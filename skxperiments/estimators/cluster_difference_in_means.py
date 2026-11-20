"""Difference-in-means estimator for cluster-randomized designs.

Aggregates individual-level outcomes to the cluster level before
comparing arms, so the effective sample size is the number of
clusters — not the number of individuals. Comparing raw individual
means would treat customers of the same store as independent
randomization units and understate the variance of the estimate.
"""

import pandas as pd

from skxperiments.core.assignment import ClusterAssignment
from skxperiments.core.base import BaseEstimator
from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.core.results import Results


class ClusterDifferenceInMeans(BaseEstimator):
    """Difference-in-means estimator for ClusterAssignment.

    Collapses the outcome to one value per cluster (the cluster mean)
    and compares arms at the cluster level. The estimand is chosen
    explicitly via ``estimand``:

    - ``"cluster_mean"`` (default): every cluster weighs equally,

        ATE_hat = mean_g(Ybar_g | treated) - mean_g(Ybar_g | control)

      where ``Ybar_g`` is the mean outcome of cluster ``g``. This is
      the average effect on the cluster-level mean, clusters weighted
      equally — the canonical estimand when clusters are the
      randomization units.

    - ``"exposure_weighted"``: clusters are weighted by a
      **pre-specified** exposure column ``weight_col`` (e.g., expected
      customer volume fixed before the experiment starts),

        ATE_hat = sum(w_g Ybar_g T_g) / sum(w_g T_g)
                  - sum(w_g Ybar_g (1 - T_g)) / sum(w_g (1 - T_g))

      The weight must be a cluster-level attribute (constant within
      cluster) and must not depend on the realized assignment or the
      outcome; weights defined post-hoc invalidate the estimand.

    This estimator computes the point estimate only. Inference for
    cluster designs is randomization-based: use ``RandomizationTest``
    (each permutation relabels whole clusters). ``NeymanCI`` and
    ``BootstrapCI`` do not support ``ClusterAssignment`` in v1.

    Parameters
    ----------
    outcome_col : str
        Name of the outcome column in ``assignment.data_``.
    estimand : {"cluster_mean", "exposure_weighted"}, optional
        Estimand selector, by default ``"cluster_mean"``.
    weight_col : str or None, optional
        Pre-specified exposure column. Required when
        ``estimand="exposure_weighted"``; must be None otherwise.

    Attributes
    ----------
    assignment_ : ClusterAssignment
        The fitted assignment, stored for downstream use.
    ate_ : float
        Point estimate of the chosen estimand.
    cluster_frame_ : pd.DataFrame
        Cluster-level aggregates used in the estimation: one row per
        cluster with columns ``outcome`` (cluster mean), ``treatment``,
        and ``weight`` (only for ``estimand="exposure_weighted"``),
        indexed by cluster label.

    Notes
    -----
    The ``Results`` returned by ``estimate()`` reports individual-level
    counts in ``n_obs`` / ``n_treated`` / ``n_control`` (the rows of
    the assignment data) and the cluster counts — the actual
    randomization units — in ``extra``: ``"n_clusters"``,
    ``"n_clusters_treated"``, ``"n_clusters_control"``, plus
    ``"estimand"``. Neither count is silently substituted for the
    other.

    Examples
    --------
    >>> from skxperiments.design.cluster import ClusterDesign
    >>> from skxperiments.design.crd import CRD
    >>> from skxperiments.estimators.cluster_difference_in_means import (
    ...     ClusterDifferenceInMeans,
    ... )
    >>> design = ClusterDesign(cluster_col="store_id",
    ...                        base_design=CRD(p=0.5, seed=42))
    >>> assignment = design.randomize(df)  # doctest: +SKIP
    >>> estimator = ClusterDifferenceInMeans(outcome_col="y")
    >>> results = estimator.fit(assignment).estimate()  # doctest: +SKIP
    """

    _VALID_ESTIMANDS = ("cluster_mean", "exposure_weighted")

    def __init__(
        self,
        outcome_col: str,
        estimand: str = "cluster_mean",
        weight_col: str | None = None,
    ) -> None:
        if estimand not in self._VALID_ESTIMANDS:
            raise InvalidDesignError(
                f"estimand must be one of {self._VALID_ESTIMANDS}, "
                f"got {estimand!r}."
            )
        if estimand == "exposure_weighted" and weight_col is None:
            raise InvalidDesignError(
                "estimand='exposure_weighted' requires a weight_col "
                "with the pre-specified exposure of each cluster."
            )
        if estimand == "cluster_mean" and weight_col is not None:
            raise InvalidDesignError(
                "weight_col is only used with "
                "estimand='exposure_weighted'; pass weight_col=None "
                "for estimand='cluster_mean'."
            )

        self.outcome_col = outcome_col
        self.estimand = estimand
        self.weight_col = weight_col

    def fit(
        self, assignment: ClusterAssignment
    ) -> "ClusterDifferenceInMeans":
        """Fit the estimator on a ClusterAssignment.

        Parameters
        ----------
        assignment : ClusterAssignment
            Assignment produced by ``ClusterDesign``.

        Returns
        -------
        ClusterDifferenceInMeans
            Returns self.

        Raises
        ------
        DesignEstimatorMismatch
            If ``assignment`` is not a ``ClusterAssignment``.
        InvalidDesignError
            If ``outcome_col`` is missing, non-numeric, or contains
            NaN; or, for ``estimand="exposure_weighted"``, if
            ``weight_col`` is missing, non-numeric, contains NaN or
            negative values, varies within a cluster, or has zero
            total weight in an arm.
        """
        self._validate_assignment_type(assignment, ClusterAssignment)

        data = assignment.data_

        if self.outcome_col not in data.columns:
            raise InvalidDesignError(
                f"Outcome column '{self.outcome_col}' not found in "
                f"assignment.data_. Available columns: "
                f"{list(data.columns)}."
            )
        if not pd.api.types.is_numeric_dtype(data[self.outcome_col]):
            raise InvalidDesignError(
                f"Outcome column '{self.outcome_col}' must be numeric. "
                f"dtype found: {data[self.outcome_col].dtype}."
            )
        if data[self.outcome_col].isna().any():
            raise InvalidDesignError(
                f"Outcome column '{self.outcome_col}' contains NaN "
                f"values. Impute or drop NaN before fitting."
            )

        cluster_col = assignment.cluster_col_
        grouped = data.groupby(cluster_col)

        cluster_frame = pd.DataFrame(
            {
                "outcome": grouped[self.outcome_col].mean(),
                "treatment": grouped[assignment.treatment_col_].first(),
            }
        )

        if self.estimand == "exposure_weighted":
            cluster_frame["weight"] = self._validated_cluster_weights(
                data, grouped
            )

        treated = cluster_frame["treatment"] == 1
        control = ~treated

        if self.estimand == "cluster_mean":
            ate = (
                cluster_frame.loc[treated, "outcome"].mean()
                - cluster_frame.loc[control, "outcome"].mean()
            )
        else:
            w_t = cluster_frame.loc[treated, "weight"]
            w_c = cluster_frame.loc[control, "weight"]
            if w_t.sum() == 0.0 or w_c.sum() == 0.0:
                raise InvalidDesignError(
                    "Total exposure weight is zero in at least one "
                    "arm; the exposure-weighted estimand is undefined. "
                    "Check weight_col."
                )
            ate = (
                (w_t * cluster_frame.loc[treated, "outcome"]).sum()
                / w_t.sum()
                - (w_c * cluster_frame.loc[control, "outcome"]).sum()
                / w_c.sum()
            )

        self.assignment_: ClusterAssignment = assignment
        self.cluster_frame_: pd.DataFrame = cluster_frame
        self.ate_: float = float(ate)

        return self

    def _validated_cluster_weights(
        self, data: pd.DataFrame, grouped: "pd.core.groupby.DataFrameGroupBy"
    ) -> "pd.Series":
        """Validate weight_col and return one weight per cluster."""
        weight_col = self.weight_col

        if weight_col not in data.columns:
            raise InvalidDesignError(
                f"Weight column '{weight_col}' not found in "
                f"assignment.data_. Available columns: "
                f"{list(data.columns)}."
            )
        if not pd.api.types.is_numeric_dtype(data[weight_col]):
            raise InvalidDesignError(
                f"Weight column '{weight_col}' must be numeric. "
                f"dtype found: {data[weight_col].dtype}."
            )
        if data[weight_col].isna().any():
            raise InvalidDesignError(
                f"Weight column '{weight_col}' contains NaN values. "
                f"Every cluster must have a pre-specified exposure."
            )
        if (data[weight_col] < 0).any():
            raise InvalidDesignError(
                f"Weight column '{weight_col}' contains negative "
                f"values. Exposure weights must be non-negative."
            )

        n_unique = grouped[weight_col].nunique()
        if (n_unique > 1).any():
            offending = n_unique[n_unique > 1].index.tolist()
            raise InvalidDesignError(
                f"Weight column '{weight_col}' must be constant "
                f"within each cluster (exposure is a cluster-level, "
                f"pre-specified attribute), but it varies within "
                f"cluster(s) {offending}."
            )

        return grouped[weight_col].first()

    def estimate(self) -> Results:
        """Return a Results object with the point estimate and metadata.

        Returns
        -------
        Results
            Results with ``ate``, individual-level ``n_obs`` /
            ``n_treated`` / ``n_control``, ``estimator_name``, and
            ``design_name`` populated; cluster counts and the chosen
            estimand in ``extra``. ``se``, ``ci``, ``p_value`` are
            ``None`` — use ``RandomizationTest`` for inference.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called.
        """
        self._check_is_fitted()

        design_name: str | None
        if self.assignment_.design_ is not None:
            design_name = type(self.assignment_.design_).__name__
        else:
            design_name = None

        return Results(
            ate=self.ate_,
            n_obs=self.assignment_.n_units_,
            n_treated=self.assignment_.n_treated_,
            n_control=self.assignment_.n_control_,
            estimator_name=type(self).__name__,
            design_name=design_name,
            extra={
                "estimand": self.estimand,
                "n_clusters": self.assignment_.n_clusters_,
                "n_clusters_treated": self.assignment_.n_clusters_treated_,
                "n_clusters_control": self.assignment_.n_clusters_control_,
            },
        )
