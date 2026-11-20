"""Rerandomized Completely Randomized Design (Morgan & Rubin 2012).

Repeatedly proposes CRD assignments and accepts the first one whose
between-group Mahalanobis distance on the specified covariates is
below a fixed threshold. Improves covariate balance over plain CRD
while preserving the validity of randomization-based inference, as
long as the same acceptance criterion is applied when generating
permutations under the null.
"""

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from skxperiments.core.assignment import CRDAssignment
from skxperiments.core.base import BaseDesign
from skxperiments.core.exceptions import (
    InsufficientDataError,
    InvalidDesignError,
)


class ReRandomizedCRD(BaseDesign):
    """Rerandomized CRD with Mahalanobis acceptance criterion.

    Treatment is randomized completely at random (as in CRD), but only
    realizations whose Mahalanobis distance between treated and control
    means on ``covariates`` is at most ``threshold`` are accepted.

    Parameters
    ----------
    covariates : list of str or None, optional
        Names of covariates used to compute the Mahalanobis distance.
        Must be non-empty unless ``covariate_roles`` is used to select
        covariates from a frozen point-in-time table at ``randomize``
        time. All numeric, no NaN. Default None.
    covariate_roles : str or sequence of str, optional
        When ``covariates`` is None, the frozen covariate table's
        covariates carrying any of these roles are used. By default
        ``("balance",)``. The selected names are frozen in
        ``rerandomization_metadata`` so the criterion is auditable.
    threshold : float
        Maximum Mahalanobis distance for acceptance. Must be > 0.
        Has the interpretation of a chi-squared quantile with ``k``
        degrees of freedom, where ``k = len(covariates)`` (e.g.,
        ``scipy.stats.chi2.ppf(0.01, df=k)`` accepts approximately
        1% of CRD randomizations under regularity conditions).
    n_treated : int or None, optional
        Number of treated units. Provide either ``n_treated`` or ``p``,
        not both. By default None.
    p : float or None, optional
        Treatment proportion in (0, 1). Provide either ``n_treated`` or
        ``p``, not both. By default None.
    seed : int or None, optional
        Random seed for reproducibility, by default None.
    treatment_col : str, optional
        Name of the treatment column added to the output, by default
        ``"treatment"``.
    max_attempts : int, optional
        Maximum number of randomizations attempted before giving up,
        by default 10_000.

    Notes
    -----
    The Mahalanobis distance between treated and control means is
    defined as (Morgan & Rubin 2012):

        M = d^T [(1/n_T + 1/n_C) * S_X]^(-1) d

    where ``d = mean_treated - mean_control`` and ``S_X`` is the sample
    covariance (``ddof=1``) of the covariates over the full DataFrame.
    The covariance matrix is computed once in ``randomize`` and reused
    in every ``draw`` call to ensure the null distribution generated
    by ``RandomizationTest`` respects the same acceptance criterion.

    References
    ----------
    Morgan, K. L., & Rubin, D. B. (2012). Rerandomization to improve
    covariate balance in experiments. Annals of Statistics, 40(2).
    """

    def __init__(
        self,
        covariates: list[str] | None = None,
        threshold: float | None = None,
        n_treated: int | None = None,
        p: float | None = None,
        seed: int | None = None,
        treatment_col: str = "treatment",
        max_attempts: int = 10_000,
        covariate_roles: str | tuple[str, ...] = "balance",
    ) -> None:
        # n_treated XOR p
        if n_treated is None and p is None:
            raise InvalidDesignError(
                "ReRandomizedCRD requires exactly one of n_treated or p; "
                "both are None."
            )
        if n_treated is not None and p is not None:
            raise InvalidDesignError(
                "ReRandomizedCRD requires exactly one of n_treated or p; "
                "both were provided."
            )

        if covariates is not None and (
            not isinstance(covariates, list) or len(covariates) == 0
        ):
            raise InvalidDesignError(
                "ReRandomizedCRD requires a non-empty list of covariates "
                "or None to select them from a frozen covariate table by "
                "role."
            )

        if isinstance(covariate_roles, str):
            roles = (covariate_roles,)
        elif isinstance(covariate_roles, (tuple, list)) and len(
            covariate_roles
        ) > 0:
            roles = tuple(covariate_roles)
        else:
            raise InvalidDesignError(
                "covariate_roles must be a role or a non-empty sequence "
                "of roles."
            )
        from skxperiments.covariates.spec import VALID_ROLES

        for role in roles:
            if role not in VALID_ROLES:
                raise InvalidDesignError(
                    f"Unknown covariate role {role!r}. Valid roles: "
                    f"{list(VALID_ROLES)}."
                )

        if threshold is None:
            raise InvalidDesignError(
                "ReRandomizedCRD requires a threshold (the maximum "
                "Mahalanobis distance for acceptance)."
            )
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or threshold <= 0
        ):
            raise InvalidDesignError(
                f"threshold must be > 0, but received {threshold}."
            )

        if p is not None and not (0.0 < p < 1.0):
            raise InvalidDesignError(
                f"Treatment proportion p must be in (0, 1), but received {p}."
            )

        if not isinstance(max_attempts, int) or max_attempts < 1:
            raise InvalidDesignError(
                f"max_attempts must be a positive integer, but received "
                f"{max_attempts}."
            )

        self.covariates = covariates
        self.covariate_roles = roles
        self.threshold = threshold
        self.n_treated = n_treated
        self.p = p
        self.seed = seed
        self.treatment_col = treatment_col
        self.max_attempts = max_attempts

    def _resolve_covariates(
        self, frozen_covariates: Any = None
    ) -> list[str]:
        """Return the covariates defining the acceptance criterion.

        Explicit ``covariates`` win. Otherwise the frozen table supplies
        the covariates carrying ``covariate_roles``; a plain DataFrame
        cannot be used, because a role-based criterion must be built
        from point-in-time values.
        """
        if self.covariates is not None:
            return list(self.covariates)

        if frozen_covariates is None:
            raise InvalidDesignError(
                "ReRandomizedCRD was declared without explicit covariates, "
                "so it needs a frozen point-in-time covariate table to "
                "select the balance covariates. Build one with "
                "PointInTimeCovariates and pass frozen_covariates= to "
                "randomize()."
            )
        selected = frozen_covariates.role_columns(*self.covariate_roles)
        if not selected:
            raise InvalidDesignError(
                f"The frozen covariate table declares no covariate with "
                f"role(s) {list(self.covariate_roles)}; the rerandomization "
                f"criterion would be empty. Declared roles: "
                f"{ {n: list(r) for n, r in frozen_covariates.roles.items()} }."
            )
        return list(selected)

    def randomize(
        self,
        df: pd.DataFrame,
        *,
        frozen_covariates: Any = None,
    ) -> CRDAssignment:
        """Perform rerandomization and return a CRDAssignment.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with experimental units. Must contain all
            criterion covariates and must not contain ``treatment_col``.

        Other Parameters
        ----------------
        frozen_covariates : CovariateFrame or None, optional
            Frozen point-in-time covariate table. Required when the
            design was declared without explicit ``covariates`` (the
            criterion is then selected by role). When supplied, the
            criterion columns are verified against the frozen values, so
            a later "latest snapshot" cannot redefine the design.

        Returns
        -------
        CRDAssignment
            Assignment whose Mahalanobis distance is at most
            ``threshold``. ``rerandomization_metadata`` is populated,
            including the resolved criterion covariates.

        Raises
        ------
        InvalidDesignError
            For any validation failure listed in the class docstring,
            or when ``max_attempts`` is reached.
        InsufficientDataError
            When ``len(df) < n_treated``.
        """
        n_total = len(df)
        covariates = self._resolve_covariates(frozen_covariates)

        if frozen_covariates is not None:
            # Verify only the criterion columns: the design needs no other
            # frozen covariate, and a caller may legitimately pass a frame
            # with just those columns plus outcomes.
            frozen_covariates.assert_columns_unchanged(df, covariates)

        if self.treatment_col in df.columns:
            raise InvalidDesignError(
                f"Treatment column '{self.treatment_col}' already exists "
                f"in DataFrame. Drop or rename it before calling randomize()."
            )

        # Validate covariates exist
        missing = [c for c in covariates if c not in df.columns]
        if missing:
            raise InvalidDesignError(
                f"Covariates not found in DataFrame: {missing}. "
                f"Available columns: {list(df.columns)}."
            )

        # Validate numeric dtype
        non_numeric = [
            c for c in covariates if not is_numeric_dtype(df[c])
        ]
        if non_numeric:
            raise InvalidDesignError(
                f"Covariates must be numeric: {non_numeric} are not."
            )

        # Validate no NaN
        cols_with_nan = [c for c in covariates if df[c].isna().any()]
        if cols_with_nan:
            raise InvalidDesignError(
                f"Covariates contain NaN values: {cols_with_nan}. "
                f"Impute or drop NaN before calling randomize()."
            )

        # Resolve n_treated
        if self.n_treated is not None:
            if n_total < self.n_treated:
                raise InsufficientDataError(
                    context="ReRandomizedCRD randomization",
                    minimum=self.n_treated,
                    received=n_total,
                )
            n_treated = self.n_treated
        else:
            n_treated = int(round(self.p * n_total))

        if n_treated <= 0 or n_treated >= n_total:
            raise InvalidDesignError(
                f"Resolved n_treated={n_treated} for N={n_total}; must "
                f"be strictly between 0 and N. Adjust n_treated or p."
            )

        # Compute covariance matrix once (ddof=1)
        cov_matrix = df[covariates].cov(ddof=1).values

        # Check for singularity
        k = len(covariates)
        rank = np.linalg.matrix_rank(cov_matrix)
        if rank < k:
            raise InvalidDesignError(
                f"Covariance matrix of covariates is singular "
                f"(rank {rank} < {k}). Covariates are likely collinear; "
                f"remove redundant variables."
            )

        rng = np.random.default_rng(self.seed)

        # Pre-resolve target n_treated and store on self for the loop
        # to use without recomputation.
        self._resolved_n_treated = n_treated
        self._resolved_covariates = covariates

        return self._randomize_with_cached_cov(
            df=df,
            cov_matrix=cov_matrix,
            rng=rng,
            frozen_covariates=frozen_covariates,
        )

    def _randomize_with_cached_cov(
        self,
        df: pd.DataFrame,
        cov_matrix: np.ndarray,
        rng: np.random.Generator,
        frozen_covariates: Any = None,
        covariates: list[str] | None = None,
    ) -> CRDAssignment:
        """Acceptance/rejection loop with a pre-computed covariance matrix.

        Used by ``randomize`` and reused by ``CRDAssignment.draw`` to
        avoid recomputing the covariance matrix.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame without the treatment column.
        cov_matrix : np.ndarray
            Sample covariance matrix of covariates (ddof=1), already
            computed once.
        rng : np.random.Generator
            Random generator driving the loop.
        frozen_covariates : CovariateFrame or None, optional
            Frozen covariate table carried onto the new assignment.
        covariates : list of str or None, optional
            Criterion covariates. ``CRDAssignment.draw`` passes the
            names recorded in ``rerandomization_metadata`` so a replay
            cannot silently use a different criterion than the original
            randomization.

        Returns
        -------
        CRDAssignment
            Accepted assignment with metadata populated.

        Raises
        ------
        InvalidDesignError
            If ``max_attempts`` is reached without acceptance.
        """
        n_total = len(df)

        # Resolve n_treated. When called from randomize(), it was set
        # via self._resolved_n_treated. When called from draw(), we
        # rederive it from self.n_treated / self.p.
        n_treated = getattr(self, "_resolved_n_treated", None)
        if n_treated is None:
            if self.n_treated is not None:
                n_treated = self.n_treated
            else:
                n_treated = int(round(self.p * n_total))

        n_control = n_total - n_treated
        scaling_factor = 1.0 / n_treated + 1.0 / n_control

        scaled_cov = scaling_factor * cov_matrix
        try:
            inv_scaled_cov = np.linalg.inv(scaled_cov)
        except np.linalg.LinAlgError as exc:
            raise InvalidDesignError(
                f"Failed to invert scaled covariance matrix: {exc}. "
                f"Covariates may be collinear."
            ) from None

        if covariates is None:
            covariates = getattr(self, "_resolved_covariates", None)
        if covariates is None:
            if self.covariates is not None:
                covariates = list(self.covariates)
            elif frozen_covariates is not None:
                covariates = frozen_covariates.role_columns(
                    *self.covariate_roles
                )
            else:
                raise InvalidDesignError(
                    "ReRandomizedCRD cannot replay the acceptance criterion: "
                    "no resolved covariates and no frozen covariate table."
                )

        cov_values = df[covariates].values

        for attempt in range(1, self.max_attempts + 1):
            treatment = np.zeros(n_total, dtype=int)
            chosen = rng.choice(n_total, size=n_treated, replace=False)
            treatment[chosen] = 1

            treated_mask = treatment == 1
            mean_t = cov_values[treated_mask].mean(axis=0)
            mean_c = cov_values[~treated_mask].mean(axis=0)
            d = mean_t - mean_c

            distance = float(d @ inv_scaled_cov @ d)

            if distance <= self.threshold:
                df_out = df.copy()
                df_out[self.treatment_col] = treatment

                metadata = {
                    "covariates": list(covariates),
                    "threshold": float(self.threshold),
                    "cov_matrix": cov_matrix,
                    "attempts": attempt,
                    "scaling_factor": scaling_factor,
                }

                return CRDAssignment(
                    data=df_out,
                    treatment_col=self.treatment_col,
                    design=self,
                    seed=self.seed,
                    rerandomization_metadata=metadata,
                    frozen_covariates=frozen_covariates,
                )

        raise InvalidDesignError(
            f"ReRandomizedCRD failed to find an assignment with "
            f"Mahalanobis distance <= {self.threshold} after "
            f"{self.max_attempts} attempts. Increase threshold or "
            f"max_attempts."
        )