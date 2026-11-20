"""Lin (2013) covariate-adjusted ATE estimator.

Computes the ATE via OLS of Y on (1, T, X_centered, T * X_centered),
where X_centered = X - mean(X). The coefficient of T is the ATE
estimate. Lin's adjustment reduces variance compared to plain
difference-in-means when covariates predict the outcome, while
remaining consistent for the ATE under any covariate distribution.

Reference: Lin, W. (2013). Agnostic notes on regression adjustments
to experimental data: Reexamining Freedman's critique. Annals of
Applied Statistics, 7(1), 295-318.
"""

import numpy as np
import pandas as pd

from skxperiments.core.assignment import (
    BlockedAssignment,
    CRDAssignment,
)
from skxperiments.core.base import BaseEstimator
from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.core.results import Results
from skxperiments.covariates.frame import resolve_covariates


class LinEstimator(BaseEstimator):
    """Lin (2013) covariate-adjusted ATE estimator.

    Estimates the ATE via OLS of Y on a design matrix that includes
    the treatment indicator, mean-centered covariates, and their
    treatment-covariate interactions:

        ATE_hat = coef_T in OLS of Y on [1, T, X_centered, T * X_centered]

    where X_centered = X - mean(X). The coefficient of T is the ATE.

    Lin's adjustment reduces the variance of the ATE estimate compared
    to plain difference-in-means whenever covariates predict the
    outcome, and remains consistent for the ATE without distributional
    assumptions on covariates.

    Decisão arquitetural fixada (item 18): ``inference_mode`` is
    documentational metadata only at this stage. ``LinEstimator``
    computes the point estimate only — it does not compute HC2,
    Neyman, or any standard error. Inference is the responsibility
    of Phase 4 classes, which read ``Results.extra["inference_mode"]``
    to decide which variance formula to apply.

    Parameters
    ----------
    outcome_col : str
        Name of the outcome column in ``assignment.data_``.
    covariates : list of str or None, optional
        Names of covariate columns. Must be a non-empty list of strings,
        or None to select the covariates carrying ``covariate_roles``
        from the assignment's frozen point-in-time table. For unadjusted
        ATE, use ``DifferenceInMeans``.
    covariate_roles : str or sequence of str, optional
        Roles used to select covariates from the frozen table when
        ``covariates`` is None. Default ``("adjustment",)``. The design
        variables a rerandomization criterion used are usually declared
        with role ``"adjustment"`` as well; name them explicitly if they
        are not.
    inference_mode : {"finite_population", "superpopulation"}, optional
        Documentational flag propagated to ``Results.extra``. The
        Phase 4 inference class reads this to pick a variance
        formula. Default ``"finite_population"``.

    Attributes
    ----------
    assignment_ : CRDAssignment or BlockedAssignment
        The fitted assignment.
    ate_ : float
        Lin-adjusted point estimate of the ATE.
    covariates_ : list of str
        The covariates actually used (resolved at fit time when they
        came from the frozen table).
    covariate_metadata_ : dict or None
        Frozen table metadata (roles, windows, fill rule, cutoff and
        history versions, fingerprint) when covariates came from a
        frozen table. Propagated to ``Results.extra``.
    coefficients_ : np.ndarray
        Full vector of OLS coefficients with shape ``(2 + 2*K,)``,
        where K is the number of covariates. Layout:

        - ``coefficients_[0]`` — intercept
        - ``coefficients_[1]`` — ATE (coefficient of T)
        - ``coefficients_[2 : 2 + K]`` — centered-covariate coefficients
        - ``coefficients_[2 + K : 2 + 2*K]`` — interaction coefficients
          for T times each centered covariate

    inference_mode_ : str
        Copy of ``inference_mode`` set during ``fit``. Follows the
        sklearn convention of trailing-underscore attributes for
        learned state, even when the value is merely mirrored from
        ``__init__`` without transformation.

    Notes
    -----
    Accepts ``CRDAssignment`` or ``BlockedAssignment``.
    ``FactorialAssignment`` is rejected via ``DesignEstimatorMismatch``.

    With ``BlockedAssignment``, ``LinEstimator`` treats the data as a
    single sample and does **not** use block structure. Users who
    want to exploit block structure should use
    ``BlockedDifferenceInMeans``.

    Constant covariates (zero variance) make the design matrix
    singular and are rejected at ``fit``.

    When the assignment carries ``frozen_covariates_``, the adjustment
    covariates are read from the frozen point-in-time table: with
    ``covariates=None`` they are selected by role; with explicit names
    each must be a frozen covariate (otherwise the fit raises), and the
    frozen values are verified unchanged before adjusting. A column that
    is not in the frozen table (even if it looks pre-period) is not
    accepted on such an assignment — build it into the table so it
    carries a cutoff and a fingerprint. Assignments without a frozen
    table keep the plain DataFrame behaviour.

    Examples
    --------
    >>> from skxperiments.design.crd import CRD
    >>> from skxperiments.estimators.lin_estimator import LinEstimator
    >>> design = CRD(p=0.5, seed=42)
    >>> assignment = design.randomize(df)  # doctest: +SKIP
    >>> estimator = LinEstimator(
    ...     outcome_col="y", covariates=["x1", "x2"]
    ... )
    >>> result = estimator.fit(assignment).estimate()  # doctest: +SKIP
    >>> result.ate  # doctest: +SKIP
    """

    _VALID_INFERENCE_MODES = ("finite_population", "superpopulation")

    def __init__(
        self,
        outcome_col: str,
        covariates: list[str] | None = None,
        inference_mode: str = "finite_population",
        covariate_roles: str | tuple[str, ...] = "adjustment",
    ) -> None:
        # Validate covariates
        if covariates is not None:
            if not isinstance(covariates, list):
                raise InvalidDesignError(
                    f"covariates must be a list of strings or None, but "
                    f"received {type(covariates).__name__}."
                )
            if len(covariates) == 0:
                raise InvalidDesignError(
                    "LinEstimator requires at least one covariate; use "
                    "DifferenceInMeans for unadjusted ATE. Pass "
                    "covariates=None to select them from a frozen "
                    "covariate table by role."
                )
            for c in covariates:
                if not isinstance(c, str):
                    raise InvalidDesignError(
                        f"covariates must be a list of strings; found "
                        f"element of type {type(c).__name__}: {c!r}."
                    )

        if isinstance(covariate_roles, str):
            roles: tuple[str, ...] = (covariate_roles,)
        elif isinstance(covariate_roles, (tuple, list)) and len(
            covariate_roles
        ) > 0:
            roles = tuple(covariate_roles)
        else:
            raise InvalidDesignError(
                "covariate_roles must be a role or a non-empty sequence "
                "of roles."
            )

        # Validate inference_mode
        if inference_mode not in self._VALID_INFERENCE_MODES:
            raise InvalidDesignError(
                f"inference_mode must be one of "
                f"{self._VALID_INFERENCE_MODES}, but received "
                f"{inference_mode!r}."
            )

        self.outcome_col = outcome_col
        self.covariates = covariates
        self.inference_mode = inference_mode
        self.covariate_roles = roles

    def fit(
        self, assignment: CRDAssignment | BlockedAssignment
    ) -> "LinEstimator":
        """Fit the Lin estimator on a CRDAssignment or BlockedAssignment.

        Parameters
        ----------
        assignment : CRDAssignment or BlockedAssignment
            The assignment to fit on.

        Returns
        -------
        LinEstimator
            Returns self.

        Raises
        ------
        DesignEstimatorMismatch
            If ``assignment`` is not a ``CRDAssignment`` or
            ``BlockedAssignment``.
        InvalidDesignError
            If ``outcome_col`` is missing, non-numeric, or has NaN;
            if any covariate is missing, non-numeric, has NaN, or is
            constant (zero variance); or if ``covariates`` is None and
            the frozen table declares no adjustment covariate.
        FrozenCovariateViolation
            If the assignment's frozen covariates changed after design.
        """
        self._validate_assignment_type(
            assignment, (CRDAssignment, BlockedAssignment)
        )

        data = assignment.data_
        frozen = getattr(assignment, "frozen_covariates_", None)
        if frozen is not None:
            # The covariates are the design feature: confirm they were
            # not replaced by a newer snapshot before adjusting.
            frozen.assert_unchanged(data)

        if self.covariates is not None:
            # With a frozen table present, explicit names must be frozen
            # covariates: an unfrozen column is exactly the latest
            # snapshot that must not enter the design feature.
            covariates = resolve_covariates(assignment, self.covariates)
        else:
            covariates = resolve_covariates(
                assignment, None, roles=self.covariate_roles
            )
            if not covariates:
                raise InvalidDesignError(
                    "LinEstimator was declared without covariates, so it "
                    "needs an assignment carrying a frozen point-in-time "
                    "covariate table to select them by role. Pass "
                    "covariates explicitly for a plain DataFrame."
                )

        # Validate outcome
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

        # Validate covariates
        missing = [c for c in covariates if c not in data.columns]
        if missing:
            raise InvalidDesignError(
                f"Covariates not found in assignment.data_: {missing}. "
                f"Available columns: {list(data.columns)}."
            )

        non_numeric = [
            c for c in covariates
            if not pd.api.types.is_numeric_dtype(data[c])
        ]
        if non_numeric:
            raise InvalidDesignError(
                f"Covariates must be numeric: {non_numeric} are not."
            )

        with_nan = [c for c in covariates if data[c].isna().any()]
        if with_nan:
            raise InvalidDesignError(
                f"Covariates contain NaN values: {with_nan}. "
                f"Impute or drop NaN before fitting."
            )

        constants = [
            c for c in covariates
            if data[c].var(ddof=0) == 0
        ]
        if constants:
            raise InvalidDesignError(
                f"Covariates are constant (zero variance): "
                f"{constants}. A constant covariate makes the design "
                f"matrix singular; remove or replace them."
            )

        # Build the Lin design matrix.
        X = data[covariates].values.astype(float)
        X_centered = X - X.mean(axis=0)

        T = (
            data[assignment.treatment_col_]
            .values.astype(float)
            .reshape(-1, 1)
        )
        y = data[self.outcome_col].values.astype(float)

        n = len(y)
        intercept = np.ones((n, 1))
        interaction = T * X_centered  # broadcasting: (n,1)*(n,K) -> (n,K)
        design_matrix = np.hstack(
            [intercept, T, X_centered, interaction]
        )

        # OLS via lstsq — more stable than inverting X.T @ X.
        coefficients, *_ = np.linalg.lstsq(
            design_matrix, y, rcond=None
        )

        # Layout: [intercept, T, X_centered (K), T*X_centered (K)].
        # coefficients[1] is the ATE.
        self.assignment_: CRDAssignment | BlockedAssignment = assignment
        self.covariates_: list[str] = list(covariates)
        self.covariate_metadata_: dict | None = (
            None if frozen is None else frozen.metadata()
        )
        self.ate_: float = float(coefficients[1])
        self.coefficients_: np.ndarray = coefficients
        self.inference_mode_: str = self.inference_mode

        return self

    def estimate(self) -> Results:
        """Return a Results object with the point estimate and metadata.

        Returns
        -------
        Results
            Results with ``ate``, ``n_obs``, ``n_treated``, ``n_control``,
            ``estimator_name``, ``design_name`` populated, and
            ``extra`` carrying ``inference_mode``, the resolved
            ``covariates``, and (when covariates came from a frozen
            point-in-time table) ``covariate_metadata``. ``se``, ``ci``,
            ``p_value`` are ``None`` — inference is Phase 4.

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

        extra: dict = {
            "inference_mode": self.inference_mode_,
            "covariates": list(self.covariates_),
        }
        if self.covariate_metadata_ is not None:
            extra["covariate_metadata"] = self.covariate_metadata_

        return Results(
            ate=self.ate_,
            n_obs=self.assignment_.n_units_,
            n_treated=self.assignment_.n_treated_,
            n_control=self.assignment_.n_control_,
            estimator_name=type(self).__name__,
            design_name=design_name,
            extra=extra,
        )