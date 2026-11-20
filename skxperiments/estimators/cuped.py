"""CUPED (Controlled-experiment Using Pre-Experiment Data) estimator.

Reduces variance of the ATE estimate by adjusting the outcome with a
pre-experiment covariate, typically the same metric measured in a
period before the experiment.

Reference: Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013).
Improving the sensitivity of online controlled experiments by
utilizing pre-experiment data. WSDM 2013.
"""

import numpy as np
import pandas as pd

from skxperiments.core.assignment import CRDAssignment
from skxperiments.core.base import BaseEstimator
from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.core.results import Results


class CUPED(BaseEstimator):
    """CUPED estimator: variance reduction via pre-experiment covariate.

    Estimates the ATE by adjusting the outcome with a pre-experiment
    covariate, typically the same metric measured before the
    experiment started:

        theta = Cov(Y, X_pre) / Var(X_pre)
        ATE = (mean(Y_T) - mean(Y_C)) - theta * (mean(X_pre_T) - mean(X_pre_C))

    Under randomization, the second term has expectation zero, so the
    estimator remains unbiased for the ATE. Variance is reduced by a
    factor of approximately ``1 - rho**2``, where ``rho`` is the
    Pearson correlation between Y and X_pre.

    With a single pre-experiment covariate and **no interaction
    term**, CUPED is asymptotically equivalent to OLS of Y on
    [1, T, X_pre]. This differs from ``LinEstimator``, which adds
    the interaction T * X_centered and is asymptotically optimal
    among linear adjustments.

    Decisão arquitetural fixada: CUPED v1 accepts only
    ``CRDAssignment``. ``BlockedAssignment`` is rejected with a
    suggestion to use ``BlockedDifferenceInMeans`` (CUPED with
    blocking is planned for v2). ``FactorialAssignment`` is rejected
    with ``DesignEstimatorMismatch``.

    Decisão arquitetural fixada: ``pre_experiment_col`` is a parameter
    of ``__init__``, not ``fit``, consistent with sklearn and the
    other estimators in the library.

    Decisão arquitetural fixada: CUPED has no ``inference_mode``
    parameter. Phase 4 will use a CUPED-specific SE formula
    without a mode flag.

    Parameters
    ----------
    outcome_col : str
        Name of the outcome column in ``assignment.data_``.
    pre_experiment_col : str or None, optional
        Name of the pre-experiment covariate column in
        ``assignment.data_``. Must differ from ``outcome_col`` and
        have non-zero variance. May be None only when the assignment
        carries a frozen covariate table, in which case the covariate is
        resolved by role at ``fit`` time. Default None.
    covariate_roles : str or sequence of str, optional
        Roles used to select the covariate from the assignment's frozen
        point-in-time table when ``pre_experiment_col`` is None. Exactly
        one covariate must match; otherwise name it explicitly. Default
        ``("adjustment",)``.

    Attributes
    ----------
    assignment_ : CRDAssignment
        The fitted assignment.
    ate_ : float
        CUPED-adjusted point estimate of the ATE.
    theta_ : float
        Adjustment coefficient ``Cov(Y, X_pre) / Var(X_pre)``.
        Interpretable as the slope of OLS regression of Y on X_pre.
    correlation_ : float
        Pearson correlation between Y and X_pre. The expected
        variance reduction relative to ``DifferenceInMeans`` is
        ``1 - correlation_**2``.
    pre_experiment_col_ : str
        The covariate column actually used; records whether it came from
        ``pre_experiment_col`` or from the frozen table.
    covariate_metadata_ : dict or None
        Frozen table metadata (roles, windows, fill rule, cutoff and
        history versions, fingerprint) when the covariate came from a
        frozen table. Propagated to ``Results.extra``.

    Notes
    -----
    Covariance and variance are computed with ``ddof=1`` (sample
    convention), consistent with ``pandas.Series.cov`` and ``.var``
    defaults. ``np.cov(y, x, ddof=1)[0, 1]`` returns the off-diagonal
    sample covariance; ``np.corrcoef(y, x)[0, 1]`` returns the
    Pearson correlation.

    When the assignment carries ``frozen_covariates_``, the covariate is
    read from that frozen table and verified to be unchanged, rather
    than trusting whatever column of the same name is currently in the
    data. CUPED with a post-assignment covariate is not CUPED.

    Standard errors, confidence intervals, and p-values are not
    computed here. The ``Results`` object returned by ``estimate()``
    has ``se``, ``ci``, ``p_value`` set to ``None``. Phase 4
    inference classes will compute them.

    Examples
    --------
    >>> from skxperiments.design.crd import CRD
    >>> from skxperiments.estimators.cuped import CUPED
    >>> design = CRD(p=0.5, seed=42)
    >>> assignment = design.randomize(df)  # doctest: +SKIP
    >>> estimator = CUPED(
    ...     outcome_col="y", pre_experiment_col="y_pre"
    ... )
    >>> result = estimator.fit(assignment).estimate()  # doctest: +SKIP
    >>> result.ate  # doctest: +SKIP
    >>> result.extra["theta"]  # doctest: +SKIP
    """

    def __init__(
        self,
        outcome_col: str,
        pre_experiment_col: str | None = None,
        covariate_roles: str | tuple[str, ...] = "adjustment",
    ) -> None:
        # Validate types and non-emptiness.
        if not isinstance(outcome_col, str) or len(outcome_col) == 0:
            raise InvalidDesignError(
                f"outcome_col must be a non-empty string, but received "
                f"{outcome_col!r}."
            )
        if pre_experiment_col is not None and (
            not isinstance(pre_experiment_col, str)
            or len(pre_experiment_col) == 0
        ):
            raise InvalidDesignError(
                f"pre_experiment_col must be a non-empty string or None, "
                f"but received {pre_experiment_col!r}."
            )

        # Validate distinctness.
        if outcome_col == pre_experiment_col:
            raise InvalidDesignError(
                f"outcome_col and pre_experiment_col must differ. Using "
                f"the same column ({outcome_col!r}) for both does not "
                f"make sense: theta would be 1, and the adjusted "
                f"outcome would be identically zero."
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

        self.outcome_col = outcome_col
        self.pre_experiment_col = pre_experiment_col
        self.covariate_roles = roles

    def _resolve_pre_experiment_col(self, assignment: CRDAssignment) -> str:
        """Resolve the covariate column, preferring the frozen table."""
        frozen = assignment.frozen_covariates_
        if frozen is not None and self.pre_experiment_col is not None:
            if self.pre_experiment_col not in frozen.names:
                raise InvalidDesignError(
                    f"pre_experiment_col {self.pre_experiment_col!r} is "
                    f"not in the assignment's frozen covariate table. "
                    f"Declared covariates: {frozen.names}. A pre-experiment "
                    f"covariate must come from the point-in-time table; "
                    f"build it with PointInTimeCovariates."
                )
            return self.pre_experiment_col

        if self.pre_experiment_col is not None:
            return self.pre_experiment_col

        if frozen is None:
            raise InvalidDesignError(
                "CUPED was declared without pre_experiment_col, so it "
                "needs an assignment carrying a frozen point-in-time "
                "covariate table to select the covariate by role. Pass "
                "pre_experiment_col explicitly for a plain DataFrame."
            )

        selected = frozen.role_columns(*self.covariate_roles)
        if len(selected) != 1:
            raise InvalidDesignError(
                f"CUPED needs exactly one covariate with role(s) "
                f"{list(self.covariate_roles)} in the frozen table; found "
                f"{selected}. Name the covariate with "
                f"pre_experiment_col=."
            )
        return selected[0]

    def fit(self, assignment: CRDAssignment) -> "CUPED":
        """Fit the CUPED estimator on a CRDAssignment.

        Parameters
        ----------
        assignment : CRDAssignment
            The assignment to fit on.

        Returns
        -------
        CUPED
            Returns self.

        Raises
        ------
        DesignEstimatorMismatch
            If ``assignment`` is not a ``CRDAssignment``. For
            ``BlockedAssignment``, suggests
            ``BlockedDifferenceInMeans``.
        InvalidDesignError
            If ``outcome_col`` or ``pre_experiment_col`` is missing,
            non-numeric, contains NaN; or if ``pre_experiment_col``
            has zero variance. Also if ``pre_experiment_col`` is None and
            no frozen covariate table selects exactly one adjustment
            covariate.
        FrozenCovariateViolation
            If the assignment's frozen covariates changed after design.
        """
        self._validate_assignment_type(assignment, CRDAssignment)

        data = assignment.data_
        frozen = getattr(assignment, "frozen_covariates_", None)
        if frozen is not None:
            # The pre-period covariate is the design feature: confirm it
            # was not replaced by a newer snapshot before using it.
            frozen.assert_unchanged(data)

        pre_col = self._resolve_pre_experiment_col(assignment)

        # Validate outcome.
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

        # Validate pre-experiment covariate.
        if pre_col not in data.columns:
            raise InvalidDesignError(
                f"Pre-experiment column '{pre_col}' "
                f"not found in assignment.data_. Available columns: "
                f"{list(data.columns)}."
            )

        if not pd.api.types.is_numeric_dtype(data[pre_col]):
            raise InvalidDesignError(
                f"Pre-experiment column '{pre_col}' "
                f"must be numeric. dtype found: "
                f"{data[pre_col].dtype}."
            )

        if data[pre_col].isna().any():
            raise InvalidDesignError(
                f"Pre-experiment column '{pre_col}' "
                f"contains NaN values. Impute or drop NaN before "
                f"fitting."
            )

        # Validate non-zero variance.
        var_x = float(np.var(data[pre_col].values, ddof=1))
        if var_x == 0:
            raise InvalidDesignError(
                f"pre_experiment_col '{pre_col}' has "
                f"zero variance; theta is undefined. CUPED requires a "
                f"non-constant pre-experiment covariate."
            )

        # Compute CUPED estimator.
        y = data[self.outcome_col].values.astype(float)
        x_pre = data[pre_col].values.astype(float)

        # theta = Cov(Y, X_pre) / Var(X_pre), over all units, ddof=1.
        cov_yx = float(np.cov(y, x_pre, ddof=1)[0, 1])
        theta = cov_yx / var_x

        # Pearson correlation, for diagnostic reporting of expected
        # variance reduction (= 1 - correlation**2).
        correlation = float(np.corrcoef(y, x_pre)[0, 1])

        # ATE on adjusted outcome:
        #   tau = (mean(Y_T) - mean(Y_C)) - theta * (mean(X_T) - mean(X_C))
        treated_idx = assignment.treated_ids()
        control_idx = assignment.control_ids()

        dim_y = float(y[treated_idx].mean() - y[control_idx].mean())
        dim_x = float(x_pre[treated_idx].mean() - x_pre[control_idx].mean())

        self.assignment_: CRDAssignment = assignment
        self.pre_experiment_col_: str = pre_col
        self.covariate_metadata_: dict | None = (
            None if frozen is None else frozen.metadata()
        )
        self.theta_: float = theta
        self.correlation_: float = correlation
        self.ate_: float = dim_y - theta * dim_x

        return self

    def estimate(self) -> Results:
        """Return a Results object with the point estimate and metadata.

        Returns
        -------
        Results
            Results with ``ate``, ``n_obs``, ``n_treated``, ``n_control``,
            ``estimator_name``, ``design_name`` populated, and ``extra``
            carrying ``theta``, ``correlation``, the resolved
            ``pre_experiment_col``, and (when the covariate came from a
            frozen point-in-time table) ``covariate_metadata``. ``se``,
            ``ci``, ``p_value`` are ``None`` — inference is Phase 4.

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
            "theta": self.theta_,
            "correlation": self.correlation_,
            "pre_experiment_col": self.pre_experiment_col_,
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