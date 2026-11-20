"""Joint randomization inference across multiple outcomes on a shared plan.

Analyzing K outcome variables of the same experiment with K independent
``RandomizationTest`` calls repeats the resampling draws independently per
outcome: each metric permutes its own assignment, so the simulated null
worlds are unrelated across outcomes and the dependence between test
statistics — induced by shared covariates and shared treatment draws — is
destroyed. Corrections that merely sort p-values (Bonferroni, Holm, BH)
then ignore that dependence.

The classes here keep the dependence structure intact:

- ``RandomizationPlan`` (``skxperiments.core.plan``) is bound to one
  Assignment population and one design version, and generates the complete
  assignment of every replicate by ``replicate_id``. All outcome
  estimators are refit on the *same* replicated assignment — the same
  simulated world — so the joint null distribution of the K statistics is
  preserved.
- ``MultiOutcomeRandomizationTest`` simulates that joint null once and
  hands it to ``MaxStatCorrection``, whose adjusted p-values come from the
  distribution of the across-outcome maximum statistic (Westfall & Young,
  1993), not from sorted p-values.
- Raw per-outcome results are preserved next to the adjusted ones, so no
  information is lost to the correction.

References
----------
Westfall, P. H., & Young, S. S. (1993). Resampling-Based Multiple Testing:
    Examples and Methods for p-Value Adjustment. Wiley.
Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be
    zero: calculating exact P-values when permutations are randomly drawn.
    Statistical Applications in Genetics and Molecular Biology, 9(1).
"""

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from skxperiments.core.assignment import (
    BlockedAssignment,
    CRDAssignment,
)
from skxperiments.core.base import BaseEstimator
from skxperiments.core.exceptions import (
    DesignEstimatorMismatch,
    InvalidDesignError,
    NotFittedError,
)
from skxperiments.core.plan import RandomizationPlan
from skxperiments.core.results import Results


class MaxStatCorrection:
    """Single-step max-statistic (maxT) multiple-testing correction.

    Consumes the **joint** simulation distribution of the outcome
    statistics — one row per replicate, one column per outcome — and
    adjusts each p-value against the distribution of the across-outcome
    maximum (Westfall & Young, 1993):

        p_adj[j] = (1 + #{r : max_k S[r, k] >= s_obs[j]}) / (1 + B)

    where ``S[r, k]`` is the (possibly absolute) statistic of outcome
    ``k`` in replicate ``r`` and ``s_obs[j]`` the observed statistic of
    outcome ``j``. Because the maximum is taken within each shared
    replicate, the correction inherits the dependence between outcomes
    from the joint null instead of assuming a worst-case (Bonferroni) or
    independence-based (BH) structure. The ``+1`` terms are the Phipson
    & Smyth (2010) continuity correction, matching ``RandomizationTest``.

    Parameters
    ----------
    alternative : {"two-sided", "greater", "less"}, optional
        Direction of the test, by default ``"two-sided"``. Two-sided
        uses ``max_k |S_k|``; ``"greater"`` uses ``max_k S_k``;
        ``"less"`` uses ``min_k S_k`` with the inequality reversed.
    alpha : float, optional
        Family-wise significance level, by default 0.05. Stored for
        reporting; ``adjust`` itself only computes p-values.

    Notes
    -----
    Single-step maxT controls the family-wise error rate (FWER) under
    the joint sharp null. The uniformly more powerful free step-down
    variant is deferred (see ``ROADMAP.md``).

    Examples
    --------
    >>> import numpy as np
    >>> observed = np.array([2.0, 1.0])
    >>> joint_null = np.array(
    ...     [[0.5, 0.2], [2.5, 0.1], [1.0, 1.5], [0.3, 0.4]]
    ... )
    >>> MaxStatCorrection().adjust(observed, joint_null)
    array([0.4, 0.6])
    """

    _VALID_ALTERNATIVES = ("two-sided", "greater", "less")

    def __init__(
        self,
        alternative: str = "two-sided",
        alpha: float = 0.05,
    ) -> None:
        if alternative not in self._VALID_ALTERNATIVES:
            raise InvalidDesignError(
                f"alternative must be one of {self._VALID_ALTERNATIVES}, "
                f"got {alternative!r}."
            )
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise InvalidDesignError(
                f"alpha must be a float, got {type(alpha).__name__}."
            )
        if not (0.0 < float(alpha) < 1.0):
            raise InvalidDesignError(
                f"alpha must be in (0, 1), got {alpha}."
            )

        self.alternative = alternative
        self.alpha = float(alpha)

    def adjust(
        self,
        observed: np.ndarray,
        joint_null: np.ndarray,
    ) -> np.ndarray:
        """Compute max-stat adjusted p-values from the joint null.

        Parameters
        ----------
        observed : np.ndarray
            Observed statistics, shape ``(m,)`` — one per outcome.
        joint_null : np.ndarray
            Joint null distribution, shape ``(B, m)``: row ``r`` holds
            the statistics of all ``m`` outcomes computed on the *same*
            replicate ``r``.

        Returns
        -------
        np.ndarray
            Adjusted p-values, shape ``(m,)``, in the same outcome order
            as ``observed``.

        Raises
        ------
        InvalidDesignError
            If shapes are inconsistent, the family is empty, or any
            statistic is NaN.
        """
        observed = np.asarray(observed, dtype=float)
        joint_null = np.asarray(joint_null, dtype=float)

        if observed.ndim != 1 or observed.size == 0:
            raise InvalidDesignError(
                f"observed must be a non-empty 1-D array, got shape "
                f"{observed.shape}."
            )
        if joint_null.ndim != 2:
            raise InvalidDesignError(
                f"joint_null must be a 2-D array (replicates x outcomes), "
                f"got shape {joint_null.shape}."
            )
        if joint_null.shape[1] != observed.shape[0]:
            raise InvalidDesignError(
                f"joint_null has {joint_null.shape[1]} outcome columns but "
                f"observed has {observed.shape[0]} entries; they must match."
            )
        if np.isnan(observed).any() or np.isnan(joint_null).any():
            raise InvalidDesignError(
                "observed and joint_null must not contain NaN statistics."
            )

        n_replicates = joint_null.shape[0]

        if self.alternative == "two-sided":
            extreme_per_replicate = np.max(np.abs(joint_null), axis=1)
            observed_stat = np.abs(observed)
            n_extreme = np.sum(
                extreme_per_replicate[:, None] >= observed_stat[None, :],
                axis=0,
            )
        elif self.alternative == "greater":
            extreme_per_replicate = np.max(joint_null, axis=1)
            n_extreme = np.sum(
                extreme_per_replicate[:, None] >= observed[None, :],
                axis=0,
            )
        else:  # "less"
            extreme_per_replicate = np.min(joint_null, axis=1)
            n_extreme = np.sum(
                extreme_per_replicate[:, None] <= observed[None, :],
                axis=0,
            )

        return (1 + n_extreme) / (1 + n_replicates)

    def __repr__(self) -> str:
        """Return string representation with parameters."""
        return (
            f"MaxStatCorrection(alternative={self.alternative!r}, "
            f"alpha={self.alpha!r})"
        )


@dataclass(frozen=True, eq=False)
class MultiOutcomeResult:
    """Result of joint multi-outcome randomization inference.

    Attributes
    ----------
    raw_results : dict
        Mapping from outcome name to its unadjusted ``Results`` (raw
        per-outcome p-value from the marginal null, Phipson & Smyth
        corrected). Carries no correction metadata, so it remains
        compatible with ``MultipleTestingCorrection``.
    adjusted_results : dict
        Mapping from outcome name to its max-stat-adjusted ``Results``.
        The raw p-value is preserved in
        ``Results.extra["original_p_values"]``.
    joint_null_distribution : np.ndarray
        The joint simulated null, shape ``(n_replicates, n_outcomes)``:
        row ``r`` holds every outcome's statistic computed on the *same*
        shared replicate ``r``.
    alternative : str
        The alternative hypothesis used.
    alpha : float
        Family-wise significance level.
    n_replicates : int
        Number of shared replicates simulated.
    table : pd.DataFrame
        One row per outcome with columns ``outcome``, ``ate``,
        ``p_value`` (raw), ``p_value_adjusted`` (max-stat), and
        ``significant``.
    """

    raw_results: dict
    adjusted_results: dict
    joint_null_distribution: np.ndarray
    alternative: str
    alpha: float
    n_replicates: int
    table: pd.DataFrame

    @property
    def significant(self) -> list[str]:
        """Outcomes significant after the max-stat correction."""
        return [
            row.outcome for row in self.table.itertuples() if row.significant
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the per-outcome table."""
        return self.table.copy()

    def to_dict(self) -> dict:
        """Return the summary fields as a plain dictionary."""
        return {
            "correction": "max_stat",
            "alpha": self.alpha,
            "n_outcomes": len(self.raw_results),
            "n_replicates": self.n_replicates,
            "alternative": self.alternative,
            "significant": self.significant,
        }

    def summary(self) -> "MultiOutcomeResult":
        """Print the per-outcome table and return self."""
        print(
            f"Multi-Outcome Randomization Test (correction=max_stat, "
            f"alpha={self.alpha}, n_replicates={self.n_replicates})"
        )
        print(self.table.to_string(index=False))
        return self


class MultiOutcomeRandomizationTest:
    """Joint Fisher randomization test across multiple outcome variables.

    Fits every outcome estimator on the *same* replicated assignments —
    generated once by a shared ``RandomizationPlan`` bound to the
    observed assignment's population and design version — so the joint
    null distribution of the outcome statistics is preserved and can
    feed the max-stat correction.

    Parameters
    ----------
    estimators : dict of str to BaseEstimator
        Mapping from outcome name to a **configured, unfitted** scalar
        estimator (e.g., ``{"retention": DifferenceInMeans("retention"),
        "revenue": CUPED("revenue", "revenue_pre")}``). Each estimator
        must expose an ``outcome_col`` attribute: the common-population
        validation resolves every outcome against the same
        ``assignment.data_`` before any draw happens. The estimators are
        deep-copied at construction, so each outcome's mutable state is
        isolated and the passed instances are never fitted or mutated.
    n_replicates : int, optional
        Number of shared replicates, by default 10_000. Must be a
        positive integer.
    alternative : {"two-sided", "greater", "less"}, optional
        Alternative hypothesis, by default ``"two-sided"``. Same
        semantics as ``RandomizationTest``.
    seed : int or None, optional
        Master seed of the shared ``RandomizationPlan``. The same
        ``seed`` reproduces the same joint null. By default None.
    alpha : float, optional
        Family-wise significance level used for the adjusted results
        and the significance table, by default 0.05.

    Attributes
    ----------
    assignment_ : CRDAssignment or BlockedAssignment
        The observed assignment passed to ``fit``.
    plan_ : RandomizationPlan
        The shared randomization plan generated at ``fit`` time.
    outcome_names_ : list of str
        Outcome names in insertion order.
    observed_statistics_ : dict
        Mapping from outcome name to its observed ATE.
    joint_null_distribution_ : np.ndarray
        Joint simulated null, shape ``(n_replicates, n_outcomes)``.
    p_values_raw_ : dict
        Mapping from outcome name to its unadjusted p-value.
    p_values_adjusted_ : dict
        Mapping from outcome name to its max-stat-adjusted p-value.

    Notes
    -----
    **Common population.** ``fit`` validates, before any draw, that
    every ``outcome_col`` exists in ``assignment.data_`` and contains no
    NaN. If outcomes are missing, define the common population first —
    one Assignment whose data carries every outcome column, restricted
    to complete cases or imputed — instead of letting each metric drop
    its own rows and draw from a different population.

    **Estimator state after fit.** Each outcome's (copied) estimator is
    refit ``n_replicates`` times; after ``fit`` it holds the state of
    the *last* replicate, mirroring ``RandomizationTest``. The results
    returned by ``estimate()`` are computed from quantities captured
    before and during the loop, so they are unaffected.

    **Why not ``BaseInference``.** ``estimate()`` returns a
    ``MultiOutcomeResult`` (a family of ``Results`` plus the joint
    null), not a single ``Results``; like ``MultipleTestingCorrection``
    and ``ExperimentComparison``, this class follows the
    ``fit``/``estimate`` verbs without inheriting the ABC.

    ``FactorialAssignment`` and multi-effect estimators are rejected,
    matching ``RandomizationTest`` v1.

    Examples
    --------
    >>> from skxperiments.design.crd import CRD
    >>> from skxperiments.estimators.difference_in_means import (
    ...     DifferenceInMeans,
    ... )
    >>> from skxperiments.inference import MultiOutcomeRandomizationTest
    >>> design = CRD(p=0.5, seed=42)
    >>> assignment = design.randomize(df)  # doctest: +SKIP
    >>> test = MultiOutcomeRandomizationTest(
    ...     estimators={
    ...         "y1": DifferenceInMeans("y1"),
    ...         "y2": DifferenceInMeans("y2"),
    ...     },
    ...     n_replicates=1000,
    ...     seed=0,
    ... )
    >>> result = test.fit(assignment).estimate()  # doctest: +SKIP
    >>> result.significant  # doctest: +SKIP
    """

    _VALID_ALTERNATIVES = ("two-sided", "greater", "less")

    def __init__(
        self,
        estimators: dict[str, BaseEstimator],
        n_replicates: int = 10_000,
        alternative: str = "two-sided",
        seed: int | None = None,
        alpha: float = 0.05,
    ) -> None:
        if not isinstance(estimators, dict) or len(estimators) == 0:
            raise InvalidDesignError(
                "estimators must be a non-empty dict mapping outcome name "
                f"to a configured BaseEstimator, got "
                f"{type(estimators).__name__}."
            )
        for name, est in estimators.items():
            if not isinstance(name, str):
                raise InvalidDesignError(
                    f"estimator keys must be strings (outcome names), got "
                    f"{type(name).__name__}."
                )
            if not isinstance(est, BaseEstimator):
                raise InvalidDesignError(
                    f"Estimator for outcome '{name}' must be an instance "
                    f"of BaseEstimator, got {type(est).__name__}."
                )
            outcome_col = getattr(est, "outcome_col", None)
            if not isinstance(outcome_col, str):
                raise InvalidDesignError(
                    f"Estimator for outcome '{name}' "
                    f"({type(est).__name__}) does not expose an "
                    "'outcome_col' attribute. The common-population "
                    "validation requires every outcome estimator to "
                    "declare its outcome column."
                )

        if not isinstance(n_replicates, int) or isinstance(
            n_replicates, bool
        ):
            raise InvalidDesignError(
                f"n_replicates must be an integer, got "
                f"{type(n_replicates).__name__}."
            )
        if n_replicates <= 0:
            raise InvalidDesignError(
                f"n_replicates must be > 0, got {n_replicates}."
            )

        if alternative not in self._VALID_ALTERNATIVES:
            raise InvalidDesignError(
                f"alternative must be one of {self._VALID_ALTERNATIVES}, "
                f"got {alternative!r}."
            )

        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise InvalidDesignError(
                f"alpha must be a float, got {type(alpha).__name__}."
            )
        if not (0.0 < float(alpha) < 1.0):
            raise InvalidDesignError(
                f"alpha must be in (0, 1), got {alpha}."
            )

        # Deep-copy so each outcome's mutable estimator state is
        # isolated and the caller's instances are never fitted.
        self.estimators = copy.deepcopy(estimators)
        self.n_replicates = n_replicates
        self.alternative = alternative
        self.seed = seed
        self.alpha = float(alpha)

    def fit(
        self,
        assignment: CRDAssignment | BlockedAssignment,
    ) -> "MultiOutcomeRandomizationTest":
        """Run the joint simulation and compute raw and adjusted p-values.

        Parameters
        ----------
        assignment : CRDAssignment or BlockedAssignment
            Observed assignment whose data carries **all** outcome
            columns (the common population).

        Returns
        -------
        MultiOutcomeRandomizationTest
            Returns self.

        Raises
        ------
        DesignEstimatorMismatch
            If ``assignment`` is not a ``CRDAssignment`` or
            ``BlockedAssignment``.
        InvalidDesignError
            If any outcome column is missing from or has NaN in the
            common population, or if an estimator produces a
            multi-effect ``Results``.
        """
        if not isinstance(assignment, (CRDAssignment, BlockedAssignment)):
            raise DesignEstimatorMismatch(
                estimator_name=type(self).__name__,
                received_type=type(assignment).__name__,
                expected_type="CRDAssignment or BlockedAssignment",
            )

        names = list(self.estimators.keys())
        data = assignment.data_

        # --- Common population, defined before any draw -------------
        missing = {
            name: est.outcome_col
            for name, est in self.estimators.items()
            if est.outcome_col not in data.columns
        }
        if missing:
            raise InvalidDesignError(
                "MultiOutcomeRandomizationTest requires a common "
                "population: outcome column(s) "
                f"{sorted(set(missing.values()))!r} (for outcome(s) "
                f"{sorted(missing.keys())!r}) not found in "
                "assignment.data_. Define the common population first — "
                "one Assignment whose data carries every outcome column "
                "— instead of fitting each metric on its own population."
            )

        with_nan = [
            name
            for name, est in self.estimators.items()
            if data[est.outcome_col].isna().any()
        ]
        if with_nan:
            raise InvalidDesignError(
                f"Outcome column(s) for {with_nan!r} contain NaN in the "
                "common population. Restrict the data to a common "
                "complete-case population (or impute) before "
                "randomizing, so every outcome is evaluated on the same "
                "units; do not let each metric drop its own rows and "
                "draw from a different population."
            )

        # --- Shared plan: one population, one design version --------
        plan = RandomizationPlan(
            assignment,
            n_replicates=self.n_replicates,
            seed=self.seed,
        )

        # --- Observed statistics on the observed assignment ---------
        observed_results: dict[str, Results] = {}
        for name, est in self.estimators.items():
            est.fit(assignment)
            res = est.estimate()
            if res.ate is None:
                raise InvalidDesignError(
                    "MultiOutcomeRandomizationTest supports only "
                    "estimators producing a scalar ATE (Results.ate). "
                    f"The estimator for outcome '{name}' "
                    f"({type(est).__name__}) produced a multi-effect "
                    "Results (Results.effects)."
                )
            observed_results[name] = res

        # --- Joint simulation: all outcomes in the same world -------
        m = len(names)
        joint_null = np.empty((self.n_replicates, m), dtype=float)
        for r in range(self.n_replicates):
            shared = plan.assignment(r)
            for j, name in enumerate(names):
                est = self.estimators[name]
                est.fit(shared)
                joint_null[r, j] = est.estimate().ate

        observed_vec = np.array(
            [float(observed_results[name].ate) for name in names]
        )

        # --- Raw p-values: marginal null, Phipson & Smyth -----------
        p_values_raw = self._marginal_p_values(observed_vec, joint_null)

        # --- Adjusted p-values: the joint null feeds max-stat -------
        correction = MaxStatCorrection(
            alternative=self.alternative, alpha=self.alpha
        )
        adjusted_vec = correction.adjust(observed_vec, joint_null)

        self.assignment_ = assignment
        self.plan_ = plan
        self.outcome_names_ = names
        self.observed_statistics_ = {
            name: float(observed_results[name].ate) for name in names
        }
        self.joint_null_distribution_ = joint_null
        self.p_values_raw_ = {
            name: float(p_values_raw[j]) for j, name in enumerate(names)
        }
        self.p_values_adjusted_ = {
            name: float(adjusted_vec[j]) for j, name in enumerate(names)
        }
        self._observed_results = observed_results

        return self

    def estimate(self) -> MultiOutcomeResult:
        """Return the joint inference result.

        Builds, per outcome, a raw ``Results`` (unadjusted p-value) and
        an adjusted ``Results`` (max-stat p-value; raw preserved in
        ``extra["original_p_values"]``), plus the joint null matrix and
        a per-outcome table.

        Returns
        -------
        MultiOutcomeResult

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called.
        """
        self._check_is_fitted()

        raw_results: dict[str, Results] = {}
        adjusted_results: dict[str, Results] = {}
        rows: list[dict[str, Any]] = []

        for j, name in enumerate(self.outcome_names_):
            base = self._observed_results[name]
            marginal_null = self.joint_null_distribution_[:, j].copy()

            raw_results[name] = Results(
                ate=self.observed_statistics_[name],
                p_value=self.p_values_raw_[name],
                alpha=self.alpha,
                n_obs=base.n_obs,
                n_treated=base.n_treated,
                n_control=base.n_control,
                estimator_name=base.estimator_name,
                design_name=base.design_name,
                inference_name=type(self).__name__,
                extra={
                    "n_replicates": self.n_replicates,
                    "null_distribution": marginal_null,
                    "alternative": self.alternative,
                },
            )

            adjusted_results[name] = Results(
                ate=self.observed_statistics_[name],
                p_value=self.p_values_adjusted_[name],
                alpha=self.alpha,
                n_obs=base.n_obs,
                n_treated=base.n_treated,
                n_control=base.n_control,
                estimator_name=base.estimator_name,
                design_name=base.design_name,
                inference_name=type(self).__name__,
                extra={
                    "correction_method": "max_stat",
                    "original_p_values": dict(self.p_values_raw_),
                    "family_wise_alpha": self.alpha,
                    "n_tests": len(self.outcome_names_),
                    "n_replicates": self.n_replicates,
                    "null_distribution": marginal_null,
                    "alternative": self.alternative,
                },
            )

            rows.append(
                {
                    "outcome": name,
                    "ate": self.observed_statistics_[name],
                    "p_value": self.p_values_raw_[name],
                    "p_value_adjusted": self.p_values_adjusted_[name],
                    "significant": bool(
                        self.p_values_adjusted_[name] < self.alpha
                    ),
                }
            )

        table = pd.DataFrame(
            rows,
            columns=[
                "outcome",
                "ate",
                "p_value",
                "p_value_adjusted",
                "significant",
            ],
        )

        return MultiOutcomeResult(
            raw_results=raw_results,
            adjusted_results=adjusted_results,
            joint_null_distribution=self.joint_null_distribution_,
            alternative=self.alternative,
            alpha=self.alpha,
            n_replicates=self.n_replicates,
            table=table,
        )

    def _marginal_p_values(
        self,
        observed_vec: np.ndarray,
        joint_null: np.ndarray,
    ) -> np.ndarray:
        """Raw per-outcome p-values from each outcome's marginal null.

        Uses the same criterion as ``RandomizationTest`` (Phipson &
        Smyth continuity correction), applied column-wise.
        """
        if self.alternative == "greater":
            n_extreme = np.sum(joint_null >= observed_vec[None, :], axis=0)
        elif self.alternative == "less":
            n_extreme = np.sum(joint_null <= observed_vec[None, :], axis=0)
        else:  # "two-sided"
            n_extreme = np.sum(
                np.abs(joint_null) >= np.abs(observed_vec)[None, :],
                axis=0,
            )
        return (1 + n_extreme) / (1 + joint_null.shape[0])

    def _check_is_fitted(self) -> None:
        """Check if the test has been fitted.

        Raises
        ------
        NotFittedError
            If no attributes ending in underscore are found.
        """
        fitted_attrs = [
            attr for attr in self.__dict__ if attr.endswith("_")
        ]
        if not fitted_attrs:
            raise NotFittedError(
                class_name=type(self).__name__,
                required_methods=["fit"],
            )

    def __repr__(self) -> str:
        """Return string representation with parameters."""
        return (
            f"{type(self).__name__}(estimators="
            f"{list(self.estimators.keys())!r}, "
            f"n_replicates={self.n_replicates!r}, "
            f"alternative={self.alternative!r}, "
            f"seed={self.seed!r}, alpha={self.alpha!r})"
        )
