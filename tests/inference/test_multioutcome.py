"""Tests for skxperiments.inference.multioutcome.

Covers ``MaxStatCorrection``, ``MultiOutcomeRandomizationTest``, and
``MultiOutcomeResult``.
"""

import numpy as np
import pandas as pd
import pytest

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
from skxperiments.core.results import Results
from skxperiments.design.blocked_crd import BlockedCRD
from skxperiments.design.crd import CRD
from skxperiments.design.factorial import FactorialDesign
from skxperiments.estimators.blocked_difference_in_means import (
    BlockedDifferenceInMeans,
)
from skxperiments.estimators.difference_in_means import DifferenceInMeans
from skxperiments.inference import (
    MaxStatCorrection,
    MultipleTestingCorrection,
    MultiOutcomeRandomizationTest,
    MultiOutcomeResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_assignment_multioutcome(
    n: int = 100,
    seed: int = 42,
    ate1: float = 0.0,
    ate2: float = 0.0,
    shared_covariate: bool = True,
) -> CRDAssignment:
    """Build a CRDAssignment with two outcomes.

    When ``shared_covariate`` is True, both outcomes load on the same
    covariate ``x`` (y1 = x + ate1*T + e1, y2 = x + ate2*T + e2), which
    induces dependence between the two test statistics under the null.
    Otherwise y2 = e2 (independent of y1).
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    df = pd.DataFrame({"x": x})

    design = CRD(p=0.5, seed=seed)
    assignment = design.randomize(df)
    t = assignment.data_["treatment"].values.astype(float)

    df_out = assignment.data_.copy()
    df_out["y1"] = x + ate1 * t + rng.normal(0.0, 1.0, n)
    if shared_covariate:
        df_out["y2"] = x + ate2 * t + rng.normal(0.0, 1.0, n)
    else:
        df_out["y2"] = ate2 * t + rng.normal(0.0, 1.0, n)

    return CRDAssignment(
        data=df_out,
        treatment_col=assignment.treatment_col_,
        design=design,
        seed=seed,
    )


def _make_blocked_assignment_multioutcome(
    n_per_block: int = 10, n_blocks: int = 4, seed: int = 42
) -> BlockedAssignment:
    """Build a BlockedAssignment with two outcomes sharing a covariate."""
    rng = np.random.default_rng(seed)
    n = n_per_block * n_blocks
    df = pd.DataFrame(
        {
            "x": rng.normal(0.0, 1.0, n),
            "block": np.repeat(np.arange(n_blocks), n_per_block),
        }
    )
    design = BlockedCRD(block_col="block", p=0.5, seed=seed)
    assignment = design.randomize(df)
    t = assignment.data_["treatment"].values.astype(float)
    x = assignment.data_["x"].values

    df_out = assignment.data_.copy()
    df_out["y1"] = x + rng.normal(0.0, 1.0, n)
    df_out["y2"] = x + rng.normal(0.0, 1.0, n)

    return BlockedAssignment(
        data=df_out,
        treatment_col=assignment.treatment_col_,
        design=design,
        block_col=assignment.block_col_,
        block_sizes=assignment.block_sizes_,
        seed=seed,
    )


def _dim_two_outcomes() -> dict:
    return {
        "y1": DifferenceInMeans("y1"),
        "y2": DifferenceInMeans("y2"),
    }


# ---------------------------------------------------------------------------
# 1. MaxStatCorrection
# ---------------------------------------------------------------------------


class TestMaxStatCorrection:
    """Unit tests for the max-statistic adjustment."""

    def test_hand_computed_two_sided(self) -> None:
        """Adjusted p-values match a hand-computed maxT example."""
        observed = np.array([2.0, 1.0])
        joint_null = np.array(
            [
                [0.5, 0.2],
                [2.5, 0.1],
                [1.0, 1.5],
                [0.3, 0.4],
            ]
        )
        # Row-wise max |S|: [0.5, 2.5, 1.5, 0.4].
        # |obs1| = 2.0: 1 extreme row (2.5) -> (1+1)/(1+4) = 0.4
        # |obs2| = 1.0: 2 extreme rows (2.5, 1.5) -> (1+2)/5 = 0.6
        adjusted = MaxStatCorrection().adjust(observed, joint_null)
        np.testing.assert_allclose(adjusted, [0.4, 0.6])

    def test_hand_computed_greater(self) -> None:
        """'greater' uses the row-wise max without absolute values."""
        observed = np.array([1.0, -2.0])
        joint_null = np.array([[0.5, 3.0], [1.5, -1.0]])
        # Row-wise max: [3.0, 1.5].
        # obs1 = 1.0: 2 extreme -> 3/3 = 1.0
        # obs2 = -2.0: 2 extreme -> 3/3 = 1.0
        adjusted = MaxStatCorrection(alternative="greater").adjust(
            observed, joint_null
        )
        np.testing.assert_allclose(adjusted, [1.0, 1.0])

    def test_hand_computed_less(self) -> None:
        """'less' uses the row-wise min with the reversed inequality."""
        observed = np.array([0.4, 0.25])
        joint_null = np.array(
            [
                [0.5, 0.3],
                [1.5, 1.0],
                [-0.2, 0.8],
            ]
        )
        # Row-wise min: [0.3, 1.0, -0.2].
        # obs1 = 0.4: 2 extreme (0.3, -0.2) -> (1+2)/4 = 0.75
        # obs2 = 0.25: 1 extreme (-0.2) -> (1+1)/4 = 0.5
        adjusted = MaxStatCorrection(alternative="less").adjust(
            observed, joint_null
        )
        np.testing.assert_allclose(adjusted, [0.75, 0.5])

    def test_single_outcome_equals_marginal(self) -> None:
        """With one outcome, max-stat coincides with the marginal test."""
        rng = np.random.default_rng(0)
        joint_null = rng.normal(0.0, 1.0, (500, 1))
        observed = np.array([1.3])
        adjusted = MaxStatCorrection().adjust(observed, joint_null)
        expected = (1 + np.sum(np.abs(joint_null) >= 1.3)) / (1 + 500)
        np.testing.assert_allclose(adjusted, [expected])

    def test_adjusted_pvalues_bounded_away_from_zero(self) -> None:
        """The continuity correction bounds p-values away from zero."""
        observed = np.array([100.0, 100.0])
        joint_null = np.zeros((10, 2))
        adjusted = MaxStatCorrection().adjust(observed, joint_null)
        np.testing.assert_allclose(adjusted, [1 / 11, 1 / 11])

    def test_rejects_invalid_alternative(self) -> None:
        with pytest.raises(InvalidDesignError, match="alternative"):
            MaxStatCorrection(alternative="sideways")

    def test_rejects_invalid_alpha(self) -> None:
        with pytest.raises(InvalidDesignError, match="alpha"):
            MaxStatCorrection(alpha=1.5)

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(InvalidDesignError, match="columns"):
            MaxStatCorrection().adjust(
                np.array([1.0, 2.0]), np.zeros((10, 3))
            )

    def test_rejects_1d_joint_null(self) -> None:
        with pytest.raises(InvalidDesignError, match="2-D"):
            MaxStatCorrection().adjust(np.array([1.0]), np.zeros(10))

    def test_rejects_nan(self) -> None:
        with pytest.raises(InvalidDesignError, match="NaN"):
            MaxStatCorrection().adjust(
                np.array([np.nan]), np.zeros((10, 1))
            )


# ---------------------------------------------------------------------------
# 2. MultiOutcomeRandomizationTest creation
# ---------------------------------------------------------------------------


class TestMultiOutcomeCreation:
    """Tests for __init__ validation and estimator isolation."""

    def test_defaults(self) -> None:
        test = MultiOutcomeRandomizationTest(estimators=_dim_two_outcomes())
        assert test.n_replicates == 10_000
        assert test.alternative == "two-sided"
        assert test.alpha == 0.05
        assert test.seed is None

    def test_estimators_are_deep_copied(self) -> None:
        """The stored estimators are copies, not the user's instances."""
        estimators = _dim_two_outcomes()
        test = MultiOutcomeRandomizationTest(estimators=estimators)
        for name, est in estimators.items():
            assert test.estimators[name] is not est

    def test_rejects_non_dict(self) -> None:
        with pytest.raises(InvalidDesignError, match="dict"):
            MultiOutcomeRandomizationTest(
                estimators=[DifferenceInMeans("y1")]  # type: ignore[arg-type]
            )

    def test_rejects_empty_dict(self) -> None:
        with pytest.raises(InvalidDesignError, match="non-empty"):
            MultiOutcomeRandomizationTest(estimators={})

    def test_rejects_non_string_key(self) -> None:
        with pytest.raises(InvalidDesignError, match="strings"):
            MultiOutcomeRandomizationTest(
                estimators={1: DifferenceInMeans("y1")}  # type: ignore[dict-item]
            )

    def test_rejects_non_estimator_value(self) -> None:
        with pytest.raises(InvalidDesignError, match="BaseEstimator"):
            MultiOutcomeRandomizationTest(
                estimators={"y1": "not-an-estimator"}  # type: ignore[dict-item]
            )

    def test_rejects_estimator_without_outcome_col(self) -> None:
        """Estimators must declare outcome_col for population validation."""

        class NoOutcomeCol(BaseEstimator):
            def fit(self, assignment):  # pragma: no cover - not reached
                return self

            def estimate(self):  # pragma: no cover - not reached
                return Results(ate=0.0)

        with pytest.raises(InvalidDesignError, match="outcome_col"):
            MultiOutcomeRandomizationTest(
                estimators={"y1": NoOutcomeCol()}
            )

    @pytest.mark.parametrize("bad", [0, -1, 10.5, True])
    def test_rejects_invalid_n_replicates(self, bad: object) -> None:
        with pytest.raises(InvalidDesignError, match="n_replicates"):
            MultiOutcomeRandomizationTest(
                estimators=_dim_two_outcomes(),
                n_replicates=bad,  # type: ignore[arg-type]
            )

    def test_rejects_invalid_alternative(self) -> None:
        with pytest.raises(InvalidDesignError, match="alternative"):
            MultiOutcomeRandomizationTest(
                estimators=_dim_two_outcomes(), alternative="sideways"
            )

    def test_rejects_invalid_alpha(self) -> None:
        with pytest.raises(InvalidDesignError, match="alpha"):
            MultiOutcomeRandomizationTest(
                estimators=_dim_two_outcomes(), alpha=0.0
            )


# ---------------------------------------------------------------------------
# 3. Common-population validation
# ---------------------------------------------------------------------------


class TestCommonPopulation:
    """Missing outcomes must be caught before any replicate is drawn."""

    def test_missing_outcome_column_raises_before_plan(self) -> None:
        """A missing outcome column fails fast, naming the population."""
        assignment = _make_assignment_multioutcome()
        test = MultiOutcomeRandomizationTest(
            estimators={
                "y1": DifferenceInMeans("y1"),
                "ghost": DifferenceInMeans("y_missing"),
            },
            n_replicates=10,
            seed=0,
        )
        with pytest.raises(InvalidDesignError, match="common population"):
            test.fit(assignment)
        # The error fires before the shared plan is even built.
        assert not hasattr(test, "plan_")

    def test_missing_outcome_message_names_column_and_outcome(self) -> None:
        assignment = _make_assignment_multioutcome()
        test = MultiOutcomeRandomizationTest(
            estimators={"ghost": DifferenceInMeans("y_missing")},
            n_replicates=10,
        )
        with pytest.raises(InvalidDesignError, match="y_missing"):
            test.fit(assignment)

    def test_nan_outcome_raises_with_complete_case_guidance(self) -> None:
        """NaN outcomes must not let each metric drop its own rows."""
        assignment = _make_assignment_multioutcome()
        assignment.data_.loc[0, "y2"] = np.nan
        test = MultiOutcomeRandomizationTest(
            estimators=_dim_two_outcomes(), n_replicates=10, seed=0
        )
        with pytest.raises(InvalidDesignError, match="complete-case"):
            test.fit(assignment)

    def test_nan_error_mentions_common_population(self) -> None:
        assignment = _make_assignment_multioutcome()
        assignment.data_.loc[3, "y1"] = np.nan
        test = MultiOutcomeRandomizationTest(
            estimators=_dim_two_outcomes(), n_replicates=10, seed=0
        )
        with pytest.raises(InvalidDesignError, match="common"):
            test.fit(assignment)

    def test_rejects_factorial_assignment(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(0.0, 1.0, 40)})
        assignment = FactorialDesign(
            factors=["A", "B"], n_per_cell=10, seed=1
        ).randomize(df)
        test = MultiOutcomeRandomizationTest(
            estimators=_dim_two_outcomes(), n_replicates=10
        )
        with pytest.raises(DesignEstimatorMismatch):
            test.fit(assignment)

    def test_rejects_multi_effect_estimator(self) -> None:
        """Estimators producing Results.effects are rejected at fit."""

        class FakeMultiEffect(BaseEstimator):
            def __init__(self, outcome_col: str) -> None:
                self.outcome_col = outcome_col

            def fit(self, assignment):
                self.assignment_ = assignment
                return self

            def estimate(self):
                return Results(
                    effects={("A",): 1.0}, p_value={("A",): 0.5}
                )

        assignment = _make_assignment_multioutcome()
        test = MultiOutcomeRandomizationTest(
            estimators={"y1": FakeMultiEffect("y1")}, n_replicates=10
        )
        with pytest.raises(InvalidDesignError, match="scalar ATE"):
            test.fit(assignment)


# ---------------------------------------------------------------------------
# 4. Joint simulation behavior
# ---------------------------------------------------------------------------


class TestJointSimulation:
    """The joint null must come from shared draws of one population."""

    def test_joint_null_matches_shared_plan_draws(self) -> None:
        """Column j of the joint null equals refitting outcome j's
        estimator on plan.assignment(r) — the same world for all outcomes."""
        assignment = _make_assignment_multioutcome(n=80, seed=1)
        test = MultiOutcomeRandomizationTest(
            estimators=_dim_two_outcomes(), n_replicates=30, seed=11
        )
        test.fit(assignment)

        est1 = DifferenceInMeans("y1")
        est2 = DifferenceInMeans("y2")
        for r in range(30):
            shared = test.plan_.assignment(r)
            np.testing.assert_allclose(
                est1.fit(shared).estimate().ate,
                test.joint_null_distribution_[r, 0],
            )
            np.testing.assert_allclose(
                est2.fit(shared).estimate().ate,
                test.joint_null_distribution_[r, 1],
            )

    def test_joint_null_preserves_dependence(self) -> None:
        """Outcomes sharing a covariate have correlated null statistics."""
        assignment = _make_assignment_multioutcome(
            n=200, seed=2, shared_covariate=True
        )
        test = MultiOutcomeRandomizationTest(
            estimators=_dim_two_outcomes(), n_replicates=2000, seed=3
        )
        test.fit(assignment)
        corr = np.corrcoef(test.joint_null_distribution_.T)[0, 1]
        # Var(x) = Var(e) = 1 -> theoretical corr ~ 0.5.
        assert corr > 0.35

    def test_joint_null_independent_for_independent_outcomes(self) -> None:
        """Outcomes not sharing the covariate have ~independent nulls."""
        assignment = _make_assignment_multioutcome(
            n=200, seed=2, shared_covariate=False
        )
        test = MultiOutcomeRandomizationTest(
            estimators=_dim_two_outcomes(), n_replicates=2000, seed=3
        )
        test.fit(assignment)
        corr = np.corrcoef(test.joint_null_distribution_.T)[0, 1]
        assert abs(corr) < 0.15

    def test_reproducibility_same_seed(self) -> None:
        """Same master seed -> identical joint null distribution."""
        a1 = _make_assignment_multioutcome(seed=5)
        a2 = _make_assignment_multioutcome(seed=5)
        t1 = MultiOutcomeRandomizationTest(
            estimators=_dim_two_outcomes(), n_replicates=50, seed=9
        ).fit(a1)
        t2 = MultiOutcomeRandomizationTest(
            estimators=_dim_two_outcomes(), n_replicates=50, seed=9
        ).fit(a2)
        np.testing.assert_array_equal(
            t1.joint_null_distribution_, t2.joint_null_distribution_
        )

    def test_different_seed_different_null(self) -> None:
        a1 = _make_assignment_multioutcome(seed=5)
        a2 = _make_assignment_multioutcome(seed=5)
        t1 = MultiOutcomeRandomizationTest(
            estimators=_dim_two_outcomes(), n_replicates=50, seed=9
        ).fit(a1)
        t2 = MultiOutcomeRandomizationTest(
            estimators=_dim_two_outcomes(), n_replicates=50, seed=10
        ).fit(a2)
        assert not np.array_equal(
            t1.joint_null_distribution_, t2.joint_null_distribution_
        )

    def test_works_on_blocked_assignment(self) -> None:
        """Blocked assignments keep their block structure in the plan."""
        assignment = _make_blocked_assignment_multioutcome()
        test = MultiOutcomeRandomizationTest(
            estimators={
                "y1": BlockedDifferenceInMeans("y1"),
                "y2": BlockedDifferenceInMeans("y2"),
            },
            n_replicates=30,
            seed=4,
        )
        result = test.fit(assignment).estimate()
        assert result.joint_null_distribution.shape == (30, 2)
        for r in range(30):
            rep = test.plan_.assignment(r)
            assert isinstance(rep, BlockedAssignment)

    def test_plan_binding_enforced_after_fit(self) -> None:
        """The fitted plan refuses to draw once the population mutates."""
        assignment = _make_assignment_multioutcome(n=60, seed=6)
        test = MultiOutcomeRandomizationTest(
            estimators=_dim_two_outcomes(), n_replicates=20, seed=0
        ).fit(assignment)
        assignment.data_.loc[0, "x"] = 12345.0
        with pytest.raises(InvalidDesignError, match="population"):
            test.plan_.assignment(0)


# ---------------------------------------------------------------------------
# 5. Estimator state isolation
# ---------------------------------------------------------------------------


class TestEstimatorIsolation:
    """Each outcome's mutable estimator state is isolated."""

    def test_user_estimators_never_fitted(self) -> None:
        """The instances passed in are deep-copied, never fitted."""
        estimators = _dim_two_outcomes()
        assignment = _make_assignment_multioutcome(n=60, seed=7)
        MultiOutcomeRandomizationTest(
            estimators=estimators, n_replicates=20, seed=0
        ).fit(assignment)
        for est in estimators.values():
            fitted_attrs = [
                a for a in est.__dict__ if a.endswith("_")
            ]
            assert fitted_attrs == []

    def test_outcome_estimators_are_distinct_objects(self) -> None:
        """Each outcome gets its own estimator copy."""
        estimators = _dim_two_outcomes()
        test = MultiOutcomeRandomizationTest(
            estimators=estimators, n_replicates=10
        )
        assert test.estimators["y1"] is not test.estimators["y2"]

    def test_internal_estimators_fitted_after_fit(self) -> None:
        """The internal copies were used (and hold last-replicate state)."""
        assignment = _make_assignment_multioutcome(n=60, seed=8)
        test = MultiOutcomeRandomizationTest(
            estimators=_dim_two_outcomes(), n_replicates=20, seed=0
        ).fit(assignment)
        for est in test.estimators.values():
            assert hasattr(est, "ate_")


# ---------------------------------------------------------------------------
# 6. Raw and adjusted p-values
# ---------------------------------------------------------------------------


class TestPValues:
    """Raw results are preserved; the joint null feeds the correction."""

    def _fitted(self, seed: int = 0) -> MultiOutcomeRandomizationTest:
        assignment = _make_assignment_multioutcome(n=80, seed=1)
        return MultiOutcomeRandomizationTest(
            estimators=_dim_two_outcomes(), n_replicates=200, seed=seed
        ).fit(assignment)

    def test_raw_pvalues_match_marginal_null(self) -> None:
        """Raw p-values equal Phipson-Smyth on each marginal column."""
        test = self._fitted()
        joint = test.joint_null_distribution_
        for j, name in enumerate(test.outcome_names_):
            obs = test.observed_statistics_[name]
            n_extreme = int(np.sum(np.abs(joint[:, j]) >= abs(obs)))
            expected = (1 + n_extreme) / (1 + joint.shape[0])
            assert test.p_values_raw_[name] == pytest.approx(expected)

    def test_adjusted_pvalues_come_from_joint_null(self) -> None:
        """Adjusted p-values equal the max-stat formula recomputed from
        the joint null matrix — not from sorted p-values."""
        test = self._fitted()
        joint = test.joint_null_distribution_
        max_per_replicate = np.max(np.abs(joint), axis=1)
        for name in test.outcome_names_:
            obs = abs(test.observed_statistics_[name])
            n_extreme = int(np.sum(max_per_replicate >= obs))
            expected = (1 + n_extreme) / (1 + joint.shape[0])
            assert test.p_values_adjusted_[name] == pytest.approx(expected)

    def test_adjusted_geq_raw(self) -> None:
        """Max-stat adjustment never shrinks a p-value."""
        test = self._fitted()
        for name in test.outcome_names_:
            assert (
                test.p_values_adjusted_[name]
                >= test.p_values_raw_[name]
            )

    def test_adjusted_differs_from_sorted_pvalue_correction(self) -> None:
        """Max-stat adjusted p-values are not Holm-on-raw-p-values."""
        test = self._fitted()
        raw_list = [test.p_values_raw_[n] for n in test.outcome_names_]
        holm = np.array([min(1.0, p * (2 - i)) for i, p in enumerate(sorted(raw_list))])
        adjusted = np.array(
            [test.p_values_adjusted_[n] for n in test.outcome_names_]
        )
        assert not np.allclose(np.sort(adjusted), np.sort(holm))

    def test_single_outcome_adjusted_equals_raw(self) -> None:
        """With one outcome the max-stat correction is the identity."""
        assignment = _make_assignment_multioutcome(n=60, seed=3)
        test = MultiOutcomeRandomizationTest(
            estimators={"y1": DifferenceInMeans("y1")},
            n_replicates=100,
            seed=0,
        ).fit(assignment)
        assert (
            test.p_values_adjusted_["y1"]
            == test.p_values_raw_["y1"]
        )

    def test_strong_effect_survives_correction(self) -> None:
        """A large true effect on y1 stays significant after adjustment."""
        assignment = _make_assignment_multioutcome(
            n=200, seed=10, ate1=1.0, ate2=0.0
        )
        result = (
            MultiOutcomeRandomizationTest(
                estimators=_dim_two_outcomes(),
                n_replicates=500,
                seed=0,
            )
            .fit(assignment)
            .estimate()
        )
        assert result.significant == ["y1"]
        assert result.adjusted_results["y1"].p_value < 0.05
        assert result.adjusted_results["y2"].p_value > 0.05


# ---------------------------------------------------------------------------
# 7. Result surface
# ---------------------------------------------------------------------------


class TestResultSurface:
    """Tests for MultiOutcomeResult and the per-outcome Results."""

    def _result(self) -> MultiOutcomeResult:
        assignment = _make_assignment_multioutcome(n=80, seed=1)
        return (
            MultiOutcomeRandomizationTest(
                estimators=_dim_two_outcomes(),
                n_replicates=100,
                seed=0,
            )
            .fit(assignment)
            .estimate()
        )

    def test_estimate_before_fit_raises(self) -> None:
        test = MultiOutcomeRandomizationTest(
            estimators=_dim_two_outcomes(), n_replicates=10
        )
        with pytest.raises(NotFittedError):
            test.estimate()

    def test_result_type_and_joint_null(self) -> None:
        result = self._result()
        assert isinstance(result, MultiOutcomeResult)
        assert result.joint_null_distribution.shape == (100, 2)
        assert result.n_replicates == 100

    def test_raw_results_preserved(self) -> None:
        """Raw results carry the uncorrected p-values and no correction
        metadata — they remain valid input for MultipleTestingCorrection."""
        result = self._result()
        raw = result.raw_results
        assert set(raw) == {"y1", "y2"}
        for name, res in raw.items():
            assert res.p_value == pytest.approx(
                result.table.set_index("outcome").loc[name, "p_value"]
            )
            assert res.extra is not None
            assert "correction_method" not in res.extra
            assert res.extra["n_replicates"] == 100
            assert res.extra["null_distribution"].shape == (100,)
        # Raw results remain correctable by the sorted-p machinery.
        corrected = MultipleTestingCorrection(method="holm").correct(
            list(raw.values())
        )
        assert len(corrected) == 2

    def test_adjusted_results_carry_correction_metadata(self) -> None:
        """Adjusted results record the correction and the raw p-values."""
        result = self._result()
        for name, res in result.adjusted_results.items():
            assert res.extra is not None
            assert res.extra["correction_method"] == "max_stat"
            assert res.extra["n_tests"] == 2
            assert res.extra["family_wise_alpha"] == 0.05
            assert set(res.extra["original_p_values"]) == {"y1", "y2"}
            assert res.alpha == 0.05

    def test_adjusted_results_rejected_by_mtc(self) -> None:
        """The reserved keys prevent a second, stacked correction."""
        result = self._result()
        with pytest.raises(InvalidDesignError, match="reserved"):
            MultipleTestingCorrection(method="holm").correct(
                list(result.adjusted_results.values())
            )

    def test_table_columns_and_order(self) -> None:
        result = self._result()
        assert list(result.table.columns) == [
            "outcome",
            "ate",
            "p_value",
            "p_value_adjusted",
            "significant",
        ]
        assert list(result.table["outcome"]) == ["y1", "y2"]

    def test_to_dataframe_is_copy(self) -> None:
        result = self._result()
        df = result.to_dataframe()
        df.loc[0, "ate"] = 999.0
        assert result.table.loc[0, "ate"] != 999.0

    def test_to_dict_keys(self) -> None:
        d = self._result().to_dict()
        assert set(d) == {
            "correction",
            "alpha",
            "n_outcomes",
            "n_replicates",
            "alternative",
            "significant",
        }
        assert d["correction"] == "max_stat"
        assert d["n_outcomes"] == 2

    def test_summary_prints_and_returns_self(self, capsys) -> None:
        result = self._result()
        assert result.summary() is result
        out = capsys.readouterr().out
        assert "Multi-Outcome Randomization Test" in out
        assert "max_stat" in out

    def test_observed_statistics_match_independent_fit(self) -> None:
        """Observed ATEs equal fitting each estimator on the assignment."""
        assignment = _make_assignment_multioutcome(n=80, seed=1)
        test = MultiOutcomeRandomizationTest(
            estimators=_dim_two_outcomes(), n_replicates=10, seed=0
        ).fit(assignment)
        for name in ("y1", "y2"):
            expected = (
                DifferenceInMeans(name).fit(assignment).estimate().ate
            )
            assert test.observed_statistics_[name] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 8. Numerics (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMultiOutcomeNumerics:
    """Statistical property tests for the joint test (slow)."""

    def test_maxstat_controls_fwer_under_complete_null(self) -> None:
        """Under the complete null, FWER of the adjusted test ~ alpha.

        150 simulations, 3 correlated null outcomes, n_replicates=400.
        The family-wise rejection rate at alpha=0.05 must stay near
        0.05 (bound 0.10 is ~3 Monte Carlo SEs above 0.05).
        """
        n_sims = 150
        rejections = 0
        for i in range(n_sims):
            rng = np.random.default_rng(20_000 + i)
            n = 50
            x = rng.normal(0.0, 1.0, n)
            df = pd.DataFrame({"x": x})
            design = CRD(p=0.5, seed=30_000 + i)
            base = design.randomize(df)
            df_out = base.data_.copy()
            df_out["y1"] = x + rng.normal(0.0, 1.0, n)
            df_out["y2"] = x + rng.normal(0.0, 1.0, n)
            df_out["y3"] = x + rng.normal(0.0, 1.0, n)
            assignment = CRDAssignment(
                data=df_out,
                treatment_col="treatment",
                design=design,
                seed=30_000 + i,
            )
            test = MultiOutcomeRandomizationTest(
                estimators={
                    "y1": DifferenceInMeans("y1"),
                    "y2": DifferenceInMeans("y2"),
                    "y3": DifferenceInMeans("y3"),
                },
                n_replicates=400,
                seed=40_000 + i,
            )
            result = test.fit(assignment).estimate()
            if result.significant:
                rejections += 1

        fwer = rejections / n_sims
        assert fwer <= 0.10, (
            f"Max-stat FWER too high under complete null: {fwer:.3f} "
            f"({rejections}/{n_sims})"
        )

    def test_raw_family_rejects_more_than_adjusted(self) -> None:
        """Without correction, the family-wise rate exceeds alpha."""
        n_sims = 100
        raw_rejections = 0
        adjusted_rejections = 0
        for i in range(n_sims):
            rng = np.random.default_rng(50_000 + i)
            n = 50
            x = rng.normal(0.0, 1.0, n)
            df = pd.DataFrame({"x": x})
            design = CRD(p=0.5, seed=60_000 + i)
            base = design.randomize(df)
            df_out = base.data_.copy()
            df_out["y1"] = x + rng.normal(0.0, 1.0, n)
            df_out["y2"] = x + rng.normal(0.0, 1.0, n)
            df_out["y3"] = x + rng.normal(0.0, 1.0, n)
            assignment = CRDAssignment(
                data=df_out,
                treatment_col="treatment",
                design=design,
                seed=60_000 + i,
            )
            test = MultiOutcomeRandomizationTest(
                estimators={
                    "y1": DifferenceInMeans("y1"),
                    "y2": DifferenceInMeans("y2"),
                    "y3": DifferenceInMeans("y3"),
                },
                n_replicates=400,
                seed=70_000 + i,
            ).fit(assignment)
            if any(p < 0.05 for p in test.p_values_raw_.values()):
                raw_rejections += 1
            if any(p < 0.05 for p in test.p_values_adjusted_.values()):
                adjusted_rejections += 1

        assert raw_rejections > adjusted_rejections
