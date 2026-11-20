"""Tests for skxperiments.core.plan.RandomizationPlan."""

import numpy as np
import pandas as pd
import pytest

from skxperiments.core.assignment import (
    BlockedAssignment,
    CRDAssignment,
)
from skxperiments.core.exceptions import (
    DesignEstimatorMismatch,
    InvalidDesignError,
)
from skxperiments.core.plan import RandomizationPlan
from skxperiments.design.blocked_crd import BlockedCRD
from skxperiments.design.crd import CRD
from skxperiments.design.factorial import FactorialDesign


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_crd_assignment(n: int = 60, seed: int = 42) -> CRDAssignment:
    """Build a CRDAssignment with a covariate and an outcome."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(0.0, 1.0, n)})
    design = CRD(p=0.5, seed=seed)
    assignment = design.randomize(df)
    df_out = assignment.data_.copy()
    df_out["y"] = df_out["x"] + rng.normal(0.0, 1.0, n)
    return CRDAssignment(
        data=df_out,
        treatment_col=assignment.treatment_col_,
        design=design,
        seed=seed,
    )


def _make_blocked_assignment(
    n_per_block: int = 10, n_blocks: int = 4, seed: int = 42
) -> BlockedAssignment:
    """Build a BlockedAssignment with a covariate and an outcome."""
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
    df_out = assignment.data_.copy()
    df_out["y"] = df_out["x"] + rng.normal(0.0, 1.0, n)
    return BlockedAssignment(
        data=df_out,
        treatment_col=assignment.treatment_col_,
        design=design,
        block_col=assignment.block_col_,
        block_sizes=assignment.block_sizes_,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# 1. Creation and validation
# ---------------------------------------------------------------------------


class TestRandomizationPlanCreation:
    """Tests for RandomizationPlan.__init__ validations."""

    def test_defaults_stored(self) -> None:
        """n_replicates and seed are stored; seeds are pre-generated."""
        plan = RandomizationPlan(_make_crd_assignment(), 100, seed=0)
        assert plan.n_replicates == 100
        assert plan.seed == 0
        assert plan.replicate_seeds_.shape == (100,)

    def test_len(self) -> None:
        """len(plan) equals n_replicates."""
        assert len(RandomizationPlan(_make_crd_assignment(), 7)) == 7

    def test_repr(self) -> None:
        """repr names the design and the replicate count."""
        r = repr(RandomizationPlan(_make_crd_assignment(), 10, seed=1))
        assert "RandomizationPlan" in r
        assert "CRD" in r

    def test_rejects_factorial_assignment(self) -> None:
        """FactorialAssignment is rejected (matches RandomizationTest v1)."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(0.0, 1.0, 40)})
        assignment = FactorialDesign(
            factors=["A", "B"], n_per_cell=10, seed=1
        ).randomize(df)
        with pytest.raises(DesignEstimatorMismatch):
            RandomizationPlan(assignment, 10)

    def test_rejects_assignment_without_design(self) -> None:
        """An assignment with design_=None cannot anchor a plan."""
        df = pd.DataFrame({"x": [1, 2, 3, 4], "treatment": [1, 0, 1, 0]})
        assignment = CRDAssignment(data=df, treatment_col="treatment", design=None)
        with pytest.raises(InvalidDesignError, match="design"):
            RandomizationPlan(assignment, 10)

    @pytest.mark.parametrize("bad", [0, -1, 10.5, True, "10"])
    def test_rejects_invalid_n_replicates(self, bad: object) -> None:
        """n_replicates must be a positive integer (bool rejected)."""
        with pytest.raises(InvalidDesignError, match="n_replicates"):
            RandomizationPlan(_make_crd_assignment(), bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Replicate generation
# ---------------------------------------------------------------------------


class TestReplicateGeneration:
    """Tests for RandomizationPlan.assignment(replicate_id)."""

    def test_same_replicate_id_same_assignment(self) -> None:
        """The same replicate_id always reproduces the same assignment."""
        plan = RandomizationPlan(_make_crd_assignment(), 10, seed=0)
        t1 = plan.assignment(7).data_["treatment"].values
        t2 = plan.assignment(7).data_["treatment"].values
        np.testing.assert_array_equal(t1, t2)

    def test_replicate_matches_direct_draw_with_replicate_seed(self) -> None:
        """plan.assignment(r) equals assignment.draw(seed=replicate_seeds_[r])."""
        base = _make_crd_assignment()
        plan = RandomizationPlan(base, 10, seed=3)
        for r in range(10):
            via_plan = plan.assignment(r).data_["treatment"].values
            via_draw = base.draw(
                seed=int(plan.replicate_seeds_[r])
            ).data_["treatment"].values
            np.testing.assert_array_equal(via_plan, via_draw)

    def test_distinct_replicates_differ(self) -> None:
        """Different replicate_ids produce different treatment vectors."""
        plan = RandomizationPlan(_make_crd_assignment(), 10, seed=0)
        t0 = plan.assignment(0).data_["treatment"].values
        t1 = plan.assignment(1).data_["treatment"].values
        assert not np.array_equal(t0, t1)

    def test_replicates_preserve_design_constraints(self) -> None:
        """Every replicate keeps n_treated and shares the population data."""
        base = _make_crd_assignment(n=60)
        plan = RandomizationPlan(base, 25, seed=1)
        for r in range(len(plan)):
            rep = plan.assignment(r)
            assert isinstance(rep, CRDAssignment)
            assert rep.n_units_ == 60
            assert rep.n_treated_ == 30
            # Same population: covariate columns identical to the bound one.
            pd.testing.assert_series_equal(
                rep.data_["x"], base.data_["x"]
            )

    def test_blocked_replicates_preserve_block_structure(self) -> None:
        """Blocked replicates rerandomize within blocks only."""
        base = _make_blocked_assignment()
        plan = RandomizationPlan(base, 10, seed=2)
        for r in range(len(plan)):
            rep = plan.assignment(r)
            assert isinstance(rep, BlockedAssignment)
            assert rep.block_sizes_ == base.block_sizes_
            # Within-block treated counts are preserved.
            for block, size in base.block_sizes_.items():
                expected = int(round(0.5 * size))
                actual = int(
                    rep.data_.loc[
                        rep.data_["block"] == block, "treatment"
                    ].sum()
                )
                assert actual == expected

    def test_reproducibility_same_seed(self) -> None:
        """Two plans with the same seed generate identical replicates."""
        base = _make_crd_assignment()
        p1 = RandomizationPlan(base, 10, seed=5)
        p2 = RandomizationPlan(base, 10, seed=5)
        np.testing.assert_array_equal(p1.replicate_seeds_, p2.replicate_seeds_)
        for r in range(10):
            np.testing.assert_array_equal(
                p1.assignment(r).data_["treatment"].values,
                p2.assignment(r).data_["treatment"].values,
            )

    def test_different_seed_different_plan(self) -> None:
        """Different master seeds give different replicate seeds."""
        base = _make_crd_assignment()
        p1 = RandomizationPlan(base, 10, seed=5)
        p2 = RandomizationPlan(base, 10, seed=6)
        assert not np.array_equal(p1.replicate_seeds_, p2.replicate_seeds_)

    @pytest.mark.parametrize("bad", [-1, 10, 100, 1.5, True, "0"])
    def test_replicate_id_out_of_range_or_wrong_type(
        self, bad: object
    ) -> None:
        """replicate_id must be an int in [0, n_replicates)."""
        plan = RandomizationPlan(_make_crd_assignment(), 10, seed=0)
        with pytest.raises(InvalidDesignError, match="replicate_id"):
            plan.assignment(bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. Population and design-version binding
# ---------------------------------------------------------------------------


class TestBinding:
    """The plan must refuse to draw once its bindings are broken."""

    def test_population_mutation_invalidates_plan(self) -> None:
        """Mutating assignment.data_ after creation breaks the binding."""
        base = _make_crd_assignment()
        plan = RandomizationPlan(base, 10, seed=0)
        plan.assignment(0)  # works before mutation
        base.data_.loc[0, "x"] = 999.0
        with pytest.raises(InvalidDesignError, match="population"):
            plan.assignment(0)

    def test_treatment_column_mutation_invalidates_plan(self) -> None:
        """The fingerprint covers the observed treatment vector too."""
        base = _make_crd_assignment()
        plan = RandomizationPlan(base, 10, seed=0)
        base.data_.loc[0, "treatment"] = (
            1 - base.data_.loc[0, "treatment"]
        )
        with pytest.raises(InvalidDesignError, match="population"):
            plan.assignment(0)

    def test_design_version_mutation_invalidates_plan(self) -> None:
        """Changing a mechanism parameter breaks the design binding."""
        base = _make_crd_assignment()
        design = base.design_
        plan = RandomizationPlan(base, 10, seed=0)
        design.set_params(p=0.7)
        with pytest.raises(InvalidDesignError, match="design version"):
            plan.assignment(0)

    def test_design_seed_change_does_not_invalidate(self) -> None:
        """The design's own seed is not part of the design version."""
        base = _make_crd_assignment()
        design = base.design_
        plan = RandomizationPlan(base, 10, seed=0)
        design.seed = 999
        rep = plan.assignment(0)  # must not raise
        assert isinstance(rep, CRDAssignment)

    def test_validate_passes_when_untouched(self) -> None:
        """validate() is a no-op while the bindings hold."""
        plan = RandomizationPlan(_make_crd_assignment(), 10, seed=0)
        plan.validate()

    def test_fingerprints_exposed(self) -> None:
        """Population and design fingerprints are inspectable attributes."""
        plan = RandomizationPlan(_make_crd_assignment(), 10, seed=0)
        assert isinstance(plan.population_fingerprint_, str)
        assert isinstance(plan.design_fingerprint_, str)
        assert plan.design_version_["p"] == 0.5
        assert "seed" not in plan.design_version_
