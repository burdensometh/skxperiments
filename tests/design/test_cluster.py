"""Tests for skxperiments.design.cluster.ClusterDesign and
skxperiments.core.assignment.ClusterAssignment.
"""

import numpy as np
import pandas as pd
import pytest

from skxperiments.core.assignment import (
    BlockedAssignment,
    ClusterAssignment,
    CRDAssignment,
)
from skxperiments.core.exceptions import (
    InsufficientDataError,
    InvalidDesignError,
)
from skxperiments.design.blocked_crd import BlockedCRD
from skxperiments.design.cluster import ClusterDesign
from skxperiments.design.crd import CRD
from skxperiments.design.factorial import FactorialDesign
from skxperiments.design.rerandomized_crd import ReRandomizedCRD


# --- Helpers ---


def _df_stores(
    sizes: list[int] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Build an individual-level DataFrame: one row per customer, with
    a ``store_id`` cluster column. ``sizes[i]`` is the number of
    customers of store i.
    """
    if sizes is None:
        sizes = [5] * 10
    rng = np.random.default_rng(seed)
    store_ids: list[str] = []
    for i, s in enumerate(sizes):
        store_ids.extend([f"store_{i}"] * s)
    n = len(store_ids)
    return pd.DataFrame(
        {
            "store_id": store_ids,
            "customer_id": np.arange(n),
            "x": rng.normal(size=n),
        }
    )


def _df_stores_blocked(
    sizes: list[int] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Individual-level df with stores nested in two regions (blocks):
    the first half of the stores in region "A", the rest in "B".
    """
    df = _df_stores(sizes=sizes, seed=seed)
    n_clusters = df["store_id"].nunique()
    half = n_clusters // 2
    store_block = {
        f"store_{i}": ("A" if i < half else "B") for i in range(n_clusters)
    }
    df["region"] = df["store_id"].map(store_block)
    return df


def _cluster_treatment_map(assignment: ClusterAssignment) -> dict:
    """Map cluster label -> its (unique) treatment label."""
    data = assignment.data_
    return data.groupby(assignment.cluster_col_)[
        assignment.treatment_col_
    ].first().to_dict()


# --- ClusterDesign creation ---


class TestClusterDesignCreation:
    """Tests for ClusterDesign instantiation."""

    def test_basic_creation_crd_base(self) -> None:
        """Should instantiate wrapping a CRD base design."""
        base = CRD(p=0.5, seed=42)
        design = ClusterDesign(cluster_col="store_id", base_design=base)
        assert design.cluster_col == "store_id"
        assert design.base_design is base

    def test_basic_creation_blocked_base(self) -> None:
        """Should instantiate wrapping a BlockedCRD base design."""
        base = BlockedCRD(block_col="region", p=0.5, seed=42)
        design = ClusterDesign(cluster_col="store_id", base_design=base)
        assert design.base_design is base

    def test_repr_includes_base_design(self) -> None:
        """repr should show cluster_col and the wrapped base design."""
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        r = repr(design)
        assert "ClusterDesign" in r
        assert "store_id" in r
        assert "CRD" in r

    def test_rerandomized_base_rejected(self) -> None:
        """ReRandomizedCRD is not a supported base design in v1."""
        with pytest.raises(InvalidDesignError):
            ClusterDesign(
                cluster_col="store_id",
                base_design=ReRandomizedCRD(
                    p=0.5, covariates=["x"], threshold=0.1
                ),
            )

    def test_factorial_base_rejected(self) -> None:
        """FactorialDesign is not a supported base design in v1."""
        with pytest.raises(InvalidDesignError):
            ClusterDesign(
                cluster_col="store_id",
                base_design=FactorialDesign(factors=["A", "B"], n_per_cell=4),
            )

    def test_non_design_base_rejected(self) -> None:
        """A non-design object must be rejected."""
        with pytest.raises(InvalidDesignError):
            ClusterDesign(cluster_col="store_id", base_design="CRD")


# --- ClusterDesign.randomize with CRD base ---


class TestClusterDesignRandomizeCRD:
    """Tests for ClusterDesign.randomize with a CRD base design."""

    def test_returns_cluster_assignment(self) -> None:
        """randomize must return a ClusterAssignment."""
        df = _df_stores()
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        assignment = design.randomize(df)
        assert isinstance(assignment, ClusterAssignment)

    def test_does_not_modify_input_dataframe(self) -> None:
        """Should not modify the input DataFrame."""
        df = _df_stores()
        snapshot = df.copy()
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        _ = design.randomize(df)
        pd.testing.assert_frame_equal(df, snapshot)

    def test_treatment_column_added(self) -> None:
        """Output assignment must carry the treatment column."""
        df = _df_stores()
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        assignment = design.randomize(df)
        assert "treatment" in assignment.data_.columns
        assert set(assignment.data_["treatment"].unique()) <= {0, 1}

    def test_custom_treatment_col_propagates(self) -> None:
        """The base design's treatment_col names the output column."""
        df = _df_stores()
        design = ClusterDesign(
            cluster_col="store_id",
            base_design=CRD(p=0.5, seed=42, treatment_col="T"),
        )
        assignment = design.randomize(df)
        assert assignment.treatment_col_ == "T"
        assert "T" in assignment.data_.columns

    def test_treatment_constant_within_cluster(self) -> None:
        """All individuals of a cluster must share one label."""
        df = _df_stores(sizes=[3, 7, 2, 9, 4, 5])
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        assignment = design.randomize(df)
        n_unique = assignment.data_.groupby("store_id")[
            "treatment"
        ].nunique()
        assert (n_unique == 1).all()

    def test_treated_cluster_count_matches_p(self) -> None:
        """Number of treated clusters must equal round(p * n_clusters)."""
        df = _df_stores(sizes=[4] * 10)
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        assignment = design.randomize(df)
        assert assignment.n_clusters_ == 10
        assert assignment.n_clusters_treated_ == 5
        assert assignment.n_clusters_control_ == 5

    def test_treated_cluster_count_matches_n_treated(self) -> None:
        """Absolute n_treated of the base design counts clusters."""
        df = _df_stores(sizes=[4] * 10)
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(n_treated=3, seed=42)
        )
        assignment = design.randomize(df)
        assert assignment.n_clusters_treated_ == 3
        assert assignment.n_clusters_control_ == 7

    def test_cluster_sizes_and_counts(self) -> None:
        """cluster_sizes_ must map each cluster to its row count."""
        sizes = [3, 7, 2, 9, 4, 5]
        df = _df_stores(sizes=sizes)
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        assignment = design.randomize(df)
        expected = {f"store_{i}": s for i, s in enumerate(sizes)}
        assert assignment.cluster_sizes_ == expected
        assert assignment.n_clusters_ == len(sizes)

    def test_individual_counts_preserved(self) -> None:
        """n_units_/n_treated_ count individuals, not clusters."""
        sizes = [3, 7, 2, 9, 4, 5]
        df = _df_stores(sizes=sizes)
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        assignment = design.randomize(df)
        assert assignment.n_units_ == sum(sizes)
        treated_clusters = {
            k for k, v in _cluster_treatment_map(assignment).items()
            if v == 1
        }
        expected_treated = sum(
            s for i, s in enumerate(sizes) if f"store_{i}" in treated_clusters
        )
        assert assignment.n_treated_ == expected_treated
        assert assignment.n_control_ == sum(sizes) - expected_treated

    def test_within_cluster_identity_preserved(self) -> None:
        """Every input row must appear, unchanged, in the output."""
        df = _df_stores(sizes=[3, 5, 2])
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        assignment = design.randomize(df)
        out = assignment.data_
        assert len(out) == len(df)
        assert list(out["customer_id"]) == list(df["customer_id"])
        assert list(out["store_id"]) == list(df["store_id"])
        pd.testing.assert_series_equal(out["x"], df["x"], check_names=False)

    def test_reproducibility_same_seed(self) -> None:
        """Same seed must produce identical labelings."""
        df = _df_stores()
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        a1 = design.randomize(df)
        a2 = design.randomize(df)
        pd.testing.assert_frame_equal(a1.data_, a2.data_)

    def test_missing_cluster_col_raises(self) -> None:
        """Missing cluster column must raise InvalidDesignError."""
        df = _df_stores().drop(columns=["store_id"])
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        with pytest.raises(InvalidDesignError):
            design.randomize(df)

    def test_nan_cluster_col_raises(self) -> None:
        """NaN cluster labels must raise InvalidDesignError."""
        df = _df_stores()
        df.loc[0, "store_id"] = None
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        with pytest.raises(InvalidDesignError):
            design.randomize(df)

    def test_existing_treatment_col_raises(self) -> None:
        """A pre-existing treatment column must raise."""
        df = _df_stores()
        df["treatment"] = 0
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        with pytest.raises(InvalidDesignError):
            design.randomize(df)

    def test_too_few_clusters_raises(self) -> None:
        """A single cluster cannot be randomized (base CRD check)."""
        df = _df_stores(sizes=[10])
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        with pytest.raises(InsufficientDataError):
            design.randomize(df)


# --- ClusterDesign.randomize with BlockedCRD base ---


class TestClusterDesignRandomizeBlocked:
    """Tests for ClusterDesign.randomize with a BlockedCRD base."""

    def test_returns_cluster_assignment(self) -> None:
        """Blocked base design must also yield a ClusterAssignment."""
        df = _df_stores_blocked(sizes=[4] * 10)
        design = ClusterDesign(
            cluster_col="store_id",
            base_design=BlockedCRD(block_col="region", p=0.5, seed=42),
        )
        assignment = design.randomize(df)
        assert isinstance(assignment, ClusterAssignment)

    def test_proportion_per_block_at_cluster_level(self) -> None:
        """Within each block, round(p * n_clusters_block) clusters
        must be treated."""
        df = _df_stores_blocked(sizes=[4] * 10)
        design = ClusterDesign(
            cluster_col="store_id",
            base_design=BlockedCRD(block_col="region", p=0.5, seed=42),
        )
        assignment = design.randomize(df)
        data = assignment.data_
        cluster_block = data.groupby("store_id")["region"].first()
        cluster_treatment = data.groupby("store_id")["treatment"].first()
        for region in ("A", "B"):
            clusters_in_block = cluster_block[cluster_block == region]
            n_treated = int(
                cluster_treatment[clusters_in_block.index].sum()
            )
            assert n_treated == int(round(0.5 * len(clusters_in_block)))

    def test_cluster_spanning_blocks_raises(self) -> None:
        """A cluster with individuals in two blocks must raise."""
        df = _df_stores_blocked(sizes=[4] * 6)
        # Move one customer of store_0 into region "B".
        df.loc[0, "region"] = "B"
        design = ClusterDesign(
            cluster_col="store_id",
            base_design=BlockedCRD(block_col="region", p=0.5, seed=42),
        )
        with pytest.raises(InvalidDesignError):
            design.randomize(df)

    def test_missing_block_col_raises(self) -> None:
        """Missing block column must raise InvalidDesignError."""
        df = _df_stores(sizes=[4] * 6)
        design = ClusterDesign(
            cluster_col="store_id",
            base_design=BlockedCRD(block_col="region", p=0.5, seed=42),
        )
        with pytest.raises(InvalidDesignError):
            design.randomize(df)


# --- ClusterAssignment construction validation ---


class TestClusterAssignmentValidation:
    """Tests for ClusterAssignment integrity checks."""

    def test_mixed_treatment_within_cluster_raises(self) -> None:
        """Individuals of one cluster with different labels must raise."""
        df = pd.DataFrame(
            {
                "store_id": ["A", "A", "B", "B"],
                "treatment": [1, 0, 0, 0],
            }
        )
        with pytest.raises(InvalidDesignError):
            ClusterAssignment(
                data=df,
                treatment_col="treatment",
                design=None,
                cluster_col="store_id",
                cluster_sizes={"A": 2, "B": 2},
            )

    def test_cluster_sizes_mismatch_raises(self) -> None:
        """cluster_sizes keys must match the labels in the data."""
        df = pd.DataFrame(
            {
                "store_id": ["A", "A", "B", "B"],
                "treatment": [1, 1, 0, 0],
            }
        )
        with pytest.raises(InvalidDesignError):
            ClusterAssignment(
                data=df,
                treatment_col="treatment",
                design=None,
                cluster_col="store_id",
                cluster_sizes={"A": 2, "C": 2},
            )

    def test_missing_cluster_col_raises(self) -> None:
        """Missing cluster column must raise InvalidDesignError."""
        df = pd.DataFrame({"treatment": [1, 0]})
        with pytest.raises(InvalidDesignError):
            ClusterAssignment(
                data=df,
                treatment_col="treatment",
                design=None,
                cluster_col="store_id",
                cluster_sizes={"A": 1, "B": 1},
            )

    def test_cluster_labels_by_arm(self) -> None:
        """treated/control_cluster_labels must list cluster labels."""
        df = pd.DataFrame(
            {
                "store_id": ["A", "A", "B", "B", "C", "C"],
                "treatment": [1, 1, 0, 0, 1, 1],
            }
        )
        assignment = ClusterAssignment(
            data=df,
            treatment_col="treatment",
            design=None,
            cluster_col="store_id",
            cluster_sizes={"A": 2, "B": 2, "C": 2},
        )
        assert set(assignment.treated_cluster_labels()) == {"A", "C"}
        assert set(assignment.control_cluster_labels()) == {"B"}
        assert assignment.n_clusters_treated_ == 2
        assert assignment.n_clusters_control_ == 1

    def test_ids_are_individual_level(self) -> None:
        """treated_ids/control_ids must be individual row positions."""
        df = pd.DataFrame(
            {
                "store_id": ["A", "A", "B", "B", "B"],
                "treatment": [1, 1, 0, 0, 0],
            }
        )
        assignment = ClusterAssignment(
            data=df,
            treatment_col="treatment",
            design=None,
            cluster_col="store_id",
            cluster_sizes={"A": 2, "B": 3},
        )
        np.testing.assert_array_equal(assignment.treated_ids(), [0, 1])
        np.testing.assert_array_equal(assignment.control_ids(), [2, 3, 4])


# --- ClusterAssignment.draw ---


class TestClusterAssignmentDraw:
    """Tests for ClusterAssignment.draw (whole-cluster relabeling)."""

    def test_draw_returns_cluster_assignment(self) -> None:
        """draw must return a ClusterAssignment."""
        df = _df_stores()
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        assignment = design.randomize(df)
        drawn = assignment.draw(seed=1)
        assert isinstance(drawn, ClusterAssignment)

    def test_draw_preserves_individuals(self) -> None:
        """A draw must keep the same individuals and clusters."""
        df = _df_stores(sizes=[3, 5, 2, 4])
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        assignment = design.randomize(df)
        drawn = assignment.draw(seed=1)
        assert drawn.n_units_ == assignment.n_units_
        assert drawn.n_clusters_ == assignment.n_clusters_
        assert drawn.cluster_sizes_ == assignment.cluster_sizes_
        assert list(drawn.data_["customer_id"]) == list(
            assignment.data_["customer_id"]
        )

    def test_draw_relabels_whole_clusters(self) -> None:
        """After a draw, treatment must remain constant per cluster."""
        df = _df_stores(sizes=[3, 7, 2, 9, 4, 5])
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        assignment = design.randomize(df)
        for seed in range(5):
            drawn = assignment.draw(seed=seed)
            n_unique = drawn.data_.groupby("store_id")[
                "treatment"
            ].nunique()
            assert (n_unique == 1).all()

    def test_draw_preserves_treated_cluster_count(self) -> None:
        """Each draw must keep the design's treated-cluster count."""
        df = _df_stores(sizes=[4] * 10)
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(n_treated=4, seed=42)
        )
        assignment = design.randomize(df)
        for seed in range(5):
            drawn = assignment.draw(seed=seed)
            assert drawn.n_clusters_treated_ == 4

    def test_draw_reproducible_per_seed(self) -> None:
        """draw(seed) must be reproducible for the same seed."""
        df = _df_stores()
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        assignment = design.randomize(df)
        d1 = assignment.draw(seed=7)
        d2 = assignment.draw(seed=7)
        pd.testing.assert_frame_equal(d1.data_, d2.data_)

    def test_draw_varies_across_seeds(self) -> None:
        """Different seeds should (with high probability) give
        different cluster labelings."""
        df = _df_stores(sizes=[4] * 12)
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        assignment = design.randomize(df)
        labelings = {
            tuple(
                assignment.draw(seed=s).data_["treatment"].tolist()
            )
            for s in range(10)
        }
        assert len(labelings) > 1

    def test_draw_does_not_mutate_self(self) -> None:
        """draw must not modify the original assignment."""
        df = _df_stores()
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        assignment = design.randomize(df)
        snapshot = assignment.data_.copy()
        _ = assignment.draw(seed=1)
        pd.testing.assert_frame_equal(assignment.data_, snapshot)

    def test_draw_does_not_corrupt_base_design_seed(self) -> None:
        """draw must restore the base design's seed afterwards."""
        df = _df_stores()
        base = CRD(p=0.5, seed=42)
        design = ClusterDesign(cluster_col="store_id", base_design=base)
        assignment = design.randomize(df)
        _ = assignment.draw(seed=999)
        assert base.seed == 42

    def test_draw_without_design_raises(self) -> None:
        """draw on a design-less assignment must raise."""
        df = pd.DataFrame(
            {
                "store_id": ["A", "A", "B", "B"],
                "treatment": [1, 1, 0, 0],
            }
        )
        assignment = ClusterAssignment(
            data=df,
            treatment_col="treatment",
            design=None,
            cluster_col="store_id",
            cluster_sizes={"A": 2, "B": 2},
        )
        with pytest.raises(InvalidDesignError):
            assignment.draw(seed=1)

    def test_draw_blocked_base(self) -> None:
        """draw must respect the blocked mechanism at cluster level."""
        df = _df_stores_blocked(sizes=[4] * 10)
        design = ClusterDesign(
            cluster_col="store_id",
            base_design=BlockedCRD(block_col="region", p=0.5, seed=42),
        )
        assignment = design.randomize(df)
        drawn = assignment.draw(seed=3)
        assert isinstance(drawn, ClusterAssignment)
        data = drawn.data_
        cluster_block = data.groupby("store_id")["region"].first()
        cluster_treatment = data.groupby("store_id")["treatment"].first()
        for region in ("A", "B"):
            clusters_in_block = cluster_block[cluster_block == region]
            n_treated = int(
                cluster_treatment[clusters_in_block.index].sum()
            )
            assert n_treated == int(round(0.5 * len(clusters_in_block)))
