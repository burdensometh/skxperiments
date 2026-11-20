"""Integration test: ClusterDesign -> ClusterAssignment ->
ClusterDifferenceInMeans -> RandomizationTest -> Results.

Covers the store-randomized / customer-recorded scenario: stores are
the randomization units, outcomes are recorded per customer, and the
analysis must aggregate to the cluster level.
"""

import numpy as np
import pandas as pd
import pytest

from skxperiments.core.results import Results
from skxperiments.design.blocked_crd import BlockedCRD
from skxperiments.design.cluster import ClusterDesign
from skxperiments.design.crd import CRD
from skxperiments.estimators.cluster_difference_in_means import (
    ClusterDifferenceInMeans,
)
from skxperiments.inference import RandomizationTest


def _store_customer_df(
    n_clusters: int,
    customers_per_cluster: int,
    seed: int,
    blocked: bool = False,
) -> pd.DataFrame:
    """Individual-level df: customers nested in stores (clusters)."""
    rng = np.random.default_rng(seed)
    store_ids: list[str] = []
    for i in range(n_clusters):
        store_ids.extend([f"store_{i}"] * customers_per_cluster)
    n = len(store_ids)
    df = pd.DataFrame(
        {
            "store_id": store_ids,
            "customer_id": np.arange(n),
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )
    if blocked:
        half = n_clusters // 2
        store_block = {
            f"store_{i}": ("A" if i < half else "B")
            for i in range(n_clusters)
        }
        df["region"] = df["store_id"].map(store_block)
    return df


class TestClusterCRDtoClusterDifferenceInMeans:
    """End-to-end: ClusterDesign(CRD) -> assignment -> estimator."""

    def test_full_pipeline_produces_results(self) -> None:
        """Full pipeline must produce Results with honest two-level
        metadata and recover the injected cluster-level ATE."""
        df = _store_customer_df(
            n_clusters=40, customers_per_cluster=10, seed=42
        )
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=42)
        )
        assignment = design.randomize(df)

        # Inject a constant cluster-level effect into treated stores.
        true_ate = 1.0
        treated = assignment.treated_ids()
        assignment.data_.iloc[
            treated, assignment.data_.columns.get_loc("y")
        ] += true_ate

        estimator = ClusterDifferenceInMeans(outcome_col="y")
        result = estimator.fit(assignment).estimate()

        assert isinstance(result, Results)
        assert result.ate is not None
        assert isinstance(result.ate, float)
        assert result.ate == pytest.approx(true_ate, abs=0.25)

        assert result.estimator_name == "ClusterDifferenceInMeans"
        assert result.design_name == "ClusterDesign"

        # Two-level metadata: individuals in n_obs, clusters in extra.
        assert result.n_obs == 400
        assert result.extra is not None
        assert result.extra["n_clusters"] == 40
        assert result.extra["n_clusters_treated"] == 20
        assert result.extra["estimand"] == "cluster_mean"

    def test_randomization_test_detects_cluster_effect(self) -> None:
        """RandomizationTest on the cluster assignment must detect a
        strong cluster-level effect."""
        df = _store_customer_df(
            n_clusters=40, customers_per_cluster=10, seed=7
        )
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=7)
        )
        assignment = design.randomize(df)
        treated = assignment.treated_ids()
        assignment.data_.iloc[
            treated, assignment.data_.columns.get_loc("y")
        ] += 1.5

        rt = RandomizationTest(
            estimator=ClusterDifferenceInMeans(outcome_col="y"),
            n_permutations=500,
            seed=0,
        )
        result = rt.fit(assignment).estimate()

        assert result.p_value is not None
        assert result.p_value < 0.05
        assert result.inference_name == "RandomizationTest"
        assert result.n_obs == 400
        assert result.extra is not None
        assert result.extra["n_clusters"] == 40

    def test_randomization_test_null_not_extreme(self) -> None:
        """With no effect, the randomization p-value must not be
        extreme."""
        df = _store_customer_df(
            n_clusters=40, customers_per_cluster=10, seed=11
        )
        design = ClusterDesign(
            cluster_col="store_id", base_design=CRD(p=0.5, seed=11)
        )
        assignment = design.randomize(df)

        rt = RandomizationTest(
            estimator=ClusterDifferenceInMeans(outcome_col="y"),
            n_permutations=500,
            seed=0,
        )
        result = rt.fit(assignment).estimate()
        assert result.p_value is not None
        assert result.p_value > 0.01


class TestClusterBlockedPipeline:
    """End-to-end: ClusterDesign(BlockedCRD) -> estimator -> test."""

    def test_blocked_cluster_pipeline(self) -> None:
        """Blocked cluster randomization must flow through estimation
        and randomization inference."""
        df = _store_customer_df(
            n_clusters=20, customers_per_cluster=5, seed=3, blocked=True
        )
        design = ClusterDesign(
            cluster_col="store_id",
            base_design=BlockedCRD(block_col="region", p=0.5, seed=3),
        )
        assignment = design.randomize(df)

        # Balance check: 5 treated clusters per region.
        data = assignment.data_
        cluster_block = data.groupby("store_id")["region"].first()
        cluster_t = data.groupby("store_id")["treatment"].first()
        for region in ("A", "B"):
            labels = cluster_block[cluster_block == region].index
            assert int(cluster_t[labels].sum()) == 5

        estimator = ClusterDifferenceInMeans(outcome_col="y")
        result = estimator.fit(assignment).estimate()
        assert isinstance(result, Results)
        assert result.extra is not None
        assert result.extra["n_clusters"] == 20

        rt = RandomizationTest(
            estimator=ClusterDifferenceInMeans(outcome_col="y"),
            n_permutations=200,
            seed=0,
        )
        rt_result = rt.fit(assignment).estimate()
        assert rt_result.p_value is not None
        assert 0.0 < rt_result.p_value <= 1.0
