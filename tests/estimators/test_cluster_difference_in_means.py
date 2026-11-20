"""Tests for skxperiments.estimators.cluster_difference_in_means."""

import numpy as np
import pandas as pd
import pytest

from skxperiments.core.assignment import (
    BlockedAssignment,
    ClusterAssignment,
    CRDAssignment,
)
from skxperiments.core.exceptions import (
    DesignEstimatorMismatch,
    InvalidDesignError,
    NotFittedError,
)
from skxperiments.core.results import Results
from skxperiments.design.blocked_crd import BlockedCRD
from skxperiments.design.cluster import ClusterDesign
from skxperiments.design.crd import CRD
from skxperiments.estimators.blocked_difference_in_means import (
    BlockedDifferenceInMeans,
)
from skxperiments.estimators.cluster_difference_in_means import (
    ClusterDifferenceInMeans,
)
from skxperiments.estimators.difference_in_means import DifferenceInMeans


# --- Helpers ---


def _uneven_cluster_assignment(
    weights: list[float] | None = None,
) -> ClusterAssignment:
    """Hand-built ClusterAssignment with unequal cluster sizes.

    Clusters A, B are treated; C, D are control. Cluster means:
    A=2 (size 2), B=6 (size 4), C=2 (size 4), D=3 (size 2).

    - cluster_mean estimand: (2+6)/2 - (2+3)/2 = 1.5
    - individual-level DiM:  28/6 - 14/6 = 2.333... (differs, because
      the individual mean size-weights the cluster means)
    - exposure_weighted with weights [1, 3, 2, 2]:
      (1*2 + 3*6)/4 - (2*2 + 2*3)/4 = 5 - 2.5 = 2.5
    """
    df = pd.DataFrame(
        {
            "store_id": [
                "A", "A",
                "B", "B", "B", "B",
                "C", "C", "C", "C",
                "D", "D",
            ],
            "treatment": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "y": [
                1.0, 3.0,                # A: mean 2
                5.0, 5.0, 7.0, 7.0,      # B: mean 6
                0.0, 0.0, 0.0, 8.0,      # C: mean 2
                2.0, 4.0,                # D: mean 3
            ],
        }
    )
    if weights is not None:
        weight_map = {
            k: w for k, w in zip(["A", "B", "C", "D"], weights)
        }
        df["w"] = df["store_id"].map(weight_map)
    return ClusterAssignment(
        data=df,
        treatment_col="treatment",
        design=None,
        cluster_col="store_id",
        cluster_sizes={"A": 2, "B": 4, "C": 4, "D": 2},
    )


def _randomized_cluster_assignment(
    n_clusters: int = 20,
    customers_per_cluster: int = 5,
    true_ate: float = 0.0,
    seed: int = 0,
) -> ClusterAssignment:
    """Cluster-randomize stores, then build outcomes with a constant
    cluster-level effect ``true_ate`` on treated clusters.
    """
    rng = np.random.default_rng(seed)
    store_ids: list[str] = []
    for i in range(n_clusters):
        store_ids.extend([f"store_{i}"] * customers_per_cluster)
    n = len(store_ids)
    df = pd.DataFrame(
        {
            "store_id": store_ids,
            "x": rng.normal(size=n),
        }
    )
    design = ClusterDesign(
        cluster_col="store_id", base_design=CRD(p=0.5, seed=seed)
    )
    assignment = design.randomize(df)
    t = assignment.data_["treatment"].values.astype(float)
    assignment.data_["y"] = (
        assignment.data_["x"].values
        + true_ate * t
        + rng.normal(size=n)
    )
    return assignment


# --- Creation ---


class TestClusterDifferenceInMeansCreation:
    """Tests for ClusterDifferenceInMeans instantiation."""

    def test_default_estimand(self) -> None:
        """Default estimand is cluster_mean with no weight column."""
        est = ClusterDifferenceInMeans(outcome_col="y")
        assert est.estimand == "cluster_mean"
        assert est.weight_col is None

    def test_exposure_weighted_requires_weight_col(self) -> None:
        """exposure_weighted without weight_col must raise."""
        with pytest.raises(InvalidDesignError):
            ClusterDifferenceInMeans(
                outcome_col="y", estimand="exposure_weighted"
            )

    def test_cluster_mean_rejects_weight_col(self) -> None:
        """weight_col with cluster_mean estimand must raise."""
        with pytest.raises(InvalidDesignError):
            ClusterDifferenceInMeans(
                outcome_col="y", estimand="cluster_mean", weight_col="w"
            )

    def test_invalid_estimand_raises(self) -> None:
        """Unknown estimand strings must raise."""
        with pytest.raises(InvalidDesignError):
            ClusterDifferenceInMeans(outcome_col="y", estimand="individual")

    def test_exposure_weighted_creation_ok(self) -> None:
        """Valid exposure_weighted configuration must instantiate."""
        est = ClusterDifferenceInMeans(
            outcome_col="y",
            estimand="exposure_weighted",
            weight_col="w",
        )
        assert est.weight_col == "w"


# --- Fit: assignment type validation ---


class TestClusterDifferenceInMeansAssignmentType:
    """ClusterDifferenceInMeans accepts only ClusterAssignment."""

    def test_rejects_crd_assignment(self) -> None:
        """CRDAssignment must be rejected."""
        df = pd.DataFrame({"x": range(10), "y": range(10)})
        assignment = CRD(p=0.5, seed=1).randomize(df)
        est = ClusterDifferenceInMeans(outcome_col="y")
        with pytest.raises(DesignEstimatorMismatch):
            est.fit(assignment)

    def test_rejects_blocked_assignment(self) -> None:
        """BlockedAssignment must be rejected."""
        df = pd.DataFrame(
            {
                "x": range(10),
                "y": range(10),
                "block": ["A"] * 5 + ["B"] * 5,
            }
        )
        assignment = BlockedCRD(block_col="block", p=0.5, seed=1).randomize(df)
        est = ClusterDifferenceInMeans(outcome_col="y")
        with pytest.raises(DesignEstimatorMismatch):
            est.fit(assignment)


# --- Fit: point estimates ---


class TestClusterDifferenceInMeansFit:
    """Point-estimate correctness at the cluster level."""

    def test_cluster_mean_estimand_uses_cluster_means(self) -> None:
        """ATE must be the difference of per-cluster mean averages,
        not the individual-level difference in means."""
        assignment = _uneven_cluster_assignment()
        est = ClusterDifferenceInMeans(outcome_col="y")
        est.fit(assignment)
        # Cluster means: A=2, B=6 (treated); C=2, D=3 (control).
        assert est.ate_ == pytest.approx((2 + 6) / 2 - (2 + 3) / 2)
        # The individual-level DiM differs (unequal cluster sizes):
        y = assignment.data_["y"].values
        t = assignment.data_["treatment"].values
        individual_dim = y[t == 1].mean() - y[t == 0].mean()
        assert est.ate_ != pytest.approx(individual_dim)

    def test_cluster_mean_equal_sizes_matches_individual(self) -> None:
        """With equal cluster sizes, cluster_mean equals the
        individual-level difference in means."""
        assignment = _randomized_cluster_assignment(
            n_clusters=20, customers_per_cluster=5, seed=1
        )
        est = ClusterDifferenceInMeans(outcome_col="y")
        est.fit(assignment)
        y = assignment.data_["y"].values
        t = assignment.data_["treatment"].values
        individual_dim = y[t == 1].mean() - y[t == 0].mean()
        assert est.ate_ == pytest.approx(individual_dim)

    def test_exposure_weighted_estimand(self) -> None:
        """ATE must weight cluster means by the exposure column."""
        # weights: A=1, B=3 (treated); C=2, D=2 (control)
        assignment = _uneven_cluster_assignment(weights=[1, 3, 2, 2])
        est = ClusterDifferenceInMeans(
            outcome_col="y",
            estimand="exposure_weighted",
            weight_col="w",
        )
        est.fit(assignment)
        # (1*2 + 3*6)/4 - (2*2 + 2*3)/4 = 5 - 2.5
        assert est.ate_ == pytest.approx(5.0 - 2.5)

    def test_cluster_frame_has_one_row_per_cluster(self) -> None:
        """cluster_frame_ must hold one row per cluster."""
        assignment = _uneven_cluster_assignment()
        est = ClusterDifferenceInMeans(outcome_col="y")
        est.fit(assignment)
        assert len(est.cluster_frame_) == 4
        assert set(est.cluster_frame_.index) == {"A", "B", "C", "D"}
        assert list(est.cluster_frame_.columns) == ["outcome", "treatment"]

    def test_cluster_frame_includes_weight_when_weighted(self) -> None:
        """cluster_frame_ must carry the weight column when used."""
        assignment = _uneven_cluster_assignment(weights=[1, 3, 2, 2])
        est = ClusterDifferenceInMeans(
            outcome_col="y",
            estimand="exposure_weighted",
            weight_col="w",
        )
        est.fit(assignment)
        assert "weight" in est.cluster_frame_.columns

    def test_recovers_constant_cluster_effect(self) -> None:
        """A constant cluster-level effect must be recovered."""
        assignment = _randomized_cluster_assignment(
            n_clusters=40, customers_per_cluster=5, true_ate=2.0, seed=3
        )
        est = ClusterDifferenceInMeans(outcome_col="y")
        est.fit(assignment)
        assert est.ate_ == pytest.approx(2.0, abs=0.3)


# --- Fit: input validation ---


class TestClusterDifferenceInMeansValidation:
    """Outcome and weight column validation."""

    def test_missing_outcome_col_raises(self) -> None:
        assignment = _uneven_cluster_assignment()
        est = ClusterDifferenceInMeans(outcome_col="nope")
        with pytest.raises(InvalidDesignError):
            est.fit(assignment)

    def test_non_numeric_outcome_raises(self) -> None:
        assignment = _uneven_cluster_assignment()
        assignment.data_["y"] = assignment.data_["y"].astype(str)
        est = ClusterDifferenceInMeans(outcome_col="y")
        with pytest.raises(InvalidDesignError):
            est.fit(assignment)

    def test_nan_outcome_raises(self) -> None:
        assignment = _uneven_cluster_assignment()
        assignment.data_.loc[0, "y"] = np.nan
        est = ClusterDifferenceInMeans(outcome_col="y")
        with pytest.raises(InvalidDesignError):
            est.fit(assignment)

    def test_missing_weight_col_raises(self) -> None:
        assignment = _uneven_cluster_assignment()
        est = ClusterDifferenceInMeans(
            outcome_col="y",
            estimand="exposure_weighted",
            weight_col="w",
        )
        with pytest.raises(InvalidDesignError):
            est.fit(assignment)

    def test_non_numeric_weight_raises(self) -> None:
        assignment = _uneven_cluster_assignment(weights=[1, 3, 2, 2])
        assignment.data_["w"] = assignment.data_["w"].astype(str)
        est = ClusterDifferenceInMeans(
            outcome_col="y",
            estimand="exposure_weighted",
            weight_col="w",
        )
        with pytest.raises(InvalidDesignError):
            est.fit(assignment)

    def test_nan_weight_raises(self) -> None:
        assignment = _uneven_cluster_assignment(weights=[1, 3, 2, 2])
        assignment.data_.loc[0, "w"] = np.nan
        est = ClusterDifferenceInMeans(
            outcome_col="y",
            estimand="exposure_weighted",
            weight_col="w",
        )
        with pytest.raises(InvalidDesignError):
            est.fit(assignment)

    def test_negative_weight_raises(self) -> None:
        assignment = _uneven_cluster_assignment(weights=[1, -3, 2, 2])
        est = ClusterDifferenceInMeans(
            outcome_col="y",
            estimand="exposure_weighted",
            weight_col="w",
        )
        with pytest.raises(InvalidDesignError):
            est.fit(assignment)

    def test_weight_varying_within_cluster_raises(self) -> None:
        """Exposure is a cluster-level attribute; within-cluster
        variation must raise."""
        assignment = _uneven_cluster_assignment(weights=[1, 3, 2, 2])
        assignment.data_.loc[0, "w"] = 99.0  # store A now has two weights
        est = ClusterDifferenceInMeans(
            outcome_col="y",
            estimand="exposure_weighted",
            weight_col="w",
        )
        with pytest.raises(InvalidDesignError):
            est.fit(assignment)

    def test_zero_total_weight_in_arm_raises(self) -> None:
        """Zero total exposure in an arm leaves the estimand
        undefined."""
        assignment = _uneven_cluster_assignment(weights=[0, 0, 2, 2])
        est = ClusterDifferenceInMeans(
            outcome_col="y",
            estimand="exposure_weighted",
            weight_col="w",
        )
        with pytest.raises(InvalidDesignError):
            est.fit(assignment)


# --- Estimate: Results metadata ---


class TestClusterDifferenceInMeansEstimate:
    """Results must carry both individual and cluster counts."""

    def test_estimate_before_fit_raises(self) -> None:
        """estimate() before fit() must raise NotFittedError."""
        est = ClusterDifferenceInMeans(outcome_col="y")
        with pytest.raises(NotFittedError):
            est.estimate()

    def test_results_metadata(self) -> None:
        """n_obs counts individuals; cluster counts live in extra."""
        assignment = _randomized_cluster_assignment(
            n_clusters=20, customers_per_cluster=5, seed=1
        )
        est = ClusterDifferenceInMeans(outcome_col="y")
        result = est.fit(assignment).estimate()

        assert isinstance(result, Results)
        assert result.estimator_name == "ClusterDifferenceInMeans"
        assert result.design_name == "ClusterDesign"

        # Individual-level counts (rows of the assignment data).
        assert result.n_obs == 100
        assert result.n_treated + result.n_control == 100

        # Cluster counts — the actual randomization units.
        assert result.extra is not None
        assert result.extra["n_clusters"] == 20
        assert (
            result.extra["n_clusters_treated"]
            + result.extra["n_clusters_control"]
            == 20
        )
        assert result.extra["estimand"] == "cluster_mean"

    def test_results_extra_estimand_exposure_weighted(self) -> None:
        """extra must record the exposure_weighted estimand."""
        assignment = _uneven_cluster_assignment(weights=[1, 3, 2, 2])
        est = ClusterDifferenceInMeans(
            outcome_col="y",
            estimand="exposure_weighted",
            weight_col="w",
        )
        result = est.fit(assignment).estimate()
        assert result.extra is not None
        assert result.extra["estimand"] == "exposure_weighted"
        assert result.extra["n_clusters"] == 4

    def test_inference_fields_none(self) -> None:
        """se, ci, p_value must be None (inference is separate)."""
        assignment = _uneven_cluster_assignment()
        est = ClusterDifferenceInMeans(outcome_col="y")
        result = est.fit(assignment).estimate()
        assert result.se is None
        assert result.ci is None
        assert result.p_value is None


# --- Routing: other estimators reject ClusterAssignment ---


class TestOtherEstimatorsRejectCluster:
    """Unit-level DiM estimators must reject ClusterAssignment with a
    pointer to ClusterDifferenceInMeans."""

    def test_difference_in_means_rejects_cluster(self) -> None:
        assignment = _uneven_cluster_assignment()
        with pytest.raises(DesignEstimatorMismatch) as exc_info:
            DifferenceInMeans(outcome_col="y").fit(assignment)
        assert "ClusterDifferenceInMeans" in str(exc_info.value)

    def test_blocked_difference_in_means_rejects_cluster(self) -> None:
        assignment = _uneven_cluster_assignment()
        with pytest.raises(DesignEstimatorMismatch) as exc_info:
            BlockedDifferenceInMeans(outcome_col="y").fit(assignment)
        assert "ClusterDifferenceInMeans" in str(exc_info.value)
