"""Integration tests: frozen covariates through design, diagnostics, estimates."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from skxperiments.core.assignment import CRDAssignment
from skxperiments.core.exceptions import (
    FrozenCovariateViolation,
    InvalidDesignError,
)
from skxperiments.covariates import CovariateSpec, PointInTimeCovariates
from skxperiments.design.balance import check_balance
from skxperiments.design.crd import CRD
from skxperiments.design.rerandomized_crd import ReRandomizedCRD
from skxperiments.diagnostics.balance_report import BalanceReport
from skxperiments.estimators.cuped import CUPED
from skxperiments.estimators.difference_in_means import DifferenceInMeans
from skxperiments.estimators.lin_estimator import LinEstimator
from skxperiments.inference import NeymanCI
from skxperiments.pipeline import ExperimentPipeline

CUTOFF = pd.Timestamp("2026-02-01 00:00:00")


def _make_units(n=120, seed=0):
    return pd.DataFrame(
        {
            "unit_id": [f"u{i}" for i in range(n)],
            "assigned_at": CUTOFF,
        }
    )


def _make_history(units, seed=1):
    """History with in-window and post-cutoff rows per unit."""
    rng = np.random.default_rng(seed)
    rows = []
    for uid in units["unit_id"]:
        # Two pre-window events, visible on time.
        for day in (-3, -1):
            rows.append(
                (
                    uid,
                    CUTOFF + pd.Timedelta(days=day),
                    CUTOFF + pd.Timedelta(days=day, hours=1),
                    rng.normal(),
                )
            )
        # One event whose value is known before cutoff but arrived late:
        # this is what a latest-snapshot join would leak.
        rows.append(
            (
                uid,
                CUTOFF - pd.Timedelta(hours=6),
                CUTOFF + pd.Timedelta(days=2),
                1000.0,
            )
        )
        # A truly post-assignment event with a huge value.
        rows.append(
            (
                uid,
                CUTOFF + pd.Timedelta(days=1),
                CUTOFF + pd.Timedelta(days=1),
                5000.0,
            )
        )
    return pd.DataFrame(
        {
            "unit_id": [r[0] for r in rows],
            "event_at": [r[1] for r in rows],
            "arrived_at": [r[2] for r in rows],
            "spend": [r[3] for r in rows],
        }
    )


def _frame(units, history, fillna=0.0):
    builder = PointInTimeCovariates(
        unit_col="unit_id",
        event_time_col="event_at",
        arrival_time_col="arrived_at",
        cutoff_col="assigned_at",
        specs=[
            CovariateSpec(
                "spend_7d",
                column="spend",
                agg="sum",
                role=["balance", "adjustment"],
            ),
        ],
        window="7D",
        fillna=fillna,
        history_version="snap-1",
        cutoff_version="v1",
    )
    return builder.build(units, history)


def _attached(units, history, extra=None):
    frame = _frame(units, history)
    df = frame.attach(units)
    if extra is not None:
        df = df.join(extra)
    return frame, df


# --- Design balance --------------------------------------------------------


class TestRerandomizationUsesFrozenCovariates:
    def _setup(self, seed=0):
        units = _make_units(seed=seed)
        history = _make_history(units, seed=seed + 10)
        frame, df = _attached(units, history)
        return frame, df

    def test_by_role_selection(self):
        frame, df = self._setup()
        design = ReRandomizedCRD(
            covariates=None, threshold=10.0, p=0.5, seed=42
        )
        assignment = design.randomize(df, frozen_covariates=frame)
        meta = assignment.rerandomization_metadata
        assert meta["covariates"] == ["spend_7d"]
        assert assignment.frozen_covariates_ is frame

    def test_no_covariates_and_no_frame_raises(self):
        units = _make_units()
        design = ReRandomizedCRD(covariates=None, threshold=5.0, p=0.5)
        with pytest.raises(InvalidDesignError, match="frozen"):
            design.randomize(units)

    def test_role_with_no_match_raises(self):
        units = _make_units()
        history = _make_history(units)
        builder = PointInTimeCovariates(
            unit_col="unit_id",
            event_time_col="event_at",
            specs=[CovariateSpec("spend_7d", column="spend", agg="sum")],
            cutoff_col="assigned_at",
        )
        frame = builder.build(units, history)
        design = ReRandomizedCRD(
            covariates=None,
            threshold=5.0,
            p=0.5,
            covariate_roles="balance",
        )
        with pytest.raises(InvalidDesignError, match="no covariate"):
            design.randomize(frame.attach(units), frozen_covariates=frame)

    def test_latest_snapshot_column_cannot_redefine_criterion(self):
        """The named failure mode: same name, newer values."""
        frame, df = self._setup()
        design = ReRandomizedCRD(
            covariates=None, threshold=10.0, p=0.5, seed=42
        )
        tampered = df.copy()
        tampered["spend_7d"] = np.linspace(0, 1, len(tampered))
        with pytest.raises(FrozenCovariateViolation, match="changed"):
            design.randomize(tampered, frozen_covariates=frame)

    def test_criterion_columns_verified_against_frozen(self):
        """A newer spend_7d under the same name cannot redefine the design."""
        frame, df = self._setup()
        design = ReRandomizedCRD(
            covariates=None, threshold=10.0, p=0.5, seed=42
        )
        tampered = df.copy()
        tampered["spend_7d"] = tampered["spend_7d"] + 1.0
        with pytest.raises(FrozenCovariateViolation):
            design.randomize(tampered, frozen_covariates=frame)

    def test_draw_keeps_the_same_criterion_and_frame(self):
        frame, df = self._setup()
        design = ReRandomizedCRD(
            covariates=None, threshold=10.0, p=0.5, seed=42
        )
        assignment = design.randomize(df, frozen_covariates=frame)
        new = assignment.draw(seed=7)
        assert new.rerandomization_metadata["covariates"] == ["spend_7d"]
        assert new.frozen_covariates_ is frame
        # Same frozen values, still attached.
        frame.assert_unchanged(new.data_)

    def test_criterion_is_actually_balanced(self):
        """The rerandomized feature must be balanced on the frozen value."""
        frame, df = self._setup()
        design = ReRandomizedCRD(
            covariates=None,
            threshold=float(chi2.ppf(0.10, df=1)),
            p=0.5,
            seed=3,
            max_attempts=20_000,
        )
        assignment = design.randomize(df, frozen_covariates=frame)
        balance = check_balance(assignment)
        assert list(balance["covariate"]) == ["spend_7d"]
        assert abs(float(balance["smd"].iloc[0])) < 1.0


# --- Balance diagnostics ---------------------------------------------------


class TestBalancePrefersFrozenCovariates:
    def test_check_balance_default_uses_declared_only(self):
        units = _make_units()
        history = _make_history(units)
        frame, df = _attached(units, history)
        # Add a post-assignment metric: it must NOT appear by default.
        df["revenue_post"] = np.arange(len(df), dtype=float)
        assignment = CRD(p=0.5, seed=0).randomize(
            df, frozen_covariates=frame
        )
        table = check_balance(assignment)
        assert list(table["covariate"]) == ["spend_7d"]

    def test_check_balance_rejects_undeclared_explicit_name(self):
        units = _make_units()
        frame, df = _attached(units, _make_history(units))
        assignment = CRD(p=0.5, seed=0).randomize(
            df, frozen_covariates=frame
        )
        with pytest.raises(InvalidDesignError, match="frozen"):
            check_balance(assignment, covariates=["latest_spend"])

    def test_balance_report_carries_metadata(self):
        units = _make_units()
        frame, df = _attached(units, _make_history(units))
        assignment = CRD(p=0.5, seed=0).randomize(
            df, frozen_covariates=frame
        )
        result = BalanceReport().run(assignment)
        assert result.covariate_metadata is not None
        assert result.covariate_metadata["history_version"] == "snap-1"
        assert (
            result.covariate_metadata["fingerprint"]
            == frame.fingerprint()
        )

    def test_balance_report_without_frozen_table_still_autoselects(self):
        df = pd.DataFrame({"x": [1.0, 2, 3, 4], "treatment": [1, 0, 1, 0]})
        assignment = CRDAssignment(
            data=df, treatment_col="treatment", design=None
        )
        result = BalanceReport().run(assignment)
        assert list(result.table["covariate"]) == ["x"]


# --- Estimators ------------------------------------------------------------


class TestAdjustmentEstimatorsUseFrozenCovariates:
    def _setup(self, seed=0):
        units = _make_units(seed=seed)
        history = _make_history(units, seed=seed + 10)
        frame, df = _attached(units, history)
        # Outcome correlated with frozen spend; post metric is noise.
        rng = np.random.default_rng(seed + 99)
        df = df.copy()
        df["y"] = 0.8 * frame.data.loc[df["unit_id"], "spend_7d"].to_numpy()
        df["y"] += rng.normal(scale=0.5, size=len(df))
        df["treated_effect"] = np.nan
        assignment = CRD(p=0.5, seed=seed + 1).randomize(
            df.drop(columns=["treated_effect"]),
            frozen_covariates=frame,
        )
        treated = assignment.treated_ids()
        assignment.data_.loc[
            assignment.data_.index[treated], "y"
        ] += 1.0
        return frame, assignment

    def test_cuped_resolves_by_role(self):
        frame, assignment = self._setup()
        estimator = CUPED(outcome_col="y", pre_experiment_col=None)
        result = estimator.fit(assignment).estimate()
        assert estimator.pre_experiment_col_ == "spend_7d"
        assert result.extra["covariate_metadata"]["history_version"] == "snap-1"
        assert result.extra["pre_experiment_col"] == "spend_7d"

    def test_cuped_rejects_column_outside_frozen_table(self):
        _, assignment = self._setup()
        estimator = CUPED(outcome_col="y", pre_experiment_col="raw_spend")
        with pytest.raises(InvalidDesignError, match="not in the assignment"):
            estimator.fit(assignment)

    def test_cuped_without_frame_and_without_column_raises(self):
        df = pd.DataFrame({"y_pre": [1.0, 2, 3, 4], "y": [1.0, 2, 3, 4]})
        assignment = CRD(p=0.5, seed=0).randomize(df)
        estimator = CUPED(outcome_col="y", pre_experiment_col=None)
        with pytest.raises(InvalidDesignError, match="frozen"):
            estimator.fit(assignment)

    def test_lin_explicit_unfrozen_column_rejected(self):
        _, assignment = self._setup()
        assignment.data_["region"] = 1.0
        estimator = LinEstimator(outcome_col="y", covariates=["region"])
        with pytest.raises(InvalidDesignError, match="frozen"):
            estimator.fit(assignment)

    def test_lin_explicit_frozen_column_accepted(self):
        _, assignment = self._setup()
        estimator = LinEstimator(
            outcome_col="y", covariates=["spend_7d"]
        )
        result = estimator.fit(assignment).estimate()
        assert estimator.covariates_ == ["spend_7d"]
        assert result.extra["covariates"] == ["spend_7d"]

    def test_lin_resolves_by_role_and_records_covariates(self):
        _, assignment = self._setup()
        estimator = LinEstimator(outcome_col="y", covariates=None)
        result = estimator.fit(assignment).estimate()
        assert estimator.covariates_ == ["spend_7d"]
        assert result.extra["covariates"] == ["spend_7d"]
        assert (
            result.extra["covariate_metadata"]["fingerprint"]
            == assignment.frozen_covariates_.fingerprint()
        )

    def test_lin_without_frame_and_without_columns_raises(self):
        df = pd.DataFrame(
            {"x": [1.0, 2, 3, 4], "y": [1.0, 2, 3, 4]}
        )
        assignment = CRD(p=0.5, seed=0).randomize(df)
        estimator = LinEstimator(outcome_col="y", covariates=None)
        with pytest.raises(InvalidDesignError, match="frozen"):
            estimator.fit(assignment)

    @pytest.mark.parametrize("estimator_cls", [CUPED, LinEstimator])
    def test_tampered_frozen_column_blocks_adjustment(self, estimator_cls):
        _, assignment = self._setup()
        assignment.data_["spend_7d"] = np.arange(
            len(assignment.data_), dtype=float
        )
        if estimator_cls is CUPED:
            estimator = estimator_cls(outcome_col="y", pre_experiment_col=None)
        else:
            estimator = estimator_cls(outcome_col="y", covariates=None)
        with pytest.raises(FrozenCovariateViolation):
            estimator.fit(assignment)

    def test_adjusted_estimate_is_not_polluted_by_post_rows(self):
        """Values from late-arriving rows would move the estimate a lot."""
        _, assignment = self._setup()
        assert assignment.data_["spend_7d"].max() < 100.0

        result = CUPED(
            outcome_col="y", pre_experiment_col=None
        ).fit(assignment).estimate()
        assert result.ate == pytest.approx(1.0, abs=0.35)


# --- Assignment integrity --------------------------------------------------


class TestAssignmentFreezing:
    def _setup(self):
        units = _make_units()
        frame, df = _attached(units, _make_history(units))
        assignment = CRD(p=0.5, seed=0).randomize(
            df, frozen_covariates=frame
        )
        return frame, assignment

    def test_metadata_available_on_assignment(self):
        frame, assignment = self._setup()
        assert assignment.covariate_metadata_["fingerprint"] == (
            frame.fingerprint()
        )

    def test_appending_post_data_is_allowed(self):
        _, assignment = self._setup()
        assignment.data_["revenue_post"] = 1.0
        assignment.verify_frozen_covariates()  # no raise
        frozen = assignment.frozen_covariates_
        frozen.assert_unchanged(assignment.data_)

    def test_overwriting_frozen_column_is_rejected(self):
        _, assignment = self._setup()
        assignment.data_["spend_7d"] = 0.0
        with pytest.raises(FrozenCovariateViolation):
            assignment.verify_frozen_covariates()

    def test_plain_assignment_verify_raises(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "treatment": [1, 0]})
        assignment = CRDAssignment(
            data=df, treatment_col="treatment", design=None
        )
        with pytest.raises(InvalidDesignError, match="no frozen"):
            assignment.verify_frozen_covariates()

    def test_draw_carries_frozen_covariates(self):
        frame, assignment = self._setup()
        new = assignment.draw(seed=11)
        assert new.frozen_covariates_ is frame
        frame.assert_unchanged(new.data_)
        assert new.covariate_metadata_["fingerprint"] == frame.fingerprint()

    def test_construction_verifies_frozen_values(self):
        frame, assignment = self._setup()
        tampered = assignment.data_.copy()
        tampered["spend_7d"] = 0.0
        with pytest.raises(FrozenCovariateViolation):
            CRDAssignment(
                data=tampered,
                treatment_col="treatment",
                design=None,
                frozen_covariates=frame,
            )


# --- Pipeline --------------------------------------------------------------


class TestPipelineCarriesFrozenMetadata:
    def test_pipeline_records_covariate_metadata(self):
        units = _make_units()
        frame, df = _attached(units, _make_history(units))
        rng = np.random.default_rng(0)
        df = df.copy()
        df["y"] = rng.normal(size=len(df))
        assignment = CRD(p=0.5, seed=0).randomize(
            df, frozen_covariates=frame
        )
        pipeline = ExperimentPipeline(
            inference=NeymanCI(estimator=DifferenceInMeans(outcome_col="y")),
            diagnostics=[BalanceReport()],
        )
        result = pipeline.run(assignment)
        assert result.covariate_metadata["fingerprint"] == (
            frame.fingerprint()
        )

    def test_pipeline_blocks_tampered_design_feature(self):
        units = _make_units()
        frame, df = _attached(units, _make_history(units))
        df = df.copy()
        df["y"] = np.arange(len(df), dtype=float)
        assignment = CRD(p=0.5, seed=0).randomize(
            df, frozen_covariates=frame
        )
        assignment.data_["spend_7d"] *= 2.0
        pipeline = ExperimentPipeline(
            inference=NeymanCI(estimator=DifferenceInMeans(outcome_col="y")),
            diagnostics=[],
        )
        with pytest.raises(FrozenCovariateViolation):
            pipeline.run(assignment)
