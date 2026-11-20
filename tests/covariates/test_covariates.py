"""Tests for skxperiments.covariates."""

import pandas as pd
import pytest

from skxperiments.core.exceptions import (
    FrozenCovariateViolation,
    InvalidDesignError,
)
from skxperiments.covariates import (
    CovariateFrame,
    CovariateSpec,
    PointInTimeCovariates,
)
from skxperiments.covariates.spec import parse_window


# --- Helpers ---------------------------------------------------------------


def _units(cutoffs=None, unit_ids=None):
    """Two-unit frames with explicit assignment cutoffs."""
    if unit_ids is None:
        unit_ids = ["u1", "u2", "u3"]
    if cutoffs is None:
        cutoffs = ["2026-01-10 00:00", "2026-01-10 00:00", "2026-01-10 00:00"]
    return pd.DataFrame(
        {"unit_id": unit_ids, "assigned_at": pd.to_datetime(cutoffs)}
    )


def _history(rows):
    """Build a history DataFrame from (unit, event, arrival, value) rows."""
    return pd.DataFrame(
        {
            "unit_id": [r[0] for r in rows],
            "event_at": pd.to_datetime([r[1] for r in rows]),
            "arrived_at": pd.to_datetime([r[2] for r in rows]),
            "spend": [r[3] for r in rows],
        }
    )


def _builder(**kwargs):
    params = dict(
        unit_col="unit_id",
        event_time_col="event_at",
        arrival_time_col="arrived_at",
        cutoff_col="assigned_at",
        specs=[CovariateSpec("spend_7d", column="spend", agg="sum")],
        window="7D",
    )
    params.update(kwargs)
    return PointInTimeCovariates(**params)


# --- Spec ------------------------------------------------------------------


class TestParseWindow:
    def test_none_is_all_history(self):
        assert parse_window(None) is None

    def test_string_parses(self):
        assert parse_window("7D") == pd.Timedelta(days=7)

    def test_timedelta_accepted(self):
        assert parse_window(pd.Timedelta(hours=2)) == pd.Timedelta(hours=2)

    def test_zero_rejected(self):
        with pytest.raises(InvalidDesignError, match="positive"):
            parse_window("0D")

    def test_negative_rejected(self):
        with pytest.raises(InvalidDesignError, match="positive"):
            parse_window("-1D")

    def test_garbage_rejected(self):
        with pytest.raises(InvalidDesignError, match="window"):
            parse_window("not-a-window")

    def test_wrong_type_rejected(self):
        with pytest.raises(InvalidDesignError, match="window"):
            parse_window(7)  # type: ignore[arg-type]


class TestCovariateSpec:
    def test_default_column_is_name(self):
        spec = CovariateSpec("spend", agg="sum")
        assert spec.column == "spend"
        assert spec.roles == ("adjustment",)

    def test_count_has_no_column(self):
        spec = CovariateSpec("n_events", agg="count")
        assert spec.column is None
        assert spec.requires_numeric is False

    def test_roles_normalized_and_deduplicated(self):
        spec = CovariateSpec(
            "spend", agg="sum", role=["balance", "balance", "adjustment"]
        )
        assert spec.roles == ("balance", "adjustment")

    def test_unknown_role_rejected(self):
        with pytest.raises(InvalidDesignError, match="role"):
            CovariateSpec("spend", agg="sum", role="nonsense")

    def test_unknown_agg_rejected(self):
        with pytest.raises(InvalidDesignError, match="agg"):
            CovariateSpec("spend", agg="median_absolute_deviation")

    def test_to_dict_round_trips(self):
        spec = CovariateSpec("spend", agg="sum", window="7D", role="balance")
        d = spec.to_dict()
        assert d == {
            "name": "spend",
            "column": "spend",
            "agg": "sum",
            "window": "7 days 00:00:00",
            "roles": ["balance"],
        }

    def test_repr_mentions_name(self):
        assert "spend" in repr(CovariateSpec("spend", agg="sum"))


class TestBuilderValidation:
    def test_missing_cutoff_rejected_at_build(self):
        builder = _builder(cutoff_col=None)
        with pytest.raises(InvalidDesignError, match="cutoff"):
            builder.build(_units(), _history([]))

    def test_both_cutoffs_rejected(self):
        builder = _builder()
        with pytest.raises(InvalidDesignError, match="not both"):
            builder.build(_units(), _history([]), cutoff="2026-01-01")

    def test_strategy_mismatch_with_agg_rejected(self):
        with pytest.raises(InvalidDesignError, match="agg must be None"):
            PointInTimeCovariates(
                unit_col="unit_id",
                event_time_col="event_at",
                strategy="last_visible",
                specs=[CovariateSpec("spend", column="spend", agg="sum")],
            )

    def test_last_visible_rejects_per_column_window(self):
        with pytest.raises(InvalidDesignError, match="window"):
            PointInTimeCovariates(
                unit_col="unit_id",
                event_time_col="event_at",
                strategy="last_visible",
                specs=[CovariateSpec("spend", window="7D")],
            )

    def test_reserved_name_rejected(self):
        with pytest.raises(InvalidDesignError, match="structural"):
            _builder(specs=[CovariateSpec("unit_id", agg="sum")])

    def test_duplicate_names_rejected(self):
        with pytest.raises(InvalidDesignError, match="Duplicate"):
            _builder(
                specs=[
                    CovariateSpec("spend", agg="sum"),
                    CovariateSpec("spend", agg="mean"),
                ]
            )

    def test_unknown_fillna_key_rejected(self):
        with pytest.raises(InvalidDesignError, match="unknown"):
            _builder(fillna={"other": 1.0})

    def test_non_numeric_fillna_rejected(self):
        with pytest.raises(InvalidDesignError, match="fillna"):
            _builder(fillna={"spend_7d": "zero"})

    def test_non_finite_fillna_rejected(self):
        with pytest.raises(InvalidDesignError, match="finite"):
            _builder(fillna=float("nan"))
        with pytest.raises(InvalidDesignError, match="finite"):
            _builder(fillna={"spend_7d": float("inf")})

    def test_non_numeric_source_rejected(self):
        history = _history([("u1", "2026-01-05", "2026-01-05", 1.0)])
        history["spend"] = history["spend"].astype(str)
        with pytest.raises(InvalidDesignError, match="numeric"):
            _builder().build(_units(), history)

    def test_missing_history_column_rejected(self):
        history = _history([]).drop(columns=["spend"])
        with pytest.raises(InvalidDesignError, match="missing"):
            _builder().build(_units(), history)

    def test_duplicate_units_rejected(self):
        units = pd.concat([_units(), _units()])
        with pytest.raises(InvalidDesignError, match="duplicate"):
            _builder().build(units, _history([]))

    def test_missing_cutoff_value_rejected(self):
        units = _units()
        units.loc[1, "assigned_at"] = pd.NaT
        with pytest.raises(InvalidDesignError, match="missing"):
            _builder().build(units, _history([]))


# --- Aggregation semantics -------------------------------------------------


class TestBuildAggregate:
    def test_window_is_relative_to_each_cutoff(self):
        units = _units(["2026-01-10", "2026-01-10", "2026-01-12"])
        history = _history(
            [
                ("u1", "2026-01-09", "2026-01-09", 1.0),  # inside 7D
                ("u1", "2026-01-01", "2026-01-01", 100.0),  # outside 7D
                ("u3", "2026-01-06", "2026-01-06", 5.0),  # inside u3's 7D
            ]
        )
        frame = _builder().build(units, history)
        assert frame.data.loc["u1", "spend_7d"] == 1.0
        assert frame.data.loc["u3", "spend_7d"] == 5.0

    def test_event_before_cutoff_but_arrival_after_is_excluded(self):
        """The core look-ahead case: known late, not knowable at cutoff."""
        history = _history(
            [
                ("u1", "2026-01-09", "2026-01-11", 50.0),
                ("u1", "2026-01-09", "2026-01-09", 2.0),
            ]
        )
        frame = _builder().build(_units(), history)
        assert frame.data.loc["u1", "spend_7d"] == 2.0
        assert frame.n_invisible_rows == 1
        # The late-arriving row is exactly the look-ahead row.
        assert frame.n_lookahead_rows == 1

    def test_event_after_cutoff_excluded(self):
        history = _history([("u1", "2026-01-11", "2026-01-11", 50.0)])
        frame = _builder().build(_units(), history)
        assert frame.data.loc["u1", "spend_7d"] == 0.0
        assert frame.filled_counts["spend_7d"] == 3
        # Post-cutoff event: invisible, but not a look-ahead row.
        assert frame.n_invisible_rows == 1
        assert frame.n_lookahead_rows == 0

    def test_missing_arrival_is_not_provable(self):
        history = _history([("u1", "2026-01-09", "2026-01-09", 1.0)])
        history.loc[0, "arrived_at"] = pd.NaT
        frame = _builder().build(_units(), history)
        assert frame.data.loc["u1", "spend_7d"] == 0.0
        assert frame.n_invisible_rows == 1

    def test_without_arrival_column_uses_event_time(self):
        builder = _builder(arrival_time_col=None)
        history = _history([("u1", "2026-01-09", "2026-01-11", 4.0)])
        frame = builder.build(_units(), history)
        # No arrival channel: the row is treated as visible at event time.
        assert frame.data.loc["u1", "spend_7d"] == 4.0

    def test_aggregations(self):
        history = _history(
            [
                ("u1", "2026-01-09", "2026-01-09", 1.0),
                ("u1", "2026-01-08", "2026-01-08", 3.0),
            ]
        )
        specs = [
            CovariateSpec("total", column="spend", agg="sum"),
            CovariateSpec("avg", column="spend", agg="mean"),
            CovariateSpec("hi", column="spend", agg="max"),
            CovariateSpec("lo", column="spend", agg="min"),
            CovariateSpec("n", agg="count"),
        ]
        frame = _builder(specs=specs).build(_units(), history)
        row = frame.data.loc["u1"]
        assert row["total"] == 4.0
        assert row["avg"] == 2.0
        assert row["hi"] == 3.0
        assert row["lo"] == 1.0
        assert row["n"] == 2.0

    def test_per_column_window_overrides_builder(self):
        history = _history(
            [
                ("u1", "2026-01-09", "2026-01-09", 1.0),
                ("u1", "2026-01-04", "2026-01-04", 10.0),
            ]
        )
        specs = [
            CovariateSpec("d1", column="spend", agg="sum", window="1D"),
            CovariateSpec("d7", column="spend", agg="sum", window=None),
        ]
        frame = _builder(specs=specs).build(_units(), history)
        assert frame.data.loc["u1", "d1"] == 1.0
        # window=None inherits the builder-level window (7D here), so
        # both Jan 9 and Jan 2 are inside.
        assert frame.data.loc["u1", "d7"] == 11.0

    def test_builder_window_applies_when_spec_has_none(self):
        history = _history(
            [
                ("u1", "2026-01-09", "2026-01-09", 1.0),
                ("u1", "2026-01-02", "2026-01-02", 10.0),
            ]
        )
        frame = _builder(window="1D").build(_units(), history)
        assert frame.data.loc["u1", "spend_7d"] == 1.0

    def test_first_and_last_respect_event_order(self):
        history = _history(
            [
                # Rows handed out of event order on purpose.
                ("u1", "2026-01-09", "2026-01-09", 20.0),
                ("u1", "2026-01-08", "2026-01-08", 10.0),
            ]
        )
        specs = [
            CovariateSpec("first_v", column="spend", agg="first"),
            CovariateSpec("last_v", column="spend", agg="last"),
        ]
        frame = _builder(specs=specs).build(_units(), history)
        assert frame.data.loc["u1", "first_v"] == 10.0
        assert frame.data.loc["u1", "last_v"] == 20.0

    def test_fill_rule_is_frozen_and_reported(self):
        frame = _builder(fillna=-1.0).build(_units(), _history([]))
        assert (frame.data["spend_7d"] == -1.0).all()
        assert frame.fill_rules == {"spend_7d": -1.0}
        assert frame.filled_counts == {"spend_7d": 3}

    def test_fillna_per_column(self):
        frame = _builder(
            fillna={"spend_7d": 99.0},
            specs=[
                CovariateSpec("spend_7d", column="spend", agg="sum"),
                CovariateSpec("events", agg="count"),
            ],
        ).build(_units(), _history([]))
        assert (frame.data["spend_7d"] == 99.0).all()
        assert (frame.data["events"] == 0.0).all()
        # A true zero count is not a fill.
        assert frame.filled_counts["events"] == 0

    def test_history_of_unrelated_units_ignored(self):
        history = _history([("ghost", "2026-01-09", "2026-01-09", 7.0)])
        frame = _builder().build(_units(), history)
        assert "ghost" not in frame.data.index
        assert frame.n_invisible_rows == 0  # no cutoff, not "invisible"

    def test_row_order_does_not_change_values(self):
        history = _history(
            [
                ("u1", "2026-01-09", "2026-01-09", 1.0),
                ("u1", "2026-01-08", "2026-01-08", 2.0),
            ]
        )
        f1 = _builder().build(_units(), history)
        f2 = _builder().build(_units(), history.iloc[::-1].reset_index(drop=True))
        pd.testing.assert_frame_equal(f1.data, f2.data)

    def test_metadata_records_versions_and_roles(self):
        frame = _builder(
            specs=[
                CovariateSpec(
                    "spend_7d", column="spend", agg="sum", role="balance"
                )
            ],
            history_version="snapshot-3",
            cutoff_version="2026-01-10",
        ).build(_units(), _history([]))
        meta = frame.metadata()
        assert meta["history_version"] == "snapshot-3"
        assert meta["cutoff_version"] == "2026-01-10"
        assert meta["roles"] == {"spend_7d": ["balance"]}
        assert meta["fill_rules"] == {"spend_7d": 0.0}
        assert isinstance(meta["fingerprint"], str)

    def test_build_time_version_overrides(self):
        frame = _builder(history_version="default").build(
            _units(), _history([]), history_version="explicit"
        )
        assert frame.history_version == "explicit"


class TestBuildLastVisible:
    def _builder_last(self, **kwargs):
        params = dict(
            unit_col="unit_id",
            event_time_col="event_at",
            arrival_time_col="arrived_at",
            cutoff_col="assigned_at",
            strategy="last_visible",
            specs=[
                CovariateSpec("status_value", column="spend"),
            ],
        )
        params.update(kwargs)
        return PointInTimeCovariates(**params)

    def test_picks_latest_visible_event(self):
        history = _history(
            [
                ("u1", "2026-01-05", "2026-01-05", 2.0),
                ("u1", "2026-01-09", "2026-01-09", 9.0),
            ]
        )
        frame = self._builder_last().build(_units(), history)
        assert frame.data.loc["u1", "status_value"] == 9.0

    def test_skips_row_that_arrived_late(self):
        """An event at the end of the window that arrived late loses."""
        history = _history(
            [
                ("u1", "2026-01-05", "2026-01-05", 2.0),
                ("u1", "2026-01-09", "2026-01-12", 900.0),
            ]
        )
        frame = self._builder_last().build(_units(), history)
        assert frame.data.loc["u1", "status_value"] == 2.0

    def test_builder_window_limits_visibility(self):
        history = _history([("u1", "2026-01-01", "2026-01-01", 1.0)])
        frame = self._builder_last(window="2D").build(_units(), history)
        assert frame.data.loc["u1", "status_value"] == 0.0
        assert frame.filled_counts["status_value"] == 3


# --- Freezing / integrity --------------------------------------------------


class TestFrozenIntegrity:
    def _frame(self):
        history = _history([("u1", "2026-01-09", "2026-01-09", 5.0)])
        return _builder().build(_units(), history)

    def test_fingerprint_is_stable_and_order_independent(self):
        frame = self._frame()
        assert frame.fingerprint() == frame.fingerprint()
        reordered = frame.data.iloc[::-1].copy()
        other = CovariateFrame(
            data=reordered,
            roles=frame.roles,
            specs=frame.specs,
            strategy=frame.strategy,
            window=frame.window,
            cutoffs=frame.cutoffs,
            fill_rules=frame.fill_rules,
            filled_counts=frame.filled_counts,
            unit_col=frame.unit_col,
            event_time_col=frame.event_time_col,
            arrival_time_col=frame.arrival_time_col,
            n_history_rows=0,
            n_invisible_rows=0,
            n_lookahead_rows=0,
        )
        assert other.fingerprint() == frame.fingerprint()

    def test_appended_rows_do_not_change_frozen_values(self):
        units = _units()
        early = _history([("u1", "2026-01-09", "2026-01-09", 5.0)])
        f1 = _builder().build(units, early)

        appended = pd.concat(
            [
                early,
                _history(
                    [
                        ("u1", "2026-01-11", "2026-01-11", 500.0),
                        ("u2", "2026-01-09", "2026-01-20", 700.0),
                    ]
                ),
            ],
            ignore_index=True,
        )
        f2 = _builder().build(units, appended)
        pd.testing.assert_frame_equal(f1.data, f2.data)
        assert f1.fingerprint() == f2.fingerprint()

    def test_rebuild_from_newer_cutoffs_changes_fingerprint(self):
        history = _history([("u1", "2026-01-09", "2026-01-09", 5.0)])
        f1 = _builder().build(_units(), history)
        later = _units(["2026-01-15", "2026-01-15", "2026-01-15"])
        f2 = _builder(window=None).build(later, history)
        assert f1.fingerprint() != f2.fingerprint()

    def test_assert_unchanged_accepts_same_frame(self):
        frame = self._frame()
        frame.assert_unchanged(frame)  # no raise

    def test_assert_unchanged_detects_tampering(self):
        frame = self._frame()
        tampered = frame.data.copy()
        tampered.loc["u1", "spend_7d"] = 999.0
        with pytest.raises(FrozenCovariateViolation, match="changed"):
            frame.assert_unchanged(tampered)

    def test_assert_unchanged_on_dataframe(self):
        frame = self._frame()
        attached = frame.attach(_units())
        frame.assert_unchanged(attached)

    def test_assert_unchanged_rejects_unknown_units(self):
        frame = self._frame()
        strangers = pd.DataFrame(
            {"unit_id": ["u1", "zzz"], "spend_7d": [5.0, 0.0]}
        )
        with pytest.raises(InvalidDesignError, match="not present"):
            frame.assert_unchanged(strangers)

    def test_attach_appends_columns_preserving_order(self):
        frame = self._frame()
        units = _units().iloc[[2, 1, 0]].reset_index(drop=True)
        attached = frame.attach(units)
        assert list(attached["unit_id"]) == ["u3", "u2", "u1"]
        assert "spend_7d" in attached.columns
        assert attached.loc[0, "spend_7d"] == 0.0
        assert attached.loc[2, "spend_7d"] == 5.0

    def test_attach_accepts_matching_existing_column(self):
        frame = self._frame()
        units = frame.attach(_units())
        again = frame.attach(units)
        pd.testing.assert_frame_equal(units, again)

    def test_attach_rejects_conflicting_latest_snapshot(self):
        """The named failure mode: pre-prefix column from a newer join."""
        frame = self._frame()
        units = _units()
        units["spend_7d"] = [1.0, 2.0, 3.0]  # a latest-snapshot join
        with pytest.raises(InvalidDesignError, match="latest snapshot"):
            frame.attach(units)

    def test_attach_rejects_missing_unit_column(self):
        frame = self._frame()
        with pytest.raises(InvalidDesignError, match="unit column"):
            frame.attach(pd.DataFrame({"other": [1]}))

    def test_attach_rejects_duplicate_units(self):
        frame = self._frame()
        units = pd.concat([_units(), _units()])
        with pytest.raises(InvalidDesignError, match="duplicate"):
            frame.attach(units)

    def test_attach_rejects_units_without_frozen_row(self):
        frame = self._frame()
        units = pd.DataFrame({"unit_id": ["u1", "zzz"]})
        with pytest.raises(InvalidDesignError, match="no frozen row"):
            frame.attach(units)

    def test_assert_columns_unchanged_subset(self):
        frame = _builder(
            specs=[
                CovariateSpec("a", column="spend", agg="sum"),
                CovariateSpec("b", column="spend", agg="mean"),
            ]
        ).build(_units(), _history([]))
        df = _units().copy()
        df["a"] = frame.data.loc[df["unit_id"], "a"].to_numpy()
        df["b"] = frame.data.loc[df["unit_id"], "b"].to_numpy()
        # Only "a" needs to be present for a subset check.
        frame.assert_columns_unchanged(df[["unit_id", "a"]], ["a"])

    def test_assert_columns_unchanged_detects_change(self):
        frame = _builder().build(_units(), _history([]))
        df = frame.attach(_units())
        df.loc[0, "spend_7d"] = 42.0
        with pytest.raises(FrozenCovariateViolation):
            frame.assert_columns_unchanged(df, ["spend_7d"])

    def test_assert_columns_unchanged_rejects_non_frozen(self):
        frame = _builder().build(_units(), _history([]))
        df = frame.attach(_units())
        df["other"] = 1.0
        with pytest.raises(InvalidDesignError, match="not in the frozen"):
            frame.assert_columns_unchanged(df, ["other"])

    def test_assert_columns_unchanged_rejects_unknown_unit(self):
        frame = _builder().build(_units(), _history([]))
        df = pd.DataFrame({"unit_id": ["u1", "zzz"], "spend_7d": [0.0, 0.0]})
        with pytest.raises(InvalidDesignError, match="not present"):
            frame.assert_columns_unchanged(df, ["spend_7d"])

    def test_role_columns(self):
        frame = _builder(
            specs=[
                CovariateSpec("a", column="spend", agg="sum", role="balance"),
                CovariateSpec("b", column="spend", agg="mean", role="adjustment"),
                CovariateSpec(
                    "c", column="spend", agg="max", role="descriptive"
                ),
            ]
        ).build(_units(), _history([]))
        assert frame.role_columns("balance") == ["a"]
        assert frame.role_columns("descriptive") == ["c"]
        assert frame.role_columns() == ["a", "b", "c"]
        assert frame.names == ["a", "b", "c"]

    def test_frame_is_complete(self):
        frame = self._frame()
        assert not frame.data.isna().any().any()
        assert isinstance(frame.data.index, pd.Index)
        assert frame.data.index.name == "unit_id"
