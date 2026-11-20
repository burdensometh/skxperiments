"""Point-in-time construction of pre-assignment covariates.

The problem this module solves
------------------------------
Rerandomization and covariate adjustment (Lin, CUPED) are only valid when
the covariates are known **before** treatment assignment. Joining a
"latest users" table at analysis time silently mixes post-assignment
behavior into the design and the estimator, which invalidates the
balance criterion and biases the adjusted estimate.

``PointInTimeCovariates`` builds a frozen covariate table from an event
history, keyed by ``unit_col`` and per-unit assignment cutoff:

1. A history row is **visible** at a cutoff when
   ``event_time <= cutoff`` and ``arrival_time <= cutoff``. The arrival
   condition matters: a row whose event happened before the cutoff but
   was only written to the warehouse afterwards was not knowable at
   assignment time, and including it is look-ahead bias.
2. It is **eligible for a column** when it is visible and also within the
   declared lookback window: ``cutoff - window <= event_time``.
3. Eligible rows are reduced per unit either by an aggregation
   (``strategy="aggregate"``, e.g. ``sum`` over the last 7 days) or by
   selecting the row with the latest visible event time
   (``strategy="last_visible"``, e.g. a CRM status snapshot).
4. Units with no eligible value receive the **frozen fill rule** declared
   for the column, and the number of filled units is recorded.

Because eligibility depends only on the cutoff and on arrival time, rows
appended to the history after assignment (``arrival_time > cutoff``)
can never change the frozen values. ``CovariateFrame.fingerprint`` makes
that guarantee checkable.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from skxperiments.core.exceptions import (
    FrozenCovariateViolation,
    InvalidDesignError,
)
from skxperiments.covariates.spec import (
    CovariateSpec,
    normalize_specs,
    parse_window,
)

# Roles a balance diagnostic reports on by default. Blocking and
# descriptive covariates are excluded: blocking columns are often
# categorical and descriptive ones are not part of the design feature.
DIAGNOSTIC_ROLES = ("balance", "adjustment")


@dataclass(frozen=True, eq=False)
class CovariateFrame:
    """Frozen covariate table tied to per-unit cutoffs.

    Instances are produced by ``PointInTimeCovariates.build``; construct
    one directly only in tests.

    Attributes
    ----------
    data : pd.DataFrame
        One row per unit (index = unit id, name = ``unit_col``) and one
        column per ``CovariateSpec``, in declaration order. Contains no
        NaN: every missing value has been replaced by the frozen fill
        rule.
    roles : dict
        Mapping from covariate name to the tuple of roles it plays.
    specs : tuple of CovariateSpec
        The normalized specifications, in declaration order.
    strategy : str
        ``"aggregate"`` or ``"last_visible"``.
    window : pd.Timedelta or None
        Builder-level default window (``None`` = all history).
    cutoffs : pd.Series
        Per-unit assignment cutoff, indexed by unit id.
    fill_rules : dict
        Frozen fill value per covariate.
    filled_counts : dict
        Number of units whose value came from the fill rule. For
        ``agg="count"`` a unit with no history has a true count of 0 and
        is not counted as filled.
    n_history_rows : int
        Number of history rows supplied at build time.
    n_invisible_rows : int
        History rows (for units present in ``units``) excluded because
        they were not visible at the cutoff: ``event_time > cutoff``,
        ``arrival_time > cutoff``, or unknown arrival.
    n_lookahead_rows : int
        Of those, the rows whose ``event_time <= cutoff`` but whose
        ``arrival_time > cutoff`` or is missing. These are exactly the
        rows a "latest snapshot" join would have leaked into the design
        feature; they are the ones to investigate.
    history_version, cutoff_version : Any
        Free-form labels (a table snapshot id, a commit hash, a date)
        carried through to assignments and results.
    """

    data: pd.DataFrame
    roles: dict[str, tuple[str, ...]]
    specs: tuple[CovariateSpec, ...]
    strategy: str
    window: pd.Timedelta | None
    cutoffs: pd.Series
    fill_rules: dict[str, float]
    filled_counts: dict[str, int]
    unit_col: str
    event_time_col: str
    arrival_time_col: str | None
    n_history_rows: int
    n_invisible_rows: int
    n_lookahead_rows: int
    history_version: Any = None
    cutoff_version: Any = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def names(self) -> list[str]:
        """Covariate names, in declaration order."""
        return [spec.name for spec in self.specs]

    @property
    def n_units_(self) -> int:
        """Number of units in the frozen table."""
        return len(self.data)

    def role_columns(self, *roles: str) -> list[str]:
        """Return covariates playing any of ``roles``, in spec order.

        Parameters
        ----------
        *roles : str
            Roles to select. With no roles, every covariate is returned.
        """
        if len(roles) == 0:
            return self.names
        wanted = set(roles)
        return [
            spec.name
            for spec in self.specs
            if wanted.intersection(self.roles.get(spec.name, ()))
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the frozen table."""
        return self.data.copy()

    def metadata(self) -> dict:
        """Return a JSON-friendly description of the frozen design feature.

        The dict is what assignments and results carry in
        ``covariate_metadata``: covariates, roles, windows, fill rules,
        cutoff and history versions, and the fingerprint.
        """
        return {
            "unit_col": self.unit_col,
            "event_time_col": self.event_time_col,
            "arrival_time_col": self.arrival_time_col,
            "strategy": self.strategy,
            "window": None if self.window is None else str(self.window),
            "specs": [spec.to_dict() for spec in self.specs],
            "roles": {name: list(roles) for name, roles in self.roles.items()},
            "fill_rules": dict(self.fill_rules),
            "filled_counts": dict(self.filled_counts),
            "n_units": self.n_units_,
            "n_history_rows": self.n_history_rows,
            "n_invisible_rows": self.n_invisible_rows,
            "n_lookahead_rows": self.n_lookahead_rows,
            "history_version": self.history_version,
            "cutoff_version": self.cutoff_version,
            "fingerprint": self.fingerprint(),
        }

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------
    @property
    def fingerprint_(self) -> str:
        """Stable hash of unit ids, covariates, and per-unit cutoffs.

        Computed once at construction; the frame is frozen, so it cannot
        change. Exposed both as a property and via ``fingerprint()``.
        """
        cached = self.__dict__.get("_cached_fingerprint")
        if cached is None:
            cached = self._fingerprint(self.data, self.cutoffs)
            object.__setattr__(self, "_cached_fingerprint", cached)
        return cached

    def fingerprint(self) -> str:
        """Stable hash of unit ids, covariates, and per-unit cutoffs."""
        return self.fingerprint_

    @classmethod
    def _fingerprint(
        cls, data: pd.DataFrame, cutoffs: pd.Series
    ) -> str:
        """Hash unit ids, covariate values, and cutoffs (order-independent).

        ``data`` must be indexed by unit id; ``cutoffs`` must be a Series
        indexed the same way. Row order is normalized by sorting on the
        index, so handing the same units in a different order yields the
        same fingerprint.
        """
        # Sort by unit id so the hash is independent of row order, which
        # may change when the same units are handed in a different order.
        ordered = data.sort_index(kind="mergesort")
        unit_index = pd.Index(ordered.index)
        # Raw 8-byte representations: a value-preserving hash that stays
        # cheap enough to run on every assignment construction.
        digest = hashlib.sha256()
        digest.update(json.dumps(list(ordered.columns)).encode("utf-8"))
        _update_hash_from_index(digest, unit_index)
        for column in ordered.columns:
            values = ordered[column].to_numpy(dtype=float, copy=False)
            digest.update(np.ascontiguousarray(values).tobytes())
        sorted_cutoffs = cutoffs.sort_index(kind="mergesort")
        selected = sorted_cutoffs.loc[unit_index]
        if isinstance(selected.dtype, pd.DatetimeTZDtype):
            # Normalize to UTC so the hash is stable across tz handling.
            selected = selected.dt.tz_convert("UTC").dt.tz_localize(None)
        cutoff_values = selected.to_numpy(dtype="datetime64[ns]")
        digest.update(np.ascontiguousarray(cutoff_values).tobytes())
        return digest.hexdigest()

    def fingerprint_from_data(self, df: pd.DataFrame) -> str:
        """Fingerprint the frozen columns as they appear in ``df``.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame containing ``unit_col`` and every covariate, or
            one already indexed by ``unit_col``.

        Raises
        ------
        InvalidDesignError
            If a covariate or the unit column is missing.
        """
        covariate_missing = [c for c in self.names if c not in df]
        if covariate_missing:
            raise InvalidDesignError(
                f"Cannot fingerprint frozen covariates: columns missing "
                f"from DataFrame: {covariate_missing}."
            )

        if self.unit_col in df.columns:
            units = df[self.unit_col]
            indexed = df.set_index(self.unit_col)[self.names]
        elif df.index.name == self.unit_col:
            units = pd.Series(df.index.to_numpy())
            indexed = df[self.names]
        else:
            raise InvalidDesignError(
                f"Cannot fingerprint frozen covariates: unit column "
                f"{self.unit_col!r} is neither a column nor the index "
                f"name of the DataFrame."
            )

        unknown = ~units.isin(self.cutoffs.index).to_numpy()
        if unknown.any():
            raise InvalidDesignError(
                "Cannot fingerprint frozen covariates: DataFrame contains "
                "unit ids not present in the frozen frame, e.g. "
                f"{list(units[unknown].head(5))}."
            )
        for name in self.names:
            if not pd.api.types.is_numeric_dtype(indexed[name]):
                raise InvalidDesignError(
                    f"Cannot fingerprint frozen covariates: column "
                    f"{name!r} is not numeric in the DataFrame (dtype "
                    f"{indexed[name].dtype}). A non-numeric column under a "
                    f"frozen name is a different variable, not the design "
                    f"feature."
                )
        cutoffs = self.cutoffs.reindex(indexed.index)
        if cutoffs.isna().any():
            raise InvalidDesignError(
                "Cannot fingerprint frozen covariates: DataFrame contains "
                "unit ids not present in the frozen frame."
            )
        return self._fingerprint(indexed, cutoffs)

    def _order_preserving_match(self, df: pd.DataFrame) -> bool | None:
        """Cheap check when ``df`` has the frame's units in the same order.

        Returns True/False when the fast path applies, or None when the
        DataFrame is not in the frame's row order (callers then fall back
        to the order-independent fingerprint).

        The fast path is what keeps ``Assignment.draw`` affordable: the
        design re-derives ``df`` from data that was already verified, and
        only the array comparison is needed.
        """
        if any(name not in df.columns for name in self.names):
            return None
        if self.unit_col in df.columns:
            units = df[self.unit_col]
        elif df.index.name == self.unit_col:
            units = pd.Series(df.index.to_numpy(), index=df.index)
        else:
            return None

        if len(units) != len(self.data):
            return None
        # Row-for-row unit identity, otherwise the positional comparison
        # would be meaningless.
        if not np.array_equal(
            units.to_numpy(), self.data.index.to_numpy()
        ):
            return None

        for name in self.names:
            try:
                values = df[name].to_numpy(dtype=float, copy=False)
            except (TypeError, ValueError):
                # Not the frozen dtype: let the fingerprint path report
                # the mismatch with a clear message.
                return None
            expected = self.data[name].to_numpy(dtype=float, copy=False)
            if not np.array_equal(values, expected):
                return False
        return True

    def assert_columns_unchanged(
        self, df: pd.DataFrame, columns: list[str]
    ) -> None:
        """Verify specific frozen columns against ``df`` by unit id.

        Used by designs whose criterion is a subset of the frozen table:
        only the criterion columns need to be present, and each must
        still hold its frozen value.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing ``unit_col`` and ``columns``.
        columns : list of str
            Frozen covariates to verify.

        Raises
        ------
        InvalidDesignError
            If the unit column, a listed covariate, or a unit id is
            missing, or a listed name is not a frozen covariate.
        FrozenCovariateViolation
            If a frozen value differs.
        """
        not_frozen = [c for c in columns if c not in self.names]
        if not_frozen:
            raise InvalidDesignError(
                f"Cannot verify covariates {not_frozen}: not in the frozen "
                f"table. Declared covariates: {self.names}."
            )
        missing = [c for c in [self.unit_col, *columns] if c not in df]
        if missing:
            raise InvalidDesignError(
                f"Cannot verify frozen covariates: columns missing from "
                f"DataFrame: {missing}."
            )

        units = df[self.unit_col]
        unknown = ~units.isin(self.data.index).to_numpy()
        if unknown.any():
            raise InvalidDesignError(
                "Cannot verify frozen covariates: DataFrame contains unit "
                f"ids not present in the frozen frame, e.g. "
                f"{list(units[unknown].head(5))}."
            )
        expected = self.data.loc[units.to_numpy(), columns]
        for name in columns:
            try:
                actual = pd.to_numeric(
                    df[name], errors="raise"
                ).to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise InvalidDesignError(
                    f"Cannot verify frozen covariate {name!r}: the column "
                    f"is not numeric ({exc}). A non-numeric column under a "
                    f"frozen name is a different variable."
                ) from None
            if not np.array_equal(
                actual, expected[name].to_numpy(dtype=float, copy=False)
            ):
                raise FrozenCovariateViolation(
                    f"Frozen covariate {name!r} changed after it was built. "
                    f"The rerandomization criterion / adjustment variable "
                    f"must be the point-in-time value, not a newer "
                    f"snapshot. expected fingerprint "
                    f"{self.fingerprint_[:12]}."
                )

    def assert_unchanged(
        self,
        df_or_frame: "CovariateFrame | pd.DataFrame",
    ) -> None:
        """Verify the frozen covariates were not altered.

        Parameters
        ----------
        df_or_frame : CovariateFrame or pd.DataFrame
            Another frozen frame, or a DataFrame whose covariate columns
            should still match this frame.

        Raises
        ------
        FrozenCovariateViolation
            If the fingerprints differ. This is the guard against
            rebuilding from a newer history snapshot, appending
            post-assignment rows, or re-joining a latest table.
        """
        expected = self.fingerprint_
        if isinstance(df_or_frame, CovariateFrame):
            other = df_or_frame.fingerprint_
        else:
            # Fast path: row order preserved (e.g. a fresh draw). This
            # is an exact value comparison, not a heuristic.
            fast = self._order_preserving_match(df_or_frame)
            if fast is True:
                return
            other = self.fingerprint_from_data(df_or_frame)
        if other != expected:
            raise FrozenCovariateViolation(
                "Frozen covariate table changed after it was built. "
                "Post-assignment data must never alter the design "
                "feature: rebuild only with the original cutoffs and a "
                "history visible at those cutoffs, or start a new "
                "experiment. "
                f"expected fingerprint {expected[:12]}, "
                f"received {other[:12]}."
            )

    # ------------------------------------------------------------------
    # Attachment
    # ------------------------------------------------------------------
    def attach(
        self,
        df: pd.DataFrame,
        *,
        unit_col: str | None = None,
    ) -> pd.DataFrame:
        """Return ``df`` with the frozen covariates joined by unit id.

        Units are matched on ``unit_col`` (default: the builder's
        ``unit_col``). The output keeps ``df``'s row order and index.

        Parameters
        ----------
        df : pd.DataFrame
            Units to analyze. Must contain ``unit_col``.
        unit_col : str or None, optional
            Column used to match rows. Defaults to the builder's.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with covariate columns appended.

        Raises
        ------
        InvalidDesignError
            If ``unit_col`` is missing or duplicated in ``df``; if some
            unit has no frozen row; or if ``df`` already has a column
            with a covariate name and different values (that would mean
            a latest-snapshot column is about to be silently swapped).
        """
        key = self.unit_col if unit_col is None else unit_col
        if key not in df.columns:
            raise InvalidDesignError(
                f"Cannot attach frozen covariates: unit column {key!r} "
                f"not found. Available columns: {list(df.columns)}."
            )
        if df[key].duplicated().any():
            dupes = df.loc[df[key].duplicated(), key].unique()[:5]
            raise InvalidDesignError(
                f"Cannot attach frozen covariates: unit column {key!r} "
                f"has duplicate values, e.g. {list(dupes)}."
            )

        aligned = self.data.reindex(df[key].to_numpy())
        missing_mask = aligned.isna().any(axis=1).to_numpy()
        if missing_mask.any():
            missing_units = df.loc[missing_mask, key].head(5).tolist()
            raise InvalidDesignError(
                f"Cannot attach frozen covariates: {int(missing_mask.sum())} "
                f"unit(s) have no frozen row, e.g. {missing_units}. Build "
                f"the covariate table for exactly the analysis units."
            )

        out = df.copy()
        aligned = aligned.reset_index(drop=True)
        aligned.index = out.index
        for name in self.names:
            values = aligned[name].to_numpy()
            if name in out.columns:
                existing = pd.to_numeric(out[name], errors="coerce").to_numpy(
                    dtype=float
                )
                if not np.array_equal(
                    np.nan_to_num(existing, nan=-np.inf),
                    np.nan_to_num(values, nan=-np.inf),
                ):
                    raise InvalidDesignError(
                        f"DataFrame already has a column {name!r} with "
                        f"values differing from the frozen covariate table. "
                        f"This is the look-ahead case: do not merge a "
                        f"latest snapshot under a similar name. Drop the "
                        f"column or pass the frozen table as the only "
                        f"source."
                    )
                existing_index = out.columns.get_loc(name)
                out.iloc[:, existing_index] = values
            else:
                out[name] = values
        return out


def _update_hash_from_index(digest: "hashlib._Hash", index: pd.Index) -> None:
    """Feed unit ids into ``digest`` in a type-stable way."""
    try:
        values = index.to_numpy(dtype="int64", copy=False)
        digest.update(np.ascontiguousarray(values).tobytes())
    except (TypeError, ValueError):
        # Non-numeric ids (the common case): hash their string form.
        payload = "\x1f".join(str(v) for v in index).encode("utf-8")
        digest.update(payload)


def _to_datetime(series: pd.Series, name: str) -> pd.Series:
    """Parse ``series`` as datetimes, raising a clear error on failure."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    try:
        return pd.to_datetime(series, errors="raise")
    except (ValueError, TypeError) as exc:
        raise InvalidDesignError(
            f"Column {name!r} must hold datetimes; parsing failed "
            f"({exc}). Point-in-time covariates require event, arrival "
            f"and cutoff times on a common datetime scale."
        ) from None


class PointInTimeCovariates:
    """Build a frozen pre-assignment covariate table from event history.

    Parameters
    ----------
    unit_col : str
        Unit identifier present in both the units and the history table.
    event_time_col : str
        History column with the time the event happened.
    specs : sequence of CovariateSpec
        Declarative list of covariates to build.
    arrival_time_col : str or None, optional
        History column with the time the event became visible to the
        analysis pipeline. Strongly recommended: without it look-ahead
        cannot be detected and event time is used as arrival time.
    cutoff_col : str or None, optional
        Units column with the per-unit assignment cutoff. Either this or
        the ``cutoff`` argument of ``build`` must be supplied.
    strategy : {"aggregate", "last_visible"}, optional
        How eligible rows become one value per unit. Default
        ``"aggregate"``.
    window : str, pd.Timedelta or None, optional
        Default lookback window, overridable per spec in the
        ``"aggregate"`` strategy. ``None`` means all history up to the
        cutoff. Default None.
    fillna : float or dict, optional
        Frozen fill rule for units with no eligible value. A scalar
        applies to every covariate; a dict overrides per name. Default
        0.0. The resolved rule is recorded per covariate and never
        re-derived at analysis time.
    cutoff_version : Any, optional
        Default version label stored on every frame this builder makes.
    history_version : Any, optional
        Default history snapshot label.

    Notes
    -----
    A row is visible when ``event_time <= cutoff`` **and**
    ``arrival_time <= cutoff``. Rows with a missing arrival time are
    excluded (treated as "not provably visible") and reported in
    ``n_invisible_rows``. This is deliberately conservative: excluding a
    stale row costs efficiency, including a future row invalidates the
    design.
    """

    def __init__(
        self,
        unit_col: str,
        event_time_col: str,
        specs: list[CovariateSpec] | tuple[CovariateSpec, ...],
        arrival_time_col: str | None = None,
        cutoff_col: str | None = None,
        strategy: str = "aggregate",
        window: str | pd.Timedelta | datetime.timedelta | None = None,
        fillna: float | dict[str, float] = 0.0,
        cutoff_version: Any = None,
        history_version: Any = None,
    ) -> None:
        for label, value in (
            ("unit_col", unit_col),
            ("event_time_col", event_time_col),
            ("arrival_time_col", arrival_time_col),
            ("cutoff_col", cutoff_col),
        ):
            if value is not None and (
                not isinstance(value, str) or not value
            ):
                raise InvalidDesignError(
                    f"{label} must be a non-empty string or None; "
                    f"received {value!r}."
                )
        if arrival_time_col is not None and arrival_time_col in (
            unit_col,
            event_time_col,
        ):
            raise InvalidDesignError(
                "arrival_time_col must differ from unit_col and "
                "event_time_col."
            )
        if cutoff_col is not None and cutoff_col in (
            unit_col,
            event_time_col,
        ):
            raise InvalidDesignError(
                "cutoff_col must differ from unit_col and event_time_col."
            )

        self.specs: list[CovariateSpec] = normalize_specs(
            specs,
            strategy=strategy,
            unit_col=unit_col,
            event_time_col=event_time_col,
            arrival_time_col=arrival_time_col,
        )
        self.unit_col = unit_col
        self.event_time_col = event_time_col
        self.arrival_time_col = arrival_time_col
        self.cutoff_col = cutoff_col
        self.strategy = strategy
        self.window = parse_window(window)
        self.fillna = self._validate_fillna(fillna)
        self.cutoff_version = cutoff_version
        self.history_version = history_version

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_fillna(
        self, fillna: float | dict[str, float]
    ) -> float | dict[str, float]:
        names = {spec.name for spec in self.specs}
        if isinstance(fillna, dict):
            unknown = sorted(set(fillna) - names)
            if unknown:
                raise InvalidDesignError(
                    f"fillna has entries for unknown covariates: {unknown}. "
                    f"Known: {sorted(names)}."
                )
            cleaned: dict[str, float] = {}
            for key, value in fillna.items():
                if not isinstance(value, (int, float)) or isinstance(
                    value, bool
                ):
                    raise InvalidDesignError(
                        f"fillna[{key!r}] must be numeric; received "
                        f"{value!r}."
                    )
                if not math.isfinite(value):
                    raise InvalidDesignError(
                        f"fillna[{key!r}] must be finite; received "
                        f"{value!r}. A NaN/inf fill would leave the frozen "
                        f"table incomplete."
                    )
                cleaned[key] = float(value)
            return cleaned
        if isinstance(fillna, bool) or not isinstance(fillna, (int, float)):
            raise InvalidDesignError(
                f"fillna must be a number or a dict of numbers; received "
                f"{type(fillna).__name__}. The fill rule is frozen and "
                f"applied at build time."
            )
        if not math.isfinite(fillna):
            raise InvalidDesignError(
                f"fillna must be finite; received {fillna!r}. A NaN/inf "
                f"fill would leave the frozen table incomplete."
            )
        return float(fillna)

    def _fill_for(self, name: str) -> float:
        if isinstance(self.fillna, dict):
            return self.fillna.get(name, 0.0)
        return self.fillna

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(
        self,
        units: pd.DataFrame,
        history: pd.DataFrame,
        *,
        cutoff: Any = None,
        cutoff_version: Any = None,
        history_version: Any = None,
    ) -> CovariateFrame:
        """Build and freeze the covariate table.

        Parameters
        ----------
        units : pd.DataFrame
            One row per analysis unit. Must contain ``unit_col`` and,
            unless ``cutoff`` is given, ``cutoff_col``.
        history : pd.DataFrame
            Long event history at unit level. May contain rows after the
            cutoffs and rows for units not in ``units``; both are
            excluded.
        cutoff : datetime-like, optional
            A single cutoff applied to every unit. Mutually exclusive
            with ``cutoff_col``.
        cutoff_version, history_version : Any, optional
            Labels overriding the builder defaults for this build.

        Returns
        -------
        CovariateFrame

        Raises
        ------
        InvalidDesignError
            For missing columns, unparseable times, duplicate units,
            ambiguous cutoffs, or a spec column that cannot be aggregated.
        """
        if self.cutoff_col is not None and cutoff is not None:
            raise InvalidDesignError(
                "Provide either cutoff_col on the builder or a single "
                "cutoff at build time, not both."
            )
        if self.cutoff_col is None and cutoff is None:
            raise InvalidDesignError(
                "A per-unit cutoff is required: pass cutoff_col to the "
                "builder or cutoff= to build(). Without a cutoff the "
                "covariate table would be built from a latest snapshot."
            )

        self._validate_frames(units, history)
        cutoffs = self._resolve_cutoffs(units, cutoff)

        hist = self._prepare_history(history, cutoffs)

        if self.strategy == "last_visible":
            values = self._build_last_visible(hist, units)
            filled = {name: int(values[name].isna().sum()) for name in values}
        else:
            values, filled = self._build_aggregated(hist, units, cutoffs)

        # Apply the frozen fill rule.
        fill_rules: dict[str, float] = {}
        for name in values:
            rule = self._fill_for(name)
            fill_rules[name] = rule
            values[name] = values[name].fillna(rule).astype(float)

        data = pd.DataFrame(values, index=pd.Index(
            units[self.unit_col].to_numpy(), name=self.unit_col
        ))
        data = data[list(values.keys())]

        visible = hist["_visible"].to_numpy()
        has_cutoff = hist["_has_cutoff"].to_numpy()
        not_visible = has_cutoff & ~visible
        n_invisible = int(not_visible.sum())
        event_known = (hist["_event"] <= hist["_cutoff"]).to_numpy()
        n_lookahead = int((not_visible & event_known).sum())

        roles = {spec.name: tuple(spec.roles) for spec in self.specs}
        frame = CovariateFrame(
            data=data,
            roles=roles,
            specs=tuple(self.specs),
            strategy=self.strategy,
            window=self.window,
            cutoffs=cutoffs,
            fill_rules=fill_rules,
            filled_counts=filled,
            unit_col=self.unit_col,
            event_time_col=self.event_time_col,
            arrival_time_col=self.arrival_time_col,
            n_history_rows=int(len(history)),
            n_invisible_rows=n_invisible,
            n_lookahead_rows=n_lookahead,
            history_version=(
                self.history_version
                if history_version is None
                else history_version
            ),
            cutoff_version=(
                self.cutoff_version
                if cutoff_version is None
                else cutoff_version
            ),
        )
        # Cheap self-check: the frame must be complete and finite.
        if frame.data.isna().any().any():
            raise InvalidDesignError(
                "Internal error: frozen covariate table still contains "
                "NaN after applying the fill rule."
            )
        # Warm the fingerprint cache while we know the data is pristine.
        _ = frame.fingerprint_
        return frame

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------
    def _validate_frames(
        self, units: pd.DataFrame, history: pd.DataFrame
    ) -> None:
        if not isinstance(units, pd.DataFrame) or not isinstance(
            history, pd.DataFrame
        ):
            raise InvalidDesignError(
                "units and history must be pandas DataFrames."
            )
        if self.unit_col not in units.columns:
            raise InvalidDesignError(
                f"Unit column {self.unit_col!r} not found in units. "
                f"Available columns: {list(units.columns)}."
            )
        need_history = {self.unit_col, self.event_time_col}
        if self.arrival_time_col is not None:
            need_history.add(self.arrival_time_col)
        for spec in self.specs:
            if spec.column is not None:
                need_history.add(spec.column)
        missing = sorted(need_history - set(history.columns))
        if missing:
            raise InvalidDesignError(
                f"History is missing required columns: {missing}. "
                f"Available columns: {list(history.columns)}."
            )
        if units[self.unit_col].duplicated().any():
            dupes = units.loc[
                units[self.unit_col].duplicated(), self.unit_col
            ].unique()[:5]
            raise InvalidDesignError(
                f"units has duplicate {self.unit_col!r} values, e.g. "
                f"{list(dupes)}. One row per unit is required."
            )
        if self.cutoff_col is not None and (
            self.cutoff_col not in units.columns
        ):
            raise InvalidDesignError(
                f"Cutoff column {self.cutoff_col!r} not found in units. "
                f"Available columns: {list(units.columns)}."
            )

        for spec in self.specs:
            if spec.column is None:
                continue
            if spec.requires_numeric and not pd.api.types.is_numeric_dtype(
                history[spec.column]
            ):
                raise InvalidDesignError(
                    f"Covariate {spec.name!r} uses agg={spec.agg!r} on "
                    f"column {spec.column!r}, which must be numeric; dtype "
                    f"is {history[spec.column].dtype}."
                )

    def _resolve_cutoffs(
        self, units: pd.DataFrame, cutoff: Any
    ) -> pd.Series:
        if cutoff is not None:
            parsed = pd.to_datetime([cutoff], errors="raise")[0]
            values = pd.Series(
                parsed, index=units[self.unit_col].to_numpy()
            )
        else:
            parsed = _to_datetime(units[self.cutoff_col], self.cutoff_col)
            if parsed.isna().any():
                missing_units = units.loc[
                    parsed.isna(), self.unit_col
                ].head(5).tolist()
                raise InvalidDesignError(
                    f"Cutoff column {self.cutoff_col!r} has missing values "
                    f"for units {missing_units}. Every unit needs an "
                    f"explicit assignment cutoff."
                )
            values = pd.Series(
                parsed.to_numpy(), index=units[self.unit_col].to_numpy()
            )
        return values

    def _prepare_history(
        self, history: pd.DataFrame, cutoffs: pd.Series
    ) -> pd.DataFrame:
        hist = history.copy()
        hist["_row_order"] = np.arange(len(hist))
        hist["_event"] = _to_datetime(
            hist[self.event_time_col], self.event_time_col
        )
        if hist["_event"].isna().any():
            raise InvalidDesignError(
                f"Event time column {self.event_time_col!r} has missing "
                f"values. Every history row needs an event time."
            )
        if self.arrival_time_col is not None:
            hist["_arrival"] = _to_datetime(
                hist[self.arrival_time_col], self.arrival_time_col
            )
        else:
            # Without an arrival column, assume the event was visible as
            # soon as it happened. Look-ahead cannot be detected then,
            # which is why providing arrival_time_col is recommended.
            hist["_arrival"] = hist["_event"]

        hist["_cutoff"] = hist[self.unit_col].map(cutoffs)
        hist["_has_cutoff"] = hist["_cutoff"].notna()

        arrival_ok = hist["_arrival"].notna() & (
            hist["_arrival"] <= hist["_cutoff"]
        )
        hist["_visible"] = (
            hist["_has_cutoff"]
            & (hist["_event"] <= hist["_cutoff"])
            & arrival_ok
        )
        return hist

    def _eligible(
        self, hist: pd.DataFrame, window: pd.Timedelta | None
    ) -> pd.Series:
        if window is None:
            return hist["_visible"]
        return hist["_visible"] & (
            hist["_event"] >= hist["_cutoff"] - window
        )

    def _unit_order(self, units: pd.DataFrame) -> list:
        return units[self.unit_col].tolist()

    def _build_last_visible(
        self,
        hist: pd.DataFrame,
        units: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        """Take the latest visible row per unit (visibility pre-encoded)."""
        eligible = hist.loc[self._eligible(hist, self.window)]
        order = ["_event", "_arrival", "_row_order"]
        eligible = eligible.sort_values(order, kind="mergesort")
        last = eligible.groupby(self.unit_col, sort=False).tail(1)
        last = last.set_index(self.unit_col)

        index = pd.Index(self._unit_order(units), name=self.unit_col)
        values: dict[str, pd.Series] = {}
        for spec in self.specs:
            # ``column`` defaults to the covariate name when the spec
            # was declared without an explicit source column.
            source = spec.column if spec.column is not None else spec.name
            values[spec.name] = (
                pd.to_numeric(last[source], errors="coerce")
                .reindex(index)
                .astype(float)
            )
        return values

    def _build_aggregated(
        self,
        hist: pd.DataFrame,
        units: pd.DataFrame,
        cutoffs: pd.Series,
    ) -> tuple[dict[str, pd.Series], dict[str, int]]:
        index = pd.Index(self._unit_order(units), name=self.unit_col)
        values: dict[str, pd.Series] = {}
        filled: dict[str, int] = {}

        for spec in self.specs:
            window = spec.window if spec.window is not None else self.window
            eligible = hist.loc[self._eligible(hist, window)]

            if spec.agg == "count":
                counts = eligible.groupby(
                    self.unit_col, sort=False
                ).size()
                series = (
                    counts.reindex(index).fillna(0.0).astype(float)
                )
                # A true zero count is not "missing"; no fill applies.
                values[spec.name] = series
                filled[spec.name] = 0
                continue

            ordered = eligible.sort_values(
                ["_event", "_arrival", "_row_order"], kind="mergesort"
            )
            grouped = ordered.groupby(self.unit_col, sort=False)[spec.column]
            aggregated = grouped.agg(spec.agg)
            non_null = grouped.count()

            series = pd.to_numeric(aggregated, errors="coerce").reindex(index)
            present = non_null.reindex(index).fillna(0)
            series = series.where(present > 0)
            filled[spec.name] = int(series.isna().sum())
            values[spec.name] = series.astype(float)

        return values, filled


def resolve_covariates(
    assignment: Any,
    covariates: list[str] | None,
    roles: tuple[str, ...] | list[str] | str = DIAGNOSTIC_ROLES,
) -> list[str]:
    """Resolve a covariate list against an assignment's frozen table.

    Used by diagnostics and estimators so that, when the user does not
    name columns explicitly, the *declared* point-in-time covariates are
    used instead of every numeric column in ``assignment.data_`` (which
    may include post-assignment metrics).

    Parameters
    ----------
    assignment : BaseAssignment
        Assignment that may carry ``frozen_covariates_``.
    covariates : list of str or None
        Explicit names, or None to use declared roles.
    roles : str or sequence of str, optional
        Roles to select when ``covariates`` is None. Default is every
        diagnostic role.

    Returns
    -------
    list of str
        Explicit names unchanged, or the declared names matching
        ``roles``. An empty list means there is no frozen table and no
        explicit names: callers should fall back to their own selection.

    Raises
    ------
    InvalidDesignError
        If explicit names are given but some are absent from the frozen
        table, or if the frozen table declares no covariate with the
        requested roles.
    """
    frame = getattr(assignment, "frozen_covariates_", None)
    if covariates is not None:
        if frame is not None:
            unknown = [c for c in covariates if c not in frame.names]
            if unknown:
                raise InvalidDesignError(
                    f"Covariates {unknown} are not in the frozen "
                    f"covariate table. Declared covariates: "
                    f"{frame.names}. Build them with "
                    f"PointInTimeCovariates instead of joining another "
                    f"table."
                )
        return list(covariates)

    if frame is None:
        return []

    wanted = (roles,) if isinstance(roles, str) else tuple(roles)
    selected = frame.role_columns(*wanted)
    if not selected:
        raise InvalidDesignError(
            f"The frozen covariate table declares no covariate with "
            f"role(s) {list(wanted)}. Declared roles: "
            f"{ {n: list(r) for n, r in frame.roles.items()} }."
        )
    return selected
