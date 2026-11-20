"""Declarative specification of point-in-time covariates.

A ``CovariateSpec`` names one output covariate, the history column it is
built from, how that column is aggregated over the declared window, and
the role the covariate plays in the experiment (balance, adjustment,
blocking, or descriptive).

The specification is deliberately data-free: it can be declared once,
reviewed, and versioned alongside the experiment. The actual values are
produced by ``PointInTimeCovariates.build``.
"""

from __future__ import annotations

import datetime

import pandas as pd

from skxperiments.core.exceptions import InvalidDesignError

#: Aggregations that reduce the eligible history rows of a column to one value.
AGGREGATIONS = (
    "sum",
    "mean",
    "min",
    "max",
    "std",
    "median",
    "count",
    "last",
    "first",
)

#: Roles a covariate may play. A column may hold several roles at once.
VALID_ROLES = ("balance", "adjustment", "blocking", "descriptive")

#: Builder strategies.
VALID_STRATEGIES = ("aggregate", "last_visible")


def normalize_roles(roles: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Validate and normalize a role or collection of roles.

    Parameters
    ----------
    roles : str or sequence of str
        One role or several.

    Returns
    -------
    tuple of str
        Deduplicated roles in the declared order.

    Raises
    ------
    InvalidDesignError
        If a role is unknown or the container is empty.
    """
    if isinstance(roles, str):
        candidates = [roles]
    elif isinstance(roles, (tuple, list)):
        candidates = list(roles)
    else:
        raise InvalidDesignError(
            f"roles must be a string or a sequence of strings; received "
            f"{type(roles).__name__}."
        )
    if len(candidates) == 0:
        raise InvalidDesignError("roles must contain at least one role.")

    normalized: list[str] = []
    for role in candidates:
        if not isinstance(role, str) or role not in VALID_ROLES:
            raise InvalidDesignError(
                f"Unknown covariate role {role!r}. Valid roles: "
                f"{list(VALID_ROLES)}."
            )
        if role not in normalized:
            normalized.append(role)
    return tuple(normalized)


def parse_window(
    window: str | pd.Timedelta | datetime.timedelta | None,
) -> pd.Timedelta | None:
    """Parse a declared history window into a positive ``pd.Timedelta``.

    Parameters
    ----------
    window : str, pd.Timedelta, datetime.timedelta or None
        A pandas offset string (``"7D"``, ``"24h"``, ``"30min"``), a
        ``Timedelta``, or ``None`` for "all history up to the cutoff".

    Returns
    -------
    pd.Timedelta or None

    Raises
    ------
    InvalidDesignError
        If the window is not parseable or is not strictly positive.
    """
    if window is None:
        return None

    if isinstance(window, (pd.Timedelta, datetime.timedelta)):
        parsed = pd.Timedelta(window)
    elif isinstance(window, str):
        try:
            parsed = pd.Timedelta(window)
        except (ValueError, TypeError) as exc:
            raise InvalidDesignError(
                f"window must be a pandas offset string such as '7D', "
                f"'24h' or '30min'; received {window!r} ({exc})."
            ) from None
    else:
        raise InvalidDesignError(
            f"window must be a string, pandas.Timedelta, "
            f"datetime.timedelta or None; received {type(window).__name__}."
        )

    if parsed <= pd.Timedelta(0):
        raise InvalidDesignError(
            f"window must be strictly positive; received {window!r} "
            f"({parsed})."
        )
    return parsed


class CovariateSpec:
    """One point-in-time covariate built from a history table.

    Parameters
    ----------
    name : str
        Output column name in the frozen covariate table. Must be unique
        within a builder and must not collide with a structural column.
    column : str or None, optional
        Source column in the history table. Required for every
        aggregation except ``"count"`` (where it is ignored: the count is
        of history rows, not of non-null values). Defaults to ``name``.
    agg : str or None, optional
        How eligible history rows are reduced to one value per unit.
        Required for the ``"aggregate"`` strategy and must be ``None``
        for ``"last_visible"``, where the selected row supplies every
        column. Default None.
    window : str, pd.Timedelta or None, optional
        Per-column lookback window, overriding the builder-level window.
        Only valid for the ``"aggregate"`` strategy: with
        ``"last_visible"`` a single window decides which row is visible,
        so per-column windows would select different rows. Default None.
    role : str or sequence of str, optional
        Role(s) the covariate plays. ``"balance"`` makes it eligible for
        a design's balance criterion (e.g., rerandomization),
        ``"adjustment"`` for Lin/CUPED, ``"blocking"`` for blocking
        columns, ``"descriptive"`` for reporting only. Default
        ``"adjustment"``.

    Notes
    -----
    Windows are measured relative to each unit's assignment cutoff: a
    history row is inside the window when
    ``cutoff - window <= event_time <= cutoff``. Visibility additionally
    requires ``arrival_time <= cutoff``; a row whose event happened
    before the cutoff but arrived after it was not knowable at
    assignment time and is excluded (see ``PointInTimeCovariates``).
    """

    def __init__(
        self,
        name: str,
        column: str | None = None,
        agg: str | None = None,
        window: str | pd.Timedelta | datetime.timedelta | None = None,
        role: str | tuple[str, ...] | list[str] = "adjustment",
    ) -> None:
        if not isinstance(name, str) or not name:
            raise InvalidDesignError(
                f"CovariateSpec.name must be a non-empty string; "
                f"received {name!r}."
            )
        if column is not None and (
            not isinstance(column, str) or not column
        ):
            raise InvalidDesignError(
                f"CovariateSpec.column must be a non-empty string or None; "
                f"received {column!r}."
            )
        if agg is not None and agg not in AGGREGATIONS:
            raise InvalidDesignError(
                f"CovariateSpec.agg must be one of {list(AGGREGATIONS)} or "
                f"None; received {agg!r}."
            )

        if agg == "count":
            # Counting history rows does not read a value column.
            column = None
        elif column is None:
            # Default source column: same name as the output covariate.
            column = name

        self.name: str = name
        self.column: str | None = column
        self.agg: str | None = agg
        self.window: pd.Timedelta | None = parse_window(window)
        self.roles: tuple[str, ...] = normalize_roles(role)

    @property
    def requires_numeric(self) -> bool:
        """Whether the source column must be numeric."""
        return self.agg != "count"

    def to_dict(self) -> dict:
        """Return a JSON-friendly description of the specification."""
        return {
            "name": self.name,
            "column": self.column,
            "agg": self.agg,
            "window": None if self.window is None else str(self.window),
            "roles": list(self.roles),
        }

    def __repr__(self) -> str:
        parts = [f"name={self.name!r}"]
        if self.column is not None:
            parts.append(f"column={self.column!r}")
        if self.agg is not None:
            parts.append(f"agg={self.agg!r}")
        if self.window is not None:
            parts.append(f"window={str(self.window)!r}")
        parts.append(f"role={list(self.roles)!r}")
        return f"CovariateSpec({', '.join(parts)})"


def normalize_specs(
    specs: list[CovariateSpec] | tuple[CovariateSpec, ...],
    *,
    strategy: str,
    unit_col: str,
    event_time_col: str,
    arrival_time_col: str | None,
) -> list[CovariateSpec]:
    """Validate specs as a coherent group for a builder.

    Parameters
    ----------
    specs : sequence of CovariateSpec
        Specifications to validate.
    strategy : {"aggregate", "last_visible"}
        Builder strategy.
    unit_col, event_time_col, arrival_time_col : str or None
        Builder columns, used to reject output names that would shadow
        structural columns.

    Returns
    -------
    list of CovariateSpec
        Defensive copies of the validated specs, in declaration order.

    Raises
    ------
    InvalidDesignError
        On an empty list, duplicate names, reserved names, or a strategy
        that conflicts with the per-spec options.
    """
    if strategy not in VALID_STRATEGIES:
        raise InvalidDesignError(
            f"strategy must be one of {VALID_STRATEGIES}; "
            f"received {strategy!r}."
        )
    if not isinstance(specs, (list, tuple)) or len(specs) == 0:
        raise InvalidDesignError(
            "specs must be a non-empty sequence of CovariateSpec objects."
        )

    reserved = {unit_col, event_time_col}
    if arrival_time_col is not None:
        reserved.add(arrival_time_col)

    names: list[str] = []
    normalized: list[CovariateSpec] = []
    for spec in specs:
        if not isinstance(spec, CovariateSpec):
            raise InvalidDesignError(
                f"specs must contain CovariateSpec objects; found "
                f"{type(spec).__name__}."
            )
        if spec.name in names:
            raise InvalidDesignError(
                f"Duplicate covariate name in specs: {spec.name!r}."
            )
        if spec.name in reserved:
            raise InvalidDesignError(
                f"Covariate name {spec.name!r} collides with a structural "
                f"column. Reserved: {sorted(reserved)}."
            )

        if strategy == "aggregate":
            if spec.agg is None:
                raise InvalidDesignError(
                    f"CovariateSpec {spec.name!r}: agg is required when "
                    f"strategy='aggregate'."
                )
        else:  # last_visible
            if spec.agg is not None:
                raise InvalidDesignError(
                    f"CovariateSpec {spec.name!r}: agg must be None when "
                    f"strategy='last_visible'; the value is read from the "
                    f"last row visible at the cutoff."
                )
            if spec.window is not None:
                raise InvalidDesignError(
                    f"CovariateSpec {spec.name!r}: per-column window is not "
                    f"allowed when strategy='last_visible'; use the "
                    f"builder-level window."
                )

        names.append(spec.name)
        normalized.append(
            CovariateSpec(
                name=spec.name,
                column=spec.column,
                agg=spec.agg,
                window=spec.window,
                role=spec.roles,
            )
        )
    return normalized
