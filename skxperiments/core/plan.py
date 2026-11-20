"""Shared randomization plans bound to an Assignment population and design version.

A ``RandomizationPlan`` is the single source of replicated assignments for
resampling-based inference that must be shared across consumers — most
importantly, joint inference over multiple outcome variables of the same
experiment. Caching a list of seeds and letting each metric call
``assignment.draw(seed)`` on its own is *not* equivalent: if the metrics
hold different Assignment objects (different rows, different columns),
each one draws from a different population and the simulated worlds are
not comparable.

The plan closes that hole by binding two things at construction and
re-verifying them on every draw:

- **Population**: a fingerprint of ``assignment.data_`` (all columns,
  including the observed treatment vector). Mutating the bound
  assignment's data invalidates the plan.
- **Design version**: a fingerprint of the generating design's
  parameters (``design.get_params()``, excluding ``seed`` — the plan
  supplies its own per-replicate seeds, so the design's seed is not part
  of the randomization mechanism). Changing any mechanism parameter
  (e.g., ``p``, ``n_treated``, ``block_col``, rerandomization threshold)
  invalidates the plan.

Each replicate is identified by a ``replicate_id`` in
``[0, n_replicates)`` and maps to one pre-generated seed; the same
``replicate_id`` always reproduces the same complete Assignment.
"""

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from skxperiments.core.assignment import (
    BaseAssignment,
    BlockedAssignment,
    CRDAssignment,
)
from skxperiments.core.exceptions import (
    DesignEstimatorMismatch,
    InvalidDesignError,
)


class RandomizationPlan:
    """Shared randomization plan over one Assignment population and design version.

    Pre-generates one seed per replicate and produces, on demand, the
    complete Assignment of replicate ``replicate_id`` by delegating to
    the bound assignment's ``draw()`` — so the randomization mechanism
    (CRD, blocked CRD, rerandomized CRD) is respected exactly as in
    ``RandomizationTest``.

    Parameters
    ----------
    assignment : CRDAssignment or BlockedAssignment
        The observed assignment defining the population (units and
        covariates) and, through ``assignment.design_``, the
        randomization mechanism. Must carry a reference to its
        generating design (``design_`` not None).
    n_replicates : int
        Number of replicates in the plan. Must be a positive integer.
    seed : int or None, optional
        Master seed. A single ``np.random.default_rng(seed)`` generates
        one seed per replicate, so the same ``seed`` reproduces the same
        plan. By default None (non-deterministic).

    Attributes
    ----------
    replicate_seeds_ : np.ndarray
        One pre-generated seed per replicate, shape
        ``(n_replicates,)``. ``replicate_id`` indexes this array.
    population_fingerprint_ : str
        SHA-256 fingerprint of the bound population
        (``assignment.data_``) captured at construction.
    design_version_ : dict
        Copy of the bound design parameters (excluding ``seed``).
    design_fingerprint_ : str
        Fingerprint of ``design_version_`` captured at construction.

    Notes
    -----
    ``FactorialAssignment`` is rejected (via ``DesignEstimatorMismatch``),
    matching ``RandomizationTest`` v1.

    Every call to ``assignment()`` re-verifies both fingerprints and
    raises ``InvalidDesignError`` if the bound assignment's data or the
    design's parameters changed since construction. The plan therefore
    never silently draws from a population or mechanism different from
    the one it was bound to.

    Examples
    --------
    >>> from skxperiments.design.crd import CRD
    >>> from skxperiments.core.plan import RandomizationPlan
    >>> design = CRD(p=0.5, seed=42)
    >>> assignment = design.randomize(df)  # doctest: +SKIP
    >>> plan = RandomizationPlan(  # doctest: +SKIP
    ...     assignment, n_replicates=1000, seed=0
    ... )  # doctest: +SKIP
    >>> replicate_7 = plan.assignment(7)  # doctest: +SKIP
    """

    def __init__(
        self,
        assignment: BaseAssignment,
        n_replicates: int,
        seed: int | None = None,
    ) -> None:
        if not isinstance(assignment, (CRDAssignment, BlockedAssignment)):
            raise DesignEstimatorMismatch(
                estimator_name=type(self).__name__,
                received_type=type(assignment).__name__,
                expected_type="CRDAssignment or BlockedAssignment",
            )

        if assignment.design_ is None:
            raise InvalidDesignError(
                "RandomizationPlan requires an assignment that carries a "
                "reference to its generating design (design_ is None). "
                "Build the assignment via Design.randomize(df)."
            )

        if not isinstance(n_replicates, (int, np.integer)) or isinstance(
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

        self.n_replicates = int(n_replicates)
        self.seed = seed
        self._assignment = assignment

        rng = np.random.default_rng(seed)
        self.replicate_seeds_: np.ndarray = rng.integers(
            0, 2**32, size=self.n_replicates
        )

        self.population_fingerprint_: str = self._fingerprint_population(
            assignment
        )
        self.design_version_: dict = self._design_version(assignment.design_)
        self.design_fingerprint_: str = self._fingerprint_design(
            self.design_version_
        )

    def assignment(self, replicate_id: int) -> BaseAssignment:
        """Return the complete Assignment of one replicate.

        Parameters
        ----------
        replicate_id : int
            Replicate index in ``[0, n_replicates)``. The same
            ``replicate_id`` always yields the same assignment.

        Returns
        -------
        BaseAssignment
            Freshly drawn Assignment of the same concrete type as the
            bound assignment, sharing its data.

        Raises
        ------
        InvalidDesignError
            If ``replicate_id`` is not an integer in range, or if the
            binding no longer holds (population or design version
            changed since the plan was created).
        """
        if not isinstance(replicate_id, (int, np.integer)) or isinstance(
            replicate_id, bool
        ):
            raise InvalidDesignError(
                f"replicate_id must be an integer, got "
                f"{type(replicate_id).__name__}."
            )
        if not 0 <= replicate_id < self.n_replicates:
            raise InvalidDesignError(
                f"replicate_id must be in [0, {self.n_replicates}), got "
                f"{replicate_id}."
            )

        self.validate()
        return self._assignment.draw(
            seed=int(self.replicate_seeds_[replicate_id])
        )

    def validate(self) -> None:
        """Re-verify the population and design-version bindings.

        Raises
        ------
        InvalidDesignError
            If the bound assignment's ``data_`` changed (population
            mismatch) or the generating design's parameters changed
            (design-version mismatch) since the plan was created.
        """
        current_population = self._fingerprint_population(self._assignment)
        if current_population != self.population_fingerprint_:
            raise InvalidDesignError(
                "RandomizationPlan is bound to a specific Assignment "
                "population, but the bound assignment's data_ has changed "
                "since the plan was created. Create a new RandomizationPlan "
                "for the modified assignment instead of mutating it."
            )

        current_design = self._fingerprint_design(
            self._design_version(self._assignment.design_)
        )
        if current_design != self.design_fingerprint_:
            raise InvalidDesignError(
                "RandomizationPlan is bound to design version "
                f"{self.design_version_}, but the design now has parameters "
                f"{self._design_version(self._assignment.design_)}. The plan "
                "cannot draw under a different design version; create a new "
                "RandomizationPlan."
            )

    def __len__(self) -> int:
        """Number of replicates in the plan."""
        return self.n_replicates

    @staticmethod
    def _fingerprint_population(assignment: BaseAssignment) -> str:
        """Fingerprint the population carried by an assignment.

        Hashes the full ``data_`` (all columns, including the observed
        treatment vector) together with shape, column names, and dtypes,
        so any in-place mutation of the bound assignment is detected.
        """
        df = assignment.data_
        hashed = pd.util.hash_pandas_object(df, index=True).values
        payload = (
            repr(
                (
                    df.shape,
                    list(df.columns),
                    [str(dtype) for dtype in df.dtypes],
                )
            ).encode()
            + hashed.tobytes()
        )
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _design_version(design: Any) -> dict:
        """Extract the design version: mechanism parameters, excluding seed.

        The design's own ``seed`` is excluded because the plan supplies
        per-replicate seeds itself; the seed does not alter the
        randomization mechanism the plan is bound to.
        """
        version = dict(design.get_params())
        version.pop("seed", None)
        return version

    @staticmethod
    def _fingerprint_design(design_version: dict) -> str:
        """Fingerprint a design-version dict via a sorted repr."""
        return repr(sorted(design_version.items(), key=lambda kv: kv[0]))

    def __repr__(self) -> str:
        """Return string representation."""
        design_name = type(self._assignment.design_).__name__
        return (
            f"RandomizationPlan(n_replicates={self.n_replicates}, "
            f"design={design_name}, seed={self.seed})"
        )
