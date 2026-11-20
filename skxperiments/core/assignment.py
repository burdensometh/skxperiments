"""Assignment classes representing the contract between designs and estimators.

An Assignment object is always created by a Design via randomize() — never
instantiated directly by the user. It carries the treatment assignment
alongside a copy of the original data, ensuring no side effects.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from skxperiments.core.exceptions import InvalidDesignError


class BaseAssignment(ABC):
    """Abstract base class for all assignment objects.

    An Assignment is the contract between a design and an estimator.
    Estimators receive Assignment objects, not loose DataFrames.

    Notes
    -----
    Assignment objects are intended to be immutable after construction.
    Designs are responsible for passing a defensive copy of the input
    DataFrame (i.e., ``df.copy()``) when constructing an Assignment.
    The outcome variable is **not** part of the Assignment contract:
    estimators receive the outcome column name as a parameter at __init__
    and resolve it against ``assignment.data_`` at fit time.

    Parameters
    ----------
    data : pd.DataFrame
        Copy of the original DataFrame with treatment column added.
    treatment_col : str
        Name of the treatment column.
    design : Any
        Reference to the design that generated this assignment.

    Attributes
    ----------
    data_ : pd.DataFrame
        DataFrame with treatment column added.
    treatment_col_ : str
        Name of the treatment column.
    design_ : Any
        Reference to the generating design.
    n_units_ : int
        Total number of units.
    n_treated_ : int
        Number of units assigned to treatment.
    n_control_ : int
        Number of units assigned to control.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        design: Any,
    ) -> None:
        if treatment_col not in data.columns:
            raise InvalidDesignError(
                f"Treatment column '{treatment_col}' not found in DataFrame. "
                f"Available columns: {list(data.columns)}."
            )
    
        self.data_: pd.DataFrame = data
        self.treatment_col_: str = treatment_col
        self.design_: Any = design
        self.n_units_: int = len(data)
        self.n_treated_: int = int((data[treatment_col] == 1).sum())
        self.n_control_: int = int((data[treatment_col] == 0).sum())

    def _validate_treatment_col(self) -> None:
        """Validate that the treatment column contains only 0 and 1.

        Raises
        ------
        InvalidDesignError
            If the treatment column contains values other than 0 and 1.
        """
        unique_values = set(self.data_[self.treatment_col_].unique())
        allowed = {0, 1}
        if not unique_values.issubset(allowed):
            invalid = unique_values - allowed
            raise InvalidDesignError(
                f"Treatment column '{self.treatment_col_}' must contain only "
                f"0 and 1, but found values: {sorted(invalid)}."
            )

    @abstractmethod
    def treated_ids(self) -> np.ndarray:
        """Return iloc positions of treated units.

        Returns
        -------
        np.ndarray
            Array of integer positions where treatment == 1.
        """

    @abstractmethod
    def control_ids(self) -> np.ndarray:
        """Return iloc positions of control units.

        Returns
        -------
        np.ndarray
            Array of integer positions where treatment == 0.
        """

    @abstractmethod
    def draw(self, seed: int | None = None) -> "BaseAssignment":
        """Generate a new realization of the assignment under the same mechanism.

        This method enables randomization-based inference: each call returns
        a fresh Assignment object whose treatment vector is a new draw from
        the same randomization mechanism that produced this Assignment
        (same N, same probabilities, same constraints), but with a different
        random seed.

        The returned Assignment must be of the same concrete type as self
        and reference the same underlying data. Implementations must not
        mutate self.

        Parameters
        ----------
        seed : int or None, optional
            Random seed for the new draw. If None, a non-deterministic
            draw is performed.

        Returns
        -------
        BaseAssignment
            New Assignment of the same concrete type, with a freshly drawn
            treatment vector.

        Notes
        -----
        Consumed by RandomizationTest to generate the null distribution.
        Not intended for direct use by end users.
        """

    def __repr__(self) -> str:
        """Return string representation.

        Returns
        -------
        str
            Format: ClassName(n_treated=N, n_control=N)
        """
        class_name = type(self).__name__
        return f"{class_name}(n_treated={self.n_treated_}, n_control={self.n_control_})"


class CRDAssignment(BaseAssignment):
    """Assignment resulting from a Completely Randomized Design.

    Parameters
    ----------
    data : pd.DataFrame
        Copy of the original DataFrame with treatment column added.
    treatment_col : str
        Name of the treatment column.
    design : Any
        Reference to the design that generated this assignment.
    seed : int or None, optional
        Random seed used for the randomization, by default None.
    rerandomization_metadata : dict or None, optional
        When the assignment was produced by a rerandomization design
        (e.g., ReRandomizedCRD), this dict carries the information
        needed to replay the acceptance/rejection mechanism in
        ``draw()``. Expected keys:

        - ``"covariates"``: list[str]
        - ``"threshold"``: float
        - ``"cov_matrix"``: np.ndarray (sample covariance, ddof=1, of
          the full DataFrame)
        - ``"attempts"``: int
        - ``"scaling_factor"``: float (1/n_treated + 1/n_control)

        When None, this is a plain CRD assignment.

    Attributes
    ----------
    seed_ : int or None
        Seed used in the randomization.
    rerandomization_metadata : dict or None
        Rerandomization metadata, or None if this is a plain CRD
        assignment.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [1, 2, 3, 4], "treatment": [1, 0, 1, 0]})
    >>> assignment = CRDAssignment(
    ...     data=df, treatment_col="treatment", design=None, seed=42
    ... )
    >>> assignment.n_treated_
    2
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        design: Any,
        seed: int | None = None,
        rerandomization_metadata: dict | None = None,
    ) -> None:
        super().__init__(data=data, treatment_col=treatment_col, design=design)
        self.seed_: int | None = seed
        self.rerandomization_metadata: dict | None = rerandomization_metadata
        self._validate_treatment_col()

    def treated_ids(self) -> np.ndarray:
        """Return iloc positions of treated units."""
        return np.where(self.data_[self.treatment_col_].values == 1)[0]

    def control_ids(self) -> np.ndarray:
        """Return iloc positions of control units."""
        return np.where(self.data_[self.treatment_col_].values == 0)[0]

    def draw(self, seed: int | None = None) -> "CRDAssignment":
        """Generate a new realization under the same mechanism.

        For plain CRD, delegates to ``design_.randomize(df_clean)``.
        For rerandomized CRD, delegates to
        ``design_._randomize_with_cached_cov`` so the covariance matrix
        is reused without recomputation.

        Parameters
        ----------
        seed : int or None, optional
            Random seed for the new draw, by default None.

        Returns
        -------
        CRDAssignment
            Fresh assignment under the same mechanism.

        Raises
        ------
        InvalidDesignError
            If ``design_`` is None.
        """
        if self.design_ is None:
            raise InvalidDesignError(
                "Cannot draw a new realization: this CRDAssignment was "
                "constructed without a reference to a generating design."
            )

        df_clean = self.data_.drop(columns=[self.treatment_col_])

        if self.rerandomization_metadata is not None:
            # Rerandomized CRD: reuse cached covariance matrix.
            cov_matrix = self.rerandomization_metadata["cov_matrix"]
            rng = np.random.default_rng(seed)
            return self.design_._randomize_with_cached_cov(
                df=df_clean,
                cov_matrix=cov_matrix,
                rng=rng,
            )

        # Plain CRD: delegate to design.randomize via temporary seed override.
        original_seed = getattr(self.design_, "seed", None)
        try:
            self.design_.seed = seed
            return self.design_.randomize(df_clean)
        finally:
            self.design_.seed = original_seed


class BlockedAssignment(BaseAssignment):
    """Assignment resulting from a Blocked Completely Randomized Design.

    Treatment is randomized independently within each block, preserving
    the treatment proportion within every block.

    Parameters
    ----------
    data : pd.DataFrame
        Copy of the original DataFrame with treatment column added.
    treatment_col : str
        Name of the treatment column.
    design : Any
        Reference to the BlockedCRD design that generated this assignment.
    block_col : str
        Name of the column identifying blocks.
    block_sizes : dict
        Mapping from block label to number of units in that block.
    seed : int or None, optional
        Random seed used for the randomization, by default None.

    Attributes
    ----------
    block_col_ : str
        Name of the block column.
    block_sizes_ : dict
        Mapping from block label to block size.
    seed_ : int or None
        Seed used in the randomization.

    Notes
    -----
    The block column must exist in ``data``. The treatment proportion
    is preserved within each block; see BlockedCRD for the design that
    produces this Assignment.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        design: Any,
        block_col: str,
        block_sizes: dict,
        seed: int | None = None,
    ) -> None:
        super().__init__(data=data, treatment_col=treatment_col, design=design)
        if block_col not in data.columns:
            raise InvalidDesignError(
                f"Block column '{block_col}' not found in DataFrame. "
                f"Available columns: {list(data.columns)}."
            )
        self.block_col_: str = block_col
        self.block_sizes_: dict = dict(block_sizes)
        self.seed_: int | None = seed
        self._validate_treatment_col()

    def treated_ids(self) -> np.ndarray:
        """Return iloc positions of treated units."""
        return np.where(self.data_[self.treatment_col_].values == 1)[0]

    def control_ids(self) -> np.ndarray:
        """Return iloc positions of control units."""
        return np.where(self.data_[self.treatment_col_].values == 0)[0]

    def draw(self, seed: int | None = None) -> "BlockedAssignment":
        """Generate a new realization under the same blocked mechanism.

        Delegates to the generating design, ensuring the same block
        structure and within-block proportion are respected.

        Parameters
        ----------
        seed : int or None, optional
            Random seed for the new draw.

        Returns
        -------
        BlockedAssignment
            New BlockedAssignment with a freshly drawn treatment vector.

        Raises
        ------
        InvalidDesignError
            If this Assignment was constructed without a reference to
            a generating design.
        """
        if self.design_ is None:
            raise InvalidDesignError(
                "Cannot draw a new realization: this BlockedAssignment "
                "was constructed without a reference to a generating "
                "design."
            )

        df_clean = self.data_.drop(columns=[self.treatment_col_])

        original_seed = getattr(self.design_, "seed", None)
        try:
            self.design_.seed = seed
            return self.design_.randomize(df_clean)
        finally:
            self.design_.seed = original_seed


class ClusterAssignment(BaseAssignment):
    """Assignment resulting from a cluster-randomized design.

    Treatment is randomized over whole clusters (e.g., stores), while
    outcomes are recorded at the individual level (e.g., customers).
    Every individual inherits the treatment label of its cluster, so
    the number of independent randomization units is the number of
    clusters — **not** the number of rows.

    The individual-level data are preserved verbatim (one row per
    individual, original columns intact): within-cluster unit identity
    is retained so that cluster-level aggregations can be recomputed
    and so that ``draw()`` can re-propagate a fresh cluster-level
    randomization onto the same individuals.

    Parameters
    ----------
    data : pd.DataFrame
        Copy of the original (individual-level) DataFrame with the
        treatment column added. Treatment must be constant within
        each cluster.
    treatment_col : str
        Name of the treatment column.
    design : Any
        Reference to the ClusterDesign that generated this assignment.
    cluster_col : str
        Name of the column identifying clusters.
    cluster_sizes : dict
        Mapping from cluster label to number of individuals in that
        cluster.
    seed : int or None, optional
        Random seed used for the randomization, by default None.

    Attributes
    ----------
    cluster_col_ : str
        Name of the cluster column.
    cluster_sizes_ : dict
        Mapping from cluster label to cluster size (individuals).
    n_clusters_ : int
        Number of unique clusters (the number of independent
        randomization units).
    n_clusters_treated_ : int
        Number of clusters assigned to treatment.
    n_clusters_control_ : int
        Number of clusters assigned to control.
    seed_ : int or None
        Seed used in the randomization.

    Notes
    -----
    ``n_units_``, ``n_treated_``, and ``n_control_`` (inherited from
    ``BaseAssignment``) count **individuals**, matching the rows of
    ``data_``. The cluster-level counts are exposed separately as
    ``n_clusters_``, ``n_clusters_treated_``, and
    ``n_clusters_control_``; neither is silently substituted for the
    other.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        design: Any,
        cluster_col: str,
        cluster_sizes: dict,
        seed: int | None = None,
    ) -> None:
        super().__init__(data=data, treatment_col=treatment_col, design=design)
        if cluster_col not in data.columns:
            raise InvalidDesignError(
                f"Cluster column '{cluster_col}' not found in DataFrame. "
                f"Available columns: {list(data.columns)}."
            )
        if cluster_col == treatment_col:
            raise InvalidDesignError(
                f"Cluster column and treatment column must differ; "
                f"both are '{cluster_col}'."
            )

        data_labels = set(data[cluster_col].unique())
        if set(cluster_sizes) != data_labels:
            raise InvalidDesignError(
                "cluster_sizes keys do not match the cluster labels "
                f"present in the data. cluster_sizes keys: "
                f"{sorted(set(cluster_sizes), key=str)}; data labels: "
                f"{sorted(data_labels, key=str)}."
            )

        self.cluster_col_: str = cluster_col
        self.cluster_sizes_: dict = dict(cluster_sizes)
        self.n_clusters_: int = len(cluster_sizes)
        self.seed_: int | None = seed
        self._validate_treatment_col()
        self._validate_cluster_integrity()

        cluster_treatment = self._cluster_treatment()
        self.n_clusters_treated_: int = int((cluster_treatment == 1).sum())
        self.n_clusters_control_: int = int((cluster_treatment == 0).sum())

    def _cluster_treatment(self) -> "pd.Series":
        """Return the treatment label of each cluster (indexed by label)."""
        return self.data_.groupby(self.cluster_col_)[self.treatment_col_].first()

    def _validate_cluster_integrity(self) -> None:
        """Validate that treatment is constant within each cluster.

        Raises
        ------
        InvalidDesignError
            If any cluster contains both treated and control
            individuals — under cluster randomization all individuals
            of a cluster must share the cluster's label.
        """
        n_unique = self.data_.groupby(self.cluster_col_)[
            self.treatment_col_
        ].nunique()
        if (n_unique > 1).any():
            offending = n_unique[n_unique > 1].index.tolist()
            raise InvalidDesignError(
                f"Treatment must be constant within each cluster, but "
                f"cluster(s) {offending} contain both treated and "
                f"control individuals. Cluster-randomized labels must "
                f"be assigned to whole clusters."
            )

    def treated_ids(self) -> np.ndarray:
        """Return iloc positions of treated individuals."""
        return np.where(self.data_[self.treatment_col_].values == 1)[0]

    def control_ids(self) -> np.ndarray:
        """Return iloc positions of control individuals."""
        return np.where(self.data_[self.treatment_col_].values == 0)[0]

    def treated_cluster_labels(self) -> np.ndarray:
        """Return labels of clusters assigned to treatment."""
        cluster_treatment = self._cluster_treatment()
        return cluster_treatment[cluster_treatment == 1].index.to_numpy()

    def control_cluster_labels(self) -> np.ndarray:
        """Return labels of clusters assigned to control."""
        cluster_treatment = self._cluster_treatment()
        return cluster_treatment[cluster_treatment == 0].index.to_numpy()

    def draw(self, seed: int | None = None) -> "ClusterAssignment":
        """Generate a new realization under the same cluster mechanism.

        Re-randomizes **whole clusters** on the unique-cluster
        population (via the generating ClusterDesign and its base
        design) and re-propagates the labels to the same individuals.
        Individual rows are never relabeled independently: all members
        of a cluster switch arm together.

        Parameters
        ----------
        seed : int or None, optional
            Random seed for the new draw, by default None.

        Returns
        -------
        ClusterAssignment
            New ClusterAssignment over the same individuals and
            clusters, with a freshly drawn cluster-level treatment
            vector.

        Raises
        ------
        InvalidDesignError
            If this Assignment was constructed without a reference to
            a generating design.
        """
        if self.design_ is None:
            raise InvalidDesignError(
                "Cannot draw a new realization: this ClusterAssignment "
                "was constructed without a reference to a generating "
                "design."
            )

        df_clean = self.data_.drop(columns=[self.treatment_col_])

        # The seed lives on the wrapped base design (single source of
        # truth); override it temporarily, mirroring the pattern used
        # by CRDAssignment.draw.
        base_design = self.design_.base_design
        original_seed = getattr(base_design, "seed", None)
        try:
            base_design.seed = seed
            return self.design_.randomize(df_clean)
        finally:
            base_design.seed = original_seed


class FactorialAssignment(BaseAssignment):
    """Assignment resulting from a 2^K Factorial Design.

    Each unit is assigned to one of 2^K cells defined by the values of
    K binary factors. The synthetic column ``"_cell"`` carries an
    integer in [0, 2^K - 1] encoding the cell of each unit.

    Cell encoding convention (little-endian):

        cell_index = sum(factor_value * 2**i
                         for i, factor_value in enumerate(factor_cols))

    For K=2 with factors ``["A", "B"]``:
        A=0, B=0 -> cell 0
        A=1, B=0 -> cell 1
        A=0, B=1 -> cell 2
        A=1, B=1 -> cell 3

    This convention is part of the contract: FactorialEstimator
    (Phase 3) relies on it to reconstruct factor effects from cell
    indices.

    Notes
    -----
    Unlike CRDAssignment and BlockedAssignment, FactorialAssignment
    does **not** call ``self._validate_treatment_col()`` in __init__.
    The synthetic ``"_cell"`` column carries values in
    [0, 2^K - 1], not a binary 0/1 treatment indicator. Validating it
    against {0, 1} would always fail for K >= 1.

    Consequently, ``treated_ids()`` and ``control_ids()`` are not
    meaningful for factorial designs and raise NotImplementedError.
    Use ``cell_ids(**factor_values)`` to select units by cell.

    Parameters
    ----------
    data : pd.DataFrame
        Copy of the original DataFrame with factor columns and
        ``"_cell"`` added.
    design : Any
        Reference to the FactorialDesign that generated this assignment.
    factor_cols : list of str
        Names of the K factors, in the order used to compute cell indices.
    cell_sizes : dict
        Mapping from cell index to number of units in that cell.
    seed : int or None, optional
        Random seed used in the randomization, by default None.

    Attributes
    ----------
    factor_cols : list of str
        Names of the K factors.
    n_cells_ : int
        Number of cells (2^K).
    cell_sizes_ : dict
        Mapping from cell index to cell size.
    seed_ : int or None
        Seed used in the randomization.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        design: Any,
        factor_cols: list[str],
        cell_sizes: dict,
        seed: int | None = None,
    ) -> None:
        # Pass "_cell" as treatment_col to satisfy BaseAssignment's
        # interface, but do NOT validate it against {0, 1}: see
        # class docstring.
        super().__init__(data=data, treatment_col="_cell", design=design)
        # Override the binary-coded n_treated_ / n_control_ set by
        # BaseAssignment; they are not meaningful here.
        self.n_treated_ = 0
        self.n_control_ = 0

        self.factor_cols: list[str] = list(factor_cols)
        self.n_cells_: int = 2 ** len(factor_cols)
        self.cell_sizes_: dict = dict(cell_sizes)
        self.seed_: int | None = seed

    def treated_ids(self) -> np.ndarray:
        """Not applicable to factorial designs.

        Raises
        ------
        NotImplementedError
            Always. Use ``cell_ids(**factor_values)`` instead.
        """
        # TODO Fase 3: revisar contrato de treated_ids/control_ids
        # quando FactorialEstimator for implementado — atual
        # NotImplementedError é solução de transição.
        raise NotImplementedError(
            "FactorialAssignment não tem tratamento binário único — "
            "use cell_ids(**factor_values) para selecionar unidades "
            "por célula"
        )

    def control_ids(self) -> np.ndarray:
        """Not applicable to factorial designs.

        Raises
        ------
        NotImplementedError
            Always. Use ``cell_ids(**factor_values)`` instead.
        """
        # TODO Fase 3: revisar contrato de treated_ids/control_ids
        # quando FactorialEstimator for implementado — atual
        # NotImplementedError é solução de transição.
        raise NotImplementedError(
            "FactorialAssignment não tem tratamento binário único — "
            "use cell_ids(**factor_values) para selecionar unidades "
            "por célula"
        )

    def cell_ids(self, **factor_values: int) -> np.ndarray:
        """Return iloc positions of units in the given cell.

        Parameters
        ----------
        **factor_values : int
            Mapping of factor name to its value (0 or 1). All factors
            in ``factor_cols`` must be specified.

        Returns
        -------
        np.ndarray
            Array of iloc positions of units matching all specified
            factor values.

        Raises
        ------
        InvalidDesignError
            If a kwarg is not a known factor name, if a value is not
            0 or 1, or if not all factors are specified.

        Examples
        --------
        >>> # For a 2x2 design with factors ["A", "B"]:
        >>> # assignment.cell_ids(A=1, B=0) -> iloc positions of cell 1
        """
        # Validate factor names
        unknown = [k for k in factor_values if k not in self.factor_cols]
        if unknown:
            raise InvalidDesignError(
                f"Unknown factor(s) in cell_ids: {unknown}. "
                f"Known factors: {self.factor_cols}."
            )

        # Validate all factors specified
        missing = [k for k in self.factor_cols if k not in factor_values]
        if missing:
            raise InvalidDesignError(
                f"cell_ids requires values for all factors. "
                f"Missing: {missing}. Provided: {list(factor_values.keys())}."
            )

        # Validate values
        for name, value in factor_values.items():
            if value not in (0, 1):
                raise InvalidDesignError(
                    f"Factor '{name}' value must be 0 or 1, "
                    f"received {value!r}."
                )

        # Build mask: all factors must match.
        mask = np.ones(self.n_units_, dtype=bool)
        for name in self.factor_cols:
            mask &= self.data_[name].values == factor_values[name]

        return np.where(mask)[0]

    def draw(self, seed: int | None = None) -> "FactorialAssignment":
        """Generate a new realization preserving cell sizes.

        Drops factor columns and ``"_cell"`` from the data before
        delegating to ``design_.randomize`` to avoid name-collision
        errors in randomize().

        Parameters
        ----------
        seed : int or None, optional
            Random seed for the new draw, by default None.

        Returns
        -------
        FactorialAssignment
            New assignment with freshly drawn cell allocation.

        Raises
        ------
        InvalidDesignError
            If ``design_`` is None.
        """
        if self.design_ is None:
            raise InvalidDesignError(
                "Cannot draw a new realization: this FactorialAssignment "
                "was constructed without a reference to a generating "
                "design."
            )

        cols_to_drop = ["_cell"] + list(self.factor_cols)
        df_clean = self.data_.drop(columns=cols_to_drop)

        original_seed = getattr(self.design_, "seed", None)
        try:
            self.design_.seed = seed
            return self.design_.randomize(df_clean)
        finally:
            self.design_.seed = original_seed