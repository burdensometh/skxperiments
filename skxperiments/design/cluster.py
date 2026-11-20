"""Cluster-randomized design adapter.

Randomizes whole clusters (e.g., stores) instead of individuals (e.g.,
customers) when outcomes are recorded per individual but the
randomization unit is the cluster. The adapter reuses an existing
unit-level design (``CRD`` or ``BlockedCRD``) on the unique-cluster
population and then propagates the cluster-level labels back to the
individual rows via the cluster id column.

Treating individuals as independent randomization units in this
setting overstates the effective sample size; the cluster — not the
individual — is the unit of randomization, and all downstream
estimation and inference must respect that.
"""

import pandas as pd

from skxperiments.core.assignment import ClusterAssignment
from skxperiments.core.base import BaseDesign
from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.design.blocked_crd import BlockedCRD
from skxperiments.design.crd import CRD


class ClusterDesign(BaseDesign):
    """Cluster-randomized design wrapping a unit-level base design.

    ``randomize(df)`` proceeds in two steps:

    1. **Cluster-level randomization.** The unique-cluster population
       is extracted from ``df`` (one row per cluster) and the wrapped
       ``base_design`` (``CRD`` or ``BlockedCRD``) randomizes over
       clusters. For a blocked base design, each cluster must belong
       to exactly one block.
    2. **Label propagation.** Every individual row inherits the
       treatment label of its cluster. The returned
       ``ClusterAssignment`` keeps the individual-level data intact
       (within-cluster unit identity is preserved).

    Parameters
    ----------
    cluster_col : str
        Name of the column in the DataFrame that identifies clusters
        (e.g., ``"store_id"``).
    base_design : CRD or BlockedCRD
        The unit-level design applied to the unique-cluster
        population. Its ``treatment_col`` names the treatment column
        of the resulting assignment, and its ``seed`` governs
        reproducibility (``ClusterDesign`` holds no seed of its own).

    Notes
    -----
    ``ReRandomizedCRD`` and ``FactorialDesign`` are not supported as
    base designs in v1 and are rejected at construction.

    Examples
    --------
    >>> import pandas as pd
    >>> from skxperiments.design.cluster import ClusterDesign
    >>> from skxperiments.design.crd import CRD
    >>> df = pd.DataFrame({
    ...     "store_id": [1, 1, 2, 2, 3, 3, 4, 4],
    ...     "customer_id": range(8),
    ... })
    >>> design = ClusterDesign(cluster_col="store_id",
    ...                        base_design=CRD(p=0.5, seed=42))
    >>> assignment = design.randomize(df)
    >>> assignment.n_clusters_
    4
    """

    def __init__(self, cluster_col: str, base_design: BaseDesign) -> None:
        if not isinstance(base_design, (CRD, BlockedCRD)):
            raise InvalidDesignError(
                "ClusterDesign v1 supports only CRD or BlockedCRD as "
                f"base_design; received {type(base_design).__name__}. "
                "Rerandomized and factorial base designs are not "
                "supported yet."
            )
        self.cluster_col = cluster_col
        self.base_design = base_design

    def randomize(self, df: pd.DataFrame) -> ClusterAssignment:
        """Randomize whole clusters and propagate labels to individuals.

        Parameters
        ----------
        df : pd.DataFrame
            Individual-level DataFrame. Must contain ``cluster_col``
            (and the base design's ``block_col``, when blocked) and
            must not contain the treatment column.

        Returns
        -------
        ClusterAssignment
            Assignment with the treatment column added at the
            individual level, constant within each cluster. The
            original DataFrame is not modified.

        Raises
        ------
        InvalidDesignError
            If ``cluster_col`` is missing or contains NaN, if the
            treatment column already exists, or if any cluster spans
            more than one block (blocked base design).
        InsufficientDataError
            Propagated from the base design when the unique-cluster
            population (or a block within it) is too small to
            randomize.
        """
        treatment_col = self.base_design.treatment_col

        if self.cluster_col not in df.columns:
            raise InvalidDesignError(
                f"Cluster column '{self.cluster_col}' not found in "
                f"DataFrame. Available columns: {list(df.columns)}."
            )
        if treatment_col in df.columns:
            raise InvalidDesignError(
                f"Treatment column '{treatment_col}' already exists "
                f"in DataFrame. Drop or rename it before calling "
                f"randomize()."
            )
        if df[self.cluster_col].isna().any():
            raise InvalidDesignError(
                f"Cluster column '{self.cluster_col}' contains NaN "
                f"values. Every individual must belong to a cluster."
            )

        cluster_df = self._build_cluster_frame(df)

        # Step 1: randomize over the unique-cluster population. The
        # base design owns the seed; ClusterDesign holds none.
        base_assignment = self.base_design.randomize(cluster_df)

        # Step 2: propagate cluster labels to individuals.
        label_map = dict(
            zip(
                base_assignment.data_[self.cluster_col],
                base_assignment.data_[treatment_col],
            )
        )
        df_out = df.copy()
        df_out[treatment_col] = df_out[self.cluster_col].map(label_map)

        cluster_sizes = (
            df_out[self.cluster_col].value_counts().to_dict()
        )

        return ClusterAssignment(
            data=df_out,
            treatment_col=treatment_col,
            design=self,
            cluster_col=self.cluster_col,
            cluster_sizes=cluster_sizes,
            seed=self.base_design.seed,
        )

    def _build_cluster_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract the unique-cluster population (one row per cluster).

        For a blocked base design, also carries the block label of
        each cluster, after validating that every cluster belongs to
        exactly one block.
        """
        if isinstance(self.base_design, BlockedCRD):
            block_col = self.base_design.block_col
            if block_col not in df.columns:
                raise InvalidDesignError(
                    f"Block column '{block_col}' (required by the "
                    f"blocked base design) not found in DataFrame. "
                    f"Available columns: {list(df.columns)}."
                )
            blocks_per_cluster = df.groupby(self.cluster_col)[
                block_col
            ].nunique()
            if (blocks_per_cluster > 1).any():
                offending = blocks_per_cluster[
                    blocks_per_cluster > 1
                ].index.tolist()
                raise InvalidDesignError(
                    f"Each cluster must belong to exactly one block, "
                    f"but cluster(s) {offending} span multiple "
                    f"values of '{block_col}'. Define blocks over "
                    f"clusters, not over individuals."
                )
            cols = [self.cluster_col, block_col]
        else:
            cols = [self.cluster_col]

        return df[cols].drop_duplicates().reset_index(drop=True)
