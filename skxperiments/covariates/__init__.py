"""Point-in-time construction of pre-assignment covariates.

Rerandomization and covariate adjustment require covariates that were
known *before* assignment. Joining a latest-snapshot table at analysis
time mixes post-assignment behavior into the design and the estimator.
This module builds a frozen covariate table keyed by unit id and
assignment cutoff, so the design feature cannot change when new data
arrives.
"""

from skxperiments.covariates.frame import (
    CovariateFrame,
    PointInTimeCovariates,
    resolve_covariates,
)
from skxperiments.covariates.spec import (
    AGGREGATIONS,
    VALID_ROLES,
    VALID_STRATEGIES,
    CovariateSpec,
)

__all__ = [
    "AGGREGATIONS",
    "VALID_ROLES",
    "VALID_STRATEGIES",
    "CovariateFrame",
    "CovariateSpec",
    "PointInTimeCovariates",
    "resolve_covariates",
]
