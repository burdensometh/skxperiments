"""Experimental design module.

Provides design objects (CRD, BlockedCRD, ReRandomizedCRD, FactorialDesign,
ClusterDesign) and standalone diagnostic functions (check_balance,
power_analysis).
"""

from skxperiments.design.balance import check_balance
from skxperiments.design.blocked_crd import BlockedCRD
from skxperiments.design.cluster import ClusterDesign
from skxperiments.design.crd import CRD
from skxperiments.design.factorial import FactorialDesign
from skxperiments.design.power import PowerResult, power_analysis
from skxperiments.design.rerandomized_crd import ReRandomizedCRD

__all__ = [
    "BlockedCRD",
    "CRD",
    "ClusterDesign",
    "FactorialDesign",
    "PowerResult",
    "ReRandomizedCRD",
    "check_balance",
    "power_analysis",
]