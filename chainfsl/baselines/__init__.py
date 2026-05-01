"""Baselines: FedAvg, SplitFed, SplitFedV2, AdaptSFL, DFL."""
from .fedavg import FedAvgBaseline
from .splitfed import SplitFedBaseline
from .splitfed_v2 import SplitFedV2Baseline
from .adaptsfl import AdaptSFLBaseline
from .dfl import DFLBaseline

__all__ = [
    "FedAvgBaseline",
    "SplitFedBaseline",
    "SplitFedV2Baseline",
    "AdaptSFLBaseline",
    "DFLBaseline",
]
