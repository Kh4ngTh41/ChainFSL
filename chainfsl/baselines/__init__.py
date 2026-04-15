"""Baselines: FedAvg, SplitFed, AdaptSFL, DFL."""
from .fedavg import FedAvgBaseline
from .splitfed import SplitFedBaseline
from .adaptsfl import AdaptSFLBaseline
from .dfl import DFLBaseline

__all__ = [
    "FedAvgBaseline",
    "SplitFedBaseline",
    "AdaptSFLBaseline",
    "DFLBaseline",
]
