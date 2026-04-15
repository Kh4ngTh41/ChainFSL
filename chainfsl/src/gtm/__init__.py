"""GTM module: Game-Theoretic Tokenomics."""
from .contribution import ContributionVector, VLIComputer, compute_contribution_vector, aggregate_contributions
from .shapley import TMCSShapley, ShapleyResult, ShapleyConfig, ShapleyCalculator, validate_shapley_efficiency
from .tokenomics import TokenomicsEngine, TokenomicsConfig, NashValidator

__all__ = [
    "ContributionVector",
    "VLIComputer",
    "compute_contribution_vector",
    "aggregate_contributions",
    "TMCSShapley",
    "ShapleyResult",
    "ShapleyConfig",
    "ShapleyCalculator",
    "validate_shapley_efficiency",
    "TokenomicsEngine",
    "TokenomicsConfig",
    "NashValidator",
]