# Layer 2: Ensemble Learning

from .meta_learner import MetaLearner
from .advanced_ensemble import (
    AdvancedEnsemble,
    EnsembleMethod,
    create_advanced_ensemble,
)


# Import ensemble integrator conditionally to avoid circular imports
def create_ensemble_integrator(symbol: str, **kwargs):
    """Create ensemble integrator with lazy import."""
    from .ensemble_integrator import EnsembleIntegrator, EnsembleConfig

    config = EnsembleConfig(symbol=symbol, **kwargs)
    return EnsembleIntegrator(config)


__all__ = [
    "MetaLearner",
    "AdvancedEnsemble",
    "EnsembleMethod",
    "create_advanced_ensemble",
    "create_ensemble_integrator",
]
