"""
Risk Harmoniser v1 Module

Provides edge blending and position sizing with risk constraints as specified
in Future_instruction.txt Task E.

Public API:
- EdgeBlender: Blends multiple model edges with confidence weighting
- RiskAwarePositionSizer: Calculates position sizes with VaR constraints  
- create_edge_blender(): Factory function for EdgeBlender
- create_position_sizer(): Factory function for RiskAwarePositionSizer

Mathematical Components:
- Edge blending with decay weights and Bayesian shrinkage
- Kelly criterion position sizing with VaR impact calculation
- Asset-class specific risk limits and leverage constraints
"""

from .edge_blender import (
    EdgeBlender,
    ModelEdge,
    BlendedEdge,
    create_edge_blender
)

from .position_sizer import (
    RiskAwarePositionSizer,
    PositionSizeResult,
    create_position_sizer
)

__all__ = [
    'EdgeBlender',
    'ModelEdge', 
    'BlendedEdge',
    'RiskAwarePositionSizer',
    'PositionSizeResult',
    'create_edge_blender',
    'create_position_sizer'
]

__version__ = "1.0.0" 