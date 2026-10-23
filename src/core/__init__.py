"""
Core detection pipeline modules.

Modules:
    - chart_processor: ENC file loading and layer extraction
    - ais_processor: AIS trajectory cleaning and metrics
    - fusion_layer: Spatial join of AIS + chart data
    - detect_interactions: 5-stage interaction detection engine
    - collision_risk: Multi-vessel CPA/TCPA assessment
    - ukc_risk: Under-Keel-Clearance analysis
    - nav_constraints: Navigation rule validation
"""

__all__ = [
    "chart_processor",
    "ais_processor",
    "fusion_layer",
    "detect_interactions",
    "collision_risk",
    "ukc_risk",
    "nav_constraints",
]
