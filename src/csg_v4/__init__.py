"""Continuity–Strain Gravity (CSG) Version 4 implementation package.

CSG V4 predicts galaxy rotation curves using salience-based gravitational
modifications. Instead of dark matter, the model proposes that gravitational
response depends on the "salience" of baryonic structure—a composite measure
of continuity, coherence, retention, mass density, and morphology.

Main Components:
    - CSGV4Model: Core model for computing salience and predicting velocities
    - GalaxyData: Container for galaxy observables (rotation curves, etc.)
    - CSGConfig: Configuration with all tunable parameters
    - scan_kappa: Optimize the global calibration constant κ_c
    - compute_error_metrics: Evaluate model predictions against observations
    - summarize_galaxy_metrics: Generate summary tables

Quick Start:
    >>> from csg_v4 import CSGV4Model, GalaxyData, scan_kappa
    >>> from csg_v4.synthetic_data import load_synthetic_galaxies
    >>>
    >>> # Load test galaxies
    >>> galaxies = load_synthetic_galaxies()
    >>>
    >>> # Optimize κ_c
    >>> model = CSGV4Model()
    >>> scan_result, _ = scan_kappa(galaxies.values(), model=model)
    >>> print(f"Best κ_c: {scan_result.best_kappa:.4f}")

For detailed documentation, see README.md and docs/CONCEPTS.md.
"""

from .config import CSGConfig
from .galaxy import GalaxyData
from .model import CSGV4Model, ModelOutputs
from .optimizer import scan_kappa
from .reporting import summarize_galaxy_metrics, compute_error_metrics

__all__ = [
    "CSGConfig",
    "GalaxyData",
    "CSGV4Model",
    "ModelOutputs",
    "scan_kappa",
    "summarize_galaxy_metrics",
    "compute_error_metrics",
]
