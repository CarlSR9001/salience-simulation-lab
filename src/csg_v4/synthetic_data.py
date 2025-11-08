"""Synthetic galaxy datasets for CSG V4 validation.

This module provides synthetic test galaxies that mimic real systems:
- NGC6503-like: High surface brightness spiral
- DDO154-like: Low surface brightness dwarf
- NGC2403-like: Grand design spiral

These synthetic galaxies are useful for:
- Testing the CSG V4 pipeline
- Demonstrating typical behavior
- Quick validation without downloading SPARC data

Example:
    >>> from csg_v4.synthetic_data import load_synthetic_galaxies
    >>>
    >>> galaxies = load_synthetic_galaxies()
    >>> for name, galaxy in galaxies.items():
    ...     print(f"{name}: {galaxy.galaxy_type}, v_max = {galaxy.vmax():.1f} km/s")

Note: These are idealized test cases, not real observational data.
For production analysis, use real SPARC data via data_io.SPARCParser.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .galaxy import GalaxyData


def _make_ngc6503_like() -> GalaxyData:
    radius = np.linspace(0.5, 13.0, 16)
    v_obs = np.array([20, 45, 70, 95, 110, 120, 125, 128, 130, 131, 132, 131, 130, 129, 128, 127], dtype=float)
    v_bar = np.array([18, 40, 60, 80, 95, 105, 110, 112, 113, 112, 110, 108, 105, 102, 98, 95], dtype=float)
    sigma_v = np.array([25, 22, 20, 18, 16, 15, 14, 14, 13, 13, 14, 15, 17, 20, 22, 25], dtype=float)
    return GalaxyData(
        name="NGC6503-like",
        galaxy_type="hsb_spiral",
        radii_kpc=radius,
        v_obs=v_obs,
        v_bar=v_bar,
        sigma_v=sigma_v,
        gas_fraction=0.35,
        age_gyr=10.0,
        has_coherent_rotation=True,
    )


def _make_ddo154_like() -> GalaxyData:
    radius = np.linspace(0.3, 10.0, 16)
    v_obs = np.array([10, 20, 30, 40, 50, 58, 62, 65, 68, 70, 71, 72, 72, 71, 70, 69], dtype=float)
    v_bar = np.array([5, 10, 15, 20, 26, 32, 36, 39, 41, 42, 42, 41, 39, 36, 33, 30], dtype=float)
    sigma_v = np.array([18, 17, 16, 15, 14, 13, 13, 12, 12, 11, 11, 11, 12, 13, 14, 15], dtype=float)
    return GalaxyData(
        name="DDO154-like",
        galaxy_type="lsb_dwarf",
        radii_kpc=radius,
        v_obs=v_obs,
        v_bar=v_bar,
        sigma_v=sigma_v,
        gas_fraction=0.85,
        age_gyr=9.5,
        has_coherent_rotation=True,
    )


def _make_ngc2403_like() -> GalaxyData:
    radius = np.linspace(0.5, 16.0, 16)
    v_obs = np.array([30, 60, 90, 120, 140, 150, 160, 170, 175, 178, 180, 181, 181, 180, 178, 175], dtype=float)
    v_bar = np.array([25, 50, 75, 100, 120, 135, 145, 150, 152, 150, 148, 145, 140, 135, 130, 125], dtype=float)
    sigma_v = np.array([30, 28, 25, 22, 20, 18, 17, 17, 18, 19, 20, 22, 24, 27, 30, 33], dtype=float)
    return GalaxyData(
        name="NGC2403-like",
        galaxy_type="grand_design",
        radii_kpc=radius,
        v_obs=v_obs,
        v_bar=v_bar,
        sigma_v=sigma_v,
        gas_fraction=0.45,
        age_gyr=11.0,
        has_coherent_rotation=True,
    )


def load_synthetic_galaxies() -> Dict[str, GalaxyData]:
    galaxies = {
        "NGC6503": _make_ngc6503_like(),
        "DDO154": _make_ddo154_like(),
        "NGC2403": _make_ngc2403_like(),
    }
    return galaxies
