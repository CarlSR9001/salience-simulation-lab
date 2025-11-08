"""Galaxy data structures for CSG V4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import numpy as np

from .config import CSGConfig


@dataclass(slots=True)
class GalaxyData:
    """Container for radial galaxy observables used by the CSG pipeline.

    This dataclass holds all galaxy properties required for CSG V4 analysis:
    rotation curves, velocity dispersions, baryonic velocities, and metadata.

    Attributes:
        name: Galaxy identifier (e.g., "NGC6503" or "DDO154").
        galaxy_type: Morphological classification (e.g., "grand_design", "hsb_spiral", "lsb_dwarf").
        radii_kpc: Radial sampling points [kpc].
        v_obs: Observed circular velocities [km/s].
        v_bar: Baryonic circular velocities from visible matter [km/s].
        sigma_v: Velocity dispersion / measurement uncertainty [km/s].
        gas_fraction: Fractional gas mass: M_gas / (M_gas + M_stars), in [0, 1].
        age_gyr: Estimated galaxy age [Gyr].
        has_coherent_rotation: Boolean flag for kinematically coherent rotation.
        metadata: Optional dict for additional properties (distance, inclination, etc.).

    All radial arrays (radii_kpc, v_obs, v_bar, sigma_v) must have the same length.
    Validation is performed in __post_init__.
    """

    name: str
    galaxy_type: str
    radii_kpc: np.ndarray
    v_obs: np.ndarray
    v_bar: np.ndarray
    sigma_v: np.ndarray
    gas_fraction: float
    age_gyr: float
    has_coherent_rotation: bool
    metadata: Optional[Mapping[str, float]] = None

    def __post_init__(self) -> None:
        # Validate array types
        arrays: Iterable[np.ndarray] = (self.radii_kpc, self.v_obs, self.v_bar, self.sigma_v)
        array_names = ["radii_kpc", "v_obs", "v_bar", "sigma_v"]

        for arr, name in zip(arrays, array_names):
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"{name} must be a numpy array, got {type(arr)}.\n"
                    f"Use np.array() to convert your data."
                )
            if arr.size == 0:
                raise ValueError(
                    f"{name} is empty.\n"
                    f"Galaxy data must contain at least one radial sample point."
                )

        # Validate array lengths match
        lengths = {arr.size for arr in arrays}
        if len(lengths) != 1:
            length_info = ", ".join(f"{name}={arr.size}" for name, arr in zip(array_names, arrays))
            raise ValueError(
                f"All radial arrays must have the same length.\n"
                f"Got: {length_info}\n"
                f"Each array represents measurements at the same radial points."
            )

        # Validate radii are positive
        if np.any(self.radii_kpc <= 0):
            invalid_count = np.sum(self.radii_kpc <= 0)
            min_radius = np.min(self.radii_kpc)
            raise ValueError(
                f"Radii must be strictly positive to avoid singular accelerations.\n"
                f"Found {invalid_count} non-positive radius value(s), minimum={min_radius:.6f} kpc.\n"
                f"Check your input data for zeros or negative values."
            )

        # Validate gas fraction
        if not (0.0 <= self.gas_fraction <= 1.0):
            raise ValueError(
                f"gas_fraction must lie in [0, 1], got {self.gas_fraction}.\n"
                f"Gas fraction represents the ratio of gas mass to total baryonic mass.\n"
                f"A value outside [0, 1] indicates invalid or incorrectly scaled data."
            )

        # Validate age
        if self.age_gyr <= 0:
            raise ValueError(
                f"age_gyr must be positive, got {self.age_gyr}.\n"
                f"Age should represent the galaxy's age in billions of years (typical: 8-13 Gyr)."
            )

        # Validate velocities are non-negative
        for arr, name in zip([self.v_obs, self.v_bar, self.sigma_v], ["v_obs", "v_bar", "sigma_v"]):
            if np.any(arr < 0):
                invalid_count = np.sum(arr < 0)
                min_value = np.min(arr)
                raise ValueError(
                    f"{name} must be non-negative (velocity magnitudes).\n"
                    f"Found {invalid_count} negative value(s), minimum={min_value:.3f} km/s.\n"
                    f"Check your input data for sign errors or improper conversions."
                )

        # Validate finite values
        for arr, name in zip(arrays, array_names):
            if not np.all(np.isfinite(arr)):
                nan_count = np.sum(np.isnan(arr))
                inf_count = np.sum(np.isinf(arr))
                raise ValueError(
                    f"{name} contains non-finite values (NaN={nan_count}, Inf={inf_count}).\n"
                    f"All values must be finite numbers for physical calculations."
                )

    @property
    def n_radii(self) -> int:
        return self.radii_kpc.size

    def baryonic_mass_profile(self, config: CSGConfig) -> np.ndarray:
        """Return enclosed baryonic mass M_baryon(r) inferred from v_bar.

        Uses the Newtonian relation: M(r) = v²(r) * r / G

        Args:
            config: Configuration providing gravitational constant G.

        Returns:
            Enclosed baryonic mass [M_sun] at each radius.
        """
        v_bar_sq = np.square(self.v_bar)
        m_baryon = v_bar_sq * self.radii_kpc / config.G
        return m_baryon

    def baryonic_acceleration(self, config: CSGConfig) -> np.ndarray:
        """Compute baryonic gravitational acceleration g_bar = G M(r) / r².

        Args:
            config: Configuration providing gravitational constant G.

        Returns:
            Baryonic acceleration [kpc/Gyr²] at each radius.
        """
        m_baryon = self.baryonic_mass_profile(config)
        g_bar = config.G * m_baryon / np.square(self.radii_kpc)
        return g_bar

    def vmax(self) -> float:
        """Return maximum observed velocity in the rotation curve [km/s]."""
        return float(np.max(self.v_obs))
