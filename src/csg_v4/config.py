"""Configuration primitives for Continuity–Strain Gravity (CSG) Version 4."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class CSGConfig:
    """Holds tunable constants for the CSG V4 scoring and prediction pipeline.

    This frozen dataclass centralizes all hyperparameters, physical constants,
    and optimization settings. Key parameter groups:

    **Physical Constants:**
    - G, c, H0: Gravitational constant, speed of light, Hubble constant
    - V0: Reference velocity scale (~50 km/s)
    - a0: MOND acceleration scale (computed from c*H0/1000)

    **Salience Weights:**
    - alpha_n, gamma_X, gamma_Sh, gamma_W: HYPE term coefficients
    - beta_phi, beta_J: PENALTY term coefficients
    - gamma_workhorse: Bonus for high-quality spirals

    **Quality Factor Blending:**
    - aura_mix: Blend weight for global vs. local quality (default 0.5)
    - min_s_prime, min_q_final: Numerical stability floors

    **Optimizer Settings:**
    - kappa_min, kappa_max, kappa_samples: Grid search range
    - gss_tol, gss_max_iter: Golden section search parameters
    - bootstrap_samples: Number of bootstrap resamples for uncertainty

    **Workhorse Thresholds:**
    - workhorse_r/k/c/m_threshold: Quality criteria for workhorse galaxies
    """

    G: float = 4.302e-6
    c: float = 299_792.458
    H0: float = 70.0
    V0: float = 50.0

    alpha_n: float = 0.35
    gamma_X: float = 0.25
    gamma_Sh: float = 0.30
    gamma_W: float = 0.55
    gamma_workhorse: float = 0.35
    beta_phi: float = 0.60
    beta_J: float = 0.45

    aura_mix: float = 0.5
    min_s_prime: float = 1.0e-10
    min_q_final: float = 1.0e-6

    kappa_min: float = 0.001
    kappa_max: float = 1.0
    kappa_samples: int = 2000
    zoom_samples: int = 400
    quadratic_window: int = 7
    gss_tol: float = 1.0e-5
    gss_max_iter: int = 128
    bootstrap_samples: int = 200
    bootstrap_seed: int = 1337

    inner_radius_fraction: float = 0.3
    min_inner_points: int = 3

    curvature_floor: float = 1.0e-6
    curvature_scale: float = 0.25
    workhorse_w_scale: float = 0.7
    phi_workhorse_drop: float = 0.15
    workhorse_r_threshold: float = 0.85
    workhorse_k_threshold: float = 0.75
    workhorse_c_threshold: float = 0.7
    workhorse_m_threshold: float = 0.3

    sigma_clip_floor: float = 1.0e-6

    _kappa_grid: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        grid = np.linspace(self.kappa_min, self.kappa_max, self.kappa_samples)
        object.__setattr__(self, "_kappa_grid", grid)

    @property
    def a0(self) -> float:
        """MOND acceleration scale: a0 = c * H0 / 1000 [kpc/Gyr²].

        This is the characteristic acceleration below which MOND effects dominate.
        Typical value: ~2.1e4 kpc/Gyr² ≈ 1.2e-10 m/s².
        """
        return self.c * self.H0 / 1000.0

    @property
    def kappa_grid(self) -> np.ndarray:
        """Pre-computed grid of kappa_c values for scanning."""
        return self._kappa_grid
