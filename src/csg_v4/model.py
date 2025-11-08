"""CSG V4 model pipeline.

This module implements the Continuity-Strain Gravity (CSG) Version 4 model,
which predicts galaxy rotation curves by computing salience-based modifications
to gravitational acceleration. The key innovation is that regions with higher
"salience" (a composite measure of continuity, coherence, mass density, etc.)
exhibit different gravitational behavior, potentially explaining flat rotation
curves without invoking dark matter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import CSGConfig
from .galaxy import GalaxyData
from .scores import compute_scores


@dataclass(slots=True)
class ModelOutputs:
    """Container for all outputs from a CSG V4 model prediction.

    Attributes:
        kappa_c: Calibration constant that scales the acceleration a0.
        scores: Dictionary of all computed scoring components (C_local, R_galaxy, etc.).
        s_prime: Raw salience values at each radial point (before Q transformation).
        q_local: Local salience quality factor (s_prime^0.25) at each radius.
        a_galaxy: Galaxy-averaged quality factor from inner region.
        q_final: Final quality factor blending local and global components.
        a_eff: Effective acceleration scale at each radius [kpc/Gyr²].
        v_pred: Predicted circular velocity at each radius [km/s].
        g_obs_pred: Predicted observed acceleration at each radius [kpc/Gyr²].
    """
    kappa_c: float
    scores: dict[str, np.ndarray | float]
    s_prime: np.ndarray
    q_local: np.ndarray
    a_galaxy: float
    q_final: np.ndarray
    a_eff: np.ndarray
    v_pred: np.ndarray
    g_obs_pred: np.ndarray


class CSGV4Model:
    """Main CSG V4 model for predicting galaxy rotation curves via salience.

    This class encapsulates the full CSG V4 pipeline:
    1. Compute salience scores from galaxy properties (continuity, mass, coherence, etc.)
    2. Transform salience into a quality factor Q
    3. Use Q to modulate the interpolating function between Newtonian and MOND regimes
    4. Predict circular velocities that match observations without dark matter

    Attributes:
        config: Configuration object containing all tunable parameters.
    """

    def __init__(self, config: CSGConfig | None = None) -> None:
        """Initialize the CSG V4 model.

        Args:
            config: Optional configuration. If None, uses default CSGConfig().
        """
        self.config = config or CSGConfig()

    def compute_salience(self, galaxy: GalaxyData) -> dict[str, np.ndarray | float]:
        """Compute the full salience profile for a galaxy.

        This is the heart of the CSG V4 model. Salience S' is computed as:
            S' = CORE * HYPE * PENALTY

        Where:
            CORE = C_local * R_galaxy * K_galaxy * M_local
                (continuity × retention × coherence × mass)
            HYPE = 1 + (boost terms) - (penalty terms)
                (morphology-dependent amplification)
            PENALTY = 1 - beta_phi * phi_galaxy - beta_J * J_galaxy
                (structural disorder penalties)

        Then S' is converted to a quality factor:
            Q_local = S'^0.25  (fourth-root scaling)
            Q_final = (1-aura_mix)*Q_local + aura_mix*A_galaxy

        Args:
            galaxy: Galaxy data including rotation curve and properties.

        Returns:
            Dictionary containing all intermediate scores plus S_prime, Q_local,
            A_galaxy, and Q_final arrays/scalars.
        """
        config = self.config
        # Compute all base scores: continuity, retention, coherence, mass, etc.
        scores = compute_scores(galaxy, config)

        # Extract key scoring components for salience formula
        c_local = scores["C_local"]      # Local continuity (velocity/dispersion ratio)
        r_g = scores["R_galaxy"]         # Galaxy retention (age + kinematics)
        k_g = scores["K_galaxy"]         # Coherence fraction (how many points are ordered)
        m_local = scores["M_local"]      # Local mass density score
        delta_a = scores["DeltaA_local"] # Morphology-dependent boost
        unfair_flag = float(scores["unfair_edge_flag"])
        workhorse_flag = float(scores.get("Workhorse_flag", 0.0))
        workhorse_bonus = float(scores.get("HYPE_workhorse", 0.0))
        c_mean = float(scores.get("C_local_mean", 0.0))
        m_mean = float(scores.get("M_local_mean", 0.0))

        # CORE: Product of continuity, retention, coherence, and mass
        # This captures the "fundamental salience" from basic galaxy properties
        core = (c_local * r_g * k_g) * m_local

        # HYPE: Morphology-dependent amplification factor (always >= 0.1)
        # Boosts for organized structures (DeltaA, X, S_h)
        # Penalties for curvature irregularities (W) and workhorse bonuses
        hype = 1.0 + (
            config.alpha_n * scores["DeltaA_local"]   # Morphological structure boost
            + config.gamma_X * scores["X_galaxy"]     # Edge case boost (low-mass coherent)
            + config.gamma_Sh * scores["S_h_galaxy"]  # Stress-history boost
            - config.gamma_W * scores["W_local"]      # Curvature penalty
            + workhorse_bonus                         # Bonus for well-behaved spirals
        )
        hype = np.maximum(hype, 0.1)  # Floor to prevent negative or zero amplification

        # PENALTY: Structural disorder reductions (always in [0, 1])
        # Penalizes galaxies with high disorder (phi) or low max coherence (J)
        penalty = 1.0 - config.beta_phi * scores["phi_galaxy"] - config.beta_J * scores["J_galaxy"]
        penalty = np.clip(penalty, 0.0, 1.0)

        # Raw salience: S' = CORE × HYPE × PENALTY
        s_prime = core * hype * penalty
        s_prime = np.maximum(s_prime, config.min_s_prime)  # Numerical stability floor

        # Quality factor: Q_local = S'^(1/4)
        # Fourth-root scaling provides smooth interpolation in acceleration formula
        q_local = np.power(s_prime, 0.25)

        # Galaxy aura: Average Q in inner region (represents global "character")
        inner_count = max(int(np.ceil(galaxy.n_radii * config.inner_radius_fraction)), config.min_inner_points)
        inner_q = q_local[:inner_count]
        a_galaxy = float(np.clip(np.mean(inner_q), 0.0, 1.0))

        # Final quality factor: Blend local and global components
        # aura_mix controls how much the galaxy's "global character" influences outer regions
        q_final = (1.0 - config.aura_mix) * q_local + config.aura_mix * a_galaxy
        q_final = np.maximum(q_final, config.min_q_final)  # Numerical stability floor

        return {
            **scores,
            "CORE": core,
            "HYPE": hype,
            "PENALTY": penalty,
            "S_prime": s_prime,
            "DeltaA_local": delta_a,
            "unfair_edge_flag": unfair_flag,
            "Workhorse_flag": float(workhorse_flag),
            "HYPE_workhorse": workhorse_bonus,
            "C_local_mean": c_mean,
            "M_local_mean": m_mean,
            "Q_local": q_local,
            "A_galaxy": a_galaxy,
            "Q_final": q_final,
        }

    def predict_velocity(self, galaxy: GalaxyData, kappa_c: float) -> ModelOutputs:
        """Predict circular velocities using the CSG V4 interpolating function.

        The CSG V4 model interpolates between Newtonian gravity and a MOND-like
        regime based on salience-derived quality factors. The key equation is:

            a_eff = kappa_c * (a0 / Q_final)  [effective acceleration scale]
            x = g_bar / a_eff                   [dimensionless ratio]
            mu = x / (1 + x)                    [interpolation weight]
            g_obs = mu * g_bar + (1-mu) * sqrt(a_eff * g_bar)

        When g_bar >> a_eff (high acceleration): mu → 1, g_obs → g_bar (Newtonian)
        When g_bar << a_eff (low acceleration): mu → 0, g_obs → sqrt(a_eff*g_bar) (MOND)

        Args:
            galaxy: Galaxy data with rotation curve and properties.
            kappa_c: Global calibration constant (typically fitted across many galaxies).

        Returns:
            ModelOutputs containing all predictions and intermediate values.
        """
        config = self.config
        # Compute salience and extract final quality factor
        salience = self.compute_salience(galaxy)
        q_final = salience["Q_final"]

        # Effective acceleration scale: higher salience (lower Q) → lower a_eff → more MOND-like
        a_eff = kappa_c * (config.a0 / q_final)

        # Baryonic acceleration from visible matter
        g_bar = galaxy.baryonic_acceleration(config)

        # Interpolating function: x = g_bar/a_eff, mu = x/(1+x)
        x = g_bar / a_eff
        mu = x / (1.0 + x)

        # Observed acceleration: weighted average of Newtonian and deep-MOND regimes
        g_obs = mu * g_bar + (1.0 - mu) * np.sqrt(a_eff * g_bar)

        # Circular velocity: v = sqrt(g * r)
        v_pred = np.sqrt(g_obs * galaxy.radii_kpc)

        return ModelOutputs(
            kappa_c=kappa_c,
            scores=salience,
            s_prime=salience["S_prime"],
            q_local=salience["Q_local"],
            a_galaxy=salience["A_galaxy"],
            q_final=q_final,
            a_eff=a_eff,
            v_pred=v_pred,
            g_obs_pred=g_obs,
        )

    @staticmethod
    def residuals(galaxy: GalaxyData, outputs: ModelOutputs) -> np.ndarray:
        """Compute fractional velocity residuals.

        Args:
            galaxy: Galaxy data with observed velocities.
            outputs: Model predictions.

        Returns:
            Fractional residuals: (v_pred - v_obs) / v_obs at each radius.
        """
        return (outputs.v_pred - galaxy.v_obs) / (galaxy.v_obs + 1.0e-6)

    @staticmethod
    def chisq(residuals: Iterable[np.ndarray]) -> float:
        """Compute mean squared fractional residual across multiple galaxies.

        This is the optimization objective for kappa_c fitting.

        Args:
            residuals: Iterable of residual arrays, one per galaxy.

        Returns:
            Mean squared fractional residual averaged over all radial points
            across all galaxies.
        """
        accum = 0.0
        count = 0
        for resid in residuals:
            resid = np.asarray(resid)
            accum += float(np.sum(np.square(resid)))
            count += resid.size
        return accum / max(count, 1)
