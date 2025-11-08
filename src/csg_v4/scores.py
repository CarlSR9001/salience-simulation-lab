"""Scoring functions for Continuity–Strain Gravity (CSG) Version 4.

This module computes all the individual scoring components that feed into the
salience formula. Each score captures a different aspect of galaxy structure,
dynamics, or morphology:

- C_local: Local continuity (ordered rotation vs. random motion)
- R_galaxy: Retention (age-weighted kinematic stability)
- K_galaxy: Coherence fraction (percentage of ordered regions)
- M_local: Mass density score
- W_local: Curvature irregularity (velocity profile smoothness)
- phi_galaxy: Structural disorder penalty
- J_galaxy: Maximum coherence gap
- X_galaxy: Edge case flag (low-mass coherent systems)
- S_h_galaxy: Stress-history score
- DeltaA_local: Morphology-dependent boost
"""

from __future__ import annotations

import numpy as np

from .config import CSGConfig
from .galaxy import GalaxyData


def clip(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    """Convenience wrapper for np.clip to ensure values stay in [low, high]."""
    return np.clip(arr, low, high)


def compute_c_local(galaxy: GalaxyData, config: CSGConfig) -> np.ndarray:
    """Compute local continuity score at each radius.

    Continuity measures the dominance of ordered rotation (v_obs) over random
    motion (sigma_v). Higher continuity indicates a more coherently rotating
    disk with less turbulence or measurement uncertainty.

    Formula: C = 1 / (1 + sigma/V0)
    - When sigma << V0: C → 1 (highly ordered)
    - When sigma >> V0: C → 0 (turbulent/disordered)

    Args:
        galaxy: Galaxy data with velocity dispersion profile.
        config: Configuration with V0 reference velocity (~50 km/s).

    Returns:
        C_local score in [0, 1] at each radial point.
    """
    # Normalize dispersion by reference velocity V0
    sigma_term = galaxy.sigma_v / (config.V0 + config.sigma_clip_floor)
    # Continuity decreases as dispersion increases
    c_local = 1.0 / (1.0 + sigma_term)
    return clip(c_local, 0.0, 1.0)


def compute_r_galaxy(galaxy: GalaxyData) -> float:
    """Compute galaxy-wide retention score.

    Retention combines age and kinematic stability to measure how well a
    galaxy has "retained" its ordered structure over time. Older galaxies
    with stable, rotation-dominated kinematics score higher.

    Formula: R = 0.5 * tanh(age/5 Gyr) + 0.5 * median(v/(v+sigma))

    Args:
        galaxy: Galaxy data with age and kinematics.

    Returns:
        R_galaxy score in [0, 1].
    """
    # Age contribution: saturates around 5 Gyr
    age_term = np.tanh(galaxy.age_gyr / 5.0)
    # Kinematic contribution: ratio of ordered to total motion
    ratio = galaxy.v_obs / (galaxy.v_obs + galaxy.sigma_v + 1.0e-6)
    kinematic_term = float(np.median(ratio))
    # Equal weighting of age and kinematics
    r_galaxy = 0.5 * age_term + 0.5 * kinematic_term
    return float(np.clip(r_galaxy, 0.0, 1.0))


def compute_phi_x_flags(galaxy: GalaxyData) -> tuple[float, float, float]:
    """Compute structural disorder penalty (phi) and edge case flags (X).

    phi_galaxy measures intrinsic structural disorder based on morphology.
    - Grand design spirals: phi = 0.4 (low disorder)
    - Dwarfs/irregulars: phi = 0.6 (higher disorder)

    The "unfair edge" flag identifies low-mass, gas-rich, coherently rotating
    systems that exhibit MOND-like behavior but would be penalized by naive
    disorder metrics. These systems receive a phi reduction (-0.2) and an
    X boost to counteract the penalty.

    Args:
        galaxy: Galaxy data with type and kinematic properties.

    Returns:
        Tuple of (phi_galaxy, X_galaxy, unfair_edge_flag):
        - phi_galaxy: Disorder penalty in [0, 1]
        - X_galaxy: Edge case boost (1.0 if triggered, else 0.0)
        - unfair_edge_flag: Boolean flag cast to float
    """
    # Base disorder depends on morphology
    galaxy_type = galaxy.galaxy_type.lower()
    if galaxy_type in {"grand_design", "hsb_spiral"}:
        phi_base = 0.4  # Well-organized spirals have low disorder
    elif galaxy_type in {"lsb_dwarf", "dwarf_irregular"}:
        phi_base = 0.6  # Dwarfs/irregulars have higher disorder
    else:
        phi_base = 0.5  # Default mid-range

    # "Unfair edge" case: low-mass, gas-dominated, but coherently rotating
    # These systems deserve salience boost despite superficial disorder
    unfair_edge_flag = (
        galaxy.vmax() < 70.0             # Low maximum velocity
        and galaxy.gas_fraction > 0.7     # Gas-dominated
        and galaxy.has_coherent_rotation  # But still coherently rotating
    )
    if unfair_edge_flag:
        phi_base -= 0.2  # Reduce disorder penalty

    phi_galaxy = float(np.clip(phi_base, 0.0, 1.0))
    x_galaxy = 1.0 if unfair_edge_flag else 0.0
    return phi_galaxy, x_galaxy, float(unfair_edge_flag)


def compute_s_h_galaxy(galaxy: GalaxyData) -> float:
    """Compute stress-history score.

    S_h combines age, gas content, and stress (dispersion) to capture the
    galaxy's dynamical history. Higher scores indicate systems that have
    experienced sustained stress or retained high gas fractions.

    Args:
        galaxy: Galaxy data with age, gas fraction, and kinematics.

    Returns:
        S_h_galaxy score in [0, 1].
    """
    age_term = np.tanh(galaxy.age_gyr / 5.0)
    stress_term_raw = galaxy.sigma_v / (galaxy.v_obs + 1.0e-6)
    stress_term = clip(np.median(stress_term_raw) / 0.5, 0.0, 1.0)
    s_h = 0.4 * age_term + 0.3 * galaxy.gas_fraction + 0.3 * stress_term
    return float(np.clip(s_h, 0.0, 1.0))


def compute_k_galaxy(c_local: np.ndarray) -> float:
    """Compute coherence fraction (K_galaxy).

    K measures what fraction of radial points have "high" continuity (C >= 0.6).
    This indicates how much of the galaxy's disk is coherently rotating.

    Args:
        c_local: Local continuity scores at each radius.

    Returns:
        K_galaxy in [0, 1]: fraction of points with C >= 0.6.
    """
    n_coherent = np.count_nonzero(c_local >= 0.6)
    k_galaxy = n_coherent / c_local.size
    return float(np.clip(k_galaxy, 0.0, 1.0))


def compute_j_galaxy(c_local: np.ndarray) -> float:
    """Compute coherence gap (J_galaxy).

    J = 1 - max(C_local) measures the "gap" between observed coherence and
    perfect coherence. Higher J means even the best region has poor continuity.

    Args:
        c_local: Local continuity scores at each radius.

    Returns:
        J_galaxy in [0, 1]: coherence deficiency.
    """
    j_galaxy = 1.0 - float(np.max(c_local))
    return float(np.clip(j_galaxy, 0.0, 1.0))


def compute_w_local(galaxy: GalaxyData, config: CSGConfig) -> np.ndarray:
    """Compute local curvature irregularity (W_local).

    W measures the second derivative (curvature) of the rotation curve.
    High W indicates bumps, wiggles, or irregularities that reduce salience.

    Args:
        galaxy: Galaxy data with rotation curve.
        config: Configuration with curvature scaling parameters.

    Returns:
        W_local in [0, 1] at each radius: normalized curvature.
    """
    v = galaxy.v_obs
    # First derivative: dv/dr
    dv = np.gradient(v, galaxy.radii_kpc)
    # Second derivative: d²v/dr²
    d2v = np.gradient(dv, galaxy.radii_kpc)
    curvature = np.abs(d2v)
    # Normalize by maximum curvature
    normalized = curvature / (np.max(curvature) + config.curvature_floor)
    w_local = clip(normalized / config.curvature_scale, 0.0, 1.0)
    return w_local


def compute_m_local(galaxy: GalaxyData, config: CSGConfig) -> np.ndarray:
    """Compute local mass density score (M_local).

    M scores the baryonic surface density on a log scale. Higher densities
    contribute more to salience, as they represent regions where gravity
    should be more "responsive" to baryonic matter.

    Formula: M = (log10(Σ_baryon) - 6) / 4, clipped to [0, 1]

    Args:
        galaxy: Galaxy data with baryonic mass profile.
        config: Configuration for computing mass profile.

    Returns:
        M_local in [0, 1] at each radius.
    """
    m_baryon = galaxy.baryonic_mass_profile(config)
    # Surface density: M / (π r²)
    sigma_b = m_baryon / (np.pi * np.square(galaxy.radii_kpc))
    log_sigma = np.log10(sigma_b + 1.0e-10)
    # Map log10(Σ) from ~6-10 to [0, 1]
    m_local = clip((log_sigma - 6.0) / 4.0, 0.0, 1.0)
    return m_local


def compute_delta_a_local(galaxy: GalaxyData) -> np.ndarray:
    """Compute morphology-dependent boost (DeltaA_local).

    DeltaA provides a radial profile boost based on galaxy type, with a
    Gaussian peak at a characteristic radius r_feature:
    - Grand design spirals: high boost (0.7) peaking at ~5 kpc
    - Spirals: moderate boost (0.5) peaking at ~5 kpc
    - Dwarfs: lower boost (0.3) peaking at ~2 kpc

    Args:
        galaxy: Galaxy data with morphological type.

    Returns:
        DeltaA_local in [0, 1] at each radius.
    """
    galaxy_type = galaxy.galaxy_type.lower()
    if galaxy_type == "grand_design":
        base, r_feature = 0.7, 5.0
    elif "spiral" in galaxy_type:
        base, r_feature = 0.5, 5.0
    elif "dwarf" in galaxy_type:
        base, r_feature = 0.3, 2.0
    else:
        base, r_feature = 0.4, 5.0

    r = galaxy.radii_kpc
    # Gaussian-like radial profile centered on r_feature
    radial_factor = np.exp(-np.square((r - r_feature) / r_feature))
    delta_a = base * (0.5 + 0.5 * radial_factor)
    return clip(delta_a, 0.0, 1.0)


def compute_scores(galaxy: GalaxyData, config: CSGConfig) -> dict[str, np.ndarray | float]:
    """Compute all scoring components for a galaxy.

    This is the main entry point for computing the full suite of salience
    scores. It orchestrates all individual scoring functions and applies
    special adjustments for "workhorse" galaxies (well-behaved spirals that
    deserve reduced penalties) and metadata-driven feedback.

    The "workhorse" logic identifies high-quality spiral galaxies that meet
    strict thresholds for retention, coherence, continuity, and mass. These
    galaxies receive:
    - Reduced disorder penalty (phi drops by ~0.15)
    - Reduced curvature penalty (W scaled by ~0.7)
    - HYPE bonus (~0.35)

    Args:
        galaxy: Galaxy data with all required properties.
        config: Configuration with scoring parameters and thresholds.

    Returns:
        Dictionary containing all scores (scalars and arrays).
    """
    # Compute all base scoring components
    c_local = compute_c_local(galaxy, config)
    r_galaxy = compute_r_galaxy(galaxy)
    phi_galaxy, x_galaxy, unfair_flag = compute_phi_x_flags(galaxy)
    s_h = compute_s_h_galaxy(galaxy)
    k_galaxy = compute_k_galaxy(c_local)
    j_galaxy = compute_j_galaxy(c_local)
    w_local = compute_w_local(galaxy, config)
    m_local = compute_m_local(galaxy, config)
    delta_a = compute_delta_a_local(galaxy)

    # Compute mean values for workhorse criteria
    c_mean = float(np.mean(c_local)) if c_local.size else 0.0
    m_mean = float(np.mean(m_local)) if m_local.size else 0.0

    # "Workhorse" galaxies: well-behaved spirals meeting strict quality criteria
    # These are the "goldilocks" systems that exhibit ideal salience properties
    galaxy_type = galaxy.galaxy_type.lower()
    workhorse_candidate = galaxy_type in {"hsb_spiral", "grand_design", "spiral"}
    workhorse_flag = (
        workhorse_candidate
        and r_galaxy >= config.workhorse_r_threshold    # High retention
        and k_galaxy >= config.workhorse_k_threshold    # High coherence
        and c_mean >= config.workhorse_c_threshold      # High average continuity
        and m_mean >= config.workhorse_m_threshold      # Sufficient mass
    )

    # Apply workhorse bonuses: reduce penalties for disorder and curvature
    if workhorse_flag:
        phi_galaxy = float(np.clip(phi_galaxy - config.phi_workhorse_drop, 0.0, 1.0))
        w_local = clip(w_local * config.workhorse_w_scale, 0.0, 1.0)

    workhorse_bonus = config.gamma_workhorse if workhorse_flag else 0.0

    # Apply metadata-driven feedback adjustments (for experimental overrides)
    metadata = galaxy.metadata or {}
    feedback_hype = float(metadata.get("feedback_hype", 0.0))
    phi_feedback = float(metadata.get("feedback_phi_delta", 0.0))
    w_feedback_scale = float(metadata.get("feedback_w_scale", 1.0))

    if phi_feedback != 0.0:
        phi_galaxy = float(np.clip(phi_galaxy + phi_feedback, 0.0, 1.0))
    if w_feedback_scale != 1.0:
        w_local = clip(w_local * w_feedback_scale, 0.0, 1.0)
    workhorse_bonus += feedback_hype

    return {
        "C_local": c_local,
        "R_galaxy": r_galaxy,
        "phi_galaxy": phi_galaxy,
        "X_galaxy": x_galaxy,
        "S_h_galaxy": s_h,
        "K_galaxy": k_galaxy,
        "J_galaxy": j_galaxy,
        "W_local": w_local,
        "M_local": m_local,
        "DeltaA_local": delta_a,
        "unfair_edge_flag": unfair_flag,
        "Workhorse_flag": float(workhorse_flag),
        "HYPE_workhorse": workhorse_bonus,
        "C_local_mean": c_mean,
        "M_local_mean": m_mean,
    }
