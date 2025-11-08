"""Unit tests for scores.py module."""

import numpy as np
import pytest

from csg_v4.config import CSGConfig
from csg_v4.galaxy import GalaxyData
from csg_v4.scores import (
    clip,
    compute_c_local,
    compute_delta_a_local,
    compute_j_galaxy,
    compute_k_galaxy,
    compute_m_local,
    compute_phi_x_flags,
    compute_r_galaxy,
    compute_s_h_galaxy,
    compute_scores,
    compute_w_local,
)


def create_test_galaxy(
    galaxy_type="spiral",
    n_points=5,
    vmax=100.0,
    gas_fraction=0.5,
    age_gyr=8.0,
    has_coherent_rotation=True,
    metadata=None,
):
    """Helper function to create a test galaxy."""
    return GalaxyData(
        name="TestGalaxy",
        galaxy_type=galaxy_type,
        radii_kpc=np.linspace(1.0, 10.0, n_points),
        v_obs=np.linspace(vmax, vmax * 0.8, n_points),
        v_bar=np.linspace(vmax * 0.7, vmax * 0.6, n_points),
        sigma_v=np.linspace(20.0, 30.0, n_points),
        gas_fraction=gas_fraction,
        age_gyr=age_gyr,
        has_coherent_rotation=has_coherent_rotation,
        metadata=metadata,
    )


class TestClipFunction:
    """Test the clip utility function."""

    def test_clip_within_bounds(self):
        """Test clipping values within bounds."""
        arr = np.array([0.3, 0.5, 0.7])
        result = clip(arr, 0.0, 1.0)
        np.testing.assert_array_equal(result, arr)

    def test_clip_below_lower_bound(self):
        """Test clipping values below lower bound."""
        arr = np.array([-0.5, 0.5, 1.5])
        result = clip(arr, 0.0, 1.0)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_clip_above_upper_bound(self):
        """Test clipping values above upper bound."""
        arr = np.array([0.5, 1.5, 2.0])
        result = clip(arr, 0.0, 1.0)
        expected = np.array([0.5, 1.0, 1.0])
        np.testing.assert_array_equal(result, expected)


class TestComputeCLocal:
    """Test compute_c_local function."""

    def test_c_local_basic(self):
        """Test basic c_local computation."""
        config = CSGConfig()
        galaxy = create_test_galaxy()
        c_local = compute_c_local(galaxy, config)

        # Check shape
        assert c_local.shape == galaxy.radii_kpc.shape

        # Check bounds [0, 1]
        assert np.all(c_local >= 0.0)
        assert np.all(c_local <= 1.0)

    def test_c_local_formula(self):
        """Test c_local formula implementation."""
        config = CSGConfig()
        galaxy = create_test_galaxy()
        c_local = compute_c_local(galaxy, config)

        # Verify formula: c_local = 1 / (1 + sigma_v / (V0 + floor))
        sigma_term = galaxy.sigma_v / (config.V0 + config.sigma_clip_floor)
        expected = 1.0 / (1.0 + sigma_term)
        expected = np.clip(expected, 0.0, 1.0)
        np.testing.assert_allclose(c_local, expected)

    def test_c_local_high_sigma(self):
        """Test c_local with high sigma_v (should give low c_local)."""
        config = CSGConfig()
        galaxy = create_test_galaxy(n_points=3)
        galaxy.sigma_v[0] = 1000.0  # Very high sigma_v
        c_local = compute_c_local(galaxy, config)

        # High sigma should give low c_local
        assert c_local[0] < 0.1


class TestComputeRGalaxy:
    """Test compute_r_galaxy function."""

    def test_r_galaxy_basic(self):
        """Test basic r_galaxy computation."""
        galaxy = create_test_galaxy()
        r_galaxy = compute_r_galaxy(galaxy)

        # Check bounds [0, 1]
        assert 0.0 <= r_galaxy <= 1.0

    def test_r_galaxy_old_age(self):
        """Test r_galaxy with old galaxy (should be higher)."""
        galaxy_young = create_test_galaxy(age_gyr=1.0)
        galaxy_old = create_test_galaxy(age_gyr=20.0)

        r_young = compute_r_galaxy(galaxy_young)
        r_old = compute_r_galaxy(galaxy_old)

        # Older galaxy should have higher r_galaxy
        assert r_old > r_young

    def test_r_galaxy_formula(self):
        """Test r_galaxy formula implementation."""
        galaxy = create_test_galaxy()
        r_galaxy = compute_r_galaxy(galaxy)

        # Verify formula components
        age_term = np.tanh(galaxy.age_gyr / 5.0)
        ratio = galaxy.v_obs / (galaxy.v_obs + galaxy.sigma_v + 1.0e-6)
        kinematic_term = float(np.median(ratio))
        expected = 0.5 * age_term + 0.5 * kinematic_term
        expected = float(np.clip(expected, 0.0, 1.0))

        assert abs(r_galaxy - expected) < 1e-10


class TestComputePhiXFlags:
    """Test compute_phi_x_flags function."""

    def test_phi_x_grand_design(self):
        """Test phi and x for grand_design galaxies."""
        galaxy = create_test_galaxy(galaxy_type="grand_design", vmax=150.0)
        phi, x, unfair = compute_phi_x_flags(galaxy)

        # Grand design should have phi_base = 0.4
        assert phi == 0.4
        assert x == 0.0
        assert unfair == 0.0

    def test_phi_x_hsb_spiral(self):
        """Test phi and x for hsb_spiral galaxies."""
        galaxy = create_test_galaxy(galaxy_type="hsb_spiral", vmax=150.0)
        phi, x, unfair = compute_phi_x_flags(galaxy)

        # HSB spiral should have phi_base = 0.4
        assert phi == 0.4
        assert x == 0.0

    def test_phi_x_lsb_dwarf(self):
        """Test phi and x for lsb_dwarf galaxies."""
        galaxy = create_test_galaxy(galaxy_type="lsb_dwarf", vmax=150.0)
        phi, x, unfair = compute_phi_x_flags(galaxy)

        # LSB dwarf should have phi_base = 0.6
        assert phi == 0.6
        assert x == 0.0

    def test_phi_x_dwarf_irregular(self):
        """Test phi and x for dwarf_irregular galaxies."""
        galaxy = create_test_galaxy(galaxy_type="dwarf_irregular", vmax=150.0)
        phi, x, unfair = compute_phi_x_flags(galaxy)

        # Dwarf irregular should have phi_base = 0.6
        assert phi == 0.6
        assert x == 0.0

    def test_phi_x_other_type(self):
        """Test phi and x for other galaxy types."""
        galaxy = create_test_galaxy(galaxy_type="elliptical", vmax=150.0)
        phi, x, unfair = compute_phi_x_flags(galaxy)

        # Other types should have phi_base = 0.5
        assert phi == 0.5
        assert x == 0.0

    def test_phi_x_unfair_edge_flag(self):
        """Test unfair edge flag (vmax < 70, gas > 0.7, has rotation)."""
        galaxy = create_test_galaxy(
            galaxy_type="spiral",
            vmax=60.0,
            gas_fraction=0.8,
            has_coherent_rotation=True,
        )
        phi, x, unfair = compute_phi_x_flags(galaxy)

        # Should trigger unfair edge flag
        assert unfair == 1.0
        assert x == 1.0
        # phi should be reduced by 0.2
        assert phi == 0.3  # 0.5 - 0.2

    def test_phi_x_no_unfair_edge_high_vmax(self):
        """Test that high vmax prevents unfair edge flag."""
        galaxy = create_test_galaxy(
            galaxy_type="spiral",
            vmax=100.0,  # Too high
            gas_fraction=0.8,
            has_coherent_rotation=True,
        )
        phi, x, unfair = compute_phi_x_flags(galaxy)

        # Should not trigger unfair edge flag
        assert unfair == 0.0
        assert x == 0.0

    def test_phi_x_bounds(self):
        """Test that phi is always in [0, 1]."""
        galaxy = create_test_galaxy()
        phi, x, unfair = compute_phi_x_flags(galaxy)

        assert 0.0 <= phi <= 1.0
        assert x in [0.0, 1.0]


class TestComputeSHGalaxy:
    """Test compute_s_h_galaxy function."""

    def test_s_h_basic(self):
        """Test basic s_h computation."""
        galaxy = create_test_galaxy()
        s_h = compute_s_h_galaxy(galaxy)

        # Check bounds [0, 1]
        assert 0.0 <= s_h <= 1.0

    def test_s_h_formula(self):
        """Test s_h formula implementation."""
        galaxy = create_test_galaxy()
        s_h = compute_s_h_galaxy(galaxy)

        # Verify formula
        age_term = np.tanh(galaxy.age_gyr / 5.0)
        stress_term_raw = galaxy.sigma_v / (galaxy.v_obs + 1.0e-6)
        stress_term = np.clip(np.median(stress_term_raw) / 0.5, 0.0, 1.0)
        expected = 0.4 * age_term + 0.3 * galaxy.gas_fraction + 0.3 * stress_term
        expected = float(np.clip(expected, 0.0, 1.0))

        assert abs(s_h - expected) < 1e-10

    def test_s_h_old_galaxy(self):
        """Test s_h increases with galaxy age."""
        galaxy_young = create_test_galaxy(age_gyr=1.0)
        galaxy_old = create_test_galaxy(age_gyr=20.0)

        s_h_young = compute_s_h_galaxy(galaxy_young)
        s_h_old = compute_s_h_galaxy(galaxy_old)

        # Older galaxy should have higher s_h contribution from age
        assert s_h_old > s_h_young


class TestComputeKGalaxy:
    """Test compute_k_galaxy function."""

    def test_k_galaxy_all_coherent(self):
        """Test k_galaxy when all points are coherent."""
        c_local = np.array([0.7, 0.8, 0.9, 0.6])
        k_galaxy = compute_k_galaxy(c_local)

        # All points >= 0.6, so k should be 1.0
        assert k_galaxy == 1.0

    def test_k_galaxy_none_coherent(self):
        """Test k_galaxy when no points are coherent."""
        c_local = np.array([0.1, 0.2, 0.3, 0.4])
        k_galaxy = compute_k_galaxy(c_local)

        # No points >= 0.6, so k should be 0.0
        assert k_galaxy == 0.0

    def test_k_galaxy_partial_coherent(self):
        """Test k_galaxy with partial coherence."""
        c_local = np.array([0.7, 0.3, 0.8, 0.2, 0.9])
        k_galaxy = compute_k_galaxy(c_local)

        # 3 out of 5 points >= 0.6
        expected = 3.0 / 5.0
        assert k_galaxy == expected

    def test_k_galaxy_bounds(self):
        """Test that k_galaxy is in [0, 1]."""
        c_local = np.linspace(0.0, 1.0, 10)
        k_galaxy = compute_k_galaxy(c_local)

        assert 0.0 <= k_galaxy <= 1.0


class TestComputeJGalaxy:
    """Test compute_j_galaxy function."""

    def test_j_galaxy_basic(self):
        """Test basic j_galaxy computation."""
        c_local = np.array([0.5, 0.7, 0.9, 0.6])
        j_galaxy = compute_j_galaxy(c_local)

        # j = 1 - max(c_local) = 1 - 0.9 = 0.1
        assert abs(j_galaxy - 0.1) < 1e-10

    def test_j_galaxy_max_c(self):
        """Test j_galaxy when max c_local is 1.0."""
        c_local = np.array([0.5, 0.7, 1.0, 0.6])
        j_galaxy = compute_j_galaxy(c_local)

        # j = 1 - 1.0 = 0.0
        assert j_galaxy == 0.0

    def test_j_galaxy_min_c(self):
        """Test j_galaxy when max c_local is 0.0."""
        c_local = np.array([0.0, 0.0, 0.0, 0.0])
        j_galaxy = compute_j_galaxy(c_local)

        # j = 1 - 0.0 = 1.0
        assert j_galaxy == 1.0

    def test_j_galaxy_bounds(self):
        """Test that j_galaxy is in [0, 1]."""
        c_local = np.linspace(0.0, 1.0, 10)
        j_galaxy = compute_j_galaxy(c_local)

        assert 0.0 <= j_galaxy <= 1.0


class TestComputeWLocal:
    """Test compute_w_local function."""

    def test_w_local_basic(self):
        """Test basic w_local computation."""
        config = CSGConfig()
        galaxy = create_test_galaxy(n_points=10)
        w_local = compute_w_local(galaxy, config)

        # Check shape
        assert w_local.shape == galaxy.radii_kpc.shape

        # Check bounds [0, 1]
        assert np.all(w_local >= 0.0)
        assert np.all(w_local <= 1.0)

    def test_w_local_flat_velocity(self):
        """Test w_local with flat velocity curve (low curvature)."""
        config = CSGConfig()
        galaxy = create_test_galaxy(n_points=10)
        # Set flat velocity
        galaxy.v_obs[:] = 100.0

        w_local = compute_w_local(galaxy, config)

        # Flat curve should have near-zero curvature
        assert np.all(w_local < 0.1)

    def test_w_local_curved_velocity(self):
        """Test w_local with curved velocity profile."""
        config = CSGConfig()
        galaxy = create_test_galaxy(n_points=20)
        # Create parabolic velocity profile
        r = galaxy.radii_kpc
        galaxy.v_obs[:] = 100.0 - 0.5 * (r - 5.0) ** 2

        w_local = compute_w_local(galaxy, config)

        # Some points should have higher curvature
        assert np.max(w_local) > 0.1


class TestComputeMLocal:
    """Test compute_m_local function."""

    def test_m_local_basic(self):
        """Test basic m_local computation."""
        config = CSGConfig()
        galaxy = create_test_galaxy()
        m_local = compute_m_local(galaxy, config)

        # Check shape
        assert m_local.shape == galaxy.radii_kpc.shape

        # Check bounds [0, 1]
        assert np.all(m_local >= 0.0)
        assert np.all(m_local <= 1.0)

    def test_m_local_formula(self):
        """Test m_local formula implementation."""
        config = CSGConfig()
        galaxy = create_test_galaxy()
        m_local = compute_m_local(galaxy, config)

        # Verify formula
        m_baryon = galaxy.baryonic_mass_profile(config)
        sigma_b = m_baryon / (np.pi * np.square(galaxy.radii_kpc))
        log_sigma = np.log10(sigma_b + 1.0e-10)
        expected = np.clip((log_sigma - 6.0) / 4.0, 0.0, 1.0)

        np.testing.assert_allclose(m_local, expected)


class TestComputeDeltaALocal:
    """Test compute_delta_a_local function."""

    def test_delta_a_grand_design(self):
        """Test delta_a for grand_design galaxies."""
        galaxy = create_test_galaxy(galaxy_type="grand_design")
        delta_a = compute_delta_a_local(galaxy)

        # Check shape and bounds
        assert delta_a.shape == galaxy.radii_kpc.shape
        assert np.all(delta_a >= 0.0)
        assert np.all(delta_a <= 1.0)

    def test_delta_a_spiral(self):
        """Test delta_a for spiral galaxies."""
        galaxy = create_test_galaxy(galaxy_type="spiral")
        delta_a = compute_delta_a_local(galaxy)

        # Check shape and bounds
        assert delta_a.shape == galaxy.radii_kpc.shape
        assert np.all(delta_a >= 0.0)
        assert np.all(delta_a <= 1.0)

    def test_delta_a_dwarf(self):
        """Test delta_a for dwarf galaxies."""
        galaxy = create_test_galaxy(galaxy_type="dwarf_irregular")
        delta_a = compute_delta_a_local(galaxy)

        # Check shape and bounds
        assert delta_a.shape == galaxy.radii_kpc.shape
        assert np.all(delta_a >= 0.0)
        assert np.all(delta_a <= 1.0)

    def test_delta_a_other(self):
        """Test delta_a for other galaxy types."""
        galaxy = create_test_galaxy(galaxy_type="elliptical")
        delta_a = compute_delta_a_local(galaxy)

        # Check shape and bounds
        assert delta_a.shape == galaxy.radii_kpc.shape
        assert np.all(delta_a >= 0.0)
        assert np.all(delta_a <= 1.0)


class TestComputeScores:
    """Test compute_scores integration function."""

    def test_compute_scores_basic(self):
        """Test basic compute_scores returns all expected keys."""
        config = CSGConfig()
        galaxy = create_test_galaxy()
        scores = compute_scores(galaxy, config)

        # Check all expected keys are present
        expected_keys = {
            "C_local",
            "R_galaxy",
            "phi_galaxy",
            "X_galaxy",
            "S_h_galaxy",
            "K_galaxy",
            "J_galaxy",
            "W_local",
            "M_local",
            "DeltaA_local",
            "unfair_edge_flag",
            "Workhorse_flag",
            "HYPE_workhorse",
            "C_local_mean",
            "M_local_mean",
        }
        assert set(scores.keys()) == expected_keys

    def test_compute_scores_array_shapes(self):
        """Test that array scores have correct shapes."""
        config = CSGConfig()
        galaxy = create_test_galaxy(n_points=10)
        scores = compute_scores(galaxy, config)

        # Check array shapes
        assert scores["C_local"].shape == (10,)
        assert scores["W_local"].shape == (10,)
        assert scores["M_local"].shape == (10,)
        assert scores["DeltaA_local"].shape == (10,)

        # Check scalar values
        assert isinstance(scores["R_galaxy"], float)
        assert isinstance(scores["phi_galaxy"], float)
        assert isinstance(scores["K_galaxy"], float)

    def test_compute_scores_workhorse_flag(self):
        """Test workhorse flag computation."""
        config = CSGConfig()
        # Create a galaxy that should be a workhorse
        galaxy = create_test_galaxy(
            galaxy_type="hsb_spiral",
            n_points=20,
            vmax=150.0,
            age_gyr=12.0,
        )
        scores = compute_scores(galaxy, config)

        # Workhorse flag should be 0 or 1
        assert scores["Workhorse_flag"] in [0.0, 1.0]

        # If workhorse, bonus should be positive
        if scores["Workhorse_flag"] == 1.0:
            assert scores["HYPE_workhorse"] > 0.0

    def test_compute_scores_with_metadata(self):
        """Test compute_scores with metadata feedback."""
        config = CSGConfig()
        metadata = {
            "feedback_hype": 0.1,
            "feedback_phi_delta": 0.05,
            "feedback_w_scale": 0.9,
        }
        galaxy = create_test_galaxy(metadata=metadata)
        scores = compute_scores(galaxy, config)

        # Should include feedback in hype bonus
        assert scores["HYPE_workhorse"] >= 0.1

    def test_compute_scores_bounds(self):
        """Test that all score values are in valid bounds."""
        config = CSGConfig()
        galaxy = create_test_galaxy()
        scores = compute_scores(galaxy, config)

        # Check bounds for scalar scores
        assert 0.0 <= scores["R_galaxy"] <= 1.0
        assert 0.0 <= scores["phi_galaxy"] <= 1.0
        assert 0.0 <= scores["K_galaxy"] <= 1.0
        assert 0.0 <= scores["J_galaxy"] <= 1.0
        assert 0.0 <= scores["S_h_galaxy"] <= 1.0

        # Check bounds for array scores
        assert np.all(scores["C_local"] >= 0.0) and np.all(scores["C_local"] <= 1.0)
        assert np.all(scores["W_local"] >= 0.0) and np.all(scores["W_local"] <= 1.0)
        assert np.all(scores["M_local"] >= 0.0) and np.all(scores["M_local"] <= 1.0)
        assert np.all(scores["DeltaA_local"] >= 0.0) and np.all(
            scores["DeltaA_local"] <= 1.0
        )

    def test_compute_scores_c_mean(self):
        """Test C_local_mean calculation."""
        config = CSGConfig()
        galaxy = create_test_galaxy()
        scores = compute_scores(galaxy, config)

        # C_mean should be mean of C_local
        expected_c_mean = float(np.mean(scores["C_local"]))
        assert abs(scores["C_local_mean"] - expected_c_mean) < 1e-10

    def test_compute_scores_m_mean(self):
        """Test M_local_mean calculation."""
        config = CSGConfig()
        galaxy = create_test_galaxy()
        scores = compute_scores(galaxy, config)

        # M_mean should be mean of M_local
        expected_m_mean = float(np.mean(scores["M_local"]))
        assert abs(scores["M_local_mean"] - expected_m_mean) < 1e-10
