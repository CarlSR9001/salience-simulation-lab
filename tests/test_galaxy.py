"""Unit tests for galaxy.py module."""

import numpy as np
import pytest

from csg_v4.config import CSGConfig
from csg_v4.galaxy import GalaxyData


class TestGalaxyDataValidation:
    """Test GalaxyData validation logic."""

    def test_valid_galaxy_creation(self):
        """Test creating a valid GalaxyData instance."""
        galaxy = GalaxyData(
            name="TestGalaxy",
            galaxy_type="spiral",
            radii_kpc=np.array([1.0, 2.0, 3.0]),
            v_obs=np.array([100.0, 120.0, 110.0]),
            v_bar=np.array([80.0, 90.0, 85.0]),
            sigma_v=np.array([20.0, 25.0, 22.0]),
            gas_fraction=0.5,
            age_gyr=8.0,
            has_coherent_rotation=True,
        )
        assert galaxy.name == "TestGalaxy"
        assert galaxy.galaxy_type == "spiral"
        assert galaxy.n_radii == 3

    def test_mismatched_array_lengths(self):
        """Test that mismatched array lengths raise ValueError."""
        with pytest.raises(ValueError, match="All radial arrays must have the same length"):
            GalaxyData(
                name="BadGalaxy",
                galaxy_type="spiral",
                radii_kpc=np.array([1.0, 2.0, 3.0]),
                v_obs=np.array([100.0, 120.0]),  # Wrong length
                v_bar=np.array([80.0, 90.0, 85.0]),
                sigma_v=np.array([20.0, 25.0, 22.0]),
                gas_fraction=0.5,
                age_gyr=8.0,
                has_coherent_rotation=True,
            )

    def test_negative_radii(self):
        """Test that negative radii raise ValueError."""
        with pytest.raises(ValueError, match="Radii must be strictly positive"):
            GalaxyData(
                name="BadGalaxy",
                galaxy_type="spiral",
                radii_kpc=np.array([1.0, -2.0, 3.0]),  # Negative radius
                v_obs=np.array([100.0, 120.0, 110.0]),
                v_bar=np.array([80.0, 90.0, 85.0]),
                sigma_v=np.array([20.0, 25.0, 22.0]),
                gas_fraction=0.5,
                age_gyr=8.0,
                has_coherent_rotation=True,
            )

    def test_zero_radii(self):
        """Test that zero radii raise ValueError."""
        with pytest.raises(ValueError, match="Radii must be strictly positive"):
            GalaxyData(
                name="BadGalaxy",
                galaxy_type="spiral",
                radii_kpc=np.array([1.0, 0.0, 3.0]),  # Zero radius
                v_obs=np.array([100.0, 120.0, 110.0]),
                v_bar=np.array([80.0, 90.0, 85.0]),
                sigma_v=np.array([20.0, 25.0, 22.0]),
                gas_fraction=0.5,
                age_gyr=8.0,
                has_coherent_rotation=True,
            )

    def test_gas_fraction_below_zero(self):
        """Test that gas_fraction below 0 raises ValueError."""
        with pytest.raises(ValueError, match="gas_fraction must lie in"):
            GalaxyData(
                name="BadGalaxy",
                galaxy_type="spiral",
                radii_kpc=np.array([1.0, 2.0, 3.0]),
                v_obs=np.array([100.0, 120.0, 110.0]),
                v_bar=np.array([80.0, 90.0, 85.0]),
                sigma_v=np.array([20.0, 25.0, 22.0]),
                gas_fraction=-0.1,  # Negative gas fraction
                age_gyr=8.0,
                has_coherent_rotation=True,
            )

    def test_gas_fraction_above_one(self):
        """Test that gas_fraction above 1 raises ValueError."""
        with pytest.raises(ValueError, match="gas_fraction must lie in"):
            GalaxyData(
                name="BadGalaxy",
                galaxy_type="spiral",
                radii_kpc=np.array([1.0, 2.0, 3.0]),
                v_obs=np.array([100.0, 120.0, 110.0]),
                v_bar=np.array([80.0, 90.0, 85.0]),
                sigma_v=np.array([20.0, 25.0, 22.0]),
                gas_fraction=1.5,  # Above 1
                age_gyr=8.0,
                has_coherent_rotation=True,
            )

    def test_gas_fraction_edge_cases(self):
        """Test that gas_fraction at 0 and 1 are valid."""
        # Test gas_fraction = 0
        galaxy1 = GalaxyData(
            name="TestGalaxy1",
            galaxy_type="spiral",
            radii_kpc=np.array([1.0, 2.0, 3.0]),
            v_obs=np.array([100.0, 120.0, 110.0]),
            v_bar=np.array([80.0, 90.0, 85.0]),
            sigma_v=np.array([20.0, 25.0, 22.0]),
            gas_fraction=0.0,
            age_gyr=8.0,
            has_coherent_rotation=True,
        )
        assert galaxy1.gas_fraction == 0.0

        # Test gas_fraction = 1
        galaxy2 = GalaxyData(
            name="TestGalaxy2",
            galaxy_type="spiral",
            radii_kpc=np.array([1.0, 2.0, 3.0]),
            v_obs=np.array([100.0, 120.0, 110.0]),
            v_bar=np.array([80.0, 90.0, 85.0]),
            sigma_v=np.array([20.0, 25.0, 22.0]),
            gas_fraction=1.0,
            age_gyr=8.0,
            has_coherent_rotation=True,
        )
        assert galaxy2.gas_fraction == 1.0

    def test_negative_age(self):
        """Test that negative age raises ValueError."""
        with pytest.raises(ValueError, match="age_gyr must be positive"):
            GalaxyData(
                name="BadGalaxy",
                galaxy_type="spiral",
                radii_kpc=np.array([1.0, 2.0, 3.0]),
                v_obs=np.array([100.0, 120.0, 110.0]),
                v_bar=np.array([80.0, 90.0, 85.0]),
                sigma_v=np.array([20.0, 25.0, 22.0]),
                gas_fraction=0.5,
                age_gyr=-1.0,  # Negative age
                has_coherent_rotation=True,
            )

    def test_zero_age(self):
        """Test that zero age raises ValueError."""
        with pytest.raises(ValueError, match="age_gyr must be positive"):
            GalaxyData(
                name="BadGalaxy",
                galaxy_type="spiral",
                radii_kpc=np.array([1.0, 2.0, 3.0]),
                v_obs=np.array([100.0, 120.0, 110.0]),
                v_bar=np.array([80.0, 90.0, 85.0]),
                sigma_v=np.array([20.0, 25.0, 22.0]),
                gas_fraction=0.5,
                age_gyr=0.0,  # Zero age
                has_coherent_rotation=True,
            )

    def test_negative_velocities(self):
        """Test that negative velocities raise ValueError."""
        # Negative v_obs
        with pytest.raises(ValueError, match="v_obs must be non-negative"):
            GalaxyData(
                name="BadGalaxy",
                galaxy_type="spiral",
                radii_kpc=np.array([1.0, 2.0, 3.0]),
                v_obs=np.array([100.0, -120.0, 110.0]),
                v_bar=np.array([80.0, 90.0, 85.0]),
                sigma_v=np.array([20.0, 25.0, 22.0]),
                gas_fraction=0.5,
                age_gyr=8.0,
                has_coherent_rotation=True,
            )

        # Negative v_bar
        with pytest.raises(ValueError, match="v_bar must be non-negative"):
            GalaxyData(
                name="BadGalaxy",
                galaxy_type="spiral",
                radii_kpc=np.array([1.0, 2.0, 3.0]),
                v_obs=np.array([100.0, 120.0, 110.0]),
                v_bar=np.array([80.0, -90.0, 85.0]),
                sigma_v=np.array([20.0, 25.0, 22.0]),
                gas_fraction=0.5,
                age_gyr=8.0,
                has_coherent_rotation=True,
            )

        # Negative sigma_v
        with pytest.raises(ValueError, match="sigma_v must be non-negative"):
            GalaxyData(
                name="BadGalaxy",
                galaxy_type="spiral",
                radii_kpc=np.array([1.0, 2.0, 3.0]),
                v_obs=np.array([100.0, 120.0, 110.0]),
                v_bar=np.array([80.0, 90.0, 85.0]),
                sigma_v=np.array([20.0, -25.0, 22.0]),
                gas_fraction=0.5,
                age_gyr=8.0,
                has_coherent_rotation=True,
            )


class TestGalaxyDataProperties:
    """Test GalaxyData properties and methods."""

    def test_n_radii_property(self):
        """Test n_radii property returns correct size."""
        galaxy = GalaxyData(
            name="TestGalaxy",
            galaxy_type="spiral",
            radii_kpc=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            v_obs=np.array([100.0, 120.0, 110.0, 105.0, 100.0]),
            v_bar=np.array([80.0, 90.0, 85.0, 82.0, 78.0]),
            sigma_v=np.array([20.0, 25.0, 22.0, 21.0, 19.0]),
            gas_fraction=0.5,
            age_gyr=8.0,
            has_coherent_rotation=True,
        )
        assert galaxy.n_radii == 5

    def test_vmax_method(self):
        """Test vmax returns maximum observed velocity."""
        galaxy = GalaxyData(
            name="TestGalaxy",
            galaxy_type="spiral",
            radii_kpc=np.array([1.0, 2.0, 3.0, 4.0]),
            v_obs=np.array([100.0, 150.0, 120.0, 110.0]),
            v_bar=np.array([80.0, 90.0, 85.0, 82.0]),
            sigma_v=np.array([20.0, 25.0, 22.0, 21.0]),
            gas_fraction=0.5,
            age_gyr=8.0,
            has_coherent_rotation=True,
        )
        assert galaxy.vmax() == 150.0

    def test_baryonic_mass_profile(self):
        """Test baryonic mass profile calculation."""
        config = CSGConfig()
        galaxy = GalaxyData(
            name="TestGalaxy",
            galaxy_type="spiral",
            radii_kpc=np.array([1.0, 2.0, 3.0]),
            v_obs=np.array([100.0, 120.0, 110.0]),
            v_bar=np.array([80.0, 90.0, 85.0]),
            sigma_v=np.array([20.0, 25.0, 22.0]),
            gas_fraction=0.5,
            age_gyr=8.0,
            has_coherent_rotation=True,
        )
        m_baryon = galaxy.baryonic_mass_profile(config)

        # Verify shape
        assert m_baryon.shape == galaxy.radii_kpc.shape

        # Verify formula: m_baryon = v_bar^2 * r / G
        expected = np.square(galaxy.v_bar) * galaxy.radii_kpc / config.G
        np.testing.assert_allclose(m_baryon, expected)

    def test_baryonic_acceleration(self):
        """Test baryonic acceleration calculation."""
        config = CSGConfig()
        galaxy = GalaxyData(
            name="TestGalaxy",
            galaxy_type="spiral",
            radii_kpc=np.array([1.0, 2.0, 3.0]),
            v_obs=np.array([100.0, 120.0, 110.0]),
            v_bar=np.array([80.0, 90.0, 85.0]),
            sigma_v=np.array([20.0, 25.0, 22.0]),
            gas_fraction=0.5,
            age_gyr=8.0,
            has_coherent_rotation=True,
        )
        g_bar = galaxy.baryonic_acceleration(config)

        # Verify shape
        assert g_bar.shape == galaxy.radii_kpc.shape

        # Verify formula: g_bar = G * m_baryon / r^2
        m_baryon = galaxy.baryonic_mass_profile(config)
        expected = config.G * m_baryon / np.square(galaxy.radii_kpc)
        np.testing.assert_allclose(g_bar, expected)

    def test_metadata_optional(self):
        """Test that metadata is optional."""
        galaxy = GalaxyData(
            name="TestGalaxy",
            galaxy_type="spiral",
            radii_kpc=np.array([1.0, 2.0, 3.0]),
            v_obs=np.array([100.0, 120.0, 110.0]),
            v_bar=np.array([80.0, 90.0, 85.0]),
            sigma_v=np.array([20.0, 25.0, 22.0]),
            gas_fraction=0.5,
            age_gyr=8.0,
            has_coherent_rotation=True,
        )
        assert galaxy.metadata is None

    def test_metadata_with_values(self):
        """Test that metadata can be provided."""
        metadata = {"feedback_hype": 0.1, "feedback_phi_delta": 0.05}
        galaxy = GalaxyData(
            name="TestGalaxy",
            galaxy_type="spiral",
            radii_kpc=np.array([1.0, 2.0, 3.0]),
            v_obs=np.array([100.0, 120.0, 110.0]),
            v_bar=np.array([80.0, 90.0, 85.0]),
            sigma_v=np.array([20.0, 25.0, 22.0]),
            gas_fraction=0.5,
            age_gyr=8.0,
            has_coherent_rotation=True,
            metadata=metadata,
        )
        assert galaxy.metadata == metadata
        assert galaxy.metadata["feedback_hype"] == 0.1


class TestGalaxyDataEdgeCases:
    """Test edge cases for GalaxyData."""

    def test_single_radii_point(self):
        """Test galaxy with single radial point."""
        galaxy = GalaxyData(
            name="SinglePoint",
            galaxy_type="spiral",
            radii_kpc=np.array([1.0]),
            v_obs=np.array([100.0]),
            v_bar=np.array([80.0]),
            sigma_v=np.array([20.0]),
            gas_fraction=0.5,
            age_gyr=8.0,
            has_coherent_rotation=True,
        )
        assert galaxy.n_radii == 1
        assert galaxy.vmax() == 100.0

    def test_large_number_of_points(self):
        """Test galaxy with many radial points."""
        n_points = 100
        galaxy = GalaxyData(
            name="ManyPoints",
            galaxy_type="spiral",
            radii_kpc=np.linspace(0.1, 10.0, n_points),
            v_obs=np.ones(n_points) * 100.0,
            v_bar=np.ones(n_points) * 80.0,
            sigma_v=np.ones(n_points) * 20.0,
            gas_fraction=0.5,
            age_gyr=8.0,
            has_coherent_rotation=True,
        )
        assert galaxy.n_radii == n_points

    def test_zero_velocities_allowed(self):
        """Test that zero velocities are allowed (non-negative)."""
        galaxy = GalaxyData(
            name="ZeroVel",
            galaxy_type="spiral",
            radii_kpc=np.array([1.0, 2.0, 3.0]),
            v_obs=np.array([0.0, 120.0, 110.0]),
            v_bar=np.array([0.0, 90.0, 85.0]),
            sigma_v=np.array([0.0, 25.0, 22.0]),
            gas_fraction=0.5,
            age_gyr=8.0,
            has_coherent_rotation=True,
        )
        assert galaxy.n_radii == 3
