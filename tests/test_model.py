"""Unit tests for model.py module."""

import numpy as np
import pytest

from csg_v4.config import CSGConfig
from csg_v4.galaxy import GalaxyData
from csg_v4.model import CSGV4Model, ModelOutputs


def create_test_galaxy(
    galaxy_type="spiral",
    n_points=10,
    vmax=120.0,
    gas_fraction=0.5,
    age_gyr=8.0,
    has_coherent_rotation=True,
    metadata=None,
):
    """Helper function to create a test galaxy."""
    radii = np.linspace(1.0, 10.0, n_points)
    return GalaxyData(
        name="TestGalaxy",
        galaxy_type=galaxy_type,
        radii_kpc=radii,
        v_obs=np.linspace(vmax, vmax * 0.8, n_points),
        v_bar=np.linspace(vmax * 0.7, vmax * 0.6, n_points),
        sigma_v=np.linspace(20.0, 30.0, n_points),
        gas_fraction=gas_fraction,
        age_gyr=age_gyr,
        has_coherent_rotation=has_coherent_rotation,
        metadata=metadata,
    )


class TestCSGV4ModelInitialization:
    """Test CSGV4Model initialization."""

    def test_default_initialization(self):
        """Test model initialization with default config."""
        model = CSGV4Model()
        assert model.config is not None
        assert isinstance(model.config, CSGConfig)

    def test_custom_config_initialization(self):
        """Test model initialization with custom config."""
        custom_config = CSGConfig(alpha_n=0.5, beta_phi=0.7)
        model = CSGV4Model(config=custom_config)
        assert model.config.alpha_n == 0.5
        assert model.config.beta_phi == 0.7

    def test_none_config_uses_default(self):
        """Test that None config uses default."""
        model = CSGV4Model(config=None)
        assert model.config is not None
        default_config = CSGConfig()
        assert model.config.alpha_n == default_config.alpha_n


class TestComputeSalience:
    """Test compute_salience method."""

    def test_salience_basic(self):
        """Test basic salience computation."""
        model = CSGV4Model()
        galaxy = create_test_galaxy()
        salience = model.compute_salience(galaxy)

        # Check that all score keys are present
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
            "CORE",
            "HYPE",
            "PENALTY",
            "S_prime",
            "Q_local",
            "A_galaxy",
            "Q_final",
        }
        assert set(salience.keys()) == expected_keys

    def test_salience_array_shapes(self):
        """Test that salience arrays have correct shapes."""
        model = CSGV4Model()
        galaxy = create_test_galaxy(n_points=15)
        salience = model.compute_salience(galaxy)

        # Check array shapes
        assert salience["S_prime"].shape == (15,)
        assert salience["Q_local"].shape == (15,)
        assert salience["Q_final"].shape == (15,)
        assert salience["CORE"].shape == (15,)
        assert salience["HYPE"].shape == (15,)

        # Check scalar values
        assert isinstance(salience["A_galaxy"], float)
        assert isinstance(salience["R_galaxy"], float)
        assert isinstance(salience["PENALTY"], float)

    def test_salience_core_computation(self):
        """Test CORE computation in salience."""
        model = CSGV4Model()
        galaxy = create_test_galaxy()
        salience = model.compute_salience(galaxy)

        # CORE = (C_local * R_galaxy * K_galaxy) * M_local
        expected_core = (
            salience["C_local"]
            * salience["R_galaxy"]
            * salience["K_galaxy"]
            * salience["M_local"]
        )
        np.testing.assert_allclose(salience["CORE"], expected_core)

    def test_salience_hype_computation(self):
        """Test HYPE computation in salience."""
        model = CSGV4Model()
        galaxy = create_test_galaxy()
        salience = model.compute_salience(galaxy)
        config = model.config

        # HYPE = 1.0 + (alpha_n * DeltaA + gamma_X * X + gamma_Sh * S_h - gamma_W * W + bonus)
        expected_hype = 1.0 + (
            config.alpha_n * salience["DeltaA_local"]
            + config.gamma_X * salience["X_galaxy"]
            + config.gamma_Sh * salience["S_h_galaxy"]
            - config.gamma_W * salience["W_local"]
            + salience["HYPE_workhorse"]
        )
        expected_hype = np.maximum(expected_hype, 0.1)
        np.testing.assert_allclose(salience["HYPE"], expected_hype)

    def test_salience_penalty_computation(self):
        """Test PENALTY computation in salience."""
        model = CSGV4Model()
        galaxy = create_test_galaxy()
        salience = model.compute_salience(galaxy)
        config = model.config

        # PENALTY = 1.0 - beta_phi * phi - beta_J * J
        expected_penalty = (
            1.0
            - config.beta_phi * salience["phi_galaxy"]
            - config.beta_J * salience["J_galaxy"]
        )
        expected_penalty = np.clip(expected_penalty, 0.0, 1.0)
        assert abs(salience["PENALTY"] - expected_penalty) < 1e-10

    def test_salience_s_prime_computation(self):
        """Test S_prime computation in salience."""
        model = CSGV4Model()
        galaxy = create_test_galaxy()
        salience = model.compute_salience(galaxy)
        config = model.config

        # S_prime = CORE * HYPE * PENALTY (with minimum)
        expected_s_prime = salience["CORE"] * salience["HYPE"] * salience["PENALTY"]
        expected_s_prime = np.maximum(expected_s_prime, config.min_s_prime)
        np.testing.assert_allclose(salience["S_prime"], expected_s_prime)

    def test_salience_q_local_computation(self):
        """Test Q_local computation in salience."""
        model = CSGV4Model()
        galaxy = create_test_galaxy()
        salience = model.compute_salience(galaxy)

        # Q_local = S_prime^0.25
        expected_q_local = np.power(salience["S_prime"], 0.25)
        np.testing.assert_allclose(salience["Q_local"], expected_q_local)

    def test_salience_a_galaxy_computation(self):
        """Test A_galaxy computation from inner radii."""
        model = CSGV4Model()
        galaxy = create_test_galaxy(n_points=20)
        salience = model.compute_salience(galaxy)
        config = model.config

        # A_galaxy should be mean of inner Q_local values
        inner_count = max(
            int(np.ceil(galaxy.n_radii * config.inner_radius_fraction)),
            config.min_inner_points,
        )
        inner_q = salience["Q_local"][:inner_count]
        expected_a_galaxy = float(np.clip(np.mean(inner_q), 0.0, 1.0))

        assert abs(salience["A_galaxy"] - expected_a_galaxy) < 1e-10

    def test_salience_q_final_computation(self):
        """Test Q_final computation with aura mixing."""
        model = CSGV4Model()
        galaxy = create_test_galaxy()
        salience = model.compute_salience(galaxy)
        config = model.config

        # Q_final = (1 - aura_mix) * Q_local + aura_mix * A_galaxy
        expected_q_final = (
            1.0 - config.aura_mix
        ) * salience["Q_local"] + config.aura_mix * salience["A_galaxy"]
        expected_q_final = np.maximum(expected_q_final, config.min_q_final)
        np.testing.assert_allclose(salience["Q_final"], expected_q_final)

    def test_salience_bounds(self):
        """Test that salience values are in valid bounds."""
        model = CSGV4Model()
        galaxy = create_test_galaxy()
        salience = model.compute_salience(galaxy)

        # Check A_galaxy bounds
        assert 0.0 <= salience["A_galaxy"] <= 1.0

        # Check array bounds
        assert np.all(salience["Q_local"] >= 0.0)
        assert np.all(salience["Q_final"] >= 0.0)
        assert np.all(salience["S_prime"] >= 0.0)
        assert np.all(salience["HYPE"] >= 0.1)
        assert np.all(salience["PENALTY"] >= 0.0)
        assert np.all(salience["PENALTY"] <= 1.0)

    def test_salience_min_inner_points(self):
        """Test that min_inner_points is respected."""
        model = CSGV4Model()
        # Create galaxy with very few points
        galaxy = create_test_galaxy(n_points=5)
        salience = model.compute_salience(galaxy)

        # Should still compute A_galaxy using at least min_inner_points
        assert isinstance(salience["A_galaxy"], float)
        assert 0.0 <= salience["A_galaxy"] <= 1.0


class TestPredictVelocity:
    """Test predict_velocity method."""

    def test_predict_velocity_basic(self):
        """Test basic velocity prediction."""
        model = CSGV4Model()
        galaxy = create_test_galaxy()
        kappa_c = 0.5
        outputs = model.predict_velocity(galaxy, kappa_c)

        # Check that outputs is ModelOutputs
        assert isinstance(outputs, ModelOutputs)

        # Check that all fields are present
        assert outputs.kappa_c == kappa_c
        assert outputs.scores is not None
        assert outputs.s_prime is not None
        assert outputs.q_local is not None
        assert outputs.a_galaxy is not None
        assert outputs.q_final is not None
        assert outputs.a_eff is not None
        assert outputs.v_pred is not None
        assert outputs.g_obs_pred is not None

    def test_predict_velocity_shapes(self):
        """Test that predicted arrays have correct shapes."""
        model = CSGV4Model()
        galaxy = create_test_galaxy(n_points=12)
        kappa_c = 0.5
        outputs = model.predict_velocity(galaxy, kappa_c)

        # Check array shapes
        assert outputs.s_prime.shape == (12,)
        assert outputs.q_local.shape == (12,)
        assert outputs.q_final.shape == (12,)
        assert outputs.a_eff.shape == (12,)
        assert outputs.v_pred.shape == (12,)
        assert outputs.g_obs_pred.shape == (12,)

        # Check scalar values
        assert isinstance(outputs.kappa_c, float)
        assert isinstance(outputs.a_galaxy, float)

    def test_predict_velocity_a_eff_computation(self):
        """Test a_eff computation."""
        model = CSGV4Model()
        galaxy = create_test_galaxy()
        kappa_c = 0.6
        outputs = model.predict_velocity(galaxy, kappa_c)
        config = model.config

        # a_eff = kappa_c * (a0 / Q_final)
        expected_a_eff = kappa_c * (config.a0 / outputs.q_final)
        np.testing.assert_allclose(outputs.a_eff, expected_a_eff)

    def test_predict_velocity_interpolation_function(self):
        """Test velocity prediction uses correct interpolation function."""
        model = CSGV4Model()
        galaxy = create_test_galaxy()
        kappa_c = 0.5
        outputs = model.predict_velocity(galaxy, kappa_c)
        config = model.config

        # Verify the MOND-like interpolation
        g_bar = galaxy.baryonic_acceleration(config)
        x = g_bar / outputs.a_eff
        mu = x / (1.0 + x)
        expected_g_obs = mu * g_bar + (1.0 - mu) * np.sqrt(outputs.a_eff * g_bar)
        np.testing.assert_allclose(outputs.g_obs_pred, expected_g_obs)

    def test_predict_velocity_v_pred_computation(self):
        """Test v_pred computation from g_obs."""
        model = CSGV4Model()
        galaxy = create_test_galaxy()
        kappa_c = 0.5
        outputs = model.predict_velocity(galaxy, kappa_c)

        # v_pred = sqrt(g_obs * r)
        expected_v_pred = np.sqrt(outputs.g_obs_pred * galaxy.radii_kpc)
        np.testing.assert_allclose(outputs.v_pred, expected_v_pred)

    def test_predict_velocity_different_kappa(self):
        """Test that different kappa values give different predictions."""
        model = CSGV4Model()
        galaxy = create_test_galaxy()

        outputs1 = model.predict_velocity(galaxy, kappa_c=0.3)
        outputs2 = model.predict_velocity(galaxy, kappa_c=0.7)

        # Different kappa should give different predictions
        assert not np.allclose(outputs1.v_pred, outputs2.v_pred)
        assert not np.allclose(outputs1.a_eff, outputs2.a_eff)

    def test_predict_velocity_positive_values(self):
        """Test that predictions are positive."""
        model = CSGV4Model()
        galaxy = create_test_galaxy()
        kappa_c = 0.5
        outputs = model.predict_velocity(galaxy, kappa_c)

        # All physical quantities should be positive
        assert np.all(outputs.a_eff > 0.0)
        assert np.all(outputs.g_obs_pred >= 0.0)
        assert np.all(outputs.v_pred >= 0.0)


class TestModelOutputs:
    """Test ModelOutputs dataclass."""

    def test_model_outputs_creation(self):
        """Test creating a ModelOutputs instance."""
        kappa_c = 0.5
        scores = {"R_galaxy": 0.8}
        s_prime = np.array([0.1, 0.2])
        q_local = np.array([0.5, 0.6])
        a_galaxy = 0.55
        q_final = np.array([0.5, 0.6])
        a_eff = np.array([1.0, 1.1])
        v_pred = np.array([100.0, 110.0])
        g_obs_pred = np.array([5.0, 6.0])

        outputs = ModelOutputs(
            kappa_c=kappa_c,
            scores=scores,
            s_prime=s_prime,
            q_local=q_local,
            a_galaxy=a_galaxy,
            q_final=q_final,
            a_eff=a_eff,
            v_pred=v_pred,
            g_obs_pred=g_obs_pred,
        )

        assert outputs.kappa_c == kappa_c
        assert outputs.scores == scores
        np.testing.assert_array_equal(outputs.s_prime, s_prime)
        np.testing.assert_array_equal(outputs.q_local, q_local)
        assert outputs.a_galaxy == a_galaxy
        np.testing.assert_array_equal(outputs.q_final, q_final)
        np.testing.assert_array_equal(outputs.a_eff, a_eff)
        np.testing.assert_array_equal(outputs.v_pred, v_pred)
        np.testing.assert_array_equal(outputs.g_obs_pred, g_obs_pred)


class TestResiduals:
    """Test residuals static method."""

    def test_residuals_basic(self):
        """Test basic residuals computation."""
        model = CSGV4Model()
        galaxy = create_test_galaxy(n_points=5)
        kappa_c = 0.5
        outputs = model.predict_velocity(galaxy, kappa_c)

        residuals = CSGV4Model.residuals(galaxy, outputs)

        # Check shape
        assert residuals.shape == galaxy.v_obs.shape

    def test_residuals_formula(self):
        """Test residuals formula."""
        galaxy = create_test_galaxy(n_points=5)
        # Set up simple velocities for easy verification
        galaxy.v_obs[:] = 100.0

        model = CSGV4Model()
        outputs = model.predict_velocity(galaxy, kappa_c=0.5)
        residuals = CSGV4Model.residuals(galaxy, outputs)

        # Residuals = (v_pred - v_obs) / (v_obs + epsilon)
        expected_residuals = (outputs.v_pred - galaxy.v_obs) / (galaxy.v_obs + 1.0e-6)
        np.testing.assert_allclose(residuals, expected_residuals)

    def test_residuals_perfect_fit(self):
        """Test residuals when prediction matches observation."""
        galaxy = create_test_galaxy(n_points=5)
        model = CSGV4Model()
        outputs = model.predict_velocity(galaxy, kappa_c=0.5)

        # Artificially set v_pred to match v_obs
        outputs.v_pred[:] = galaxy.v_obs

        residuals = CSGV4Model.residuals(galaxy, outputs)

        # Residuals should be near zero
        np.testing.assert_allclose(residuals, 0.0, atol=1e-6)

    def test_residuals_handles_zero_v_obs(self):
        """Test residuals with zero observed velocity (uses epsilon)."""
        galaxy = create_test_galaxy(n_points=3)
        galaxy.v_obs[0] = 0.0

        model = CSGV4Model()
        outputs = model.predict_velocity(galaxy, kappa_c=0.5)
        residuals = CSGV4Model.residuals(galaxy, outputs)

        # Should not raise error due to epsilon in denominator
        assert np.all(np.isfinite(residuals))


class TestChisq:
    """Test chisq static method."""

    def test_chisq_single_residual_array(self):
        """Test chisq with single residual array."""
        residuals = [np.array([0.1, -0.2, 0.15, -0.1])]
        chisq = CSGV4Model.chisq(residuals)

        # chisq = sum(residuals^2) / count
        expected = np.sum(np.square(residuals[0])) / 4
        assert abs(chisq - expected) < 1e-10

    def test_chisq_multiple_residual_arrays(self):
        """Test chisq with multiple residual arrays."""
        residuals = [
            np.array([0.1, -0.2]),
            np.array([0.15, -0.1, 0.05]),
        ]
        chisq = CSGV4Model.chisq(residuals)

        # chisq = sum(all residuals^2) / total_count
        total_sum = np.sum(np.square(residuals[0])) + np.sum(np.square(residuals[1]))
        total_count = 2 + 3
        expected = total_sum / total_count
        assert abs(chisq - expected) < 1e-10

    def test_chisq_zero_residuals(self):
        """Test chisq with zero residuals."""
        residuals = [np.array([0.0, 0.0, 0.0])]
        chisq = CSGV4Model.chisq(residuals)

        assert chisq == 0.0

    def test_chisq_empty_list(self):
        """Test chisq with empty list (edge case)."""
        residuals = []
        chisq = CSGV4Model.chisq(residuals)

        # Should return 0.0 (accum=0, count=0, result=0/1)
        assert chisq == 0.0

    def test_chisq_perfect_fit(self):
        """Test chisq with perfect fit residuals."""
        residuals = [np.zeros(10)]
        chisq = CSGV4Model.chisq(residuals)

        assert chisq == 0.0

    def test_chisq_positive_value(self):
        """Test that chisq is always non-negative."""
        residuals = [np.array([0.1, -0.2, 0.3, -0.4])]
        chisq = CSGV4Model.chisq(residuals)

        assert chisq >= 0.0


class TestModelIntegration:
    """Integration tests for the full model pipeline."""

    def test_full_pipeline(self):
        """Test complete model pipeline from galaxy to residuals."""
        model = CSGV4Model()
        galaxy = create_test_galaxy(n_points=15)
        kappa_c = 0.5

        # Run full pipeline
        outputs = model.predict_velocity(galaxy, kappa_c)
        residuals = model.residuals(galaxy, outputs)
        chisq = model.chisq([residuals])

        # Verify all outputs are valid
        assert outputs.kappa_c == kappa_c
        assert residuals.shape == galaxy.v_obs.shape
        assert chisq >= 0.0
        assert np.all(np.isfinite(residuals))

    def test_multiple_galaxies_chisq(self):
        """Test chisq computation across multiple galaxies."""
        model = CSGV4Model()
        kappa_c = 0.5

        # Create multiple galaxies
        galaxy1 = create_test_galaxy(galaxy_type="spiral", n_points=10)
        galaxy2 = create_test_galaxy(galaxy_type="dwarf_irregular", n_points=8)

        # Get predictions and residuals
        outputs1 = model.predict_velocity(galaxy1, kappa_c)
        outputs2 = model.predict_velocity(galaxy2, kappa_c)
        residuals1 = model.residuals(galaxy1, outputs1)
        residuals2 = model.residuals(galaxy2, outputs2)

        # Compute combined chisq
        chisq = model.chisq([residuals1, residuals2])

        assert chisq >= 0.0
        assert np.isfinite(chisq)

    def test_different_galaxy_types(self):
        """Test model works with different galaxy types."""
        model = CSGV4Model()
        kappa_c = 0.5

        galaxy_types = [
            "spiral",
            "grand_design",
            "hsb_spiral",
            "lsb_dwarf",
            "dwarf_irregular",
            "elliptical",
        ]

        for gtype in galaxy_types:
            galaxy = create_test_galaxy(galaxy_type=gtype)
            outputs = model.predict_velocity(galaxy, kappa_c)

            # Should produce valid outputs for all types
            assert outputs.v_pred.shape == galaxy.v_obs.shape
            assert np.all(np.isfinite(outputs.v_pred))
            assert np.all(outputs.v_pred >= 0.0)

    def test_reproducibility(self):
        """Test that model produces reproducible results."""
        model1 = CSGV4Model()
        model2 = CSGV4Model()
        galaxy = create_test_galaxy()
        kappa_c = 0.5

        outputs1 = model1.predict_velocity(galaxy, kappa_c)
        outputs2 = model2.predict_velocity(galaxy, kappa_c)

        # Should produce identical results
        np.testing.assert_array_equal(outputs1.v_pred, outputs2.v_pred)
        np.testing.assert_array_equal(outputs1.a_eff, outputs2.a_eff)
