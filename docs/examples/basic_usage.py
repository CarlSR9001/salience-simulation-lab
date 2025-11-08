"""
Basic Usage Example for CSG V4

This script demonstrates the fundamental workflow:
1. Create a galaxy
2. Compute salience
3. Predict rotation curve
4. Evaluate fit quality
"""

import numpy as np
import matplotlib.pyplot as plt
from csg_v4 import CSGV4Model, CSGConfig, GalaxyData


def create_example_galaxy():
    """Create a synthetic spiral galaxy for demonstration."""
    # Radial grid from 1 to 20 kpc
    radii = np.linspace(1.0, 20.0, 25)

    # Synthetic observed rotation curve (flattened outer regions)
    v_obs = 200 * np.tanh(radii / 5.0)

    # Baryonic component (exponentially declining)
    v_bar = 180 * np.exp(-radii / 10.0)

    # Velocity dispersion (typical ~10-15 km/s)
    sigma_v = np.full_like(radii, 12.0)

    galaxy = GalaxyData(
        name="ExampleSpiral",
        galaxy_type="hsb_spiral",
        radii_kpc=radii,
        v_obs=v_obs,
        v_bar=v_bar,
        sigma_v=sigma_v,
        gas_fraction=0.3,
        age_gyr=10.0,
        has_coherent_rotation=True
    )

    return galaxy


def main():
    print("=" * 60)
    print("CSG V4 Basic Usage Example")
    print("=" * 60)

    # Step 1: Create galaxy
    print("\n[1] Creating example galaxy...")
    galaxy = create_example_galaxy()
    print(f"    Name: {galaxy.name}")
    print(f"    Type: {galaxy.galaxy_type}")
    print(f"    Radii: {galaxy.n_radii} points from {galaxy.radii_kpc[0]:.1f} to {galaxy.radii_kpc[-1]:.1f} kpc")
    print(f"    Max velocity: {galaxy.vmax():.1f} km/s")

    # Step 2: Initialize model
    print("\n[2] Initializing CSG V4 model...")
    config = CSGConfig()  # Use default parameters
    model = CSGV4Model(config)
    print(f"    Using a0 = {config.a0:.1f} kpc/Gyr²")

    # Step 3: Compute salience
    print("\n[3] Computing salience profile...")
    salience = model.compute_salience(galaxy)

    # Extract key scores
    q_local = salience["Q_local"]
    q_final = salience["Q_final"]
    s_prime = salience["S_prime"]

    print(f"    Mean S': {np.mean(s_prime):.3f}")
    print(f"    Mean Q_local: {np.mean(q_local):.3f}")
    print(f"    Mean Q_final: {np.mean(q_final):.3f}")
    print(f"    Galaxy aura (A): {salience['A_galaxy']:.3f}")

    # Step 4: Predict rotation curve
    print("\n[4] Predicting rotation curve...")
    kappa_c = 0.5  # Trial calibration constant
    outputs = model.predict_velocity(galaxy, kappa_c)

    # Compute residuals
    residuals = model.residuals(galaxy, outputs)
    rms_error = np.sqrt(np.mean((outputs.v_pred - galaxy.v_obs) ** 2))
    mean_frac_error = np.mean(np.abs(residuals)) * 100

    print(f"    Using κ_c = {kappa_c}")
    print(f"    RMS error: {rms_error:.2f} km/s")
    print(f"    Mean fractional error: {mean_frac_error:.2f}%")

    # Step 5: Visualize results
    print("\n[5] Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Rotation curves
    ax = axes[0, 0]
    ax.plot(galaxy.radii_kpc, galaxy.v_obs, 'o-', label='Observed', linewidth=2)
    ax.plot(galaxy.radii_kpc, galaxy.v_bar, 's--', label='Baryonic', linewidth=2)
    ax.plot(galaxy.radii_kpc, outputs.v_pred, '^-', label='CSG V4', linewidth=2)
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Circular Velocity [km/s]')
    ax.set_title('Rotation Curves')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: Salience profile
    ax = axes[0, 1]
    ax.plot(galaxy.radii_kpc, s_prime, '-', label="S' (raw salience)", linewidth=2)
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Salience S\'')
    ax.set_title('Salience Profile')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: Quality factors
    ax = axes[1, 0]
    ax.plot(galaxy.radii_kpc, q_local, '-', label='Q_local', linewidth=2)
    ax.plot(galaxy.radii_kpc, q_final, '-', label='Q_final (blended)', linewidth=2)
    ax.axhline(salience['A_galaxy'], color='gray', linestyle='--',
               label=f'A_galaxy = {salience["A_galaxy"]:.3f}')
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Quality Factor Q')
    ax.set_title('Quality Factor Transformation')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 4: Fractional residuals
    ax = axes[1, 1]
    ax.plot(galaxy.radii_kpc, residuals * 100, 'o-', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Fractional Residual [%]')
    ax.set_title(f'Residuals (mean |Δv/v| = {mean_frac_error:.2f}%)')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('csg_v4_basic_example.png', dpi=150)
    print("    Saved to: csg_v4_basic_example.png")

    # Step 6: Explore scoring components
    print("\n[6] Exploring scoring components...")
    print(f"    R_galaxy (retention): {salience['R_galaxy']:.3f}")
    print(f"    K_galaxy (coherence): {salience['K_galaxy']:.3f}")
    print(f"    phi_galaxy (disorder): {salience['phi_galaxy']:.3f}")
    print(f"    Mean C_local (continuity): {np.mean(salience['C_local']):.3f}")
    print(f"    Mean M_local (mass): {np.mean(salience['M_local']):.3f}")
    print(f"    Mean W_local (curvature): {np.mean(salience['W_local']):.3f}")

    print("\n" + "=" * 60)
    print("Example complete! Check csg_v4_basic_example.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
