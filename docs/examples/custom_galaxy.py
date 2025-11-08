"""
Custom Galaxy Example for CSG V4

This script demonstrates how to:
1. Create a custom galaxy from scratch
2. Explore different morphological types
3. See how salience scores change with galaxy properties
"""

import numpy as np
import matplotlib.pyplot as plt
from csg_v4 import CSGV4Model, CSGConfig, GalaxyData


def create_custom_galaxy(
    name,
    galaxy_type,
    n_points=20,
    v_max=150,
    scale_length=5.0,
    sigma=10.0,
    gas_fraction=0.3,
    age_gyr=10.0
):
    """
    Create a custom galaxy with specified properties.

    Args:
        name: Galaxy identifier
        galaxy_type: Morphology ("grand_design", "hsb_spiral", "lsb_dwarf", etc.)
        n_points: Number of radial points
        v_max: Asymptotic maximum velocity [km/s]
        scale_length: Exponential scale length [kpc]
        sigma: Velocity dispersion [km/s]
        gas_fraction: Gas mass fraction [0, 1]
        age_gyr: Galaxy age [Gyr]

    Returns:
        GalaxyData instance
    """
    # Radial grid
    radii = np.linspace(0.5, 20.0, n_points)

    # Observed velocity: tanh profile (flattens at large r)
    v_obs = v_max * np.tanh(radii / scale_length)

    # Baryonic velocity: exponentially declining
    v_bar = v_max * 0.9 * np.exp(-radii / (2 * scale_length))

    # Velocity dispersion: can be constant or radius-dependent
    sigma_v = np.full(n_points, sigma)

    galaxy = GalaxyData(
        name=name,
        galaxy_type=galaxy_type,
        radii_kpc=radii,
        v_obs=v_obs,
        v_bar=v_bar,
        sigma_v=sigma_v,
        gas_fraction=gas_fraction,
        age_gyr=age_gyr,
        has_coherent_rotation=True
    )

    return galaxy


def compare_morphologies():
    """Compare salience profiles for different morphological types."""
    print("=" * 70)
    print("Morphology Comparison")
    print("=" * 70)

    # Create galaxies of different types with same basic parameters
    morphologies = [
        ("grand_design", "Grand Design Spiral"),
        ("hsb_spiral", "HSB Spiral"),
        ("spiral", "Regular Spiral"),
        ("lsb_dwarf", "LSB Dwarf"),
        ("dwarf_irregular", "Dwarf Irregular"),
    ]

    model = CSGV4Model()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0, 1, len(morphologies)))

    for idx, (morph_type, morph_label) in enumerate(morphologies):
        print(f"\n[{morph_label}]")

        galaxy = create_custom_galaxy(
            name=morph_label,
            galaxy_type=morph_type,
            v_max=150,
            scale_length=5.0,
            sigma=12.0,
            gas_fraction=0.3,
            age_gyr=10.0
        )

        salience = model.compute_salience(galaxy)

        # Print key scores
        print(f"  R_galaxy: {salience['R_galaxy']:.3f}")
        print(f"  K_galaxy: {salience['K_galaxy']:.3f}")
        print(f"  phi_galaxy: {salience['phi_galaxy']:.3f}")
        print(f"  Mean S': {np.mean(salience['S_prime']):.3f}")
        print(f"  Mean Q_final: {np.mean(salience['Q_final']):.3f}")

        # Plot S' profile
        color = colors[idx]
        axes[0].plot(galaxy.radii_kpc, salience['S_prime'],
                     '-o', color=color, label=morph_label, linewidth=2, markersize=4)

        # Plot Q_final profile
        axes[1].plot(galaxy.radii_kpc, salience['Q_final'],
                     '-o', color=color, label=morph_label, linewidth=2, markersize=4)

        # Plot CORE components
        axes[2].plot(galaxy.radii_kpc, salience['C_local'],
                     '-o', color=color, label=morph_label, linewidth=2, markersize=4)

        # Plot DeltaA
        axes[3].plot(galaxy.radii_kpc, salience['DeltaA_local'],
                     '-o', color=color, label=morph_label, linewidth=2, markersize=4)

    # Configure subplots
    axes[0].set_xlabel('Radius [kpc]', fontsize=11)
    axes[0].set_ylabel("S' (Raw Salience)", fontsize=11)
    axes[0].set_title("Salience Profiles by Morphology", fontsize=12, weight='bold')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Radius [kpc]', fontsize=11)
    axes[1].set_ylabel('Q_final (Quality Factor)', fontsize=11)
    axes[1].set_title("Quality Factor Profiles", fontsize=12, weight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].set_xlabel('Radius [kpc]', fontsize=11)
    axes[2].set_ylabel('C_local (Continuity)', fontsize=11)
    axes[2].set_title("Local Continuity", fontsize=12, weight='bold')
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)

    axes[3].set_xlabel('Radius [kpc]', fontsize=11)
    axes[3].set_ylabel('DeltaA_local (Morphology Boost)', fontsize=11)
    axes[3].set_title("Morphology-Dependent Boost", fontsize=12, weight='bold')
    axes[3].legend(fontsize=9)
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('morphology_comparison.png', dpi=150)
    print("\n  Saved to: morphology_comparison.png")


def parameter_sensitivity():
    """Explore how salience changes with galaxy parameters."""
    print("\n" + "=" * 70)
    print("Parameter Sensitivity Analysis")
    print("=" * 70)

    model = CSGV4Model()

    # Base galaxy
    base = create_custom_galaxy(
        name="Base",
        galaxy_type="hsb_spiral",
        v_max=150,
        scale_length=5.0,
        sigma=12.0,
        gas_fraction=0.3,
        age_gyr=10.0
    )

    # Variations
    variations = {
        "Low sigma (σ=5)": {"sigma": 5.0},
        "High sigma (σ=20)": {"sigma": 20.0},
        "Gas-rich (f_gas=0.7)": {"gas_fraction": 0.7},
        "Young (age=3 Gyr)": {"age_gyr": 3.0},
        "Old (age=13 Gyr)": {"age_gyr": 13.0},
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Plot base
    base_salience = model.compute_salience(base)
    axes[0].plot(base.radii_kpc, base_salience['S_prime'],
                 'k-', linewidth=3, label='Base', zorder=10)
    axes[1].plot(base.radii_kpc, base_salience['Q_final'],
                 'k-', linewidth=3, label='Base', zorder=10)
    axes[2].plot(base.radii_kpc, base_salience['C_local'],
                 'k-', linewidth=3, label='Base', zorder=10)
    axes[3].bar(['R', 'K', 'phi', 'Mean Q'],
                [base_salience['R_galaxy'], base_salience['K_galaxy'],
                 base_salience['phi_galaxy'], np.mean(base_salience['Q_final'])],
                color='black', alpha=0.3, label='Base')

    # Plot variations
    colors = plt.cm.tab10(np.linspace(0, 1, len(variations)))

    print("\nComparing to base:")
    print(f"  Base: R={base_salience['R_galaxy']:.3f}, K={base_salience['K_galaxy']:.3f}, "
          f"mean Q={np.mean(base_salience['Q_final']):.3f}")

    for idx, (label, params) in enumerate(variations.items()):
        # Create variant
        variant_params = {
            "name": label,
            "galaxy_type": "hsb_spiral",
            "v_max": 150,
            "scale_length": 5.0,
            "sigma": 12.0,
            "gas_fraction": 0.3,
            "age_gyr": 10.0
        }
        variant_params.update(params)
        variant = create_custom_galaxy(**variant_params)

        salience = model.compute_salience(variant)

        print(f"  {label}: R={salience['R_galaxy']:.3f}, K={salience['K_galaxy']:.3f}, "
              f"mean Q={np.mean(salience['Q_final']):.3f}")

        color = colors[idx]
        axes[0].plot(variant.radii_kpc, salience['S_prime'],
                     '--', color=color, label=label, linewidth=2)
        axes[1].plot(variant.radii_kpc, salience['Q_final'],
                     '--', color=color, label=label, linewidth=2)
        axes[2].plot(variant.radii_kpc, salience['C_local'],
                     '--', color=color, label=label, linewidth=2)

    axes[0].set_xlabel('Radius [kpc]')
    axes[0].set_ylabel("S'")
    axes[0].set_title("Salience vs. Parameters", weight='bold')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Radius [kpc]')
    axes[1].set_ylabel('Q_final')
    axes[1].set_title("Quality Factor vs. Parameters", weight='bold')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].set_xlabel('Radius [kpc]')
    axes[2].set_ylabel('C_local')
    axes[2].set_title("Continuity vs. Parameters", weight='bold')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    axes[3].set_ylabel('Score Value')
    axes[3].set_title("Global Scores (Base Only)", weight='bold')
    axes[3].legend()
    axes[3].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png', dpi=150)
    print("\n  Saved to: parameter_sensitivity.png")


def main():
    print("\n" + "=" * 70)
    print("CSG V4 Custom Galaxy Examples")
    print("=" * 70)

    # Example 1: Morphology comparison
    compare_morphologies()

    # Example 2: Parameter sensitivity
    parameter_sensitivity()

    print("\n" + "=" * 70)
    print("Analysis complete! Check PNG files for results.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
