"""
Kappa Optimization Example for CSG V4

This script demonstrates how to:
1. Load multiple galaxies
2. Scan κ_c parameter space
3. Find optimal fit
4. Refine with advanced methods
5. Estimate uncertainty via bootstrap
"""

import numpy as np
import matplotlib.pyplot as plt
from csg_v4 import CSGV4Model, CSGConfig
from csg_v4.synthetic_data import load_synthetic_galaxies
from csg_v4.optimizer import scan_kappa, refine_kappa, bootstrap_kappa


def main():
    print("=" * 70)
    print("CSG V4 Kappa Optimization Example")
    print("=" * 70)

    # Step 1: Load galaxies
    print("\n[1] Loading synthetic galaxy sample...")
    galaxies_dict = load_synthetic_galaxies()
    galaxies = list(galaxies_dict.values())

    print(f"    Loaded {len(galaxies)} galaxies:")
    for gal in galaxies:
        print(f"      - {gal.name}: {gal.galaxy_type}, {gal.n_radii} points, v_max = {gal.vmax():.1f} km/s")

    # Step 2: Initialize model
    print("\n[2] Initializing model...")
    config = CSGConfig()
    model = CSGV4Model(config)
    print(f"    Kappa grid: {config.kappa_samples} points from {config.kappa_min} to {config.kappa_max}")

    # Step 3: Perform grid scan
    print("\n[3] Performing kappa scan (this may take a moment)...")
    scan_result, residual_map = scan_kappa(
        galaxies,
        model=model,
        store_profiles=True  # Needed for bootstrap
    )

    print(f"    Best κ_c: {scan_result.best_kappa:.6f}")
    print(f"    Best χ²: {scan_result.best_chisq:.6e}")
    print(f"    Located at index {scan_result.best_index} / {len(scan_result.kappa_values)}")

    # Step 4: Refine estimate
    print("\n[4] Refining estimate...")
    refinement = refine_kappa(galaxies, model, scan_result, config)

    if refinement.quadratic is not None:
        q = refinement.quadratic
        print(f"    Quadratic fit:")
        print(f"      κ_c = {q.kappa:.6f} ± {q.sigma:.6f}")
        print(f"      χ² = {q.chisq:.6e}")
        print(f"      Curvature = {q.curvature:.3e}")

    if refinement.golden is not None:
        g = refinement.golden
        print(f"    Golden section search:")
        print(f"      κ_c = {g.kappa:.6f}")
        print(f"      χ² = {g.chisq:.6e}")
        print(f"      Iterations = {g.iterations}")
        print(f"      Final bracket = [{g.bracket[0]:.6f}, {g.bracket[1]:.6f}]")

    if refinement.zoom_scan is not None:
        z = refinement.zoom_scan
        print(f"    Zoom scan:")
        print(f"      κ_c = {z.best_kappa:.6f}")
        print(f"      χ² = {z.best_chisq:.6e}")
        print(f"      Grid points = {len(z.kappa_values)}")

    # Step 5: Bootstrap uncertainty
    print("\n[5] Estimating uncertainty via bootstrap...")
    print(f"    Drawing {config.bootstrap_samples} bootstrap samples...")
    bootstrap_estimates = bootstrap_kappa(scan_result, config)

    bs_mean = np.mean(bootstrap_estimates)
    bs_std = np.std(bootstrap_estimates)
    bs_median = np.median(bootstrap_estimates)
    bs_q16, bs_q84 = np.percentile(bootstrap_estimates, [16, 84])

    print(f"    Bootstrap statistics:")
    print(f"      Mean: {bs_mean:.6f}")
    print(f"      Std: {bs_std:.6f}")
    print(f"      Median: {bs_median:.6f}")
    print(f"      68% CI: [{bs_q16:.6f}, {bs_q84:.6f}]")

    # Step 6: Visualize results
    print("\n[6] Creating visualizations...")

    # Figure 1: Kappa scan
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(scan_result.kappa_values, scan_result.chisq_values, '-', linewidth=2)
    ax.scatter([scan_result.best_kappa], [scan_result.best_chisq],
               color='red', s=100, zorder=10, label=f'Best κ_c = {scan_result.best_kappa:.4f}')
    if refinement.quadratic is not None:
        ax.scatter([refinement.quadratic.kappa], [refinement.quadratic.chisq],
                   color='orange', s=80, marker='s', zorder=9, label='Quadratic')
    ax.set_xlabel('κ_c', fontsize=12)
    ax.set_ylabel('χ² (mean squared fractional residual)', fontsize=12)
    ax.set_title('Kappa Scan Results', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Figure 2: Bootstrap distribution
    ax = axes[1]
    ax.hist(bootstrap_estimates, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(bs_mean, color='red', linestyle='--', linewidth=2, label=f'Mean = {bs_mean:.4f}')
    ax.axvline(bs_median, color='orange', linestyle='--', linewidth=2, label=f'Median = {bs_median:.4f}')
    ax.axvline(bs_q16, color='gray', linestyle=':', linewidth=1.5, label='68% CI')
    ax.axvline(bs_q84, color='gray', linestyle=':', linewidth=1.5)
    ax.set_xlabel('κ_c', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Bootstrap Distribution (σ = {bs_std:.4f})', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('csg_v4_kappa_optimization.png', dpi=150)
    print("    Saved to: csg_v4_kappa_optimization.png")

    # Figure 3: Per-galaxy fits
    fig, axes = plt.subplots(len(galaxies), 1, figsize=(10, 4 * len(galaxies)))
    if len(galaxies) == 1:
        axes = [axes]

    best_kappa = scan_result.best_kappa
    for idx, (galaxy, ax) in enumerate(zip(galaxies, axes)):
        outputs = model.predict_velocity(galaxy, best_kappa)

        ax.plot(galaxy.radii_kpc, galaxy.v_obs, 'o-', label='Observed', linewidth=2)
        ax.plot(galaxy.radii_kpc, galaxy.v_bar, 's--', label='Baryonic', linewidth=2)
        ax.plot(galaxy.radii_kpc, outputs.v_pred, '^-', label='CSG V4', linewidth=2)

        rms = np.sqrt(np.mean((outputs.v_pred - galaxy.v_obs) ** 2))
        ax.set_xlabel('Radius [kpc]')
        ax.set_ylabel('Circular Velocity [km/s]')
        ax.set_title(f'{galaxy.name} (RMS = {rms:.2f} km/s)', fontsize=12, weight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('csg_v4_galaxy_fits.png', dpi=150)
    print("    Saved to: csg_v4_galaxy_fits.png")

    # Step 7: Summary
    print("\n[7] Summary:")
    print(f"    Primary scan:     κ_c = {scan_result.best_kappa:.6f}, χ² = {scan_result.best_chisq:.6e}")
    if refinement.quadratic is not None:
        print(f"    Quadratic fit:    κ_c = {refinement.quadratic.kappa:.6f} ± {refinement.quadratic.sigma:.6f}")
    print(f"    Bootstrap:        κ_c = {bs_mean:.6f} ± {bs_std:.6f}")
    print(f"    Recommendation:   Use κ_c = {bs_median:.4f} (bootstrap median)")

    print("\n" + "=" * 70)
    print("Optimization complete! Check output PNG files for visualizations.")
    print("=" * 70)


if __name__ == "__main__":
    main()
