"""Experiment U: Cluster-Scale Salience Coupling.

Extends SPARC fits to galaxy clusters with inter-galaxy continuity constraints.
Tests whether shared continuity parameters (κ_c) across cluster members improve
fits beyond independent models, which would suggest large-scale salience effects.

Architecture:
- Load a subset of SPARC galaxies to simulate cluster membership
- Fit independently: each galaxy gets its own κ_c
- Fit jointly with shared κ_c: all galaxies share a single κ_c
- Fit jointly with continuity coupling: shared κ_c + penalty for salience drift
- Compare chi-square, per-galaxy residuals, and κ_c variance

Success metric: Discover anomalous κ_c correlations or improved fits beyond
independent models, suggesting large-scale continuity effects across clusters.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

ARTIFACT_DIR = Path("artifacts/cluster_fit")

# Import CSG V4 modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from csg_v4.config import CSGConfig
from csg_v4.data_io import SPARCParser
from csg_v4.galaxy import GalaxyData
from csg_v4.model import CSGV4Model


@dataclass
class ClusterFitResult:
    """Results from a cluster-scale fitting regime."""
    regime: str
    kappa_values: List[float]
    kappa_mean: float
    kappa_std: float
    kappa_variance: float
    total_chisq: float
    per_galaxy_chisq: List[float]
    per_galaxy_residuals: List[np.ndarray]
    continuity_penalty: float
    energy_accounting: Dict


def load_cluster_galaxies(
    max_count: int = 8,
    quality_filter: int = 1,
    seed: int = 42,
) -> List[GalaxyData]:
    """Load a subset of SPARC galaxies to simulate a cluster.

    Args:
        max_count: Maximum number of galaxies to include.
        quality_filter: SPARC quality flag threshold (1=best).
        seed: Random seed for reproducible selection.

    Returns:
        List of GalaxyData objects.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "sparc"
    table1 = data_dir / "table1.dat"
    table2 = data_dir / "table2.dat"

    if not table1.exists() or not table2.exists():
        raise FileNotFoundError(
            f"SPARC data not found. Expected:\n"
            f"  {table1}\n"
            f"  {table2}\n"
            f"Please ensure SPARC dataset is downloaded."
        )

    sparc_parser = SPARCParser(table1, table2)
    config = CSGConfig()

    # Load all galaxies
    galaxies_dict = sparc_parser.load_galaxies(
        selection=None,
        config=config,
        stellar_ml=0.5,
        bulge_ml=0.7,
    )

    # Filter by quality
    metadata_table = sparc_parser.table1
    filtered = []
    for galaxy in galaxies_dict.values():
        if galaxy.name in metadata_table.index:
            quality = int(metadata_table.loc[galaxy.name, "Q"])
            if quality <= quality_filter:
                filtered.append(galaxy)

    # Randomly sample to simulate cluster membership
    rng = np.random.default_rng(seed)
    if len(filtered) > max_count:
        indices = rng.choice(len(filtered), size=max_count, replace=False)
        filtered = [filtered[i] for i in indices]

    return filtered


def compute_chisq_single(
    galaxy: GalaxyData,
    model: CSGV4Model,
    kappa: float,
) -> Tuple[float, np.ndarray]:
    """Compute chi-square and residuals for a single galaxy."""
    outputs = model.predict_velocity(galaxy, kappa)
    residuals = model.residuals(galaxy, outputs)
    chisq = float(np.sum(residuals**2 / (galaxy.sigma_v**2 + 1e-6)))
    return chisq, residuals


def compute_chisq_cluster(
    galaxies: List[GalaxyData],
    model: CSGV4Model,
    kappa_values: List[float],
) -> Tuple[float, List[float], List[np.ndarray]]:
    """Compute total chi-square for cluster with per-galaxy kappas."""
    total_chisq = 0.0
    per_galaxy_chisq = []
    per_galaxy_residuals = []

    for galaxy, kappa in zip(galaxies, kappa_values):
        chisq, residuals = compute_chisq_single(galaxy, model, kappa)
        total_chisq += chisq
        per_galaxy_chisq.append(chisq)
        per_galaxy_residuals.append(residuals)

    return total_chisq, per_galaxy_chisq, per_galaxy_residuals


def continuity_penalty(kappa_values: List[float], strength: float = 1.0) -> float:
    """Compute inter-galaxy continuity penalty.

    Penalizes variance in κ_c across cluster members, encouraging coherent
    continuity parameters that would indicate large-scale salience coupling.

    Args:
        kappa_values: List of κ_c values for each galaxy.
        strength: Penalty coefficient.

    Returns:
        Penalty value (higher = more variance).
    """
    if len(kappa_values) <= 1:
        return 0.0

    kappa_array = np.array(kappa_values)
    mean_kappa = float(np.mean(kappa_array))
    variance = float(np.var(kappa_array))

    # Penalize deviations from mean
    penalty = strength * variance * len(kappa_values)

    return penalty


def fit_independent(
    galaxies: List[GalaxyData],
    model: CSGV4Model,
    kappa_init: float = 0.5,
) -> ClusterFitResult:
    """Fit each galaxy independently with its own κ_c.

    Args:
        galaxies: List of galaxies to fit.
        model: CSG V4 model.
        kappa_init: Initial guess for κ_c.

    Returns:
        ClusterFitResult with independent κ_c values.
    """
    kappa_values = []
    per_galaxy_chisq = []
    per_galaxy_residuals = []

    for galaxy in galaxies:
        # Optimize κ_c for this galaxy
        def objective(kappa):
            chisq, _ = compute_chisq_single(galaxy, model, kappa[0])
            return chisq

        result = minimize(
            objective,
            x0=[kappa_init],
            bounds=[(0.01, 2.0)],
            method="L-BFGS-B",
        )

        best_kappa = float(result.x[0])
        chisq, residuals = compute_chisq_single(galaxy, model, best_kappa)

        kappa_values.append(best_kappa)
        per_galaxy_chisq.append(chisq)
        per_galaxy_residuals.append(residuals)

    total_chisq = float(np.sum(per_galaxy_chisq))
    kappa_mean = float(np.mean(kappa_values))
    kappa_std = float(np.std(kappa_values))
    kappa_variance = float(np.var(kappa_values))

    # Energy accounting: measure spread
    energy_dict = {
        "regime": "independent",
        "total_optimization_calls": len(galaxies),
        "kappa_range": float(np.max(kappa_values) - np.min(kappa_values)),
        "kappa_cv": kappa_std / (kappa_mean + 1e-9),
    }

    return ClusterFitResult(
        regime="independent",
        kappa_values=kappa_values,
        kappa_mean=kappa_mean,
        kappa_std=kappa_std,
        kappa_variance=kappa_variance,
        total_chisq=total_chisq,
        per_galaxy_chisq=per_galaxy_chisq,
        per_galaxy_residuals=per_galaxy_residuals,
        continuity_penalty=0.0,
        energy_accounting=energy_dict,
    )


def fit_shared(
    galaxies: List[GalaxyData],
    model: CSGV4Model,
    kappa_init: float = 0.5,
) -> ClusterFitResult:
    """Fit all galaxies with a single shared κ_c.

    Args:
        galaxies: List of galaxies to fit.
        model: CSG V4 model.
        kappa_init: Initial guess for shared κ_c.

    Returns:
        ClusterFitResult with shared κ_c.
    """
    def objective(kappa):
        kappa_val = float(kappa[0])
        kappa_list = [kappa_val] * len(galaxies)
        total_chisq, _, _ = compute_chisq_cluster(galaxies, model, kappa_list)
        return total_chisq

    result = minimize(
        objective,
        x0=[kappa_init],
        bounds=[(0.01, 2.0)],
        method="L-BFGS-B",
    )

    best_kappa = float(result.x[0])
    kappa_list = [best_kappa] * len(galaxies)
    total_chisq, per_galaxy_chisq, per_galaxy_residuals = compute_chisq_cluster(
        galaxies, model, kappa_list
    )

    energy_dict = {
        "regime": "shared",
        "total_optimization_calls": 1,
        "kappa_range": 0.0,
        "kappa_cv": 0.0,
    }

    return ClusterFitResult(
        regime="shared",
        kappa_values=kappa_list,
        kappa_mean=best_kappa,
        kappa_std=0.0,
        kappa_variance=0.0,
        total_chisq=total_chisq,
        per_galaxy_chisq=per_galaxy_chisq,
        per_galaxy_residuals=per_galaxy_residuals,
        continuity_penalty=0.0,
        energy_accounting=energy_dict,
    )


def fit_coupled(
    galaxies: List[GalaxyData],
    model: CSGV4Model,
    kappa_init: float = 0.5,
    coupling_strength: float = 10.0,
) -> ClusterFitResult:
    """Fit with inter-galaxy continuity coupling.

    Each galaxy has its own κ_c, but they're coupled via a continuity penalty
    that encourages coherence across the cluster.

    Args:
        galaxies: List of galaxies to fit.
        model: CSG V4 model.
        kappa_init: Initial guess for κ_c values.
        coupling_strength: Strength of continuity penalty.

    Returns:
        ClusterFitResult with coupled κ_c values.
    """
    n_galaxies = len(galaxies)

    def objective(kappa_array):
        kappa_list = [float(k) for k in kappa_array]
        total_chisq, _, _ = compute_chisq_cluster(galaxies, model, kappa_list)
        penalty = continuity_penalty(kappa_list, strength=coupling_strength)
        return total_chisq + penalty

    result = minimize(
        objective,
        x0=[kappa_init] * n_galaxies,
        bounds=[(0.01, 2.0)] * n_galaxies,
        method="L-BFGS-B",
    )

    kappa_values = [float(k) for k in result.x]
    total_chisq_raw, per_galaxy_chisq, per_galaxy_residuals = compute_chisq_cluster(
        galaxies, model, kappa_values
    )
    penalty = continuity_penalty(kappa_values, strength=coupling_strength)
    total_chisq = total_chisq_raw + penalty

    kappa_mean = float(np.mean(kappa_values))
    kappa_std = float(np.std(kappa_values))
    kappa_variance = float(np.var(kappa_values))

    energy_dict = {
        "regime": "coupled",
        "total_optimization_calls": 1,
        "kappa_range": float(np.max(kappa_values) - np.min(kappa_values)),
        "kappa_cv": kappa_std / (kappa_mean + 1e-9),
        "coupling_strength": coupling_strength,
    }

    return ClusterFitResult(
        regime="coupled",
        kappa_values=kappa_values,
        kappa_mean=kappa_mean,
        kappa_std=kappa_std,
        kappa_variance=kappa_variance,
        total_chisq=total_chisq,
        per_galaxy_chisq=per_galaxy_chisq,
        per_galaxy_residuals=per_galaxy_residuals,
        continuity_penalty=penalty,
        energy_accounting=energy_dict,
    )


def compare_regimes(
    independent: ClusterFitResult,
    shared: ClusterFitResult,
    coupled: ClusterFitResult,
) -> Dict:
    """Compare fitting regimes and identify anomalies."""
    # Chi-square improvements
    chisq_indep = independent.total_chisq
    chisq_shared = shared.total_chisq
    chisq_coupled = coupled.total_chisq

    # Relative improvements
    shared_vs_indep = (chisq_indep - chisq_shared) / chisq_indep
    coupled_vs_indep = (chisq_indep - chisq_coupled) / chisq_indep
    coupled_vs_shared = (chisq_shared - chisq_coupled) / chisq_shared

    # Kappa correlation analysis
    kappa_correlation = float(np.corrcoef(
        independent.kappa_values,
        coupled.kappa_values,
    )[0, 1])

    # Variance reduction
    variance_reduction = (
        independent.kappa_variance - coupled.kappa_variance
    ) / (independent.kappa_variance + 1e-9)

    # Anomaly flags
    anomalous_correlation = abs(kappa_correlation) > 0.8
    anomalous_variance_reduction = variance_reduction > 0.5
    improved_fit = coupled_vs_indep > 0.05

    comparison = {
        "chisq_independent": chisq_indep,
        "chisq_shared": chisq_shared,
        "chisq_coupled": chisq_coupled,
        "improvement_shared_vs_indep": shared_vs_indep,
        "improvement_coupled_vs_indep": coupled_vs_indep,
        "improvement_coupled_vs_shared": coupled_vs_shared,
        "kappa_correlation_indep_coupled": kappa_correlation,
        "variance_reduction_coupled": variance_reduction,
        "kappa_mean_independent": independent.kappa_mean,
        "kappa_mean_shared": shared.kappa_mean,
        "kappa_mean_coupled": coupled.kappa_mean,
        "kappa_std_independent": independent.kappa_std,
        "kappa_std_coupled": coupled.kappa_std,
        "anomalous_correlation": anomalous_correlation,
        "anomalous_variance_reduction": anomalous_variance_reduction,
        "improved_fit": improved_fit,
    }

    return comparison


def run_experiment(
    n_galaxies: int = 8,
    coupling_strength: float = 10.0,
    seed: int = 42,
) -> Dict:
    """Run full cluster-scale salience coupling experiment."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment U: Cluster-Scale Salience Coupling")
    print("=" * 70)

    # Load cluster galaxies
    print(f"\nLoading {n_galaxies} SPARC galaxies (quality=1)...")
    galaxies = load_cluster_galaxies(max_count=n_galaxies, quality_filter=1, seed=seed)
    galaxy_names = [g.name for g in galaxies]
    print(f"Loaded cluster: {', '.join(galaxy_names)}")

    # Initialize model
    config = CSGConfig()
    model = CSGV4Model(config)

    # Fit regimes
    print("\nFitting regime 1: Independent κ_c per galaxy...")
    independent = fit_independent(galaxies, model, kappa_init=0.5)
    print(f"  κ_c range: [{min(independent.kappa_values):.4f}, {max(independent.kappa_values):.4f}]")
    print(f"  Total χ²: {independent.total_chisq:.2f}")

    print("\nFitting regime 2: Shared κ_c across cluster...")
    shared = fit_shared(galaxies, model, kappa_init=0.5)
    print(f"  Shared κ_c: {shared.kappa_mean:.4f}")
    print(f"  Total χ²: {shared.total_chisq:.2f}")

    print(f"\nFitting regime 3: Coupled κ_c with continuity penalty (strength={coupling_strength})...")
    coupled = fit_coupled(galaxies, model, kappa_init=0.5, coupling_strength=coupling_strength)
    print(f"  κ_c range: [{min(coupled.kappa_values):.4f}, {max(coupled.kappa_values):.4f}]")
    print(f"  Continuity penalty: {coupled.continuity_penalty:.2f}")
    print(f"  Total χ² (with penalty): {coupled.total_chisq:.2f}")

    # Compare regimes
    print("\nComparing regimes...")
    comparison = compare_regimes(independent, shared, coupled)

    payload = {
        "n_galaxies": n_galaxies,
        "galaxy_names": galaxy_names,
        "coupling_strength": coupling_strength,
        "seed": seed,
        "independent": {
            "kappa_values": independent.kappa_values,
            "kappa_mean": independent.kappa_mean,
            "kappa_std": independent.kappa_std,
            "total_chisq": independent.total_chisq,
            "per_galaxy_chisq": independent.per_galaxy_chisq,
        },
        "shared": {
            "kappa_value": shared.kappa_mean,
            "total_chisq": shared.total_chisq,
            "per_galaxy_chisq": shared.per_galaxy_chisq,
        },
        "coupled": {
            "kappa_values": coupled.kappa_values,
            "kappa_mean": coupled.kappa_mean,
            "kappa_std": coupled.kappa_std,
            "total_chisq": coupled.total_chisq,
            "continuity_penalty": coupled.continuity_penalty,
            "per_galaxy_chisq": coupled.per_galaxy_chisq,
        },
        "comparison": comparison,
    }

    return payload


def write_artifact(payload: Dict) -> Path:
    """Write experiment results to artifact file."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())

    record = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "experiment_name": "experiment_u_cluster_coupling",
        "run_id": run_id,
        **payload,
    }

    path = ARTIFACT_DIR / f"cluster_fit_{timestamp}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def run_coupling_sweep(
    n_galaxies: int = 8,
    coupling_strengths: List[float] = None,
    seed: int = 42,
) -> Dict:
    """Run coupling strength sweep to find optimal balance.

    Args:
        n_galaxies: Number of galaxies in cluster.
        coupling_strengths: List of coupling strengths to test.
        seed: Random seed for galaxy selection.

    Returns:
        Dictionary with sweep results and metrics.
    """
    if coupling_strengths is None:
        coupling_strengths = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment U: Coupling Strength Sweep")
    print("=" * 70)

    # Load cluster galaxies once
    print(f"\nLoading {n_galaxies} SPARC galaxies (quality=1)...")
    galaxies = load_cluster_galaxies(max_count=n_galaxies, quality_filter=1, seed=seed)
    galaxy_names = [g.name for g in galaxies]
    print(f"Loaded cluster: {', '.join(galaxy_names)}")

    # Initialize model
    config = CSGConfig()
    model = CSGV4Model(config)

    # Fit independent regime once (baseline)
    print("\nFitting independent regime (baseline)...")
    independent = fit_independent(galaxies, model, kappa_init=0.5)
    chisq_indep = independent.total_chisq
    variance_indep = independent.kappa_variance
    print(f"  Independent χ²: {chisq_indep:.2f}")
    print(f"  Independent κ_c variance: {variance_indep:.6f}")

    # Sweep coupling strengths
    sweep_results = []

    for strength in coupling_strengths:
        print(f"\nTesting coupling strength = {strength:.1f}...")

        if strength == 0.0:
            # Zero coupling is equivalent to independent
            coupled = independent
        else:
            coupled = fit_coupled(galaxies, model, kappa_init=0.5, coupling_strength=strength)

        # Calculate metrics vs independent fit
        # NOTE: Use raw chi-square WITHOUT penalty for fair comparison
        chisq_coupled_raw, _, _ = compute_chisq_cluster(galaxies, model, coupled.kappa_values)
        chisq_ratio = chisq_coupled_raw / chisq_indep

        # Variance reduction
        variance_coupled = coupled.kappa_variance
        variance_reduction = (variance_indep - variance_coupled) / (variance_indep + 1e-9)

        # Check if coupling improves fit
        improves_fit = chisq_ratio < 1.0

        result = {
            "coupling_strength": strength,
            "chisq_independent": chisq_indep,
            "chisq_coupled_raw": chisq_coupled_raw,
            "chisq_coupled_with_penalty": coupled.total_chisq,
            "chisq_ratio": chisq_ratio,
            "kappa_variance_independent": variance_indep,
            "kappa_variance_coupled": variance_coupled,
            "variance_reduction": variance_reduction,
            "improves_fit": improves_fit,
            "kappa_values": coupled.kappa_values,
            "kappa_mean": coupled.kappa_mean,
            "kappa_std": coupled.kappa_std,
            "continuity_penalty": coupled.continuity_penalty,
        }

        sweep_results.append(result)

        print(f"  χ² ratio: {chisq_ratio:.4f} {'✓ IMPROVED' if improves_fit else ''}")
        print(f"  Variance reduction: {variance_reduction:.2%}")

    # Identify sweet spots
    sweet_spots = []
    optimal = None
    best_chisq_ratio = float('inf')

    for result in sweep_results:
        # Sweet spot criteria: 30-70% variance reduction + χ² ratio ≤ 1.0
        var_red = result["variance_reduction"]
        chisq_r = result["chisq_ratio"]

        is_sweet_spot = (0.30 <= var_red <= 0.70) and (chisq_r <= 1.0)
        result["is_sweet_spot"] = is_sweet_spot

        if is_sweet_spot:
            sweet_spots.append(result)

        # Track optimal (best fit quality with coherence)
        if chisq_r < best_chisq_ratio and var_red > 0.10:  # At least 10% coherence gain
            best_chisq_ratio = chisq_r
            optimal = result

    # Find threshold where forcing coherence becomes harmful
    harm_threshold = None
    for result in sweep_results:
        if result["chisq_ratio"] > 1.05:  # 5% degradation threshold
            if harm_threshold is None or result["coupling_strength"] < harm_threshold:
                harm_threshold = result["coupling_strength"]

    payload = {
        "n_galaxies": n_galaxies,
        "galaxy_names": galaxy_names,
        "seed": seed,
        "coupling_strengths": coupling_strengths,
        "independent_baseline": {
            "chisq": chisq_indep,
            "kappa_variance": variance_indep,
            "kappa_values": independent.kappa_values,
            "kappa_mean": independent.kappa_mean,
            "kappa_std": independent.kappa_std,
        },
        "sweep_results": sweep_results,
        "sweet_spots": sweet_spots,
        "optimal_coupling": optimal,
        "harm_threshold": harm_threshold,
    }

    return payload


def plot_coupling_sweep(payload: Dict, timestamp: str) -> List[Path]:
    """Generate plots for coupling strength sweep.

    Args:
        payload: Sweep results dictionary.
        timestamp: Timestamp string for filenames.

    Returns:
        List of paths to generated plots.
    """
    results = payload["sweep_results"]

    coupling_strengths = [r["coupling_strength"] for r in results]
    chisq_ratios = [r["chisq_ratio"] for r in results]
    variance_reductions = [r["variance_reduction"] * 100 for r in results]  # Convert to %
    improves_fit = [r["improves_fit"] for r in results]

    plot_paths = []

    # Plot 1: χ² ratio vs coupling strength
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color points by whether they improve fit
    colors = ['green' if imp else 'red' for imp in improves_fit]
    ax.scatter(coupling_strengths, chisq_ratios, c=colors, s=100, alpha=0.7, edgecolors='black')
    ax.plot(coupling_strengths, chisq_ratios, 'k--', alpha=0.3)

    # Reference line at χ² ratio = 1.0
    ax.axhline(y=1.0, color='blue', linestyle='--', linewidth=2, label='χ² ratio = 1.0 (no degradation)')

    ax.set_xlabel('Coupling Strength', fontsize=12, fontweight='bold')
    ax.set_ylabel('χ² Ratio (coupled / independent)', fontsize=12, fontweight='bold')
    ax.set_title('Fit Quality vs Coupling Strength', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add annotations for interesting points
    if payload["optimal_coupling"]:
        opt = payload["optimal_coupling"]
        ax.annotate('Optimal',
                   xy=(opt["coupling_strength"], opt["chisq_ratio"]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    path1 = ARTIFACT_DIR / f"coupling_sweep_{timestamp}_chisq_ratio.png"
    plt.savefig(path1, dpi=150)
    plt.close()
    plot_paths.append(path1)

    # Plot 2: Variance reduction vs coupling strength
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(coupling_strengths, variance_reductions, c=colors, s=100, alpha=0.7, edgecolors='black')
    ax.plot(coupling_strengths, variance_reductions, 'k--', alpha=0.3)

    # Sweet spot range
    ax.axhspan(30, 70, alpha=0.2, color='green', label='Sweet spot range (30-70%)')

    ax.set_xlabel('Coupling Strength', fontsize=12, fontweight='bold')
    ax.set_ylabel('κ_c Variance Reduction (%)', fontsize=12, fontweight='bold')
    ax.set_title('Coherence vs Coupling Strength', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    path2 = ARTIFACT_DIR / f"coupling_sweep_{timestamp}_variance_reduction.png"
    plt.savefig(path2, dpi=150)
    plt.close()
    plot_paths.append(path2)

    # Plot 3: Pareto frontier (coherence vs fit quality)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot with coupling strength as color
    scatter = ax.scatter(variance_reductions, chisq_ratios,
                        c=np.log10(np.array(coupling_strengths) + 0.01),  # +0.01 for log(0)
                        s=150, alpha=0.7, cmap='viridis', edgecolors='black')

    # Annotate each point with coupling strength
    for i, strength in enumerate(coupling_strengths):
        ax.annotate(f'{strength:.1f}',
                   xy=(variance_reductions[i], chisq_ratios[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)

    # Reference lines
    ax.axhline(y=1.0, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='χ² ratio = 1.0')
    ax.axvspan(30, 70, alpha=0.1, color='green', label='Sweet spot variance reduction')

    # Highlight sweet spots
    sweet_spot_indices = [i for i, r in enumerate(results) if r.get("is_sweet_spot", False)]
    if sweet_spot_indices:
        sweet_var = [variance_reductions[i] for i in sweet_spot_indices]
        sweet_chi = [chisq_ratios[i] for i in sweet_spot_indices]
        ax.scatter(sweet_var, sweet_chi, s=300, facecolors='none',
                  edgecolors='lime', linewidths=3, label='Sweet spots')

    # Highlight optimal
    if payload["optimal_coupling"]:
        opt = payload["optimal_coupling"]
        ax.scatter([opt["variance_reduction"] * 100], [opt["chisq_ratio"]],
                  s=400, marker='*', c='gold', edgecolors='black',
                  linewidths=2, label='Optimal', zorder=10)

    ax.set_xlabel('κ_c Variance Reduction (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('χ² Ratio (coupled / independent)', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Frontier: Coherence vs Fit Quality', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log₁₀(Coupling Strength)', fontsize=10)

    plt.tight_layout()
    path3 = ARTIFACT_DIR / f"coupling_sweep_{timestamp}_pareto.png"
    plt.savefig(path3, dpi=150)
    plt.close()
    plot_paths.append(path3)

    return plot_paths


def main() -> None:
    """Main entry point - run coupling strength sweep."""
    print("\nRunning coupling strength sweep...")

    payload = run_coupling_sweep(
        n_galaxies=8,
        coupling_strengths=[0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        seed=42,
    )

    # Save results
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())

    record = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "experiment_name": "experiment_u_coupling_sweep",
        "run_id": run_id,
        **payload,
    }

    artifact_path = ARTIFACT_DIR / f"coupling_sweep_{timestamp}.json"
    artifact_path.write_text(json.dumps(record, indent=2))

    # Generate plots
    plot_paths = plot_coupling_sweep(payload, timestamp)

    # Print summary
    print("\n" + "=" * 70)
    print("Coupling Strength Sweep Results")
    print("=" * 70)

    print(f"\nBaseline (Independent Fit):")
    print(f"  χ²: {payload['independent_baseline']['chisq']:.2f}")
    print(f"  κ_c variance: {payload['independent_baseline']['kappa_variance']:.6f}")
    print(f"  κ_c: mean={payload['independent_baseline']['kappa_mean']:.4f}, std={payload['independent_baseline']['kappa_std']:.4f}")

    print(f"\nSweep Results:")
    print(f"  {'Strength':>10} {'χ² Ratio':>10} {'Var Red %':>10} {'Improves?':>10} {'Sweet Spot?':>12}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    for result in payload["sweep_results"]:
        improves = "✓" if result["improves_fit"] else ""
        sweet = "✓" if result.get("is_sweet_spot", False) else ""
        print(f"  {result['coupling_strength']:>10.1f} "
              f"{result['chisq_ratio']:>10.4f} "
              f"{result['variance_reduction']*100:>9.1f}% "
              f"{improves:>10} "
              f"{sweet:>12}")

    # Optimal coupling
    if payload["optimal_coupling"]:
        opt = payload["optimal_coupling"]
        print(f"\nOptimal Coupling Strength:")
        print(f"  Strength: {opt['coupling_strength']:.1f}")
        print(f"  χ² ratio: {opt['chisq_ratio']:.4f}")
        print(f"  Variance reduction: {opt['variance_reduction']:.2%}")
        print(f"  Improves fit: {opt['improves_fit']}")
    else:
        print(f"\nNo optimal coupling found (none maintain/improve fit with coherence)")

    # Sweet spots
    if payload["sweet_spots"]:
        print(f"\nSweet Spots Found ({len(payload['sweet_spots'])}):")
        for spot in payload["sweet_spots"]:
            print(f"  Strength={spot['coupling_strength']:.1f}: "
                  f"χ² ratio={spot['chisq_ratio']:.4f}, "
                  f"variance reduction={spot['variance_reduction']:.2%}")
    else:
        print(f"\nNo sweet spots found (30-70% variance reduction + χ² ratio ≤ 1.0)")

    # Harm threshold
    if payload["harm_threshold"] is not None:
        print(f"\nHarm Threshold:")
        print(f"  Coupling strength {payload['harm_threshold']:.1f} causes >5% fit degradation")
    else:
        print(f"\nNo harm threshold found (all strengths maintain fit quality)")

    # Key findings
    print(f"\nKey Findings:")
    any_improvements = any(r["improves_fit"] for r in payload["sweep_results"] if r["coupling_strength"] > 0)
    if any_improvements:
        print(f"  ✓ EXCITING: Coupling IMPROVES fit quality at some strengths!")
        improving = [r for r in payload["sweep_results"] if r["improves_fit"] and r["coupling_strength"] > 0]
        for r in improving:
            print(f"    - Strength {r['coupling_strength']:.1f}: {(1-r['chisq_ratio'])*100:.1f}% improvement")
    else:
        print(f"  ✗ Coupling never improves fit quality vs independent")

    max_var_red = max(r["variance_reduction"] for r in payload["sweep_results"])
    print(f"  Maximum variance reduction: {max_var_red:.2%}")

    if payload["sweet_spots"]:
        print(f"  ✓ Found {len(payload['sweet_spots'])} sweet spot(s) balancing coherence and fit")

    print(f"\nResults saved to:")
    print(f"  Data: {artifact_path}")
    for path in plot_paths:
        print(f"  Plot: {path}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
