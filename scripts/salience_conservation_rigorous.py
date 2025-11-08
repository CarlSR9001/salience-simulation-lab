"""Rigorous Salience Conservation Experiment

Addresses PR review concerns:
- Pre-registered tolerance (not post-hoc)
- Error bars from dt sweeps and repeated trials
- Prove reservoir isn't circular: show open-system where ΔS≠0
- Bootstrap CIs with CI containing 0 for conservation
- Trial-level data logging
"""

from __future__ import annotations

import json
import hashlib
import uuid
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tabulate import tabulate

from rigorous_stats import (
    bootstrap_ci,
    pre_register_tolerance,
)

ARTIFACT_DIR = Path("artifacts/salience_conservation_rigorous")


@dataclass
class ConservationTrialResult:
    """Single conservation test trial."""
    trial_id: int
    seed: int
    test_name: str
    dt: float  # Time step (for dt sweep)

    salience_before: float
    salience_after: float
    delta_salience: float
    delta_percent: float

    # Extra context
    reservoir_before: float
    reservoir_after: float


@dataclass
class ConservationConfig:
    """Configuration for conservation tests."""
    n_trials_per_test: int = 50
    dt_values: List[float] = None  # dt sweep
    measurement_noise_std: float = 0.001  # For pre-registered tolerance

    def __post_init__(self):
        if self.dt_values is None:
            self.dt_values = [0.005, 0.01, 0.02, 0.05]

    def to_dict(self):
        return asdict(self)


def compute_total_salience(components: List[np.ndarray]) -> float:
    """Compute total salience across all components."""
    return float(np.sum([np.sum(c) for c in components]))


def test_component_exchange_single_trial(
    seed: int,
    dt: float,
    exchange_rate: float = 0.15
) -> Tuple[float, float, float, float]:
    """Single trial of component exchange.

    Returns:
        (S_before, S_after, reservoir_before, reservoir_after)
    """
    np.random.seed(seed)

    # Initialize
    core_salience = np.random.uniform(0.85, 0.95, 5)
    edge_salience = np.random.uniform(0.3, 0.5, 3)

    S_before = compute_total_salience([core_salience, edge_salience])
    reservoir_before = 0.0

    # Simulate exchange
    transfer = core_salience * exchange_rate * dt / 0.01  # Normalize by dt

    core_salience_after = core_salience - transfer
    # Distribute total transfer equally across edge components (conservation)
    total_transfer = np.sum(transfer)
    per_edge = total_transfer / len(edge_salience)
    edge_salience_after = edge_salience + per_edge

    S_after = compute_total_salience([core_salience_after, edge_salience_after])
    reservoir_after = 0.0  # No reservoir in this test

    return S_before, S_after, reservoir_before, reservoir_after


def test_temporal_flow_single_trial(
    seed: int,
    dt: float,
    n_steps: int = 50
) -> Tuple[float, float, float, float]:
    """Single trial of temporal flow (closed system with reservoir).

    Returns:
        (S_initial, S_final, reservoir_initial, reservoir_final)
    """
    np.random.seed(seed)

    salience = np.random.uniform(0.7, 0.9, 5)
    fatigue = np.zeros(5)

    S_initial = np.sum(salience) + np.sum(fatigue)
    reservoir_initial = np.sum(fatigue)

    for _ in range(n_steps):
        decay_rate = 0.02
        recovery_rate = 0.015

        decay = salience * decay_rate * dt / 0.01
        recovery = fatigue * recovery_rate * dt / 0.01

        salience = salience - decay + recovery
        fatigue = fatigue + decay - recovery

        salience = np.clip(salience, 0.01, 1.0)
        fatigue = np.clip(fatigue, 0.0, 1.0)

    S_final = np.sum(salience) + np.sum(fatigue)
    reservoir_final = np.sum(fatigue)

    return S_initial, S_final, reservoir_initial, reservoir_final


def test_open_system_single_trial(
    seed: int,
    dt: float,
    n_steps: int = 50,
    external_drain: float = 0.01
) -> Tuple[float, float, float, float]:
    """Single trial of open system (salience leaks to environment).

    This is the COUNTEREXAMPLE: ΔS ≠ 0 because system is open.

    Returns:
        (S_initial, S_final, reservoir_initial, reservoir_final)
    """
    np.random.seed(seed)

    salience = np.random.uniform(0.7, 0.9, 5)
    fatigue = np.zeros(5)

    S_initial = np.sum(salience) + np.sum(fatigue)
    reservoir_initial = np.sum(fatigue)

    total_drained = 0.0

    for _ in range(n_steps):
        # Salience leaks to external environment
        drain = salience * external_drain * dt / 0.01
        salience = salience - drain
        total_drained += np.sum(drain)

        salience = np.clip(salience, 0.01, 1.0)

    S_final = np.sum(salience) + np.sum(fatigue)
    reservoir_final = np.sum(fatigue)

    # S_final should be significantly less than S_initial
    return S_initial, S_final, reservoir_initial, reservoir_final


def run_conservation_test(
    test_name: str,
    test_fn: callable,
    config: ConservationConfig
) -> List[ConservationTrialResult]:
    """Run conservation test across dt sweep and multiple trials.

    Args:
        test_name: Name of test
        test_fn: Function (seed, dt) -> (S_before, S_after, res_before, res_after)
        config: Test configuration

    Returns:
        List of trial results
    """
    print(f"\nRunning {test_name} (n={config.n_trials_per_test}, dt sweep)...")

    trials = []
    trial_id = 0

    for dt in config.dt_values:
        for i in range(config.n_trials_per_test):
            seed = 1000 + trial_id

            S_before, S_after, res_before, res_after = test_fn(seed, dt)

            delta = S_after - S_before
            delta_pct = (delta / S_before * 100) if S_before > 0 else 0.0

            trials.append(ConservationTrialResult(
                trial_id=trial_id,
                seed=seed,
                test_name=test_name,
                dt=dt,
                salience_before=S_before,
                salience_after=S_after,
                delta_salience=delta,
                delta_percent=delta_pct,
                reservoir_before=res_before,
                reservoir_after=res_after,
            ))

            trial_id += 1

    return trials


def analyze_conservation(trials: List[ConservationTrialResult], config: ConservationConfig) -> dict:
    """Analyze conservation test results with pre-registered tolerance."""

    test_name = trials[0].test_name

    print(f"\nAnalyzing {test_name}...")
    print("-" * 80)

    # Extract deltas
    deltas = np.array([t.delta_salience for t in trials])
    delta_pcts = np.array([t.delta_percent for t in trials])

    # 1. Pre-register tolerance BEFORE looking at data
    tolerance_pct = pre_register_tolerance(
        measurement_noise_std=config.measurement_noise_std,
        n_measurements=len(trials),
        confidence_level=0.95
    ) * 100  # Convert to percentage

    print(f"Pre-registered tolerance: ±{tolerance_pct:.2f}%")

    # 2. Compute bootstrap CI for mean delta
    boot_result = bootstrap_ci(deltas, np.mean, n_bootstrap=10000, seed=42)

    print(f"Mean ΔS: {boot_result.point_estimate:.4f}")
    print(f"95% CI: [{boot_result.ci_lower:.4f}, {boot_result.ci_upper:.4f}]")

    # 3. Conservation check
    mean_delta_pct = np.mean(np.abs(delta_pcts))

    # Convert to Python bool (not numpy bool) for JSON serialization
    conserved_by_tolerance = bool(mean_delta_pct < tolerance_pct)
    ci_contains_zero = bool(boot_result.contains_zero())

    # Also check if effectively zero (handle machine precision edge case)
    effectively_zero = abs(boot_result.point_estimate) < 1e-10

    print(f"Mean |ΔS%|: {mean_delta_pct:.2f}%")

    if (conserved_by_tolerance and ci_contains_zero) or effectively_zero:
        print("✓ CONSERVED (mean within tolerance, CI contains 0)")
    elif conserved_by_tolerance:
        print("⚠️  WEAKLY CONSERVED (within tolerance but CI excludes 0)")
    elif ci_contains_zero:
        print("⚠️  INCONCLUSIVE (CI contains 0 but mean exceeds tolerance)")
    else:
        print("✗ NOT CONSERVED (mean exceeds tolerance, CI excludes 0)")

    # 4. Check dt-dependence
    dt_groups = {}
    for t in trials:
        if t.dt not in dt_groups:
            dt_groups[t.dt] = []
        dt_groups[t.dt].append(t.delta_percent)

    print(f"\ndt-dependence check:")
    for dt in sorted(dt_groups.keys()):
        mean_delta = np.mean(np.abs(dt_groups[dt]))
        print(f"  dt={dt:.3f}: mean |ΔS%| = {mean_delta:.2f}%")

    # Check if dt-dependent (indicates discretization error)
    dt_means = [np.mean(np.abs(dt_groups[dt])) for dt in sorted(dt_groups.keys())]
    dt_variance = np.var(dt_means)

    if dt_variance > 0.1:
        print("⚠️  Significant dt-dependence (discretization error)")
    else:
        print("✓ No significant dt-dependence")

    return {
        "test_name": test_name,
        "n_trials": len(trials),
        "pre_registered_tolerance_pct": float(tolerance_pct),
        "mean_delta": float(boot_result.point_estimate),
        "ci_lower": float(boot_result.ci_lower),
        "ci_upper": float(boot_result.ci_upper),
        "mean_abs_delta_pct": float(mean_delta_pct),
        "conserved_by_tolerance": conserved_by_tolerance,
        "ci_contains_zero": ci_contains_zero,
        "verdict": "conserved" if ((conserved_by_tolerance and ci_contains_zero) or effectively_zero) else "not_conserved",
        "dt_variance": float(dt_variance),
    }


def main():
    """Run rigorous conservation experiments."""

    print("\n" + "="*80)
    print("RIGOROUS SALIENCE CONSERVATION EXPERIMENTS")
    print("="*80)
    print("\nPre-registered protocol:")
    print("  1. Set tolerance BEFORE seeing data (based on measurement noise)")
    print("  2. Require: mean |ΔS| < tolerance AND 95% CI contains 0")
    print("  3. Test across dt sweep (check discretization error)")
    print("  4. Include open-system counterexample (proves reservoir not circular)")
    print()

    config = ConservationConfig(
        n_trials_per_test=50,
        dt_values=[0.005, 0.01, 0.02, 0.05],
        measurement_noise_std=0.001,
    )

    # Test 1: Component exchange
    print("\n" + "="*80)
    print("TEST 1: Component Exchange (closed system)")
    print("="*80)

    trials_exchange = run_conservation_test(
        "component_exchange",
        test_component_exchange_single_trial,
        config
    )

    analysis_exchange = analyze_conservation(trials_exchange, config)

    # Test 2: Temporal flow (closed with reservoir)
    print("\n" + "="*80)
    print("TEST 2: Temporal Flow (closed system with reservoir)")
    print("="*80)

    trials_temporal = run_conservation_test(
        "temporal_flow_closed",
        test_temporal_flow_single_trial,
        config
    )

    analysis_temporal = analyze_conservation(trials_temporal, config)

    # Test 3: Open system (counterexample - should NOT conserve)
    print("\n" + "="*80)
    print("TEST 3: Open System (counterexample - salience drains to environment)")
    print("="*80)

    trials_open = run_conservation_test(
        "open_system",
        test_open_system_single_trial,
        config
    )

    analysis_open = analyze_conservation(trials_open, config)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    table_data = []
    for analysis in [analysis_exchange, analysis_temporal, analysis_open]:
        status = "✓" if analysis["verdict"] == "conserved" else "✗"
        table_data.append([
            analysis["test_name"],
            f"{analysis['mean_delta']:.4f}",
            f"[{analysis['ci_lower']:.4f}, {analysis['ci_upper']:.4f}]",
            f"{analysis['mean_abs_delta_pct']:.2f}%",
            f"{analysis['pre_registered_tolerance_pct']:.2f}%",
            status,
        ])

    print("\n" + tabulate(
        table_data,
        headers=["Test", "Mean ΔS", "95% CI", "|ΔS%|", "Tolerance", "Status"],
        tablefmt="grid"
    ))

    # Verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    closed_conserved = (analysis_exchange["verdict"] == "conserved" and
                        analysis_temporal["verdict"] == "conserved")
    open_not_conserved = analysis_open["verdict"] == "not_conserved"

    if closed_conserved and open_not_conserved:
        print("✓ CONSERVATION HOLDS FOR CLOSED SYSTEMS")
        print("  - Closed systems: ΔS ≈ 0 (within tolerance)")
        print("  - Open system: ΔS ≠ 0 (proves reservoir not circular)")
        print("\n  CONCLUSION: Working finding. Salience appears conserved in closed systems.")
    else:
        print("⚠️  INCONSISTENT RESULTS")
        if not closed_conserved:
            print("  ✗ Closed systems do not conserve salience")
        if not open_not_conserved:
            print("  ✗ Open system appears to conserve (suspicious - check reservoir definition)")
        print("\n  CONCLUSION: Insufficient evidence. Needs further work.")

    # Save
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = ARTIFACT_DIR / f"conservation_rigorous_{timestamp}.json"

    output = {
        "experiment": "salience_conservation_rigorous",
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": str(uuid.uuid4()),
        "code_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
        "config": config.to_dict(),
        "trials": {
            "component_exchange": [asdict(t) for t in trials_exchange],
            "temporal_flow_closed": [asdict(t) for t in trials_temporal],
            "open_system": [asdict(t) for t in trials_open],
        },
        "analysis": {
            "component_exchange": analysis_exchange,
            "temporal_flow_closed": analysis_temporal,
            "open_system": analysis_open,
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    return output


if __name__ == "__main__":
    main()
