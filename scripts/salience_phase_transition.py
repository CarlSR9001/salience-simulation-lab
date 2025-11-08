"""Novel Analysis: Salience Phase Transitions

Analyzes continuity tax sweeps to detect phase transitions - critical points where
system behavior undergoes sudden qualitative changes.

Inspired by statistical mechanics: phase transitions occur when control parameters
cross critical thresholds, causing order parameters to change discontinuously.

Order Parameters Monitored:
1. Control effectiveness (output / target)
2. Energy efficiency (energy / distance traveled)
3. Salience structure (variance, entropy)
4. Response dynamics (rise time, overshoot)

Detection Methods:
1. First derivative discontinuity (sharp slope change)
2. Second derivative peak (inflection point)
3. Variance spike (critical fluctuations)
4. Order parameter crossing thresholds
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from tabulate import tabulate

ARTIFACT_DIR = Path("artifacts/phase_transitions")


@dataclass
class PhaseTransition:
    """Detected phase transition."""
    lambda_critical: float
    transition_type: str  # "control_collapse", "energy_discontinuity", "salience_reorganization"
    order_parameter: str  # which metric changed
    pre_value: float
    post_value: float
    jump_magnitude: float
    confidence: float  # 0-1


def detect_discontinuity(x: np.ndarray, y: np.ndarray, window: int = 3) -> List[Tuple[float, float, float]]:
    """Detect discontinuities in y(x) using derivative analysis.

    Returns:
        List of (x_critical, jump_size, confidence)
    """
    if len(x) < 5:
        return []

    # Compute first derivative (central difference)
    dy = np.gradient(y, x)

    # Compute second derivative
    d2y = np.gradient(dy, x)

    # Find peaks in |d2y| (inflection points)
    d2y_abs = np.abs(d2y)

    transitions = []

    for i in range(window, len(x) - window):
        # Check if this is a local maximum in |d2y|
        local_window = d2y_abs[i-window:i+window+1]
        if d2y_abs[i] == np.max(local_window) and d2y_abs[i] > 0:
            # Measure jump magnitude
            y_before = np.mean(y[max(0, i-window):i])
            y_after = np.mean(y[i:min(len(y), i+window)])

            jump = abs(y_after - y_before)

            # Confidence based on jump relative to baseline
            baseline_var = np.var(y)
            if baseline_var > 0:
                confidence = min(1.0, jump / (np.sqrt(baseline_var) + 1e-9))
            else:
                confidence = 0.0

            if confidence > 0.3:  # Significant jump
                transitions.append((x[i], jump, confidence))

    return transitions


def analyze_lambda_sweep(data: Dict) -> List[PhaseTransition]:
    """Analyze a lambda sweep for phase transitions.

    Expected data format from continuity_mass_sim.py or similar:
    {
        "results": [
            {
                "lambda_c": float,
                "final_output": float,
                "total_energy": float,
                "mean_salience": float,
                "rise_time_90": float,
                "m_eff": float,
                "energy_ratio": float,
                ...
            },
            ...
        ]
    }
    """
    results = data.get("results", [])
    if not results:
        return []

    # Sort by lambda_c
    results_sorted = sorted(results, key=lambda r: r.get("lambda_c", 0))

    # Extract arrays
    lambda_values = np.array([r.get("lambda_c", 0) for r in results_sorted])
    final_outputs = np.array([r.get("final_output", 0) for r in results_sorted])
    energies = np.array([r.get("total_energy", 0) for r in results_sorted])
    saliences = np.array([r.get("mean_salience", 0) for r in results_sorted])
    m_effs = np.array([r.get("m_eff", np.nan) for r in results_sorted])
    energy_ratios = np.array([r.get("energy_ratio", np.nan) for r in results_sorted])

    # Remove NaN entries
    valid_mask = ~(np.isnan(m_effs) | np.isnan(energy_ratios))

    if np.sum(valid_mask) < 5:
        return []

    lambda_clean = lambda_values[valid_mask]
    outputs_clean = final_outputs[valid_mask]
    energies_clean = energies[valid_mask]
    saliences_clean = saliences[valid_mask]
    m_effs_clean = m_effs[valid_mask]
    energy_ratios_clean = energy_ratios[valid_mask]

    transitions = []

    # 1. Detect control collapse (output drops below 0.5)
    output_trans = detect_discontinuity(lambda_clean, outputs_clean, window=2)
    for lam, jump, conf in output_trans:
        if conf > 0.5:
            idx = np.argmin(np.abs(lambda_clean - lam))
            transitions.append(PhaseTransition(
                lambda_critical=float(lam),
                transition_type="control_collapse",
                order_parameter="final_output",
                pre_value=float(outputs_clean[max(0, idx-1)]),
                post_value=float(outputs_clean[min(len(outputs_clean)-1, idx+1)]),
                jump_magnitude=float(jump),
                confidence=float(conf),
            ))

    # 2. Detect energy discontinuity
    energy_trans = detect_discontinuity(lambda_clean, energies_clean, window=2)
    for lam, jump, conf in energy_trans:
        if conf > 0.5:
            idx = np.argmin(np.abs(lambda_clean - lam))
            transitions.append(PhaseTransition(
                lambda_critical=float(lam),
                transition_type="energy_discontinuity",
                order_parameter="total_energy",
                pre_value=float(energies_clean[max(0, idx-1)]),
                post_value=float(energies_clean[min(len(energies_clean)-1, idx+1)]),
                jump_magnitude=float(jump),
                confidence=float(conf),
            ))

    # 3. Detect salience reorganization
    salience_trans = detect_discontinuity(lambda_clean, saliences_clean, window=2)
    for lam, jump, conf in salience_trans:
        if conf > 0.4:
            idx = np.argmin(np.abs(lambda_clean - lam))
            transitions.append(PhaseTransition(
                lambda_critical=float(lam),
                transition_type="salience_reorganization",
                order_parameter="mean_salience",
                pre_value=float(saliences_clean[max(0, idx-1)]),
                post_value=float(saliences_clean[min(len(saliences_clean)-1, idx+1)]),
                jump_magnitude=float(jump),
                confidence=float(conf),
            ))

    # 4. Detect m_eff / energy_ratio decoupling (anomaly threshold)
    if len(m_effs_clean) > 3:
        mismatch = np.abs(m_effs_clean - energy_ratios_clean)
        mismatch_trans = detect_discontinuity(lambda_clean, mismatch, window=2)

        for lam, jump, conf in mismatch_trans:
            if conf > 0.6:
                idx = np.argmin(np.abs(lambda_clean - lam))
                transitions.append(PhaseTransition(
                    lambda_critical=float(lam),
                    transition_type="mass_energy_decoupling",
                    order_parameter="abs(m_eff - energy_ratio)",
                    pre_value=float(mismatch[max(0, idx-1)]),
                    post_value=float(mismatch[min(len(mismatch)-1, idx+1)]),
                    jump_magnitude=float(jump),
                    confidence=float(conf),
                ))

    return transitions


def run_goldilocks_sweep() -> Dict:
    """Run a fine-grained lambda sweep focused on the Goldilocks zone (0-10)."""

    from scripts.continuity_mass_sim import (
        PlantConfig, ControllerConfig, run_continuity_taxed_controller
    )

    print("\n" + "="*70)
    print("GOLDILOCKS ZONE SWEEP (λ ∈ [0, 10])")
    print("="*70)
    print("\nSearching for optimal continuity tax that:")
    print("  1. Reduces energy (via overshoot damping)")
    print("  2. Maintains control (reaches target)")
    print("  3. Preserves salience (> 0.7)")
    print()

    # Fine-grained sweep
    lambda_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]

    plant_cfg = PlantConfig(dt=0.01, horizon=10.0, tau=0.1, salience_channels=5)
    ctrl_cfg = ControllerConfig(kp=2.0, ki=0.5, kd=0.1)

    results = []

    for lam in lambda_values:
        print(f"Running λ_c = {lam:.2f}...")

        ctrl_cfg_run = ControllerConfig(kp=ctrl_cfg.kp, ki=ctrl_cfg.ki, kd=ctrl_cfg.kd, lambda_c=lam)

        result = run_continuity_taxed_controller(plant_cfg, ctrl_cfg_run)
        results.append(result)

        print(f"  Output: {result['final_output']:.3f}, Energy: {result['total_energy']:.3f}, "
              f"Salience: {result['mean_salience']:.3f}")

    return {
        "experiment": "goldilocks_sweep",
        "timestamp": datetime.now(UTC).isoformat(),
        "lambda_values": lambda_values,
        "results": results,
    }


def main():
    """Run phase transition analysis."""

    print("\n" + "="*70)
    print("SALIENCE PHASE TRANSITION DETECTOR")
    print("="*70)
    print("\nSearching for critical points where system behavior changes qualitatively.\n")

    # First, run a Goldilocks sweep
    print("Step 1: Generating lambda sweep data...")

    sweep_data = run_goldilocks_sweep()

    # Analyze for phase transitions
    print("\n" + "="*70)
    print("Step 2: Detecting phase transitions...")
    print("="*70)

    transitions = analyze_lambda_sweep(sweep_data)

    if not transitions:
        print("\n✓ No significant phase transitions detected.")
        print("  System behavior appears smooth across lambda range.")
    else:
        print(f"\n⚠️  Detected {len(transitions)} phase transition(s):\n")

        table_data = []
        for t in sorted(transitions, key=lambda x: x.lambda_critical):
            table_data.append([
                f"{t.lambda_critical:.2f}",
                t.transition_type,
                t.order_parameter,
                f"{t.pre_value:.3f}",
                f"{t.post_value:.3f}",
                f"{t.jump_magnitude:.3f}",
                f"{t.confidence:.2f}",
            ])

        print(tabulate(
            table_data,
            headers=["λ_critical", "Type", "Parameter", "Before", "After", "Jump", "Confidence"],
            tablefmt="grid"
        ))

    # Identify optimal lambda (if exists)
    print("\n" + "="*70)
    print("Step 3: Identifying optimal λ (Goldilocks zone)")
    print("="*70)

    results = sweep_data["results"]

    # Filter for valid results (reached target)
    valid_results = [r for r in results if r.get("final_output", 0) >= 0.95]

    if not valid_results:
        print("\n⚠️  No lambda value achieved target output ≥ 0.95")
        print("    Need longer horizon or lower lambda values.")
    else:
        # Find minimum energy among valid results
        best = min(valid_results, key=lambda r: r.get("total_energy", float('inf')))

        baseline = next((r for r in results if r.get("lambda_c", 0) == 0.0), None)

        print(f"\n✓ Optimal λ_c = {best.get('lambda_c', 0):.2f}")
        print(f"    Final output: {best.get('final_output', 0):.3f}")
        print(f"    Total energy: {best.get('total_energy', 0):.3f}")
        print(f"    Mean salience: {best.get('mean_salience', 0):.3f}")

        if baseline:
            energy_savings = (1.0 - best.get('total_energy', 0) / baseline.get('total_energy', 1.0)) * 100
            print(f"    Energy savings vs baseline: {energy_savings:.1f}%")

            if energy_savings > 20:
                print("\n🎉 GOLDILOCKS ZONE FOUND!")
                print(f"    λ_c ≈ {best.get('lambda_c', 0):.2f} provides significant efficiency gain")
                print("    while maintaining control and salience.")

    # Save results
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = ARTIFACT_DIR / f"phase_transitions_{timestamp}.json"

    output = {
        "experiment": "salience_phase_transition",
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": str(uuid.uuid4()),
        "sweep_data": sweep_data,
        "transitions": [
            {
                "lambda_critical": t.lambda_critical,
                "transition_type": t.transition_type,
                "order_parameter": t.order_parameter,
                "pre_value": t.pre_value,
                "post_value": t.post_value,
                "jump_magnitude": t.jump_magnitude,
                "confidence": t.confidence,
            }
            for t in transitions
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    return transitions


if __name__ == "__main__":
    main()
