"""Goldilocks Zone Sweep - Finding Optimal Continuity Tax

Based on the energy-mass investigation findings, this sweep searches for
the optimal λ_c value that:
1. Reduces energy (by damping overshoot)
2. Maintains control (reaches target ≥95%)
3. Preserves salience (≥0.7)

Key Insight from Investigation:
- Baseline wastes ~76% of energy correcting overshoot
- λ_c = 40 is too extreme (system fails)
- Edge component at λ = 0.05 shows genuine efficiency gains
- Hypothesis: λ_c ∈ [2, 5] provides optimal damping
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from tabulate import tabulate

ARTIFACT_DIR = Path("artifacts/goldilocks")


def run_pid_controller(
    kp: float = 2.0,
    ki: float = 0.5,
    kd: float = 0.1,
    lambda_c: float = 0.0,
    dt: float = 0.01,
    horizon: float = 15.0,
    target: float = 1.0,
) -> dict:
    """Run a simple PID controller with continuity tax."""

    # Salience tracker (simplified)
    n_channels = 5
    salience = np.random.uniform(0.75, 0.95, n_channels)

    # Controller state
    integral = 0.0
    prev_error = 0.0
    output = 0.0

    # Plant (first-order system)
    tau = 0.1

    # Tracking
    steps = int(horizon / dt)
    outputs = []
    controls = []
    saliences = []
    energies = []

    total_energy = 0.0

    for step in range(steps):
        # Error
        error = target - output

        # Update salience (simplified dynamics)
        payoff = min(1.0, 1.0 / (abs(error) + 0.1))
        fatigue = abs(error) * 0.05

        salience = 0.95 * salience + 0.05 * payoff - fatigue * 0.02
        salience = np.clip(salience, 0.01, 1.0)

        mean_salience = np.mean(salience)

        # Effective mass from continuity tax
        m_eff = 1.0 + lambda_c * mean_salience

        # PID control
        p = kp * error / m_eff
        integral += error * dt / m_eff
        i = ki * integral
        d = kd * (error - prev_error) / (dt * m_eff)

        control = p + i + d
        prev_error = error

        # Energy (control effort squared)
        energy = control**2 * dt
        total_energy += energy

        # Plant dynamics (first-order)
        output += (control - output) / tau * dt

        # Record
        outputs.append(output)
        controls.append(control)
        saliences.append(mean_salience)
        energies.append(energy)

    # Compute metrics
    outputs_arr = np.array(outputs)
    energies_arr = np.array(energies)

    # Rise time (first crossing of 0.9 * target)
    rise_threshold = 0.9 * target
    rise_idx = np.where(outputs_arr >= rise_threshold)[0]
    if len(rise_idx) > 0:
        rise_time = rise_idx[0] * dt
        rise_idx_val = rise_idx[0]
    else:
        rise_time = np.inf
        rise_idx_val = steps

    # Final output
    final_output = outputs_arr[-1]

    # Peak overshoot
    if len(rise_idx) > 0:
        outputs_after_rise = outputs_arr[rise_idx_val:]
        peak_overshoot = np.max(outputs_after_rise) - target
    else:
        peak_overshoot = 0.0

    # Energy breakdown
    if rise_idx_val < steps:
        energy_before_rise = np.sum(energies_arr[:rise_idx_val])
        energy_after_rise = np.sum(energies_arr[rise_idx_val:])
    else:
        energy_before_rise = total_energy
        energy_after_rise = 0.0

    # m_eff (simplified as mean effective mass)
    mean_m_eff = np.mean([1.0 + lambda_c * s for s in saliences])

    return {
        "lambda_c": float(lambda_c),
        "final_output": float(final_output),
        "rise_time": float(rise_time) if rise_time != np.inf else None,
        "peak_overshoot": float(peak_overshoot),
        "total_energy": float(total_energy),
        "energy_before_rise": float(energy_before_rise),
        "energy_after_rise": float(energy_after_rise),
        "overshoot_energy_fraction": float(energy_after_rise / total_energy if total_energy > 0 else 0),
        "mean_salience": float(np.mean(saliences)),
        "min_salience": float(np.min(saliences)),
        "m_eff": float(mean_m_eff),
        "reached_target": bool(final_output >= 0.95),
    }


def main():
    """Run Goldilocks sweep."""

    print("\n" + "="*70)
    print("GOLDILOCKS ZONE SWEEP")
    print("="*70)
    print("\nSearching for optimal λ_c that minimizes energy while reaching target.\n")

    # Fine-grained sweep focused on [0, 10]
    lambda_values = [
        0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
        4.5, 5.0, 6.0, 7.0, 8.0, 10.0
    ]

    results = []

    print("Running sweep...")
    for lam in lambda_values:
        result = run_pid_controller(lambda_c=lam, horizon=15.0)
        results.append(result)

        status = "✓" if result["reached_target"] else "✗"
        print(f"  λ={lam:5.2f}: output={result['final_output']:.3f} "
              f"energy={result['total_energy']:6.2f} "
              f"salience={result['mean_salience']:.3f} {status}")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    # Filter for valid results (reached target)
    valid = [r for r in results if r["reached_target"]]
    invalid = [r for r in results if not r["reached_target"]]

    baseline = next((r for r in results if r["lambda_c"] == 0.0), None)

    if not valid:
        print("\n⚠️  No lambda value reached target!")
        print("    Need longer horizon or different parameters.")
    else:
        print(f"\n✓ {len(valid)}/{len(results)} configurations reached target\n")

        # Find optimal (minimum energy among valid)
        optimal = min(valid, key=lambda r: r["total_energy"])

        print("OPTIMAL CONFIGURATION:")
        print(f"  λ_c = {optimal['lambda_c']:.2f}")
        print(f"  Final output: {optimal['final_output']:.4f}")
        print(f"  Rise time: {optimal['rise_time']:.3f} s")
        print(f"  Total energy: {optimal['total_energy']:.3f}")
        print(f"  Peak overshoot: {optimal['peak_overshoot']:+.4f}")
        print(f"  Mean salience: {optimal['mean_salience']:.3f}")

        if baseline and baseline["reached_target"]:
            energy_savings = (1 - optimal["total_energy"] / baseline["total_energy"]) * 100
            rise_slowdown = (optimal["rise_time"] / baseline["rise_time"] - 1) * 100
            overshoot_reduction = (1 - abs(optimal["peak_overshoot"]) / abs(baseline["peak_overshoot"])) * 100 if baseline["peak_overshoot"] != 0 else 0

            print(f"\nVS BASELINE (λ=0):")
            print(f"  Energy savings: {energy_savings:+.1f}%")
            print(f"  Rise time change: {rise_slowdown:+.1f}%")
            print(f"  Overshoot reduction: {overshoot_reduction:+.1f}%")

            if energy_savings > 15 and overshoot_reduction > 30:
                print(f"\n🎉 GOLDILOCKS ZONE FOUND!")
                print(f"    λ_c ≈ {optimal['lambda_c']:.1f} provides significant efficiency")
                print(f"    by damping overshoot while maintaining control.")

    # Detailed table
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)

    table_data = []
    for r in results:
        marker = "✓" if r["reached_target"] else "✗"
        overshoot_pct = r["overshoot_energy_fraction"] * 100

        table_data.append([
            f"{r['lambda_c']:.2f}",
            f"{r['final_output']:.3f}",
            f"{r['rise_time']:.3f}" if r['rise_time'] is not None else "∞",
            f"{r['total_energy']:.2f}",
            f"{overshoot_pct:.1f}%",
            f"{r['mean_salience']:.3f}",
            marker,
        ])

    print("\n" + tabulate(
        table_data,
        headers=["λ_c", "Output", "Rise(s)", "Energy", "Overshoot%", "Salience", "OK"],
        tablefmt="grid"
    ))

    # Identify transition points
    print("\n" + "="*70)
    print("PHASE TRANSITIONS")
    print("="*70)

    for i in range(len(results) - 1):
        r1, r2 = results[i], results[i+1]

        # Detect control collapse transition
        if r1["reached_target"] and not r2["reached_target"]:
            print(f"\n⚠️  CONTROL COLLAPSE between λ={r1['lambda_c']:.2f} and λ={r2['lambda_c']:.2f}")
            print(f"    Output drops from {r1['final_output']:.3f} to {r2['final_output']:.3f}")

        # Detect energy discontinuity
        if r1["reached_target"] and r2["reached_target"]:
            energy_jump = abs(r2["total_energy"] - r1["total_energy"])
            if energy_jump > 1.0:
                print(f"\n⚠️  ENERGY DISCONTINUITY between λ={r1['lambda_c']:.2f} and λ={r2['lambda_c']:.2f}")
                print(f"    Energy jumps by {energy_jump:.2f}")

    # Save results
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = ARTIFACT_DIR / f"goldilocks_{timestamp}.json"

    output = {
        "experiment": "goldilocks_sweep",
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": str(uuid.uuid4()),
        "lambda_values": lambda_values,
        "results": results,
        "optimal": optimal if valid else None,
        "baseline": baseline,
        "n_valid": len(valid),
        "n_invalid": len(invalid),
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
