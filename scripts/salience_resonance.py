"""Novel Experiment: Salience Resonance

Tests whether multiple continuity-taxed systems with synchronized salience patterns
exhibit energy coupling, transfer, or resonance effects.

Hypothesis: If salience reflects a real physical substrate (like spacetime curvature),
then systems with aligned salience patterns might couple energetically, similar to
resonant oscillators or entangled quantum systems.

Scientific Approach:
1. Run two identical PID controllers in parallel on separate plants
2. Vary the phase relationship between their salience patterns (synchronized, anti-phase, independent)
3. Measure total system energy and look for deviations from additive expectation
4. Test if "salience coherence" between systems reduces total energy cost
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tabulate import tabulate

ARTIFACT_DIR = Path("artifacts/salience_resonance")


@dataclass
class ResonanceConfig:
    dt: float = 0.01
    horizon: float = 10.0
    lambda_c: float = 2.0
    phase_modes: List[str] = None  # ["synchronized", "anti_phase", "independent", "random"]
    n_trials: int = 10

    def __post_init__(self):
        if self.phase_modes is None:
            self.phase_modes = ["synchronized", "anti_phase", "independent", "random"]


class SalienceTracker:
    """Tracks salience for a single controller."""

    def __init__(self, n_channels: int = 5):
        self.n_channels = n_channels
        self.delta_a = np.random.uniform(0.7, 1.0, n_channels)
        self.retention = np.random.uniform(0.8, 0.95, n_channels)
        self.payoff = np.ones(n_channels)
        self.fatigue = np.zeros(n_channels)

    def compute_salience(self, control_magnitude: float) -> np.ndarray:
        """Compute current salience vector."""
        # Update payoff based on control effectiveness
        self.payoff = 0.9 * self.payoff + 0.1 * min(control_magnitude, 1.0)

        # Accumulate fatigue
        self.fatigue = 0.95 * self.fatigue + 0.05 * abs(control_magnitude)

        # Salience formula from AGENTS.md
        salience = self.delta_a * self.retention * self.payoff * (1.0 - self.fatigue)
        return np.clip(salience, 0.001, 1.0)

    def get_mean_salience(self) -> float:
        """Get average salience across channels."""
        s = self.compute_salience(0.0)
        return float(np.mean(s))

    def get_salience_vector(self) -> np.ndarray:
        """Get full salience vector."""
        return self.compute_salience(0.0)


class ContinuityPID:
    """PID controller with continuity tax."""

    def __init__(self, kp: float = 2.0, ki: float = 0.5, kd: float = 0.1,
                 lambda_c: float = 2.0, salience_tracker: SalienceTracker = None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.lambda_c = lambda_c
        self.salience = salience_tracker or SalienceTracker()

        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, error: float, dt: float) -> Tuple[float, float]:
        """Execute one control step.

        Returns:
            (control_output, energy_spent)
        """
        # Compute salience
        s_vec = self.salience.compute_salience(abs(error))
        mean_salience = np.mean(s_vec)

        # Effective mass from continuity tax
        m_eff = 1.0 + self.lambda_c * mean_salience

        # PID terms
        p_term = self.kp * error / m_eff
        self.integral += error * dt / m_eff
        i_term = self.ki * self.integral
        d_term = self.kd * (error - self.prev_error) / (dt * m_eff)

        control = p_term + i_term + d_term

        # Energy cost (simplified as control effort squared)
        energy = control ** 2 * dt

        self.prev_error = error

        return control, energy


class Plant:
    """Simple first-order plant."""

    def __init__(self, tau: float = 0.1, dt: float = 0.01):
        self.tau = tau
        self.dt = dt
        self.output = 0.0

    def step(self, control: float) -> float:
        """Apply control and return new output."""
        # First-order response: dy/dt = (u - y) / tau
        self.output += (control - self.output) / self.tau * self.dt
        return self.output


def run_coupled_system(config: ResonanceConfig, phase_mode: str, seed: int = None
                       ) -> dict:
    """Run two controllers with specified phase relationship."""
    if seed is not None:
        np.random.seed(seed)

    # Initialize two identical systems
    salience_a = SalienceTracker(n_channels=5)
    salience_b = SalienceTracker(n_channels=5)

    # Set up phase relationship
    if phase_mode == "synchronized":
        # Identical salience initialization
        salience_b.delta_a = salience_a.delta_a.copy()
        salience_b.retention = salience_a.retention.copy()
    elif phase_mode == "anti_phase":
        # Inverted salience patterns
        salience_b.delta_a = 1.0 - salience_a.delta_a + 0.5
        salience_b.retention = 1.0 - salience_a.retention + 0.5
        salience_b.delta_a = np.clip(salience_b.delta_a, 0.5, 1.0)
        salience_b.retention = np.clip(salience_b.retention, 0.7, 0.95)
    elif phase_mode == "independent":
        # Keep random initialization
        pass
    elif phase_mode == "random":
        # Completely random patterns each step
        pass

    controller_a = ContinuityPID(lambda_c=config.lambda_c, salience_tracker=salience_a)
    controller_b = ContinuityPID(lambda_c=config.lambda_c, salience_tracker=salience_b)

    plant_a = Plant(dt=config.dt)
    plant_b = Plant(dt=config.dt)

    target = 1.0
    steps = int(config.horizon / config.dt)

    energy_a = 0.0
    energy_b = 0.0

    salience_coherence_sum = 0.0
    salience_a_history = []
    salience_b_history = []

    for step in range(steps):
        # Randomize salience_b each step if in random mode
        if phase_mode == "random":
            salience_b.delta_a = np.random.uniform(0.7, 1.0, 5)
            salience_b.retention = np.random.uniform(0.8, 0.95, 5)

        # Compute errors
        error_a = target - plant_a.output
        error_b = target - plant_b.output

        # Control steps
        control_a, e_a = controller_a.step(error_a, config.dt)
        control_b, e_b = controller_b.step(error_b, config.dt)

        energy_a += e_a
        energy_b += e_b

        # Plant updates
        plant_a.step(control_a)
        plant_b.step(control_b)

        # Track salience coherence (dot product of normalized vectors)
        s_a = salience_a.get_salience_vector()
        s_b = salience_b.get_salience_vector()

        s_a_norm = s_a / (np.linalg.norm(s_a) + 1e-9)
        s_b_norm = s_b / (np.linalg.norm(s_b) + 1e-9)

        coherence = np.dot(s_a_norm, s_b_norm)
        salience_coherence_sum += coherence

        salience_a_history.append(np.mean(s_a))
        salience_b_history.append(np.mean(s_b))

    total_energy = energy_a + energy_b
    mean_coherence = salience_coherence_sum / steps

    return {
        "phase_mode": phase_mode,
        "energy_a": float(energy_a),
        "energy_b": float(energy_b),
        "total_energy": float(total_energy),
        "mean_salience_coherence": float(mean_coherence),
        "mean_salience_a": float(np.mean(salience_a_history)),
        "mean_salience_b": float(np.mean(salience_b_history)),
        "final_output_a": float(plant_a.output),
        "final_output_b": float(plant_b.output),
        "lambda_c": config.lambda_c,
        "seed": seed,
    }


def run_baseline(config: ResonanceConfig, n_trials: int = 10) -> dict:
    """Run two independent controllers (no continuity tax) as baseline."""

    energies = []

    for trial in range(n_trials):
        controller_a = ContinuityPID(lambda_c=0.0)
        controller_b = ContinuityPID(lambda_c=0.0)

        plant_a = Plant(dt=config.dt)
        plant_b = Plant(dt=config.dt)

        target = 1.0
        steps = int(config.horizon / config.dt)

        energy = 0.0

        for _ in range(steps):
            error_a = target - plant_a.output
            error_b = target - plant_b.output

            control_a, e_a = controller_a.step(error_a, config.dt)
            control_b, e_b = controller_b.step(error_b, config.dt)

            energy += e_a + e_b

            plant_a.step(control_a)
            plant_b.step(control_b)

        energies.append(energy)

    return {
        "mean_total_energy": float(np.mean(energies)),
        "std_total_energy": float(np.std(energies)),
        "min_total_energy": float(np.min(energies)),
        "max_total_energy": float(np.max(energies)),
    }


def main():
    """Run salience resonance experiment."""

    print("\n" + "="*70)
    print("SALIENCE RESONANCE EXPERIMENT")
    print("="*70)
    print("\nHypothesis: Systems with synchronized salience patterns may exhibit")
    print("energy coupling beyond simple additive expectation.\n")

    config = ResonanceConfig(
        dt=0.01,
        horizon=10.0,
        lambda_c=2.0,
        n_trials=20,
    )

    # Run baseline (no continuity tax)
    print("Running baseline (λ=0, independent systems)...")
    baseline = run_baseline(config, n_trials=config.n_trials)

    print(f"  Baseline mean energy: {baseline['mean_total_energy']:.4f} ± {baseline['std_total_energy']:.4f}")

    # Run experiments across phase modes
    results = []

    for phase_mode in config.phase_modes:
        print(f"\nRunning phase_mode={phase_mode}...")

        mode_results = []
        for trial in range(config.n_trials):
            result = run_coupled_system(config, phase_mode, seed=1000 + trial)
            mode_results.append(result)

        # Aggregate
        total_energies = [r["total_energy"] for r in mode_results]
        coherences = [r["mean_salience_coherence"] for r in mode_results]

        summary = {
            "phase_mode": phase_mode,
            "lambda_c": config.lambda_c,
            "n_trials": config.n_trials,
            "mean_total_energy": float(np.mean(total_energies)),
            "std_total_energy": float(np.std(total_energies)),
            "mean_coherence": float(np.mean(coherences)),
            "std_coherence": float(np.std(coherences)),
            "baseline_mean_energy": baseline["mean_total_energy"],
            "energy_ratio": float(np.mean(total_energies) / baseline["mean_total_energy"]),
            "trials": mode_results,
        }

        results.append(summary)

        print(f"  Mean energy: {summary['mean_total_energy']:.4f} ± {summary['std_total_energy']:.4f}")
        print(f"  Mean coherence: {summary['mean_coherence']:.4f} ± {summary['std_coherence']:.4f}")
        print(f"  Ratio vs baseline: {summary['energy_ratio']:.4f}")

    # Analyze for resonance effects
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    table_data = []
    for r in results:
        table_data.append([
            r["phase_mode"],
            f"{r['mean_coherence']:.3f} ± {r['std_coherence']:.3f}",
            f"{r['mean_total_energy']:.3f} ± {r['std_total_energy']:.3f}",
            f"{r['energy_ratio']:.4f}",
        ])

    print("\n" + tabulate(
        table_data,
        headers=["Phase Mode", "Salience Coherence", "Total Energy", "Ratio vs Baseline"],
        tablefmt="grid"
    ))

    # Look for anomalies
    print("\n" + "="*70)
    print("ANOMALY DETECTION")
    print("="*70)

    # Expected: energy_ratio ~ constant regardless of coherence
    # Anomaly: energy_ratio significantly lower for high coherence

    sync_result = next(r for r in results if r["phase_mode"] == "synchronized")
    anti_result = next(r for r in results if r["phase_mode"] == "anti_phase")
    indep_result = next(r for r in results if r["phase_mode"] == "independent")

    energy_coherence_correlation = []
    for r in results:
        for trial in r["trials"]:
            energy_coherence_correlation.append((
                trial["mean_salience_coherence"],
                trial["total_energy"]
            ))

    # Compute Pearson correlation
    coherences_all = [x[0] for x in energy_coherence_correlation]
    energies_all = [x[1] for x in energy_coherence_correlation]

    if len(coherences_all) > 1:
        corr = np.corrcoef(coherences_all, energies_all)[0, 1]
        print(f"\nPearson correlation (coherence vs energy): {corr:.4f}")

        if corr < -0.3:
            print("⚠️  ANOMALY: Negative correlation detected!")
            print("    High salience coherence associated with LOWER energy.")
            print("    This suggests genuine resonance/coupling between systems.")
        elif abs(corr) < 0.15:
            print("✓  No significant correlation (expected baseline).")
        else:
            print("⚠️  Positive correlation detected.")
            print("    High coherence associated with HIGHER energy (anti-resonance?).")

    # Statistical test: synchronized vs independent
    sync_energies = [t["total_energy"] for t in sync_result["trials"]]
    indep_energies = [t["total_energy"] for t in indep_result["trials"]]

    diff_mean = np.mean(sync_energies) - np.mean(indep_energies)
    diff_std = np.sqrt(np.var(sync_energies) / len(sync_energies) +
                       np.var(indep_energies) / len(indep_energies))

    if diff_std > 0:
        z_score = diff_mean / diff_std
        print(f"\nSynchronized vs Independent:")
        print(f"  Difference in means: {diff_mean:.4f}")
        print(f"  Z-score: {z_score:.3f}")

        if abs(z_score) > 2.5:
            print(f"  ⚠️  STATISTICALLY SIGNIFICANT at p<0.01")
        elif abs(z_score) > 1.96:
            print(f"  ⚠️  SIGNIFICANT at p<0.05")
        else:
            print(f"  ✓  Not statistically significant")

    # Save results
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = ARTIFACT_DIR / f"resonance_{timestamp}.json"

    output = {
        "experiment": "salience_resonance",
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": str(uuid.uuid4()),
        "config": {
            "dt": config.dt,
            "horizon": config.horizon,
            "lambda_c": config.lambda_c,
            "n_trials": config.n_trials,
        },
        "baseline": baseline,
        "results": results,
        "correlation_coherence_energy": float(corr) if len(coherences_all) > 1 else None,
        "z_score_sync_vs_indep": float(z_score) if diff_std > 0 else None,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
