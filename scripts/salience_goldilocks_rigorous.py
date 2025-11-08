"""Rigorous Goldilocks Zone Experiment

Addresses PR review concerns:
- Fix metric bugs (rise_time=0.000, overshoot issues)
- Piecewise vs smooth model selection (ΔAIC/BIC)
- Co-occurrence requirement: ≥2 order parameters must move together
- Stability testing: ≥10 seeds × ≥3 horizons
- Show one-break fit beating smooth fit
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
    compare_piecewise_vs_smooth,
    stability_rate,
)

ARTIFACT_DIR = Path("artifacts/salience_goldilocks_rigorous")


@dataclass
class GoldilocksTrialResult:
    """Single trial result."""
    trial_id: int
    seed: int
    lambda_c: float
    horizon: float

    # Order parameters
    final_output: float
    total_energy: float
    mean_salience: float
    rise_time_90: float  # Time to reach 90% of target
    overshoot_pct: float  # Overshoot percentage


@dataclass
class GoldilocksConfig:
    """Configuration for Goldilocks sweep."""
    lambda_values: List[float] = None
    n_seeds: int = 10
    test_horizons: List[float] = None
    dt: float = 0.01
    target: float = 1.0

    def __post_init__(self):
        if self.lambda_values is None:
            # Fine-grained sweep
            self.lambda_values = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]
        if self.test_horizons is None:
            self.test_horizons = [10.0, 15.0, 20.0]

    def to_dict(self):
        return asdict(self)


class SimplePID:
    """Simple PID with continuity tax."""

    def __init__(self, kp: float, ki: float, kd: float, lambda_c: float, dt: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.lambda_c = lambda_c
        self.dt = dt

        self.integral = 0.0
        self.prev_error = 0.0

        # Salience tracker
        self.salience = 0.8  # Simple scalar salience
        self.fatigue = 0.0

    def step(self, error: float) -> Tuple[float, float]:
        """Execute one control step.

        Returns:
            (control_output, energy_spent)
        """
        # Update salience
        self.salience = 0.95 * self.salience + 0.05 * min(abs(error), 1.0)
        self.fatigue = 0.90 * self.fatigue + 0.10 * abs(error)
        self.salience = max(0.01, self.salience - 0.3 * self.fatigue)

        # Effective mass
        m_eff = 1.0 + self.lambda_c * self.salience

        # PID terms
        p_term = self.kp * error / m_eff
        self.integral += error * self.dt / m_eff
        self.integral = np.clip(self.integral, -10.0, 10.0)
        i_term = self.ki * self.integral

        d_term = self.kd * (error - self.prev_error) / (self.dt * m_eff)

        control = p_term + i_term + d_term

        energy = control ** 2 * self.dt

        self.prev_error = error

        return control, energy


class SimplePlant:
    """First-order plant."""

    def __init__(self, tau: float, dt: float):
        self.tau = tau
        self.dt = dt
        self.output = 0.0

    def step(self, control: float) -> float:
        """Apply control and return new output."""
        self.output += (control - self.output) / self.tau * self.dt
        return self.output


def compute_metrics(
    output_history: np.ndarray,
    target: float,
    dt: float
) -> Tuple[float, float]:
    """Compute rise time and overshoot (FIXED).

    Args:
        output_history: Time series of plant output
        target: Target value
        dt: Time step

    Returns:
        (rise_time_90, overshoot_pct)
    """
    # Rise time: first time output crosses 0.9 * target
    threshold_90 = 0.9 * target

    rise_idx = np.where(output_history >= threshold_90)[0]

    if len(rise_idx) > 0:
        rise_time_90 = rise_idx[0] * dt
    else:
        # Never reached 90%
        rise_time_90 = np.inf

    # Overshoot: max(output) - target, as percentage of target
    max_output = np.max(output_history)

    if max_output > target:
        overshoot_pct = (max_output - target) / target * 100
    else:
        overshoot_pct = 0.0

    return float(rise_time_90), float(overshoot_pct)


def run_single_trial(
    lambda_c: float,
    horizon: float,
    seed: int,
    trial_id: int,
    config: GoldilocksConfig
) -> GoldilocksTrialResult:
    """Run single trial."""
    np.random.seed(seed)

    controller = SimplePID(kp=1.0, ki=0.5, kd=0.05, lambda_c=lambda_c, dt=config.dt)
    plant = SimplePlant(tau=0.1, dt=config.dt)

    steps = int(horizon / config.dt)
    output_history = np.zeros(steps)
    salience_history = []

    total_energy = 0.0

    for step in range(steps):
        error = config.target - plant.output

        control, energy = controller.step(error)
        total_energy += energy

        plant.step(control)

        output_history[step] = plant.output
        salience_history.append(controller.salience)

    # Compute metrics
    rise_time_90, overshoot_pct = compute_metrics(output_history, config.target, config.dt)

    mean_salience = np.mean(salience_history)
    final_output = plant.output

    return GoldilocksTrialResult(
        trial_id=trial_id,
        seed=seed,
        lambda_c=lambda_c,
        horizon=horizon,
        final_output=float(final_output),
        total_energy=float(total_energy),
        mean_salience=float(mean_salience),
        rise_time_90=rise_time_90,
        overshoot_pct=overshoot_pct,
    )


def run_goldilocks_sweep(config: GoldilocksConfig) -> List[GoldilocksTrialResult]:
    """Run full Goldilocks sweep: λ × seeds × horizons."""

    print("\n" + "="*80)
    print("RIGOROUS GOLDILOCKS SWEEP")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Lambda values: {len(config.lambda_values)}")
    print(f"  Seeds per λ: {config.n_seeds}")
    print(f"  Horizons: {config.test_horizons}")
    print(f"  Total trials: {len(config.lambda_values) * config.n_seeds * len(config.test_horizons)}")
    print()

    all_trials = []
    trial_id = 0

    for lambda_c in config.lambda_values:
        print(f"Running λ={lambda_c:.2f}...")

        for seed_idx in range(config.n_seeds):
            for horizon in config.test_horizons:
                seed = 1000 + trial_id

                trial = run_single_trial(lambda_c, horizon, seed, trial_id, config)
                all_trials.append(trial)

                trial_id += 1

    return all_trials


def analyze_goldilocks(trials: List[GoldilocksTrialResult], config: GoldilocksConfig) -> dict:
    """Rigorous analysis with model selection and co-occurrence."""

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    # Aggregate by lambda
    lambda_aggregates = {}

    for lambda_c in config.lambda_values:
        lambda_trials = [t for t in trials if t.lambda_c == lambda_c]

        # Filter for successful trials (reached ≥95% of target)
        successful = [t for t in lambda_trials if t.final_output >= 0.95]

        if len(successful) == 0:
            # No successful trials
            lambda_aggregates[lambda_c] = {
                "success_rate": 0.0,
                "mean_energy": np.nan,
                "std_energy": np.nan,
                "mean_overshoot": np.nan,
                "std_overshoot": np.nan,
                "mean_rise_time": np.nan,
                "std_rise_time": np.nan,
                "mean_salience": np.nan,
                "n_successful": 0,
            }
        else:
            success_rate = len(successful) / len(lambda_trials)

            lambda_aggregates[lambda_c] = {
                "success_rate": success_rate,
                "mean_energy": np.mean([t.total_energy for t in successful]),
                "std_energy": np.std([t.total_energy for t in successful]),
                "mean_overshoot": np.mean([t.overshoot_pct for t in successful]),
                "std_overshoot": np.std([t.overshoot_pct for t in successful]),
                "mean_rise_time": np.mean([t.rise_time_90 for t in successful]),
                "std_rise_time": np.std([t.rise_time_90 for t in successful]),
                "mean_salience": np.mean([t.mean_salience for t in successful]),
                "n_successful": len(successful),
            }

    # Extract arrays for model selection
    lambda_array = np.array([lam for lam in config.lambda_values if not np.isnan(lambda_aggregates[lam]["mean_energy"])])
    energy_array = np.array([lambda_aggregates[lam]["mean_energy"] for lam in lambda_array])
    overshoot_array = np.array([lambda_aggregates[lam]["mean_overshoot"] for lam in lambda_array])
    rise_time_array = np.array([lambda_aggregates[lam]["mean_rise_time"] for lam in lambda_array])

    print("\n1. Model Selection: Piecewise vs Smooth")
    print("-" * 80)

    # Test energy(λ)
    if len(lambda_array) >= 5:
        energy_model = compare_piecewise_vs_smooth(lambda_array, energy_array)

        print(f"\nEnergy(λ):")
        print(f"  Smooth AIC: {energy_model.smooth_aic:.2f}")
        print(f"  Piecewise AIC: {energy_model.piecewise_aic:.2f}")
        print(f"  ΔAIC: {energy_model.delta_aic:.2f} (positive = piecewise better)")

        if energy_model.prefers_piecewise and energy_model.is_strong_evidence():
            print(f"  ✓ STRONG EVIDENCE for phase transition at λ≈{energy_model.break_point:.2f}")
        elif energy_model.prefers_piecewise:
            print(f"  ⚠️  Weak evidence for transition at λ≈{energy_model.break_point:.2f}")
        else:
            print(f"  ✗ No phase transition (smooth model preferred)")

        # Test overshoot(λ)
        overshoot_model = compare_piecewise_vs_smooth(lambda_array, overshoot_array)

        print(f"\nOvershoot(λ):")
        print(f"  Smooth AIC: {overshoot_model.smooth_aic:.2f}")
        print(f"  Piecewise AIC: {overshoot_model.piecewise_aic:.2f}")
        print(f"  ΔAIC: {overshoot_model.delta_aic:.2f}")

        if overshoot_model.prefers_piecewise and overshoot_model.is_strong_evidence():
            print(f"  ✓ STRONG EVIDENCE for transition at λ≈{overshoot_model.break_point:.2f}")
        elif overshoot_model.prefers_piecewise:
            print(f"  ⚠️  Weak evidence for transition at λ≈{overshoot_model.break_point:.2f}")
        else:
            print(f"  ✗ No transition")

        # Co-occurrence check
        print("\n2. Co-Occurrence Check (≥2 order parameters)")
        print("-" * 80)

        transitions_detected = []
        if energy_model.prefers_piecewise and energy_model.break_point is not None:
            transitions_detected.append(("energy", energy_model.break_point))
        if overshoot_model.prefers_piecewise and overshoot_model.break_point is not None:
            transitions_detected.append(("overshoot", overshoot_model.break_point))

        print(f"Transitions detected: {len(transitions_detected)}")
        for name, break_pt in transitions_detected:
            print(f"  - {name}: λ≈{break_pt:.2f}")

        if len(transitions_detected) >= 2:
            # Check if breakpoints are close (within 20% of range)
            breaks = [b for _, b in transitions_detected]
            break_range = np.max(breaks) - np.min(breaks)
            total_range = np.max(lambda_array) - np.min(lambda_array)

            if break_range / total_range < 0.2:
                print(f"\n  ✓ CO-OCCURRENCE confirmed (breaks within {break_range:.2f} of each other)")
                co_occurrence = True
            else:
                print(f"\n  ⚠️  Breaks too far apart ({break_range:.2f}), no co-occurrence")
                co_occurrence = False
        else:
            print(f"\n  ✗ Insufficient order parameters (need ≥2)")
            co_occurrence = False

    else:
        print("⚠️  Insufficient data points for model selection")
        energy_model = None
        overshoot_model = None
        co_occurrence = False

    # 3. Stability check
    print("\n3. Stability Testing")
    print("-" * 80)

    # Check success rate at optimal λ (if exists)
    successful_lambda_values = [lam for lam in config.lambda_values
                                 if lambda_aggregates[lam]["success_rate"] >= 0.70]

    if len(successful_lambda_values) > 0:
        print(f"Lambda values with ≥70% success: {successful_lambda_values}")

        # Find optimal (min energy among stable)
        optimal_lambda = min(successful_lambda_values,
                             key=lambda lam: lambda_aggregates[lam]["mean_energy"])

        print(f"\n✓ OPTIMAL λ≈{optimal_lambda:.2f}")
        print(f"  Success rate: {lambda_aggregates[optimal_lambda]['success_rate']:.1%}")
        print(f"  Mean energy: {lambda_aggregates[optimal_lambda]['mean_energy']:.3f}")
        print(f"  Mean overshoot: {lambda_aggregates[optimal_lambda]['mean_overshoot']:.1f}%")
        print(f"  Mean rise time: {lambda_aggregates[optimal_lambda]['mean_rise_time']:.3f}s")

    else:
        print("⚠️  No lambda value achieved ≥70% stability")
        optimal_lambda = None

    # 4. Summary table
    print("\n4. Summary Table")
    print("-" * 80)

    table_data = []
    for lam in config.lambda_values:
        agg = lambda_aggregates[lam]

        if np.isnan(agg["mean_energy"]):
            table_data.append([
                f"{lam:.2f}",
                f"{agg['success_rate']:.1%}",
                "—",
                "—",
                "—",
            ])
        else:
            table_data.append([
                f"{lam:.2f}",
                f"{agg['success_rate']:.1%}",
                f"{agg['mean_energy']:.2f} ± {agg['std_energy']:.2f}",
                f"{agg['mean_overshoot']:.1f} ± {agg['std_overshoot']:.1f}",
                f"{agg['mean_rise_time']:.2f} ± {agg['std_rise_time']:.2f}",
            ])

    print("\n" + tabulate(
        table_data,
        headers=["λ", "Success%", "Energy", "Overshoot%", "Rise Time (s)"],
        tablefmt="grid"
    ))

    # Verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    if energy_model and energy_model.is_strong_evidence() and co_occurrence and optimal_lambda is not None:
        print("✓ GOLDILOCKS ZONE CONFIRMED")
        print(f"  - Piecewise model strongly preferred (ΔAIC > 10)")
        print(f"  - Co-occurrence: ≥2 order parameters transition together")
        print(f"  - Optimal λ≈{optimal_lambda:.2f} with ≥70% stability")
        print("\n  CONCLUSION: Working finding. Not yet a phase transition proof.")
        print("  Requires: thermodynamic limit, universality class, critical exponents.")
    else:
        print("⚠️  GOLDILOCKS ZONE NOT ESTABLISHED")
        if not energy_model or not energy_model.is_strong_evidence():
            print("  ✗ Model selection does not strongly favor piecewise (ΔAIC ≤ 10)")
        if not co_occurrence:
            print("  ✗ Co-occurrence requirement not met (<2 order parameters)")
        if optimal_lambda is None:
            print("  ✗ No lambda achieved ≥70% stability")
        print("\n  CONCLUSION: Optimization hint, not a phase boundary.")

    # Prepare output
    analysis_result = {
        "lambda_aggregates": {str(k): v for k, v in lambda_aggregates.items()},
        "model_selection": {
            "energy": {
                "break_point": float(energy_model.break_point) if energy_model and energy_model.break_point else None,
                "delta_aic": float(energy_model.delta_aic) if energy_model else None,
                "prefers_piecewise": energy_model.prefers_piecewise if energy_model else False,
                "strong_evidence": energy_model.is_strong_evidence() if energy_model else False,
            } if energy_model else None,
            "overshoot": {
                "break_point": float(overshoot_model.break_point) if overshoot_model and overshoot_model.break_point else None,
                "delta_aic": float(overshoot_model.delta_aic) if overshoot_model else None,
                "prefers_piecewise": overshoot_model.prefers_piecewise if overshoot_model else False,
                "strong_evidence": overshoot_model.is_strong_evidence() if overshoot_model else False,
            } if overshoot_model else None,
        },
        "co_occurrence": co_occurrence,
        "optimal_lambda": float(optimal_lambda) if optimal_lambda is not None else None,
    }

    return analysis_result


def main():
    """Run rigorous Goldilocks experiment."""

    config = GoldilocksConfig(
        lambda_values=[0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0],
        n_seeds=10,
        test_horizons=[10.0, 15.0, 20.0],
        dt=0.01,
    )

    # Run sweep
    trials = run_goldilocks_sweep(config)

    # Analyze
    analysis = analyze_goldilocks(trials, config)

    # Save
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = ARTIFACT_DIR / f"goldilocks_rigorous_{timestamp}.json"

    output = {
        "experiment": "salience_goldilocks_rigorous",
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": str(uuid.uuid4()),
        "code_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
        "config": config.to_dict(),
        "trials": [asdict(t) for t in trials],
        "analysis": analysis,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    return output


if __name__ == "__main__":
    main()
