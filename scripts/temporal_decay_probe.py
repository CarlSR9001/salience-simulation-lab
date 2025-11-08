"""Experiment R: Temporal Energy Decay Probe.

Long-horizon controller with exponential fatigue amplification and scheduled perturbations.
Tests whether energy_ratio decreases over time while salience remains stable.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ARTIFACT_DIR = Path("artifacts/temporal_decay")


@dataclass
class ProbeConfig:
    dt: float = 0.01
    horizon: float = 30.0
    tau: float = 0.1
    kp: float = 2.0
    ki: float = 0.5
    kd: float = 0.1
    fatigue_decay: float = 0.85
    fatigue_gain: float = 0.15
    fatigue_amplification: float = 6.0
    salience_scale: float = 0.01
    mass_scale: float = 2.0
    integral_clip: float = 20.0
    perturbation_times: Tuple[float, ...] = (3.0, 7.0, 13.0, 17.0, 23.0, 27.0)
    perturbation_magnitude: float = 0.6
    perturbation_duration: float = 0.5
    seed: int = 2025


class TemporalController:
    def __init__(self, cfg: ProbeConfig, lambda_c: float) -> None:
        self.cfg = cfg
        self.lambda_c = lambda_c
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.derivative_est = 0.0
        self.prev_error = 0.0
        self.prev_control = 0.0
        self.salience = 1.5
        self.retention = 1.0
        self.payoff = 1.0
        self.fatigue = 0.0

    def step(self, target: float, measurement: float, time_now: float) -> Dict[str, float]:
        error = target - measurement

        # Exponential fatigue amplification
        fatigue_scale = math.exp(self.cfg.fatigue_amplification * self.fatigue)

        # Integral with continuity tax
        int_mass = 1.0 + self.lambda_c * self.cfg.mass_scale * self.salience * fatigue_scale
        raw_int_delta = error * self.cfg.dt
        applied_int_delta = raw_int_delta / int_mass
        self.integral = float(
            np.clip(self.integral + applied_int_delta, -self.cfg.integral_clip, self.cfg.integral_clip)
        )

        # Derivative with continuity tax
        raw_derivative = (error - self.prev_error) / self.cfg.dt
        deriv_delta = raw_derivative - self.derivative_est
        deriv_mass = 1.0 + self.lambda_c * self.cfg.mass_scale * self.salience * fatigue_scale
        self.derivative_est += 0.2 * (deriv_delta / deriv_mass)

        # Control signal
        control = (
            self.cfg.kp * error
            + self.cfg.ki * self.integral
            + self.cfg.kd * self.derivative_est
        )

        # Update salience
        control_delta = abs(control - self.prev_control)
        scale = max(self.cfg.salience_scale, 1e-6)
        delta_norm = control_delta / scale

        novelty_decay = math.exp(-delta_norm)
        novelty_gain = 1.0 - novelty_decay

        self.retention = 0.9 * self.retention + 0.1 * novelty_decay
        payoff_sample = math.exp(-abs(error))
        self.payoff = 0.9 * self.payoff + 0.1 * payoff_sample

        # Exponentially amplified fatigue with time-dependent accumulation
        time_fatigue = (time_now / self.cfg.horizon) ** 2.0  # Accelerating accumulation
        self.fatigue = (
            self.cfg.fatigue_decay * self.fatigue
            + self.cfg.fatigue_gain * delta_norm
            + 0.35 * time_fatigue  # Extreme time-based accumulation
        )
        phi = min(self.fatigue, 0.95)

        # Salience computation with higher baseline
        value = (
            4.0 * novelty_gain
            + 0.5 * float(np.clip(self.retention, 0.6, 1.3))
            + 0.4 * float(np.clip(self.payoff, 0.6, 1.3))
        )
        gate = max(0.6, 1.0 / (1.0 + math.exp(-1.5 * (value - 0.1))))
        self.salience = float(np.clip(value * gate * (1.0 - 0.15 * phi), 0.4, 2.0))

        self.prev_error = error
        self.prev_control = control

        return {
            "control": float(control),
            "salience": self.salience,
            "fatigue": self.fatigue,
            "fatigue_scale": fatigue_scale,
        }


def simulate(cfg: ProbeConfig, lambda_c: float) -> Dict[str, object]:
    controller = TemporalController(cfg, lambda_c)
    controller.reset()

    steps = int(cfg.horizon / cfg.dt)
    target_base = 1.0
    state = 0.0

    outputs = np.zeros(steps)
    errors = np.zeros(steps)
    salience_trace = np.zeros(steps)
    fatigue_trace = np.zeros(steps)
    control_energy_trace = np.zeros(steps)
    perturbation_trace = np.zeros(steps)

    control_energy = 0.0

    for i in range(steps):
        time_now = i * cfg.dt

        # Scheduled perturbations
        target = target_base
        perturbation_active = False
        for perturb_time in cfg.perturbation_times:
            if perturb_time <= time_now < perturb_time + cfg.perturbation_duration:
                target += cfg.perturbation_magnitude
                perturbation_active = True
                break

        perturbation_trace[i] = 1.0 if perturbation_active else 0.0

        result = controller.step(target, state, time_now)
        control = result["control"]

        # Plant dynamics
        state += cfg.dt * ((-state + control) / cfg.tau)

        outputs[i] = state
        errors[i] = target - state
        salience_trace[i] = result["salience"]
        fatigue_trace[i] = result["fatigue"]

        control_energy_step = abs(control) * cfg.dt
        control_energy += control_energy_step
        control_energy_trace[i] = control_energy_step

    # Compute windowed metrics
    early_end = int(5.0 / cfg.dt)
    mid_end = int(10.0 / cfg.dt)

    windows = {
        "early": (0, early_end),
        "mid": (early_end, mid_end),
        "late": (mid_end, steps),
    }

    windowed_metrics = {}
    for window_name, (start, end) in windows.items():
        window_energy = float(np.sum(control_energy_trace[start:end]))
        window_salience = float(np.mean(salience_trace[start:end]))
        window_fatigue = float(np.mean(fatigue_trace[start:end]))
        window_rms_error = float(np.sqrt(np.mean(np.square(errors[start:end]))))

        windowed_metrics[f"{window_name}_energy"] = window_energy
        windowed_metrics[f"{window_name}_salience"] = window_salience
        windowed_metrics[f"{window_name}_fatigue"] = window_fatigue
        windowed_metrics[f"{window_name}_rms_error"] = window_rms_error

    # Compute cumulative fatigue and salience drift
    cumulative_fatigue = float(np.mean(fatigue_trace))
    salience_drift_rate = float(
        (salience_trace[-1] - salience_trace[0]) / cfg.horizon if cfg.horizon > 0 else 0.0
    )

    # Entropy proxy: variance of control energy trace
    entropy_proxy = float(np.var(control_energy_trace))

    return {
        "lambda_c": lambda_c,
        "total_control_energy": control_energy,
        "mean_salience": float(np.mean(salience_trace)),
        "min_salience": float(np.min(salience_trace)),
        "cumulative_fatigue": cumulative_fatigue,
        "salience_drift_rate": salience_drift_rate,
        "entropy_proxy": entropy_proxy,
        **windowed_metrics,
    }


def run_sweep(cfg: ProbeConfig, lambdas: List[float]) -> List[Dict[str, float]]:
    results = []
    baseline_energy = None
    baseline_early = None
    baseline_mid = None
    baseline_late = None

    for lambda_c in lambdas:
        metrics = simulate(cfg, lambda_c)

        if baseline_energy is None:
            baseline_energy = metrics["total_control_energy"]
            baseline_early = metrics["early_energy"]
            baseline_mid = metrics["mid_energy"]
            baseline_late = metrics["late_energy"]

        # Compute energy ratios
        energy_ratio = metrics["total_control_energy"] / baseline_energy if baseline_energy else 1.0

        early_energy_ratio = metrics["early_energy"] / baseline_early if baseline_early else 1.0
        mid_energy_ratio = metrics["mid_energy"] / baseline_mid if baseline_mid else 1.0
        late_energy_ratio = metrics["late_energy"] / baseline_late if baseline_late else 1.0

        # Temporal decay metric: ratio of late to early energy ratio
        temporal_decay = late_energy_ratio / early_energy_ratio if early_energy_ratio else 1.0

        results.append({
            **metrics,
            "energy_ratio": energy_ratio,
            "early_energy_ratio": early_energy_ratio,
            "mid_energy_ratio": mid_energy_ratio,
            "late_energy_ratio": late_energy_ratio,
            "temporal_decay_factor": temporal_decay,
        })

    return results


def write_artifact(entries: List[Dict[str, float]], cfg: ProbeConfig) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())
    payload = []
    for entry in entries:
        payload.append(
            {
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "experiment_name": "experiment_r_temporal_decay",
                "run_id": run_id,
                "horizon": cfg.horizon,
                "dt": cfg.dt,
                "seed": cfg.seed,
                **entry,
            }
        )
    path = ARTIFACT_DIR / f"temporal_decay_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def main() -> None:
    cfg = ProbeConfig()
    lambdas = [0.0, 5.0, 20.0]

    results = run_sweep(cfg, lambdas)
    artifact = write_artifact(results, cfg)

    print("=== Experiment R: Temporal Energy Decay Probe ===")
    print(f"Horizon: {cfg.horizon}s, dt: {cfg.dt}, Steps: {int(cfg.horizon / cfg.dt)}")
    print(f"Perturbations at: {cfg.perturbation_times}")
    print()

    for entry in results:
        print(f"λ_c={entry['lambda_c']:>5.1f}:")
        print(f"  Total energy: {entry['total_control_energy']:.3f}, ratio: {entry['energy_ratio']:.3f}")
        print(f"  Mean salience: {entry['mean_salience']:.3f}, min: {entry['min_salience']:.3f}")
        print(f"  Cumulative fatigue: {entry['cumulative_fatigue']:.3f}")
        print(f"  Early window: energy={entry['early_energy']:.3f}, ratio={entry['early_energy_ratio']:.3f}, salience={entry['early_salience']:.3f}")
        print(f"  Mid   window: energy={entry['mid_energy']:.3f}, ratio={entry['mid_energy_ratio']:.3f}, salience={entry['mid_salience']:.3f}")
        print(f"  Late  window: energy={entry['late_energy']:.3f}, ratio={entry['late_energy_ratio']:.3f}, salience={entry['late_salience']:.3f}")
        print(f"  Temporal decay factor: {entry['temporal_decay_factor']:.3f}")
        print(f"  Salience drift rate: {entry['salience_drift_rate']:.5f}")
        print()

    # Success criteria check
    print("Success Criteria Check:")
    for entry in results:
        if entry['lambda_c'] > 0:
            late_early_ratio = entry['late_energy_ratio'] / entry['early_energy_ratio'] if entry['early_energy_ratio'] else 1.0
            success = late_early_ratio < 0.5 and entry['late_salience'] >= 0.7
            status = "PASS" if success else "FAIL"
            print(f"  λ_c={entry['lambda_c']:>5.1f}: late/early={late_early_ratio:.3f}, late_salience={entry['late_salience']:.3f} [{status}]")

    print(f"\nResults written to {artifact}")


if __name__ == "__main__":
    main()
