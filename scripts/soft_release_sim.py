"""Experiment M: Soft burst release with salience floor gating."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from salience_floor_gate import GateTelemetry, gate_subsidy

ARTIFACT_DIR = Path("artifacts/soft_release")


@dataclass
class PlantConfig:
    dt: float = 0.01
    horizon: float = 6.0
    tau: float = 0.1
    release_time: float = 1.5
    release_duration: float = 1.5
    target: float = 1.0


@dataclass
class ControllerConfig:
    kp: float = 3.0
    ki: float = 1.6
    kd: float = 0.12
    salience_scale: float = 0.05
    fatigue_decay: float = 0.9
    fatigue_gain: float = 0.08
    smoothing_alpha: float = 0.24
    integral_clip: float = 22.0
    lambda_start: float = 20.0
    lambda_end: float = 0.0
    mu_start: float = 0.0
    mu_end: float = 2.0
    assistance_cap: float = 2.5
    salience_floor: float = 0.7
    recovery_tax: float = 0.2


class SoftReleaseController:
    def __init__(self, cfg: ControllerConfig, plant_cfg: PlantConfig) -> None:
        self.cfg = cfg
        self.plant_cfg = plant_cfg
        self.dt = plant_cfg.dt
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.derivative_est = 0.0
        self.prev_error = 0.0
        self.salience = np.ones(2, dtype=float)
        self.retention = np.full(2, 0.9, dtype=float)
        self.payoff = np.full(2, 0.9, dtype=float)
        self.fatigue = np.zeros(2, dtype=float)

    def _update_salience(self, idx: int, delta: float, error: float) -> None:
        scale = max(self.cfg.salience_scale, 1e-6)
        novelty = math.exp(-abs(delta) / scale)
        self.retention[idx] = 0.9 * self.retention[idx] + 0.1 * novelty
        payoff = math.exp(-abs(error))
        self.payoff[idx] = 0.9 * self.payoff[idx] + 0.1 * payoff
        self.fatigue[idx] = self.cfg.fatigue_decay * self.fatigue[idx] + self.cfg.fatigue_gain * abs(delta) / scale
        phi = min(self.fatigue[idx], 0.95)
        salience_raw = novelty * np.clip(self.retention[idx], 0.2, 1.2) * np.clip(self.payoff[idx], 0.2, 1.2) * (1.0 - phi)
        self.salience[idx] = float(np.clip(0.65 + 0.55 * salience_raw, 0.65, 1.5))

    def _scheduled_params(self, step: int) -> Dict[str, float]:
        release_idx = int(self.plant_cfg.release_time / self.dt)
        duration_steps = max(int(self.plant_cfg.release_duration / self.dt), 1)
        if step < release_idx:
            return {
                "lambda_core": self.cfg.lambda_start,
                "mu_c": self.cfg.mu_start,
                "phase2": False,
            }
        alpha = min(max((step - release_idx) / duration_steps, 0.0), 1.0)
        lambda_core = (1 - alpha) * self.cfg.lambda_start + alpha * self.cfg.lambda_end
        mu_c = (1 - alpha) * self.cfg.mu_start + alpha * self.cfg.mu_end
        return {
            "lambda_core": lambda_core,
            "mu_c": mu_c,
            "phase2": True,
        }

    def step(self, target: float, measurement: float, step: int, telemetry: GateTelemetry) -> Dict[str, float]:
        params = self._scheduled_params(step)
        error = target - measurement

        int_mass = 1.0 + params["lambda_core"] * 0.3 * self.salience[0]
        raw_int_delta = error * self.dt
        applied_int_delta = raw_int_delta / int_mass
        self.integral = float(np.clip(self.integral + applied_int_delta, -self.cfg.integral_clip, self.cfg.integral_clip))
        prev_sal_int = self.salience[0]
        self._update_salience(0, applied_int_delta, error)
        delta_sal_int = self.salience[0] - prev_sal_int

        raw_derivative = (error - self.prev_error) / self.dt
        deriv_delta = raw_derivative - self.derivative_est
        deriv_mass = 1.0 + params["lambda_core"] * 0.3 * self.salience[1]
        self.derivative_est += self.cfg.smoothing_alpha * (deriv_delta / deriv_mass)
        prev_sal_der = self.salience[1]
        self._update_salience(1, deriv_delta, error)
        delta_sal_der = self.salience[1] - prev_sal_der

        base_control = self.cfg.kp * error + self.cfg.ki * self.integral + self.cfg.kd * self.derivative_est

        mean_salience = float(np.mean(self.salience))
        mean_delta_salience = 0.5 * (delta_sal_int + delta_sal_der)

        multiplier = 1.0
        if params["phase2"] and params["mu_c"] > 0.0:
            goal_aligned = abs(error) < abs(self.prev_error) or abs(self.prev_error) < 1e-6
            coherence_term = 0.0
            if mean_salience >= 0.8:
                coherence_term = 0.35
            strength = max(mean_delta_salience, 0.0) + coherence_term if goal_aligned else 0.0
            strength = float(np.clip(strength, 0.0, self.cfg.assistance_cap))
            desired = 1.0 + params["mu_c"] * strength
            desired = float(np.clip(desired, 1.0, 1.0 + params["mu_c"] * self.cfg.assistance_cap))
            multiplier, accelerated, blocked = gate_subsidy(
                mean_salience, self.cfg.salience_floor, desired, self.cfg.recovery_tax
            )
            telemetry.record(accelerated, blocked)
        else:
            multiplier, _, blocked = gate_subsidy(mean_salience, self.cfg.salience_floor, 1.0, self.cfg.recovery_tax)
            telemetry.record(False, blocked)

        control = base_control * multiplier
        self.prev_error = error
        return {
            "control": float(control),
            "mean_salience": mean_salience,
            "multiplier": multiplier,
        }


def simulate(cfg: ControllerConfig, plant_cfg: PlantConfig) -> Dict[str, np.ndarray | float]:
    controller = SoftReleaseController(cfg, plant_cfg)
    controller.reset()

    steps = int(plant_cfg.horizon / plant_cfg.dt)
    target = np.full(steps, plant_cfg.target, dtype=float)
    state = 0.0

    outputs = np.zeros(steps, dtype=float)
    controls = np.zeros(steps, dtype=float)
    salience_trace = np.zeros(steps, dtype=float)
    multiplier_trace = np.ones(steps, dtype=float)

    energy_phase1 = 0.0
    energy_phase2 = 0.0
    telemetry = GateTelemetry(floor=cfg.salience_floor)

    for i in range(steps):
        outputs[i] = state
        result = controller.step(target[i], state, i, telemetry)
        controls[i] = result["control"]
        salience_trace[i] = result["mean_salience"]
        multiplier_trace[i] = result["multiplier"]

        energy_term = abs(controls[i]) * plant_cfg.dt
        if i * plant_cfg.dt < plant_cfg.release_time:
            energy_phase1 += energy_term
        else:
            energy_phase2 += energy_term

        state += plant_cfg.dt * ((-state + controls[i]) / plant_cfg.tau)

    time = np.linspace(0.0, plant_cfg.horizon, steps, endpoint=False)
    errors = target - outputs

    release_idx = int(plant_cfg.release_time / plant_cfg.dt)
    post_outputs = outputs[release_idx:]

    rise_time_post = compute_rise_time(post_outputs, plant_cfg.target, plant_cfg.dt)
    peak_rate_post = compute_peak_rate(post_outputs, plant_cfg.dt, int(plant_cfg.release_duration / plant_cfg.dt))
    mean_salience_phase2 = float(np.mean(salience_trace[release_idx:]))
    min_salience_phase2 = float(np.min(salience_trace[release_idx:]))

    return {
        "time": time,
        "output": outputs,
        "control": controls,
        "salience": salience_trace,
        "multiplier": multiplier_trace,
        "errors": errors,
        "rise_time_post": rise_time_post,
        "peak_rate_post": peak_rate_post,
        "mean_salience_phase2": mean_salience_phase2,
        "min_salience_phase2": min_salience_phase2,
        "energy_phase1": energy_phase1,
        "energy_phase2": energy_phase2,
        "acceleration_pct": telemetry.acceleration_pct,
        "blocked_pct": telemetry.blocked_pct,
    }


def compute_rise_time(outputs: np.ndarray, target: float, dt: float, threshold: float = 0.9) -> float:
    cutoff = threshold * target
    for idx, value in enumerate(outputs):
        if value >= cutoff:
            return idx * dt
    return float("nan")


def compute_peak_rate(outputs: np.ndarray, dt: float, window_steps: int) -> float:
    diffs = np.diff(outputs, prepend=outputs[0]) / dt
    window = diffs[:window_steps]
    return float(np.max(window)) if len(window) else float("nan")


def run_experiment() -> List[Dict[str, float]]:
    plant_cfg = PlantConfig()
    hard_cfg = ControllerConfig(mu_end=0.0)  # reference: pure release without subsidy
    soft_cfg = ControllerConfig()

    traces_hard = simulate(hard_cfg, plant_cfg)
    traces_soft = simulate(soft_cfg, plant_cfg)

    metrics_hard = collect_metrics(traces_hard, "hard_release")
    metrics_soft = collect_metrics(traces_soft, "soft_release")

    return [metrics_hard, metrics_soft]


def collect_metrics(traces: Dict[str, np.ndarray | float], label: str) -> Dict[str, float]:
    rise_time = float(traces["rise_time_post"])
    peak_rate = float(traces["peak_rate_post"])
    record = {
        "scenario": label,
        "configuration": label,
        "rise_time_post": rise_time,
        "rise_time_90": rise_time,
        "peak_rate_post": peak_rate,
        "post_release_peak_rate": peak_rate,
        "mean_salience_phase2": float(traces["mean_salience_phase2"]),
        "min_salience_phase2": float(traces["min_salience_phase2"]),
        "energy_phase1": float(traces["energy_phase1"]),
        "energy_phase2": float(traces["energy_phase2"]),
        "acceleration_pct": float(traces["acceleration_pct"]),
        "blocked_pct": float(traces["blocked_pct"]),
    }
    return record


def write_artifact(records: List[Dict[str, float]]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())
    payload = []
    for record in records:
        payload.append(
            {
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "experiment_name": "experiment_m_soft_release",
                "run_id": run_id,
                **record,
            }
        )
    path = ARTIFACT_DIR / f"soft_release_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def print_summary(records: List[Dict[str, float]]) -> None:
    from tabulate import tabulate

    rows = [
        (
            record["scenario"],
            record["rise_time_post"],
            record["peak_rate_post"],
            record["mean_salience_phase2"],
            record["min_salience_phase2"],
            record["energy_phase1"],
            record["energy_phase2"],
            record["acceleration_pct"],
            record["blocked_pct"],
        )
        for record in records
    ]
    print(
        tabulate(
            rows,
            headers=[
                "scenario",
                "rise_time_post",
                "peak_rate_post",
                "mean_salience_phase2",
                "min_salience_phase2",
                "energy_phase1",
                "energy_phase2",
                "accel_pct",
                "blocked_pct",
            ],
            tablefmt="github",
            floatfmt=".6f",
        )
    )


def main() -> None:
    records = run_experiment()
    artifact = write_artifact(records)
    print("=== Experiment M: soft burst release ===")
    print_summary(records)
    print(f"Results written to {artifact}")


if __name__ == "__main__":
    main()
