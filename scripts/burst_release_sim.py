"""Experiment H: Burst release / stored inertia collapse test."""

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

ARTIFACT_DIR = Path("artifacts/burst_release")


@dataclass
class PlantConfig:
    dt: float = 0.01
    horizon: float = 6.0
    tau: float = 0.1
    release_time: float = 1.5
    analysis_window: float = 0.5


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
    lambda_core_phase1: float = 20.0
    lambda_core_phase2: float = 0.0
    mu_c_phase2: float = 0.0
    assist_cap: float = 2.5
    salience_floor: float = 0.0
    recovery_tax: float = 0.2


class BurstController:
    def __init__(self, cfg: ControllerConfig, dt: float) -> None:
        self.cfg = cfg
        self.dt = dt
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
        self.retention[idx] = 0.92 * self.retention[idx] + 0.08 * novelty
        payoff = math.exp(-abs(error))
        self.payoff[idx] = 0.9 * self.payoff[idx] + 0.1 * payoff
        self.fatigue[idx] = (
            self.cfg.fatigue_decay * self.fatigue[idx]
            + self.cfg.fatigue_gain * abs(delta) / scale
        )
        phi = min(self.fatigue[idx], 0.95)
        salience = novelty * np.clip(self.retention[idx], 0.2, 1.2) * np.clip(self.payoff[idx], 0.2, 1.2) * (1.0 - phi)
        self.salience[idx] = float(np.clip(salience, 1e-3, 1.5))

    def _lambda_core(self, phase2: bool) -> float:
        return self.cfg.lambda_core_phase2 if phase2 else self.cfg.lambda_core_phase1

    def _assist_factor(
        self, error: float, mean_salience: float, mean_delta_salience: float, phase2: bool, telemetry: GateTelemetry
    ) -> float:
        if not phase2 or self.cfg.mu_c_phase2 <= 0.0:
            if self.cfg.salience_floor > 0.0 and mean_salience < self.cfg.salience_floor:
                gated, _, blocked = gate_subsidy(
                    mean_salience, self.cfg.salience_floor, 1.0, self.cfg.recovery_tax
                )
                telemetry.record(False, blocked)
                return gated
            return 1.0
        prev_err_mag = abs(self.prev_error)
        curr_err_mag = abs(error)
        goal_aligned = prev_err_mag <= 1e-6 or curr_err_mag < prev_err_mag
        salience_coherent = mean_salience >= 0.8
        salience_gain = max(mean_delta_salience, 0.0)
        if not goal_aligned:
            return 1.0
        if salience_gain <= 0.0 and not salience_coherent:
            gated, _, blocked = gate_subsidy(mean_salience, self.cfg.salience_floor, 1.0, self.cfg.recovery_tax)
            telemetry.record(False, blocked)
            return gated
        coherence_term = 0.35 if salience_coherent else 0.0
        strength = float(np.clip(salience_gain + coherence_term, 0.0, self.cfg.assist_cap))
        desired = 1.0 + self.cfg.mu_c_phase2 * strength
        max_boost = 1.0 + self.cfg.mu_c_phase2 * self.cfg.assist_cap
        desired = float(np.clip(desired, 1.0, max_boost))
        gated, accelerated, blocked = gate_subsidy(
            mean_salience, self.cfg.salience_floor, desired, self.cfg.recovery_tax
        )
        telemetry.record(accelerated, blocked)
        return gated

    def step(
        self, target: float, measurement: float, phase2: bool, telemetry: GateTelemetry
    ) -> Dict[str, float]:
        error = target - measurement
        lambda_core = self._lambda_core(phase2)

        int_mass = 1.0 + lambda_core * 0.3 * self.salience[0]
        raw_int_delta = error * self.dt
        applied_int_delta = raw_int_delta / int_mass
        self.integral = float(
            np.clip(self.integral + applied_int_delta, -self.cfg.integral_clip, self.cfg.integral_clip)
        )
        prev_sal_int = self.salience[0]
        self._update_salience(0, applied_int_delta, error)
        delta_sal_int = self.salience[0] - prev_sal_int

        raw_derivative = (error - self.prev_error) / self.dt
        deriv_delta = raw_derivative - self.derivative_est
        deriv_mass = 1.0 + lambda_core * 0.3 * self.salience[1]
        self.derivative_est += self.cfg.smoothing_alpha * (deriv_delta / deriv_mass)
        prev_sal_der = self.salience[1]
        self._update_salience(1, deriv_delta, error)
        delta_sal_der = self.salience[1] - prev_sal_der

        base_control = (
            self.cfg.kp * error
            + self.cfg.ki * self.integral
            + self.cfg.kd * self.derivative_est
        )

        mean_salience = float(np.mean(self.salience))
        mean_delta_salience = 0.5 * (delta_sal_int + delta_sal_der)
        assist_factor = self._assist_factor(error, mean_salience, mean_delta_salience, phase2, telemetry)

        if assist_factor > 1.0:
            self.fatigue *= 0.92
            self.retention = np.clip(0.96 * self.retention + 0.04 * self.salience, 0.2, 1.2)

        self.prev_error = error
        control = base_control * assist_factor
        return {
            "control": float(control),
            "salience_mean": mean_salience,
            "assist_factor": assist_factor,
        }


def simulate(cfg: ControllerConfig, plant_cfg: PlantConfig) -> Dict[str, np.ndarray | float]:
    controller = BurstController(cfg, plant_cfg.dt)
    controller.reset()

    steps = int(plant_cfg.horizon / plant_cfg.dt)
    release_idx = int(plant_cfg.release_time / plant_cfg.dt)
    analysis_steps = int(plant_cfg.analysis_window / plant_cfg.dt)

    target = np.ones(steps, dtype=float)
    state = 0.0

    outputs = np.zeros(steps, dtype=float)
    controls = np.zeros(steps, dtype=float)
    salience_trace = np.zeros(steps, dtype=float)
    assist_trace = np.ones(steps, dtype=float)

    control_energy_phase1 = 0.0
    control_energy_phase2 = 0.0
    telemetry = GateTelemetry(floor=cfg.salience_floor)

    for i in range(steps):
        phase2 = i >= release_idx
        outputs[i] = state
        result = controller.step(target[i], state, phase2, telemetry)
        controls[i] = result["control"]
        salience_trace[i] = result["salience_mean"]
        assist_trace[i] = result["assist_factor"]
        energy_term = abs(controls[i]) * plant_cfg.dt
        if phase2:
            control_energy_phase2 += energy_term
        else:
            control_energy_phase1 += energy_term
        state += plant_cfg.dt * ((-state + controls[i]) / plant_cfg.tau)

    time = np.linspace(0.0, plant_cfg.horizon, steps, endpoint=False)
    errors = target - outputs

    post_outputs = outputs[release_idx:]
    post_time = time[release_idx:] - plant_cfg.release_time

    rise_time_post = compute_rise_time(post_outputs, 1.0, plant_cfg.dt)
    peak_rate = compute_peak_rate(post_outputs, plant_cfg.dt, analysis_steps)
    overshoot = compute_peak_overshoot(post_outputs, analysis_steps)
    mean_salience_phase2 = float(np.mean(salience_trace[release_idx:]))
    min_salience_phase2 = float(np.min(salience_trace[release_idx:]))

    return {
        "time": time,
        "output": outputs,
        "control": controls,
        "salience": salience_trace,
        "assist_factor": assist_trace,
        "errors": errors,
        "rise_time_post": rise_time_post,
        "peak_rate_post": peak_rate,
        "overshoot_post": overshoot,
        "mean_salience_phase2": mean_salience_phase2,
        "min_salience_phase2": min_salience_phase2,
        "acceleration_pct": telemetry.acceleration_pct,
        "blocked_pct": telemetry.blocked_pct,
        "control_energy_phase1": control_energy_phase1,
        "control_energy_phase2": control_energy_phase2,
    }


def compute_rise_time(outputs: np.ndarray, target: float, dt: float) -> float:
    threshold = 0.9 * target
    indices = np.where(outputs >= threshold)[0]
    if indices.size == 0:
        return float("nan")
    return float(indices[0] * dt)


def compute_peak_rate(outputs: np.ndarray, dt: float, window_steps: int) -> float:
    diffs = np.diff(outputs, prepend=outputs[0]) / dt
    return float(np.max(diffs[:window_steps]))


def compute_peak_overshoot(outputs: np.ndarray, window_steps: int) -> float:
    window = outputs[:window_steps]
    return float(np.max(window) - 1.0)


def run_experiment() -> List[Dict[str, float]]:
    plant_cfg = PlantConfig()
    base_cfg = ControllerConfig(mu_c_phase2=0.0)
    subsidy_cfg = ControllerConfig(mu_c_phase2=2.0)

    traces_base = simulate(base_cfg, plant_cfg)
    traces_subsidy = simulate(subsidy_cfg, plant_cfg)

    base_metrics = collect_metrics(traces_base, "release_only")
    subsidy_metrics = collect_metrics(traces_subsidy, "release_plus_subsidy")

    return [base_metrics, subsidy_metrics]


def collect_metrics(traces: Dict[str, np.ndarray | float], label: str) -> Dict[str, float]:
    return {
        "scenario": label,
        "rise_time_post": float(traces["rise_time_post"]),
        "post_release_peak_rate": float(traces["peak_rate_post"]),
        "post_release_overshoot": float(traces["overshoot_post"]),
        "mean_salience_phase2": float(traces["mean_salience_phase2"]),
        "min_salience_phase2": float(traces["min_salience_phase2"]),
        "acceleration_pct": float(traces["acceleration_pct"]),
        "blocked_pct": float(traces["blocked_pct"]),
        "control_energy_phase1": float(traces["control_energy_phase1"]),
        "control_energy_phase2": float(traces["control_energy_phase2"]),
    }


def write_artifact(records: List[Dict[str, float]]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())
    payload = []
    for record in records:
        payload.append(
            {
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "experiment_name": "experiment_h_burst_release",
                "run_id": run_id,
                **record,
            }
        )
    path = ARTIFACT_DIR / f"burst_release_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def print_summary(records: List[Dict[str, float]]) -> None:
    from tabulate import tabulate

    rows = [
        (
            record["scenario"],
            record["rise_time_post"],
            record["post_release_peak_rate"],
            record["post_release_overshoot"],
            record["mean_salience_phase2"],
            record["control_energy_phase1"],
            record["control_energy_phase2"],
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
                "overshoot_post",
                "mean_salience_phase2",
                "min_salience_phase2",
                "accel_pct",
                "blocked_pct",
                "energy_phase1",
                "energy_phase2",
            ],
            tablefmt="github",
            floatfmt=".6f",
        )
    )


def main() -> None:
    records = run_experiment()
    artifact = write_artifact(records)
    print("=== Experiment H: burst release ===")
    print_summary(records)
    print(f"Results written to {artifact}")


if __name__ == "__main__":
    main()
