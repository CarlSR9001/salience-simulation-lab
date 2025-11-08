"""Experiment G: Continuity subsidy (superfluid mass test)."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from tabulate import tabulate

ARTIFACT_DIR = Path("artifacts/mass_sweep")


@dataclass
class PlantConfig:
    dt: float = 0.01
    horizon: float = 5.0
    tau: float = 0.1


@dataclass
class ControllerConfig:
    kp: float = 3.0
    ki: float = 1.6
    kd: float = 0.12
    salience_scale: float = 0.05
    fatigue_decay: float = 0.9
    fatigue_gain: float = 0.1
    smoothing_alpha: float = 0.24
    integral_clip: float = 22.0
    mu_c: float = 0.0
    assist_cap: float = 2.5
    salience_floor: float = 0.0
    recovery_tax: float = 0.2


class ContinuitySubsidyPID:
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

    def _update_salience(self, idx: int, delta: float, error: float) -> tuple[float, float]:
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
        delta_a = novelty
        retention_term = np.clip(self.retention[idx], 0.2, 1.1)
        payoff_term = np.clip(self.payoff[idx], 0.2, 1.1)
        salience = delta_a * retention_term * payoff_term * (1.0 - phi)
        prev_salience = self.salience[idx]
        new_salience = float(np.clip(salience, 1e-3, 1.5))
        self.salience[idx] = new_salience
        return new_salience, new_salience - prev_salience

    def _compute_assist_factor(
        self, error: float, mean_salience: float, mean_delta_salience: float
    ) -> tuple[float, bool, bool]:
        if self.cfg.mu_c <= 0.0:
            if self.cfg.salience_floor > 0.0 and mean_salience < self.cfg.salience_floor:
                safe_factor = 1.0 / (1.0 + self.cfg.recovery_tax)
                return safe_factor, False, True
            return 1.0, False, False
        below_floor = self.cfg.salience_floor > 0.0 and mean_salience < self.cfg.salience_floor
        if below_floor:
            safe_factor = 1.0 / (1.0 + self.cfg.recovery_tax)
            return safe_factor, False, True
        prev_error_mag = abs(self.prev_error)
        curr_error_mag = abs(error)
        goal_aligned = prev_error_mag <= 1e-6 or curr_error_mag < prev_error_mag
        salience_coherent = mean_salience >= 0.8
        salience_gain = max(mean_delta_salience, 0.0)
        if not goal_aligned:
            return 1.0, False, False
        if salience_gain <= 0.0 and not salience_coherent:
            return 1.0, False, False
        coherence_term = 0.35 if salience_coherent else 0.0
        strength = salience_gain + coherence_term
        strength = float(np.clip(strength, 0.0, self.cfg.assist_cap))
        boost = 1.0 + self.cfg.mu_c * strength
        max_boost = 1.0 + self.cfg.mu_c * self.cfg.assist_cap
        boost = float(np.clip(boost, 1.0, max_boost))
        return boost, True, False

    def step(self, target: float, measurement: float) -> Dict[str, float]:
        error = target - measurement

        raw_int_delta = error * self.dt
        self.integral = float(
            np.clip(self.integral + raw_int_delta, -self.cfg.integral_clip, self.cfg.integral_clip)
        )
        int_sal, int_delta_sal = self._update_salience(0, raw_int_delta, error)

        raw_derivative = (error - self.prev_error) / self.dt
        deriv_delta = raw_derivative - self.derivative_est
        self.derivative_est += self.cfg.smoothing_alpha * deriv_delta
        der_sal, der_delta_sal = self._update_salience(1, deriv_delta, error)

        base_control = (
            self.cfg.kp * error
            + self.cfg.ki * self.integral
            + self.cfg.kd * self.derivative_est
        )

        mean_salience = float(np.mean(self.salience))
        mean_delta_salience = 0.5 * (int_delta_sal + der_delta_sal)
        assist_factor, assisted, floor_blocked = self._compute_assist_factor(
            error, mean_salience, mean_delta_salience
        )
        if assisted:
            self.fatigue *= 0.92
            self.retention = np.clip(0.96 * self.retention + 0.04 * self.salience, 0.2, 1.2)

        self.prev_error = error
        control = base_control * assist_factor
        return {
            "control": float(control),
            "salience_mean": mean_salience,
            "assist_factor": assist_factor,
            "floor_blocked": floor_blocked,
        }


def simulate(
    mu_c: float,
    plant_cfg: PlantConfig,
    salience_floor: float = 0.0,
    recovery_tax: float = 0.2,
) -> Dict[str, float]:
    controller = ContinuitySubsidyPID(
        ControllerConfig(mu_c=mu_c, salience_floor=salience_floor, recovery_tax=recovery_tax),
        plant_cfg.dt,
    )
    controller.reset()

    steps = int(plant_cfg.horizon / plant_cfg.dt)
    target = np.ones(steps, dtype=float)
    state = 0.0

    outputs = np.zeros(steps, dtype=float)
    salience_trace = np.zeros(steps, dtype=float)
    control_energy = 0.0
    accel_active = 0
    accel_blocked = 0

    for i in range(steps):
        outputs[i] = state
        result = controller.step(target[i], state)
        control = result["control"]
        salience_trace[i] = result["salience_mean"]
        control_energy += abs(control) * plant_cfg.dt
        state += plant_cfg.dt * ((-state + control) / plant_cfg.tau)
        if result["assist_factor"] > 1.0 + 1e-6:
            accel_active += 1
        if result["floor_blocked"]:
            accel_blocked += 1

    errors = target - outputs
    rise_time = compute_rise_time(outputs, target[-1], plant_cfg.dt)
    peak_overshoot = float(np.max(outputs) - target[-1])
    settling = compute_settling_time(outputs, target[-1], plant_cfg.dt)
    rms_error = float(np.sqrt(np.mean(np.square(errors))))
    mean_salience = float(np.mean(salience_trace))
    min_salience = float(np.min(salience_trace))

    return {
        "rise_time_90": rise_time,
        "peak_overshoot": peak_overshoot,
        "settling_time_2pct": settling,
        "rms_error": rms_error,
        "mean_salience": mean_salience,
        "min_salience": min_salience,
        "control_energy": control_energy,
        "acceleration_steps": accel_active,
        "blocked_steps": accel_blocked,
        "total_steps": steps,
    }


def compute_rise_time(outputs: np.ndarray, target: float, dt: float) -> float:
    threshold = 0.9 * target
    indices = np.where(outputs >= threshold)[0]
    if indices.size == 0:
        return float("nan")
    return float(indices[0] * dt)


def compute_settling_time(outputs: np.ndarray, target: float, dt: float, tol: float = 0.02) -> float:
    upper = target + tol
    lower = target - tol
    for idx in range(len(outputs)):
        window = outputs[idx:]
        if window.size == 0:
            break
        if np.all((window >= lower) & (window <= upper)):
            return float(idx * dt)
    return float("nan")


def run_subsidy_sweep(
    coeffs: Iterable[float], salience_floor: float = 0.0, recovery_tax: float = 0.2
) -> List[Dict[str, float]]:
    plant_cfg = PlantConfig()
    results: List[Dict[str, float]] = []
    baseline_rise: float | None = None

    for mu in coeffs:
        metrics = simulate(mu, plant_cfg, salience_floor=salience_floor, recovery_tax=recovery_tax)
        if baseline_rise is None:
            baseline_rise = metrics["rise_time_90"]
        if baseline_rise and not math.isnan(metrics["rise_time_90"]):
            m_eff = metrics["rise_time_90"] / baseline_rise
        else:
            m_eff = float("nan")
        accel_steps = metrics.get("acceleration_steps", 0)
        total_steps = max(metrics.get("total_steps", 1), 1)
        blocked_steps = metrics.get("blocked_steps", 0)
        entry = {
            **metrics,
            "mu_c": mu,
            "m_eff": m_eff,
            "salience_floor": salience_floor,
            "acceleration_pct": accel_steps / total_steps,
            "blocked_pct": blocked_steps / total_steps,
        }
        results.append(entry)
    return results


def write_artifact(entries: List[Dict[str, float]]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())
    payload = []
    for entry in entries:
        payload.append(
            {
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "experiment_name": "experiment_g_continuity_subsidy",
                "run_id": run_id,
                **entry,
            }
        )
    path = ARTIFACT_DIR / f"continuity_subsidy_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def print_summary(entries: List[Dict[str, float]]) -> None:
    rows = [
        (
            entry["mu_c"],
            entry["salience_floor"],
            entry["rise_time_90"],
            entry["peak_overshoot"],
            entry["settling_time_2pct"],
            entry["rms_error"],
            entry["mean_salience"],
            entry["min_salience"],
            entry["control_energy"],
            entry["m_eff"],
            entry["acceleration_pct"],
            entry["blocked_pct"],
        )
        for entry in entries
    ]
    print(
        tabulate(
            rows,
            headers=[
                "mu_c",
                "S_floor",
                "rise_time_90",
                "peak_overshoot",
                "settling_time_2pct",
                "rms_error",
                "mean_salience",
                "min_salience",
                "control_energy",
                "m_eff",
                "accel_pct",
                "blocked_pct",
            ],
            tablefmt="github",
            floatfmt=".6f",
        )
    )


def main() -> None:
    coeffs = [0.0, 0.5, 1.0, 2.0]
    floors = [0.0, 0.6, 0.7, 0.8]
    all_entries: List[Dict[str, float]] = []
    for floor in floors:
        entries = run_subsidy_sweep(coeffs, salience_floor=floor)
        all_entries.extend(entries)
    artifact = write_artifact(all_entries)
    print("=== Experiment G: continuity subsidy with salience gating ===")
    print_summary(all_entries)
    print(f"Results written to {artifact}")


if __name__ == "__main__":
    main()
