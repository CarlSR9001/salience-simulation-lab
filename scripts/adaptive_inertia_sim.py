"""Experiment B: Adaptive inertia subspace.

Simulates a continuity-taxed controller with core and edge latent components
using different continuity penalties. Logs metrics required by AGENTS.md.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

ARTIFACT_DIR = Path("artifacts/mass_sweep")


@dataclass
class ExperimentConfig:
    dt: float = 0.01
    horizon: float = 5.0
    tau: float = 0.12
    core_gain: float = 2.0
    edge_gain: float = 0.2
    kp: float = 0.0
    salience_scale_core: float = 0.06
    salience_scale_edge: float = 0.1
    fatigue_decay: float = 0.9
    fatigue_gain: float = 0.08


@dataclass
class ContinuitySettings:
    lambda_core: float
    lambda_edge: float


@dataclass
class RunResult:
    config: ContinuitySettings
    lag_core: float
    lag_edge: float
    overshoot_core: float
    overshoot_edge: float
    m_eff_core: float
    m_eff_edge: float
    mean_salience_core: float
    mean_salience_edge: float
    control_energy: float


class AdaptiveController:
    def __init__(self, cfg: ExperimentConfig, settings: ContinuitySettings) -> None:
        self.cfg = cfg
        self.settings = settings
        self.reset()

    def reset(self) -> None:
        self.core_state = 0.0
        self.edge_state = 0.0
        self.prev_error = 0.0
        self.salience_core = 1.0
        self.salience_edge = 1.0
        self.retention_core = 0.9
        self.retention_edge = 0.85
        self.payoff_core = 0.9
        self.payoff_edge = 0.9
        self.fatigue_core = 0.0
        self.fatigue_edge = 0.0

    def _update_salience(self, component: str, delta: float, error: float) -> None:
        if component == "core":
            scale = self.cfg.salience_scale_core
            novelty = math.exp(np.clip(-abs(delta) / scale, -10.0, 0.0))
            payoff = math.exp(np.clip(-abs(error), -10.0, 0.0))
            self.retention_core = 0.9 * self.retention_core + 0.1 * novelty
            self.payoff_core = 0.9 * self.payoff_core + 0.1 * payoff
            self.fatigue_core = (
                self.cfg.fatigue_decay * self.fatigue_core
                + self.cfg.fatigue_gain * abs(delta) / scale
            )
            phi = min(self.fatigue_core, 0.95)
            salience = novelty * np.clip(self.retention_core, 0.3, 1.1) * np.clip(self.payoff_core, 0.3, 1.1) * (1.0 - phi)
            self.salience_core = float(np.clip(0.6 + 0.4 * salience, 0.85, 1.5))
        else:
            scale = self.cfg.salience_scale_edge
            novelty = math.exp(np.clip(-abs(delta) / scale, -10.0, 0.0))
            payoff = math.exp(np.clip(-abs(error), -10.0, 0.0))
            self.retention_edge = 0.88 * self.retention_edge + 0.12 * novelty
            self.payoff_edge = 0.9 * self.payoff_edge + 0.1 * payoff
            self.fatigue_edge = (
                self.cfg.fatigue_decay * self.fatigue_edge
                + self.cfg.fatigue_gain * abs(delta) / scale
            )
            phi = min(self.fatigue_edge, 0.95)
            salience = novelty * np.clip(self.retention_edge, 0.2, 1.1) * np.clip(self.payoff_edge, 0.2, 1.1) * (1.0 - phi)
            self.salience_edge = float(np.clip(salience, 0.05, 1.2))

    def step(self, dt: float, error: float) -> Dict[str, float]:
        error_clipped = float(np.clip(error, -5.0, 5.0))

        raw_core_delta = error_clipped * dt
        mass_core = 1.0 + self.settings.lambda_core * self.salience_core
        self.core_state += raw_core_delta / mass_core
        self._update_salience("core", raw_core_delta, error)

        raw_edge_delta = (error_clipped - self.prev_error) / dt
        raw_edge_delta = float(np.clip(raw_edge_delta, -50.0, 50.0))
        mass_edge = 1.0 + self.settings.lambda_edge * self.salience_edge
        self.edge_state += raw_edge_delta / mass_edge
        self.edge_state *= 0.8
        self._update_salience("edge", raw_edge_delta, error)

        core_output = np.clip(self.cfg.core_gain * self.core_state, -10.0, 10.0)
        edge_output = np.clip(self.cfg.edge_gain * self.edge_state, -10.0, 10.0)
        proportional = self.cfg.kp * error_clipped
        total_control = core_output + edge_output + proportional
        self.prev_error = error

        return {
            "control": total_control,
            "core_output": core_output,
            "edge_output": edge_output,
            "salience_core": self.salience_core,
            "salience_edge": self.salience_edge,
        }


def simulate(settings: ContinuitySettings, cfg: ExperimentConfig) -> Dict[str, np.ndarray]:
    controller = AdaptiveController(cfg, settings)
    controller.reset()

    steps = int(cfg.horizon / cfg.dt)
    target = np.ones(steps)
    output = np.zeros(steps)
    core_trace = np.zeros(steps)
    edge_trace = np.zeros(steps)
    salience_core = np.zeros(steps)
    salience_edge = np.zeros(steps)
    control_energy = 0.0

    state = 0.0

    for i in range(steps):
        error = target[i] - state
        result = controller.step(cfg.dt, error)
        state += cfg.dt * ((-state + result["control"]) / cfg.tau)
        output[i] = state
        core_trace[i] = result["core_output"]
        edge_trace[i] = result["edge_output"]
        salience_core[i] = result["salience_core"]
        salience_edge[i] = result["salience_edge"]
        control_energy += abs(result["control"]) * cfg.dt

    return {
        "time": np.arange(0.0, cfg.horizon, cfg.dt),
        "output": output,
        "core": core_trace,
        "edge": edge_trace,
        "salience_core": salience_core,
        "salience_edge": salience_edge,
        "control_energy": control_energy,
    }


def compute_metrics(traces: Dict[str, np.ndarray], cfg: ExperimentConfig) -> Dict[str, float]:
    time = traces["time"]
    core = traces["core"]
    edge = traces["edge"]
    dt = cfg.dt

    final_core = float(np.mean(core[int(0.9 * len(core)) :]))
    core_abs = np.abs(core)
    core_area = float(np.sum(core_abs) * dt)
    lag_core = np.nan
    if core_area > 1e-6:
        cumulative = np.cumsum(core_abs) * dt
        target = 0.5 * core_area
        idx = int(np.searchsorted(cumulative, target))
        lag_core = float(time[min(idx, len(time) - 1)])

    peak_core = float(np.max(core))
    overshoot_core = peak_core - final_core

    edge_abs = np.abs(edge)
    edge_area = float(np.sum(edge_abs) * dt)
    lag_edge = np.nan
    if edge_area > 1e-6:
        cumulative_edge = np.cumsum(edge_abs) * dt
        target_edge = 0.5 * cumulative_edge[-1]
        idx = int(np.searchsorted(cumulative_edge, target_edge))
        lag_edge = float(time[min(idx, len(time) - 1)])
    peak_edge = float(np.max(edge_abs))
    overshoot_edge = peak_edge

    mean_salience_core = float(np.mean(traces["salience_core"]))
    mean_salience_edge = float(np.mean(traces["salience_edge"]))

    return {
        "lag_core": lag_core,
        "lag_edge": lag_edge,
        "overshoot_core": overshoot_core,
        "overshoot_edge": overshoot_edge,
        "mean_salience_core": mean_salience_core,
        "mean_salience_edge": mean_salience_edge,
        "control_energy": float(traces["control_energy"]),
    }


def run_experiment() -> List[RunResult]:
    cfg = ExperimentConfig()
    baseline_settings = ContinuitySettings(lambda_core=0.0, lambda_edge=0.0)
    taxed_settings = ContinuitySettings(lambda_core=40.0, lambda_edge=0.05)

    baseline_traces = simulate(baseline_settings, cfg)
    baseline_metrics = compute_metrics(baseline_traces, cfg)

    taxed_traces = simulate(taxed_settings, cfg)
    taxed_metrics = compute_metrics(taxed_traces, cfg)

    lag_core_baseline = baseline_metrics["lag_core"]
    lag_edge_baseline = baseline_metrics["lag_edge"]

    results = [
        RunResult(
            config=baseline_settings,
            lag_core=baseline_metrics["lag_core"],
            lag_edge=baseline_metrics["lag_edge"],
            overshoot_core=baseline_metrics["overshoot_core"],
            overshoot_edge=baseline_metrics["overshoot_edge"],
            m_eff_core=1.0,
            m_eff_edge=1.0,
            mean_salience_core=baseline_metrics["mean_salience_core"],
            mean_salience_edge=baseline_metrics["mean_salience_edge"],
            control_energy=baseline_metrics["control_energy"],
        ),
        RunResult(
            config=taxed_settings,
            lag_core=taxed_metrics["lag_core"],
            lag_edge=taxed_metrics["lag_edge"],
            overshoot_core=taxed_metrics["overshoot_core"],
            overshoot_edge=taxed_metrics["overshoot_edge"],
            m_eff_core=(taxed_metrics["lag_core"] / lag_core_baseline)
            if lag_core_baseline and not math.isnan(taxed_metrics["lag_core"])
            else float("nan"),
            m_eff_edge=(taxed_metrics["lag_edge"] / lag_edge_baseline)
            if lag_edge_baseline and not math.isnan(taxed_metrics["lag_edge"])
            else float("nan"),
            mean_salience_core=taxed_metrics["mean_salience_core"],
            mean_salience_edge=taxed_metrics["mean_salience_edge"],
            control_energy=taxed_metrics["control_energy"],
        ),
    ]

    return results


def write_artifact(results: List[RunResult]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())
    payload = []
    for res in results:
        payload.append(
            {
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "experiment_name": "experiment_b_adaptive_mass",
                "run_id": run_id,
                "lambda_core": res.config.lambda_core,
                "lambda_edge": res.config.lambda_edge,
                "lag_core": res.lag_core,
                "lag_edge": res.lag_edge,
                "overshoot_core": res.overshoot_core,
                "overshoot_edge": res.overshoot_edge,
                "m_eff_core": res.m_eff_core,
                "m_eff_edge": res.m_eff_edge,
                "mean_salience_core": res.mean_salience_core,
                "mean_salience_edge": res.mean_salience_edge,
                "control_energy": res.control_energy,
            }
        )
    path = ARTIFACT_DIR / f"adaptive_mass_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def main() -> None:
    results = run_experiment()
    artifact_path = write_artifact(results)
    print(f"Results written to {artifact_path}")
    for res in results:
        print(
            f"lambda_core={res.config.lambda_core:.2f}, lambda_edge={res.config.lambda_edge:.2f} -> "
            f"lag_core={res.lag_core:.4f}, lag_edge={res.lag_edge:.4f}, "
            f"m_eff_core={res.m_eff_core:.3f}, m_eff_edge={res.m_eff_edge:.3f}, "
            f"mean_salience_core={res.mean_salience_core:.3f}, mean_salience_edge={res.mean_salience_edge:.3f}"
        )


if __name__ == "__main__":
    main()
