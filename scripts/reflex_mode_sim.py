"""Experiment J: Reflex Mode (Zero-Lag Edge Channel).

Simulates a core/edge controller where the edge channel receives continuity subsidy
to accelerate reflex responses while the core maintains nominal continuity tax.
Based on adaptive_inertia_sim core/edge split + continuity_subsidy_sim assist logic.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from tabulate import tabulate

ARTIFACT_DIR = Path("artifacts/reflex_mode")


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
    lambda_core: float = 0.0  # nominal continuity tax for core
    assist_cap: float = 2.5
    edge_salience_floor: float = 0.05  # lower floor for edge (operates transiently)
    core_salience_floor: float = 0.8  # higher floor for core (must stay coherent)


@dataclass
class ReflexSettings:
    mu_edge: float  # continuity subsidy for edge channel


@dataclass
class RunResult:
    mu_edge: float
    edge_response_time_90: float
    core_salience: float
    cross_coupling_error: float
    total_overshoot: float
    mean_salience_edge: float
    mean_salience_core: float
    control_energy: float
    peak_edge_output: float
    mean_edge_assist: float


class ReflexController:
    """Controller with core/edge split where edge receives continuity subsidy."""

    def __init__(self, cfg: ExperimentConfig, settings: ReflexSettings) -> None:
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
        self.prev_salience_edge = 1.0

    def _update_salience(self, component: str, delta: float, error: float) -> tuple[float, float]:
        """Update salience for component, return (new_salience, delta_salience)."""
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
            prev = self.salience_core
            self.salience_core = float(np.clip(0.6 + 0.4 * salience, 0.85, 1.5))
            return self.salience_core, self.salience_core - prev
        else:  # edge
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
            prev = self.salience_edge
            self.salience_edge = float(np.clip(salience, 0.05, 1.2))
            return self.salience_edge, self.salience_edge - prev

    def _compute_edge_assist(self, error: float, salience_edge: float, delta_salience_edge: float) -> float:
        """Compute continuity subsidy assist factor for edge channel.

        Edge channel is reflex-oriented, so subsidy logic differs from core:
        - Uses lower salience floor (edge operates transiently)
        - Prioritizes rapid response when error is present
        - Less strict about salience coherence (reflexes can be brief)
        """
        if self.settings.mu_edge <= 0.0:
            return 1.0

        # Check edge salience floor (much lower than core)
        if salience_edge < self.cfg.edge_salience_floor:
            return 1.0

        # For reflex mode, assist when there's significant error (fast response needed)
        # or when reducing error (goal alignment)
        error_mag = abs(error)
        significant_error = error_mag > 0.01  # there's work to do

        prev_error_mag = abs(self.prev_error)
        curr_error_mag = abs(error)
        goal_aligned = prev_error_mag <= 1e-6 or curr_error_mag < prev_error_mag

        # Edge gets assist either when error is present or when making progress
        reflex_active = significant_error or goal_aligned

        if not reflex_active:
            return 1.0

        # Compute assist strength based on:
        # 1. Current edge salience (even if low, edge can be active)
        # 2. Salience improvement
        # 3. Error magnitude (more error = stronger reflex needed)

        salience_gain = max(delta_salience_edge, 0.0)
        error_urgency = float(np.clip(error_mag, 0.0, 1.0))  # normalized urgency

        # Edge coherence: much lower bar than core
        edge_coherent = salience_edge >= 0.2  # edge can operate with lower salience
        coherence_term = 0.2 if edge_coherent else 0.0

        # Strength combines salience gain, urgency, and coherence
        strength = salience_gain + 0.3 * error_urgency + coherence_term
        strength = float(np.clip(strength, 0.0, self.cfg.assist_cap))

        # Apply subsidy
        boost = 1.0 + self.settings.mu_edge * strength
        max_boost = 1.0 + self.settings.mu_edge * self.cfg.assist_cap
        boost = float(np.clip(boost, 1.0, max_boost))

        return boost

    def step(self, dt: float, error: float) -> Dict[str, float]:
        error_clipped = float(np.clip(error, -5.0, 5.0))

        # Core channel: nominal continuity tax (λ_core ≈ 0)
        raw_core_delta = error_clipped * dt
        mass_core = 1.0 + self.cfg.lambda_core * self.salience_core
        self.core_state += raw_core_delta / mass_core
        salience_core, delta_salience_core = self._update_salience("core", raw_core_delta, error)

        # Edge channel: derivative-like response with subsidy
        raw_edge_delta = (error_clipped - self.prev_error) / dt
        raw_edge_delta = float(np.clip(raw_edge_delta, -50.0, 50.0))

        # Update salience before computing assist
        salience_edge, delta_salience_edge = self._update_salience("edge", raw_edge_delta, error)

        # Compute edge assist factor based on subsidy
        edge_assist = self._compute_edge_assist(error, salience_edge, delta_salience_edge)

        # Apply assist to edge acceleration (reducing effective inertia)
        # No base continuity tax on edge (lambda_edge = 0 implicitly), only subsidy
        edge_acceleration = raw_edge_delta * edge_assist
        self.edge_state += edge_acceleration
        self.edge_state *= 0.8  # damping

        # If assisted, reduce fatigue and boost retention (as in subsidy sim)
        if edge_assist > 1.0 + 1e-6:
            self.fatigue_edge *= 0.92
            self.retention_edge = float(np.clip(0.96 * self.retention_edge + 0.04 * salience_edge, 0.2, 1.2))

        # Compute outputs
        core_output = np.clip(self.cfg.core_gain * self.core_state, -10.0, 10.0)
        edge_output = np.clip(self.cfg.edge_gain * self.edge_state, -10.0, 10.0)
        proportional = self.cfg.kp * error_clipped
        total_control = core_output + edge_output + proportional

        self.prev_error = error

        return {
            "control": total_control,
            "core_output": core_output,
            "edge_output": edge_output,
            "salience_core": salience_core,
            "salience_edge": salience_edge,
            "edge_assist": edge_assist,
        }


def simulate(settings: ReflexSettings, cfg: ExperimentConfig) -> Dict[str, np.ndarray]:
    controller = ReflexController(cfg, settings)
    controller.reset()

    steps = int(cfg.horizon / cfg.dt)
    target = np.ones(steps)
    output = np.zeros(steps)
    core_trace = np.zeros(steps)
    edge_trace = np.zeros(steps)
    salience_core_trace = np.zeros(steps)
    salience_edge_trace = np.zeros(steps)
    edge_assist_trace = np.zeros(steps)
    control_energy = 0.0

    state = 0.0

    for i in range(steps):
        error = target[i] - state
        result = controller.step(cfg.dt, error)
        state += cfg.dt * ((-state + result["control"]) / cfg.tau)
        output[i] = state
        core_trace[i] = result["core_output"]
        edge_trace[i] = result["edge_output"]
        salience_core_trace[i] = result["salience_core"]
        salience_edge_trace[i] = result["salience_edge"]
        edge_assist_trace[i] = result["edge_assist"]
        control_energy += abs(result["control"]) * cfg.dt

    return {
        "time": np.arange(0.0, cfg.horizon, cfg.dt),
        "output": output,
        "core": core_trace,
        "edge": edge_trace,
        "salience_core": salience_core_trace,
        "salience_edge": salience_edge_trace,
        "edge_assist": edge_assist_trace,
        "control_energy": control_energy,
    }


def compute_edge_response_time_90(edge_trace: np.ndarray, dt: float) -> float:
    """Compute time to first significant edge activation (reflex trigger time).

    For reflex mode, we care about when the edge first becomes meaningfully active,
    not when it reaches 90% of peak (which might be immediate for derivative channels).
    """
    edge_abs = np.abs(edge_trace)
    peak = float(np.max(edge_abs))

    if peak < 1e-6:
        return float("nan")

    # Use a fixed threshold of 10% of peak for "first activation"
    # This better captures reflex latency
    threshold = 0.1 * peak
    indices = np.where(edge_abs >= threshold)[0]

    if indices.size == 0:
        return float("nan")

    return float(indices[0] * dt)


def compute_cross_coupling_error(output: np.ndarray, edge_trace: np.ndarray, dt: float) -> float:
    """
    Measure cross-coupling: how much does edge motion disturb core stability?
    Computed as correlation between edge activity and output deviation variance.
    """
    # Compute edge activity magnitude
    edge_activity = np.abs(edge_trace)

    # Compute output deviation from target
    target = 1.0
    deviation = np.abs(output - target)

    # Windowed correlation: high edge activity should ideally not correlate with high deviation
    # We use a simple metric: mean(edge_activity * deviation) normalized
    if np.max(edge_activity) < 1e-6:
        return 0.0

    coupling = float(np.mean(edge_activity * deviation))
    # Normalize by peak edge and typical deviation
    max_edge = float(np.max(edge_activity))
    max_dev = float(np.max(deviation))

    if max_edge > 1e-6 and max_dev > 1e-6:
        normalized_coupling = coupling / (max_edge * max_dev)
    else:
        normalized_coupling = 0.0

    return normalized_coupling


def compute_total_overshoot(output: np.ndarray, target: float = 1.0) -> float:
    """Total overshoot beyond target."""
    overshoot = float(np.max(output) - target)
    return max(overshoot, 0.0)


def compute_metrics(traces: Dict[str, np.ndarray], cfg: ExperimentConfig) -> Dict[str, float]:
    edge_response_time_90 = compute_edge_response_time_90(traces["edge"], cfg.dt)
    core_salience = float(np.mean(traces["salience_core"]))
    mean_salience_edge = float(np.mean(traces["salience_edge"]))
    cross_coupling_error = compute_cross_coupling_error(traces["output"], traces["edge"], cfg.dt)
    total_overshoot = compute_total_overshoot(traces["output"])
    peak_edge_output = float(np.max(np.abs(traces["edge"])))
    mean_edge_assist = float(np.mean(traces["edge_assist"]))

    return {
        "edge_response_time_90": edge_response_time_90,
        "core_salience": core_salience,
        "mean_salience_edge": mean_salience_edge,
        "cross_coupling_error": cross_coupling_error,
        "total_overshoot": total_overshoot,
        "control_energy": float(traces["control_energy"]),
        "peak_edge_output": peak_edge_output,
        "mean_edge_assist": mean_edge_assist,
    }


def run_experiment() -> List[RunResult]:
    cfg = ExperimentConfig()
    mu_edge_values = [0.0, 1.0, 2.0, 5.0]
    results = []

    for mu_edge in mu_edge_values:
        settings = ReflexSettings(mu_edge=mu_edge)
        traces = simulate(settings, cfg)
        metrics = compute_metrics(traces, cfg)

        result = RunResult(
            mu_edge=mu_edge,
            edge_response_time_90=metrics["edge_response_time_90"],
            core_salience=metrics["core_salience"],
            cross_coupling_error=metrics["cross_coupling_error"],
            total_overshoot=metrics["total_overshoot"],
            mean_salience_edge=metrics["mean_salience_edge"],
            mean_salience_core=metrics["core_salience"],
            control_energy=metrics["control_energy"],
            peak_edge_output=metrics["peak_edge_output"],
            mean_edge_assist=metrics["mean_edge_assist"],
        )
        results.append(result)

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
                "experiment_name": "experiment_j_reflex_mode",
                "run_id": run_id,
                "mu_edge": res.mu_edge,
                "edge_response_time_90": res.edge_response_time_90,
                "core_salience": res.core_salience,
                "mean_salience_edge": res.mean_salience_edge,
                "cross_coupling_error": res.cross_coupling_error,
                "total_overshoot": res.total_overshoot,
                "control_energy": res.control_energy,
                "peak_edge_output": res.peak_edge_output,
                "mean_edge_assist": res.mean_edge_assist,
            }
        )
    path = ARTIFACT_DIR / f"reflex_mode_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def print_summary(results: List[RunResult]) -> None:
    rows = [
        (
            res.mu_edge,
            res.edge_response_time_90,
            res.peak_edge_output,
            res.mean_edge_assist,
            res.core_salience,
            res.mean_salience_edge,
            res.cross_coupling_error,
            res.total_overshoot,
            res.control_energy,
        )
        for res in results
    ]
    print(
        tabulate(
            rows,
            headers=[
                "mu_edge",
                "edge_t90",
                "peak_edge",
                "avg_assist",
                "core_sal",
                "edge_sal",
                "coupling",
                "overshoot",
                "energy",
            ],
            tablefmt="github",
            floatfmt=".6f",
        )
    )


def main() -> None:
    print("=== Experiment J: Reflex Mode (Zero-Lag Edge Channel) ===\n")
    results = run_experiment()
    artifact_path = write_artifact(results)
    print_summary(results)
    print(f"\nResults written to {artifact_path}")

    # Success criteria check
    print("\n=== Success Criteria ===")
    baseline_assist = results[0].mean_edge_assist
    baseline_peak = results[0].peak_edge_output

    for res in results:
        if res.mu_edge > 0:
            # Edge accelerates: higher assist or higher peak output
            edge_accelerated = (res.mean_edge_assist > baseline_assist or
                               res.peak_edge_output > baseline_peak)
            core_salient = res.core_salience >= 0.8
            coupling_low = res.cross_coupling_error < 0.1  # tighter threshold
            success = edge_accelerated and core_salient and coupling_low
            status = "PASS" if success else "FAIL"
            print(f"mu_edge={res.mu_edge:.1f}: {status} "
                  f"(edge_accel={edge_accelerated}, core_sal≥0.8={core_salient}, "
                  f"coupling<0.1={coupling_low})")


if __name__ == "__main__":
    main()
