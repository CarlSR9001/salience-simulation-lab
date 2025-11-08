"""Experiment F: Inertia–Time Coupling Probe.

Combines programmable inertia (continuity tax on core state) with continuity strain on
learning dynamics to test whether inertia and subjective time remain coupled.
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

ARTIFACT_DIR = Path("artifacts/unified_probe")

SALIENCE_THRESHOLD = 0.5
SALIENCE_VALID_MIN = 0.2
SALIENCE_ACTIVE_FRACTION = 0.2
EPS = 1e-9


def safe_exp(value: float, limit: float = 50.0) -> float:
    return float(math.exp(np.clip(value, -limit, limit)))


@dataclass
class SimulationConfig:
    dt: float = 0.01
    horizon: float = 6.0
    plant_tau: float = 0.12
    core_gain: float = 1.6
    edge_gain: float = 0.6
    lambda_edge: float = 0.05
    core_tau: float = 0.08
    edge_tau: float = 0.02
    fatigue_decay: float = 0.9
    fatigue_gain: float = 0.08
    salience_scale_core: float = 0.08
    salience_scale_learning: float = 0.12
    base_lr: float = 0.045
    grad_clip: float = 3.0
    dropout_base: float = 0.35
    gradient_noise_scale: float = 0.05
    replay_decay: float = 0.92
    integral_decay: float = 0.995
    derivative_filter: float = 0.6
    error_threshold: float = 0.05
    convergence_fraction: float = 0.9
    seed: int = 2025


@dataclass
class RunConfig:
    lambda_core: float
    strain_intensity: float


class SalienceTracker:
    def __init__(self, scale: float, fatigue_decay: float, fatigue_gain: float) -> None:
        self.scale = scale
        self.fatigue_decay = fatigue_decay
        self.fatigue_gain = fatigue_gain
        self.retention = 0.9
        self.payoff = 0.9
        self.fatigue = 0.0

    def update(self, delta: float, signal: float) -> float:
        novelty = math.exp(np.clip(-abs(delta) / self.scale, -10.0, 0.0))
        payoff_term = math.exp(np.clip(-abs(signal), -10.0, 0.0))
        self.retention = 0.9 * self.retention + 0.1 * novelty
        self.payoff = 0.9 * self.payoff + 0.1 * payoff_term
        self.fatigue = self.fatigue_decay * self.fatigue + self.fatigue_gain * abs(delta) / self.scale
        phi = min(self.fatigue, 0.95)
        salience = novelty * np.clip(self.retention, 0.3, 1.2) * np.clip(self.payoff, 0.3, 1.2) * (1.0 - phi)
        return float(np.clip(salience, 0.2, 1.5))


class UnifiedController:
    def __init__(self, sim_cfg: SimulationConfig, run_cfg: RunConfig, rng: np.random.Generator) -> None:
        self.cfg = sim_cfg
        self.run_cfg = run_cfg
        self.rng = rng
        self.core_tracker = SalienceTracker(sim_cfg.salience_scale_core, sim_cfg.fatigue_decay, sim_cfg.fatigue_gain)
        self.learning_tracker = SalienceTracker(sim_cfg.salience_scale_learning, 0.92, 0.05)
        self.reset()

    def reset(self) -> None:
        self.core_state = 0.0
        self.edge_state = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        self.weights = np.array([1.8, 0.6, 0.2], dtype=float)
        self.core_salience = 1.0
        self.learning_salience = 1.0
        self.replay_error = 0.0

    def step(self, error: float, dt: float, step_index: int) -> Dict[str, float]:
        self.integral = self.cfg.integral_decay * self.integral + error * dt
        derivative = (error - self.prev_error) / dt
        derivative = self.cfg.derivative_filter * derivative + (1 - self.cfg.derivative_filter) * (self.prev_error / dt)
        self.prev_error = error

        features = np.array([error, self.integral, derivative], dtype=float)
        features = np.clip(features, -3.0, 3.0)

        if self.run_cfg.strain_intensity > 0:
            noise = self.rng.normal(0.0, 0.05 * self.run_cfg.strain_intensity, size=features.shape)
            features = features * (1.0 + noise)

        command = float(np.dot(self.weights, features))

        # Core path with programmable inertia
        core_delta = command - self.core_state
        prev_core_state = self.core_state
        self.core_salience = self.core_tracker.update(core_delta, error)
        eff_tau_core = self.cfg.core_tau * (1.0 + self.run_cfg.lambda_core * self.core_salience)
        eff_tau_core = np.clip(eff_tau_core, self.cfg.core_tau, self.cfg.core_tau * 50.0)
        self.core_state += dt * core_delta / eff_tau_core
        core_update = self.core_state - prev_core_state

        # Edge path remains agile
        edge_delta = command - self.edge_state
        eff_tau_edge = self.cfg.edge_tau * (1.0 + self.cfg.lambda_edge * self.core_salience)
        self.edge_state += dt * edge_delta / eff_tau_edge

        control = (
            self.cfg.core_gain * self.core_state + self.cfg.edge_gain * self.edge_state
        )

        # Learning update
        grad = error * features + 0.1 * self.replay_error * features
        grad = np.clip(grad, -self.cfg.grad_clip, self.cfg.grad_clip)

        if self.run_cfg.strain_intensity > 0:
            drop_prob = np.clip(self.cfg.dropout_base * self.run_cfg.strain_intensity, 0.0, 0.95)
            mask = self.rng.random(len(grad)) > drop_prob
            grad *= mask
            grad += self.rng.normal(0.0, self.cfg.gradient_noise_scale * self.run_cfg.strain_intensity, size=grad.shape)

        effective_lr = self.cfg.base_lr / (1.0 + self.run_cfg.strain_intensity * self.learning_salience)
        weight_delta = -effective_lr * grad
        self.weights += weight_delta

        delta_norm = float(np.linalg.norm(weight_delta))
        self.learning_salience = self.learning_tracker.update(delta_norm, error)
        self.replay_error = self.cfg.replay_decay * self.replay_error + (1 - self.cfg.replay_decay) * error

        return {
            "control": control,
            "core_state": self.core_state,
            "edge_state": self.edge_state,
            "core_delta": core_delta,
            "core_update": core_update,
            "weight_delta_norm": delta_norm,
            "core_salience": self.core_salience,
            "learning_salience": self.learning_salience,
        }


def simulate(sim_cfg: SimulationConfig, run_cfg: RunConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    controller = UnifiedController(sim_cfg, run_cfg, rng)
    controller.reset()

    steps = int(sim_cfg.horizon / sim_cfg.dt)
    target = 1.0
    state = 0.0
    outputs = np.zeros(steps)
    errors = np.zeros(steps)
    controls = np.zeros(steps)
    core_states = np.zeros(steps)
    core_deltas = np.zeros(steps)
    core_updates = np.zeros(steps)
    learning_deltas = np.zeros(steps)
    core_saliences = np.zeros(steps)
    learning_saliences = np.zeros(steps)

    control_energy = 0.0

    for i in range(steps):
        error = target - state
        data = controller.step(error, sim_cfg.dt, i + 1)
        state += sim_cfg.dt * ((-state + data["control"]) / sim_cfg.plant_tau)

        outputs[i] = state
        errors[i] = error
        controls[i] = data["control"]
        core_states[i] = data["core_state"]
        core_deltas[i] = abs(data["core_delta"])
        core_updates[i] = data["core_update"]
        learning_deltas[i] = data["weight_delta_norm"]
        core_saliences[i] = data["core_salience"]
        learning_saliences[i] = data["learning_salience"]
        control_energy += abs(data["control"]) * sim_cfg.dt

    time = np.linspace(0.0, sim_cfg.horizon, steps, endpoint=False)

    return {
        "time": time,
        "output": outputs,
        "error": errors,
        "control": controls,
        "core_state": core_states,
        "core_delta_trace": core_deltas,
        "core_update_trace": core_updates,
        "learning_delta_trace": learning_deltas,
        "core_salience": core_saliences,
        "learning_salience": learning_saliences,
        "control_energy": control_energy,
    }


def compute_rise_time(time: np.ndarray, output: np.ndarray, target: float = 1.0, fraction: float = 0.9) -> float:
    threshold = target * fraction
    for t, y in zip(time, output):
        if y >= threshold:
            return float(t)
    return float(time[-1])


def compute_convergence_steps(errors: np.ndarray, threshold: float, fraction: float) -> int:
    total = len(errors)
    target_count = int(math.ceil(total * fraction))
    for idx in range(total):
        window = errors[idx:]
        if len(window) < target_count:
            break
        mask = np.sum(np.abs(window) < threshold)
        if mask >= target_count:
            return idx + 1
    return total


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize_run(sim_cfg: SimulationConfig, traces: Dict[str, np.ndarray]) -> Dict[str, float | int]:
    rise_time = compute_rise_time(traces["time"], traces["output"])
    convergence_steps = compute_convergence_steps(
        traces["error"], sim_cfg.error_threshold, sim_cfg.convergence_fraction
    )
    internal_latency = float(np.mean(np.abs(traces["core_update_trace"])) + EPS)
    learning_rate_effective = float(np.mean(traces["learning_delta_trace"]))
    mean_salience_core = float(np.mean(traces["core_salience"]))
    var_salience_core = float(np.var(traces["core_salience"]))
    active_fraction = float(np.mean(traces["core_salience"] > SALIENCE_THRESHOLD))
    min_salience_core = float(np.min(traces["core_salience"]))
    mean_salience_learning = float(np.mean(traces["learning_salience"]))
    var_salience_learning = float(np.var(traces["learning_salience"]))
    control_energy = float(traces["control_energy"])

    return {
        "rise_time": float(rise_time),
        "convergence_steps": int(convergence_steps),
        "internal_update_latency": internal_latency,
        "learning_rate_effective": learning_rate_effective,
        "mean_salience_core": mean_salience_core,
        "var_salience_core": var_salience_core,
        "salience_active_fraction": active_fraction,
        "min_salience_core": min_salience_core,
        "mean_salience_learning": mean_salience_learning,
        "var_salience_learning": var_salience_learning,
        "control_energy": control_energy,
    }


def determine_regime(summary: Dict[str, float | int], coupling_ratio: float, pearson: float) -> Tuple[str, bool, str]:
    mean_sal = float(summary["mean_salience_core"])
    active_fraction = float(summary["salience_active_fraction"])

    salience_valid = mean_sal >= SALIENCE_VALID_MIN and active_fraction >= SALIENCE_ACTIVE_FRACTION

    if not salience_valid:
        if active_fraction < SALIENCE_ACTIVE_FRACTION and mean_sal < SALIENCE_VALID_MIN:
            status = "low_mean_and_fraction"
        elif active_fraction < SALIENCE_ACTIVE_FRACTION:
            status = "inactive_fraction"
        else:
            status = "low_mean"
        return "invalid", False, status

    if math.isnan(coupling_ratio):
        return "anomalous", True, "valid_salience"

    deviation = abs(coupling_ratio - 1.0)
    if deviation <= 0.25 and (not math.isnan(pearson) and pearson >= 0.5):
        return "locked", True, "valid_salience"
    if deviation > 0.25 and mean_sal >= 0.6:
        return "decoupled", True, "valid_salience"
    return "anomalous", True, "valid_salience"


def run_grid(sim_cfg: SimulationConfig, grid: Iterable[RunConfig]) -> List[Dict[str, float | int | bool | str]]:
    results: List[Dict[str, float | int | bool | str]] = []
    baseline_cfg = RunConfig(lambda_core=0.0, strain_intensity=0.0)
    base_seq = np.random.SeedSequence(sim_cfg.seed)

    for run_cfg in grid:
        baseline_seq, run_seq = base_seq.spawn(2)
        baseline_rng = np.random.default_rng(baseline_seq)
        run_rng = np.random.default_rng(run_seq)

        baseline_traces = simulate(sim_cfg, baseline_cfg, baseline_rng)
        baseline_summary = summarize_run(sim_cfg, baseline_traces)

        run_traces = simulate(sim_cfg, run_cfg, run_rng)
        run_summary = summarize_run(sim_cfg, run_traces)

        pearson = pearson_corr(run_traces["core_delta_trace"], run_traces["learning_delta_trace"])

        if run_cfg.lambda_core == 0.0 and run_cfg.strain_intensity == 0.0:
            m_eff_core = 1.0
            internal_latency_ratio = 1.0
            convergence_ratio = 1.0
            coupling_ratio = 1.0
            regime_label = "baseline"
            salience_valid = True
            salience_status = "baseline"
        else:
            m_eff_core = run_summary["rise_time"] / (baseline_summary["rise_time"] + EPS)
            internal_latency_ratio = run_summary["internal_update_latency"] / (
                baseline_summary["internal_update_latency"] + EPS
            )
            convergence_ratio = run_summary["convergence_steps"] / (
                baseline_summary["convergence_steps"] + EPS
            )
            coupling_ratio = internal_latency_ratio / (m_eff_core + EPS)
            regime_label, salience_valid, salience_status = determine_regime(run_summary, coupling_ratio, pearson)

        record: Dict[str, float | int | bool | str] = {
            "lambda_core": float(run_cfg.lambda_core),
            "strain_intensity": float(run_cfg.strain_intensity),
            "rise_time": float(run_summary["rise_time"]),
            "m_eff_core": float(m_eff_core),
            "convergence_steps": int(run_summary["convergence_steps"]),
            "convergence_ratio": float(convergence_ratio),
            "internal_update_latency": float(run_summary["internal_update_latency"]),
            "internal_latency_ratio": float(internal_latency_ratio),
            "learning_rate_effective": float(run_summary["learning_rate_effective"]),
            "mean_salience_core": float(run_summary["mean_salience_core"]),
            "var_salience_core": float(run_summary["var_salience_core"]),
            "salience_active_fraction": float(run_summary["salience_active_fraction"]),
            "min_salience_core": float(run_summary["min_salience_core"]),
            "mean_salience_learning": float(run_summary["mean_salience_learning"]),
            "var_salience_learning": float(run_summary["var_salience_learning"]),
            "salience_valid": bool(salience_valid),
            "salience_status": salience_status,
            "pearson_correlation": float(pearson),
            "coupling_ratio": float(coupling_ratio),
            "control_energy": float(run_summary["control_energy"]),
            "regime_label": regime_label,
            "baseline_rise_time": float(baseline_summary["rise_time"]),
            "baseline_convergence_steps": int(baseline_summary["convergence_steps"]),
            "baseline_internal_update_latency": float(baseline_summary["internal_update_latency"]),
            "baseline_mean_salience_core": float(baseline_summary["mean_salience_core"]),
        }

        results.append(record)

    return results


def write_artifact(records: List[Dict[str, float | int | bool | str]], metadata: Dict[str, object]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())
    payload = []
    for record in records:
        payload.append(
            {
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "experiment_name": "experiment_f_inertia_time",
                "run_id": run_id,
                **metadata,
                **record,
            }
        )
    path = ARTIFACT_DIR / f"unified_probe_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def main() -> None:
    sim_cfg = SimulationConfig()
    lambda_values = [0.0, 5.0, 10.0, 20.0]
    strain_values = [0.0, 0.5, 1.0]
    grid = [RunConfig(lambda_core=lam, strain_intensity=strain) for lam in lambda_values for strain in strain_values]

    results = run_grid(sim_cfg, grid)
    metadata = {
        "lambda_values": lambda_values,
        "strain_values": strain_values,
        "seed": sim_cfg.seed,
    }
    artifact = write_artifact(results, metadata)

    print(f"Unified probe results saved to {artifact}")
    for record in results:
        print(
            f"λ_core={record['lambda_core']:>5.1f}, strain={record['strain_intensity']:>4.1f}, "
            f"m_eff={record['m_eff_core']:.3f}, internal_ratio={record['internal_latency_ratio']:.3f}, "
            f"coupling={record['coupling_ratio']:.3f}, regime={record['regime_label']}"
        )


if __name__ == "__main__":
    main()
