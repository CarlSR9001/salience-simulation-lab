"""Experiment Q: Coupled domain energy leak probe."""

from __future__ import annotations

import argparse
import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from scripts.continuity_mass_sim import (
    CONTROL_AUTHORITY_THRESHOLD,
    ControllerConfig,
    ContinuityPID,
    PlantConfig,
    compute_rise_time,
    compute_settling_time,
)

ARTIFACT_DIR = Path("artifacts/coupled_domain")

ML_SALIENCE_FLOOR = 0.6
ML_OUTPUT_GAIN = 0.4
ML_OUTPUT_CLIP = 1.5
ML_MEASUREMENT_CLIP = 1.0


class Regime(str, Enum):
    CONTROL_ONLY = "control_only"
    FORWARD = "forward"
    REVERSE = "reverse"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class SimulationConfig:
    dt: float = 0.01
    horizon: float = 6.0
    plant_tau: float = 0.1
    noise_sigma: float = 0.0
    seed: int = 0


@dataclass
class RunConfig:
    lambda_c: float
    mu_shared: float
    regime: Regime


class SalienceAwareMLP:
    """Small MLP with salience-aware continuity mass."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rng: np.random.Generator,
        salience_scale: float = 0.1,
        mass_scale: float = 0.25,
        learning_rate: float = 0.05,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rng = rng
        bound = 1.0 / math.sqrt(input_dim)
        self.w1 = rng.uniform(-bound, bound, size=(hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = rng.uniform(-bound, bound, size=(hidden_dim,))
        self.b2 = 0.0
        self.lr = learning_rate
        self.mass_scale = mass_scale
        self.salience_scale = max(salience_scale, 1e-6)
        self.salience = np.ones(2, dtype=float)
        self.retention = np.full(2, 0.9, dtype=float)
        self.payoff = np.full(2, 0.9, dtype=float)
        self.fatigue = np.zeros(2, dtype=float)

    def _forward(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        h = np.tanh(self.w1 @ x + self.b1)
        y = float(self.w2 @ h + self.b2)
        return y, h

    def _update_salience(self, idx: int, delta: float, error: float) -> None:
        novelty = math.exp(-abs(delta) / self.salience_scale)
        self.retention[idx] = 0.92 * self.retention[idx] + 0.08 * novelty
        payoff_term = math.exp(-abs(error))
        self.payoff[idx] = 0.9 * self.payoff[idx] + 0.1 * payoff_term
        self.fatigue[idx] = 0.9 * self.fatigue[idx] + 0.1 * abs(delta) / self.salience_scale
        phi = min(self.fatigue[idx], 0.95)
        retention_term = float(np.clip(self.retention[idx], 0.2, 1.1))
        payoff_clamped = float(np.clip(self.payoff[idx], 0.2, 1.1))
        salience_val = novelty * retention_term * payoff_clamped * (1.0 - phi)
        self.salience[idx] = float(np.clip(salience_val, 1e-3, 1.5))

    def step(
        self,
        x: np.ndarray,
        error: float,
        lambda_c: float,
        *,
        train: bool = True,
    ) -> Dict[str, float]:
        y, h = self._forward(x)

        if not train:
            return {
                "output": y,
                "weight_delta_norm": 0.0,
                "salience_mean": float(np.mean(self.salience)),
                "salience_min": float(np.min(self.salience)),
            }

        grad_output = -error  # encourage reduction in error directly
        dW2 = grad_output * h
        db2 = grad_output
        dh = grad_output * self.w2
        dpre = dh * (1.0 - h ** 2)
        dW1 = np.outer(dpre, x)
        db1 = dpre

        mass_factor_w1 = 1.0 + lambda_c * self.mass_scale * self.salience[0]
        mass_factor_w2 = 1.0 + lambda_c * self.mass_scale * self.salience[1]

        delta_w1 = -(self.lr / mass_factor_w1) * dW1
        delta_b1 = -(self.lr / mass_factor_w1) * db1
        delta_w2 = -(self.lr / mass_factor_w2) * dW2
        delta_b2 = -(self.lr / mass_factor_w2) * db2

        delta_w1 = np.clip(delta_w1, -1.5, 1.5)
        delta_b1 = np.clip(delta_b1, -1.5, 1.5)
        delta_w2 = np.clip(delta_w2, -1.5, 1.5)
        delta_b2 = float(np.clip(delta_b2, -1.5, 1.5))

        self.w1 += delta_w1
        self.b1 += delta_b1
        self.w2 += delta_w2
        self.b2 += delta_b2

        delta_primary = float(np.linalg.norm(delta_w1, ord=1) + np.linalg.norm(delta_w2, ord=1))
        self._update_salience(0, float(np.linalg.norm(delta_w1)), error)
        self._update_salience(1, float(np.linalg.norm(delta_w2)), error)

        return {
            "output": y,
            "weight_delta_norm": delta_primary + abs(delta_b1).sum() + abs(delta_b2),
            "salience_mean": float(np.mean(self.salience)),
            "salience_min": float(np.min(self.salience)),
        }


def simulate_regime(
    run_cfg: RunConfig,
    sim_cfg: SimulationConfig,
    plant_cfg: PlantConfig,
    baseline_rise: float,
    baseline_energy: float,
    rng_seed: int,
    *,
    train: bool = True,
) -> Dict[str, float]:
    controller_cfg = ControllerConfig(lambda_c=run_cfg.lambda_c, salience_channels=5)
    controller = ContinuityPID(controller_cfg, plant_cfg.dt)
    controller.reset()

    rng = np.random.default_rng(rng_seed)
    ml_model = SalienceAwareMLP(input_dim=4, hidden_dim=8, rng=rng)

    steps = int(sim_cfg.horizon / plant_cfg.dt)
    target = 1.0
    state = 0.0
    last_control = 0.0
    last_output = 0.0

    outputs = np.zeros(steps, dtype=float)
    controller_salience_trace = np.zeros(steps, dtype=float)
    ml_delta_trace = np.zeros(steps, dtype=float)
    ml_salience_trace = np.zeros(steps, dtype=float)

    control_energy = 0.0
    ml_update_energy = 0.0
    controller_energy_trace = np.zeros(steps, dtype=float)
    ml_control_energy_trace = np.zeros(steps, dtype=float)
    noise_energy_trace = np.zeros(steps, dtype=float)
    measurement_inject_energy_trace = np.zeros(steps, dtype=float)

    for idx in range(steps):
        error = target - state
        features = np.array([state, error, last_control, last_output], dtype=float)

        ml_train = train and run_cfg.regime != Regime.CONTROL_ONLY
        if ml_train:
            current_salience = float(np.mean(ml_model.salience))
            ml_train = current_salience >= ML_SALIENCE_FLOOR
        ml_result = ml_model.step(features, error, run_cfg.lambda_c, train=ml_train)

        ml_salience_mean = ml_result["salience_mean"]
        gating = 0.0
        if run_cfg.mu_shared != 0.0 and ml_salience_mean >= ML_SALIENCE_FLOOR:
            activation = (ml_salience_mean - ML_SALIENCE_FLOOR) / (1.0 - ML_SALIENCE_FLOOR + 1e-6)
            gating = float(np.clip(activation, 0.0, 1.0))
        ml_raw_output = float(np.tanh(ml_result["output"] * ML_OUTPUT_GAIN))
        ml_contrib = gating * run_cfg.mu_shared * ml_raw_output
        ml_contrib = float(np.clip(ml_contrib, -ML_OUTPUT_CLIP, ML_OUTPUT_CLIP))

        measurement = state
        if run_cfg.regime in (Regime.REVERSE, Regime.BIDIRECTIONAL):
            measurement = state + float(np.clip(ml_contrib, -ML_MEASUREMENT_CLIP, ML_MEASUREMENT_CLIP))

        controller_result = controller.step(target, measurement)
        controller_control = controller_result["control"]
        control_signal = controller_control
        ml_control_energy_step = 0.0
        if run_cfg.regime in (Regime.FORWARD, Regime.BIDIRECTIONAL):
            control_signal += ml_contrib
            control_signal = float(np.clip(control_signal, -ML_OUTPUT_CLIP, ML_OUTPUT_CLIP))
            ml_control_energy_step = abs(ml_contrib) * plant_cfg.dt

        control_energy_step = abs(control_signal) * plant_cfg.dt
        control_energy += control_energy_step
        controller_energy_trace[idx] = abs(controller_control) * plant_cfg.dt
        ml_control_energy_trace[idx] = ml_control_energy_step
        ml_update_energy += gating * ml_result["weight_delta_norm"]

        state += plant_cfg.dt * ((-state + control_signal) / plant_cfg.tau)
        if sim_cfg.noise_sigma > 0.0:
            noise_term = rng.normal(0.0, sim_cfg.noise_sigma)
            state += noise_term
            noise_energy_trace[idx] = abs(noise_term)
        else:
            noise_energy_trace[idx] = 0.0

        outputs[idx] = state
        controller_salience_trace[idx] = controller_result["salience_mean"]
        ml_delta_trace[idx] = ml_result["weight_delta_norm"]
        ml_salience_trace[idx] = ml_result["salience_mean"]
        measurement_inject_energy_trace[idx] = abs(measurement - state)

        last_control = control_signal
        last_output = ml_contrib

    rise_time = compute_rise_time(outputs, target, plant_cfg.dt)
    settling_time = compute_settling_time(outputs, target, plant_cfg.dt)
    rms_error = float(np.sqrt(np.mean((target - outputs) ** 2)))

    authority_ratio = float("nan")
    controller_energy_rise = float("nan")
    external_energy_rise = float("nan")
    authority_ok = False
    if not math.isnan(rise_time):
        if baseline_rise:
            m_eff = rise_time / baseline_rise
        else:
            m_eff = float("nan")

        rise_idx = int(rise_time / plant_cfg.dt) if rise_time > 0 else 0
        rise_idx = min(max(rise_idx, 0), steps - 1)
        controller_energy_rise = float(np.sum(controller_energy_trace[: rise_idx + 1]))
        external_energy_rise = float(
            np.sum(ml_control_energy_trace[: rise_idx + 1])
            + np.sum(noise_energy_trace[: rise_idx + 1])
            + np.sum(measurement_inject_energy_trace[: rise_idx + 1])
        )
        if external_energy_rise > 1e-9:
            authority_ratio = controller_energy_rise / external_energy_rise
        else:
            authority_ratio = float("inf")
        authority_ok = (
            not math.isnan(authority_ratio)
            and (
                authority_ratio >= CONTROL_AUTHORITY_THRESHOLD
                or math.isinf(authority_ratio)
            )
        )
    else:
        m_eff = float("nan")

    baseline_energy_safe = max(baseline_energy, 1e-9)
    combined_energy = control_energy + ml_update_energy
    if authority_ok:
        energy_ratio = combined_energy / baseline_energy_safe
    else:
        energy_ratio = float("nan")
        m_eff = float("nan")

    if np.std(controller_salience_trace) > 1e-6 and np.std(ml_delta_trace) > 1e-6:
        salience_corr = float(np.corrcoef(controller_salience_trace, ml_delta_trace)[0, 1])
    else:
        salience_corr = float("nan")

    return {
        "lambda_c": run_cfg.lambda_c,
        "mu_shared": run_cfg.mu_shared,
        "regime": run_cfg.regime.value,
        "rise_time_90": float(rise_time),
        "settling_time_2pct": float(settling_time),
        "rms_error": rms_error,
        "control_energy": control_energy,
        "ml_update_energy": ml_update_energy,
        "combined_energy_ratio": energy_ratio,
        "m_eff_combined": m_eff,
        "controller_salience_mean": float(np.mean(controller_salience_trace)),
        "ml_salience_mean": float(np.mean(ml_salience_trace)),
        "ml_salience_min": float(np.min(ml_salience_trace)),
        "salience_corr": salience_corr,
        "controller_energy_rise": controller_energy_rise,
        "external_energy_rise": external_energy_rise,
        "authority_ratio": authority_ratio,
        "authority_ok": authority_ok,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment Q: coupled domain energy probe")
    parser.add_argument("--lambda-values", type=str, default="0,1,5", help="Comma-separated λ_c values")
    parser.add_argument("--mu-values", type=str, default="0,1.0", help="Comma-separated μ_shared values")
    parser.add_argument(
        "--regimes",
        type=str,
        default="control_only,forward,reverse,bidirectional",
        help="Comma-separated regimes to simulate",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation time step")
    parser.add_argument("--horizon", type=float, default=6.0, help="Simulation horizon (seconds)")
    parser.add_argument("--tau", type=float, default=0.1, help="Plant time constant")
    parser.add_argument("--noise-sigma", type=float, default=0.0, help="Gaussian noise sigma")
    parser.add_argument("--seed", type=int, default=2025, help="Base random seed")
    return parser.parse_args(argv)


def parse_float_list(payload: str) -> List[float]:
    values = []
    for part in payload.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def parse_regimes(payload: str) -> List[Regime]:
    selected: List[Regime] = []
    for part in payload.split(","):
        label = part.strip().lower()
        if not label:
            continue
        try:
            regime = Regime(label)
        except ValueError as exc:
            raise ValueError(f"Unknown regime: {label}") from exc
        selected.append(regime)
    if not selected:
        raise ValueError("No regimes selected")
    return selected


def write_artifact(entries: Iterable[Dict[str, float]]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())
    payload = []
    for entry in entries:
        payload.append(
            {
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "experiment_name": "experiment_q_coupled_domain",
                "run_id": run_id,
                **entry,
            }
        )
    out_path = ARTIFACT_DIR / f"coupled_domain_{timestamp}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def print_summary(entries: Iterable[Dict[str, float]]) -> None:
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    rows = [
        (
            e["regime"],
            e["lambda_c"],
            e["mu_shared"],
            e["rise_time_90"],
            e["m_eff_combined"],
            e["control_energy"],
            e["ml_update_energy"],
            e["combined_energy_ratio"],
            e["salience_corr"],
        )
        for e in entries
    ]

    headers = [
        "regime",
        "lambda_c",
        "mu_shared",
        "rise_time_90",
        "m_eff",
        "control_energy",
        "ml_update_energy",
        "energy_ratio",
        "salience_corr",
    ]

    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="github", floatfmt=".6f"))
    else:
        for row in rows:
            print(dict(zip(headers, row)))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    lambda_values = parse_float_list(args.lambda_values)
    mu_values = parse_float_list(args.mu_values)
    regimes = parse_regimes(args.regimes)

    sim_cfg = SimulationConfig(
        dt=args.dt,
        horizon=args.horizon,
        plant_tau=args.tau,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
    )
    plant_cfg = PlantConfig(dt=sim_cfg.dt, horizon=sim_cfg.horizon, tau=sim_cfg.plant_tau)

    entries: List[Dict[str, float]] = []

    seed_counter = args.seed
    for lambda_c in lambda_values:
        baseline_run = RunConfig(lambda_c=lambda_c, mu_shared=0.0, regime=Regime.CONTROL_ONLY)
        baseline_result = simulate_regime(
            baseline_run,
            sim_cfg,
            plant_cfg,
            baseline_rise=1.0,
            baseline_energy=1.0,
            rng_seed=seed_counter,
            train=False,
        )
        baseline_rise = baseline_result["rise_time_90"]
        baseline_energy = baseline_result["control_energy"] + baseline_result["ml_update_energy"]
        entries.append(baseline_result)

        for regime in regimes:
            for mu in mu_values:
                if regime is Regime.CONTROL_ONLY and mu != 0.0:
                    continue
                if regime is Regime.CONTROL_ONLY and lambda_c == baseline_run.lambda_c and mu == 0.0:
                    continue
                run_cfg = RunConfig(lambda_c=lambda_c, mu_shared=mu, regime=regime)
                seed_counter += 1
                result = simulate_regime(
                    run_cfg,
                    sim_cfg,
                    plant_cfg,
                    baseline_rise=baseline_rise,
                    baseline_energy=baseline_energy,
                    rng_seed=seed_counter,
                )
                entries.append(result)

    artifact = write_artifact(entries)
    print("=== Experiment Q: Coupled Domain Energy Probe ===")
    print_summary(entries)
    print(f"Results written to {artifact}")


if __name__ == "__main__":
    main()
