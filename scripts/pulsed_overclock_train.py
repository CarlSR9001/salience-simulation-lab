"""Experiment L: Pulsed overclock training with salience floor gating."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from salience_floor_gate import GateTelemetry, gate_gradient_scale

ARTIFACT_DIR = Path("artifacts/pulsed_overclock")


@dataclass
class TrainingConfig:
    input_dim: int = 4
    hidden_dim: int = 64
    output_dim: int = 2
    dataset_size: int = 2000
    val_size: int = 400
    batch_size: int = 64
    steps: int = 4000
    eval_interval: int = 40
    lr: float = 0.01
    pulse_on_steps: int = 20
    pulse_off_steps: int = 10
    alpha_boost: float = 1.0
    salience_floor: float = 0.7
    salience_threshold: float = 0.7
    recovery_tax: float = 0.3
    seed: int = 4242
    salience_lambda: float = 1e-4
    phi_coeff: float = 0.55
    continuity_beta: float = 0.35
    w1: float = 0.6
    w2: float = 0.25
    w3: float = 0.15


@dataclass
class ChannelSalienceState:
    template: np.ndarray
    fatigue: np.ndarray
    retention: np.ndarray


def generate_dataset(cfg: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    X = rng.normal(0.0, 1.0, size=(cfg.dataset_size + cfg.val_size, cfg.input_dim))
    weights_true = rng.normal(0.0, 1.0, size=(cfg.input_dim, cfg.output_dim))
    logits = X @ weights_true
    probs = 1 / (1 + np.exp(-logits))
    y = (probs[:, 0] > probs[:, 1]).astype(int)
    X_train = X[: cfg.dataset_size]
    y_train = y[: cfg.dataset_size]
    X_val = X[cfg.dataset_size :]
    y_val = y[cfg.dataset_size :]
    return X_train, y_train, X_val, y_val


def init_weights(cfg: TrainingConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    scale = math.sqrt(2.0 / cfg.input_dim)
    return {
        "W1": rng.normal(0.0, scale, size=(cfg.input_dim, cfg.hidden_dim)),
        "b1": np.zeros(cfg.hidden_dim),
        "W2": rng.normal(0.0, scale, size=(cfg.hidden_dim, cfg.output_dim)),
        "b2": np.zeros(cfg.output_dim),
    }


def forward(weights: Dict[str, np.ndarray], x: np.ndarray) -> Dict[str, np.ndarray]:
    z1 = x @ weights["W1"] + weights["b1"]
    h = np.tanh(z1)
    logits = h @ weights["W2"] + weights["b2"]
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return {"z1": z1, "h": h, "probs": probs}


def cross_entropy(probs: np.ndarray, targets: np.ndarray) -> float:
    eps = 1e-9
    targets_one_hot = np.eye(probs.shape[1])[targets]
    return float(-np.mean(np.sum(targets_one_hot * np.log(probs + eps), axis=1)))


def accuracy(probs: np.ndarray, targets: np.ndarray) -> float:
    preds = np.argmax(probs, axis=1)
    return float(np.mean(preds == targets))


def backward(weights: Dict[str, np.ndarray], cache: Dict[str, np.ndarray], x: np.ndarray, targets: np.ndarray) -> Dict[str, np.ndarray]:
    batch_size = len(x)
    probs = cache["probs"].copy()
    targets_one_hot = np.eye(probs.shape[1])[targets]
    probs -= targets_one_hot
    probs /= batch_size

    grads = {}
    grads["W2"] = cache["h"].T @ probs
    grads["b2"] = probs.sum(axis=0)

    dh = probs @ weights["W2"].T
    dz1 = dh * (1 - cache["h"] ** 2)
    grads["W1"] = x.T @ dz1
    grads["b1"] = dz1.sum(axis=0)
    grads["latent"] = dz1
    return grads


def apply_gradients(weights: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], lr: float) -> None:
    weights["W1"] -= lr * grads["W1"]
    weights["b1"] -= lr * grads["b1"]
    weights["W2"] -= lr * grads["W2"]
    weights["b2"] -= lr * grads["b2"]


def init_salience_state(cfg: TrainingConfig) -> ChannelSalienceState:
    return ChannelSalienceState(
        template=np.zeros(cfg.hidden_dim), fatigue=np.zeros(cfg.hidden_dim), retention=np.full(cfg.hidden_dim, 0.9)
    )


def compute_channel_salience(
    hidden_mean: np.ndarray,
    loss: float,
    state: ChannelSalienceState,
    cfg: TrainingConfig,
    step: int,
) -> Tuple[np.ndarray, ChannelSalienceState]:
    delta = hidden_mean - state.template
    scale = math.sqrt(cfg.hidden_dim)
    delta_norm = np.abs(delta) / (scale + 1e-6)
    novelty = 1.0 - np.exp(-delta_norm)

    state.retention = 0.9 * state.retention + 0.1 * np.exp(-delta_norm)
    payoff = np.clip(1.0 - loss, 0.0, 1.0)

    state.fatigue = cfg.phi_coeff * state.fatigue + 0.1 * delta_norm
    phi = np.tanh(state.fatigue)

    weighted = cfg.w1 * novelty + cfg.w2 * state.retention + cfg.w3 * payoff
    continuity = np.exp(-cfg.continuity_beta * delta_norm)
    decay = np.exp(-cfg.salience_lambda * step)
    salience = weighted * continuity * decay * (1.0 - phi)
    salience = np.clip(0.6 + 0.6 * salience, 1e-4, 1.5)

    state.template = state.template + 0.05 * delta
    return salience, state


def train_regime(
    cfg: TrainingConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pulsed: bool,
) -> Dict[str, object]:
    rng = np.random.default_rng(cfg.seed + (123 if pulsed else 0))
    weights = init_weights(cfg, rng)
    sal_state = init_salience_state(cfg)

    num_batches = int(np.ceil(len(X_train) / cfg.batch_size))
    batches = np.array_split(np.arange(len(X_train)), num_batches)

    history = {
        "steps": [],
        "val_loss": [],
        "val_acc": [],
        "salience": [],
        "boost_active": [],
        "boost_blocked": [],
    }
    telemetry = GateTelemetry(floor=cfg.salience_floor)
    active_fraction_steps = 0

    t_converge_090 = None
    t_converge_099 = None

    cycle = cfg.pulse_on_steps + cfg.pulse_off_steps if pulsed else 0

    for step in range(1, cfg.steps + 1):
        batch_idx = batches[(step - 1) % num_batches]
        x_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]

        cache = forward(weights, x_batch)
        hidden = cache["h"]
        loss = cross_entropy(cache["probs"], y_batch)

        hidden_mean = np.mean(hidden, axis=0)
        channel_salience, sal_state = compute_channel_salience(hidden_mean, loss, sal_state, cfg, step)
        mean_salience = float(np.mean(channel_salience))
        history["salience"].append(mean_salience)

        grads = backward(weights, cache, x_batch, y_batch)

        accelerated = False
        blocked = False
        if pulsed:
            phase_idx = (step - 1) % cycle
            in_boost = phase_idx < cfg.pulse_on_steps

            if in_boost:
                gate_mult, accelerated, blocked = gate_gradient_scale(
                    mean_salience, cfg.salience_floor, cfg.alpha_boost, cfg.recovery_tax
                )
                if accelerated:
                    grads["latent"] *= gate_mult
                    grads["W1"] *= gate_mult
                    grads["b1"] *= gate_mult
                    grads["W2"] *= gate_mult
                elif blocked:
                    decay = 1.0 / (1.0 + cfg.recovery_tax)
                    grads["latent"] *= decay
                    grads["W1"] *= decay
                    grads["b1"] *= decay
                    grads["W2"] *= decay
                else:
                    # gate returned 1.0 being neutral
                    pass
            else:
                cooldown = 1.0 / (1.0 + cfg.recovery_tax)
                gate_mult, _, _ = gate_gradient_scale(mean_salience, cfg.salience_floor, cooldown, cfg.recovery_tax)
                grads["latent"] *= gate_mult
                grads["W1"] *= gate_mult
                grads["b1"] *= gate_mult
                grads["W2"] *= gate_mult
        else:
            # baseline: no boosts, mild consistency decay to compare fairly
            grads["latent"] *= 1.0

        apply_gradients(weights, grads, cfg.lr)
        telemetry.record(accelerated, blocked)
        history["boost_active"].append(1 if accelerated else 0)
        history["boost_blocked"].append(1 if blocked else 0)

        active_fraction = np.mean(channel_salience >= cfg.salience_threshold)
        if active_fraction >= 0.2:
            active_fraction_steps += 1

        if step % cfg.eval_interval == 0:
            val_cache = forward(weights, X_val)
            val_loss = cross_entropy(val_cache["probs"], y_val)
            val_acc = accuracy(val_cache["probs"], y_val)
            history["steps"].append(step)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            if t_converge_090 is None and val_acc >= 0.9:
                t_converge_090 = step
            if t_converge_099 is None and val_acc >= 0.99:
                t_converge_099 = step

    final_eval = forward(weights, X_val)
    final_acc = accuracy(final_eval["probs"], y_val)
    final_loss = cross_entropy(final_eval["probs"], y_val)

    mean_sal = float(np.mean(history["salience"])) if history["salience"] else 0.0
    min_sal = float(np.min(history["salience"])) if history["salience"] else 0.0
    var_sal = float(np.var(history["salience"])) if history["salience"] else 0.0

    return {
        "history": history,
        "t_converge_0.90": t_converge_090,
        "t_converge_0.99": t_converge_099,
        "final_acc": final_acc,
        "final_loss": final_loss,
        "mean_salience": mean_sal,
        "min_salience": min_sal,
        "var_salience": var_sal,
        "salience_active_fraction": active_fraction_steps / cfg.steps,
        "acceleration_pct": telemetry.acceleration_pct,
        "blocked_pct": telemetry.blocked_pct,
    }


def run_pulsed_experiment(alpha_values: Tuple[float, ...]) -> Dict[str, object]:
    cfg = TrainingConfig()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_val, y_val = generate_dataset(cfg)

    results: List[Dict[str, object]] = []
    best_entry: Dict[str, object] | None = None
    for alpha in alpha_values:
        base_cfg = TrainingConfig(**{**cfg.__dict__, "alpha_boost": 1.0})
        pulsed_cfg = TrainingConfig(**{**cfg.__dict__, "alpha_boost": alpha})

        baseline = train_regime(base_cfg, X_train, y_train, X_val, y_val, pulsed=False)
        pulsed = train_regime(pulsed_cfg, X_train, y_train, X_val, y_val, pulsed=True)

        # Determine speedups using 0.9 and 0.99 thresholds
        base_t90 = baseline["t_converge_0.90"] or base_cfg.steps
        base_t99 = baseline["t_converge_0.99"] or base_cfg.steps
        pulsed_t90 = pulsed["t_converge_0.90"] or pulsed_cfg.steps
        pulsed_t99 = pulsed["t_converge_0.99"] or pulsed_cfg.steps

        speedup_90 = base_t90 / pulsed_t90 if pulsed_t90 else float("nan")
        speedup_99 = base_t99 / pulsed_t99 if pulsed_t99 else float("nan")

        summary = {
            "t_converge_baseline_0.90": baseline["t_converge_0.90"],
            "t_converge_baseline_0.99": baseline["t_converge_0.99"],
            "t_converge_pulsed_0.90": pulsed["t_converge_0.90"],
            "t_converge_pulsed_0.99": pulsed["t_converge_0.99"],
            "effective_baseline_0.90": base_t90,
            "effective_baseline_0.99": base_t99,
            "effective_pulsed_0.90": pulsed_t90,
            "effective_pulsed_0.99": pulsed_t99,
            "speedup_0.90": speedup_90,
            "speedup_0.99": speedup_99,
        }

        run_entry = {
            "alpha_boost": alpha,
            "baseline": baseline,
            "pulsed": pulsed,
            "summary": summary,
        }
        results.append(run_entry)
        if best_entry is None or summary["effective_pulsed_0.99"] < best_entry["summary"]["effective_pulsed_0.99"]:
            best_entry = run_entry

    payload = {
        "config": cfg.__dict__,
        "runs": results,
    }
    if best_entry is not None:
        payload["best_run"] = {
            "alpha_boost": best_entry["alpha_boost"],
            **best_entry["summary"],
        }
    return payload


def write_artifact(payload: Dict[str, object]) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())
    record = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "experiment_name": "experiment_l_pulsed_overclock",
        "run_id": run_id,
        **payload,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"pulsed_overclock_{timestamp}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def main() -> None:
    alpha_values = (2.0, 4.0)
    payload = run_pulsed_experiment(alpha_values)
    artifact = write_artifact(payload)
    print(f"Pulsed overclock results written to {artifact}")
    for run in payload["runs"]:
        summary = run["summary"]
        print(
            "alpha={:.1f} | baseline t99={} | pulsed t99={} | speedup_0.99={:.3f}".format(
                run["alpha_boost"],
                summary["t_converge_baseline_0.99"] if summary["t_converge_baseline_0.99"] is not None else "NA",
                summary["t_converge_pulsed_0.99"] if summary["t_converge_pulsed_0.99"] is not None else "NA",
                summary["speedup_0.99"],
            )
        )


if __name__ == "__main__":
    main()
