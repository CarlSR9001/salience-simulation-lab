"""Experiment C: Time dilation via continuity strain in learning."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ARTIFACT_DIR = Path("artifacts/time_dilation")


@dataclass
class TrainingConfig:
    input_dim: int = 4
    hidden_dim: int = 64
    output_dim: int = 2
    dataset_size: int = 2000
    val_size: int = 400
    batch_size: int = 64
    steps: int = 5000
    eval_interval: int = 50
    lr: float = 0.01
    strain_period: int = 20
    strain_label_noise: float = 0.55
    strain_dropout: float = 0.5
    strain_tax: float = 35.0
    seed: int = 42
    salience_lambda: float = 1e-4
    phi_coeff: float = 0.6
    continuity_beta: float = 0.4
    w1: float = 0.6
    w2: float = 0.25
    w3: float = 0.15


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, targets: np.ndarray) -> float:
    n = probs.shape[0]
    clipped = np.clip(probs, 1e-9, 1.0)
    log_likelihood = -np.log(clipped[np.arange(n), targets])
    return float(np.mean(log_likelihood))


def accuracy(probs: np.ndarray, targets: np.ndarray) -> float:
    preds = np.argmax(probs, axis=1)
    return float(np.mean(preds == targets))


def generate_dataset(cfg: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    centers = rng.normal(size=(cfg.output_dim, cfg.input_dim)) * 2.0
    cov = np.eye(cfg.input_dim) * 0.5
    X = []
    y = []
    for _ in range(cfg.dataset_size):
        label = rng.integers(0, cfg.output_dim)
        sample = rng.multivariate_normal(centers[label], cov)
        X.append(sample)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    val_idx = rng.choice(cfg.dataset_size, size=cfg.val_size, replace=False)
    train_mask = np.ones(cfg.dataset_size, dtype=bool)
    train_mask[val_idx] = False
    return X[train_mask], y[train_mask], X[val_idx], y[val_idx]


def init_weights(cfg: TrainingConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    limit1 = math.sqrt(6.0 / (cfg.input_dim + cfg.hidden_dim))
    limit2 = math.sqrt(6.0 / (cfg.hidden_dim + cfg.output_dim))
    return {
        "W1": rng.uniform(-limit1, limit1, size=(cfg.input_dim, cfg.hidden_dim)),
        "b1": np.zeros(cfg.hidden_dim),
        "W2": rng.uniform(-limit2, limit2, size=(cfg.hidden_dim, cfg.output_dim)),
        "b2": np.zeros(cfg.output_dim),
    }


def forward(weights: Dict[str, np.ndarray], x: np.ndarray) -> Dict[str, np.ndarray]:
    h_preact = x @ weights["W1"] + weights["b1"]
    h = relu(h_preact)
    logits = h @ weights["W2"] + weights["b2"]
    probs = softmax(logits)
    return {"h": h, "logits": logits, "probs": probs}


def backward(weights: Dict[str, np.ndarray], cache: Dict[str, np.ndarray], x: np.ndarray, targets: np.ndarray) -> Dict[str, np.ndarray]:
    probs = cache["probs"].copy()
    probs[np.arange(len(targets)), targets] -= 1.0
    probs /= targets.shape[0]

    grad_W2 = cache["h"].T @ probs
    grad_b2 = np.sum(probs, axis=0)

    dh = probs @ weights["W2"].T
    dh[cache["h"] <= 0] = 0.0

    grad_W1 = x.T @ dh
    grad_b1 = np.sum(dh, axis=0)

    return {"W1": grad_W1, "b1": grad_b1, "W2": grad_W2, "b2": grad_b2}


def apply_gradients(weights: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], lr: float) -> None:
    for key in weights:
        weights[key] -= lr * grads[key]


@dataclass
class SalienceState:
    template: np.ndarray
    fatigue: float
    retention: float


def compute_salience(hidden_mean: np.ndarray, loss: float, state: SalienceState, cfg: TrainingConfig, step: int) -> Tuple[float, SalienceState]:
    delta = hidden_mean - state.template
    scale = math.sqrt(cfg.hidden_dim)
    delta_norm = np.linalg.norm(delta) / (scale + 1e-6)
    delta_a = 1.0 - math.exp(-delta_norm)

    retention_new = 0.9 * state.retention + 0.1 * math.exp(-delta_norm)
    payoff = np.clip(1.0 - loss, 0.0, 1.0)

    fatigue = 0.85 * state.fatigue + 0.15 * delta_norm
    phi = math.tanh(fatigue)

    weighted = cfg.w1 * delta_a + cfg.w2 * retention_new + cfg.w3 * payoff
    continuity = math.exp(-cfg.continuity_beta * delta_norm)
    decay = math.exp(-cfg.salience_lambda * step)
    salience = weighted * continuity * decay * (1.0 - cfg.phi_coeff * phi)
    salience = float(np.clip(salience, 1e-4, 1.5))

    template = state.template + 0.05 * delta
    new_state = SalienceState(template=template, fatigue=fatigue, retention=retention_new)
    return salience, new_state


def train_regime(cfg: TrainingConfig, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, strained: bool) -> Dict[str, object]:
    rng = np.random.default_rng(cfg.seed + (123 if strained else 0))
    weights = init_weights(cfg, rng)
    sal_state = SalienceState(template=np.zeros(cfg.hidden_dim), fatigue=0.0, retention=0.9)

    num_batches = int(np.ceil(len(X_train) / cfg.batch_size))
    batches = np.array_split(np.arange(len(X_train)), num_batches)

    history = {
        "steps": [],
        "val_loss": [],
        "val_acc": [],
        "salience": [],
    }

    t_converge = None

    for step in range(1, cfg.steps + 1):
        batch_idx = batches[(step - 1) % num_batches]
        x_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx].copy()

        if strained and step % cfg.strain_period == 0:
            flip_mask = rng.random(len(y_batch)) < cfg.strain_label_noise
            y_batch[flip_mask] = 1 - y_batch[flip_mask]

        cache = forward(weights, x_batch)
        hidden = cache["h"]
        probs = cache["probs"]
        loss = cross_entropy(probs, y_batch)

        hidden_mean = np.mean(hidden, axis=0)
        s_prime, sal_state = compute_salience(hidden_mean, loss, sal_state, cfg, step)
        if strained:
            mask = rng.random(hidden.shape) > cfg.strain_dropout
            hidden *= mask
            cache["h"] = hidden

        grads = backward(weights, cache, x_batch, y_batch)
        if strained:
            grads = {k: grads[k] / (1.0 + cfg.strain_tax * s_prime) for k in grads}
        apply_gradients(weights, grads, cfg.lr)

        if step % cfg.eval_interval == 0:
            val_cache = forward(weights, X_val)
            val_loss = cross_entropy(val_cache["probs"], y_val)
            val_acc = accuracy(val_cache["probs"], y_val)
            history["steps"].append(step)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["salience"].append(s_prime)
            if t_converge is None and val_acc >= 0.9:
                t_converge = step

    final_eval = forward(weights, X_val)
    final_acc = accuracy(final_eval["probs"], y_val)
    final_loss = cross_entropy(final_eval["probs"], y_val)

    mean_sal = float(np.mean(history["salience"]) if history["salience"] else s_prime)

    return {
        "history": history,
        "t_converge": t_converge,
        "final_acc": final_acc,
        "final_loss": final_loss,
        "mean_salience": mean_sal,
    }


def run_experiment() -> Dict[str, object]:
    cfg = TrainingConfig()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_val, y_val = generate_dataset(cfg)

    baseline = train_regime(cfg, X_train, y_train, X_val, y_val, strained=False)
    strained = train_regime(cfg, X_train, y_train, X_val, y_val, strained=True)

    tA = baseline["t_converge"] or cfg.steps
    tB = strained["t_converge"] or cfg.steps
    time_dilation = tB / tA if tA else float("nan")

    return {
        "config": cfg.__dict__,
        "baseline": baseline,
        "strained": strained,
        "time_dilation_factor": time_dilation,
    }


def write_artifact(data: Dict[str, object]) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())
    payload = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "experiment_name": "experiment_c_time_dilation",
        "run_id": run_id,
        **data,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"time_dilation_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2, default=lambda o: o if isinstance(o, (int, float, str, list, dict)) else str(o)))
    return path


def main() -> None:
    data = run_experiment()
    path = write_artifact(data)
    print(f"Results written to {path}")
    print(
        f"Baseline converge: {data['baseline']['t_converge']} steps, final acc {data['baseline']['final_acc']:.3f}, "
        f"mean S' {data['baseline']['mean_salience']:.3f}"
    )
    print(
        f"Strained converge: {data['strained']['t_converge']} steps, final acc {data['strained']['final_acc']:.3f}, "
        f"mean S' {data['strained']['mean_salience']:.3f}"
    )
    print(f"Time dilation factor: {data['time_dilation_factor']:.3f}")


if __name__ == "__main__":
    main()
