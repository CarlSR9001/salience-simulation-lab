"""Experiment I: Subjective time overclock via continuity-guided gradient boosts."""

from __future__ import annotations

import json
import math
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ARTIFACT_DIR = Path("artifacts/time_overclock")

from salience_floor_gate import GateTelemetry, gate_gradient_scale


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
    boost_interval: int = 2
    alpha_boost: float = 1.0
    salience_floor: float = 0.0
    salience_threshold: float = 0.72
    active_fraction_target: float = 0.3
    extra_step_scale: float = 1.5
    recovery_tax: float = 0.2
    seed: int = 42
    salience_lambda: float = 1e-5
    phi_coeff: float = 0.3
    continuity_beta: float = 0.2
    w1: float = 0.6
    w2: float = 0.25
    w3: float = 0.15
    strain_label_noise: float = 0.0
    strain_dropout: float = 0.2
    curriculum_low_pct: float = 0.4
    curriculum_high_pct: float = 0.8
    delta_variance_tau: float = 0.08
    align_beta: float = 0.9
    fast_lr: float = 0.025
    brake_window: int = 20
    brake_var_threshold: float = 0.003
    brake_min_factor: float = 0.35
    boost_align_min: float = 0.45
    boost_delta_cap: float = 0.18
    boost_loss_drop_min: float = 0.08


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
    weights = {
        "W1": rng.normal(0.0, scale, size=(cfg.input_dim, cfg.hidden_dim)),
        "b1": np.zeros(cfg.hidden_dim),
        "W2": rng.normal(0.0, scale, size=(cfg.hidden_dim, cfg.output_dim)),
        "b2": np.zeros(cfg.output_dim),
    }
    return weights


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
        template=np.zeros(cfg.hidden_dim),
        fatigue=np.zeros(cfg.hidden_dim),
        retention=np.full(cfg.hidden_dim, 0.9),
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
    salience = np.clip(salience, 1e-4, 1.5)

    state.template = state.template + 0.05 * delta
    return salience, state


def boost_gradients(
    grads: Dict[str, np.ndarray],
    salience: np.ndarray,
    cfg: TrainingConfig,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    if cfg.alpha_boost <= 1.0:
        return grads, np.ones_like(salience)

    boosted = grads.copy()
    salience_clipped = np.clip(salience, 1e-6, None)
    max_salience = float(np.max(salience_clipped))
    threshold = cfg.salience_threshold

    high_mask = salience_clipped >= threshold
    low_mask = ~high_mask

    scale = np.ones_like(salience_clipped)

    if max_salience > threshold:
        high_norm = (salience_clipped[high_mask] - threshold) / (max_salience - threshold + 1e-6)
        scale_high = 1.0 + (cfg.alpha_boost - 1.0) * np.clip(high_norm, 0.0, 1.0)
        scale[high_mask] = scale_high

    if np.any(low_mask):
        deficit = (threshold - salience_clipped[low_mask]) / threshold
        scale_low = np.clip(1.0 - 0.5 * deficit, 0.6, 1.0)
        scale[low_mask] = scale_low

    boosted_latent = boosted["latent"].copy()
    boosted_latent *= scale
    boosted["latent"] = boosted_latent
    boosted["W1"] *= scale
    boosted["b1"] *= scale
    boosted["W2"] *= scale[:, None]

    return boosted, scale


def train_regime(
    cfg: TrainingConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fast: bool,
) -> Dict[str, object]:
    rng = np.random.default_rng(cfg.seed + (321 if fast else 0))
    weights = init_weights(cfg, rng)
    sal_state = init_salience_state(cfg)

    history = {
        "steps": [],
        "val_loss": [],
        "val_acc": [],
        "salience": [],
        "boost_active": [],
        "stage": [],
        "brake": [],
        "loss_var": [],
    }
    active_steps = 0
    telemetry = GateTelemetry(floor=cfg.salience_floor)

    t_converge = None

    delta_cache = np.full(len(X_train), np.nan)
    filled_mask = np.zeros(len(X_train), dtype=bool)
    indices_all = np.arange(len(X_train))
    g_ema = None
    prev_loss = None
    grad_energy = 0.0
    loss_window: deque[float] = deque(maxlen=cfg.brake_window)
    stage_counts: Dict[str, int] = {"low": 0, "mid": 0, "full": 0}
    boost_counts: Dict[str, int] = {"low": 0, "mid": 0, "full": 0}

    def curriculum_stage(current_step: int) -> str:
        if current_step <= 500:
            return "low"
        if current_step <= 1000:
            return "mid"
        return "full"

    def sample_batch_indices(current_step: int) -> np.ndarray:
        if not fast:
            return rng.choice(indices_all, size=cfg.batch_size, replace=False)
        stage = curriculum_stage(current_step)
        filled_valid = filled_mask & ~np.isnan(delta_cache)
        if stage == "full" or np.count_nonzero(filled_valid) < cfg.batch_size:
            pool_idx = indices_all
        else:
            valid_values = delta_cache[filled_valid]
            if valid_values.size < cfg.batch_size:
                pool_idx = indices_all
            else:
                low_thr = float(np.quantile(valid_values, cfg.curriculum_low_pct))
                if stage == "low":
                    mask = (delta_cache <= low_thr) & filled_valid
                else:
                    high_thr = float(np.quantile(valid_values, cfg.curriculum_high_pct))
                    mask = (delta_cache >= low_thr) & (delta_cache <= high_thr) & filled_valid
                pool_idx = np.where(mask)[0]
                if pool_idx.size < cfg.batch_size:
                    pool_idx = indices_all
        return rng.choice(pool_idx, size=cfg.batch_size, replace=False)

    for step in range(1, cfg.steps + 1):
        current_stage = curriculum_stage(step)
        for attempt in range(2):
            batch_idx = sample_batch_indices(step)
            x_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            cache = forward(weights, x_batch)
            hidden = cache["h"]
            hidden_mean = np.mean(hidden, axis=0)
            if fast and attempt == 0 and np.var(hidden_mean) > cfg.delta_variance_tau:
                continue
            break

        loss = cross_entropy(cache["probs"], y_batch)
        loss_window.append(loss)

        template_snapshot = sal_state.template.copy()
        scale = math.sqrt(cfg.hidden_dim) + 1e-6
        delta_samples = np.linalg.norm(hidden - template_snapshot, axis=1) / scale
        delta_cache[batch_idx] = delta_samples
        filled_mask[batch_idx] = True
        batch_delta_mean = float(np.mean(delta_samples))

        hidden_mean = np.mean(hidden, axis=0)
        channel_salience, sal_state = compute_channel_salience(hidden_mean, loss, sal_state, cfg, step)
        mean_salience = float(np.mean(channel_salience))
        history["salience"].append(mean_salience)

        grads = backward(weights, cache, x_batch, y_batch)

        accelerated = False
        blocked = False

        lr_base = cfg.fast_lr if fast else cfg.lr
        brake_factor = 1.0
        loss_var = 0.0
        if fast and len(loss_window) > 1:
            loss_var = float(np.var(loss_window))
            if loss_var > 0.0 and cfg.brake_var_threshold > 0.0:
                ratio = loss_var / cfg.brake_var_threshold
                brake_factor = float(np.clip(1.0 - ratio, cfg.brake_min_factor, 1.0))
        lr_eff = lr_base * brake_factor

        flat_grads = np.concatenate([
            grads["W1"].ravel(),
            grads["b1"].ravel(),
            grads["W2"].ravel(),
            grads["b2"].ravel(),
        ])

        if g_ema is None:
            align = 1.0
        else:
            denom = float(np.linalg.norm(flat_grads) * np.linalg.norm(g_ema) + 1e-12)
            align = float(np.dot(flat_grads, g_ema) / denom) if denom > 0 else 0.0

        loss_drop_rel = 0.0
        if prev_loss is not None and prev_loss > 0.0:
            loss_drop_rel = (prev_loss - loss) / abs(prev_loss)

        retention_mean = float(np.mean(sal_state.retention))

        retention_ok = True if current_stage in ("low", "mid") else retention_mean >= 0.90

        if (
            fast
            and cfg.alpha_boost > 1.0
            and align >= cfg.boost_align_min
            and batch_delta_mean <= cfg.boost_delta_cap
            and loss_drop_rel >= cfg.boost_loss_drop_min
            and retention_ok
        ):
            accelerated = True
            lr_eff = lr_eff * (1.0 + 2.0 * align)

        apply_gradients(weights, grads, lr_eff)
        telemetry.record(accelerated, blocked)

        grad_energy += float(np.sum(np.abs(flat_grads)))
        stage_counts[current_stage] += 1
        if accelerated:
            boost_counts[current_stage] += 1

        history["brake"].append(brake_factor)
        history["loss_var"].append(loss_var)

        if g_ema is None:
            g_ema = flat_grads
        else:
            g_ema = cfg.align_beta * g_ema + (1.0 - cfg.align_beta) * flat_grads

        active_fraction = np.mean(channel_salience >= cfg.salience_threshold)
        if active_fraction >= cfg.active_fraction_target:
            active_steps += 1

        if step % cfg.eval_interval == 0:
            val_cache = forward(weights, X_val)
            val_loss = cross_entropy(val_cache["probs"], y_val)
            val_acc = accuracy(val_cache["probs"], y_val)
            history["steps"].append(step)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            if t_converge is None and val_acc >= 0.99:
                t_converge = step
        history["boost_active"].append(1 if accelerated else 0)
        history["stage"].append(current_stage)
        prev_loss = loss

    final_eval = forward(weights, X_val)
    final_acc = accuracy(final_eval["probs"], y_val)
    final_loss = cross_entropy(final_eval["probs"], y_val)

    mean_sal = float(np.mean(history["salience"])) if history["salience"] else 0.0
    salience_active_fraction = active_steps / cfg.steps

    return {
        "history": history,
        "t_converge": t_converge,
        "final_acc": final_acc,
        "final_loss": final_loss,
        "mean_salience": mean_sal,
        "salience_active_fraction": salience_active_fraction,
        "acceleration_pct": telemetry.acceleration_pct,
        "blocked_pct": telemetry.blocked_pct,
        "grad_energy": grad_energy,
        "boost_counts": boost_counts,
        "stage_counts": stage_counts,
        "brake_min": float(np.min(history["brake"])) if history["brake"] else 1.0,
        "brake_mean": float(np.mean(history["brake"])) if history["brake"] else 1.0,
    }


def run_experiment(alpha_values: Tuple[float, ...]) -> Dict[str, object]:
    cfg = TrainingConfig()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_val, y_val = generate_dataset(cfg)

    results = []
    best_entry = None
    for alpha in alpha_values:
        regime_cfg = cfg
        regime_cfg = TrainingConfig(**{**cfg.__dict__, "alpha_boost": alpha})
        baseline = train_regime(regime_cfg, X_train, y_train, X_val, y_val, fast=False)
        fast = train_regime(regime_cfg, X_train, y_train, X_val, y_val, fast=True)
        tA = baseline["t_converge"] if baseline["t_converge"] is not None else regime_cfg.steps
        tB = fast["t_converge"] if fast["t_converge"] is not None else regime_cfg.steps
        speedup = tA / tB if tB else float("nan")
        summary = {
            "t_converge_baseline": baseline["t_converge"],
            "t_converge_fast": fast["t_converge"],
            "t_converge_0.99": tB,
            "effective_baseline_steps": tA,
            "speedup_0.99": speedup,
        }
        run_entry = {
            "alpha_boost": alpha,
            "baseline": baseline,
            "fast": fast,
            "summary": summary,
        }
        results.append(run_entry)
        if best_entry is None or summary["t_converge_0.99"] < best_entry["summary"]["t_converge_0.99"]:
            best_entry = run_entry

    payload = {"config": cfg.__dict__, "runs": results}
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
        "experiment_name": "experiment_i_time_overclock",
        "run_id": run_id,
        **payload,
    }
    path = ARTIFACT_DIR / f"time_overclock_{timestamp}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def main() -> None:
    alpha_values = (1.0, 2.0, 4.0)
    payload = run_experiment(alpha_values)
    artifact = write_artifact(payload)
    print(f"Time overclock results written to {artifact}")
    for run in payload["runs"]:
        summary = run["summary"]
        baseline_t = summary["t_converge_baseline"]
        fast_t = summary["t_converge_fast"]
        speedup = summary["speedup_0.99"]
        print(
            "alpha={:.1f} | baseline t={} | fast t={} | effective_fast={} | speedup={:.3f}".format(
                run["alpha_boost"],
                baseline_t if baseline_t is not None else "NA",
                fast_t if fast_t is not None else "NA",
                summary["t_converge_0.99"],
                speedup,
            )
        )


if __name__ == "__main__":
    main()
