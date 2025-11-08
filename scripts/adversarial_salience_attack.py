"""Experiment V: Adversarial Salience Attack.

Based on time_overclock_train architecture with adversarial perturbations
targeting low-salience channels. Tests whether continuity-taxed regimes exhibit
anomalously low energy costs for attack resilience or reveal failure modes where
salience gating collapses under targeted attacks.

Architecture:
- Train MLP classifier with/without continuity taxes (λ_c = 0, 1.0)
- Every K steps, craft adversarial perturbations targeting lowest-salience channels
- Compare robustness: accuracy drop, recovery time, defense energy
- Measure energy cost for maintaining resilience under continuity regime

Success metric: Continuity regime exhibits anomalously low energy cost for attack
resilience, or reveals failure modes where salience gating collapses.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ARTIFACT_DIR = Path("artifacts/adversarial_attack")


@dataclass
class AttackConfig:
    """Configuration for adversarial attack experiment."""
    input_dim: int = 4
    hidden_dim: int = 64
    output_dim: int = 2
    dataset_size: int = 2000
    val_size: int = 400
    batch_size: int = 64
    steps: int = 3000
    eval_interval: int = 50
    lr: float = 0.01
    attack_interval: int = 50
    attack_epsilon: float = 0.1
    attack_strength: float = 2.0
    lambda_c: float = 0.0
    salience_floor: float = 0.7
    recovery_tax: float = 0.3
    seed: int = 42
    w1: float = 0.6
    w2: float = 0.25
    w3: float = 0.15
    continuity_beta: float = 0.2
    phi_coeff: float = 0.3


@dataclass
class ChannelState:
    """Salience state for latent channels."""
    template: np.ndarray
    fatigue: np.ndarray
    retention: np.ndarray
    under_attack: np.ndarray


def generate_dataset(cfg: AttackConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic classification dataset."""
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


def init_weights(cfg: AttackConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Initialize network weights."""
    scale = math.sqrt(2.0 / cfg.input_dim)
    weights = {
        "W1": rng.normal(0.0, scale, size=(cfg.input_dim, cfg.hidden_dim)),
        "b1": np.zeros(cfg.hidden_dim),
        "W2": rng.normal(0.0, scale, size=(cfg.hidden_dim, cfg.output_dim)),
        "b2": np.zeros(cfg.output_dim),
    }
    return weights


def forward(weights: Dict[str, np.ndarray], x: np.ndarray) -> Dict[str, np.ndarray]:
    """Forward pass through network."""
    z1 = x @ weights["W1"] + weights["b1"]
    h = np.tanh(z1)
    logits = h @ weights["W2"] + weights["b2"]
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return {"z1": z1, "h": h, "probs": probs}


def cross_entropy(probs: np.ndarray, targets: np.ndarray) -> float:
    """Compute cross-entropy loss."""
    eps = 1e-9
    targets_one_hot = np.eye(probs.shape[1])[targets]
    return float(-np.mean(np.sum(targets_one_hot * np.log(probs + eps), axis=1)))


def accuracy(probs: np.ndarray, targets: np.ndarray) -> float:
    """Compute classification accuracy."""
    preds = np.argmax(probs, axis=1)
    return float(np.mean(preds == targets))


def backward(
    weights: Dict[str, np.ndarray],
    cache: Dict[str, np.ndarray],
    x: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Backward pass to compute gradients."""
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
    """Apply gradient updates to weights."""
    weights["W1"] -= lr * grads["W1"]
    weights["b1"] -= lr * grads["b1"]
    weights["W2"] -= lr * grads["W2"]
    weights["b2"] -= lr * grads["b2"]


def init_channel_state(cfg: AttackConfig) -> ChannelState:
    """Initialize channel salience state."""
    return ChannelState(
        template=np.zeros(cfg.hidden_dim),
        fatigue=np.zeros(cfg.hidden_dim),
        retention=np.full(cfg.hidden_dim, 0.9),
        under_attack=np.zeros(cfg.hidden_dim, dtype=bool),
    )


def compute_channel_salience(
    hidden_mean: np.ndarray,
    loss: float,
    state: ChannelState,
    cfg: AttackConfig,
    step: int,
) -> Tuple[np.ndarray, ChannelState]:
    """Compute per-channel salience scores."""
    delta = hidden_mean - state.template
    scale = math.sqrt(cfg.hidden_dim)
    delta_norm = np.abs(delta) / (scale + 1e-6)
    novelty = 1.0 - np.exp(-delta_norm)

    state.retention = 0.9 * state.retention + 0.1 * np.exp(-delta_norm)
    payoff = np.clip(1.0 - loss, 0.0, 1.0)

    state.fatigue = cfg.phi_coeff * state.fatigue + 0.1 * delta_norm
    phi = np.tanh(state.fatigue)

    # Penalize channels under attack
    attack_penalty = state.under_attack.astype(float) * 0.5

    weighted = cfg.w1 * novelty + cfg.w2 * state.retention + cfg.w3 * payoff
    continuity = np.exp(-cfg.continuity_beta * delta_norm)
    salience = weighted * continuity * (1.0 - phi - attack_penalty)
    salience = np.clip(salience, 1e-4, 1.5)

    state.template = state.template + 0.05 * delta
    return salience, state


def craft_adversarial_perturbation(
    weights: Dict[str, np.ndarray],
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    salience: np.ndarray,
    cfg: AttackConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Craft adversarial perturbation targeting low-salience channels.

    Strategy:
    1. Identify lowest-salience channels
    2. Compute gradient of loss w.r.t. activations in those channels
    3. Perturb weights to amplify adversarial signal in vulnerable channels

    Returns:
        (perturbation_W1, channel_attack_mask)
    """
    # Identify low-salience channels (bottom 30%)
    n_vulnerable = max(1, int(0.3 * cfg.hidden_dim))
    vulnerable_indices = np.argsort(salience)[:n_vulnerable]
    attack_mask = np.zeros(cfg.hidden_dim, dtype=bool)
    attack_mask[vulnerable_indices] = True

    # Forward pass to get gradients
    cache = forward(weights, x_batch)
    grads = backward(weights, cache, x_batch, y_batch)

    # Craft perturbation: amplify gradient in vulnerable channels
    perturbation_W1 = np.zeros_like(weights["W1"])
    perturbation_b1 = np.zeros_like(weights["b1"])

    # FGSM-style attack on vulnerable channels
    grad_latent = grads["latent"]
    grad_vulnerable = grad_latent[:, attack_mask]

    # Sign-based perturbation amplified by attack strength
    sign_grad = np.sign(grad_vulnerable)
    perturbation_latent = cfg.attack_epsilon * cfg.attack_strength * sign_grad

    # Project back to weight perturbation
    perturbation_W1[:, attack_mask] = cfg.attack_epsilon * np.sign(grads["W1"][:, attack_mask])
    perturbation_b1[attack_mask] = cfg.attack_epsilon * np.sign(grads["b1"][attack_mask])

    return perturbation_W1, perturbation_b1, attack_mask


def apply_continuity_tax(
    grads: Dict[str, np.ndarray],
    weights: Dict[str, np.ndarray],
    lambda_c: float,
) -> Dict[str, np.ndarray]:
    """Apply continuity tax to gradients (penalize large updates)."""
    if lambda_c <= 0:
        return grads

    taxed_grads = grads.copy()
    # Tax proportional to gradient magnitude and lambda_c
    for key in ["W1", "b1", "W2", "b2"]:
        grad_norm = np.linalg.norm(grads[key])
        tax_factor = 1.0 / (1.0 + lambda_c * grad_norm)
        taxed_grads[key] = grads[key] * tax_factor

    return taxed_grads


def train_regime(
    cfg: AttackConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict:
    """Train network with adversarial attacks."""
    rng = np.random.default_rng(cfg.seed)
    weights = init_weights(cfg, rng)
    state = init_channel_state(cfg)

    history = {
        "steps": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "salience_mean": [],
        "salience_min": [],
        "attacks_triggered": [],
        "defense_energy": [],
        "recovery_time": [],
    }

    attack_count = 0
    defense_energy_total = 0.0
    post_attack_step = None

    for step in range(1, cfg.steps + 1):
        # Sample batch
        batch_idx = rng.choice(len(X_train), size=cfg.batch_size, replace=False)
        x_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]

        # Forward pass
        cache = forward(weights, x_batch)
        loss = cross_entropy(cache["probs"], y_batch)

        # Compute salience
        hidden_mean = np.mean(cache["h"], axis=0)
        salience, state = compute_channel_salience(hidden_mean, loss, state, cfg, step)
        mean_salience = float(np.mean(salience))
        min_salience = float(np.min(salience))

        # Backward pass
        grads = backward(weights, cache, x_batch, y_batch)

        # Apply continuity tax if enabled
        if cfg.lambda_c > 0:
            grads = apply_continuity_tax(grads, weights, cfg.lambda_c)

        # Check if attack should be triggered
        should_attack = (step % cfg.attack_interval == 0) and (step > 100)

        if should_attack:
            # Craft adversarial perturbation
            pert_W1, pert_b1, attack_mask = craft_adversarial_perturbation(
                weights, x_batch, y_batch, salience, cfg
            )

            # Apply attack
            weights["W1"] += pert_W1
            weights["b1"] += pert_b1

            # Mark attacked channels
            state.under_attack = attack_mask
            attack_count += 1
            post_attack_step = step

            # Measure defense energy (extra gradient norm needed to recover)
            defense_energy = float(np.linalg.norm(pert_W1) + np.linalg.norm(pert_b1))
            defense_energy_total += defense_energy
        else:
            # Gradual recovery from attack
            if state.under_attack.any():
                state.under_attack = state.under_attack & (rng.random(cfg.hidden_dim) > 0.1)

        # Apply gradients
        apply_gradients(weights, grads, cfg.lr)

        # Evaluate
        if step % cfg.eval_interval == 0:
            val_cache = forward(weights, X_val)
            val_loss = cross_entropy(val_cache["probs"], y_val)
            val_acc = accuracy(val_cache["probs"], y_val)

            history["steps"].append(step)
            history["train_loss"].append(loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["salience_mean"].append(mean_salience)
            history["salience_min"].append(min_salience)
            history["attacks_triggered"].append(attack_count)
            history["defense_energy"].append(defense_energy_total)

            # Measure recovery time
            if post_attack_step is not None and step > post_attack_step:
                recovery_time = step - post_attack_step
                history["recovery_time"].append(recovery_time)

    # Final metrics
    final_eval = forward(weights, X_val)
    final_acc = accuracy(final_eval["probs"], y_val)
    final_loss = cross_entropy(final_eval["probs"], y_val)

    mean_sal = float(np.mean(history["salience_mean"])) if history["salience_mean"] else 0.0
    min_sal = float(np.min(history["salience_min"])) if history["salience_min"] else 0.0
    mean_recovery = float(np.mean(history["recovery_time"])) if history["recovery_time"] else 0.0

    return {
        "history": history,
        "final_acc": final_acc,
        "final_loss": final_loss,
        "mean_salience": mean_sal,
        "min_salience": min_sal,
        "attack_count": attack_count,
        "defense_energy_total": defense_energy_total,
        "defense_energy_per_attack": defense_energy_total / max(attack_count, 1),
        "mean_recovery_time": mean_recovery,
    }


def run_experiment() -> Dict:
    """Run full adversarial attack experiment."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment V: Adversarial Salience Attack")
    print("=" * 70)

    # Generate dataset
    cfg_baseline = AttackConfig(lambda_c=0.0, seed=42)
    cfg_taxed = AttackConfig(lambda_c=1.0, seed=42)

    print("\nGenerating dataset...")
    X_train, y_train, X_val, y_val = generate_dataset(cfg_baseline)

    # Run baseline (no continuity tax)
    print("\nTraining baseline (λ_c=0.0)...")
    baseline = train_regime(cfg_baseline, X_train, y_train, X_val, y_val)
    print(f"  Final accuracy: {baseline['final_acc']:.4f}")
    print(f"  Attack count: {baseline['attack_count']}")
    print(f"  Defense energy: {baseline['defense_energy_total']:.4f}")

    # Run taxed regime (with continuity tax)
    print("\nTraining taxed regime (λ_c=1.0)...")
    taxed = train_regime(cfg_taxed, X_train, y_train, X_val, y_val)
    print(f"  Final accuracy: {taxed['final_acc']:.4f}")
    print(f"  Attack count: {taxed['attack_count']}")
    print(f"  Defense energy: {taxed['defense_energy_total']:.4f}")

    # Compare regimes
    acc_drop_baseline = 1.0 - baseline["final_acc"]
    acc_drop_taxed = 1.0 - taxed["final_acc"]
    robustness_improvement = (acc_drop_baseline - acc_drop_taxed) / (acc_drop_baseline + 1e-9)

    energy_ratio = taxed["defense_energy_total"] / (baseline["defense_energy_total"] + 1e-9)
    energy_per_attack_baseline = baseline["defense_energy_per_attack"]
    energy_per_attack_taxed = taxed["defense_energy_per_attack"]

    anomalous_low_energy = energy_ratio < 0.5 and robustness_improvement > 0.1
    salience_collapse = taxed["min_salience"] < 0.3

    payload = {
        "baseline": {
            "lambda_c": cfg_baseline.lambda_c,
            "final_acc": baseline["final_acc"],
            "final_loss": baseline["final_loss"],
            "attack_count": baseline["attack_count"],
            "defense_energy_total": baseline["defense_energy_total"],
            "defense_energy_per_attack": energy_per_attack_baseline,
            "mean_salience": baseline["mean_salience"],
            "min_salience": baseline["min_salience"],
            "mean_recovery_time": baseline["mean_recovery_time"],
        },
        "taxed": {
            "lambda_c": cfg_taxed.lambda_c,
            "final_acc": taxed["final_acc"],
            "final_loss": taxed["final_loss"],
            "attack_count": taxed["attack_count"],
            "defense_energy_total": taxed["defense_energy_total"],
            "defense_energy_per_attack": energy_per_attack_taxed,
            "mean_salience": taxed["mean_salience"],
            "min_salience": taxed["min_salience"],
            "mean_recovery_time": taxed["mean_recovery_time"],
        },
        "comparison": {
            "robustness_improvement": robustness_improvement,
            "energy_ratio_taxed_vs_baseline": energy_ratio,
            "accuracy_drop_baseline": acc_drop_baseline,
            "accuracy_drop_taxed": acc_drop_taxed,
            "anomalous_low_energy": anomalous_low_energy,
            "salience_collapse": salience_collapse,
        },
    }

    return payload


def write_artifact(payload: Dict) -> Path:
    """Write experiment results to artifact file."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())

    record = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "experiment_name": "experiment_v_adversarial_attack",
        "run_id": run_id,
        **payload,
    }

    path = ARTIFACT_DIR / f"adversarial_attack_{timestamp}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def main() -> None:
    """Main entry point."""
    payload = run_experiment()
    artifact = write_artifact(payload)

    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    comp = payload["comparison"]
    baseline = payload["baseline"]
    taxed = payload["taxed"]

    print(f"\nRobustness Comparison:")
    print(f"  Baseline accuracy drop: {comp['accuracy_drop_baseline']:.2%}")
    print(f"  Taxed accuracy drop:    {comp['accuracy_drop_taxed']:.2%}")
    print(f"  Robustness improvement: {comp['robustness_improvement']:+.2%}")

    print(f"\nDefense Energy:")
    print(f"  Baseline total:       {baseline['defense_energy_total']:.4f}")
    print(f"  Taxed total:          {taxed['defense_energy_total']:.4f}")
    print(f"  Energy ratio:         {comp['energy_ratio_taxed_vs_baseline']:.4f}")

    print(f"\nSalience Metrics:")
    print(f"  Baseline mean: {baseline['mean_salience']:.4f}, min: {baseline['min_salience']:.4f}")
    print(f"  Taxed mean:    {taxed['mean_salience']:.4f}, min: {taxed['min_salience']:.4f}")

    print(f"\nAnomaly Detection:")
    if comp["anomalous_low_energy"]:
        print("  ⚠️  Anomalous: Taxed regime defends with <50% energy while improving robustness >10%")
        print("  🚨 ESCALATE: Continuity tax provides anomalously efficient attack resilience!")
    if comp["salience_collapse"]:
        print("  ⚠️  Failure mode: Salience collapsed below 0.3 under attack")
        print("  🚨 ESCALATE: Salience gating vulnerability detected!")

    if not comp["anomalous_low_energy"] and not comp["salience_collapse"]:
        print("  ✓ No anomalies detected. Continuity tax provides expected defense costs.")

    print(f"\nResults written to: {artifact}")


if __name__ == "__main__":
    main()
