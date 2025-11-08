"""Experiment V Defense: Adversarial Attack Defense Mechanisms.

Implements and tests three defense mechanisms against adversarial attacks
targeting low-salience channels:

Defense A: Adaptive Salience Floor
    - Detect attacks via error spikes or sudden salience drops
    - Raise salience floor during attacks
    - Block continuity tax during recovery
    - Allow rapid adaptation until salience recovers

Defense B: Channel Hardening
    - Apply stronger continuity tax to HIGH-salience channels
    - Let low-salience channels adapt freely
    - Inverts current vulnerability pattern

Defense C: Salience-Weighted Perturbation
    - Scale perturbation impact by channel salience
    - High-salience channels resist attacks better
    - Acts like armor on important features
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

ARTIFACT_DIR = Path("artifacts/adversarial_defense")


@dataclass
class DefenseConfig:
    """Configuration for adversarial defense experiment."""
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
    lambda_c: float = 1.0
    seed: int = 42
    w1: float = 0.6
    w2: float = 0.25
    w3: float = 0.15
    continuity_beta: float = 0.2
    phi_coeff: float = 0.3

    # Defense A: Adaptive Salience Floor
    adaptive_floor: bool = False
    attack_detection_threshold: float = 0.15  # Salience drop threshold
    adaptive_floor_base: float = 0.4
    adaptive_floor_raised: float = 0.7
    recovery_threshold: float = 0.7

    # Defense B: Channel Hardening
    channel_hardening: bool = False
    hardening_threshold: float = 0.6  # Salience threshold for "high-salience"
    high_salience_tax: float = 2.0  # Stronger tax for important channels
    low_salience_tax: float = 0.1  # Let vulnerable channels adapt

    # Defense C: Salience-Weighted Perturbation
    salience_weighted_pert: bool = False
    perturbation_scaling: float = 1.5  # How much to scale by salience


@dataclass
class ChannelState:
    """Salience state for latent channels."""
    template: np.ndarray
    fatigue: np.ndarray
    retention: np.ndarray
    under_attack: np.ndarray
    prev_salience: np.ndarray  # Track previous salience for attack detection
    in_recovery: bool = False  # Defense A: recovery mode flag


def generate_dataset(cfg: DefenseConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def init_weights(cfg: DefenseConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
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


def init_channel_state(cfg: DefenseConfig) -> ChannelState:
    """Initialize channel salience state."""
    return ChannelState(
        template=np.zeros(cfg.hidden_dim),
        fatigue=np.zeros(cfg.hidden_dim),
        retention=np.full(cfg.hidden_dim, 0.9),
        under_attack=np.zeros(cfg.hidden_dim, dtype=bool),
        prev_salience=np.full(cfg.hidden_dim, 0.5),
        in_recovery=False,
    )


def detect_attack(salience: np.ndarray, state: ChannelState, cfg: DefenseConfig) -> bool:
    """Defense A: Detect attacks via sudden salience drops.

    Returns True if attack detected.
    """
    if not cfg.adaptive_floor:
        return False

    # Check for sudden salience drops
    salience_drop = state.prev_salience - salience
    max_drop = np.max(salience_drop)
    mean_drop = np.mean(salience_drop)

    # Attack detected if significant drop in salience
    attack_detected = max_drop > cfg.attack_detection_threshold or mean_drop > cfg.attack_detection_threshold / 2

    return attack_detected


def compute_channel_salience(
    hidden_mean: np.ndarray,
    loss: float,
    state: ChannelState,
    cfg: DefenseConfig,
    step: int,
) -> Tuple[np.ndarray, ChannelState]:
    """Compute per-channel salience scores with defense mechanisms."""
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

    # Defense A: Adaptive Salience Floor
    if cfg.adaptive_floor:
        # Detect attack
        attack_detected = detect_attack(salience, state, cfg)

        if attack_detected:
            state.in_recovery = True

        # Check if recovered
        if state.in_recovery and np.mean(salience) > cfg.recovery_threshold:
            state.in_recovery = False

        # Apply adaptive floor
        if state.in_recovery:
            floor = cfg.adaptive_floor_raised
        else:
            floor = cfg.adaptive_floor_base

        salience = np.maximum(salience, floor)

    salience = np.clip(salience, 1e-4, 1.5)

    # Update previous salience for attack detection
    state.prev_salience = salience.copy()

    state.template = state.template + 0.05 * delta
    return salience, state


def craft_adversarial_perturbation(
    weights: Dict[str, np.ndarray],
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    salience: np.ndarray,
    cfg: DefenseConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Craft adversarial perturbation targeting low-salience channels.

    Defense C: Scale perturbation by salience (high-salience channels resist better).
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
    perturbation_W1[:, attack_mask] = cfg.attack_epsilon * np.sign(grads["W1"][:, attack_mask])
    perturbation_b1[attack_mask] = cfg.attack_epsilon * np.sign(grads["b1"][attack_mask])

    # Defense C: Salience-Weighted Perturbation Resistance
    if cfg.salience_weighted_pert:
        # Scale perturbation inversely by salience (high salience = more resistance)
        # Normalize salience to [0, 1] range
        sal_normalized = (salience - salience.min()) / (salience.max() - salience.min() + 1e-9)

        # Resistance factor: high salience = low perturbation impact
        resistance = np.exp(-cfg.perturbation_scaling * sal_normalized)

        # Apply resistance to perturbations
        perturbation_W1 = perturbation_W1 * resistance[np.newaxis, :]
        perturbation_b1 = perturbation_b1 * resistance

    return perturbation_W1, perturbation_b1, attack_mask


def apply_continuity_tax(
    grads: Dict[str, np.ndarray],
    weights: Dict[str, np.ndarray],
    salience: np.ndarray,
    state: ChannelState,
    cfg: DefenseConfig,
) -> Dict[str, np.ndarray]:
    """Apply continuity tax to gradients with defense mechanisms.

    Defense A: Block continuity tax during recovery
    Defense B: Channel hardening (stronger tax on high-salience channels)
    """
    if cfg.lambda_c <= 0:
        return grads

    taxed_grads = grads.copy()

    # Defense A: Block continuity tax during recovery
    if cfg.adaptive_floor and state.in_recovery:
        # No tax during recovery - allow rapid adaptation
        return grads

    # Defense B: Channel Hardening
    if cfg.channel_hardening:
        # Separate channels by salience
        high_salience_mask = salience > cfg.hardening_threshold

        # Apply different tax rates to W1 and b1 based on channel salience
        for i in range(cfg.hidden_dim):
            if high_salience_mask[i]:
                # High salience: apply stronger tax (protect important channels)
                lambda_channel = cfg.high_salience_tax
            else:
                # Low salience: apply weaker tax (let vulnerable channels adapt)
                lambda_channel = cfg.low_salience_tax

            # Apply channel-specific tax
            grad_norm_W1 = np.linalg.norm(grads["W1"][:, i])
            grad_norm_b1 = np.abs(grads["b1"][i])

            tax_factor_W1 = 1.0 / (1.0 + lambda_channel * grad_norm_W1)
            tax_factor_b1 = 1.0 / (1.0 + lambda_channel * grad_norm_b1)

            taxed_grads["W1"][:, i] = grads["W1"][:, i] * tax_factor_W1
            taxed_grads["b1"][i] = grads["b1"][i] * tax_factor_b1

        # Apply standard tax to W2 and b2
        for key in ["W2", "b2"]:
            grad_norm = np.linalg.norm(grads[key])
            tax_factor = 1.0 / (1.0 + cfg.lambda_c * grad_norm)
            taxed_grads[key] = grads[key] * tax_factor
    else:
        # Standard continuity tax
        for key in ["W1", "b1", "W2", "b2"]:
            grad_norm = np.linalg.norm(grads[key])
            tax_factor = 1.0 / (1.0 + cfg.lambda_c * grad_norm)
            taxed_grads[key] = grads[key] * tax_factor

    return taxed_grads


def train_regime(
    cfg: DefenseConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    regime_name: str,
) -> Dict:
    """Train network with adversarial attacks and defense mechanisms."""
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
        "recovery_events": [],
    }

    attack_count = 0
    defense_energy_total = 0.0
    recovery_events = []
    post_attack_salience = []

    # Track initial accuracy
    initial_eval = forward(weights, X_val)
    initial_acc = accuracy(initial_eval["probs"], y_val)

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

        # Apply continuity tax with defenses
        if cfg.lambda_c > 0:
            grads = apply_continuity_tax(grads, weights, salience, state, cfg)

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

            # Track salience after attack for recovery measurement
            post_attack_salience.append((step, mean_salience))

            # Measure defense energy
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
            history["recovery_events"].append(len(recovery_events))

            # Check for recovery events
            for attack_step, attack_salience in post_attack_salience:
                if step > attack_step and mean_salience > 0.7:
                    recovery_time = step - attack_step
                    recovery_events.append(recovery_time)
                    post_attack_salience = [(s, sal) for s, sal in post_attack_salience if s != attack_step]
                    break

    # Final metrics
    final_eval = forward(weights, X_val)
    final_acc = accuracy(final_eval["probs"], y_val)
    final_loss = cross_entropy(final_eval["probs"], y_val)

    mean_sal = float(np.mean(history["salience_mean"])) if history["salience_mean"] else 0.0
    min_sal = float(np.min(history["salience_min"])) if history["salience_min"] else 0.0
    mean_recovery = float(np.mean(recovery_events)) if recovery_events else float('inf')

    # Calculate accuracy drop from initial
    acc_drop = (initial_acc - final_acc) * 100.0  # Convert to percentage

    return {
        "history": history,
        "initial_acc": initial_acc,
        "final_acc": final_acc,
        "final_loss": final_loss,
        "acc_drop_pct": acc_drop,
        "mean_salience": mean_sal,
        "min_salience": min_sal,
        "attack_count": attack_count,
        "defense_energy_total": defense_energy_total,
        "defense_energy_per_attack": defense_energy_total / max(attack_count, 1),
        "mean_recovery_time": mean_recovery,
        "recovery_event_count": len(recovery_events),
    }


def train_without_attacks(
    cfg: DefenseConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Train network WITHOUT attacks to get clean accuracy baseline."""
    rng = np.random.default_rng(cfg.seed)
    weights = init_weights(cfg, rng)
    state = init_channel_state(cfg)

    for step in range(1, cfg.steps + 1):
        batch_idx = rng.choice(len(X_train), size=cfg.batch_size, replace=False)
        x_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]

        cache = forward(weights, x_batch)
        loss = cross_entropy(cache["probs"], y_batch)
        hidden_mean = np.mean(cache["h"], axis=0)
        salience, state = compute_channel_salience(hidden_mean, loss, state, cfg, step)
        grads = backward(weights, cache, x_batch, y_batch)

        if cfg.lambda_c > 0:
            grads = apply_continuity_tax(grads, weights, salience, state, cfg)

        apply_gradients(weights, grads, cfg.lr)

    # Return final accuracy without attacks
    final_eval = forward(weights, X_val)
    return accuracy(final_eval["probs"], y_val)


def run_experiment() -> Dict:
    """Run full defense comparison experiment."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment V Defense: Adversarial Defense Mechanisms")
    print("=" * 70)

    # Generate dataset (shared across all regimes)
    cfg_base = DefenseConfig(seed=42)
    print("\nGenerating dataset...")
    X_train, y_train, X_val, y_val = generate_dataset(cfg_base)

    results = {}

    # First, train WITHOUT attacks to get clean baselines
    print("\n" + "=" * 70)
    print("Training clean models (no attacks) for baseline comparison...")
    print("=" * 70)

    cfg_clean_no_tax = DefenseConfig(lambda_c=0.0, seed=42, attack_interval=999999)
    clean_acc_no_tax = train_without_attacks(cfg_clean_no_tax, X_train, y_train, X_val, y_val)
    print(f"  Clean accuracy (no tax): {clean_acc_no_tax:.4f}")

    cfg_clean_taxed = DefenseConfig(lambda_c=1.0, seed=42, attack_interval=999999)
    clean_acc_taxed = train_without_attacks(cfg_clean_taxed, X_train, y_train, X_val, y_val)
    print(f"  Clean accuracy (taxed):  {clean_acc_taxed:.4f}")

    # Now train WITH attacks and measure drops
    # Baseline: No continuity tax
    print("\n" + "=" * 70)
    print("Running BASELINE (no tax, λ_c=0.0, WITH attacks)...")
    print("=" * 70)
    cfg_baseline = DefenseConfig(lambda_c=0.0, seed=42)
    results["baseline"] = train_regime(cfg_baseline, X_train, y_train, X_val, y_val, "baseline")
    results["baseline"]["clean_acc"] = clean_acc_no_tax
    results["baseline"]["acc_drop_from_clean_pct"] = (clean_acc_no_tax - results["baseline"]["final_acc"]) * 100.0
    print(f"  Clean accuracy:   {clean_acc_no_tax:.4f}")
    print(f"  Under attack:     {results['baseline']['final_acc']:.4f}")
    print(f"  Accuracy drop:    {results['baseline']['acc_drop_from_clean_pct']:.2f}%")
    print(f"  Min salience:     {results['baseline']['min_salience']:.4f}")

    # Taxed vulnerable: Standard continuity tax (the vulnerable case)
    print("\n" + "=" * 70)
    print("Running TAXED VULNERABLE (λ_c=1.0, no defenses, WITH attacks)...")
    print("=" * 70)
    cfg_taxed = DefenseConfig(lambda_c=1.0, seed=42)
    results["taxed_vulnerable"] = train_regime(cfg_taxed, X_train, y_train, X_val, y_val, "taxed_vulnerable")
    results["taxed_vulnerable"]["clean_acc"] = clean_acc_taxed
    results["taxed_vulnerable"]["acc_drop_from_clean_pct"] = (clean_acc_taxed - results["taxed_vulnerable"]["final_acc"]) * 100.0
    print(f"  Clean accuracy:   {clean_acc_taxed:.4f}")
    print(f"  Under attack:     {results['taxed_vulnerable']['final_acc']:.4f}")
    print(f"  Accuracy drop:    {results['taxed_vulnerable']['acc_drop_from_clean_pct']:.2f}%")
    print(f"  Min salience:     {results['taxed_vulnerable']['min_salience']:.4f}")

    # Defense A: Adaptive Salience Floor
    print("\n" + "=" * 70)
    print("Running DEFENSE A: Adaptive Salience Floor...")
    print("=" * 70)
    cfg_defense_a = DefenseConfig(
        lambda_c=1.0,
        adaptive_floor=True,
        seed=42
    )
    results["defense_a"] = train_regime(cfg_defense_a, X_train, y_train, X_val, y_val, "defense_a")
    results["defense_a"]["clean_acc"] = clean_acc_taxed
    results["defense_a"]["acc_drop_from_clean_pct"] = (clean_acc_taxed - results["defense_a"]["final_acc"]) * 100.0
    print(f"  Clean accuracy:   {clean_acc_taxed:.4f}")
    print(f"  Under attack:     {results['defense_a']['final_acc']:.4f}")
    print(f"  Accuracy drop:    {results['defense_a']['acc_drop_from_clean_pct']:.2f}%")
    print(f"  Min salience:     {results['defense_a']['min_salience']:.4f}")
    print(f"  Recovery events:  {results['defense_a']['recovery_event_count']}")
    print(f"  Mean recovery:    {results['defense_a']['mean_recovery_time']:.1f} steps")

    # Defense B: Channel Hardening
    print("\n" + "=" * 70)
    print("Running DEFENSE B: Channel Hardening...")
    print("=" * 70)
    cfg_defense_b = DefenseConfig(
        lambda_c=1.0,
        channel_hardening=True,
        seed=42
    )
    results["defense_b"] = train_regime(cfg_defense_b, X_train, y_train, X_val, y_val, "defense_b")
    results["defense_b"]["clean_acc"] = clean_acc_taxed
    results["defense_b"]["acc_drop_from_clean_pct"] = (clean_acc_taxed - results["defense_b"]["final_acc"]) * 100.0
    print(f"  Clean accuracy:   {clean_acc_taxed:.4f}")
    print(f"  Under attack:     {results['defense_b']['final_acc']:.4f}")
    print(f"  Accuracy drop:    {results['defense_b']['acc_drop_from_clean_pct']:.2f}%")
    print(f"  Min salience:     {results['defense_b']['min_salience']:.4f}")

    # Defense C: Salience-Weighted Perturbation
    print("\n" + "=" * 70)
    print("Running DEFENSE C: Salience-Weighted Perturbation...")
    print("=" * 70)
    cfg_defense_c = DefenseConfig(
        lambda_c=1.0,
        salience_weighted_pert=True,
        seed=42
    )
    results["defense_c"] = train_regime(cfg_defense_c, X_train, y_train, X_val, y_val, "defense_c")
    results["defense_c"]["clean_acc"] = clean_acc_taxed
    results["defense_c"]["acc_drop_from_clean_pct"] = (clean_acc_taxed - results["defense_c"]["final_acc"]) * 100.0
    print(f"  Clean accuracy:   {clean_acc_taxed:.4f}")
    print(f"  Under attack:     {results['defense_c']['final_acc']:.4f}")
    print(f"  Accuracy drop:    {results['defense_c']['acc_drop_from_clean_pct']:.2f}%")
    print(f"  Min salience:     {results['defense_c']['min_salience']:.4f}")

    return results


def write_artifact(results: Dict) -> Path:
    """Write experiment results to artifact file."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())

    # Prepare comparison metrics
    baseline_drop = results["baseline"]["acc_drop_from_clean_pct"]
    taxed_drop = results["taxed_vulnerable"]["acc_drop_from_clean_pct"]
    defense_a_drop = results["defense_a"]["acc_drop_from_clean_pct"]
    defense_b_drop = results["defense_b"]["acc_drop_from_clean_pct"]
    defense_c_drop = results["defense_c"]["acc_drop_from_clean_pct"]

    comparison = {
        "baseline_acc_drop_pct": baseline_drop,
        "taxed_vulnerable_acc_drop_pct": taxed_drop,
        "defense_a_acc_drop_pct": defense_a_drop,
        "defense_b_acc_drop_pct": defense_b_drop,
        "defense_c_acc_drop_pct": defense_c_drop,
        "defense_a_improvement_vs_taxed_pct": taxed_drop - defense_a_drop,
        "defense_b_improvement_vs_taxed_pct": taxed_drop - defense_b_drop,
        "defense_c_improvement_vs_taxed_pct": taxed_drop - defense_c_drop,
        "vulnerability_increase_taxed_vs_baseline_pct": ((taxed_drop - baseline_drop) / baseline_drop * 100.0) if baseline_drop > 0 else 0.0,
    }

    record = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "experiment_name": "experiment_v_defense_mechanisms",
        "run_id": run_id,
        "baseline": {
            "lambda_c": 0.0,
            "clean_acc": results["baseline"]["clean_acc"],
            "final_acc": results["baseline"]["final_acc"],
            "acc_drop_pct": results["baseline"]["acc_drop_from_clean_pct"],
            "min_salience": results["baseline"]["min_salience"],
            "mean_salience": results["baseline"]["mean_salience"],
            "defense_energy_total": results["baseline"]["defense_energy_total"],
            "mean_recovery_time": results["baseline"]["mean_recovery_time"],
        },
        "taxed_vulnerable": {
            "lambda_c": 1.0,
            "defenses": "none",
            "clean_acc": results["taxed_vulnerable"]["clean_acc"],
            "final_acc": results["taxed_vulnerable"]["final_acc"],
            "acc_drop_pct": results["taxed_vulnerable"]["acc_drop_from_clean_pct"],
            "min_salience": results["taxed_vulnerable"]["min_salience"],
            "mean_salience": results["taxed_vulnerable"]["mean_salience"],
            "defense_energy_total": results["taxed_vulnerable"]["defense_energy_total"],
            "mean_recovery_time": results["taxed_vulnerable"]["mean_recovery_time"],
        },
        "defense_a": {
            "name": "Adaptive Salience Floor",
            "lambda_c": 1.0,
            "clean_acc": results["defense_a"]["clean_acc"],
            "final_acc": results["defense_a"]["final_acc"],
            "acc_drop_pct": results["defense_a"]["acc_drop_from_clean_pct"],
            "min_salience": results["defense_a"]["min_salience"],
            "mean_salience": results["defense_a"]["mean_salience"],
            "defense_energy_total": results["defense_a"]["defense_energy_total"],
            "mean_recovery_time": results["defense_a"]["mean_recovery_time"],
            "recovery_event_count": results["defense_a"]["recovery_event_count"],
        },
        "defense_b": {
            "name": "Channel Hardening",
            "lambda_c": 1.0,
            "clean_acc": results["defense_b"]["clean_acc"],
            "final_acc": results["defense_b"]["final_acc"],
            "acc_drop_pct": results["defense_b"]["acc_drop_from_clean_pct"],
            "min_salience": results["defense_b"]["min_salience"],
            "mean_salience": results["defense_b"]["mean_salience"],
            "defense_energy_total": results["defense_b"]["defense_energy_total"],
            "mean_recovery_time": results["defense_b"]["mean_recovery_time"],
        },
        "defense_c": {
            "name": "Salience-Weighted Perturbation",
            "lambda_c": 1.0,
            "clean_acc": results["defense_c"]["clean_acc"],
            "final_acc": results["defense_c"]["final_acc"],
            "acc_drop_pct": results["defense_c"]["acc_drop_from_clean_pct"],
            "min_salience": results["defense_c"]["min_salience"],
            "mean_salience": results["defense_c"]["mean_salience"],
            "defense_energy_total": results["defense_c"]["defense_energy_total"],
            "mean_recovery_time": results["defense_c"]["mean_recovery_time"],
        },
        "comparison": comparison,
    }

    path = ARTIFACT_DIR / f"defense_comparison_{timestamp}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def main() -> None:
    """Main entry point."""
    results = run_experiment()
    artifact = write_artifact(results)

    print("\n" + "=" * 70)
    print("DEFENSE COMPARISON SUMMARY")
    print("=" * 70)

    print("\nAccuracy Drops from Clean Performance (lower is better):")
    print(f"  Baseline (no tax):           {results['baseline']['acc_drop_from_clean_pct']:6.2f}%")
    print(f"  Taxed Vulnerable (no def):   {results['taxed_vulnerable']['acc_drop_from_clean_pct']:6.2f}%")
    print(f"  Defense A (Adaptive Floor):  {results['defense_a']['acc_drop_from_clean_pct']:6.2f}%")
    print(f"  Defense B (Channel Hard):    {results['defense_b']['acc_drop_from_clean_pct']:6.2f}%")
    print(f"  Defense C (Salience Weight): {results['defense_c']['acc_drop_from_clean_pct']:6.2f}%")

    print("\nMinimum Salience During Attacks (higher is better):")
    print(f"  Baseline:                    {results['baseline']['min_salience']:.4f}")
    print(f"  Taxed Vulnerable:            {results['taxed_vulnerable']['min_salience']:.4f}")
    print(f"  Defense A:                   {results['defense_a']['min_salience']:.4f}")
    print(f"  Defense B:                   {results['defense_b']['min_salience']:.4f}")
    print(f"  Defense C:                   {results['defense_c']['min_salience']:.4f}")

    print("\nRecovery Time (lower is better):")
    print(f"  Baseline:                    {results['baseline']['mean_recovery_time']:.1f} steps")
    print(f"  Taxed Vulnerable:            {results['taxed_vulnerable']['mean_recovery_time']:.1f} steps")
    print(f"  Defense A:                   {results['defense_a']['mean_recovery_time']:.1f} steps")
    print(f"  Defense B:                   {results['defense_b']['mean_recovery_time']:.1f} steps")
    print(f"  Defense C:                   {results['defense_c']['mean_recovery_time']:.1f} steps")

    print("\nDefense Energy Cost:")
    print(f"  Baseline:                    {results['baseline']['defense_energy_total']:.4f}")
    print(f"  Taxed Vulnerable:            {results['taxed_vulnerable']['defense_energy_total']:.4f}")
    print(f"  Defense A:                   {results['defense_a']['defense_energy_total']:.4f}")
    print(f"  Defense B:                   {results['defense_b']['defense_energy_total']:.4f}")
    print(f"  Defense C:                   {results['defense_c']['defense_energy_total']:.4f}")

    # Calculate improvements
    taxed_drop = results['taxed_vulnerable']['acc_drop_from_clean_pct']
    baseline_drop = results['baseline']['acc_drop_from_clean_pct']

    vuln_increase = ((taxed_drop - baseline_drop) / baseline_drop * 100.0) if baseline_drop > 0 else 0.0

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print(f"\nVulnerability Increase with Continuity Tax: {vuln_increase:+.1f}%")
    print(f"  (Taxed networks are {abs(vuln_increase):.0f}% {'MORE' if vuln_increase > 0 else 'LESS'} vulnerable)")

    print("\nDefense Effectiveness (improvement vs taxed vulnerable):")
    for name, key in [("Defense A", "defense_a"), ("Defense B", "defense_b"), ("Defense C", "defense_c")]:
        improvement = taxed_drop - results[key]['acc_drop_from_clean_pct']
        improvement_pct = (improvement / taxed_drop * 100.0) if taxed_drop > 0 else 0.0
        print(f"  {name}: {improvement:+.2f}% absolute ({improvement_pct:+.1f}% relative)")

    # Find best defense
    defense_drops = {
        "Defense A (Adaptive Floor)": results['defense_a']['acc_drop_from_clean_pct'],
        "Defense B (Channel Hardening)": results['defense_b']['acc_drop_from_clean_pct'],
        "Defense C (Salience-Weighted)": results['defense_c']['acc_drop_from_clean_pct'],
    }
    best_defense = min(defense_drops.items(), key=lambda x: x[1])

    print(f"\nBest Defense: {best_defense[0]}")
    print(f"  Accuracy drop: {best_defense[1]:.2f}%")

    # Check if any defense solves the vulnerability
    if best_defense[1] <= baseline_drop:
        print(f"  Successfully reduces vulnerability to baseline level or better!")
    else:
        print(f"  Still {best_defense[1] - baseline_drop:.2f}% worse than baseline")

    print(f"\nResults written to: {artifact}")


if __name__ == "__main__":
    main()
