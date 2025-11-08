"""Parameter sweep v2: Alternative approaches for protective strain.

Findings from v1: Standard strain mechanisms (tax, dropout, noise) all DECREASE salience.
New hypothesis: We need mechanisms that REWARD salience or create adaptive strain.

Alternative approaches to test:
1. Inverse tax: BOOST gradients when salience is high (reward coherence)
2. Salience-adaptive strain: Reduce disruption when salience drops
3. Coherence bonus: Add term to salience that increases under strain
4. Selective strain: Only apply to low-salience updates
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import math

from time_dilation_train import (
    TrainingConfig,
    generate_dataset,
    init_weights,
    forward,
    backward,
    apply_gradients,
    SalienceState,
    cross_entropy,
    accuracy,
)

ARTIFACT_DIR = Path("artifacts/time_dilation")


def compute_salience_with_strain_bonus(
    hidden_mean: np.ndarray,
    loss: float,
    state: SalienceState,
    cfg: TrainingConfig,
    step: int,
    under_strain: bool,
    strain_bonus: float,
) -> tuple[float, SalienceState]:
    """Modified salience that rewards stability under strain."""
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

    # Base salience
    salience = weighted * continuity * decay * (1.0 - cfg.phi_coeff * phi)

    # Add strain bonus: reward high retention under strain
    if under_strain and strain_bonus > 0:
        coherence_bonus = strain_bonus * retention_new * continuity
        salience = salience + coherence_bonus

    salience = float(np.clip(salience, 1e-4, 1.5))

    template = state.template + 0.05 * delta
    new_state = SalienceState(template=template, fatigue=fatigue, retention=retention_new)
    return salience, new_state


def train_with_protective_mechanisms(
    cfg: TrainingConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    strained: bool,
    mechanism: str,
    mechanism_strength: float,
) -> Dict[str, object]:
    """Training with different protective mechanisms.

    Mechanisms:
    - 'inverse_tax': Boost gradients when salience is high (opposite of penalty)
    - 'adaptive_strain': Reduce dropout/noise when salience drops
    - 'strain_bonus': Add salience bonus under strain
    - 'selective_strain': Only apply strain to low-salience updates
    """
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
        "adaptive_factor": [],
    }

    t_converge = None
    salience_ema = 0.3  # Exponential moving average of salience

    for step in range(1, cfg.steps + 1):
        batch_idx = batches[(step - 1) % num_batches]
        x_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx].copy()

        # Adaptive strain: modulate disruption based on recent salience
        if mechanism == 'adaptive_strain' and strained:
            # If salience is dropping, reduce disruption
            adaptive_dropout = cfg.strain_dropout * (1.0 - mechanism_strength * (1.0 - salience_ema))
            adaptive_noise = cfg.strain_label_noise * (1.0 - mechanism_strength * (1.0 - salience_ema))
            adaptive_dropout = np.clip(adaptive_dropout, 0.0, cfg.strain_dropout)
            adaptive_noise = np.clip(adaptive_noise, 0.0, cfg.strain_label_noise)
        else:
            adaptive_dropout = cfg.strain_dropout
            adaptive_noise = cfg.strain_label_noise

        # Apply label noise
        if strained and step % cfg.strain_period == 0:
            flip_mask = rng.random(len(y_batch)) < adaptive_noise
            y_batch[flip_mask] = 1 - y_batch[flip_mask]

        cache = forward(weights, x_batch)
        hidden = cache["h"]
        probs = cache["probs"]
        loss = cross_entropy(probs, y_batch)

        hidden_mean = np.mean(hidden, axis=0)

        # Compute salience (with optional strain bonus)
        if mechanism == 'strain_bonus':
            s_prime, sal_state = compute_salience_with_strain_bonus(
                hidden_mean, loss, sal_state, cfg, step,
                under_strain=strained,
                strain_bonus=mechanism_strength
            )
        else:
            # Standard salience computation (from imported function)
            from time_dilation_train import compute_salience
            s_prime, sal_state = compute_salience(hidden_mean, loss, sal_state, cfg, step)

        # Update salience EMA
        salience_ema = 0.9 * salience_ema + 0.1 * s_prime

        # Apply dropout
        if strained:
            if mechanism == 'selective_strain':
                # Only apply dropout if salience is below threshold
                if s_prime < (0.3 * mechanism_strength):
                    mask = rng.random(hidden.shape) > cfg.strain_dropout
                    hidden *= mask
                # else: skip dropout when salience is high (protective)
            else:
                mask = rng.random(hidden.shape) > adaptive_dropout
                hidden *= mask
            cache["h"] = hidden

        grads = backward(weights, cache, x_batch, y_batch)

        # Apply strain tax or inverse tax
        if strained:
            if mechanism == 'inverse_tax':
                # BOOST gradients when salience is high (opposite of penalty)
                boost = 1.0 + mechanism_strength * s_prime
                grads = {k: grads[k] * boost for k in grads}
            elif mechanism == 'selective_strain':
                # Only tax low-salience updates
                if s_prime < (0.3 * mechanism_strength):
                    grads = {k: grads[k] / (1.0 + cfg.strain_tax * s_prime) for k in grads}
                # else: no tax when salience is high
            else:
                # Standard strain tax
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
            history["adaptive_factor"].append(salience_ema)
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


def run_protective_sweep():
    """Test alternative protective mechanisms."""
    base_cfg = TrainingConfig()
    X_train, y_train, X_val, y_val = generate_dataset(base_cfg)

    # Test configurations
    mechanisms = [
        ('inverse_tax', [0.5, 1.0, 2.0, 5.0]),  # Boost factor
        ('adaptive_strain', [0.3, 0.5, 0.7, 0.9]),  # Adaptation strength
        ('strain_bonus', [0.1, 0.2, 0.3, 0.5]),  # Bonus magnitude
        ('selective_strain', [0.5, 1.0, 1.5, 2.0]),  # Threshold multiplier
    ]

    # Also vary base strain parameters
    tax_values = [1.0, 5.0, 15.0, 35.0]
    dropout_values = [0.0, 0.2, 0.5]
    noise_values = [0.0, 0.2, 0.55]

    results = []

    print(f"\n{'='*80}")
    print(f"PROTECTIVE STRAIN SWEEP V2: Alternative Mechanisms")
    print(f"{'='*80}\n")

    total_tests = sum(len(strengths) for _, strengths in mechanisms) * len(tax_values)
    test_num = 0

    for mechanism_name, strength_values in mechanisms:
        print(f"\n--- Testing mechanism: {mechanism_name.upper()} ---\n")

        for tax in tax_values:
            for strength in strength_values:
                test_num += 1

                # Use moderate disruption for these tests
                cfg = replace(
                    base_cfg,
                    strain_tax=tax,
                    strain_dropout=0.2,
                    strain_label_noise=0.2,
                )

                print(f"[{test_num}/{total_tests}] {mechanism_name}, λ={tax}, strength={strength}...", end=" ")

                # Run baseline
                baseline = train_with_protective_mechanisms(
                    cfg, X_train, y_train, X_val, y_val,
                    strained=False,
                    mechanism=mechanism_name,
                    mechanism_strength=strength
                )

                # Run strained with protective mechanism
                strained = train_with_protective_mechanisms(
                    cfg, X_train, y_train, X_val, y_val,
                    strained=True,
                    mechanism=mechanism_name,
                    mechanism_strength=strength
                )

                # Compute metrics
                tA = baseline["t_converge"] or cfg.steps
                tB = strained["t_converge"] or cfg.steps
                time_dilation = tB / tA if tA else float("nan")

                sal_baseline = baseline["mean_salience"]
                sal_strained = strained["mean_salience"]
                salience_ratio = sal_strained / sal_baseline if sal_baseline > 0 else float("nan")
                salience_delta = sal_strained - sal_baseline

                is_protective = (
                    salience_ratio > 1.0 and
                    time_dilation > 1.2 and
                    strained["final_acc"] >= 0.9
                )

                result = {
                    "mechanism": mechanism_name,
                    "params": {
                        "continuity_tax": tax,
                        "mechanism_strength": strength,
                        "dropout": cfg.strain_dropout,
                        "noise": cfg.strain_label_noise,
                    },
                    "baseline": {
                        "t_converge": baseline["t_converge"],
                        "final_acc": baseline["final_acc"],
                        "mean_salience": sal_baseline,
                    },
                    "strained": {
                        "t_converge": strained["t_converge"],
                        "final_acc": strained["final_acc"],
                        "mean_salience": sal_strained,
                    },
                    "metrics": {
                        "time_dilation_factor": time_dilation,
                        "salience_ratio": salience_ratio,
                        "salience_delta": salience_delta,
                        "is_protective": is_protective,
                    },
                }
                results.append(result)

                protective_mark = "✓ PROTECTIVE!" if is_protective else ""
                print(
                    f"dilation={time_dilation:.2f}, "
                    f"sal_ratio={salience_ratio:.3f}, "
                    f"acc={strained['final_acc']:.3f} "
                    f"{protective_mark}"
                )

    # Analyze results
    protective_configs = [r for r in results if r["metrics"]["is_protective"]]

    print(f"\n{'='*80}")
    print(f"V2 SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"Total configurations tested: {len(results)}")
    print(f"Protective configurations found: {len(protective_configs)}")

    if protective_configs:
        print(f"\n{'='*80}")
        print(f"PROTECTIVE CONFIGURATIONS FOUND!")
        print(f"{'='*80}")

        # Sort by salience ratio
        protective_configs.sort(key=lambda x: x["metrics"]["salience_ratio"], reverse=True)

        for i, cfg in enumerate(protective_configs[:15], 1):
            m = cfg["mechanism"]
            p = cfg["params"]
            metrics = cfg["metrics"]
            s = cfg["strained"]
            print(f"\n{i}. Mechanism: {m}")
            print(f"   Parameters: λ={p['continuity_tax']}, strength={p['mechanism_strength']}, dropout={p['dropout']}, noise={p['noise']}")
            print(f"   Salience ratio: {metrics['salience_ratio']:.3f} (Δ={metrics['salience_delta']:+.4f})")
            print(f"   Time dilation: {metrics['time_dilation_factor']:.2f}×")
            print(f"   Final accuracy: {s['final_acc']:.3f}")

        # Best by mechanism
        print(f"\n{'='*80}")
        print(f"BEST CONFIGURATION BY MECHANISM")
        print(f"{'='*80}")

        for mechanism_name, _ in mechanisms:
            mech_configs = [r for r in protective_configs if r["mechanism"] == mechanism_name]
            if mech_configs:
                best = max(mech_configs, key=lambda x: x["metrics"]["salience_ratio"])
                p = best["params"]
                m = best["metrics"]
                print(f"\n{mechanism_name}:")
                print(f"  Best: λ={p['continuity_tax']}, strength={p['mechanism_strength']}")
                print(f"  Salience ratio: {m['salience_ratio']:.3f}, Dilation: {m['time_dilation_factor']:.2f}×")

    else:
        print("\n⚠ Still no protective configurations found!")
        print("Analysis of why mechanisms didn't work:\n")

        # Analyze by mechanism
        for mechanism_name, _ in mechanisms:
            mech_results = [r for r in results if r["mechanism"] == mechanism_name]
            if mech_results:
                best_sal = max(mech_results, key=lambda x: x["metrics"]["salience_ratio"])
                best_dil = max(mech_results, key=lambda x: x["metrics"]["time_dilation_factor"])

                print(f"\n{mechanism_name}:")
                print(f"  Best salience ratio: {best_sal['metrics']['salience_ratio']:.3f} (target: >1.0)")
                print(f"  Best time dilation: {best_dil['metrics']['time_dilation_factor']:.2f}× (target: >1.2)")
                print(f"  Issue: {'Salience still decreases' if best_sal['metrics']['salience_ratio'] < 1.0 else 'No time dilation'}")

    # Save results
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "experiment_name": "protective_strain_sweep_v2",
        "run_id": str(uuid.uuid4()),
        "results": results,
        "summary": {
            "total_tested": len(results),
            "protective_count": len(protective_configs),
        },
    }

    path = ARTIFACT_DIR / f"protective_strain_sweep_v2_{timestamp}.json"
    path.write_text(
        json.dumps(payload, indent=2, default=lambda o: o if isinstance(o, (int, float, str, list, dict, bool)) else str(o))
    )

    print(f"\n{'='*80}")
    print(f"Results saved to: {path}")
    print(f"{'='*80}\n")

    return results, protective_configs


if __name__ == "__main__":
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    run_protective_sweep()
