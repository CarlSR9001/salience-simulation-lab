"""Parameter sweep to find protective strain regime.

Goal: Find configurations where salience INCREASES under strain while still
achieving time dilation. This would indicate protective coherence rather than
destructive chaos.

Hypothesis: Gentle continuity tax WITHOUT heavy disruption might create
protective inertia that strengthens salience while slowing learning.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from time_dilation_train import (
    TrainingConfig,
    generate_dataset,
    init_weights,
    forward,
    backward,
    apply_gradients,
    SalienceState,
    compute_salience,
    cross_entropy,
    accuracy,
)

ARTIFACT_DIR = Path("artifacts/time_dilation")


@dataclass
class SweepConfig:
    """Parameter sweep configuration."""
    continuity_taxes: List[float]  # λ values to test
    dropouts: List[float]  # Dropout rates to test
    noises: List[float]  # Label noise rates to test
    salience_floor: float = 0.0  # Optional: block updates if S' < threshold
    use_salience_gating: bool = False  # Enable salience floor gating


def train_regime_with_gating(
    cfg: TrainingConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    strained: bool,
    salience_floor: float = 0.0,
    use_gating: bool = False,
) -> Dict[str, object]:
    """Training regime with optional salience floor gating."""
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
        "gated_updates": 0,  # Count of blocked updates
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

        # Apply strain tax if strained
        if strained:
            grads = {k: grads[k] / (1.0 + cfg.strain_tax * s_prime) for k in grads}

        # Salience floor gating: block updates if salience too low
        if use_gating and strained and s_prime < salience_floor:
            history["gated_updates"] += 1
            # Skip weight update to protect coherence
        else:
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
        "gated_updates": history["gated_updates"],
    }


def run_parameter_combination(
    base_cfg: TrainingConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    continuity_tax: float,
    dropout: float,
    noise: float,
    salience_floor: float,
    use_gating: bool,
) -> Dict[str, object]:
    """Run both baseline and strained regime for a parameter combination."""
    cfg = replace(
        base_cfg,
        strain_tax=continuity_tax,
        strain_dropout=dropout,
        strain_label_noise=noise,
    )

    # Run baseline (no strain)
    baseline = train_regime_with_gating(
        cfg, X_train, y_train, X_val, y_val,
        strained=False,
        salience_floor=salience_floor,
        use_gating=use_gating
    )

    # Run strained regime
    strained = train_regime_with_gating(
        cfg, X_train, y_train, X_val, y_val,
        strained=True,
        salience_floor=salience_floor,
        use_gating=use_gating
    )

    # Compute metrics
    tA = baseline["t_converge"] or cfg.steps
    tB = strained["t_converge"] or cfg.steps
    time_dilation = tB / tA if tA else float("nan")

    sal_baseline = baseline["mean_salience"]
    sal_strained = strained["mean_salience"]
    salience_ratio = sal_strained / sal_baseline if sal_baseline > 0 else float("nan")
    salience_delta = sal_strained - sal_baseline

    # Check if this is a "protective" configuration
    is_protective = (
        salience_ratio > 1.0 and  # Salience increases!
        time_dilation > 1.2 and   # Modest slowdown achieved
        strained["final_acc"] >= 0.9  # Still converges
    )

    return {
        "params": {
            "continuity_tax": continuity_tax,
            "dropout": dropout,
            "noise": noise,
            "salience_floor": salience_floor,
            "use_gating": use_gating,
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
            "gated_updates": strained["gated_updates"],
        },
        "metrics": {
            "time_dilation_factor": time_dilation,
            "salience_ratio": salience_ratio,
            "salience_delta": salience_delta,
            "is_protective": is_protective,
        },
    }


def run_parameter_sweep(sweep_cfg: SweepConfig) -> Dict[str, object]:
    """Run full parameter sweep."""
    base_cfg = TrainingConfig()
    X_train, y_train, X_val, y_val = generate_dataset(base_cfg)

    results = []
    total_combinations = (
        len(sweep_cfg.continuity_taxes) *
        len(sweep_cfg.dropouts) *
        len(sweep_cfg.noises)
    )

    print(f"\n{'='*80}")
    print(f"PROTECTIVE STRAIN PARAMETER SWEEP")
    print(f"{'='*80}")
    print(f"Testing {total_combinations} parameter combinations")
    print(f"Continuity taxes (λ): {sweep_cfg.continuity_taxes}")
    print(f"Dropouts: {sweep_cfg.dropouts}")
    print(f"Noises: {sweep_cfg.noises}")
    print(f"Salience floor gating: {sweep_cfg.use_salience_gating} (threshold={sweep_cfg.salience_floor})")
    print(f"{'='*80}\n")

    combination_num = 0
    for tax, dropout, noise in product(
        sweep_cfg.continuity_taxes,
        sweep_cfg.dropouts,
        sweep_cfg.noises,
    ):
        combination_num += 1
        print(f"[{combination_num}/{total_combinations}] Testing λ={tax}, dropout={dropout}, noise={noise}...", end=" ")

        result = run_parameter_combination(
            base_cfg, X_train, y_train, X_val, y_val,
            tax, dropout, noise,
            sweep_cfg.salience_floor,
            sweep_cfg.use_salience_gating,
        )
        results.append(result)

        # Print quick summary
        metrics = result["metrics"]
        protective_mark = "✓ PROTECTIVE" if metrics["is_protective"] else ""
        print(
            f"dilation={metrics['time_dilation_factor']:.2f}, "
            f"sal_ratio={metrics['salience_ratio']:.3f}, "
            f"acc={result['strained']['final_acc']:.3f} "
            f"{protective_mark}"
        )

    # Analyze results
    protective_configs = [r for r in results if r["metrics"]["is_protective"]]

    print(f"\n{'='*80}")
    print(f"SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"Total configurations tested: {len(results)}")
    print(f"Protective configurations found: {len(protective_configs)}")

    if protective_configs:
        print(f"\n{'='*80}")
        print(f"PROTECTIVE CONFIGURATIONS (salience increases + time dilation)")
        print(f"{'='*80}")

        # Sort by salience ratio (descending)
        protective_configs.sort(key=lambda x: x["metrics"]["salience_ratio"], reverse=True)

        for i, cfg in enumerate(protective_configs[:10], 1):  # Top 10
            p = cfg["params"]
            m = cfg["metrics"]
            s = cfg["strained"]
            print(f"\n{i}. λ={p['continuity_tax']}, dropout={p['dropout']}, noise={p['noise']}")
            print(f"   Salience ratio: {m['salience_ratio']:.3f} (Δ={m['salience_delta']:+.4f})")
            print(f"   Time dilation: {m['time_dilation_factor']:.2f}×")
            print(f"   Final accuracy: {s['final_acc']:.3f}")
            print(f"   Convergence: {s['t_converge']} steps")

        # Find best for different objectives
        best_salience = max(protective_configs, key=lambda x: x["metrics"]["salience_ratio"])
        best_balanced = max(protective_configs, key=lambda x: x["metrics"]["salience_ratio"] * x["metrics"]["time_dilation_factor"])

        print(f"\n{'='*80}")
        print(f"RECOMMENDATIONS")
        print(f"{'='*80}")

        print(f"\nHighest salience boost:")
        p = best_salience["params"]
        m = best_salience["metrics"]
        print(f"  Parameters: λ={p['continuity_tax']}, dropout={p['dropout']}, noise={p['noise']}")
        print(f"  Salience ratio: {m['salience_ratio']:.3f}")
        print(f"  Time dilation: {m['time_dilation_factor']:.2f}×")

        print(f"\nBest balanced (salience × dilation):")
        p = best_balanced["params"]
        m = best_balanced["metrics"]
        print(f"  Parameters: λ={p['continuity_tax']}, dropout={p['dropout']}, noise={p['noise']}")
        print(f"  Salience ratio: {m['salience_ratio']:.3f}")
        print(f"  Time dilation: {m['time_dilation_factor']:.2f}×")

    else:
        print("\n⚠ No protective configurations found in this sweep!")
        print("Consider adjusting parameter ranges or convergence criteria.")

        # Show what came closest
        best_attempt = max(results, key=lambda x: x["metrics"]["salience_ratio"])
        p = best_attempt["params"]
        m = best_attempt["metrics"]
        print(f"\nClosest attempt:")
        print(f"  Parameters: λ={p['continuity_tax']}, dropout={p['dropout']}, noise={p['noise']}")
        print(f"  Salience ratio: {m['salience_ratio']:.3f} (target: >1.0)")
        print(f"  Time dilation: {m['time_dilation_factor']:.2f}× (target: >1.2)")
        print(f"  Final accuracy: {best_attempt['strained']['final_acc']:.3f} (target: ≥0.9)")

    return {
        "sweep_config": {
            "continuity_taxes": sweep_cfg.continuity_taxes,
            "dropouts": sweep_cfg.dropouts,
            "noises": sweep_cfg.noises,
            "salience_floor": sweep_cfg.salience_floor,
            "use_salience_gating": sweep_cfg.use_salience_gating,
        },
        "results": results,
        "summary": {
            "total_tested": len(results),
            "protective_count": len(protective_configs),
            "best_salience_config": best_salience["params"] if protective_configs else None,
            "best_balanced_config": best_balanced["params"] if protective_configs else None,
        },
    }


def main():
    """Run protective strain parameter sweep."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Define sweep configuration
    sweep_cfg = SweepConfig(
        continuity_taxes=[1.0, 5.0, 15.0, 35.0],
        dropouts=[0.0, 0.2, 0.5],
        noises=[0.0, 0.2, 0.55],
        salience_floor=0.1,  # Block updates if S' < 0.1
        use_salience_gating=False,  # Start without gating
    )

    # Run sweep
    results = run_parameter_sweep(sweep_cfg)

    # Save results
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())

    payload = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "experiment_name": "protective_strain_sweep",
        "run_id": run_id,
        **results,
    }

    path = ARTIFACT_DIR / f"protective_strain_sweep_{timestamp}.json"
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=lambda o: o if isinstance(o, (int, float, str, list, dict, bool)) else str(o)
        )
    )

    print(f"\n{'='*80}")
    print(f"Results saved to: {path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
