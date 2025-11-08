"""Critical sweep and universality analysis for SAL wormhole experiment.

This script scans the continuity strain parameter lambda_h, runs the wormhole
simulation for each value, and evaluates critical behavior:

* Throughput vs. soft-mode availability (heavy-traffic / critical slowing)
* Relaxation steps vs. soft-mode availability (time dilation exponent)
* Soft-mode rank collapse (feasible cone geometry)
* Percolation span statistics (jamming analogy)
* Effective update ratio (causal cone proxy)

Outputs a JSON artifact summarizing the sweep plus fitted exponents.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from tabulate import tabulate

from scripts.wormhole_tunnel_sim import WormholeConfig, run_experiment

ARTIFACT_DIR = Path("artifacts/wormhole")


@dataclass
class SweepResult:
    lambda_h: float
    throughput: float
    relaxation_steps: float
    soft_rank: float
    soft_rank_fraction: float
    sigma: float
    percolation_mean: float
    percolation_max: float
    update_ratio_mean: float
    candidate_norm_mean: float
    invariant_ok: bool


def compute_sigma(active_rank: float, soft_rank: float) -> float:
    if soft_rank <= 0:
        return 1.0
    frac = np.clip(active_rank / soft_rank, 0.0, 1.0)
    return 1.0 - frac


def fit_power_law(x: np.ndarray, y: np.ndarray) -> Dict[str, float] | None:
    mask = (x > 1e-6) & (y > 1e-9) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    try:
        slope, intercept = np.polyfit(lx, ly, 1)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None
    return {"slope": float(slope), "intercept": float(intercept)}


def analyse_sweep(results: List[SweepResult], critical_fraction: float) -> Dict[str, float | None]:
    soft_fracs = np.array([r.soft_rank_fraction for r in results])
    lambdas = np.array([r.lambda_h for r in results])

    # Estimate critical lambda when soft-mode fraction falls below the threshold
    below = np.where(soft_fracs <= critical_fraction)[0]
    if below.size == 0:
        lambda_crit = float(lambdas.max())
    else:
        idx = below[0]
        if idx == 0:
            lambda_crit = float(lambdas[0])
        else:
            lambda_crit = float(np.interp(critical_fraction, soft_fracs[idx - 1 : idx + 1], lambdas[idx - 1 : idx + 1]))

    # Prepare arrays for exponent fitting
    one_minus_sigma = np.clip(1.0 - np.array([r.sigma for r in results]), 1e-8, 1.0)
    throughput = np.array([r.throughput for r in results])
    relaxation = np.array([r.relaxation_steps for r in results])

    # Use only points with lambda <= lambda_crit for throughput exponent
    mask_sub = lambdas <= lambda_crit
    throughput_fit = fit_power_law(one_minus_sigma[mask_sub], throughput[mask_sub])

    relaxation_fit = None
    if np.all(relaxation[mask_sub] > 0):
        # Expect tau ∝ (1 - sigma)^(-z)
        inv_relax = 1.0 / (relaxation[mask_sub] + 1e-12)
        relaxation_fit = fit_power_law(one_minus_sigma[mask_sub], inv_relax)

    return {
        "lambda_crit_est": lambda_crit,
        "throughput_alpha": throughput_fit["slope"] if throughput_fit else None,
        "throughput_logA": throughput_fit["intercept"] if throughput_fit else None,
        "relaxation_z": -(relaxation_fit["slope"]) if relaxation_fit else None,
        "relaxation_logB": relaxation_fit["intercept"] if relaxation_fit else None,
    }


def run_sweep(
    lambda_values: np.ndarray,
    cfg_template: WormholeConfig,
    seed: int,
) -> List[SweepResult]:
    results: List[SweepResult] = []
    for lam in lambda_values:
        cfg = WormholeConfig(
            steps=cfg_template.steps,
            dim=cfg_template.dim,
            horizon_steps=cfg_template.horizon_steps,
            epsilon=cfg_template.epsilon,
            base_step_scale=cfg_template.base_step_scale,
            lambda_h=float(lam),
            budget_total=cfg_template.budget_total,
            wormhole_budget_fraction=cfg_template.wormhole_budget_fraction,
            wormhole_step_scale=cfg_template.wormhole_step_scale,
            soft_rank=cfg_template.soft_rank,
            verify_tolerance=cfg_template.verify_tolerance,
            debt_repay_rate=cfg_template.debt_repay_rate,
            lambda_c=cfg_template.lambda_c,
            max_packets=cfg_template.max_packets,
        )
        exp_results = run_experiment(cfg, seed)
        wormhole = next(r for r in exp_results if r.name == "wormhole")
        soft_rank_fraction = wormhole.metrics.get("soft_active_rank_mean", 0.0) / max(cfg.soft_rank, 1e-6)
        sigma = compute_sigma(wormhole.metrics.get("soft_active_rank_mean", 0.0), cfg.soft_rank)
        results.append(
            SweepResult(
                lambda_h=float(lam),
                throughput=wormhole.throughput,
                relaxation_steps=wormhole.metrics.get("relaxation_steps", float("nan")),
                soft_rank=float(wormhole.metrics.get("soft_active_rank_mean", 0.0)),
                soft_rank_fraction=float(soft_rank_fraction),
                sigma=float(sigma),
                percolation_mean=float(wormhole.metrics.get("percolation_span_mean", 0.0)),
                percolation_max=float(wormhole.metrics.get("percolation_span_max", 0.0)),
                update_ratio_mean=float(wormhole.metrics.get("update_ratio_mean", 0.0)),
                candidate_norm_mean=float(wormhole.metrics.get("candidate_norm_mean", 0.0)),
                invariant_ok=wormhole.invariant_ok,
            )
        )
    return results


def format_table(results: List[SweepResult]) -> str:
    rows = []
    for r in results:
        rows.append(
            (
                f"{r.lambda_h:.3f}",
                f"{r.sigma:.4f}",
                f"{r.soft_rank_fraction:.3f}",
                f"{r.throughput:.4f}",
                f"{r.relaxation_steps:.1f}",
                f"{r.percolation_mean:.2f}",
                f"{r.update_ratio_mean:.3f}",
                "yes" if r.invariant_ok else "no",
            )
        )
    headers = [
        "lambda_h",
        "sigma",
        "soft_frac",
        "throughput",
        "relax_steps",
        "percolation",
        "update_ratio",
        "invariants",
    ]
    return tabulate(rows, headers=headers, tablefmt="github")


def write_artifact(
    results: List[SweepResult],
    analysis: Dict[str, float | None],
    args: argparse.Namespace,
) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": timestamp,
        "lambda_start": args.lambda_start,
        "lambda_end": args.lambda_end,
        "lambda_points": args.lambda_points,
        "seed": args.seed,
        "analysis": analysis,
        "results": [
            {
                "lambda_h": r.lambda_h,
                "sigma": r.sigma,
                "soft_rank": r.soft_rank,
                "soft_rank_fraction": r.soft_rank_fraction,
                "throughput": r.throughput,
                "relaxation_steps": r.relaxation_steps,
                "percolation_mean": r.percolation_mean,
                "percolation_max": r.percolation_max,
                "update_ratio_mean": r.update_ratio_mean,
                "candidate_norm_mean": r.candidate_norm_mean,
                "invariant_ok": r.invariant_ok,
            }
            for r in results
        ],
    }
    path = ARTIFACT_DIR / f"wormhole_critical_sweep_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAL wormhole critical sweep")
    parser.add_argument("--lambda-start", type=float, default=60.0)
    parser.add_argument("--lambda-end", type=float, default=220.0)
    parser.add_argument("--lambda-points", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--horizon-steps", type=int, default=40)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--base-step-scale", type=float, default=0.06)
    parser.add_argument("--wormhole-step-scale", type=float, default=0.3)
    parser.add_argument("--wormhole-budget-fraction", type=float, default=0.12)
    parser.add_argument("--verify-tolerance", type=float, default=0.12)
    parser.add_argument("--debt-repay-rate", type=float, default=0.02)
    parser.add_argument("--max-packets", type=int, default=72)
    parser.add_argument("--soft-rank", type=int, default=8)
    parser.add_argument("--critical-fraction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lambda_values = np.linspace(args.lambda_start, args.lambda_end, args.lambda_points)

    template_cfg = WormholeConfig(
        steps=args.steps,
        horizon_steps=args.horizon_steps,
        epsilon=args.epsilon,
        base_step_scale=args.base_step_scale,
        lambda_h=args.lambda_start,
        wormhole_step_scale=args.wormhole_step_scale,
        wormhole_budget_fraction=args.wormhole_budget_fraction,
        verify_tolerance=args.verify_tolerance,
        debt_repay_rate=args.debt_repay_rate,
        max_packets=args.max_packets,
        soft_rank=args.soft_rank,
    )

    results = run_sweep(lambda_values, template_cfg, args.seed)
    analysis = analyse_sweep(results, args.critical_fraction)
    artifact = write_artifact(results, analysis, args)

    print(format_table(results))
    print("\nAnalysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    print(f"\nArtifact written to {artifact}")


if __name__ == "__main__":
    main()
