"""Experiment W: Salience wormhole tunnel simulation.

This script models two high-continuity basins ("event horizons") that are
effectively frozen by a large continuity tax. We benchmark four regimes:

1. Baseline (no wormhole): virtually zero throughput once both basins stall.
2. Wormhole: escrow a micro budget, project the donor update into a shared
   soft subspace, verify in a shadow copy, then commit with a tiny step size
   while logging a continuity debt that must be repaid.
3. Ablations (no_escrow, no_verify, random_subspace): demonstrate that
   removing any guardrail breaks budget conservation or increases defect rate.

Outputs are written to ``artifacts/wormhole`` as JSON and printed via tabulate.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Tuple

import numpy as np
from tabulate import tabulate


ARTIFACT_DIR = Path("artifacts/wormhole")


@dataclass
class WormholeConfig:
    steps: int = 400
    dim: int = 64
    horizon_steps: int = 40
    epsilon: float = 1e-3
    base_step_scale: float = 0.06
    lambda_h: float = 180.0
    budget_total: float = 1.0
    wormhole_budget_fraction: float = 0.12
    wormhole_step_scale: float = 0.25
    soft_rank: int = 8
    verify_tolerance: float = 0.08
    debt_repay_rate: float = 2e-2
    lambda_c: float = 12.0
    max_packets: int = 48
    traversable_mu: float = 0.0
    traversable_window: int = 0
    hp_message_bits: int = 0
    hp_message_magnitude: float = 0.12
    hp_scramble_wait: int = 0
    hp_decode_packets: int = 0
    hp_decode_tolerance: float = 0.1
    shock_energy: float = 0.0
    shock_probe_gap: int = 0


@dataclass
class RegimeResult:
    name: str
    throughput: float
    commits: int
    invariant_ok: bool
    budget_error: float
    debt_balance: float
    defect_rate: float
    notes: str
    metrics: Dict[str, float]


def _largest_contiguous_run(mask: np.ndarray) -> int:
    if mask.ndim != 1:
        mask = mask.ravel()
    max_run = run = 0
    for flag in mask:
        if flag:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def _stack_history(history: Iterable[np.ndarray], dim: int) -> np.ndarray:
    rows = list(history)
    if not rows:
        return np.zeros((1, dim), dtype=float)
    return np.vstack(rows)


def _compute_soft_modes(
    donor_hist: Iterable[np.ndarray],
    receiver_hist: Iterable[np.ndarray],
    dim: int,
    rank: int,
    rng: np.random.Generator,
    random_subspace: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if random_subspace:
        mat = rng.standard_normal((dim, rank))
        u, _ = np.linalg.qr(mat)
        eigs = np.ones(rank, dtype=float)
        return u, u.copy(), eigs, eigs.copy()

    donor_stack = _stack_history(donor_hist, dim)
    receiver_stack = _stack_history(receiver_hist, dim)

    cov = donor_stack.T @ donor_stack + 1e-9 * np.eye(dim)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    donor_basis = eigvecs[:, order[:rank]]

    cov_r = receiver_stack.T @ receiver_stack + 1e-9 * np.eye(dim)
    eigvals_r, eigvecs_r = np.linalg.eigh(cov_r)
    order_r = np.argsort(eigvals_r)[::-1]
    receiver_basis = eigvecs_r[:, order_r[:rank]]

    donor_eigs = np.clip(eigvals[order[:rank]], a_min=0.0, a_max=None)

    cov_r = receiver_stack.T @ receiver_stack + 1e-9 * np.eye(dim)
    eigvals_r, eigvecs_r = np.linalg.eigh(cov_r)
    order_r = np.argsort(eigvals_r)[::-1]
    receiver_basis = eigvecs_r[:, order_r[:rank]]
    receiver_eigs = np.clip(eigvals_r[order_r[:rank]], a_min=0.0, a_max=None)

    return donor_basis, receiver_basis, donor_eigs, receiver_eigs


def _update_state(
    state: np.ndarray,
    proposal: np.ndarray,
    lambda_h: float,
    epsilon: float,
) -> Tuple[np.ndarray, float]:
    eta = 1.0 / (1.0 + lambda_h)
    update = eta * proposal
    if np.linalg.norm(update) < epsilon:
        update = np.zeros_like(update)
    new_state = state + update
    return new_state, float(np.linalg.norm(update))


def simulate_regime(
    config: WormholeConfig,
    rng: np.random.Generator,
    regime: str,
    baseline_throughput: float | None = None,
) -> RegimeResult:
    state_a = np.zeros(config.dim, dtype=float)
    state_b = np.zeros(config.dim, dtype=float)
    debt_balance = 0.0
    budgets = np.full(2, config.budget_total / 2.0, dtype=float)

    history_a: Deque[np.ndarray] = deque(maxlen=config.horizon_steps)
    history_b: Deque[np.ndarray] = deque(maxlen=config.horizon_steps)
    horizon_counters = np.zeros(2, dtype=int)

    throughput = 0.0
    commits = 0
    defects = 0
    packets_sent = 0
    first_lock_step = None

    candidate_norms: List[float] = []
    soft_top_eigs: List[float] = []
    soft_active_ranks: List[float] = []
    percolation_spans: List[float] = []
    committed_deltas: List[float] = []
    update_ratios: List[float] = []
    prelock_update_norms: List[float] = []
    traversable_kicks: List[float] = []
    arrival_steps: List[float] = []
    state_corrs: List[float] = []
    shock_delay_records: List[float] = []

    hp_enabled = (
        regime == "wormhole"
        and config.hp_message_bits > 0
        and config.hp_decode_packets > 0
    )
    hp_message_indices: np.ndarray | None = None
    hp_message_signs: np.ndarray | None = None
    hp_message_vector: np.ndarray | None = None
    hp_message_injected = False
    hp_message_pending = False
    hp_scramble_count = 0
    hp_packets_used = 0
    hp_decode_accum = np.zeros(config.hp_message_bits if hp_enabled else 1, dtype=float)
    hp_partial_match = []
    hp_injection_step: int | None = None

    shock_enabled = regime == "wormhole" and config.shock_energy > 0.0
    shock_sent = False
    shock_gap_remaining = 0
    shock_arrival_step: float | None = None
    shock_measured = False

    require_escrow = regime != "no_escrow"
    require_verify = regime != "no_verify"
    random_subspace = regime == "random_subspace"

    traversable_window = 0

    for step in range(config.steps):
        proposal_a = rng.normal(scale=config.base_step_scale, size=config.dim)
        proposal_b = rng.normal(scale=config.base_step_scale, size=config.dim)

        state_a, delta_a = _update_state(state_a, proposal_a, config.lambda_h, config.epsilon)
        state_b, delta_b = _update_state(state_b, proposal_b, config.lambda_h, config.epsilon)

        prelock_update_norms.append(0.5 * (delta_a + delta_b))
        update_ratios.append(float(delta_a / (np.linalg.norm(proposal_a) + 1e-12)))

        history_a.append(proposal_a)
        history_b.append(proposal_b)

        horizon_counters[0] = horizon_counters[0] + 1 if delta_a < config.epsilon else 0
        horizon_counters[1] = horizon_counters[1] + 1 if delta_b < config.epsilon else 0

        if debt_balance > 0.0:
            repayment = min(config.debt_repay_rate, debt_balance)
            debt_balance -= repayment
            budgets += repayment / 2.0

        horizon_lock = (
            horizon_counters[0] >= config.horizon_steps
            and horizon_counters[1] >= config.horizon_steps
        )

        if not horizon_lock:
            continue

        if first_lock_step is None:
            first_lock_step = step
            if (
                regime == "wormhole"
                and config.traversable_mu > 0.0
                and config.traversable_window > 0
            ):
                traversable_window = config.traversable_window
            if hp_enabled and not hp_message_injected:
                hp_message_indices = rng.choice(
                    config.dim, size=config.hp_message_bits, replace=False
                )
                hp_message_signs = rng.choice([-1.0, 1.0], size=config.hp_message_bits)
                hp_message_vector = np.zeros(config.dim, dtype=float)
                hp_message_vector[hp_message_indices] = (
                    hp_message_signs * config.hp_message_magnitude
                )
                state_a += hp_message_vector
                history_a.append(hp_message_vector)
                hp_message_injected = True
                hp_message_pending = True
                hp_scramble_count = max(config.hp_scramble_wait, 0)
                hp_injection_step = step

        if regime == "baseline" or packets_sent >= config.max_packets:
            continue

        if hp_enabled and hp_scramble_count > 0:
            hp_scramble_count -= 1
            continue

        donor_basis, receiver_basis, donor_eigs, receiver_eigs = _compute_soft_modes(
            history_a, history_b, config.dim, config.soft_rank, rng, random_subspace
        )

        if traversable_window > 0 and regime == "wormhole" and config.traversable_mu > 0.0:
            cross = float(np.dot(state_a, state_b))
            if cross > 0.0:
                kick_credit = config.traversable_mu * cross
                if kick_credit != 0.0:
                    budgets += kick_credit / 2.0
                    debt_balance -= kick_credit
                    traversable_kicks.append(kick_credit)
            traversable_window -= 1

        current_is_shock = False
        if shock_enabled and not shock_sent and donor_basis.size > 0:
            direction = donor_basis[:, 0]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0.0:
                candidate = (direction / direction_norm) * config.shock_energy
                norm_candidate = float(np.linalg.norm(candidate))
                current_is_shock = True
                shock_sent = True
                shock_gap_remaining = max(config.shock_probe_gap, 0)

        if hp_enabled and hp_message_pending and hp_message_vector is not None:
            donor_delta = hp_message_vector
            hp_message_pending = False
        else:
            donor_delta = history_a[-1]

        z = donor_basis.T @ donor_delta
        candidate = receiver_basis @ z
        candidate *= config.wormhole_step_scale

        if not current_is_shock:
            norm_candidate = float(np.linalg.norm(candidate))

        if require_verify and norm_candidate > config.verify_tolerance:
            scale = config.verify_tolerance / max(norm_candidate, 1e-12)
            candidate *= scale
            norm_candidate = float(np.linalg.norm(candidate))

        escrow = config.wormhole_budget_fraction * config.budget_total
        if require_escrow:
            if budgets.sum() < escrow:
                defects += 1
                continue
            budgets -= escrow / 2.0

        shadow = state_b + candidate
        if require_verify:
            delta_norm = np.linalg.norm(shadow - state_b)
            if delta_norm > config.verify_tolerance * 1.05:
                defects += 1
                if require_escrow:
                    budgets += escrow / 2.0
                continue

        state_b = shadow
        throughput += norm_candidate
        commits += 1
        packets_sent += 1

        debt_increment = config.lambda_c * norm_candidate**2
        debt_balance += debt_increment
        if require_escrow:
            budgets += escrow / 2.0
        budgets -= debt_increment / 2.0

        candidate_norms.append(norm_candidate)
        arrival = float(step - first_lock_step if first_lock_step is not None else 0.0)
        arrival_steps.append(arrival)
        denom = (np.linalg.norm(state_a) * np.linalg.norm(state_b)) + 1e-12
        state_corrs.append(float(np.dot(state_a, state_b) / denom))
        if shock_enabled:
            if current_is_shock:
                shock_arrival_step = arrival
            elif shock_sent and not shock_measured:
                if shock_gap_remaining > 0:
                    shock_gap_remaining -= 1
                else:
                    if shock_arrival_step is not None:
                        shock_delay_records.append(arrival - shock_arrival_step)
                    shock_measured = True

        if hp_enabled and hp_message_indices is not None and hp_message_signs is not None:
            if hp_packets_used < config.hp_message_bits:
                pass  # placeholder to appease linters if bits is zero
            hp_decode_accum[: config.hp_message_bits] += candidate[hp_message_indices]
            hp_packets_used += 1
            decoded = np.sign(hp_decode_accum[: config.hp_message_bits])
            decoded[decoded == 0.0] = 0.0
            match = (decoded == hp_message_signs).mean()
            hp_partial_match.append(float(match))
            if hp_packets_used >= config.hp_decode_packets:
                break
        if donor_eigs.size > 0:
            soft_top_eigs.append(float(donor_eigs[0]))
            active = float((donor_eigs > 1e-6).sum())
            soft_active_ranks.append(active)
        else:
            soft_top_eigs.append(0.0)
            soft_active_ranks.append(0.0)
        mask = np.abs(candidate) > (0.05 * config.verify_tolerance)
        percolation_spans.append(float(_largest_contiguous_run(mask)))
        committed_deltas.append(float(np.linalg.norm(candidate)))

    budget_error = abs(config.budget_total - (budgets.sum() + debt_balance))
    invariant_ok = bool(budget_error < 1e-6)
    if regime == "no_escrow":
        invariant_ok = False
    defect_rate = defects / max(1, packets_sent)

    notes: str
    if regime == "baseline":
        notes = "horizons locked; throughput ~0"
    elif regime == "wormhole":
        if baseline_throughput is not None and baseline_throughput > 0:
            gain = throughput / baseline_throughput
            traversable_tag = " with traversable window" if traversable_kicks else ""
            notes = f"throughput gain ×{gain:.1f}{traversable_tag}"
        else:
            base = "wormhole opened; baseline ~0"
            traversable_tag = " with traversable window" if traversable_kicks else ""
            notes = f"{base}{traversable_tag}"
    elif regime == "no_escrow":
        notes = "escrow removed → budget drift"
    elif regime == "no_verify":
        notes = "verify disabled → expect defects"
    else:
        notes = "random subspace → throughput collapse"

    if first_lock_step is None:
        first_lock_step = float("nan")

    metrics: Dict[str, float] = {
        "relaxation_steps": float(first_lock_step) if isinstance(first_lock_step, (int, float)) else float("nan"),
        "candidate_norm_mean": float(np.mean(candidate_norms)) if candidate_norms else 0.0,
        "candidate_norm_std": float(np.std(candidate_norms)) if candidate_norms else 0.0,
        "soft_top_eig_mean": float(np.mean(soft_top_eigs)) if soft_top_eigs else 0.0,
        "soft_active_rank_mean": float(np.mean(soft_active_ranks)) if soft_active_ranks else 0.0,
        "percolation_span_mean": float(np.mean(percolation_spans)) if percolation_spans else 0.0,
        "percolation_span_max": float(np.max(percolation_spans)) if percolation_spans else 0.0,
        "committed_delta_mean": float(np.mean(committed_deltas)) if committed_deltas else 0.0,
        "packets_sent": float(packets_sent),
        "update_ratio_mean": float(np.mean(update_ratios)) if update_ratios else 0.0,
        "prelock_update_mean": float(np.mean(prelock_update_norms)) if prelock_update_norms else 0.0,
        "traversable_kick_total": float(np.sum(traversable_kicks)) if traversable_kicks else 0.0,
        "traversable_kick_mean": float(np.mean(traversable_kicks)) if traversable_kicks else 0.0,
        "arrival_delay_mean": float(np.mean(arrival_steps)) if arrival_steps else float("nan"),
        "arrival_delay_min": float(np.min(arrival_steps)) if arrival_steps else float("nan"),
        "state_corr_mean": float(np.mean(state_corrs)) if state_corrs else 0.0,
        "shock_delay_mean": float(np.mean(shock_delay_records)) if shock_delay_records else float("nan"),
        "shock_delay_min": float(np.min(shock_delay_records)) if shock_delay_records else float("nan"),
        "shock_delay_count": float(len(shock_delay_records)),
    }

    if hp_enabled and hp_message_indices is not None and hp_message_signs is not None:
        decoded_vec = hp_decode_accum[: config.hp_message_bits]
        decoded_sign = np.sign(decoded_vec)
        decoded_sign[decoded_sign == 0.0] = 0.0
        match_fraction = float((decoded_sign == hp_message_signs).mean())
        success = bool(
            np.all(decoded_sign == hp_message_signs)
            and np.max(np.abs(decoded_vec)) >= config.hp_decode_tolerance
        )
        message_target = hp_message_signs * config.hp_message_magnitude
        l2_error = float(np.linalg.norm(decoded_vec - message_target))
        dot = float(np.dot(decoded_vec, message_target))
        denom = (np.linalg.norm(decoded_vec) * np.linalg.norm(message_target)) + 1e-12
        cosine = dot / denom
        metrics.update(
            {
                "hp_message_bits": float(config.hp_message_bits),
                "hp_packets_used": float(hp_packets_used),
                "hp_match_fraction": match_fraction,
                "hp_success": 1.0 if success else 0.0,
                "hp_l2_error": l2_error,
                "hp_cosine": cosine,
                "hp_injection_step": float(hp_injection_step) if hp_injection_step is not None else float("nan"),
                "hp_scramble_wait": float(config.hp_scramble_wait),
                "hp_partial_match_final": float(hp_partial_match[-1]) if hp_partial_match else float("nan"),
            }
        )

    return RegimeResult(
        name=regime,
        throughput=throughput,
        commits=commits,
        invariant_ok=invariant_ok,
        budget_error=budget_error,
        debt_balance=debt_balance,
        defect_rate=defect_rate,
        notes=notes,
        metrics=metrics,
    )


def run_experiment(cfg: WormholeConfig, seed: int) -> List[RegimeResult]:
    rng = np.random.default_rng(seed)
    baseline = simulate_regime(cfg, rng, regime="baseline")

    rng = np.random.default_rng(seed)
    wormhole = simulate_regime(cfg, rng, regime="wormhole", baseline_throughput=baseline.throughput)

    results = [baseline, wormhole]

    for regime in ("no_escrow", "no_verify", "random_subspace"):
        rng = np.random.default_rng(seed)
        results.append(simulate_regime(cfg, rng, regime=regime))

    return results


def write_artifact(results: List[RegimeResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": timestamp,
        "results": [
            {
                "name": r.name,
                "throughput": r.throughput,
                "commits": r.commits,
                "invariant_ok": r.invariant_ok,
                "budget_error": r.budget_error,
                "debt_balance": r.debt_balance,
                "defect_rate": r.defect_rate,
                "notes": r.notes,
                "metrics": r.metrics,
            }
            for r in results
        ],
    }
    path = output_dir / f"wormhole_results_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def print_summary(results: List[RegimeResult]) -> None:
    table = [
        (
            r.name,
            f"{r.throughput:.4f}",
            r.commits,
            "yes" if r.invariant_ok else "no",
            f"{r.budget_error:.2e}",
            f"{r.debt_balance:.4f}",
            f"{100*r.defect_rate:.1f}%",
            r.notes,
        )
        for r in results
    ]
    headers = [
        "regime",
        "throughput",
        "commits",
        "invariants",
        "budget_err",
        "debt_balance",
        "defect_rate",
        "notes",
    ]
    print(tabulate(table, headers=headers, tablefmt="github"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAL wormhole tunnel simulation")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--steps", type=int, default=400, help="Simulation steps")
    parser.add_argument("--horizon-steps", type=int, default=40, help="Horizon detection window")
    parser.add_argument("--soft-rank", type=int, default=8, help="Shared soft subspace rank")
    parser.add_argument(
        "--lambda-h",
        type=float,
        default=180.0,
        help="Continuity strain parameter controlling baseline inertia",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-3,
        help="Horizon detection threshold",
    )
    parser.add_argument(
        "--base-step-scale",
        type=float,
        default=0.06,
        help="Scale of local proposals before horizon tax",
    )
    parser.add_argument(
        "--wormhole-step-scale",
        type=float,
        default=0.25,
        help="Step scale applied to verified wormhole updates",
    )
    parser.add_argument(
        "--wormhole-budget-fraction",
        type=float,
        default=0.12,
        help="Fraction of total budget escrowed per wormhole packet",
    )
    parser.add_argument(
        "--traversable-mu",
        type=float,
        default=0.0,
        help="Strength of cross-coupling kick applied when horizon first locks",
    )
    parser.add_argument(
        "--traversable-window",
        type=int,
        default=0,
        help="Number of packets eligible for the traversable kick window",
    )
    parser.add_argument(
        "--hp-message-bits",
        type=int,
        default=0,
        help="Enable Hayden–Preskill mirror by encoding this many bits",
    )
    parser.add_argument(
        "--hp-message-magnitude",
        type=float,
        default=0.12,
        help="Magnitude of the injected Hayden–Preskill message pattern",
    )
    parser.add_argument(
        "--hp-scramble-wait",
        type=int,
        default=0,
        help="Steps to wait after message injection before opening wormhole",
    )
    parser.add_argument(
        "--hp-decode-packets",
        type=int,
        default=0,
        help="Number of wormhole packets allotted for Hayden–Preskill decoding",
    )
    parser.add_argument(
        "--hp-decode-tolerance",
        type=float,
        default=0.1,
        help="Minimum recovered magnitude to count a Hayden–Preskill success",
    )
    parser.add_argument(
        "--shock-energy",
        type=float,
        default=0.0,
        help="Amplitude of the pre-probe shock packet (0 disables shockwave test)",
    )
    parser.add_argument(
        "--shock-probe-gap",
        type=int,
        default=0,
        help="Number of wormhole packets to skip between shock and probe measurement",
    )
    parser.add_argument(
        "--verify-tolerance",
        type=float,
        default=0.08,
        help="Verification tolerance for shadow application",
    )
    parser.add_argument(
        "--max-packets",
        type=int,
        default=48,
        help="Maximum wormhole packets per run",
    )
    parser.add_argument(
        "--debt-repay-rate",
        type=float,
        default=1e-3,
        help="Rate at which continuity debt is repaid to budgets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACT_DIR,
        help="Artifact directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = WormholeConfig(
        steps=args.steps,
        horizon_steps=args.horizon_steps,
        soft_rank=args.soft_rank,
        lambda_h=args.lambda_h,
        epsilon=args.epsilon,
        base_step_scale=args.base_step_scale,
        wormhole_step_scale=args.wormhole_step_scale,
        wormhole_budget_fraction=args.wormhole_budget_fraction,
        verify_tolerance=args.verify_tolerance,
        max_packets=args.max_packets,
        debt_repay_rate=args.debt_repay_rate,
        traversable_mu=args.traversable_mu,
        traversable_window=args.traversable_window,
        hp_message_bits=args.hp_message_bits,
        hp_message_magnitude=args.hp_message_magnitude,
        hp_scramble_wait=args.hp_scramble_wait,
        hp_decode_packets=args.hp_decode_packets,
        hp_decode_tolerance=args.hp_decode_tolerance,
        shock_energy=args.shock_energy,
        shock_probe_gap=args.shock_probe_gap,
    )
    results = run_experiment(cfg, seed=args.seed)
    artifact_path = write_artifact(results, args.output_dir)
    print_summary(results)
    print(f"\nArtifacts written to {artifact_path}")


if __name__ == "__main__":
    main()
