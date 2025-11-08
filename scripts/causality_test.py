"""Causality violation probe between entangled subsystems A (transmitter) and B (receiver).

Protocol
--------
1. Prepare a Bell state |Φ+> across qubits A and B.
2. For each trial:
   a. Flip a fair coin.
   b. If heads, apply a high-amplitude-damping channel to qubit A ("slam").
      If tails, leave qubit A untouched ("quiet").
   c. Immediately sample qubit B in the fixed Z basis for `shots_per_trial` measurements.
3. Record the measurement outcomes alongside the transmitter action.
4. After `num_trials`, compare B's outcome distributions conditioned on the transmitter action
   using a z-test for difference in proportions and a chi-square statistic on aggregated counts.

If B's distribution depends on whether A was slammed (statistically significant in this model),
that would indicate a signaling channel in the simulator. Otherwise, no broadcast layer is detected.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List

import json
import math
import os
import random

import numpy as np
import qutip as qt

LOG_DIR = Path("artifacts/coherence_scanner")
LOG_DIR.mkdir(parents=True, exist_ok=True)

HADAMARD = qt.Qobj((1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex))
SIGMA_X = qt.sigmax()
Q_ID = qt.qeye(2)


def apply_single(ket: qt.Qobj, gate: qt.Qobj, target: int) -> qt.Qobj:
    ops = [Q_ID, Q_ID]
    ops[target] = gate
    return qt.tensor(ops) * ket


def apply_cnot(ket: qt.Qobj, control: int, target: int) -> qt.Qobj:
    zero = qt.basis(2, 0)
    one = qt.basis(2, 1)
    proj0 = zero * zero.dag()
    proj1 = one * one.dag()

    ops0 = [Q_ID, Q_ID]
    ops1 = [Q_ID, Q_ID]
    ops0[control] = proj0
    ops1[control] = proj1
    ops1[target] = SIGMA_X

    operator = qt.tensor(ops0) + qt.tensor(ops1)
    return operator * ket


def bell_state() -> qt.Qobj:
    ket = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
    ket = apply_single(ket, HADAMARD, 0)
    ket = apply_cnot(ket, 0, 1)
    return ket.unit()


def amplitude_damp_on_a(rho: qt.Qobj, gamma: float) -> qt.Qobj:
    gamma = float(np.clip(gamma, 0.0, 1.0))
    e0 = qt.Qobj([[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma)]])
    e1 = qt.Qobj([[0.0, math.sqrt(gamma)], [0.0, 0.0]])
    k0 = qt.tensor(e0, Q_ID)
    k1 = qt.tensor(e1, Q_ID)
    return k0 * rho * k0.dag() + k1 * rho * k1.dag()


@dataclass
class TrialConfig:
    num_trials: int = 200
    shots_per_trial: int = 500
    slam_gamma: float = 0.8
    quiet_gamma: float = 0.0
    seed: int | None = None


@dataclass
class TrialResult:
    action: int  # 0 = quiet, 1 = slam
    ones_count: int
    zeros_count: int


class CausalityProbe:
    def __init__(self, config: TrialConfig | None = None) -> None:
        self.config = config or TrialConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def _prepare_conditioned_state(self, action: int) -> qt.Qobj:
        rho = qt.ket2dm(bell_state())
        gamma = self.config.slam_gamma if action == 1 else self.config.quiet_gamma
        rho = amplitude_damp_on_a(rho, gamma)
        return rho

    def _measure_b_z(self, rho: qt.Qobj, shots: int) -> TrialResult:
        # Projectors on B (second qubit)
        proj0 = qt.tensor(Q_ID, qt.basis(2, 0) * qt.basis(2, 0).dag())
        prob_zero = float((proj0 * rho).tr().real)
        prob_zero = np.clip(prob_zero, 0.0, 1.0)
        prob_one = 1.0 - prob_zero
        counts = self.rng.choice([0, 1], size=shots, p=[prob_zero, prob_one])
        zeros = int(shots - counts.sum())
        ones = int(counts.sum())
        return TrialResult(action=0, ones_count=ones, zeros_count=zeros)

    def run(self) -> Dict[str, object]:
        trials: List[TrialResult] = []
        actions = self.rng.integers(0, 2, size=self.config.num_trials)
        for action in actions:
            rho = self._prepare_conditioned_state(int(action))
            result = self._measure_b_z(rho, self.config.shots_per_trial)
            result.action = int(action)
            trials.append(result)

        summary = analyze_trials(trials)
        metadata = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "hostname": os.uname().nodename if hasattr(os, "uname") else os.getenv("COMPUTERNAME", "unknown"),
            "config": {
                "num_trials": self.config.num_trials,
                "shots_per_trial": self.config.shots_per_trial,
                "slam_gamma": self.config.slam_gamma,
                "quiet_gamma": self.config.quiet_gamma,
                "seed": self.config.seed,
            },
        }

        return {
            "trials": [result.__dict__ for result in trials],
            "summary": summary,
            "metadata": metadata,
        }


def analyze_trials(trials: List[TrialResult]) -> Dict[str, float]:
    slam_ones = slam_zeros = quiet_ones = quiet_zeros = 0
    for result in trials:
        if result.action == 1:
            slam_ones += result.ones_count
            slam_zeros += result.zeros_count
        else:
            quiet_ones += result.ones_count
            quiet_zeros += result.zeros_count

    total_slam = slam_ones + slam_zeros
    total_quiet = quiet_ones + quiet_zeros

    p_slam = slam_ones / total_slam if total_slam else 0.0
    p_quiet = quiet_ones / total_quiet if total_quiet else 0.0

    pooled = (slam_ones + quiet_ones) / (total_slam + total_quiet)
    denom = pooled * (1 - pooled) * (1 / total_slam + 1 / total_quiet)
    z_score = (p_slam - p_quiet) / math.sqrt(denom) if denom > 0 else float("inf")
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / math.sqrt(2))))

    # Chi-square on 2x2 contingency table
    chi2 = 0.0
    for observed, expected_prop in (
        (slam_ones, pooled),
        (slam_zeros, 1 - pooled),
        (quiet_ones, pooled),
        (quiet_zeros, 1 - pooled),
    ):
        expected = expected_prop * (total_slam if observed in (slam_ones, slam_zeros) else total_quiet)
        if expected > 0:
            chi2 += (observed - expected) ** 2 / expected

    return {
        "p_slam": p_slam,
        "p_quiet": p_quiet,
        "z_score": z_score,
        "p_value": p_value,
        "chi2": chi2,
        "total_slam_counts": total_slam,
        "total_quiet_counts": total_quiet,
    }


def write_report(data: Dict[str, object]) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"causality_probe_{timestamp}.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def main() -> None:
    config = TrialConfig()
    probe = CausalityProbe(config)
    data = probe.run()
    path = write_report(data)
    summary = data["summary"]

    print(f"Report saved to {path}")
    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
