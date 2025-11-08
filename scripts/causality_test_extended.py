"""Experiment D: Extended causality probe with basis/time sweeps."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import qutip as qt

ARTIFACT_DIR = Path("artifacts/causality_probe")

HADAMARD = qt.Qobj((1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex))
SIGMA_X = qt.sigmax()
SIGMA_Z = qt.sigmaz()
SIGMA_Y = qt.sigmay()
Q_ID = qt.qeye(2)

BASES = {
    "Z": (
        qt.tensor(Q_ID, qt.basis(2, 0) * qt.basis(2, 0).dag()),
        qt.tensor(Q_ID, qt.basis(2, 1) * qt.basis(2, 1).dag()),
    ),
    "X": (
        qt.tensor(Q_ID, qt.basis(2, 0) + qt.basis(2, 1))
        * (qt.tensor(Q_ID, qt.basis(2, 0) + qt.basis(2, 1)).dag())
        / 2.0,
        qt.tensor(Q_ID, qt.basis(2, 0) - qt.basis(2, 1))
        * (qt.tensor(Q_ID, qt.basis(2, 0) - qt.basis(2, 1)).dag())
        / 2.0,
    ),
    "Y": (
        qt.tensor(Q_ID, qt.basis(2, 0) + 1j * qt.basis(2, 1))
        * (qt.tensor(Q_ID, qt.basis(2, 0) + 1j * qt.basis(2, 1)).dag())
        / 2.0,
        qt.tensor(Q_ID, qt.basis(2, 0) - 1j * qt.basis(2, 1))
        * (qt.tensor(Q_ID, qt.basis(2, 0) - 1j * qt.basis(2, 1)).dag())
        / 2.0,
    ),
}

TIME_OFFSETS = {
    "t0": 0.0,
    "t1": 0.5,
    "t2": 1.0,
}


@dataclass
class ProbeConfig:
    trials_per_bin: int = 400
    shots_per_trial: int = 500
    slam_gamma: float = 0.8
    quiet_gamma: float = 0.0
    decoherence_tau: float = 2.0
    seed: int = 2025


@dataclass
class TrialResult:
    action: int
    ones: int
    zeros: int


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


def amplitude_damp(rho: qt.Qobj, gamma: float, target: int) -> qt.Qobj:
    gamma = float(np.clip(gamma, 0.0, 1.0))
    e0 = qt.Qobj([[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma)]])
    e1 = qt.Qobj([[0.0, math.sqrt(gamma)], [0.0, 0.0]])
    ops0 = [Q_ID, Q_ID]
    ops1 = [Q_ID, Q_ID]
    ops0[target] = e0
    ops1[target] = e1
    k0 = qt.tensor(ops0)
    k1 = qt.tensor(ops1)
    return k0 * rho * k0.dag() + k1 * rho * k1.dag()


def decoherence_channel(rho: qt.Qobj, time_offset: float, tau: float) -> qt.Qobj:
    if time_offset <= 0:
        return rho
    gamma = 1.0 - math.exp(-time_offset / max(tau, 1e-9))
    rho = amplitude_damp(rho, gamma, target=1)
    return rho


class ExtendedCausalityProbe:
    def __init__(self, config: ProbeConfig) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

    def _prepare_state(self, action: int) -> qt.Qobj:
        rho = qt.ket2dm(bell_state())
        gamma = self.cfg.slam_gamma if action == 1 else self.cfg.quiet_gamma
        rho = amplitude_damp(rho, gamma, target=0)
        return rho

    def _measure(self, rho: qt.Qobj, basis: str, shots: int) -> TrialResult:
        projectors = BASES[basis]
        prob_zero = float(np.clip((projectors[0] * rho).tr().real, 0.0, 1.0))
        prob_one = float(np.clip((projectors[1] * rho).tr().real, 0.0, 1.0))
        probs = np.array([prob_zero, prob_one])
        probs = np.maximum(probs, 0.0)
        probs = probs / probs.sum() if probs.sum() else np.array([1.0, 0.0])
        samples = self.rng.choice([0, 1], size=shots, p=probs)
        ones = int(np.sum(samples))
        zeros = int(shots - ones)
        return TrialResult(action=0, ones=ones, zeros=zeros)

    def run_bin(self, basis: str, offset_label: str, offset_value: float) -> Dict[str, float]:
        slam_trials: List[TrialResult] = []
        quiet_trials: List[TrialResult] = []

        for _ in range(self.cfg.trials_per_bin):
            action = int(self.rng.integers(0, 2))
            rho = self._prepare_state(action)
            rho = decoherence_channel(rho, offset_value, self.cfg.decoherence_tau)
            result = self._measure(rho, basis, self.cfg.shots_per_trial)
            result.action = action
            if action == 1:
                slam_trials.append(result)
            else:
                quiet_trials.append(result)

        return compute_statistics(
            basis=basis,
            offset_label=offset_label,
            offset_value=offset_value,
            slam_trials=slam_trials,
            quiet_trials=quiet_trials,
            shots=self.cfg.shots_per_trial,
        )

    def run(self) -> List[Dict[str, float]]:
        records: List[Dict[str, float]] = []
        for basis in BASES.keys():
            for offset_label, offset_value in TIME_OFFSETS.items():
                stats = self.run_bin(basis, offset_label, offset_value)
                records.append(stats)
        return records


def compute_statistics(
    basis: str,
    offset_label: str,
    offset_value: float,
    slam_trials: List[TrialResult],
    quiet_trials: List[TrialResult],
    shots: int,
) -> Dict[str, float]:
    slam_ones = sum(t.ones for t in slam_trials)
    slam_zeros = sum(t.zeros for t in slam_trials)
    quiet_ones = sum(t.ones for t in quiet_trials)
    quiet_zeros = sum(t.zeros for t in quiet_trials)

    total_slam = slam_ones + slam_zeros
    total_quiet = quiet_ones + quiet_zeros

    p_slam = slam_ones / total_slam if total_slam else 0.0
    p_quiet = quiet_ones / total_quiet if total_quiet else 0.0

    pooled = (slam_ones + quiet_ones) / max(total_slam + total_quiet, 1)
    denom = pooled * (1 - pooled) * (1 / max(total_slam, 1) + 1 / max(total_quiet, 1))
    z_score = (p_slam - p_quiet) / math.sqrt(denom) if denom > 0 else 0.0
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / math.sqrt(2))))

    chi2 = 0.0
    for observed, expected_prop, total in (
        (slam_ones, pooled, total_slam),
        (slam_zeros, 1 - pooled, total_slam),
        (quiet_ones, pooled, total_quiet),
        (quiet_zeros, 1 - pooled, total_quiet),
    ):
        expected = expected_prop * total
        if expected > 0:
            chi2 += (observed - expected) ** 2 / expected

    bonferroni_sig = p_value < 0.001

    return {
        "basis": basis,
        "time_offset_label": offset_label,
        "time_offset": offset_value,
        "trials_per_bin": len(slam_trials) + len(quiet_trials),
        "shots_per_trial": shots,
        "slam_trials": len(slam_trials),
        "quiet_trials": len(quiet_trials),
        "total_slam_counts": total_slam,
        "total_quiet_counts": total_quiet,
        "p_slam": p_slam,
        "p_quiet": p_quiet,
        "z_score": z_score,
        "p_value": p_value,
        "chi2": chi2,
        "bonferroni_significant": bonferroni_sig,
    }


def write_artifact(records: List[Dict[str, float]], config: ProbeConfig) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())
    payload = []
    for record in records:
        payload.append(
            {
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "experiment_name": "experiment_d_extended_probe",
                "run_id": run_id,
                "config": {
                    "trials_per_bin": config.trials_per_bin,
                    "shots_per_trial": config.shots_per_trial,
                    "slam_gamma": config.slam_gamma,
                    "quiet_gamma": config.quiet_gamma,
                    "decoherence_tau": config.decoherence_tau,
                },
                **record,
            }
        )
    path = ARTIFACT_DIR / f"causality_probe_extended_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def main() -> None:
    config = ProbeConfig()
    probe = ExtendedCausalityProbe(config)
    records = probe.run()
    path = write_artifact(records, config)
    print(f"Extended probe results saved to {path}")
    hits = [r for r in records if r["bonferroni_significant"]]
    if hits:
        print("Potential signaling detected in bins:")
        for hit in hits:
            print(
                f"  basis={hit['basis']} offset={hit['time_offset_label']} z={hit['z_score']:.3f} p={hit['p_value']:.6f}"
            )
    else:
        print("All bins within null hypothesis (no significant signaling).")


if __name__ == "__main__":
    main()
