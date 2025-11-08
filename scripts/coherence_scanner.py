"""Continuity coherence scanner prototype.

Sweeps pseudo-qubit parameters, performs simulated measurements,
and logs anomalies vs classical references.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable

import json
import math
import os
import random
import statistics
import sys
import time

import numpy as np
from tabulate import tabulate
import qutip as qt

LOG_DIR = Path("artifacts/coherence_scanner")
LOG_DIR.mkdir(parents=True, exist_ok=True)

NB_MEASUREMENTS = 10_000
PHASE_GRID = np.linspace(0.0, math.pi, 9)
DAMPING_GRID = np.linspace(0.0, 0.3, 7)
HADAMARD = qt.Qobj((1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex))


def apply_single(ket: qt.Qobj, gate: qt.Qobj, target: int) -> qt.Qobj:
    ops = [qt.qeye(2), qt.qeye(2)]
    ops[target] = gate
    return qt.tensor(ops) * ket


def apply_cnot(ket: qt.Qobj, control: int, target: int) -> qt.Qobj:
    zero = qt.basis(2, 0)
    one = qt.basis(2, 1)
    proj0 = zero * zero.dag()
    proj1 = one * one.dag()

    ops0 = [qt.qeye(2), qt.qeye(2)]
    ops1 = [qt.qeye(2), qt.qeye(2)]
    ops0[control] = proj0
    ops1[control] = proj1
    ops1[target] = qt.sigmax()

    operator = qt.tensor(ops0) + qt.tensor(ops1)
    return operator * ket
BASES = [
    qt.tensor(qt.sigmax(), qt.sigmax()),
    qt.tensor(qt.sigmay(), qt.sigmay()),
    qt.tensor(qt.sigmaz(), qt.sigmaz()),
]


@dataclass
class ScannerConfig:
    measurements: int = NB_MEASUREMENTS
    phase_values: Iterable[float] = tuple(PHASE_GRID)
    damping_values: Iterable[float] = tuple(DAMPING_GRID)


@dataclass
class ScannerResult:
    phase: float
    damping: float
    basis_index: int
    correlator: float
    p_value: float
    anomaly_score: float


class CoherenceScanner:
    def __init__(self, config: ScannerConfig | None = None) -> None:
        self.config = config or ScannerConfig()

    def _prepare_state(self, phase: float, damping: float) -> qt.Qobj:
        ket = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
        ket = apply_single(ket, HADAMARD, 0)
        ket = apply_cnot(ket, 0, 1)
        rho = qt.ket2dm(ket)
        if damping > 0:
            e0 = qt.tensor(qt.qeye(2), qt.qeye(2))
            e1 = qt.tensor(qt.sigmax(), qt.qeye(2))
            rho = (1 - damping) * rho + damping * e1 * rho * e1.dag()
        phase_op = qt.tensor(qt.Qobj([[1, 0], [0, np.exp(1j * phase)]]), qt.qeye(2))
        rho = phase_op * rho * phase_op.dag()
        return rho

    def _simulate_measurements(self, rho: qt.Qobj, basis: qt.Qobj, measurements: int) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(basis.full())
        probs = []
        for vec in eigenvectors.T:
            projector = qt.Qobj(np.outer(vec, vec.conjugate()), dims=[[2, 2], [2, 2]])
            probs.append(float((projector * rho).tr().real))
        probs = np.clip(probs, 0.0, 1.0)
        probs /= np.sum(probs)
        outcomes = np.random.choice(len(probs), size=measurements, p=probs)
        return eigenvalues[outcomes]

    def _correlator(self, samples: np.ndarray) -> float:
        return float(np.mean(samples))

    def _classical_reference(self, size: int, seed: int) -> np.ndarray:
        rng = random.Random(seed)
        return np.array([rng.choice([-1.0, 1.0]) for _ in range(size)], dtype=float)

    def _anomaly_score(self, quantum_corr: float, classical_corr: float, std_dev: float) -> float:
        if std_dev <= 1e-12:
            return float('inf') if quantum_corr != classical_corr else 0.0
        return abs(quantum_corr - classical_corr) / std_dev

    def scan(self) -> Dict[str, object]:
        results: list[ScannerResult] = []
        metadata: Dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "hostname": os.uname().nodename if hasattr(os, "uname") else os.getenv("COMPUTERNAME", "unknown"),
            "entropy_seed": random.getrandbits(64),
            "temperature_placeholder": None,
            "cpu_load": os.getloadavg() if hasattr(os, "getloadavg") else None,
        }

        measurements = self.config.measurements

        for phase in self.config.phase_values:
            for damping in self.config.damping_values:
                rho = self._prepare_state(phase, damping)
                for idx, basis in enumerate(BASES):
                    samples = self._simulate_measurements(rho, basis, measurements)
                    quantum_corr = self._correlator(samples)

                    classical_samples = self._classical_reference(measurements, seed=int(phase * 1e6) ^ idx)
                    classical_corr = self._correlator(classical_samples)

                    std_dev = np.std(classical_samples) / math.sqrt(measurements)
                    anomaly = self._anomaly_score(quantum_corr, classical_corr, std_dev)

                    results.append(ScannerResult(
                        phase=phase,
                        damping=damping,
                        basis_index=idx,
                        correlator=quantum_corr,
                        p_value=0.0,
                        anomaly_score=anomaly,
                    ))

        return {
            "results": results,
            "metadata": metadata,
        }


def write_report(data: Dict[str, object]) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = LOG_DIR / f"coherence_scan_{timestamp}.json"
    serializable = {
        "metadata": data["metadata"],
        "results": [vars(res) for res in data["results"]],
    }
    report_path.write_text(json.dumps(serializable, indent=2))
    return report_path


def print_summary(results: Iterable[ScannerResult]) -> None:
    rows = []
    for res in results:
        rows.append((
            res.phase,
            res.damping,
            res.basis_index,
            res.correlator,
            res.anomaly_score,
        ))
    print(tabulate(rows[:20], headers=["phase", "damping", "basis", "corr", "anomaly"], tablefmt="github", floatfmt=".6g"))
    if len(rows) > 20:
        print(f"... truncated {len(rows) - 20} rows ...")


def main() -> None:
    scanner = CoherenceScanner()
    data = scanner.scan()
    report_path = write_report(data)
    print(f"Report saved to {report_path}")
    print_summary(data["results"])


if __name__ == "__main__":
    main()
