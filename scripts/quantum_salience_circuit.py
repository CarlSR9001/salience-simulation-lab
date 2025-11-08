"""Prototype quantum circuit for salience invariant emulation."""

from __future__ import annotations

import numpy as np
from tabulate import tabulate
from qutip import Qobj, tensor, basis, qeye

np.pi

GATES = {
    "h": (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "r": lambda theta: np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex),
}

W1, W2, W3 = 0.5, 0.3, 0.2
LAMBDA = 0.5
K_COEFF = 0.7


def apply_gate(state: Qobj, gate: np.ndarray, qubit: int, n_qubits: int) -> Qobj:
    ops = [qeye(2)] * n_qubits
    ops[qubit] = Qobj(gate)
    return tensor(ops) * state


def apply_cnot(state: Qobj, control: int, target: int, n_qubits: int) -> Qobj:
    zero = basis(2, 0)
    one = basis(2, 1)
    proj0 = zero * zero.dag()
    proj1 = one * one.dag()

    ops0 = [qeye(2)] * n_qubits
    ops1 = [qeye(2)] * n_qubits
    ops0[control] = Qobj(proj0)
    ops1[control] = Qobj(proj1)

    ident_target = Qobj(qeye(2))
    pauli_x = Qobj(GATES["x"])
    ops1[target] = pauli_x

    term0 = tensor(ops0) * state
    term1 = tensor(ops1) * state
    return term0 + term1


def salience_scores(rho: Qobj, reference: Qobj, phase_param: float, decay_param: float) -> tuple[float, float, float]:
    diff = rho - reference
    delta = np.linalg.norm(diff.full(), "fro") / (np.linalg.norm(reference.full(), "fro") + 1.0e-12)
    overlap = float((rho * reference).tr().real)
    payoff = float(np.clip(rho.tr().real, 0.0, 1.0))
    delta_a = delta
    retention = max(overlap, 0.0)
    payoff_factor = min(payoff, 1.0)
    hype = W1 * delta_a + W2 * retention + W3 * payoff_factor
    continuity = max(0.0, np.cos(phase_param))
    fatigue = min(1.0, decay_param)
    s_prime = hype * continuity * np.exp(-LAMBDA * decay_param) * (1.0 - K_COEFF * fatigue)
    return float(delta_a), float(retention), float(s_prime)


def run_circuit(phase_param: float, decay_param: float) -> dict[str, float]:
    n_qubits = 2
    ket = tensor(basis(2, 0), basis(2, 0))

    ket = apply_gate(ket, GATES["h"], 0, n_qubits)
    ket = apply_cnot(ket, 0, 1, n_qubits)

    ket = apply_gate(ket, GATES["r"](phase_param), 0, n_qubits)
    ket = apply_gate(ket, GATES["z"], 1, n_qubits)
    ket = apply_cnot(ket, 1, 0, n_qubits)

    ket = ket.unit()
    ket_x = apply_gate(ket, GATES["x"], 0, n_qubits).unit()
    rho = (1.0 - decay_param) * (ket * ket.dag()) + decay_param * (ket_x * ket_x.dag())

    reference = tensor(basis(2, 0), basis(2, 0)).proj()
    delta_a, retention, s_prime = salience_scores(rho, reference, phase_param, decay_param)

    energy = float(decay_param * phase_param**2)
    sigma_tilde = energy / max(s_prime, 1.0e-12)

    return {
        "phase": phase_param,
        "decay": decay_param,
        "delta_a": delta_a,
        "retention": retention,
        "s_prime": s_prime,
        "energy": energy,
        "sigma_tilde": sigma_tilde,
    }


def main() -> None:
    phases = np.linspace(0.0, np.pi, 6)
    decays = np.linspace(0.0, 0.5, 6)

    rows = []
    for phase in phases:
        for decay in decays:
            result = run_circuit(float(phase), float(decay))
            rows.append((
                result["phase"],
                result["decay"],
                result["delta_a"],
                result["retention"],
                result["s_prime"],
                result["energy"],
                result["sigma_tilde"],
            ))

    print(tabulate(rows, headers=[
        "phase",
        "decay",
        "ΔA",
        "Retention",
        "S'",
        "Energy",
        "σ̃",
    ], tablefmt="github", floatfmt=".6g"))


if __name__ == "__main__":
    main()
