"""Quantum salience invariant on a three-qubit register.

Qubit layout
------------
q0 -> ΔA (novelty amplitude)
q1 -> R (retention / coherence)
q2 -> M (payoff / entropy carrier)

The circuit prepares an entangled state where controlled rotations propagate
ΔA into R and M. Controlled noise channels (bit-flip on q0, phase-flip on q1)
model continuity strain and fatigue. Salience S' is reconstructed from the
resulting density matrix and compared against an effective energy budget to
extract σ̃ and σ_info.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from tabulate import tabulate
import qutip as qt

# Salience weights / parameters
W1, W2, W3 = 0.5, 0.3, 0.2
LAMBDA = 0.4
K_COEFF = 0.6

HBAR = 1.054_571_817e-34  # J*s
OMEGA = 2.0 * np.pi * 5.0e9  # 5 GHz oscillator frequency
VOLUME = (20e-9) ** 3  # (20 nm)^3 effective volume

GATE_X = qt.sigmax()
GATE_Z = qt.sigmaz()


def gate_ry(theta: float) -> qt.Qobj:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    return qt.Qobj([[cos, -sin], [sin, cos]])


def apply_single(state: qt.Qobj, gate: qt.Qobj, target: int, n_qubits: int) -> qt.Qobj:
    ops = [qt.qeye(2)] * n_qubits
    ops[target] = gate
    return qt.tensor(ops) * state


def apply_cnot(state: qt.Qobj, control: int, target: int, n_qubits: int) -> qt.Qobj:
    zero = qt.basis(2, 0)
    one = qt.basis(2, 1)
    proj0 = zero * zero.dag()
    proj1 = one * one.dag()

    ops0 = [qt.qeye(2)] * n_qubits
    ops1 = [qt.qeye(2)] * n_qubits
    ops0[control] = proj0
    ops1[control] = proj1
    ops1[target] = GATE_X

    return qt.tensor(ops0) * state + qt.tensor(ops1) * state


def projector_on_state(bits: Iterable[int], n_qubits: int) -> qt.Qobj:
    bra = qt.ket2dm(qt.tensor(*[qt.basis(2, b) for b in bits]))
    return bra


@dataclass
class SalienceResult:
    phase_a: float
    phase_r: float
    phase_m: float
    gamma_amp: float
    gamma_phase: float
    delta_a: float
    retention: float
    payoff: float
    salience: float
    energy_density: float
    sigma_tilde: float
    sigma_info: float


def build_state(phase_a: float, phase_r: float, phase_m: float, *, noise_amp: float, noise_phase: float) -> qt.Qobj:
    n_qubits = 3
    ket = qt.tensor(qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0))

    ket = apply_single(ket, gate_ry(phase_a), 0, n_qubits)
    ket = apply_cnot(ket, 0, 1, n_qubits)
    ket = apply_single(ket, gate_ry(phase_r), 1, n_qubits)
    ket = apply_cnot(ket, 1, 2, n_qubits)
    ket = apply_single(ket, gate_ry(phase_m), 2, n_qubits)
    ket = apply_cnot(ket, 0, 2, n_qubits)

    ket = ket.unit()
    rho = ket * ket.dag()

    if noise_amp > 0:
        flipped = apply_single(ket, GATE_X, 0, n_qubits)
        rho = (1 - noise_amp) * rho + noise_amp * (flipped * flipped.dag())

    if noise_phase > 0:
        phased = apply_single(ket, GATE_Z, 1, n_qubits)
        rho = (1 - noise_phase) * rho + noise_phase * (phased * phased.dag())

    return rho


def compute_metrics(rho: qt.Qobj) -> tuple[float, float, float]:
    # ΔA from occupation of |1> on qubit 0
    proj1 = qt.tensor(qt.basis(2, 1) * qt.basis(2, 1).dag(), qt.qeye(2), qt.qeye(2))
    delta_a = float((proj1 * rho).tr().real)

    # Retention via correlation (probability q0 == q1)
    proj00 = projector_on_state([0, 0, 0], 3) + projector_on_state([0, 0, 1], 3)
    proj11 = projector_on_state([1, 1, 0], 3) + projector_on_state([1, 1, 1], 3)
    retention = float(((proj00 + proj11) * rho).tr().real)

    # Payoff from entropy of q2 subsystem
    rho_q2 = qt.ptrace(rho, 2)
    entropy = qt.entropy_vn(rho_q2, 2)
    payoff = float(entropy / np.log(2.0))

    return delta_a, retention, payoff


def compute_salience(delta_a: float, retention: float, payoff: float, *, gamma_amp: float, gamma_phase: float) -> float:
    hype = W1 * delta_a + W2 * retention + W3 * payoff
    continuity = np.clip(1.0 - 0.5 * (gamma_amp + gamma_phase), 0.0, 1.0)
    fatigue = np.clip(gamma_amp + 0.5 * gamma_phase, 0.0, 1.0)
    salience = hype * continuity * np.exp(-LAMBDA * (gamma_amp + gamma_phase)) * (1.0 - K_COEFF * fatigue)
    return float(max(salience, 1.0e-12))


def energy_density(rho: qt.Qobj) -> float:
    n_op = qt.tensor(qt.num(2), qt.qeye(2), qt.qeye(2)) + qt.tensor(qt.qeye(2), qt.num(2), qt.qeye(2)) + qt.tensor(qt.qeye(2), qt.qeye(2), qt.num(2))
    expectation = float(qt.expect(n_op, rho))
    energy = expectation * HBAR * OMEGA
    return energy / VOLUME


def run_grid(phases: Iterable[float], noises: Iterable[tuple[float, float]]) -> list[SalienceResult]:
    results: list[SalienceResult] = []
    for theta_a in phases:
        for theta_r in phases:
            for theta_m in phases:
                for gamma_amp, gamma_phase in noises:
                    rho = build_state(theta_a, theta_r, theta_m, noise_amp=gamma_amp, noise_phase=gamma_phase)
                    delta_a, retention, payoff = compute_metrics(rho)
                    salience = compute_salience(delta_a, retention, payoff, gamma_amp=gamma_amp, gamma_phase=gamma_phase)
                    rho_e = energy_density(rho)
                    sigma_tilde = rho_e / salience
                    s_ref = salience if salience > 0 else 1.0
                    sigma_info = rho_e / (salience / s_ref)
                    results.append(SalienceResult(
                        phase_a=theta_a,
                        phase_r=theta_r,
                        phase_m=theta_m,
                        gamma_amp=gamma_amp,
                        gamma_phase=gamma_phase,
                        delta_a=delta_a,
                        retention=retention,
                        payoff=payoff,
                        salience=salience,
                        energy_density=rho_e,
                        sigma_tilde=sigma_tilde,
                        sigma_info=sigma_info,
                    ))
    return results


def main() -> None:
    phase_set = np.linspace(0.0, np.pi / 2, 3)
    noise_set = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1)]

    results = run_grid(phase_set, noise_set)
    rows = []
    for res in results:
        rows.append((
            res.phase_a,
            res.phase_r,
            res.phase_m,
            res.gamma_amp,
            res.gamma_phase,
            res.delta_a,
            res.retention,
            res.payoff,
            res.salience,
            res.energy_density,
            res.sigma_tilde,
            res.sigma_info,
        ))

    print(tabulate(rows, headers=[
        "θΔ",
        "θR",
        "θM",
        "γ_amp",
        "γ_phase",
        "ΔA",
        "R",
        "M",
        "S'",
        "ρ_energy [J/m³]",
        "σ̃",
        "σ_info",
    ], tablefmt="github", floatfmt=".6g"))


if __name__ == "__main__":
    main()
