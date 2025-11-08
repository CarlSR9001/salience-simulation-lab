"""Quantum continuity-strain experiment across multiple decoherence channels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tabulate import tabulate
import qutip as qt

MSUN_KG = 1.98847e30  # placeholder, not used directly
HBAR = 1.054_571_817e-34  # J*s
OMEGA = 2.0 * np.pi * 5.0e9  # 5 GHz
VOLUME = (10.0e-9) ** 3  # 10 nm cube -> m^3

GAMMAS = np.linspace(0.0, 0.5, 11)

WEIGHTS = (0.5, 0.3, 0.2)
LAMBDA = 0.5
K_COEFF = 0.8


def bell_state() -> qt.Qobj:
    zero = qt.basis(2, 0)
    one = qt.basis(2, 1)
    psi = (qt.tensor(zero, zero) + qt.tensor(one, one)).unit()
    return psi


def amplitude_damp_first_qubit(rho: qt.Qobj, gamma: float) -> qt.Qobj:
    gamma = float(np.clip(gamma, 0.0, 1.0))
    e0 = qt.Qobj([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]])
    e1 = qt.Qobj([[0.0, np.sqrt(gamma)], [0.0, 0.0]])
    k0 = qt.tensor(e0, qt.qeye(2))
    k1 = qt.tensor(e1, qt.qeye(2))
    return k0 * rho * k0.dag() + k1 * rho * k1.dag()


def phase_damp_first_qubit(rho: qt.Qobj, gamma: float) -> qt.Qobj:
    gamma = float(np.clip(gamma, 0.0, 1.0))
    e0 = qt.Qobj([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]])
    e1 = qt.Qobj([[0.0, 0.0], [0.0, np.sqrt(gamma)]])
    k0 = qt.tensor(e0, qt.qeye(2))
    k1 = qt.tensor(e1, qt.qeye(2))
    return k0 * rho * k0.dag() + k1 * rho * k1.dag()


def depolarize_first_qubit(rho: qt.Qobj, gamma: float) -> qt.Qobj:
    gamma = float(np.clip(gamma, 0.0, 1.0))
    p = gamma / 4.0
    identity = qt.qeye(2)
    pauli_x = qt.sigmax()
    pauli_y = qt.sigmay()
    pauli_z = qt.sigmaz()
    tensors = [
        qt.tensor(identity, identity),
        qt.tensor(pauli_x, identity),
        qt.tensor(pauli_y, identity),
        qt.tensor(pauli_z, identity),
    ]
    result = (1.0 - 3 * p) * tensors[0] * rho * tensors[0].dag()
    for op in tensors[1:]:
        result += p * op * rho * op.dag()
    return result


def compute_salience(rho: qt.Qobj, rho_ref: qt.Qobj, gamma: float) -> float:
    delta = (rho - rho_ref).norm('fro')
    ref_norm = max(rho_ref.norm('fro'), 1.0e-12)
    delta_a = delta / ref_norm

    retention = qt.metrics.fidelity(rho, rho_ref)

    reduced = qt.ptrace(rho, 0)
    entropy = qt.entropy_vn(reduced, 2)
    payoff = entropy / np.log(2.0)

    w1, w2, w3 = WEIGHTS
    c_factor = max(0.0, 1.0 - gamma)
    fatigue = gamma

    s_prime = (w1 * delta_a + w2 * retention + w3 * payoff)
    s_prime *= c_factor
    s_prime *= np.exp(-LAMBDA * gamma)
    s_prime *= max(0.0, 1.0 - K_COEFF * fatigue)
    return float(max(s_prime, 1.0e-12))


def energy_density(rho: qt.Qobj, rho_ref: qt.Qobj) -> float:
    n_op = qt.tensor(qt.num(2), qt.qeye(2)) + qt.tensor(qt.qeye(2), qt.num(2))
    n_ref = qt.expect(n_op, rho_ref)
    n_cur = qt.expect(n_op, rho)
    energy = (n_ref - n_cur) * HBAR * OMEGA
    return float(energy / VOLUME)


def run_channel(name: str, transform) -> Dict[str, float]:
    rho_ref = bell_state().proj()
    data = []
    s_values = []
    energy_values = []

    for gamma in GAMMAS:
        rho_gamma = transform(rho_ref, gamma)
        s_prime = compute_salience(rho_gamma, rho_ref, gamma)
        rho_e = energy_density(rho_gamma, rho_ref)

        s_values.append(s_prime)
        energy_values.append(rho_e)

        sigma_tilde = rho_e / s_prime if s_prime > 0 else 0.0
        data.append((gamma, s_prime, rho_e, sigma_tilde))

    s_values = np.asarray(s_values)
    energy_values = np.asarray(energy_values)
    mask = s_values > 1.0e-9
    s_sel = s_values[mask]
    e_sel = energy_values[mask]

    median_s = float(np.median(s_sel)) if s_sel.size else 1.0
    s_info = s_sel / (median_s + 1.0e-12)

    sigma_tilde_samples = np.where(s_sel > 0.0, e_sel / s_sel, 0.0)
    sigma_info_samples = np.where(s_info > 0.0, e_sel / s_info, 0.0)

    rows = []
    for (gamma, s_prime, rho_e, sigma_tilde) in data:
        s_inf_unit = s_prime / (median_s + 1.0e-12)
        sigma_info = rho_e / s_inf_unit if s_inf_unit > 0.0 else 0.0
        rows.append((gamma, s_prime, rho_e, sigma_tilde, sigma_info))

    print(f"\n=== Channel: {name} ===")
    print(tabulate(rows, headers=[
        "gamma",
        "S'",
        "rho_energy [J/m^3]",
        "sigma_tilde",
        "sigma_info",
    ], tablefmt="github", floatfmt=".6g"))

    summary = {
        "sigma_tilde_median": float(np.median(sigma_tilde_samples)) if sigma_tilde_samples.size else 0.0,
        "sigma_tilde_mean": float(np.mean(sigma_tilde_samples)) if sigma_tilde_samples.size else 0.0,
        "sigma_info_median": float(np.median(sigma_info_samples)) if sigma_info_samples.size else 0.0,
        "sigma_info_mean": float(np.mean(sigma_info_samples)) if sigma_info_samples.size else 0.0,
    }

    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.6g}")
    return summary


def main() -> None:
    channels = {
        "Amplitude damping": amplitude_damp_first_qubit,
        "Phase damping": phase_damp_first_qubit,
        "Depolarizing": depolarize_first_qubit,
    }

    summary_rows = []
    for name, fn in channels.items():
        summary = run_channel(name, fn)
        summary_rows.append((
            name,
            summary["sigma_tilde_median"],
            summary["sigma_tilde_mean"],
            summary["sigma_info_median"],
            summary["sigma_info_mean"],
        ))

    print("\n=== Coupling summary across channels ===")
    print(tabulate(summary_rows, headers=[
        "Channel",
        "σ̃ median",
        "σ̃ mean",
        "σ_info median",
        "σ_info mean",
    ], tablefmt="github", floatfmt=".6g"))


if __name__ == "__main__":
    main()
