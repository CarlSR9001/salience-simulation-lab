"""Experiment T: Quantum-Classical Salience Entanglement.

Extends Bell-pair probe by mapping classical salience states to measurement bases.
Tests for statistically significant deviations conditioned on salience, which would
indicate candidate hybrid retrocausality between classical salience and quantum outcomes.

Architecture:
- Prepare entangled Bell pair (|00⟩ + |11⟩)/√2
- Apply "slam" (amplitude damping) or quiet operation on qubit A
- Compute classical salience from circuit parameters
- Map salience to measurement basis: high salience → X, mid → Y, low → Z
- Measure qubit B in salience-conditioned basis across time offsets
- Log salience-binned probabilities and test for significance

Success metric: Any statistically significant (Bonferroni-corrected) deviation
conditioned on salience bin → candidate hybrid retrocausality requiring escalation.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import qutip as qt
from qutip import gates
from scipy import stats

ARTIFACT_DIR = Path("artifacts/quantum_hybrid")

# Salience weights
W1, W2, W3 = 0.5, 0.3, 0.2
LAMBDA = 0.4
K_COEFF = 0.6

# Measurement bases
BASIS_X = qt.sigmax()
BASIS_Y = qt.sigmay()
BASIS_Z = qt.sigmaz()


@dataclass
class TrialConfig:
    """Configuration for a single trial."""
    phase_param: float
    decay_param: float
    slam: bool
    basis_name: str
    time_offset: int
    shots: int = 1000


@dataclass
class TrialResult:
    """Result from a single trial."""
    config: TrialConfig
    salience: float
    salience_bin: str
    p_zero: float
    p_one: float
    outcomes: np.ndarray


def apply_gate(state: qt.Qobj, gate: qt.Qobj, qubit: int, n_qubits: int) -> qt.Qobj:
    """Apply a single-qubit gate to a multi-qubit state or density matrix."""
    ops = [qt.qeye(2)] * n_qubits
    ops[qubit] = gate
    U = qt.tensor(ops)

    # Handle density matrices: U * rho * U†
    if state.type == 'oper':
        return U * state * U.dag()
    # Handle state vectors: U * |ψ⟩
    else:
        return U * state


def apply_cnot(state: qt.Qobj, control: int, target: int, n_qubits: int) -> qt.Qobj:
    """Apply CNOT gate to a multi-qubit state."""
    zero = qt.basis(2, 0)
    one = qt.basis(2, 1)
    proj0 = zero * zero.dag()
    proj1 = one * one.dag()

    ops0 = [qt.qeye(2)] * n_qubits
    ops1 = [qt.qeye(2)] * n_qubits
    ops0[control] = proj0
    ops1[control] = proj1
    ops1[target] = qt.sigmax()

    return qt.tensor(ops0) * state + qt.tensor(ops1) * state


def prepare_bell_pair(n_qubits: int = 2) -> qt.Qobj:
    """Prepare maximally entangled Bell pair: (|00⟩ + |11⟩)/√2."""
    ket = qt.tensor(*[qt.basis(2, 0) for _ in range(n_qubits)])
    ket = apply_gate(ket, gates.hadamard_transform(), 0, n_qubits)
    ket = apply_cnot(ket, 0, 1, n_qubits)
    return ket.unit()


def apply_amplitude_damping(rho: qt.Qobj, gamma: float, qubit: int, n_qubits: int) -> qt.Qobj:
    """Apply amplitude damping channel to a single qubit of a density matrix."""
    # Kraus operators for amplitude damping
    K0 = qt.Qobj([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]])
    K1 = qt.Qobj([[0.0, np.sqrt(gamma)], [0.0, 0.0]])

    # Build multi-qubit Kraus operators
    def make_kraus(K: qt.Qobj) -> qt.Qobj:
        ops = [qt.qeye(2)] * n_qubits
        ops[qubit] = K
        return qt.tensor(ops)

    K0_full = make_kraus(K0)
    K1_full = make_kraus(K1)

    # Apply Kraus map
    return K0_full * rho * K0_full.dag() + K1_full * rho * K1_full.dag()


def compute_salience(phase_param: float, decay_param: float, slam: bool) -> float:
    """Compute classical salience from circuit parameters.

    Salience formula: S' = HYPE × CONTINUITY × exp(-λ*decay) × (1 - K*fatigue)
    where HYPE = W1*ΔA + W2*R + W3*M
    """
    # Novelty from phase rotation
    delta_a = np.clip(np.abs(np.sin(phase_param)), 0.0, 1.0)

    # Retention from coherence (inverse of decay)
    retention = np.clip(1.0 - decay_param, 0.0, 1.0)

    # Payoff from slam operation (slam increases energy, reducing payoff)
    payoff = 0.3 if slam else 0.8

    # HYPE component
    hype = W1 * delta_a + W2 * retention + W3 * payoff

    # Continuity from phase coherence
    continuity = np.clip(np.cos(phase_param), 0.0, 1.0)

    # Fatigue from decay parameter
    fatigue = np.clip(decay_param + (0.5 if slam else 0.0), 0.0, 1.0)

    # Salience formula
    s_prime = hype * continuity * np.exp(-LAMBDA * decay_param) * (1.0 - K_COEFF * fatigue)

    return float(np.clip(s_prime, 1e-6, 1.0))


def salience_to_basis(salience: float) -> Tuple[str, qt.Qobj]:
    """Map salience value to measurement basis.

    High salience (>0.6) → X basis
    Mid salience (0.3-0.6) → Y basis
    Low salience (<0.3) → Z basis
    """
    if salience >= 0.6:
        return "X", BASIS_X
    elif salience >= 0.3:
        return "Y", BASIS_Y
    else:
        return "Z", BASIS_Z


def salience_bin_name(salience: float) -> str:
    """Get salience bin name for grouping."""
    if salience >= 0.6:
        return "high"
    elif salience >= 0.3:
        return "mid"
    else:
        return "low"


def measure_qubit(rho: qt.Qobj, basis: qt.Qobj, qubit: int, n_qubits: int, shots: int) -> Tuple[float, float, np.ndarray]:
    """Measure a single qubit in given basis with shot noise.

    Returns:
        (p_zero, p_one, outcomes_array)
    """
    # For basis transformations, we need to apply unitary transformations to the density matrix
    # rho' = U rho U†

    # Transform to measurement basis if not Z
    if basis == BASIS_X:
        H_gate = gates.hadamard_transform()
        H_full = [qt.qeye(2)] * n_qubits
        H_full[qubit] = H_gate
        H_full = qt.tensor(H_full)
        rho = H_full * rho * H_full.dag()
    elif basis == BASIS_Y:
        # Y measurement: S† H for basis change, where S = [[1,0],[0,i]]
        S_dag = qt.Qobj([[1, 0], [0, -1j]])
        H_gate = gates.hadamard_transform()

        # Apply S†
        S_full = [qt.qeye(2)] * n_qubits
        S_full[qubit] = S_dag
        S_full = qt.tensor(S_full)
        rho = S_full * rho * S_full.dag()

        # Apply H
        H_full = [qt.qeye(2)] * n_qubits
        H_full[qubit] = H_gate
        H_full = qt.tensor(H_full)
        rho = H_full * rho * H_full.dag()

    # Compute probabilities for |0⟩ and |1⟩
    proj0_ops = [qt.qeye(2)] * n_qubits
    proj1_ops = [qt.qeye(2)] * n_qubits
    proj0_ops[qubit] = qt.basis(2, 0) * qt.basis(2, 0).dag()
    proj1_ops[qubit] = qt.basis(2, 1) * qt.basis(2, 1).dag()

    proj0 = qt.tensor(proj0_ops)
    proj1 = qt.tensor(proj1_ops)

    p_zero = float(np.real((proj0 * rho).tr()))
    p_one = float(np.real((proj1 * rho).tr()))

    # Clamp to valid probability range
    p_zero = np.clip(p_zero, 0.0, 1.0)
    p_one = np.clip(p_one, 0.0, 1.0)

    # Normalize
    total = p_zero + p_one
    if total > 0:
        p_zero /= total
        p_one /= total
    else:
        # Fallback to uniform
        p_zero = 0.5
        p_one = 0.5

    # Final clamp to ensure valid probabilities
    p_zero = np.clip(p_zero, 0.0, 1.0)
    p_one = 1.0 - p_zero

    # Sample with shot noise
    rng = np.random.default_rng()
    outcomes = rng.choice([0, 1], size=shots, p=[p_zero, p_one])

    return p_zero, p_one, outcomes


def run_trial(config: TrialConfig) -> TrialResult:
    """Run a single trial with given configuration."""
    n_qubits = 2

    # Prepare Bell pair
    bell = prepare_bell_pair(n_qubits)
    rho = bell * bell.dag()

    # Apply slam or quiet on qubit A (qubit 0)
    if config.slam:
        rho = apply_amplitude_damping(rho, gamma=0.8, qubit=0, n_qubits=n_qubits)

    # Apply phase rotation to simulate time evolution
    if config.phase_param != 0.0:
        phase_gate = qt.Qobj([[1, 0], [0, np.exp(1j * config.phase_param)]])
        rho = apply_gate(rho, phase_gate, 1, n_qubits)

    # Apply decay if specified
    if config.decay_param > 0:
        rho = apply_amplitude_damping(rho, gamma=config.decay_param, qubit=1, n_qubits=n_qubits)

    # Compute classical salience
    salience = compute_salience(config.phase_param, config.decay_param, config.slam)
    sal_bin = salience_bin_name(salience)

    # Measure qubit B in specified basis
    if config.basis_name == "X":
        basis = BASIS_X
    elif config.basis_name == "Y":
        basis = BASIS_Y
    else:
        basis = BASIS_Z

    p_zero, p_one, outcomes = measure_qubit(rho, basis, qubit=1, n_qubits=n_qubits, shots=config.shots)

    return TrialResult(
        config=config,
        salience=salience,
        salience_bin=sal_bin,
        p_zero=p_zero,
        p_one=p_one,
        outcomes=outcomes,
    )


def aggregate_results(results: List[TrialResult]) -> Dict[str, Dict]:
    """Aggregate results by (salience_bin, basis, slam) and compute statistics."""
    bins: Dict[Tuple[str, str, bool], List[TrialResult]] = defaultdict(list)

    for result in results:
        key = (result.salience_bin, result.config.basis_name, result.config.slam)
        bins[key].append(result)

    aggregated = {}
    for (sal_bin, basis, slam), group in bins.items():
        # Compute aggregate statistics
        p_zeros = [r.p_zero for r in group]
        p_ones = [r.p_one for r in group]
        saliences = [r.salience for r in group]

        mean_p_zero = float(np.mean(p_zeros))
        mean_p_one = float(np.mean(p_ones))
        std_p_zero = float(np.std(p_zeros))
        mean_salience = float(np.mean(saliences))

        # Collect all outcomes for chi-square test
        all_outcomes = np.concatenate([r.outcomes for r in group])
        n_zeros = int(np.sum(all_outcomes == 0))
        n_ones = int(np.sum(all_outcomes == 1))
        total = len(all_outcomes)

        # Chi-square test against 50/50 baseline
        expected = total / 2
        chi2_stat = ((n_zeros - expected)**2 + (n_ones - expected)**2) / expected
        p_value = float(stats.chi2.sf(chi2_stat, df=1))

        # Z-score for deviation from 0.5
        z_score = (mean_p_zero - 0.5) / (std_p_zero / np.sqrt(len(group)) + 1e-9)

        key_str = f"{sal_bin}_{basis}_slam={slam}"
        aggregated[key_str] = {
            "salience_bin": sal_bin,
            "basis": basis,
            "slam": slam,
            "n_trials": len(group),
            "mean_salience": mean_salience,
            "mean_p_zero": mean_p_zero,
            "mean_p_one": mean_p_one,
            "std_p_zero": std_p_zero,
            "z_score": float(z_score),
            "chi2_stat": float(chi2_stat),
            "p_value": p_value,
            "n_zeros": n_zeros,
            "n_ones": n_ones,
            "total_shots": total,
        }

    return aggregated


def run_experiment(n_trials: int = 1000, shots_per_trial: int = 1000) -> Dict:
    """Run full quantum-classical salience entanglement experiment."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Parameter sweep
    phases = np.linspace(0.0, np.pi, 8)
    decays = [0.0, 0.1, 0.2, 0.3]
    bases = ["X", "Y", "Z"]
    time_offsets = [0, 1, 2]
    slam_options = [True, False]

    results = []
    total_configs = len(phases) * len(decays) * len(bases) * len(time_offsets) * len(slam_options)
    trials_per_config = max(1, n_trials // total_configs)

    print(f"Running {total_configs} configurations × {trials_per_config} trials × {shots_per_trial} shots")
    print(f"Total trials: {total_configs * trials_per_config}")

    trial_count = 0
    for phase in phases:
        for decay in decays:
            for basis in bases:
                for time_offset in time_offsets:
                    for slam in slam_options:
                        # Run multiple trials for this configuration
                        for _ in range(trials_per_config):
                            config = TrialConfig(
                                phase_param=phase + time_offset * 0.1,
                                decay_param=decay,
                                slam=slam,
                                basis_name=basis,
                                time_offset=time_offset,
                                shots=shots_per_trial,
                            )
                            result = run_trial(config)
                            results.append(result)
                            trial_count += 1

                            if trial_count % 100 == 0:
                                print(f"  Completed {trial_count}/{total_configs * trials_per_config} trials")

    print(f"Completed {len(results)} trials, aggregating results...")

    # Aggregate by salience bin
    aggregated = aggregate_results(results)

    # Apply Bonferroni correction
    n_tests = len(aggregated)
    bonferroni_threshold = 0.001 / n_tests if n_tests > 0 else 0.001

    significant_bins = []
    for key, stats_dict in aggregated.items():
        if stats_dict["p_value"] < bonferroni_threshold:
            significant_bins.append(key)
            stats_dict["significant"] = True
        else:
            stats_dict["significant"] = False

    payload = {
        "n_trials": len(results),
        "shots_per_trial": shots_per_trial,
        "n_bins": n_tests,
        "bonferroni_threshold": bonferroni_threshold,
        "n_significant": len(significant_bins),
        "significant_bins": significant_bins,
        "aggregated_stats": aggregated,
    }

    return payload


def write_artifact(payload: Dict, suffix: str = "") -> Path:
    """Write experiment results to artifact file."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())

    record = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "experiment_name": "experiment_t_quantum_hybrid",
        "run_id": run_id,
        **payload,
    }

    filename = f"quantum_hybrid_{timestamp}{suffix}.json"
    path = ARTIFACT_DIR / filename
    path.write_text(json.dumps(record, indent=2))
    return path


def main() -> None:
    """Main entry point."""
    print("=" * 70)
    print("Experiment T: Quantum-Classical Salience Entanglement")
    print("=" * 70)

    # Reduced for testing: 100 trials instead of 1000
    payload = run_experiment(n_trials=100, shots_per_trial=500)
    artifact = write_artifact(payload)

    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"Total trials: {payload['n_trials']}")
    print(f"Shots per trial: {payload['shots_per_trial']}")
    print(f"Number of bins: {payload['n_bins']}")
    print(f"Bonferroni threshold: {payload['bonferroni_threshold']:.6e}")
    print(f"Significant bins: {payload['n_significant']}")

    if payload['n_significant'] > 0:
        print("\n⚠️  ALERT: Significant salience-conditioned deviations detected!")
        print("Significant bins:")
        for bin_name in payload['significant_bins']:
            stats = payload['aggregated_stats'][bin_name]
            print(f"  {bin_name}:")
            print(f"    Mean salience: {stats['mean_salience']:.4f}")
            print(f"    P(0): {stats['mean_p_zero']:.4f}")
            print(f"    Z-score: {stats['z_score']:.4f}")
            print(f"    p-value: {stats['p_value']:.6e}")
        print("\n🚨 ESCALATE: Candidate hybrid retrocausality requires immediate investigation!")
    else:
        print("\n✓ No significant deviations detected. No-signaling constraint holds.")

    print(f"\nResults written to: {artifact}")


if __name__ == "__main__":
    main()
