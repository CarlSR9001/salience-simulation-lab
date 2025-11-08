"""Experiment S: Neuro-Salience Continuity Trial.

Spiking neural network (Izhikevich) with continuity taxes on synaptic updates.
Tests whether continuity constraints introduce cognitive-style slowdowns and energy anomalies.

The script now supports command-line overrides for sweep parameters so shorter
runs can be executed during exploratory audits without modifying the defaults
recorded in the experimental plan.
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ARTIFACT_DIR = Path("artifacts/neuro_salience")


def parse_float_list(payload: str) -> list[float]:
    values: list[float] = []
    for part in payload.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment S: neuro-salience continuity trial"
    )
    parser.add_argument(
        "--lambda-values",
        type=str,
        default="0.0,0.5,2.0",
        help="Comma-separated λ_c values to sweep",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs per λ_c",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of synthetic samples in the dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Base random seed",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=40,
        help="Hidden neuron count",
    )
    return parser.parse_args()


@dataclass
class NetworkConfig:
    n_input: int = 8
    n_hidden: int = 40
    n_output: int = 2
    dt: float = 0.5  # ms
    spike_threshold: float = 30.0  # mV
    # Izhikevich neuron parameters (regular spiking, tuned for more excitability)
    a: float = 0.03
    b: float = 0.25
    c: float = -65.0
    d: float = 10.0
    v_rest: float = -65.0
    i_app: float = 10.0  # Bias current for neurons
    # Learning parameters
    learning_rate: float = 0.01
    stdp_window: float = 20.0  # ms
    stdp_a_plus: float = 2.0
    stdp_a_minus: float = 1.0
    recurrent_weight_scale: float = 0.15  # Scale for recurrent connections (reduced to avoid over-correlation)
    # Continuity parameters
    fatigue_decay: float = 0.88
    fatigue_gain: float = 0.12
    salience_scale: float = 0.1
    mass_scale: float = 0.5
    # Training parameters
    n_samples: int = 100
    n_epochs: int = 100
    presentation_time: float = 50.0  # ms
    rest_time: float = 30.0  # ms
    seed: int = 2025


class IzhikevichNeuron:
    """Izhikevich spiking neuron model."""

    def __init__(self, cfg: NetworkConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.v = self.cfg.v_rest
        self.u = self.cfg.b * self.v
        self.fired = False
        self.last_spike_time = -float("inf")

    def step(self, current: float, time_now: float) -> bool:
        """Update neuron state and return True if it fires."""
        # Izhikevich dynamics with bias current
        dv = 0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + current + self.cfg.i_app
        du = self.cfg.a * (self.cfg.b * self.v - self.u)

        self.v += self.cfg.dt * dv
        self.u += self.cfg.dt * du

        # Check for spike
        if self.v >= self.cfg.spike_threshold:
            self.v = self.cfg.c
            self.u += self.cfg.d
            self.fired = True
            self.last_spike_time = time_now
            return True
        else:
            self.fired = False
            return False


class SpikingNetwork:
    """Spiking neural network with continuity-taxed synaptic plasticity."""

    def __init__(self, cfg: NetworkConfig, lambda_c: float, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.lambda_c = lambda_c
        self.rng = rng

        # Initialize neurons
        self.hidden_neurons = [IzhikevichNeuron(cfg) for _ in range(cfg.n_hidden)]
        self.output_neurons = [IzhikevichNeuron(cfg) for _ in range(cfg.n_output)]

        # Initialize synaptic weights (stronger initial weights)
        self.W_in_hidden = rng.normal(1.0, 0.3, size=(cfg.n_input, cfg.n_hidden))
        self.W_hidden_out = rng.normal(1.0, 0.3, size=(cfg.n_hidden, cfg.n_output))
        # Add recurrent connections in hidden layer
        self.W_recurrent = rng.normal(0.0, cfg.recurrent_weight_scale, size=(cfg.n_hidden, cfg.n_hidden))
        np.fill_diagonal(self.W_recurrent, 0.0)  # No self-connections

        # Salience tracking for synapses
        self.salience_in_hidden = np.ones((cfg.n_input, cfg.n_hidden))
        self.salience_hidden_out = np.ones((cfg.n_hidden, cfg.n_output))
        self.fatigue_in_hidden = np.zeros((cfg.n_input, cfg.n_hidden))
        self.fatigue_hidden_out = np.zeros((cfg.n_hidden, cfg.n_output))

        # Spike history for STDP
        self.input_spike_times = [[] for _ in range(cfg.n_input)]
        self.hidden_spike_times = [[] for _ in range(cfg.n_hidden)]
        self.output_spike_times = [[] for _ in range(cfg.n_output)]

        self.synaptic_energy = 0.0

    def reset_neurons(self) -> None:
        """Reset all neurons to resting state."""
        for neuron in self.hidden_neurons:
            neuron.reset()
        for neuron in self.output_neurons:
            neuron.reset()
        # Clear spike histories
        self.input_spike_times = [[] for _ in range(self.cfg.n_input)]
        self.hidden_spike_times = [[] for _ in range(self.cfg.n_hidden)]
        self.output_spike_times = [[] for _ in range(self.cfg.n_output)]

    def present_pattern(self, pattern: np.ndarray, time_now: float) -> Tuple[np.ndarray, np.ndarray]:
        """Present input pattern and simulate network for presentation_time."""
        steps = int(self.cfg.presentation_time / self.cfg.dt)
        hidden_spikes = np.zeros(self.cfg.n_hidden)
        output_spikes = np.zeros(self.cfg.n_output)

        for step in range(steps):
            t = time_now + step * self.cfg.dt

            # Record input spikes (Poisson-like based on pattern intensity - increased rate)
            for i in range(self.cfg.n_input):
                spike_prob = pattern[i] * self.cfg.dt / 2.0  # Increased from 10.0 to 2.0
                if self.rng.random() < spike_prob:
                    self.input_spike_times[i].append(t)

            # Compute input currents to hidden layer
            hidden_currents = np.zeros(self.cfg.n_hidden)
            for i in range(self.cfg.n_input):
                if self.input_spike_times[i] and abs(self.input_spike_times[i][-1] - t) < self.cfg.dt:
                    for j in range(self.cfg.n_hidden):
                        hidden_currents[j] += self.W_in_hidden[i, j]

            # Add recurrent currents from hidden layer
            for j in range(self.cfg.n_hidden):
                if self.hidden_neurons[j].fired:
                    for k in range(self.cfg.n_hidden):
                        hidden_currents[k] += self.W_recurrent[j, k]

            # Update hidden neurons
            for j, neuron in enumerate(self.hidden_neurons):
                if neuron.step(hidden_currents[j], t):
                    self.hidden_spike_times[j].append(t)
                    hidden_spikes[j] += 1

            # Compute currents to output layer
            output_currents = np.zeros(self.cfg.n_output)
            for j in range(self.cfg.n_hidden):
                if self.hidden_neurons[j].fired:
                    for k in range(self.cfg.n_output):
                        output_currents[k] += self.W_hidden_out[j, k]

            # Update output neurons
            for k, neuron in enumerate(self.output_neurons):
                if neuron.step(output_currents[k], t):
                    self.output_spike_times[k].append(t)
                    output_spikes[k] += 1

        return hidden_spikes, output_spikes

    def apply_stdp(self, target_class: int) -> float:
        """Apply STDP-like learning with continuity tax."""
        delta_energy = 0.0

        # Update input-to-hidden weights
        for i in range(self.cfg.n_input):
            for j in range(self.cfg.n_hidden):
                delta_w = 0.0
                # STDP: potentiation if pre before post
                for t_pre in self.input_spike_times[i]:
                    for t_post in self.hidden_spike_times[j]:
                        dt_spike = t_post - t_pre
                        if 0 < dt_spike < self.cfg.stdp_window:
                            delta_w += self.cfg.stdp_a_plus * math.exp(-dt_spike / self.cfg.stdp_window)
                        elif -self.cfg.stdp_window < dt_spike < 0:
                            delta_w -= self.cfg.stdp_a_minus * math.exp(dt_spike / self.cfg.stdp_window)

                # Apply continuity tax
                if delta_w != 0:
                    scale = max(self.cfg.salience_scale, 1e-6)
                    delta_norm = abs(delta_w) / scale
                    self.fatigue_in_hidden[i, j] = (
                        self.cfg.fatigue_decay * self.fatigue_in_hidden[i, j]
                        + self.cfg.fatigue_gain * delta_norm
                    )
                    phi = min(self.fatigue_in_hidden[i, j], 0.95)
                    self.salience_in_hidden[i, j] = float(np.clip((1.0 - 0.6 * phi), 0.1, 1.5))

                    mass = 1.0 + self.lambda_c * self.cfg.mass_scale * self.salience_in_hidden[i, j]
                    delta_w_applied = delta_w / mass
                    self.W_in_hidden[i, j] += self.cfg.learning_rate * delta_w_applied
                    delta_energy += abs(delta_w_applied)

        # Update hidden-to-output weights (supervised by target class)
        for j in range(self.cfg.n_hidden):
            for k in range(self.cfg.n_output):
                delta_w = 0.0
                # Encourage connections to target output, discourage others
                for t_pre in self.hidden_spike_times[j]:
                    for t_post in self.output_spike_times[k]:
                        dt_spike = t_post - t_pre
                        reward = 1.0 if k == target_class else -0.5
                        if 0 < dt_spike < self.cfg.stdp_window:
                            delta_w += reward * self.cfg.stdp_a_plus * math.exp(-dt_spike / self.cfg.stdp_window)
                        elif -self.cfg.stdp_window < dt_spike < 0:
                            delta_w -= reward * self.cfg.stdp_a_minus * math.exp(dt_spike / self.cfg.stdp_window)

                # Apply continuity tax
                if delta_w != 0:
                    scale = max(self.cfg.salience_scale, 1e-6)
                    delta_norm = abs(delta_w) / scale
                    self.fatigue_hidden_out[j, k] = (
                        self.cfg.fatigue_decay * self.fatigue_hidden_out[j, k]
                        + self.cfg.fatigue_gain * delta_norm
                    )
                    phi = min(self.fatigue_hidden_out[j, k], 0.95)
                    self.salience_hidden_out[j, k] = float(np.clip((1.0 - 0.6 * phi), 0.1, 1.5))

                    mass = 1.0 + self.lambda_c * self.cfg.mass_scale * self.salience_hidden_out[j, k]
                    delta_w_applied = delta_w / mass
                    self.W_hidden_out[j, k] += self.cfg.learning_rate * delta_w_applied
                    delta_energy += abs(delta_w_applied)

        return delta_energy


def generate_dataset(cfg: NetworkConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Generate simple classification dataset."""
    patterns = []
    labels = []

    # Class 0: first half of inputs active (higher intensity for stronger spikes)
    for _ in range(cfg.n_samples // 2):
        pattern = np.zeros(cfg.n_input)
        pattern[:cfg.n_input // 2] = rng.uniform(0.8, 1.0, cfg.n_input // 2)
        patterns.append(pattern)
        labels.append(0)

    # Class 1: second half of inputs active (higher intensity for stronger spikes)
    for _ in range(cfg.n_samples // 2):
        pattern = np.zeros(cfg.n_input)
        pattern[cfg.n_input // 2:] = rng.uniform(0.8, 1.0, cfg.n_input - cfg.n_input // 2)
        patterns.append(pattern)
        labels.append(1)

    # Shuffle
    indices = rng.permutation(len(patterns))
    return np.array(patterns)[indices], np.array(labels)[indices]


def train_and_evaluate(cfg: NetworkConfig, lambda_c: float) -> Dict[str, float]:
    """Train network and return metrics."""
    rng = np.random.default_rng(cfg.seed + int(lambda_c * 1000))
    patterns, labels = generate_dataset(cfg, rng)

    # Split into train/test
    split = int(0.8 * len(patterns))
    train_patterns, test_patterns = patterns[:split], patterns[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    network = SpikingNetwork(cfg, lambda_c, rng)

    # Training history
    accuracy_history = []
    learning_speed_history = []
    synaptic_energy_history = []

    total_synaptic_energy = 0.0
    time_now = 0.0

    for epoch in range(cfg.n_epochs):
        epoch_energy = 0.0

        for pattern, target in zip(train_patterns, train_labels):
            network.reset_neurons()
            _, output_spikes = network.present_pattern(pattern, time_now)
            delta_energy = network.apply_stdp(target)
            epoch_energy += delta_energy
            time_now += cfg.presentation_time + cfg.rest_time

        total_synaptic_energy += epoch_energy
        synaptic_energy_history.append(epoch_energy)

        # Evaluate accuracy
        correct = 0
        for pattern, target in zip(test_patterns, test_labels):
            network.reset_neurons()
            _, output_spikes = network.present_pattern(pattern, time_now)
            predicted = int(np.argmax(output_spikes))
            if predicted == target:
                correct += 1

        accuracy = correct / len(test_labels)
        accuracy_history.append(accuracy)

        # Learning speed: rate of weight change
        learning_speed = epoch_energy / len(train_patterns) if len(train_patterns) > 0 else 0.0
        learning_speed_history.append(learning_speed)

    # Compute final metrics
    final_accuracy = accuracy_history[-1] if accuracy_history else 0.0
    mean_accuracy = float(np.mean(accuracy_history))

    # Learning speed: epochs to reach 70% accuracy
    epochs_to_threshold = cfg.n_epochs
    for i, acc in enumerate(accuracy_history):
        if acc >= 0.7:
            epochs_to_threshold = i + 1
            break

    # Mean synaptic energy per epoch
    mean_synaptic_energy = float(np.mean(synaptic_energy_history))

    # Firing rate correlations (hidden layer)
    # Compute correlation between firing rates across training
    firing_rates = []
    for pattern in train_patterns[:10]:  # Sample subset
        network.reset_neurons()
        hidden_spikes, _ = network.present_pattern(pattern, time_now)
        firing_rates.append(hidden_spikes)

    if len(firing_rates) > 1:
        firing_rates = np.array(firing_rates)
        corr_matrix = np.corrcoef(firing_rates.T)
        mean_firing_correlation = float(np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])))
    else:
        mean_firing_correlation = 0.0

    # Mean salience across all synapses
    mean_salience = float(
        (np.mean(network.salience_in_hidden) + np.mean(network.salience_hidden_out)) / 2.0
    )

    return {
        "lambda_c": lambda_c,
        "final_accuracy": final_accuracy,
        "mean_accuracy": mean_accuracy,
        "epochs_to_threshold": epochs_to_threshold,
        "total_synaptic_energy": total_synaptic_energy,
        "mean_synaptic_energy_per_epoch": mean_synaptic_energy,
        "mean_firing_correlation": mean_firing_correlation,
        "mean_salience": mean_salience,
    }


def run_sweep(cfg: NetworkConfig, lambdas: List[float]) -> List[Dict[str, float]]:
    """Run experiment across lambda_c values."""
    results = []
    baseline_epochs = None
    baseline_energy = None

    for lambda_c in lambdas:
        print(f"Training with λ_c={lambda_c}...")
        metrics = train_and_evaluate(cfg, lambda_c)

        if baseline_epochs is None:
            baseline_epochs = metrics["epochs_to_threshold"]
            baseline_energy = metrics["mean_synaptic_energy_per_epoch"]

        # Compute relative metrics
        learning_slowdown = metrics["epochs_to_threshold"] / baseline_epochs if baseline_epochs else 1.0
        energy_ratio = metrics["mean_synaptic_energy_per_epoch"] / baseline_energy if baseline_energy else 1.0

        results.append({
            **metrics,
            "learning_slowdown_factor": learning_slowdown,
            "energy_ratio": energy_ratio,
        })

    return results


def write_artifact(entries: List[Dict[str, float]], cfg: NetworkConfig) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())
    payload = []
    for entry in entries:
        payload.append(
            {
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "experiment_name": "experiment_s_neuro_salience",
                "run_id": run_id,
                "n_input": cfg.n_input,
                "n_hidden": cfg.n_hidden,
                "n_output": cfg.n_output,
                "n_epochs": cfg.n_epochs,
                "seed": cfg.seed,
                **entry,
            }
        )
    path = ARTIFACT_DIR / f"neuro_salience_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def main() -> None:
    args = parse_args()
    cfg = NetworkConfig(
        n_hidden=args.hidden,
        n_samples=max(2, args.samples),
        n_epochs=max(1, args.epochs),
        seed=args.seed,
    )
    lambdas = parse_float_list(args.lambda_values)

    print("=== Experiment S: Neuro-Salience Continuity Trial ===")
    print(
        f"Network: {cfg.n_input} inputs, {cfg.n_hidden} hidden (spiking), {cfg.n_output} outputs"
    )
    print(f"Training: {cfg.n_epochs} epochs, {cfg.n_samples} samples")
    print()

    results = run_sweep(cfg, lambdas)
    artifact = write_artifact(results, cfg)

    print("\n=== Results Summary ===")
    for entry in results:
        print(f"λ_c={entry['lambda_c']:>5.1f}:")
        print(f"  Classification accuracy: {entry['final_accuracy']:.3f} (mean: {entry['mean_accuracy']:.3f})")
        print(f"  Learning speed: {entry['epochs_to_threshold']} epochs to 70% (slowdown: {entry['learning_slowdown_factor']:.2f}x)")
        print(f"  Synaptic energy: {entry['mean_synaptic_energy_per_epoch']:.4f} per epoch (ratio: {entry['energy_ratio']:.3f})")
        print(f"  Firing correlations: {entry['mean_firing_correlation']:.3f}")
        print(f"  Mean salience: {entry['mean_salience']:.3f}")
        print()

    # Success criteria check
    print("Success Criteria Check:")
    for entry in results:
        if entry['lambda_c'] > 0:
            cognitive_slowdown = entry['learning_slowdown_factor'] > 1.2
            energy_anomaly = entry['energy_ratio'] < 0.8 or entry['energy_ratio'] > 1.5
            status_parts = []
            if cognitive_slowdown:
                status_parts.append("slowdown observed")
            if energy_anomaly:
                status_parts.append("energy anomaly")
            status = ", ".join(status_parts) if status_parts else "nominal"
            print(f"  λ_c={entry['lambda_c']:>5.1f}: slowdown={entry['learning_slowdown_factor']:.2f}x, energy_ratio={entry['energy_ratio']:.3f} [{status}]")

    print(f"\nResults written to {artifact}")


if __name__ == "__main__":
    main()
