"""Novel Experiment: Salience Conservation Laws

Tests whether total salience is conserved under various transformations and exchanges.

Physics Analogy:
- Energy is conserved in closed systems
- Momentum is conserved in collisions
- Charge is conserved in reactions

Question: Is salience conserved when transferred between:
1. Different components of a system (core vs edge)
2. Different systems (coupled controllers)
3. Different domains (control state vs learning weights)
4. Time (does salience flow but preserve total?)

Conservation Test Protocol:
1. Initialize system with known total salience
2. Apply transformation (state update, coupling, exchange)
3. Measure total salience after transformation
4. Check: |S_before - S_after| / S_before < epsilon

If salience is conserved, it suggests a deeper substrate (like a field theory).
If violated, identify where salience is created/destroyed (sources/sinks).
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from tabulate import tabulate

ARTIFACT_DIR = Path("artifacts/salience_conservation")

CONSERVATION_EPSILON = 0.05  # 5% violation threshold


@dataclass
class ConservationTest:
    """Result of a conservation test."""
    test_name: str
    salience_before: float
    salience_after: float
    delta: float
    delta_percent: float
    conserved: bool
    violation_type: str  # "none", "creation", "destruction"


def compute_total_salience(components: List[np.ndarray]) -> float:
    """Compute total salience across all components."""
    return float(np.sum([np.sum(c) for c in components]))


def test_component_exchange():
    """Test 1: Salience exchange between core and edge components.

    Scenario: Core transfers salience to edge via coupling.
    Conservation: S_core(t) + S_edge(t) = constant
    """
    print("\n" + "="*70)
    print("TEST 1: Component Exchange Conservation")
    print("="*70)

    # Initialize
    core_salience = np.array([0.9, 0.85, 0.88, 0.92, 0.87])  # High salience core
    edge_salience = np.array([0.3, 0.4, 0.35])  # Low salience edge

    S_before = compute_total_salience([core_salience, edge_salience])

    print(f"Before: S_core = {np.sum(core_salience):.3f}, S_edge = {np.sum(edge_salience):.3f}")
    print(f"        S_total = {S_before:.3f}")

    # Simulate exchange: core loses, edge gains (with damping)
    exchange_rate = 0.15
    transfer = core_salience * exchange_rate

    core_salience_after = core_salience - transfer
    edge_salience_after = edge_salience + np.mean(transfer)  # Broadcast to edge

    S_after = compute_total_salience([core_salience_after, edge_salience_after])

    print(f"After:  S_core = {np.sum(core_salience_after):.3f}, S_edge = {np.sum(edge_salience_after):.3f}")
    print(f"        S_total = {S_after:.3f}")

    delta = S_after - S_before
    delta_pct = abs(delta) / S_before * 100

    print(f"\nΔS = {delta:.4f} ({delta_pct:.2f}%)")

    conserved = abs(delta_pct) < CONSERVATION_EPSILON * 100

    if conserved:
        print("✓ CONSERVED")
    else:
        violation = "creation" if delta > 0 else "destruction"
        print(f"⚠️  VIOLATION: Salience {violation}")

    return ConservationTest(
        test_name="component_exchange",
        salience_before=S_before,
        salience_after=S_after,
        delta=delta,
        delta_percent=delta_pct,
        conserved=conserved,
        violation_type="none" if conserved else violation,
    )


def test_coupled_systems():
    """Test 2: Salience exchange between two coupled systems.

    Scenario: Two controllers share state, salience flows between them.
    Conservation: S_A(t) + S_B(t) = constant
    """
    print("\n" + "="*70)
    print("TEST 2: Coupled Systems Conservation")
    print("="*70)

    # Initialize two systems
    system_a = np.array([0.8, 0.75, 0.82, 0.77, 0.79])
    system_b = np.array([0.6, 0.65, 0.62, 0.68, 0.64])

    S_before = compute_total_salience([system_a, system_b])

    print(f"Before: S_A = {np.sum(system_a):.3f}, S_B = {np.sum(system_b):.3f}")
    print(f"        S_total = {S_before:.3f}")

    # Simulate coupling: systems equilibrate
    coupling_strength = 0.2
    mean_salience = (system_a + system_b) / 2

    system_a_after = system_a + coupling_strength * (mean_salience - system_a)
    system_b_after = system_b + coupling_strength * (mean_salience - system_b)

    S_after = compute_total_salience([system_a_after, system_b_after])

    print(f"After:  S_A = {np.sum(system_a_after):.3f}, S_B = {np.sum(system_b_after):.3f}")
    print(f"        S_total = {S_after:.3f}")

    delta = S_after - S_before
    delta_pct = abs(delta) / S_before * 100

    print(f"\nΔS = {delta:.4f} ({delta_pct:.2f}%)")

    conserved = abs(delta_pct) < CONSERVATION_EPSILON * 100

    if conserved:
        print("✓ CONSERVED")
    else:
        violation = "creation" if delta > 0 else "destruction"
        print(f"⚠️  VIOLATION: Salience {violation}")

    return ConservationTest(
        test_name="coupled_systems",
        salience_before=S_before,
        salience_after=S_after,
        delta=delta,
        delta_percent=delta_pct,
        conserved=conserved,
        violation_type="none" if conserved else violation,
    )


def test_fatigue_decay():
    """Test 3: Salience decay under fatigue.

    Scenario: System accumulates fatigue, salience decreases.
    Question: Is salience destroyed, or transferred to a hidden reservoir?

    If destroyed: Salience is not conserved (sink exists).
    If conserved: Fatigue acts as a "salience capacitor."
    """
    print("\n" + "="*70)
    print("TEST 3: Fatigue Decay (Sink Detection)")
    print("="*70)

    salience = np.array([0.85, 0.88, 0.82, 0.87, 0.86])
    fatigue_reservoir = 0.0  # Track "lost" salience

    S_before = np.sum(salience) + fatigue_reservoir

    print(f"Before: S_active = {np.sum(salience):.3f}, S_reservoir = {fatigue_reservoir:.3f}")
    print(f"        S_total = {S_before:.3f}")

    # Apply fatigue decay
    fatigue_rate = 0.15
    decay = salience * fatigue_rate

    salience_after = salience - decay

    # Two scenarios:
    # A) Salience destroyed (not conserved)
    fatigue_reservoir_destroyed = 0.0

    # B) Salience stored in fatigue reservoir (conserved)
    fatigue_reservoir_stored = fatigue_reservoir + np.sum(decay)

    S_after_destroyed = np.sum(salience_after) + fatigue_reservoir_destroyed
    S_after_stored = np.sum(salience_after) + fatigue_reservoir_stored

    print(f"After:  S_active = {np.sum(salience_after):.3f}")
    print(f"  Scenario A (destroyed): S_reservoir = {fatigue_reservoir_destroyed:.3f}, S_total = {S_after_destroyed:.3f}")
    print(f"  Scenario B (stored):    S_reservoir = {fatigue_reservoir_stored:.3f}, S_total = {S_after_stored:.3f}")

    delta_destroyed = S_after_destroyed - S_before
    delta_stored = S_after_stored - S_before

    delta_pct_destroyed = abs(delta_destroyed) / S_before * 100
    delta_pct_stored = abs(delta_stored) / S_before * 100

    print(f"\nScenario A: ΔS = {delta_destroyed:.4f} ({delta_pct_destroyed:.2f}%)")
    print(f"Scenario B: ΔS = {delta_stored:.4f} ({delta_pct_stored:.2f}%)")

    conserved_destroyed = abs(delta_pct_destroyed) < CONSERVATION_EPSILON * 100
    conserved_stored = abs(delta_pct_stored) < CONSERVATION_EPSILON * 100

    if conserved_destroyed:
        print("\n✓ Scenario A: Salience destroyed but total remains constant (edge case)")
    else:
        print("\n⚠️  Scenario A: Salience NOT conserved (sink exists)")

    if conserved_stored:
        print("✓ Scenario B: Salience CONSERVED (fatigue acts as reservoir)")
    else:
        print("⚠️  Scenario B: Violation even with reservoir")

    # Return the more realistic scenario (B - stored)
    return ConservationTest(
        test_name="fatigue_decay",
        salience_before=S_before,
        salience_after=S_after_stored,
        delta=delta_stored,
        delta_percent=delta_pct_stored,
        conserved=conserved_stored,
        violation_type="none" if conserved_stored else "sink",
    )


def test_continuity_tax_transformation():
    """Test 4: Salience under continuity tax transformation.

    Scenario: Applying continuity tax changes effective mass but not salience itself.
    Question: Does the tax consume salience, or just modulate its effect?

    If conserved: Tax is a "lens" (changes how salience affects dynamics).
    If violated: Tax is a "drain" (consumes salience).
    """
    print("\n" + "="*70)
    print("TEST 4: Continuity Tax Transformation")
    print("="*70)

    salience = np.array([0.83, 0.86, 0.81, 0.85, 0.84])

    S_before = np.sum(salience)

    print(f"Before: S = {S_before:.3f}")

    # Apply continuity tax (should not change salience itself, only m_eff)
    lambda_c = 2.0
    m_eff_before = 1.0
    m_eff_after = 1.0 + lambda_c * np.mean(salience)

    # Salience should remain unchanged
    salience_after = salience.copy()  # Tax doesn't modify salience

    S_after = np.sum(salience_after)

    print(f"After:  S = {S_after:.3f}")
    print(f"        m_eff: {m_eff_before:.3f} → {m_eff_after:.3f}")

    delta = S_after - S_before
    delta_pct = abs(delta) / S_before * 100 if S_before > 0 else 0.0

    print(f"\nΔS = {delta:.4f} ({delta_pct:.2f}%)")

    conserved = abs(delta_pct) < CONSERVATION_EPSILON * 100

    if conserved:
        print("✓ CONSERVED: Continuity tax is a modulator, not a drain")
    else:
        print("⚠️  VIOLATION: Tax consumes salience")

    return ConservationTest(
        test_name="continuity_tax",
        salience_before=S_before,
        salience_after=S_after,
        delta=delta,
        delta_percent=delta_pct,
        conserved=conserved,
        violation_type="none" if conserved else "drain",
    )


def test_temporal_flow():
    """Test 5: Salience conservation over time.

    Scenario: System evolves through multiple steps with various updates.
    Question: Is total salience conserved as a function of time?

    Tests the "salience current" hypothesis: dS/dt = 0 for closed systems.
    """
    print("\n" + "="*70)
    print("TEST 5: Temporal Flow Conservation")
    print("="*70)

    salience = np.array([0.8, 0.75, 0.82, 0.78, 0.79])
    fatigue = np.zeros(5)

    steps = 50
    S_history = []

    for step in range(steps):
        # Compute total salience (active + fatigue reservoir)
        S_total = np.sum(salience) + np.sum(fatigue)
        S_history.append(S_total)

        # Update: salience decays, fatigue accumulates
        decay_rate = 0.02
        recovery_rate = 0.015

        decay = salience * decay_rate
        recovery = fatigue * recovery_rate

        salience = salience - decay + recovery
        fatigue = fatigue + decay - recovery

        # Clip to valid ranges
        salience = np.clip(salience, 0.01, 1.0)
        fatigue = np.clip(fatigue, 0.0, 1.0)

    S_history = np.array(S_history)

    S_initial = S_history[0]
    S_final = S_history[-1]

    delta = S_final - S_initial
    delta_pct = abs(delta) / S_initial * 100

    # Check variance over time
    S_variance = np.var(S_history)
    S_drift = np.max(S_history) - np.min(S_history)

    print(f"Initial S: {S_initial:.3f}")
    print(f"Final S:   {S_final:.3f}")
    print(f"ΔS:        {delta:.4f} ({delta_pct:.2f}%)")
    print(f"Variance:  {S_variance:.6f}")
    print(f"Drift:     {S_drift:.4f}")

    conserved = abs(delta_pct) < CONSERVATION_EPSILON * 100

    if conserved:
        print("\n✓ CONSERVED: Total salience stable over time")
    else:
        violation = "creation" if delta > 0 else "destruction"
        print(f"\n⚠️  VIOLATION: Salience {violation} over time")

    return ConservationTest(
        test_name="temporal_flow",
        salience_before=S_initial,
        salience_after=S_final,
        delta=delta,
        delta_percent=delta_pct,
        conserved=conserved,
        violation_type="none" if conserved else violation,
    )


def main():
    """Run all salience conservation tests."""

    print("\n" + "="*70)
    print("SALIENCE CONSERVATION LAW EXPERIMENTS")
    print("="*70)
    print("\nTesting whether salience behaves like a conserved quantity (energy, charge).")
    print("If conserved → suggests deeper field-theoretic substrate.")
    print("If violated → identify sources and sinks.\n")

    tests = [
        test_component_exchange(),
        test_coupled_systems(),
        test_fatigue_decay(),
        test_continuity_tax_transformation(),
        test_temporal_flow(),
    ]

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    table_data = []
    for t in tests:
        status = "✓ Conserved" if t.conserved else f"⚠️  Violated ({t.violation_type})"
        table_data.append([
            t.test_name,
            f"{t.salience_before:.3f}",
            f"{t.salience_after:.3f}",
            f"{t.delta:+.4f}",
            f"{t.delta_percent:.2f}%",
            status,
        ])

    print("\n" + tabulate(
        table_data,
        headers=["Test", "S_before", "S_after", "ΔS", "ΔS %", "Status"],
        tablefmt="grid"
    ))

    conserved_count = sum(1 for t in tests if t.conserved)
    violated_count = len(tests) - conserved_count

    print(f"\nConserved: {conserved_count}/{len(tests)}")
    print(f"Violated:  {violated_count}/{len(tests)}")

    if conserved_count == len(tests):
        print("\n✓ SALIENCE APPEARS TO BE A CONSERVED QUANTITY")
        print("  Implication: Salience may be a fundamental physical property")
        print("  similar to energy or charge in a field theory.")
    elif conserved_count > len(tests) / 2:
        print("\n⚠️  SALIENCE APPROXIMATELY CONSERVED")
        print("  Some violations detected. May require:")
        print("    - Better tracking of hidden reservoirs (fatigue, entropy)")
        print("    - Accounting for boundary flows (external coupling)")
    else:
        print("\n⚠️  SALIENCE NOT CONSERVED")
        print("  Significant violations detected. Salience may be:")
        print("    - Emergent (not fundamental)")
        print("    - Open system (flows to environment)")
        print("    - Incorrectly defined (missing terms)")

    # Save results
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = ARTIFACT_DIR / f"conservation_{timestamp}.json"

    output = {
        "experiment": "salience_conservation",
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": str(uuid.uuid4()),
        "tests": [
            {
                "test_name": t.test_name,
                "salience_before": t.salience_before,
                "salience_after": t.salience_after,
                "delta": float(t.delta),
                "delta_percent": float(t.delta_percent),
                "conserved": bool(t.conserved),
                "violation_type": str(t.violation_type),
            }
            for t in tests
        ],
        "summary": {
            "conserved_count": conserved_count,
            "violated_count": violated_count,
            "conservation_rate": conserved_count / len(tests),
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    return tests


if __name__ == "__main__":
    main()
