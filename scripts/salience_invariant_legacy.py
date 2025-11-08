"""Legacy salience invariant snapshot.

This module preserves the multiplicative salience formulation used prior to
introducing additive/soft-gated variants. It is intentionally frozen so we can
compare legacy behaviour against revised designs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class LegacySalienceParams:
    """Tunable constants carried over from the original ContinuityPID."""

    salience_floor: float = 0.0
    salience_scale: float = 1.0
    fatigue_decay: float = 0.9
    fatigue_gain: float = 0.1


@dataclass
class LegacySalienceState:
    """Container for per-channel state variables."""

    salience: float = 1.0
    retention: float = 0.9
    payoff: float = 0.9
    fatigue: float = 0.0


def legacy_update(
    state: LegacySalienceState,
    params: LegacySalienceParams,
    delta: float,
    error: float,
) -> LegacySalienceState:
    """Apply the original multiplicative salience invariant update."""

    scale = max(params.salience_scale, 1e-6)
    novelty = math.exp(-abs(delta) / scale)
    retention = 0.92 * state.retention + 0.08 * novelty
    payoff = 0.9 * state.payoff + 0.1 * math.exp(-abs(error))
    fatigue = params.fatigue_decay * state.fatigue + params.fatigue_gain * abs(delta) / scale
    phi = min(fatigue, 0.95)

    retention_term = float(np.clip(retention, 0.2, 1.1))
    payoff_term = float(np.clip(payoff, 0.2, 1.1))
    salience = novelty * retention_term * payoff_term * (1.0 - phi)

    lower_bound = max(params.salience_floor, 1e-3)
    updated_salience = float(np.clip(salience, lower_bound, 1.5))

    return LegacySalienceState(
        salience=updated_salience,
        retention=retention,
        payoff=payoff,
        fatigue=fatigue,
    )
