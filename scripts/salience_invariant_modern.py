"""Modern salience invariant snapshot.

This mirrors the additive + soft-gated salience update currently embedded in
ContinuityPID. Keeping it here lets us evolve the controller while preserving
this formulation for later reuse or analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class ModernSalienceParams:
    salience_floor: float = 0.0
    salience_scale: float = 0.05
    fatigue_decay: float = 0.9
    fatigue_gain: float = 0.1
    weight_novelty: float = 0.6
    weight_retention: float = 0.25
    weight_payoff: float = 0.15
    gate_bias: float = 0.5
    gate_continuity: float = 4.0
    gate_fatigue: float = 3.0
    gate_temp: float = 0.06
    gate_floor: float = 0.1
    credit_gain: float = 6.0
    credit_bias: float = 0.4
    credit_floor: float = 0.1
    credit_alpha: float = 0.65
    continuity_scale: float = 1.0


@dataclass
class ModernSalienceState:
    salience: float = 1.0
    retention: float = 0.9
    payoff: float = 0.9
    fatigue: float = 0.0


def _sigmoid(x: float, temp: float) -> float:
    scale = max(temp, 1e-6)
    z = x / scale
    if z >= 0.0:
        exp_neg = math.exp(-z)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(z)
    return exp_pos / (1.0 + exp_pos)


def _compute_credit(params: ModernSalienceParams, control_delta: float, measurement_delta: float) -> float:
    weighted_measurement = params.credit_alpha * measurement_delta
    denom = control_delta + weighted_measurement + 1e-9
    ratio = 0.0 if denom <= 0.0 else control_delta / denom
    ratio = max(0.0, min(1.0, ratio))
    credit_input = params.credit_bias + params.credit_gain * (ratio - 0.5)
    credit = _sigmoid(credit_input, params.gate_temp)
    return max(params.credit_floor, credit)


def update_salience(
    state: ModernSalienceState,
    params: ModernSalienceParams,
    delta: float,
    error: float,
    control_delta: float,
    measurement_delta: float,
) -> ModernSalienceState:
    scale = max(params.salience_scale, 1e-6)
    delta_mag = abs(delta)

    novelty_decay = math.exp(-delta_mag / scale)
    novelty_gain = 1.0 - novelty_decay

    retention = 0.92 * state.retention + 0.08 * novelty_decay
    payoff_sample = math.exp(-abs(error))
    payoff = 0.9 * state.payoff + 0.1 * payoff_sample

    fatigue = params.fatigue_decay * state.fatigue + params.fatigue_gain * delta_mag / scale
    phi = min(fatigue, 0.95)

    credit = _compute_credit(params, abs(control_delta), abs(measurement_delta))

    value = (
        params.weight_novelty * (novelty_gain * credit)
        + params.weight_retention * float(np.clip(retention, 0.2, 1.1))
        + params.weight_payoff * float(np.clip(payoff, 0.2, 1.1))
    )

    continuity_scale = max(params.continuity_scale, 1e-6)
    strain = math.tanh(delta_mag / (scale * continuity_scale))
    gate_input = params.gate_bias + params.gate_continuity * strain - params.gate_fatigue * phi
    gate = max(params.gate_floor, _sigmoid(gate_input, params.gate_temp))

    salience = value * gate
    lower_bound = max(params.salience_floor, 1e-3)
    salience = float(np.clip(salience, lower_bound, 1.8))

    return ModernSalienceState(
        salience=salience,
        retention=retention,
        payoff=payoff,
        fatigue=fatigue,
    )
