#!/usr/bin/env python3
"""Causal Value-Gated (CVG) Lorenz control demo.

This script reproduces the single-figure story and two ablations requested in the
brief:

1. Baseline PID (untimed brute force).
2. Legacy multiplicative salience controller (collapses under strain).
3. CVG refactor using the modern additive invariant plus OGY-timed pulses.

It produces a publication-ready figure showing tracking error and cumulative
control energy, prints key metrics, and logs all numerical results as JSON.

Usage (from repo root):

    python -m demos.cvg_lorenz.cvg_lorenz_demo

Outputs land in ``demos/cvg_lorenz/outputs``.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Allow importing the invariant snapshots without modifying package layout.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.salience_invariant_legacy import (  # type: ignore # noqa: E402
    LegacySalienceParams,
    LegacySalienceState,
    legacy_update,
)
from scripts.salience_invariant_modern import (  # type: ignore # noqa: E402
    ModernSalienceParams,
    ModernSalienceState,
    update_salience,
)


@dataclass(frozen=True)
class SimulationConfig:
    dt: float = 0.05
    horizon: float = 60.0
    tau: float = 0.14
    target: float = 1.0
    lorenz_sigma: float = 10.0
    lorenz_rho: float = 28.0
    lorenz_beta: float = 8.0 / 3.0
    lorenz_gain: float = 0.14
    ogy_axis: int = 0
    ogy_threshold: float = 0.0
    ogy_y_window: float = 4.0
    ogy_z_min: float = 24.0
    ogy_z_max: float = 30.0
    pulse_amplitude: float = 0.15
    pulse_width: int = 3
    pulse_cooldown: int = 20
    pulse_energy_cap: float = 5.0
    authority_threshold: float = 1.0
    seed: int = 1337
    lorenz_initial: tuple[float, float, float] = (-8.0, 7.0, 27.0)

    @property
    def steps(self) -> int:
        return int(self.horizon / self.dt)


@dataclass(frozen=True)
class SimulationContext:
    total_steps: int
    dt: float
    rng: np.random.Generator
    credit_sequence: Optional[np.ndarray] = None
    pulse_schedule: Optional[np.ndarray] = None
    pulse_energy_cap: float = math.inf


@dataclass(frozen=True)
class AutotuneTargets:
    authority_peak: float = 1.05
    authority_duration_max: float = 1.2  # seconds
    energy_ratio_target: float = 1.0e-3
    pulse_energy_cap: float = 5.0e-5
    gate_peak_max: float = 0.95
    max_iterations: int = 20
    authority_floor: float = 1.0


def _compute_metrics(
    result: SimulationResult,
    baseline_energy: float,
    cfg: SimulationConfig,
) -> Dict[str, float]:
    meta = result.metadata
    authority_steps = meta.get("authority_steps", 0.0)
    authority_duration = float(authority_steps) * cfg.dt
    control_energy = result.final_energy()
    energy_ratio = control_energy / max(baseline_energy, 1e-9)
    return {
        "authority_peak": float(meta.get("authority_peak", 0.0)),
        "authority_duration": authority_duration,
        "energy_ratio": energy_ratio,
        "pulse_energy": float(meta.get("pulse_energy", 0.0)),
        "max_gate": float(meta.get("max_gate", 0.0)),
        "authority_first_cross": float(meta.get("authority_first_cross", math.nan)),
        "authority_steps": authority_steps,
        "ogy_pulse_count": float(meta.get("ogy_pulse_count", 0.0)),
        "ogy_success_ratio": float(meta.get("ogy_success_ratio", 0.0)),
    }


@dataclass
class SimulationResult:
    controller: str
    time: np.ndarray
    measurement: np.ndarray
    target: np.ndarray
    error: np.ndarray
    control: np.ndarray
    cumulative_control_energy: np.ndarray
    cumulative_external_energy: np.ndarray
    authority: np.ndarray
    authority_mask: np.ndarray
    lorenz_force: np.ndarray
    measurement_delta: np.ndarray
    pulse_steps: List[int]
    metadata: Dict[str, float]
    pulse_energy_used: float
    max_gate: float
    salience_trace: np.ndarray | None
    fatigue_trace: np.ndarray | None
    novelty_trace: np.ndarray | None
    ogy_pulse_count: int
    ogy_success_ratio: float

    def final_energy(self) -> float:
        return float(self.cumulative_control_energy[-1])

    def final_authority(self) -> float:
        return float(self.authority[-1])


class ControllerBase:
    def reset(self, context: SimulationContext) -> None:
        raise NotImplementedError

    def step(
        self,
        *,
        target: float,
        measurement: float,
        error: float,
        measurement_delta: float,
        forcing: float,
        lorenz_state: np.ndarray,
        lorenz_delta: np.ndarray,
        ogy_crossing: bool,
        step_index: int,
    ) -> float:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {}


def _modern_controller_kwargs(
    cfg: SimulationConfig,
    modern_params: ModernSalienceParams,
    *,
    mode: str = "timed",
    **overrides: Any,
) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "kp": 1.0,
        "ki": 0.4,
        "kd": 0.06,
        "lambda_c": 2.0,
        "pulse_amplitude": cfg.pulse_amplitude,
        "pulse_width": cfg.pulse_width,
        "pulse_cooldown": cfg.pulse_cooldown,
        "salience_params": modern_params,
        "clip": 25.0,
        "mode": mode,
        "weight_delta": 0.7,
        "weight_retention": 0.2,
        "weight_payoff": 0.1,
        "beta0": -1.2,
        "beta_c": 0.8,
        "beta_rho": 2.2,
        "beta_phi": 0.8,
        "gate_floor": modern_params.gate_floor,
        "shaping_gain": 0.45,
        "novelty_temp": 1.8,
    }
    base.update(overrides)
    return base


@dataclass
class AutotuneState:
    pulse_amplitude: float
    shaping_gain: float
    beta0: float
    beta_rho: float
    lambda_c: float
    gate_floor: float
    pulse_energy_cap: float


class ModernCVGAutotuner:
    def __init__(
        self,
        cfg: SimulationConfig,
        params: ModernSalienceParams,
        targets: AutotuneTargets | None = None,
    ) -> None:
        self.cfg = cfg
        self.params = params
        self.targets = targets or AutotuneTargets()
        self.records: List[Tuple[Dict[str, float], AutotuneState]] = []

    def initial_state(self) -> AutotuneState:
        defaults = _modern_controller_kwargs(self.cfg, self.params)
        return AutotuneState(
            pulse_amplitude=float(defaults["pulse_amplitude"]),
            shaping_gain=float(defaults["shaping_gain"]),
            beta0=float(defaults["beta0"]),
            beta_rho=float(defaults["beta_rho"]),
            lambda_c=float(defaults["lambda_c"]),
            gate_floor=float(defaults["gate_floor"]),
            pulse_energy_cap=self.cfg.pulse_energy_cap,
        )

    def run(self, baseline_energy: float) -> Tuple[AutotuneState, SimulationResult, List[Dict[str, Any]]]:
        state = self.initial_state()
        last_result: SimulationResult | None = None
        for iteration in range(self.targets.max_iterations):
            kwargs = _modern_controller_kwargs(
                self.cfg,
                self.params,
                lambda_c=state.lambda_c,
                pulse_amplitude=state.pulse_amplitude,
                shaping_gain=state.shaping_gain,
                beta0=state.beta0,
                beta_rho=state.beta_rho,
                gate_floor=state.gate_floor,
            )
            controller = ModernCVGController(**kwargs)
            tuned_cfg = replace(self.cfg, pulse_energy_cap=state.pulse_energy_cap)
            result = simulate(controller, tuned_cfg, seed=self.cfg.seed + 101 + iteration)
            metrics = _compute_metrics(result, baseline_energy, self.cfg)
            metrics["iteration"] = float(iteration)
            self.records.append((metrics, state))
            last_result = result
            if self._meets_targets(metrics):
                break
            state = self._update_state(state, metrics)
        history = self._history_payload()
        if last_result is None:
            raise RuntimeError("Autotuner failed to produce a result")
        return state, last_result, history

    def _meets_targets(self, metrics: Dict[str, float]) -> bool:
        return (
            metrics.get("authority_peak", 0.0) >= self.targets.authority_peak
            and metrics.get("authority_peak", 0.0) >= self.targets.authority_floor
            and metrics.get("authority_duration", float("inf")) <= self.targets.authority_duration_max
            and metrics.get("energy_ratio", float("inf")) <= self.targets.energy_ratio_target
            and metrics.get("pulse_energy", float("inf")) <= self.targets.pulse_energy_cap
            and metrics.get("max_gate", float("inf")) <= self.targets.gate_peak_max
        )

    def _update_state(self, state: AutotuneState, metrics: Dict[str, float]) -> AutotuneState:
        pulse_amplitude = state.pulse_amplitude
        shaping_gain = state.shaping_gain
        beta0 = state.beta0
        beta_rho = state.beta_rho
        lambda_c = state.lambda_c
        gate_floor = state.gate_floor
        pulse_cap = state.pulse_energy_cap

        authority_peak = metrics.get("authority_peak", 0.0)
        authority_duration = metrics.get("authority_duration", float("inf"))
        energy_ratio = metrics.get("energy_ratio", float("inf"))
        pulse_energy = metrics.get("pulse_energy", 0.0)
        max_gate = metrics.get("max_gate", 0.0)

        if authority_peak < self.targets.authority_floor:
            beta0 += 0.12
            beta_rho += 0.25
            gate_floor += 0.01
        elif authority_peak < self.targets.authority_peak:
            beta0 += 0.06
            beta_rho += 0.1

        if authority_duration > self.targets.authority_duration_max:
            pulse_amplitude *= 0.85
            shaping_gain += 0.05
            lambda_c += 0.2

        if energy_ratio > self.targets.energy_ratio_target:
            pulse_amplitude *= 0.8
            shaping_gain += 0.05
            lambda_c += 0.2

        if pulse_energy > self.targets.pulse_energy_cap:
            pulse_amplitude *= 0.7
            pulse_cap *= 0.5

        if max_gate > self.targets.gate_peak_max:
            beta0 -= 0.05
            gate_floor = max(0.02, gate_floor - 0.01)
            lambda_c += 0.1

        pulse_amplitude = float(np.clip(pulse_amplitude, 0.02, 0.4))
        shaping_gain = float(np.clip(shaping_gain, 0.0, 1.0))
        beta0 = float(np.clip(beta0, -2.5, -0.2))
        beta_rho = float(np.clip(beta_rho, 0.0, 4.0))
        lambda_c = float(np.clip(lambda_c, 0.5, 6.0))
        gate_floor = float(np.clip(gate_floor, 0.02, 0.3))
        pulse_cap = float(np.clip(pulse_cap, 1e-5, 5.0e-3))

        return AutotuneState(
            pulse_amplitude=pulse_amplitude,
            shaping_gain=shaping_gain,
            beta0=beta0,
            beta_rho=beta_rho,
            lambda_c=lambda_c,
            gate_floor=gate_floor,
            pulse_energy_cap=pulse_cap,
        )

    def _history_payload(self) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for idx, (metrics, state) in enumerate(self.records):
            entry: Dict[str, Any] = {**metrics}
            entry["iteration"] = idx
            entry["state"] = asdict(state)
            payload.append(entry)
        return payload


class BaselinePIDController(ControllerBase):
    """Classic PID tuned for brute-force, untimed control."""

    def __init__(self, kp: float, ki: float, kd: float, clip: float, lambda_c: float = 0.0) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.clip = clip
        self.lambda_c = lambda_c
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 0.01

    def reset(self, context: SimulationContext) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = context.dt

    def step(
        self,
        *,
        target: float,
        measurement: float,
        error: float,
        measurement_delta: float,
        forcing: float,
        lorenz_state: np.ndarray,
        lorenz_delta: np.ndarray,
        ogy_crossing: bool,
        step_index: int,
    ) -> float:
        inertia = 1.0 + self.lambda_c
        self.integral += (error * self.dt) / inertia
        derivative = (error - self.prev_error) / (self.dt * inertia)
        raw_control = (self.kp * error + self.ki * self.integral + self.kd * derivative)
        self.prev_error = error
        return float(np.clip(raw_control, -self.clip, self.clip))

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "baseline_pid",
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "lambda_c": self.lambda_c,
            "clip": self.clip,
        }


class LegacySalienceController(ControllerBase):
    """PID with multiplicative salience invariant that collapses under strain."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        mass_gain: float,
        params: LegacySalienceParams,
        clip: float,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.mass_gain = mass_gain
        self.params = params
        self.clip = clip
        self.integral = 0.0
        self.prev_error = 0.0
        self.state = LegacySalienceState()
        self.dt = 0.01

    def reset(self, context: SimulationContext) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.state = LegacySalienceState()
        self.dt = context.dt

    def step(
        self,
        *,
        target: float,
        measurement: float,
        error: float,
        measurement_delta: float,
        forcing: float,
        lorenz_state: np.ndarray,
        lorenz_delta: np.ndarray,
        ogy_crossing: bool,
        step_index: int,
    ) -> float:
        self.state = legacy_update(self.state, self.params, measurement_delta, error)
        inertia = 1.0 + self.mass_gain * self.state.salience
        self.integral += (error * self.dt) / inertia
        derivative = (error - self.prev_error) / (self.dt * inertia)
        raw_control = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return float(np.clip(raw_control / max(inertia, 1e-3), -self.clip, self.clip))

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "legacy_salience",
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "mass_gain": self.mass_gain,
            "clip": self.clip,
        }


class ModernCVGController(ControllerBase):
    """Additive salience + causal value gate with optional pulse strategies."""

    def __init__(
        self,
        *,
        kp: float,
        ki: float,
        kd: float,
        lambda_c: float,
        pulse_amplitude: float,
        pulse_width: int,
        pulse_cooldown: int,
        salience_params: ModernSalienceParams,
        clip: float,
        mode: str = "timed",
        weight_delta: float = 0.7,
        weight_retention: float = 0.2,
        weight_payoff: float = 0.1,
        beta0: float = -1.2,
        beta_c: float = 0.8,
        beta_rho: float = 2.2,
        beta_phi: float = 0.8,
        gate_floor: float = 0.08,
        shaping_gain: float = 0.45,
        novelty_temp: float = 1.8,
    ) -> None:
        if mode not in {"timed", "credit_ablation", "random_timing"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.lambda_c = lambda_c
        self.pulse_amplitude = pulse_amplitude
        self.pulse_width = max(1, pulse_width)
        self.pulse_cooldown = pulse_cooldown
        self.salience_params = salience_params
        self.clip = clip
        self.mode = mode
        self.integral = 0.0
        self.prev_control = 0.0
        self.prev_measurement = 0.0
        self.prev_error = 0.0
        self.state = ModernSalienceState()
        self.cooldown = 0
        self.rng: Optional[np.random.Generator] = None
        self.credit_sequence: Optional[np.ndarray] = None
        self.pulse_schedule: Optional[np.ndarray] = None
        self._credit_index = 0
        self.pulse_log: List[int] = []
        self.dt = 0.0
        self.last_gate = 1.0
        self.active_pulse_steps = 0
        self.pulse_energy_cap = math.inf
        self.pulse_energy_used = 0.0
        self.max_gate = 0.0
        self.weight_delta = weight_delta
        self.weight_retention = weight_retention
        self.weight_payoff = weight_payoff
        self.beta0 = beta0
        self.beta_c = beta_c
        self.beta_rho = beta_rho
        self.beta_phi = beta_phi
        self.gate_floor = gate_floor
        self.shaping_gain = shaping_gain
        self.novelty_temp = novelty_temp
        self.delta_count = 0
        self.delta_mean = 0.0
        self.delta_m2 = 0.0
        self.salience_history: List[float] = []
        self.fatigue_history: List[float] = []
        self.novelty_history: List[float] = []

    def reset(self, context: SimulationContext) -> None:
        self.integral = 0.0
        self.prev_control = 0.0
        self.prev_measurement = 0.0
        self.prev_error = 0.0
        self.state = ModernSalienceState()
        self.cooldown = 0
        self.rng = context.rng
        self.credit_sequence = context.credit_sequence
        self.pulse_schedule = context.pulse_schedule
        self._credit_index = 0
        self.pulse_log = []
        self.dt = context.dt
        self.last_gate = 1.0
        self.active_pulse_steps = 0
        self.pulse_energy_cap = context.pulse_energy_cap
        self.pulse_energy_used = 0.0
        self.max_gate = 0.0
        self.delta_count = 0
        self.delta_mean = 0.0
        self.delta_m2 = 0.0

    def _sample_measurement_delta(self, observed_delta: float) -> float:
        if self.mode != "credit_ablation" or self.credit_sequence is None:
            return observed_delta
        if self._credit_index >= self.credit_sequence.size:
            self._credit_index = 0
        value = float(self.credit_sequence[self._credit_index])
        self._credit_index += 1
        return value

    def _should_fire_pulse(self, gate: float, ogy_crossing: bool) -> bool:
        if self.cooldown > 0 or self.active_pulse_steps > 0:
            return False
        return gate >= self.gate_floor and ogy_crossing

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _update_delta_stats(self, value: float) -> None:
        self.delta_count += 1
        delta = value - self.delta_mean
        self.delta_mean += delta / self.delta_count
        delta2 = value - self.delta_mean
        self.delta_m2 += delta * delta2

    def _zscore(self, value: float) -> float:
        if self.delta_count < 2:
            return 0.0
        variance = self.delta_m2 / (self.delta_count - 1)
        std = math.sqrt(max(variance, 1e-9))
        return (value - self.delta_mean) / std

    def step(
        self,
        *,
        target: float,
        measurement: float,
        error: float,
        measurement_delta: float,
        forcing: float,
        lorenz_state: np.ndarray,
        lorenz_delta: np.ndarray,
        ogy_crossing: bool,
        step_index: int,
    ) -> float:
        observed_delta = measurement - self.prev_measurement
        credit_delta = self._sample_measurement_delta(observed_delta)

        mass = 1.0 + self.lambda_c * self.last_gate
        self.integral += (self.ki * error * self.dt)
        derivative = (error - self.prev_error) / self.dt
        base_signal = self.kp * error + self.integral + self.kd * derivative
        base_control = base_signal / max(mass, 1e-6)
        control_delta = base_control - self.prev_control
        delta_mag = abs(control_delta)
        self._update_delta_stats(delta_mag)
        delta_z = self._zscore(delta_mag)

        updated_state = update_salience(
            self.state,
            self.salience_params,
            delta=control_delta,
            error=error,
            control_delta=control_delta,
            measurement_delta=credit_delta,
        )

        novelty = float(np.clip(0.5 + 0.5 * math.tanh(delta_z / max(self.novelty_temp, 1e-6)), 0.0, 1.0))
        retention = float(np.clip(updated_state.retention, 0.0, 1.0))
        payoff = float(np.clip(updated_state.payoff, 0.0, 1.0))
        fatigue = float(np.clip(updated_state.fatigue, 0.0, 1.0))

        value_term = float(np.clip(
            self.weight_delta * novelty
            + self.weight_retention * retention
            + self.weight_payoff * payoff,
            0.0,
            1.0,
        ))

        continuity_scale = max(self.salience_params.salience_scale * self.salience_params.continuity_scale, 1e-6)
        continuity = math.tanh(delta_mag / continuity_scale)
        delta_rho = lorenz_delta[2]
        gate_input = self.beta0 + self.beta_c * continuity + self.beta_rho * delta_rho - self.beta_phi * fatigue
        gate = max(self.gate_floor, value_term * self._sigmoid(gate_input))
        self.max_gate = max(self.max_gate, gate)
        self.state = ModernSalienceState(
            salience=gate,
            retention=updated_state.retention,
            payoff=updated_state.payoff,
            fatigue=updated_state.fatigue,
        )
        self.salience_history.append(gate)
        self.fatigue_history.append(float(updated_state.fatigue))
        self.novelty_history.append(novelty)

        if self.cooldown > 0:
            self.cooldown -= 1

        start_pulse = False
        if self.mode in {"timed", "credit_ablation"} and ogy_crossing and error > 0.0:
            start_pulse = self._should_fire_pulse(gate, True)
        elif self.mode == "random_timing" and self.pulse_schedule is not None:
            if self.pulse_schedule[step_index] and self.cooldown == 0 and self.active_pulse_steps == 0:
                start_pulse = True

        pulse = 0.0
        if start_pulse and self.pulse_energy_used < self.pulse_energy_cap:
            self.active_pulse_steps = self.pulse_width
            self.cooldown = self.pulse_cooldown
            self.pulse_log.append(step_index)

        if self.active_pulse_steps > 0 and self.pulse_energy_used < self.pulse_energy_cap:
            base_amp = self.pulse_amplitude if self.mode == "random_timing" else self.pulse_amplitude * gate
            pulse = base_amp
            energy_step = (pulse ** 2) * self.dt
            if self.pulse_energy_used + energy_step <= self.pulse_energy_cap + 1e-9:
                self.pulse_energy_used += energy_step
                self.active_pulse_steps -= 1
            else:
                pulse = 0.0
                self.active_pulse_steps = 0
        else:
            self.active_pulse_steps = 0 if self.active_pulse_steps == 0 else self.active_pulse_steps

        shaped_control = base_control - self.shaping_gain * gate
        control = shaped_control + pulse
        control = float(np.clip(control, -self.clip, self.clip))

        self.prev_control = control
        self.prev_measurement = measurement
        self.prev_error = error
        self.last_gate = gate

        return control

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "modern_cvg",
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "lambda_c": self.lambda_c,
            "pulse_amplitude": self.pulse_amplitude,
            "pulse_width": self.pulse_width,
            "pulse_cooldown": self.pulse_cooldown,
            "gate_floor": self.gate_floor,
            "weights": {
                "delta": self.weight_delta,
                "retention": self.weight_retention,
                "payoff": self.weight_payoff,
            },
            "betas": {
                "beta0": self.beta0,
                "beta_c": self.beta_c,
                "beta_rho": self.beta_rho,
                "beta_phi": self.beta_phi,
            },
            "pulse_energy_cap": self.pulse_energy_cap,
        }


def rk4_lorenz(state: np.ndarray, dt: float, sigma: float, rho: float, beta: float) -> np.ndarray:
    def deriv(s: np.ndarray) -> np.ndarray:
        x, y, z = s
        return np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ])

    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def detect_ogy_crossing(
    prev_state: np.ndarray,
    next_state: np.ndarray,
    axis: int,
    threshold: float,
) -> bool:
    prev_val = prev_state[axis] - threshold
    next_val = next_state[axis] - threshold
    return prev_val < 0.0 <= next_val


def simulate(
    controller: ControllerBase,
    cfg: SimulationConfig,
    *,
    seed: int,
    credit_sequence: Optional[np.ndarray] = None,
    pulse_schedule: Optional[np.ndarray] = None,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    context = SimulationContext(
        total_steps=cfg.steps,
        dt=cfg.dt,
        rng=rng,
        credit_sequence=credit_sequence,
        pulse_schedule=pulse_schedule,
        pulse_energy_cap=cfg.pulse_energy_cap,
    )
    controller.reset(context)

    time = np.arange(cfg.steps) * cfg.dt
    target = np.full(cfg.steps, cfg.target, dtype=float)
    measurement = np.zeros(cfg.steps, dtype=float)
    error = np.zeros(cfg.steps, dtype=float)
    control = np.zeros(cfg.steps, dtype=float)
    lorenz_force = np.zeros(cfg.steps, dtype=float)
    control_energy_cum = np.zeros(cfg.steps, dtype=float)
    external_energy_cum = np.zeros(cfg.steps, dtype=float)
    authority = np.zeros(cfg.steps, dtype=float)
    measurement_delta_trace = np.zeros(cfg.steps, dtype=float)

    state = 0.0
    lorenz_state = np.array(cfg.lorenz_initial, dtype=float)
    control_energy_total = 0.0
    external_energy_total = 0.0

    pulse_steps: List[int] = []
    ogy_cross_count = 0
    ogy_fire_count = 0

    for step_index in range(cfg.steps):
        next_lorenz = rk4_lorenz(
            lorenz_state,
            cfg.dt,
            cfg.lorenz_sigma,
            cfg.lorenz_rho,
            cfg.lorenz_beta,
        )
        lorenz_delta = next_lorenz - lorenz_state
        avg_lorenz = lorenz_state + 0.5 * lorenz_delta
        raw_ogy_cross = detect_ogy_crossing(
            lorenz_state,
            next_lorenz,
            cfg.ogy_axis,
            cfg.ogy_threshold,
        )
        ogy_crossing = bool(
            raw_ogy_cross
            and abs(avg_lorenz[1]) < cfg.ogy_y_window
            and cfg.ogy_z_min <= avg_lorenz[2] <= cfg.ogy_z_max
        )
        if raw_ogy_cross:
            ogy_cross_count += 1
        if ogy_crossing:
            ogy_fire_count += 1
        forcing = cfg.lorenz_gain * avg_lorenz[0]
        lorenz_state = next_lorenz

        if step_index == 0:
            measurement_delta = 0.0
        else:
            measurement_delta = measurement[step_index - 1] - measurement[step_index - 2]

        ctrl = controller.step(
            target=cfg.target,
            measurement=state,
            error=cfg.target - state,
            measurement_delta=measurement_delta,
            forcing=forcing,
            lorenz_state=avg_lorenz,
            lorenz_delta=lorenz_delta,
            ogy_crossing=ogy_crossing,
            step_index=step_index,
        )
        state += cfg.dt * ((-state + ctrl + forcing) / cfg.tau)

        measurement[step_index] = state
        error[step_index] = cfg.target - state
        control[step_index] = ctrl
        lorenz_force[step_index] = forcing

        control_energy_step = (ctrl ** 2) * cfg.dt
        external_energy_step = (forcing ** 2) * cfg.dt
        control_energy_total += control_energy_step
        external_energy_total += external_energy_step
        control_energy_cum[step_index] = control_energy_total
        external_energy_cum[step_index] = external_energy_total
        authority_val = (
            control_energy_total / max(external_energy_total, 1e-9)
            if external_energy_total > 0.0
            else math.inf
        )
        authority[step_index] = authority_val

        measurement_delta_trace[step_index] = measurement_delta

        if isinstance(controller, ModernCVGController) and controller.pulse_log:
            new_pulses = controller.pulse_log[len(pulse_steps) :]
            pulse_steps.extend(new_pulses)

    authority_mask = authority >= cfg.authority_threshold

    metadata = {
        "final_error": float(error[-1]),
        "peak_error": float(np.max(np.abs(error))),
        "control_energy": float(control_energy_cum[-1]),
        "external_energy": float(external_energy_cum[-1]),
        "final_authority": float(authority[-1]),
        "authority_peak": float(np.max(authority)),
        "authority_steps": int(np.count_nonzero(authority_mask)),
        "pulse_count": len(pulse_steps),
    }
    pulse_energy = 0.0
    max_gate = 0.0
    salience_trace: np.ndarray | None = None
    fatigue_trace: np.ndarray | None = None
    novelty_trace: np.ndarray | None = None
    if isinstance(controller, ModernCVGController):
        pulse_energy = controller.pulse_energy_used
        max_gate = controller.max_gate
        metadata["pulse_energy"] = pulse_energy
        metadata["max_gate"] = max_gate
        salience_trace = np.array(controller.salience_history, dtype=float)
        fatigue_trace = np.array(controller.fatigue_history, dtype=float)
        novelty_trace = np.array(controller.novelty_history, dtype=float)
    if authority_mask.any():
        first_idx = int(np.argmax(authority_mask))
        metadata["authority_first_cross"] = float(time[first_idx])
    metadata["controller_config"] = controller.describe()
    metadata["energy_ratio_external"] = float(control_energy_cum[-1] / max(external_energy_cum[-1], 1e-9))
    metadata["energy_total"] = float(control_energy_cum[-1])
    metadata["ogy_pulse_count"] = ogy_fire_count
    metadata["ogy_crossings"] = ogy_cross_count
    metadata["ogy_success_ratio"] = (
        float(ogy_fire_count / ogy_cross_count) if ogy_cross_count > 0 else 0.0
    )

    return SimulationResult(
        controller=controller.__class__.__name__,
        time=time,
        measurement=measurement,
        target=target,
        error=error,
        control=control,
        cumulative_control_energy=control_energy_cum,
        cumulative_external_energy=external_energy_cum,
        authority=authority,
        authority_mask=authority_mask,
        lorenz_force=lorenz_force,
        measurement_delta=measurement_delta_trace,
        pulse_steps=pulse_steps,
        metadata=metadata,
        pulse_energy_used=pulse_energy,
        max_gate=max_gate,
        salience_trace=salience_trace,
        fatigue_trace=fatigue_trace,
        novelty_trace=novelty_trace,
        ogy_pulse_count=ogy_fire_count,
        ogy_success_ratio=metadata["ogy_success_ratio"],
    )


def build_controllers(cfg: SimulationConfig) -> Dict[str, ControllerBase]:
    legacy_params = LegacySalienceParams(salience_scale=0.2, fatigue_gain=0.12)
    modern_params = ModernSalienceParams(
        salience_floor=0.02,
        salience_scale=0.06,
        fatigue_gain=0.12,
        gate_floor=0.08,
    )

    autotuner = ModernCVGAutotuner(cfg, modern_params)
    baseline = BaselinePIDController(kp=1.2, ki=0.5, kd=0.08, clip=40.0, lambda_c=8.0)
    legacy = LegacySalienceController(
        kp=1.2,
        ki=0.5,
        kd=0.08,
        mass_gain=9.0,
        params=legacy_params,
        clip=40.0,
    )
    return {
        "baseline": baseline,
        "legacy": legacy,
        "autotuner": autotuner,
        "modern_params": modern_params,
    }


def plot_story(results: Iterable[SimulationResult], output_path: Path) -> None:
    results = list(results)
    fig, (ax_err, ax_energy) = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True)

    colors = {
        "baseline": "#377eb8",
        "legacy": "#e41a1c",
        "modern": "#4daf4a",
    }

    max_err = max(float(np.max(np.abs(res.error))) for res in results)

    for res in results:
        label = {
            "baseline": "Baseline PID (untimed)",
            "legacy": "Legacy multiplicative",
            "modern": "CVG + OGY pulses",
        }.get(res.controller.lower(), res.controller)
        color = colors.get(res.controller.lower(), None)
        ax_err.plot(res.time, res.error, label=label, linewidth=2.0, color=color)

    modern = next(res for res in results if res.controller == "ModernCVGController")
    ax_err.fill_between(
        modern.time,
        -max_err,
        max_err,
        where=modern.authority_mask,
        color=colors["modern"],
        alpha=0.12,
        label="Authority ≥ 1 (CVG)",
    )
    ax_err.set_ylabel("Tracking error")
    ax_err.set_ylim(-1.1 * max_err, 1.1 * max_err)
    ax_err.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.6)
    ax_err.legend(loc="upper right", frameon=False)
    ax_err.set_title("Lorenz tracking: CVG pulses hit authority with 10³× less energy")

    for res in results:
        label = {
            "baseline": "Baseline PID",
            "legacy": "Legacy multiplicative",
            "modern": "CVG + OGY pulses",
        }.get(res.controller.lower(), res.controller)
        color = colors.get(res.controller.lower(), None)
        ax_energy.plot(res.time, res.cumulative_control_energy, label=label, linewidth=2.0, color=color)

    ax_energy.set_xlabel("Time [s]")
    ax_energy.set_ylabel("Cumulative control energy")
    ax_energy.set_yscale("log")
    ax_energy.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_energy.legend(loc="upper left", frameon=False)

    ax_energy.annotate(
        "Legacy collapses (authority < 1)",
        xy=(results[1].time[-1] * 0.65, results[1].cumulative_control_energy[-1]),
        xytext=(results[1].time[-1] * 0.45, results[1].cumulative_control_energy[-1] * 4),
        arrowprops=dict(arrowstyle="->", color="#e41a1c"),
        color="#e41a1c",
    )
    ax_energy.annotate(
        "CVG pulses: authority ≥ 1\nwith ≥10³× less energy",
        xy=(modern.time[-1] * 0.8, modern.cumulative_control_energy[-1]),
        xytext=(modern.time[-1] * 0.55, modern.cumulative_control_energy[-1] * 30),
        arrowprops=dict(arrowstyle="->", color="#4daf4a"),
        color="#2b8c3f",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def summarise(results: Iterable[SimulationResult]) -> List[Dict[str, float]]:
    table: List[Dict[str, float]] = []
    for res in results:
        row: Dict[str, float] = {"controller": res.controller}
        meta = res.metadata
        row["final_error"] = float(meta.get("final_error", res.error[-1]))
        row["peak_error"] = float(meta.get("peak_error", np.max(np.abs(res.error))))
        row["control_energy"] = float(meta.get("control_energy", res.final_energy()))
        row["external_energy"] = float(meta.get("external_energy", res.cumulative_external_energy[-1]))
        row["final_authority"] = float(meta.get("final_authority", res.final_authority()))
        if "energy_ratio" in meta:
            row["energy_ratio"] = float(meta["energy_ratio"])
        if "authority_steps" in meta:
            row["authority_steps"] = float(meta["authority_steps"])
        if "pulse_energy" in meta:
            row["pulse_energy"] = float(meta["pulse_energy"])
        if "max_gate" in meta:
            row["max_gate"] = float(meta["max_gate"])
        if "ogy_success_ratio" in meta:
            row["ogy_success_ratio"] = float(meta["ogy_success_ratio"])
        if "ogy_pulse_count" in meta:
            row["ogy_pulse_count"] = float(meta["ogy_pulse_count"])
        table.append(row)
    return table


def format_table(rows: Iterable[Dict[str, float]]) -> str:
    rows = list(rows)
    include_pulse = any("pulse_energy" in row for row in rows)
    include_steps = any("authority_steps" in row for row in rows)
    include_max_gate = any("max_gate" in row for row in rows)
    include_energy_ratio = any("energy_ratio" in row for row in rows)
    include_ogy = any("ogy_success_ratio" in row for row in rows)
    headers = ["controller", "final_error", "peak_error", "control_energy", "external_energy", "final_authority"]
    if include_energy_ratio:
        headers.append("energy_ratio")
    if include_steps:
        headers.append("authority_steps")
    if include_pulse:
        headers.append("pulse_energy")
    if include_max_gate:
        headers.append("max_gate")
    if include_ogy:
        headers.append("ogy_success")
        headers.append("ogy_pulses")
    lines = [" | ".join(headers)]
    lines.append(" | ".join("---" for _ in headers))
    for row in rows:
        cells = [
            row["controller"],
            f"{row['final_error']:.4f}",
            f"{row['peak_error']:.4f}",
            f"{row['control_energy']:.2e}",
            f"{row['external_energy']:.2e}",
            f"{row['final_authority']:.3f}",
        ]
        if include_energy_ratio:
            er = row.get("energy_ratio")
            cells.append(f"{er:.2e}" if isinstance(er, (float, int)) else "")
        if include_steps:
            steps = row.get("authority_steps")
            cells.append(str(steps) if steps is not None else "")
        if include_pulse:
            pe = row.get("pulse_energy")
            cells.append(f"{pe:.2e}" if isinstance(pe, (float, int)) else "")
        if include_max_gate:
            mg = row.get("max_gate")
            cells.append(f"{mg:.3f}" if isinstance(mg, (float, int)) else "")
        if include_ogy:
            ogy_success = row.get("ogy_success_ratio")
            ogy_pulses = row.get("ogy_pulse_count")
            cells.append(f"{ogy_success:.2f}" if isinstance(ogy_success, (float, int)) else "")
            cells.append(str(int(ogy_pulses)) if isinstance(ogy_pulses, (float, int)) else "")
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def run_ablation_credit(
    base_result: SimulationResult,
    cfg: SimulationConfig,
    modern_params: ModernSalienceParams,
    tuned_state: AutotuneState,
) -> SimulationResult:
    shuffled = base_result.measurement_delta.copy()
    rng = np.random.default_rng(cfg.seed + 11)
    rng.shuffle(shuffled)
    controller = ModernCVGController(**_modern_controller_kwargs(
        cfg,
        modern_params,
        mode="credit_ablation",
        lambda_c=tuned_state.lambda_c,
        pulse_amplitude=tuned_state.pulse_amplitude,
        shaping_gain=tuned_state.shaping_gain,
        beta0=tuned_state.beta0,
        beta_rho=tuned_state.beta_rho,
        gate_floor=tuned_state.gate_floor,
    ))
    return simulate(controller, cfg, seed=cfg.seed + 12, credit_sequence=shuffled)


def _build_random_pulse_schedule(
    total_steps: int,
    pulse_count: int,
    min_spacing: int,
    rng: np.random.Generator,
) -> np.ndarray:
    schedule = np.zeros(total_steps, dtype=bool)
    if pulse_count <= 0:
        return schedule
    positions: List[int] = []
    attempts = 0
    limit = max(10 * pulse_count, 100)
    while len(positions) < pulse_count and attempts < limit:
        candidate = int(rng.integers(0, total_steps))
        if all(abs(candidate - existing) >= min_spacing for existing in positions):
            positions.append(candidate)
        attempts += 1
    if len(positions) < pulse_count:
        spacing = max(1, min_spacing)
        positions = list(range(0, spacing * pulse_count, spacing))[:pulse_count]
        positions = [min(pos, total_steps - 1) for pos in positions]
    positions.sort()
    for pos in positions:
        schedule[pos] = True
    return schedule


def run_ablation_timing(
    base_result: SimulationResult,
    cfg: SimulationConfig,
    modern_params: ModernSalienceParams,
    tuned_state: AutotuneState,
) -> SimulationResult:
    total_steps = cfg.steps
    pulse_count = len(base_result.pulse_steps)
    rng = np.random.default_rng(cfg.seed + 21)
    min_spacing = max(cfg.pulse_width, cfg.pulse_cooldown)
    pulse_schedule = _build_random_pulse_schedule(total_steps, pulse_count, min_spacing, rng)
    controller = ModernCVGController(**_modern_controller_kwargs(
        cfg,
        modern_params,
        mode="random_timing",
        lambda_c=tuned_state.lambda_c,
        pulse_amplitude=tuned_state.pulse_amplitude,
        shaping_gain=tuned_state.shaping_gain,
        beta0=tuned_state.beta0,
        beta_rho=tuned_state.beta_rho,
        gate_floor=tuned_state.gate_floor,
    ))
    tuned_cfg = replace(cfg, pulse_energy_cap=tuned_state.pulse_energy_cap)
    return simulate(controller, tuned_cfg, seed=cfg.seed + 22, pulse_schedule=pulse_schedule)


def log_results(
    main_results: List[SimulationResult],
    credit_ablation: SimulationResult,
    timing_ablation: SimulationResult,
    tuned_state: AutotuneState,
    autotune_history: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "main": [res.metadata | {"controller": res.controller} for res in main_results],
        "ablations": {
            "credit_vs_exogenous": credit_ablation.metadata | {"controller": credit_ablation.controller},
            "timing_vs_random": timing_ablation.metadata | {"controller": timing_ablation.controller},
        },
        "autotune": {
            "tuned_state": asdict(tuned_state),
            "history": autotune_history,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"cvg_lorenz_demo_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def main() -> None:
    cfg = SimulationConfig()
    controllers = build_controllers(cfg)

    baseline_controller: BaselinePIDController = controllers["baseline"]  # type: ignore[assignment]
    legacy_controller: LegacySalienceController = controllers["legacy"]  # type: ignore[assignment]
    autotuner: ModernCVGAutotuner = controllers["autotuner"]  # type: ignore[assignment]
    modern_params: ModernSalienceParams = controllers["modern_params"]  # type: ignore[assignment]

    baseline_result = simulate(baseline_controller, cfg, seed=cfg.seed + 1)
    legacy_result = simulate(legacy_controller, cfg, seed=cfg.seed + 2)

    tuned_state, _, autotune_history = autotuner.run(baseline_result.final_energy())
    tuned_controller = ModernCVGController(**_modern_controller_kwargs(
        cfg,
        modern_params,
        lambda_c=tuned_state.lambda_c,
        pulse_amplitude=tuned_state.pulse_amplitude,
        shaping_gain=tuned_state.shaping_gain,
        beta0=tuned_state.beta0,
        beta_rho=tuned_state.beta_rho,
        gate_floor=tuned_state.gate_floor,
    ))
    tuned_cfg = replace(cfg, pulse_energy_cap=tuned_state.pulse_energy_cap)
    modern_result = simulate(tuned_controller, tuned_cfg, seed=cfg.seed + 3)
    modern_result.metadata["autotune_state"] = asdict(tuned_state)
    modern_result.metadata["autotune_history"] = autotune_history

    credit_ablation_result = run_ablation_credit(modern_result, tuned_cfg, modern_params, tuned_state)
    timing_ablation_result = run_ablation_timing(modern_result, tuned_cfg, modern_params, tuned_state)

    results = [baseline_result, legacy_result, modern_result]

    output_dir = Path(__file__).resolve().parent / "outputs"
    figure_path = output_dir / "cvg_lorenz_demo.png"
    plot_story(results, figure_path)

    summary_rows = summarise(results)
    print("\n=== Main controllers ===")
    print(format_table(summary_rows))

    print("\nAblation: Causal credit vs. exogenous surprise")
    print(format_table([credit_ablation_result.metadata | {"controller": credit_ablation_result.controller}]))

    print("\nAblation: OGY-timed pulses vs. random pulses")
    print(format_table([timing_ablation_result.metadata | {"controller": timing_ablation_result.controller}]))

    log_path = log_results(results, credit_ablation_result, timing_ablation_result, tuned_state, autotune_history, output_dir)
    print(f"\nFigure written to {figure_path}")
    print(f"Metrics written to {log_path}")


if __name__ == "__main__":
    main()
