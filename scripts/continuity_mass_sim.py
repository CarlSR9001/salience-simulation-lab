"""Experiment A & P: Continuity-taxed controller mass and stress sweeps."""

from __future__ import annotations

import argparse
import json
import math
import uuid
from collections import deque
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from tabulate import tabulate

CONTROL_AUTHORITY_THRESHOLD = 1.2

ARTIFACT_DIR = Path("artifacts/mass_sweep")


@dataclass
class PlantConfig:
    dt: float = 0.01
    horizon: float = 5.0
    tau: float = 0.1
    noise_sigma: float = 0.0
    noise_delay: float = 0.0
    breath_period: float = 0.0
    breath_in_fraction: float = 0.0
    breath_in_scale: float = 0.1
    breath_out_scale: float = 1.0
    breath_pulse_count: int = 1
    breath_pulse_width: float = 0.5
    adaptive_breathing: bool = False
    adapt_authority_threshold: float = 1.1
    adapt_salience_threshold: float = 0.55
    adapt_scale_min: float = 0.05
    adapt_scale_max: float = 1.0
    adapt_step_gain: float = 0.05
    salience_breathing: bool = False
    salience_breath_floor: float = 0.55
    salience_breath_gain: float = 3.0
    salience_breath_slope_gain: float = 0.25
    salience_breath_hope_gain: float = 0.2
    salience_breath_smoothing: float = 0.1
    salience_breath_recovery_scale: float = 0.15
    salience_breath_attack_scale: float = 1.0
    salience_osc_smoothing: float = 0.2
    salience_osc_flip_gain: float = 0.5
    salience_osc_variance_gain: float = 0.1
    salience_lambda_gain: float = 0.0
    salience_gate_bias_gain: float = 0.0
    salience_gate_bias_damp: float = 0.0
    salience_gate_floor_gain: float = 0.0
    salience_assist_amp_floor: float = 1.0
    salience_assist_amp_gain: float = 0.0
    salience_assist_amp_max: float = 1.0
    salience_assist_cooldown_gain: float = 0.0
    lorenz_gate_enabled: bool = False
    lorenz_gate_threshold: float = 1.0
    lorenz_gate_release_steps: int = 10
    lorenz_ramp_steps: int = 20
    lorenz_hot_tag_enabled: bool = False
    lorenz_hot_tag_window: int = 15
    lorenz_hot_tag_slope_threshold: float = 0.25
    lorenz_hot_tag_pulse_steps: int = 50
    lorenz_hot_tag_cooldown_steps: int = 60
    lorenz_hot_tag_control_boost: float = 1.3
    hot_tag_heat_steps: int = 200
    hot_tag_heat_authority: float = 0.8
    hot_tag_near_tag_threshold: float = 0.95
    hot_tag_near_tag_upper: float = 1.1
    hot_tag_near_tag_attempts: int = 3
    hot_tag_partner_lambda_scale: float = 0.5
    hot_tag_partner_mass_scale: float = 0.15
    hot_tag_partner_salience_floor: float = 0.6
    hot_tag_assist_gain: float = 0.0
    hot_tag_auto_start_steps: int = 0
    assist_poincare_enabled: bool = False
    assist_poincare_axis: int = 0
    assist_poincare_threshold: float = 0.0
    assist_poincare_direction: int = 1
    assist_poincare_window: int = 0
    hope_spot_lower: float = 0.7
    hope_spot_upper: float = 0.95
    hope_spot_min_slope: float = 0.0
    authority_break_window: int = 10
    assist_amplitude: float = 0.0
    assist_pulse_width: int = 1
    assist_cooldown_steps: int = 15
    assist_salience_floor: float = 0.4
    assist_authority_slope_min: float = 0.0
    salience_channels: int = 2
    lorenz_gain: float = 0.0
    lorenz_sigma: float = 10.0
    lorenz_rho: float = 28.0
    lorenz_beta: float = 8.0 / 3.0
    lorenz_initial: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class ControllerConfig:
    kp: float = 3.0
    ki: float = 1.6
    kd: float = 0.12
    lambda_c: float = 0.0
    salience_scale: float = 0.05
    fatigue_decay: float = 0.9
    fatigue_gain: float = 0.1
    smoothing_alpha: float = 0.24
    state_smoothing_alpha: float = 0.12
    integral_clip: float = 22.0
    mass_scale: float = 0.3
    salience_channels: int = 2
    salience_floor: float = 0.0
    salience_weight_novelty: float = 0.6
    salience_weight_retention: float = 0.25
    salience_weight_payoff: float = 0.15
    salience_gate_bias: float = 0.5
    salience_gate_continuity: float = 4.0
    salience_gate_fatigue: float = 3.0
    salience_gate_temp: float = 0.06
    salience_gate_floor: float = 0.1
    salience_credit_gain: float = 6.0
    salience_credit_bias: float = 0.4
    salience_credit_floor: float = 0.1
    salience_credit_alpha: float = 0.65
    salience_continuity_scale: float = 1.0


class ContinuityPID:
    def __init__(self, cfg: ControllerConfig, dt: float) -> None:
        self.cfg = cfg
        self.dt = dt
        self.salience_floor = max(0.0, cfg.salience_floor)
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.derivative_est = 0.0
        self.prev_error = 0.0
        channels = max(1, self.cfg.salience_channels)
        self.salience = np.ones(channels, dtype=float)
        if self.salience_floor > 0.0:
            self.salience = np.maximum(self.salience, self.salience_floor)
        self.retention = np.full(channels, 0.9, dtype=float)
        self.payoff = np.full(channels, 0.9, dtype=float)
        self.fatigue = np.zeros(channels, dtype=float)
        self.prev_control = 0.0
        self.prev_measurement = 0.0
        self.state_estimate = 0.0

    @staticmethod
    def _sigmoid(x: float, temp: float = 1.0) -> float:
        scale = max(temp, 1e-6)
        z = x / scale
        if z >= 0.0:
            exp_neg = math.exp(-z)
            return 1.0 / (1.0 + exp_neg)
        exp_pos = math.exp(z)
        return exp_pos / (1.0 + exp_pos)

    def _compute_credit(self, control_delta: float, measurement_delta: float) -> float:
        weighted_measurement = self.cfg.salience_credit_alpha * measurement_delta
        denom = control_delta + weighted_measurement + 1e-9
        ratio = 0.0 if denom <= 0.0 else control_delta / denom
        ratio = max(0.0, min(1.0, ratio))
        credit_input = self.cfg.salience_credit_bias + self.cfg.salience_credit_gain * (ratio - 0.5)
        credit = self._sigmoid(credit_input, self.cfg.salience_gate_temp)
        return max(self.cfg.salience_credit_floor, credit)

    def _update_salience(self, idx: int, delta: float, error: float, credit: float) -> None:
        scale = max(self.cfg.salience_scale, 1e-6)
        delta_mag = abs(delta)

        novelty_decay = math.exp(-delta_mag / scale)
        novelty_gain = 1.0 - novelty_decay

        self.retention[idx] = 0.92 * self.retention[idx] + 0.08 * novelty_decay
        payoff_sample = math.exp(-abs(error))
        self.payoff[idx] = 0.9 * self.payoff[idx] + 0.1 * payoff_sample

        self.fatigue[idx] = (
            self.cfg.fatigue_decay * self.fatigue[idx]
            + self.cfg.fatigue_gain * delta_mag / scale
        )
        phi = min(self.fatigue[idx], 0.95)

        credit_factor = max(self.cfg.salience_credit_floor, credit)

        retention_term = float(np.clip(self.retention[idx], 0.2, 1.1))
        payoff_term = float(np.clip(self.payoff[idx], 0.2, 1.1))
        novelty_term = novelty_gain * credit_factor

        value = (
            self.cfg.salience_weight_novelty * novelty_term
            + self.cfg.salience_weight_retention * retention_term
            + self.cfg.salience_weight_payoff * payoff_term
        )

        continuity_scale = max(self.cfg.salience_continuity_scale, 1e-6)
        strain = math.tanh(delta_mag / (scale * continuity_scale))
        gate_input = (
            self.cfg.salience_gate_bias
            + self.cfg.salience_gate_continuity * strain
            - self.cfg.salience_gate_fatigue * phi
        )
        gate = max(self.cfg.salience_gate_floor, self._sigmoid(gate_input, self.cfg.salience_gate_temp))

        salience = value * gate
        lower_bound = max(self.salience_floor, 1e-3)
        self.salience[idx] = float(np.clip(salience, lower_bound, 1.8))

    def step(self, target: float, measurement: float) -> Dict[str, float]:
        error = target - measurement

        int_mass = 1.0 + self.cfg.lambda_c * self.cfg.mass_scale * self.salience[0]
        raw_int_delta = error * self.dt
        applied_int_delta = raw_int_delta / int_mass
        self.integral = float(
            np.clip(self.integral + applied_int_delta, -self.cfg.integral_clip, self.cfg.integral_clip)
        )

        raw_derivative = (error - self.prev_error) / self.dt
        deriv_delta = raw_derivative - self.derivative_est
        deriv_mass = 1.0 + self.cfg.lambda_c * self.cfg.mass_scale * self.salience[1]
        self.derivative_est += self.cfg.smoothing_alpha * (deriv_delta / deriv_mass)

        control = (
            self.cfg.kp * error
            + self.cfg.ki * self.integral
            + self.cfg.kd * self.derivative_est
        )

        if self.salience.size >= 3:
            pass

        control_delta = control - self.prev_control
        if self.salience.size >= 4:
            pass

        estimate_prev = self.state_estimate
        self.state_estimate += self.cfg.state_smoothing_alpha * (measurement - self.state_estimate)
        if self.salience.size >= 5:
            noise_delta = measurement - estimate_prev
        else:
            noise_delta = 0.0

        control_delta_abs = abs(control_delta)
        measurement_delta_abs = abs(measurement - self.prev_measurement)
        credit = self._compute_credit(control_delta_abs, measurement_delta_abs)

        self._update_salience(0, applied_int_delta, error, credit)
        self._update_salience(1, deriv_delta, error, credit)

        if self.salience.size >= 3:
            self._update_salience(2, error, error, credit)

        if self.salience.size >= 4:
            self._update_salience(3, control_delta, error, credit)

        if self.salience.size >= 5:
            self._update_salience(4, noise_delta, error, credit)

        self.prev_error = error
        self.prev_control = control
        self.prev_measurement = measurement
        return {
            "control": float(control),
            "salience_mean": float(np.mean(self.salience)),
            "salience_min": float(np.min(self.salience)),
            "salience_channels": self.salience.tolist(),
        }


def simulate(controller_cfg: ControllerConfig, plant_cfg: PlantConfig, seed: int | None = None) -> Dict[str, float]:
    main_controller = ContinuityPID(controller_cfg, plant_cfg.dt)
    main_controller.reset()
    partner_controller: ContinuityPID | None = None
    partner_cfg: ControllerConfig | None = None
    if plant_cfg.lorenz_hot_tag_enabled:
        partner_cfg = replace(
            controller_cfg,
            lambda_c=controller_cfg.lambda_c * plant_cfg.hot_tag_partner_lambda_scale,
            mass_scale=controller_cfg.mass_scale * plant_cfg.hot_tag_partner_mass_scale,
            salience_floor=max(controller_cfg.salience_floor, plant_cfg.hot_tag_partner_salience_floor),
        )
        partner_controller = ContinuityPID(partner_cfg, plant_cfg.dt)
        partner_controller.reset()

    active_controller = main_controller
    using_partner = False

    steps = int(plant_cfg.horizon / plant_cfg.dt)
    target = np.ones(steps, dtype=float)
    state = 0.0

    outputs = np.zeros(steps, dtype=float)
    salience_mean_trace = np.zeros(steps, dtype=float)
    salience_min_trace = np.zeros(steps, dtype=float)
    salience_channel_trace = np.zeros((steps, controller_cfg.salience_channels), dtype=float)
    control_energy = 0.0
    noise_energy = 0.0
    lorenz_energy = 0.0
    assist_energy = 0.0

    control_energy_trace = np.zeros(steps, dtype=float)
    noise_energy_trace = np.zeros(steps, dtype=float)
    lorenz_energy_trace = np.zeros(steps, dtype=float)
    external_energy_trace = np.zeros(steps, dtype=float)
    adaptive_scale_trace = np.ones(steps, dtype=float)
    adaptive_scale_min = float("inf")
    adaptive_scale_max = 0.0

    rng = np.random.default_rng(seed)
    lorenz_state = np.array(plant_cfg.lorenz_initial, dtype=float)
    adaptive_scale = 1.0
    last_authority_ratio = float("inf")
    last_salience = 1.0
    control_energy_cum = 0.0
    external_energy_cum = 0.0
    authority_history = [float("inf")] * max(plant_cfg.lorenz_gate_release_steps, plant_cfg.lorenz_hot_tag_window)
    lorenz_active_trace = np.zeros(steps, dtype=float)
    lorenz_scale_trace = np.zeros(steps, dtype=float)
    lorenz_scale = 1.0 if not plant_cfg.lorenz_gate_enabled else 0.0
    hot_tag_active_trace = np.zeros(steps, dtype=float)
    partner_active_trace = np.zeros(steps, dtype=float)
    assist_energy_trace = np.zeros(steps, dtype=float)
    assist_window_trace = np.zeros(steps, dtype=float)
    assist_pulse_active_trace = np.zeros(steps, dtype=float)
    assist_trigger_trace = np.zeros(steps, dtype=float)
    hot_tag_trigger_trace = np.zeros(steps, dtype=float)
    hope_flag_trace = np.zeros(steps, dtype=float)
    heat_flag_trace = np.zeros(steps, dtype=float)
    salience_charge_trace = np.zeros(steps, dtype=float)
    salience_flip_trace = np.zeros(steps, dtype=float)
    salience_modulated_lambda = np.zeros(steps, dtype=float)
    assist_amplitude_trace = np.zeros(steps, dtype=float)
    assist_cooldown_trace = np.zeros(steps, dtype=float)
    hot_tag_steps_remaining = max(0, plant_cfg.hot_tag_auto_start_steps)
    hot_tag_cooldown = 0
    hot_tag_trigger_count = 0
    heat_counter = 0
    heat_peak = 0
    near_tag_attempts = 0
    near_tag_total = 0
    near_tag_active = False
    partner_switch_count = 0
    assist_window_remaining = 0
    poincare_axis = int(np.clip(plant_cfg.assist_poincare_axis, 0, 2)) if plant_cfg.assist_poincare_enabled else 0
    poincare_direction = 1 if not plant_cfg.assist_poincare_enabled or plant_cfg.assist_poincare_direction >= 0 else -1
    assist_trigger_count = 0
    assist_pulse_steps = 0
    assist_cooldown = 0
    assist_pulse_count = 0
    assist_pulse_pending = False
    prev_authority_step_ratio = 0.0
    authority_ratio_trace = np.zeros(steps, dtype=float)
    authority_slope_trace = np.zeros(steps, dtype=float)
    authority_window_ratio_trace = np.full(steps, float("nan"))
    hope_spot_indices: list[int] = []
    hope_spot_salience_sum = 0.0
    hope_spot_step_count = 0
    hope_spot_event_count = 0
    hope_spot_assist_count = 0
    hope_spot_hot_tag_count = 0
    first_hope_index: int | None = None
    in_hope = False
    assist_trigger_indices: list[int] = []
    heat_salience_sum = 0.0
    heat_step_count = 0
    assist_crossing_count = 0
    authority_window_queue: deque[float] = deque(maxlen=max(1, plant_cfg.authority_break_window))
    first_break_index: int | None = None
    authority_break_threshold = 1.0
    salience_charge = 0.0
    salience_flip_score = 0.0
    prev_positive_slope = None

    if hot_tag_steps_remaining > 0 and partner_controller is not None:
        active_controller = partner_controller
        using_partner = True
        partner_switch_count += 1

    for i in range(steps):
        if assist_cooldown > 0:
            assist_cooldown -= 1
        outputs[i] = state
        result = active_controller.step(target[i], state)
        control = result["control"]
        hot_tag_active = plant_cfg.lorenz_hot_tag_enabled and hot_tag_steps_remaining > 0
        if hot_tag_active:
            control *= plant_cfg.lorenz_hot_tag_control_boost
        hot_tag_triggered_this_step = False
        assist_trigger_trace[i] = 0.0
        hope_flag_trace[i] = 0.0
        heat_flag_trace[i] = 0.0
        salience_mean_trace[i] = result["salience_mean"]
        salience_min_trace[i] = result["salience_min"]
        salience_channel_trace[i] = result["salience_channels"]
        control_energy_step = abs(control) * plant_cfg.dt
        control_energy += control_energy_step
        control_energy_trace[i] = control_energy_step
        state_derivative = (-state + control) / plant_cfg.tau

        charge_factor = 1.0
        if plant_cfg.salience_breathing:
            charge_norm_prev = float(np.clip(salience_charge, 0.0, 1.0))
            recovery = max(0.0, plant_cfg.salience_breath_recovery_scale)
            attack = max(recovery, plant_cfg.salience_breath_attack_scale)
            charge_factor = recovery + (attack - recovery) * charge_norm_prev

        if assist_pulse_pending and assist_pulse_steps == 0:
            assist_pulse_steps = max(1, plant_cfg.assist_pulse_width)
            assist_pulse_pending = False
            assist_pulse_count += 1

        assist_force = 0.0
        assist_energy_step = 0.0

        assist_pulse_active = assist_pulse_steps > 0
        if assist_pulse_active and plant_cfg.assist_amplitude > 0.0:
            pulse_force = plant_cfg.assist_amplitude * (target[i] - state)
            assist_pulse_active_trace[i] = 1.0
            assist_force += pulse_force
            assist_energy_step += abs(pulse_force) * plant_cfg.dt
            assist_pulse_steps = max(0, assist_pulse_steps - 1)
        else:
            assist_pulse_active_trace[i] = 0.0

        legacy_assist_allowed = hot_tag_active and plant_cfg.hot_tag_assist_gain > 0.0
        if plant_cfg.assist_poincare_enabled:
            legacy_assist_allowed = legacy_assist_allowed and assist_window_remaining > 0
        if legacy_assist_allowed:
            legacy_force = plant_cfg.hot_tag_assist_gain * (target[i] - state)
            assist_force += legacy_force
            assist_energy_step += abs(legacy_force) * plant_cfg.dt

        if assist_force != 0.0:
            state_derivative += assist_force

        assist_energy += assist_energy_step
        assist_energy_trace[i] = assist_energy_step

        hot_tag_active = plant_cfg.lorenz_hot_tag_enabled and hot_tag_steps_remaining > 0
        if plant_cfg.lorenz_gain != 0.0:
            ramp_step = 1.0 / max(1, plant_cfg.lorenz_ramp_steps)
            if hot_tag_active:
                lorenz_scale = 0.0
            else:
                if plant_cfg.lorenz_hot_tag_enabled and hot_tag_cooldown > 0:
                    lorenz_scale = max(0.0, lorenz_scale - ramp_step)
                    hot_tag_cooldown -= 1
                elif plant_cfg.lorenz_gate_enabled:
                    release_ready = False
                    if len(authority_history) >= plant_cfg.lorenz_gate_release_steps:
                        recent = authority_history[-plant_cfg.lorenz_gate_release_steps :]
                        release_ready = all(auth >= plant_cfg.lorenz_gate_threshold for auth in recent)
                    if release_ready:
                        lorenz_scale = min(1.0, lorenz_scale + ramp_step)
                    else:
                        lorenz_scale = max(0.0, lorenz_scale - ramp_step)
                else:
                    lorenz_scale = 1.0

            x, y, z = lorenz_state
            sigma = plant_cfg.lorenz_sigma
            rho = plant_cfg.lorenz_rho
            beta = plant_cfg.lorenz_beta
        else:
            lorenz_scale = 0.0

        if plant_cfg.lorenz_gain != 0.0:
            prev_lorenz_state = lorenz_state.copy() if plant_cfg.assist_poincare_enabled else None
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            lorenz_state = lorenz_state + plant_cfg.dt * np.array([dx, dy, dz], dtype=float)
            if plant_cfg.assist_poincare_enabled:
                prev_val = prev_lorenz_state[poincare_axis] - plant_cfg.assist_poincare_threshold
                curr_val = lorenz_state[poincare_axis] - plant_cfg.assist_poincare_threshold
                crossed_pos = prev_val <= 0.0 and curr_val > 0.0
                crossed_neg = prev_val >= 0.0 and curr_val < 0.0
                if (poincare_direction > 0 and crossed_pos) or (poincare_direction < 0 and crossed_neg):
                    assist_window_remaining = max(
                        plant_cfg.assist_poincare_window,
                        assist_window_remaining,
                    )
                    assist_crossing_count += 1
            raw_force = plant_cfg.lorenz_gain * lorenz_state[0]
            scaled_force = lorenz_scale * raw_force
            if plant_cfg.salience_breathing:
                scaled_force *= charge_factor
            state_derivative += scaled_force
            lorenz_energy_step = abs(scaled_force) * plant_cfg.dt
            lorenz_energy += lorenz_energy_step
            lorenz_active_trace[i] = 1.0 if lorenz_scale > 0.0 else 0.0
        else:
            scaled_force = 0.0
            lorenz_energy_step = 0.0
            lorenz_active_trace[i] = 0.0

        if plant_cfg.lorenz_hot_tag_enabled:
            if hot_tag_active:
                hot_tag_steps_remaining -= 1
                if hot_tag_steps_remaining <= 0:
                    hot_tag_steps_remaining = 0
                    hot_tag_cooldown = max(hot_tag_cooldown, plant_cfg.lorenz_hot_tag_cooldown_steps)
                    if using_partner:
                        using_partner = False
                        active_controller = main_controller
                else:
                    if partner_controller is not None and not using_partner:
                        using_partner = True
                        partner_controller.reset()
                        active_controller = partner_controller
                        partner_switch_count += 1
            elif hot_tag_cooldown > 0 and plant_cfg.lorenz_gain != 0.0:
                hot_tag_cooldown = max(0, hot_tag_cooldown - 1)
            elif using_partner and hot_tag_steps_remaining == 0:
                using_partner = False
                active_controller = main_controller
            hot_tag_active_trace[i] = 1.0 if hot_tag_active else 0.0
            partner_active_trace[i] = 1.0 if using_partner else 0.0
        else:
            hot_tag_active_trace[i] = 0.0
            partner_active_trace[i] = 0.0

        if plant_cfg.assist_poincare_enabled:
            assist_window_trace[i] = 1.0 if assist_window_remaining > 0 else 0.0

        lorenz_energy_trace[i] = lorenz_energy_step
        lorenz_scale_trace[i] = lorenz_scale

        state += plant_cfg.dt * state_derivative

        time_now = i * plant_cfg.dt
        phase_scale = 1.0
        if plant_cfg.breath_period > 0.0:
            phase_pos = (time_now % plant_cfg.breath_period) / plant_cfg.breath_period
            if plant_cfg.breath_pulse_count <= 1:
                if plant_cfg.breath_in_fraction < 1e-9:
                    phase_scale = plant_cfg.breath_out_scale
                elif phase_pos < plant_cfg.breath_in_fraction:
                    phase_scale = plant_cfg.breath_in_scale
                else:
                    phase_scale = plant_cfg.breath_out_scale
            else:
                spacing = 1.0 / plant_cfg.breath_pulse_count
                local_phase = (phase_pos % spacing) / spacing if spacing > 0 else phase_pos
                if local_phase < plant_cfg.breath_pulse_width:
                    phase_scale = plant_cfg.breath_in_scale
                else:
                    phase_scale = plant_cfg.breath_out_scale
        effective_scale = phase_scale
        if plant_cfg.adaptive_breathing:
            if last_authority_ratio >= plant_cfg.adapt_authority_threshold and last_salience >= plant_cfg.adapt_salience_threshold:
                adaptive_scale = min(adaptive_scale + plant_cfg.adapt_step_gain, plant_cfg.adapt_scale_max)
            else:
                adaptive_scale = max(adaptive_scale - plant_cfg.adapt_step_gain, plant_cfg.adapt_scale_min)
            effective_scale *= adaptive_scale
        if plant_cfg.salience_breathing:
            effective_scale *= charge_factor
        effective_scale = max(effective_scale, 0.0)
        adaptive_scale_trace[i] = effective_scale
        effective_sigma = plant_cfg.noise_sigma * effective_scale

        if effective_sigma > 0.0 and time_now >= plant_cfg.noise_delay:
            noise_term = rng.normal(0.0, effective_sigma)
            state += noise_term
            noise_energy_step = abs(noise_term)
            noise_energy += noise_energy_step
            noise_energy_trace[i] = noise_energy_step
        else:
            noise_energy_trace[i] = 0.0

        if plant_cfg.lorenz_gain != 0.0:
            external_step = lorenz_energy_trace[i] + noise_energy_trace[i]
        else:
            external_step = noise_energy_trace[i]
        control_energy_cum += control_energy_step
        external_energy_cum += external_step
        if external_energy_cum > 1e-9:
            last_authority_ratio = control_energy_cum / external_energy_cum
        else:
            last_authority_ratio = float("inf")
        authority_ratio_trace[i] = last_authority_ratio
        if i > 0:
            authority_slope_trace[i] = (authority_ratio_trace[i] - authority_ratio_trace[i - 1]) / plant_cfg.dt
        else:
            authority_slope_trace[i] = 0.0
        authority_window_queue.append(last_authority_ratio)
        if len(authority_window_queue) == authority_window_queue.maxlen:
            authority_window_ratio_trace[i] = float(sum(authority_window_queue) / len(authority_window_queue))
            if (
                first_break_index is None
                and authority_window_ratio_trace[i] >= CONTROL_AUTHORITY_THRESHOLD
            ):
                first_break_index = i
        external_energy_trace[i] = external_step
        if plant_cfg.adaptive_breathing:
            adaptive_scale_min = min(adaptive_scale_min, effective_scale)
            adaptive_scale_max = max(adaptive_scale_max, effective_scale)
            last_salience = result["salience_mean"]
        if last_authority_ratio < plant_cfg.hot_tag_heat_authority:
            heat_salience_sum += result["salience_mean"]
            heat_step_count += 1
            heat_flag_trace[i] = 1.0

        current_slope = authority_slope_trace[i]
        in_hope_window = (
            plant_cfg.hope_spot_lower <= last_authority_ratio <= plant_cfg.hope_spot_upper
            and current_slope >= plant_cfg.hope_spot_min_slope
        )
        hope_active_current = in_hope_window
        if in_hope_window and not in_hope:
            in_hope = True
            hope_spot_event_count += 1
            if first_hope_index is None:
                first_hope_index = i
        if in_hope_window:
            hope_flag_trace[i] = 1.0
            hope_spot_salience_sum += result["salience_mean"]
            hope_spot_step_count += 1
        else:
            in_hope = False

        if plant_cfg.salience_breathing:
            salience_error = max(0.0, result["salience_mean"] - plant_cfg.salience_breath_floor)
            slope_boost = plant_cfg.salience_breath_slope_gain * max(current_slope, 0.0)
            hope_boost = plant_cfg.salience_breath_hope_gain if hope_active_current else 0.0
            target_charge = plant_cfg.salience_breath_gain * salience_error + slope_boost + hope_boost
            target_charge = float(np.clip(target_charge, 0.0, 2.0))
            smoothing = float(np.clip(plant_cfg.salience_breath_smoothing, 0.0, 1.0))
            salience_charge += smoothing * (target_charge - salience_charge)
            salience_charge = float(np.clip(salience_charge, 0.0, 1.0))
        else:
            salience_charge = 0.0
        salience_charge_trace[i] = salience_charge

        if plant_cfg.salience_breathing:
            slope_sign = 1 if current_slope > 0 else -1 if current_slope < 0 else 0
            if prev_positive_slope is not None and slope_sign != 0 and slope_sign != prev_positive_slope:
                salience_flip_score = plant_cfg.salience_osc_smoothing * (1.0 - salience_flip_score) + (1.0 - plant_cfg.salience_osc_smoothing) * salience_flip_score
            else:
                salience_flip_score *= (1.0 - plant_cfg.salience_osc_smoothing)
            prev_positive_slope = slope_sign if slope_sign != 0 else prev_positive_slope
            salience_flip_trace[i] = salience_flip_score

            variance_metric = float(np.var(salience_channel_trace[max(0, i - 20) : i + 1]))
            variance_gain = plant_cfg.salience_osc_variance_gain * variance_metric

            lambda_mod = plant_cfg.salience_lambda_gain * salience_charge
            gate_bias_mod = plant_cfg.salience_gate_bias_gain * salience_charge - plant_cfg.salience_gate_bias_damp * salience_flip_score
            gate_floor_mod = plant_cfg.salience_gate_floor_gain * salience_charge

            active_controller.cfg.lambda_c = max(0.0, main_controller.cfg.lambda_c + lambda_mod)
            active_controller.cfg.salience_gate_bias = main_controller.cfg.salience_gate_bias + gate_bias_mod
            active_controller.cfg.salience_gate_floor = max(1e-3, main_controller.cfg.salience_gate_floor + gate_floor_mod)

            assist_amp = plant_cfg.salience_assist_amp_floor + plant_cfg.salience_assist_amp_gain * salience_charge
            assist_amp = float(np.clip(assist_amp, 0.0, plant_cfg.salience_assist_amp_max))
            active_assist_amp = assist_amp * (1.0 - variance_gain)
            active_assist_amp = max(0.0, active_assist_amp)
            plant_cfg.assist_amplitude = active_assist_amp
            assist_amplitude_trace[i] = active_assist_amp

            cooldown_mod = int(max(0.0, plant_cfg.salience_assist_cooldown_gain * (1.0 - salience_charge)))
            plant_cfg.assist_cooldown_steps = max(1, assist_cooldown + cooldown_mod)
            assist_cooldown_trace[i] = plant_cfg.assist_cooldown_steps

            salience_modulated_lambda[i] = active_controller.cfg.lambda_c
        else:
            salience_flip_trace[i] = 0.0
            assist_amplitude_trace[i] = plant_cfg.assist_amplitude
            assist_cooldown_trace[i] = plant_cfg.assist_cooldown_steps
            salience_modulated_lambda[i] = main_controller.cfg.lambda_c

        assist_triggered_this_step = False
        if plant_cfg.assist_poincare_enabled:
            if assist_window_remaining > 0:
                assist_window_remaining = max(0, assist_window_remaining - 1)
            if assist_window_remaining > 0:
                assist_window_trace[i] = 1.0
                slope = 0.0
                slope_window = min(i, plant_cfg.lorenz_hot_tag_window)
                if slope_window > 0:
                    slope = authority_ratio_trace[i] - authority_ratio_trace[i - slope_window]
                recent_avg = authority_window_ratio_trace[i]
                if math.isnan(recent_avg):
                    recent_avg = last_authority_ratio
                if (
                    assist_cooldown == 0
                    and plant_cfg.assist_authority_slope_min <= slope
                    and result["salience_mean"] >= plant_cfg.assist_salience_floor
                    and assist_pulse_steps == 0
                    and recent_avg >= plant_cfg.hope_spot_lower
                    and recent_avg <= plant_cfg.hope_spot_upper
                ):
                    assist_pulse_pending = True
                    assist_cooldown = max(assist_cooldown, plant_cfg.assist_cooldown_steps)
                    assist_trigger_count += 1
                    assist_trigger_trace[i] = 1.0
                    assist_triggered_this_step = True
                    assist_trigger_indices.append(i)
        if hope_active_current and assist_triggered_this_step:
            hope_spot_assist_count += 1

        authority_history.append(last_authority_ratio)
        if plant_cfg.lorenz_hot_tag_enabled:
            if last_authority_ratio < plant_cfg.hot_tag_heat_authority:
                heat_counter = min(heat_counter + 1, plant_cfg.hot_tag_heat_steps)
            else:
                heat_counter = max(0, heat_counter - 1)
            heat_peak = max(heat_peak, heat_counter)

            in_near_window = (
                plant_cfg.hot_tag_near_tag_threshold
                <= last_authority_ratio
                <= plant_cfg.hot_tag_near_tag_upper
            )
            if in_near_window and not near_tag_active:
                near_tag_attempts += 1
                near_tag_total += 1
                near_tag_active = True
            elif not in_near_window:
                near_tag_active = False

            window = max(2, plant_cfg.lorenz_hot_tag_window)
            if len(authority_history) >= window:
                recent_window = authority_history[-window:]
                slope = (recent_window[-1] - recent_window[0]) / (window - 1)
            else:
                slope = 0.0

            hot_tag_ready = (
                heat_counter >= plant_cfg.hot_tag_heat_steps
                and near_tag_attempts >= plant_cfg.hot_tag_near_tag_attempts
                and slope >= plant_cfg.lorenz_hot_tag_slope_threshold
                and authority_history[-1] >= plant_cfg.hot_tag_near_tag_threshold
            )
            if (
                hot_tag_ready
                and hot_tag_steps_remaining == 0
                and hot_tag_cooldown == 0
            ):
                hot_tag_steps_remaining = plant_cfg.lorenz_hot_tag_pulse_steps
                hot_tag_active_trace[i] = 1.0
                hot_tag_trigger_count += 1
                hot_tag_trigger_trace[i] = 1.0
                hot_tag_triggered_this_step = True
                heat_counter = 0
                near_tag_attempts = 0
                if partner_controller is not None:
                    partner_controller.reset()
                    active_controller = partner_controller
                    using_partner = True
                    partner_switch_count += 1
        if hope_active_current and hot_tag_triggered_this_step:
            hope_spot_hot_tag_count += 1

    errors = target - outputs
    rise_time = compute_rise_time(outputs, target[-1], plant_cfg.dt)
    peak_overshoot = float(np.max(outputs) - target[-1])
    settling = compute_settling_time(outputs, target[-1], plant_cfg.dt)
    rms_error = float(np.sqrt(np.mean(np.square(errors))))
    mean_salience = float(np.mean(salience_mean_trace))
    min_salience = float(np.min(salience_min_trace))
    salience_variance = float(np.var(salience_channel_trace))

    indices = np.where(outputs >= 0.9 * target[-1])[0]
    if indices.size > 0:
        rise_idx = int(indices[0])
        control_energy_rise = float(np.sum(control_energy_trace[: rise_idx + 1]))
        external_energy_rise = float(
            np.sum(noise_energy_trace[: rise_idx + 1])
            + np.sum(lorenz_energy_trace[: rise_idx + 1])
        )
        if external_energy_rise > 1e-9:
            control_authority_ratio = float(control_energy_rise / external_energy_rise)
        else:
            control_authority_ratio = float("inf")
    else:
        rise_idx = None
        control_energy_rise = float("nan")
        external_energy_rise = float("nan")
        control_authority_ratio = float("nan")

    heat_salience_mean = heat_salience_sum / heat_step_count if heat_step_count > 0 else None
    hope_salience_mean = hope_spot_salience_sum / hope_spot_step_count if hope_spot_step_count > 0 else None

    hope_assist_ratio = (
        hope_spot_assist_count / hope_spot_event_count if hope_spot_event_count > 0 else None
    )
    hope_hot_tag_ratio = (
        hope_spot_hot_tag_count / hope_spot_event_count if hope_spot_event_count > 0 else None
    )

    energy_between_hope_and_break = None
    external_between_hope_and_break = None
    if (
        first_hope_index is not None
        and first_break_index is not None
        and first_break_index > first_hope_index
    ):
        energy_between_hope_and_break = float(
            np.sum(control_energy_trace[first_hope_index : first_break_index + 1])
        )
        external_between_hope_and_break = float(
            np.sum(external_energy_trace[first_hope_index : first_break_index + 1])
        )

    lead_lag = None
    lead_corr = None
    max_lag_steps = min(5, steps - 2)
    best_corr = -1.0
    best_lag = None
    salience_centered = salience_mean_trace - np.mean(salience_mean_trace)
    slope_centered = authority_slope_trace - np.mean(authority_slope_trace)
    salience_std = float(np.std(salience_centered))
    slope_std = float(np.std(slope_centered))
    if salience_std > 1e-9 and slope_std > 1e-9 and max_lag_steps >= 1:
        for lag in range(1, max_lag_steps + 1):
            x = salience_centered[:-lag]
            y = slope_centered[lag:]
            if x.size > 1 and y.size > 1:
                corr = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag
        if best_lag is not None:
            lead_lag = best_lag
            lead_corr = best_corr

    return {
        "rise_time_90": rise_time,
        "peak_overshoot": peak_overshoot,
        "settling_time_2pct": settling,
        "rms_error": rms_error,
        "mean_salience": mean_salience,
        "min_salience": min_salience,
        "salience_variance": salience_variance,
        "control_energy": control_energy,
        "noise_energy": noise_energy,
        "lorenz_energy": lorenz_energy,
        "assist_energy": assist_energy,
        "external_energy": noise_energy + lorenz_energy,
        "control_energy_rise": control_energy_rise,
        "external_energy_rise": external_energy_rise,
        "control_authority_ratio": control_authority_ratio,
        "adaptive_scale_mean": float(np.mean(adaptive_scale_trace)) if plant_cfg.adaptive_breathing else None,
        "adaptive_scale_min": adaptive_scale_min if plant_cfg.adaptive_breathing else None,
        "adaptive_scale_max": adaptive_scale_max if plant_cfg.adaptive_breathing else None,
        "lorenz_gate_active_mean": float(np.mean(lorenz_active_trace)) if plant_cfg.lorenz_gate_enabled else None,
        "lorenz_gate_release_steps": plant_cfg.lorenz_gate_release_steps if plant_cfg.lorenz_gate_enabled else None,
        "lorenz_scale_mean": float(np.mean(lorenz_scale_trace)) if plant_cfg.lorenz_gate_enabled else None,
        "lorenz_scale_min": float(np.min(lorenz_scale_trace)) if plant_cfg.lorenz_gate_enabled else None,
        "lorenz_scale_max": float(np.max(lorenz_scale_trace)) if plant_cfg.lorenz_gate_enabled else None,
        "breath_pulse_count": plant_cfg.breath_pulse_count,
        "breath_pulse_width": plant_cfg.breath_pulse_width,
        "hot_tag_active_mean": float(np.mean(hot_tag_active_trace)) if plant_cfg.lorenz_hot_tag_enabled else None,
        "hot_tag_trigger_count": hot_tag_trigger_count if plant_cfg.lorenz_hot_tag_enabled else None,
        "hot_tag_heat_peak": heat_peak if plant_cfg.lorenz_hot_tag_enabled else None,
        "hot_tag_near_tag_total": near_tag_total if plant_cfg.lorenz_hot_tag_enabled else None,
        "hot_tag_partner_active_mean": float(np.mean(partner_active_trace)) if plant_cfg.lorenz_hot_tag_enabled else None,
        "hot_tag_partner_switches": partner_switch_count if plant_cfg.lorenz_hot_tag_enabled else None,
        "assist_energy_mean": float(np.mean(assist_energy_trace))
        if (plant_cfg.hot_tag_assist_gain > 0.0 or plant_cfg.assist_amplitude > 0.0)
        else None,
        "hot_tag_auto_start_steps": plant_cfg.hot_tag_auto_start_steps if plant_cfg.lorenz_hot_tag_enabled else None,
        "assist_window_mean": float(np.mean(assist_window_trace)) if plant_cfg.assist_poincare_enabled else None,
        "assist_trigger_count": assist_trigger_count if plant_cfg.assist_poincare_enabled else None,
        "assist_crossing_count": assist_crossing_count if plant_cfg.assist_poincare_enabled else None,
        "assist_pulse_count": assist_pulse_count if plant_cfg.assist_amplitude > 0.0 else None,
        "hope_spot_events": hope_spot_event_count,
        "hope_salience_mean": hope_salience_mean,
        "hope_assist_ratio": hope_assist_ratio,
        "hope_hot_tag_ratio": hope_hot_tag_ratio,
        "heat_salience_mean": heat_salience_mean,
        "heat_step_fraction": heat_step_count / steps if heat_step_count > 0 else 0.0,
        "first_hope_index": first_hope_index,
        "first_break_index": first_break_index,
        "energy_between_hope_and_break": energy_between_hope_and_break,
        "external_between_hope_and_break": external_between_hope_and_break,
        "salience_authority_lead_lag": lead_lag,
        "salience_authority_lead_corr": lead_corr,
    }


def compute_rise_time(outputs: np.ndarray, target: float, dt: float) -> float:
    threshold = 0.9 * target
    indices = np.where(outputs >= threshold)[0]
    if indices.size == 0:
        return float("nan")
    return float(indices[0] * dt)


def compute_settling_time(outputs: np.ndarray, target: float, dt: float, tol: float = 0.02) -> float:
    upper = target + tol
    lower = target - tol
    for idx in range(len(outputs)):
        window = outputs[idx:]
        if window.size == 0:
            break
        if np.all((window >= lower) & (window <= upper)):
            return float(idx * dt)
    return float("nan")


def run_mass_sweep(
    lambdas: Iterable[float],
    seed: int | None = None,
    *,
    salience_floor: float = 0.0,
    controller_template: ControllerConfig | None = None,
) -> List[Dict[str, float]]:
    plant_cfg = PlantConfig()
    template = controller_template or ControllerConfig()
    results: List[Dict[str, float]] = []
    baseline_rise: float | None = None
    baseline_energy: float | None = None

    for idx, lam in enumerate(lambdas):
        controller_cfg = replace(
            template,
            lambda_c=lam,
            salience_floor=salience_floor,
        )
        metrics = simulate(controller_cfg, plant_cfg, seed=None if seed is None else seed + idx)
        if baseline_rise is None:
            baseline_rise = metrics["rise_time_90"]
            baseline_energy = metrics["control_energy"]
        if baseline_rise and not math.isnan(metrics["rise_time_90"]):
            m_eff = metrics["rise_time_90"] / baseline_rise
        else:
            m_eff = float("nan")
        energy_ratio = (
            metrics["control_energy"] / baseline_energy if baseline_energy else float("nan")
        )
        results.append({**metrics, "lambda_c": lam, "m_eff": m_eff, "energy_ratio": energy_ratio})
    return results


def run_stress_sweep(
    lambdas: Sequence[float],
    plant_cfg: PlantConfig,
    seed: int | None = None,
    *,
    salience_floor: float = 0.0,
    controller_template: ControllerConfig | None = None,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    baseline_rise: float | None = None
    baseline_energy: float | None = None
    template = controller_template or ControllerConfig()

    for idx, lam in enumerate(lambdas):
        controller_cfg = replace(
            template,
            lambda_c=lam,
            salience_channels=plant_cfg.salience_channels,
            salience_floor=salience_floor,
        )
        metrics = simulate(controller_cfg, plant_cfg, seed=None if seed is None else seed + idx)
        authority_ratio = metrics.get("control_authority_ratio", float("nan"))
        authority_ok = bool(
            not math.isnan(authority_ratio)
            and (
                authority_ratio >= CONTROL_AUTHORITY_THRESHOLD
                or math.isinf(authority_ratio)
            )
        )
        if authority_ok and baseline_rise is None:
            baseline_rise = metrics["rise_time_90"]
            baseline_energy = metrics["control_energy"]
        if authority_ok and baseline_rise and not math.isnan(metrics["rise_time_90"]):
            m_eff = metrics["rise_time_90"] / baseline_rise
        else:
            m_eff = float("nan")
        if authority_ok and baseline_energy:
            energy_ratio = metrics["control_energy"] / baseline_energy
        else:
            energy_ratio = float("nan")
        results.append(
            {
                **metrics,
                "lambda_c": lam,
                "m_eff": m_eff,
                "energy_ratio": energy_ratio,
                "salience_channels": plant_cfg.salience_channels,
                "noise_sigma": plant_cfg.noise_sigma,
                "lorenz_gain": plant_cfg.lorenz_gain,
                "salience_floor": salience_floor,
                "adaptive_breathing": plant_cfg.adaptive_breathing,
                "lorenz_gate_enabled": plant_cfg.lorenz_gate_enabled,
                "lorenz_hot_tag_enabled": plant_cfg.lorenz_hot_tag_enabled,
                "authority_ok": authority_ok,
            }
        )
    return results


def write_artifact(
    entries: List[Dict[str, float]],
    *,
    experiment_name: str = "experiment_a_mass_sweep",
    prefix: str = "mass_sweep",
) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())
    payload = []
    for entry in entries:
        payload.append(
            {
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "experiment_name": experiment_name,
                "run_id": run_id,
                **entry,
            }
        )
    path = ARTIFACT_DIR / f"{prefix}_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def print_summary(entries: List[Dict[str, float]]) -> None:
    rows = [
        (
            entry["lambda_c"],
            entry["rise_time_90"],
            entry["peak_overshoot"],
            entry["settling_time_2pct"],
            entry["rms_error"],
            entry["mean_salience"],
            entry.get("min_salience"),
            entry.get("salience_variance"),
            entry["control_energy"],
            entry["m_eff"],
            entry.get("energy_ratio"),
            entry.get("control_authority_ratio"),
            entry.get("authority_ok"),
            entry.get("adaptive_scale_mean"),
            entry.get("adaptive_scale_min"),
            entry.get("adaptive_scale_max"),
            entry.get("lorenz_gate_active_mean"),
            entry.get("lorenz_scale_mean"),
            entry.get("hot_tag_active_mean"),
            entry.get("hot_tag_trigger_count"),
            entry.get("hot_tag_heat_peak"),
            entry.get("hot_tag_near_tag_total"),
            entry.get("hot_tag_partner_active_mean"),
            entry.get("hot_tag_partner_switches"),
            entry.get("assist_energy_mean"),
            entry.get("hot_tag_auto_start_steps"),
            entry.get("assist_window_mean"),
            entry.get("assist_trigger_count"),
            entry.get("assist_crossing_count"),
            entry.get("assist_pulse_count"),
            entry.get("salience_authority_lead_lag"),
            entry.get("salience_authority_lead_corr"),
            entry.get("hope_spot_events"),
            entry.get("hope_salience_mean"),
            entry.get("hope_assist_ratio"),
            entry.get("hope_hot_tag_ratio"),
            entry.get("heat_salience_mean"),
            entry.get("heat_step_fraction"),
            entry.get("energy_between_hope_and_break"),
            entry.get("external_between_hope_and_break"),
        )
        for entry in entries
    ]
    print(
        tabulate(
            rows,
            headers=[
                "lambda_c",
                "rise_time_90",
                "peak_overshoot",
                "settling_time_2pct",
                "rms_error",
                "mean_salience",
                "min_salience",
                "salience_variance",
                "control_energy",
                "m_eff",
                "energy_ratio",
                "authority_ratio",
                "authority_ok",
                "adapt_scale_mean",
                "adapt_scale_min",
                "adapt_scale_max",
                "lorenz_gate_active",
                "lorenz_scale_mean",
                "hot_tag_active",
                "hot_tag_triggers",
                "hot_tag_heat_peak",
                "hot_tag_near_attempts",
                "partner_active",
                "partner_switches",
                "assist_energy_mean",
                "auto_start_steps",
                "assist_window_mean",
                "assist_triggers",
                "assist_crossings",
                "assist_pulses",
                "lead_lag",
                "lead_corr",
                "hope_events",
                "hope_salience",
                "hope_assist",
                "hope_hot_tag",
                "heat_salience",
                "heat_frac",
                "energy_hope_to_break",
                "external_hope_to_break",
            ],
            tablefmt="github",
            floatfmt=".6f",
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuity mass and stress sweeps")
    parser.add_argument(
        "--mode",
        choices=["mass", "stress"],
        default="mass",
        help="Select mass sweep (Experiment A) or stress sweep (Experiment P)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Base random seed for stochastic runs")
    parser.add_argument(
        "--lambda-values",
        type=str,
        default=None,
        help="Comma-separated list of lambda_c values to sweep (overrides defaults)",
    )
    parser.add_argument("--horizon", type=float, default=None, help="Override horizon duration (seconds)")
    parser.add_argument("--dt", type=float, default=None, help="Override simulation timestep")
    parser.add_argument("--noise-sigma", type=float, default=0.15, help="Gaussian noise sigma for stress mode")
    parser.add_argument(
        "--noise-delay",
        type=float,
        default=0.0,
        help="Time in seconds before stochastic forcing activates",
    )
    parser.add_argument(
        "--breath-period",
        type=float,
        default=0.0,
        help="Breathing cycle period in seconds (0 disables modulation)",
    )
    parser.add_argument(
        "--breath-in-fraction",
        type=float,
        default=0.0,
        help="Fraction of the period allocated to the low-noise inhale phase",
    )
    parser.add_argument(
        "--breath-in-scale",
        type=float,
        default=0.1,
        help="Noise scale multiplier during inhale (relative to base sigma)",
    )
    parser.add_argument(
        "--breath-out-scale",
        type=float,
        default=1.0,
        help="Noise scale multiplier during exhale (relative to base sigma)",
    )
    parser.add_argument(
        "--breath-pulse-count",
        type=int,
        default=1,
        help="Number of rest pulses per breathing period (1 = continuous inhale/exhale)",
    )
    parser.add_argument(
        "--breath-pulse-width",
        type=float,
        default=0.5,
        help="Fraction of each pulse interval spent in low-noise inhale mode",
    )
    parser.add_argument(
        "--adaptive-breathing",
        action="store_true",
        help="Enable adaptive adjustment of breathing scale based on authority/salience",
    )
    parser.add_argument(
        "--salience-breathing",
        action="store_true",
        help="Let measured salience charge drive breathing/forcing scale",
    )
    parser.add_argument(
        "--salience-breath-floor",
        type=float,
        default=0.55,
        help="Salience level that marks full recovery for charge accumulation",
    )
    parser.add_argument(
        "--salience-breath-gain",
        type=float,
        default=3.0,
        help="Gain applied to salience error when building charge",
    )
    parser.add_argument(
        "--salience-breath-slope-gain",
        type=float,
        default=0.25,
        help="Boost added to charge from positive authority slope",
    )
    parser.add_argument(
        "--salience-breath-hope-gain",
        type=float,
        default=0.2,
        help="Bonus charge applied while in hope window",
    )
    parser.add_argument(
        "--salience-breath-smoothing",
        type=float,
        default=0.1,
        help="Smoothing factor for charge integration (0=stiff, 1=immediate)",
    )
    parser.add_argument(
        "--salience-breath-recovery-scale",
        type=float,
        default=0.15,
        help="Minimum forcing scale during recovery (when charge is zero)",
    )
    parser.add_argument(
        "--salience-breath-attack-scale",
        type=float,
        default=1.0,
        help="Maximum forcing scale when charge is saturated",
    )
    parser.add_argument(
        "--salience-osc-smoothing",
        type=float,
        default=0.2,
        help="Memory factor for detecting slope flips (0=quick,1=slow)",
    )
    parser.add_argument(
        "--salience-osc-flip-gain",
        type=float,
        default=0.5,
        help="Gain applied when slope flips sign, used to damp oscillations",
    )
    parser.add_argument(
        "--salience-osc-variance-gain",
        type=float,
        default=0.1,
        help="Gain mapping salience variance into assist attenuation",
    )
    parser.add_argument(
        "--salience-lambda-gain",
        type=float,
        default=0.0,
        help="How strongly charge modulates controller lambda_c",
    )
    parser.add_argument(
        "--salience-gate-bias-gain",
        type=float,
        default=0.0,
        help="Charge to gate-bias gain (opens gate as charge grows)",
    )
    parser.add_argument(
        "--salience-gate-bias-damp",
        type=float,
        default=0.0,
        help="Amount slope flips reduce gate bias (stability control)",
    )
    parser.add_argument(
        "--salience-gate-floor-gain",
        type=float,
        default=0.0,
        help="Charge to gate-floor gain (prevents shutdown at high charge)",
    )
    parser.add_argument(
        "--salience-assist-amp-floor",
        type=float,
        default=1.0,
        help="Base assist amplitude when charge is zero",
    )
    parser.add_argument(
        "--salience-assist-amp-gain",
        type=float,
        default=0.0,
        help="Charge to assist amplitude gain",
    )
    parser.add_argument(
        "--salience-assist-amp-max",
        type=float,
        default=1.0,
        help="Clamp on salience-modulated assist amplitude",
    )
    parser.add_argument(
        "--salience-assist-cooldown-gain",
        type=float,
        default=0.0,
        help="How strongly low charge lengthens assist cooldown",
    )
    parser.add_argument(
        "--adapt-authority-threshold",
        type=float,
        default=1.1,
        help="Authority ratio required to increase disturbance scale during adaptive breathing",
    )
    parser.add_argument(
        "--adapt-salience-threshold",
        type=float,
        default=0.55,
        help="Salience threshold required alongside authority to increase disturbance scale",
    )
    parser.add_argument(
        "--adapt-scale-min",
        type=float,
        default=0.05,
        help="Minimum adaptive scale multiplier",
    )
    parser.add_argument(
        "--adapt-scale-max",
        type=float,
        default=1.0,
        help="Maximum adaptive scale multiplier",
    )
    parser.add_argument(
        "--adapt-step-gain",
        type=float,
        default=0.05,
        help="Step size for adaptive scale adjustments",
    )
    parser.add_argument(
        "--lorenz-gain",
        type=float,
        default=0.05,
        help="Lorenz coupling gain for stress mode (set 0 to disable)",
    )
    parser.add_argument(
        "--lorenz-gate",
        action="store_true",
        help="Enable authority-based gating of Lorenz forcing",
    )
    parser.add_argument(
        "--lorenz-gate-threshold",
        type=float,
        default=1.0,
        help="Authority ratio required before Lorenz forcing is allowed",
    )
    parser.add_argument(
        "--lorenz-gate-release-steps",
        type=int,
        default=10,
        help="Number of consecutive steps above threshold required to re-enable Lorenz forcing",
    )
    parser.add_argument(
        "--lorenz-ramp-steps",
        type=int,
        default=20,
        help="Number of steps to ramp Lorenz forcing from 0 to full strength",
    )
    parser.add_argument(
        "--lorenz-hot-tag",
        action="store_true",
        help="Enable authority-slope-triggered Lorenz hot tag pulses",
    )
    parser.add_argument(
        "--lorenz-hot-tag-window",
        type=int,
        default=15,
        help="Window (steps) for computing authority slope before triggering hot tag",
    )
    parser.add_argument(
        "--lorenz-hot-tag-slope-threshold",
        type=float,
        default=0.25,
        help="Minimum authority slope required within the window to trigger hot tag",
    )
    parser.add_argument(
        "--lorenz-hot-tag-pulse-steps",
        type=int,
        default=50,
        help="Duration (steps) to hold Lorenz forcing at full strength once hot tag triggers",
    )
    parser.add_argument(
        "--lorenz-hot-tag-cooldown-steps",
        type=int,
        default=60,
        help="Cooldown (steps) before another hot tag pulse can trigger",
    )
    parser.add_argument(
        "--lorenz-hot-tag-control-boost",
        type=float,
        default=1.3,
        help="Multiplier applied to control effort during hot tag window",
    )
    parser.add_argument(
        "--hot-tag-heat-steps",
        type=int,
        default=200,
        help="Number of consecutive heat steps (authority below threshold) required before hot tag is possible",
    )
    parser.add_argument(
        "--hot-tag-heat-authority",
        type=float,
        default=0.8,
        help="Authority ratio defining the heat zone (below this counts as being worked over)",
    )
    parser.add_argument(
        "--hot-tag-near-tag-threshold",
        type=float,
        default=0.95,
        help="Lower bound for near-tag authority window",
    )
    parser.add_argument(
        "--hot-tag-near-tag-upper",
        type=float,
        default=1.1,
        help="Upper bound for near-tag authority window",
    )
    parser.add_argument(
        "--hot-tag-near-tag-attempts",
        type=int,
        default=3,
        help="Number of near-tag attempts required before hot tag can trigger",
    )
    parser.add_argument(
        "--hot-tag-partner-lambda-scale",
        type=float,
        default=0.5,
        help="Scaling factor applied to controller lambda_c when partner tags in",
    )
    parser.add_argument(
        "--hot-tag-partner-mass-scale",
        type=float,
        default=0.15,
        help="Scaling factor applied to controller mass_scale for partner",
    )
    parser.add_argument(
        "--hot-tag-partner-salience-floor",
        type=float,
        default=0.6,
        help="Minimum salience floor enforced for partner controller",
    )
    parser.add_argument(
        "--hot-tag-assist-gain",
        type=float,
        default=0.0,
        help="Assist gain applied during hot tag (simulates partner subsidy)",
    )
    parser.add_argument(
        "--hot-tag-auto-start-steps",
        type=int,
        default=0,
        help="Preload hot tag for a fixed number of steps at simulation start",
    )
    parser.add_argument(
        "--assist-poincare",
        action="store_true",
        help="Enable Poincaré-gated assist windows keyed on Lorenz crossings",
    )
    parser.add_argument(
        "--assist-axis",
        type=int,
        default=0,
        help="Lorenz state axis (0=x,1=y,2=z) used for Poincaré surface",
    )
    parser.add_argument(
        "--assist-threshold",
        type=float,
        default=0.0,
        help="Threshold value of selected Lorenz axis defining the Poincaré surface",
    )
    parser.add_argument(
        "--assist-direction",
        type=int,
        choices=[-1, 1],
        default=1,
        help="Crossing direction required to open assist window (1=positive, -1=negative)",
    )
    parser.add_argument(
        "--assist-window",
        type=int,
        default=0,
        help="Assist window length in steps after a qualifying crossing",
    )
    parser.add_argument(
        "--assist-amplitude",
        type=float,
        default=0.0,
        help="Amplitude for assist pulse applied during hope-spot windows",
    )
    parser.add_argument(
        "--assist-width",
        type=int,
        default=1,
        help="Number of steps for each assist pulse",
    )
    parser.add_argument(
        "--assist-cooldown",
        type=int,
        default=15,
        help="Cooldown steps between scripted assist pulses",
    )
    parser.add_argument(
        "--assist-salience-floor",
        type=float,
        default=0.4,
        help="Minimum salience required to permit assist pulses",
    )
    parser.add_argument(
        "--assist-slope-min",
        type=float,
        default=0.0,
        help="Minimum authority slope (averaged) required to permit assist pulses",
    )
    parser.add_argument(
        "--hope-lower",
        type=float,
        default=0.7,
        help="Lower bound for authority hope-spot window",
    )
    parser.add_argument(
        "--hope-upper",
        type=float,
        default=0.95,
        help="Upper bound for authority hope-spot window",
    )
    parser.add_argument(
        "--hope-min-slope",
        type=float,
        default=0.0,
        help="Minimum slope within hope-spot to consider it valid",
    )
    parser.add_argument(
        "--authority-break-window",
        type=int,
        default=10,
        help="Window size (steps) for authority average when detecting breaks",
    )
    parser.add_argument(
        "--salience-weight-novelty",
        type=float,
        default=0.6,
        help="Weight applied to novelty term inside salience value computation",
    )
    parser.add_argument(
        "--salience-weight-retention",
        type=float,
        default=0.25,
        help="Weight applied to retention term inside salience value computation",
    )
    parser.add_argument(
        "--salience-weight-payoff",
        type=float,
        default=0.15,
        help="Weight applied to payoff term inside salience value computation",
    )
    parser.add_argument(
        "--gate-bias",
        type=float,
        default=0.5,
        help="Sigmoid gate bias for salience gating",
    )
    parser.add_argument(
        "--gate-continuity",
        type=float,
        default=4.0,
        help="Continuity sensitivity coefficient for gate",
    )
    parser.add_argument(
        "--gate-fatigue",
        type=float,
        default=3.0,
        help="Fatigue penalty coefficient for gate",
    )
    parser.add_argument(
        "--gate-temp",
        type=float,
        default=0.06,
        help="Temperature (softness) for gate sigmoid",
    )
    parser.add_argument(
        "--gate-floor",
        type=float,
        default=0.1,
        help="Minimum gate value to prevent total shutdown",
    )
    parser.add_argument(
        "--credit-gain",
        type=float,
        default=6.0,
        help="Gain applied to controller credit logistic argument",
    )
    parser.add_argument(
        "--credit-bias",
        type=float,
        default=0.4,
        help="Bias applied to controller credit logistic argument",
    )
    parser.add_argument(
        "--credit-floor",
        type=float,
        default=0.1,
        help="Minimum credit factor allowed",
    )
    parser.add_argument(
        "--credit-alpha",
        type=float,
        default=0.65,
        help="Weighting factor for measurement delta in controller credit",
    )
    parser.add_argument(
        "--continuity-scale",
        type=float,
        default=1.0,
        help="Scaling for continuity strain in salience gate",
    )
    parser.add_argument("--lorenz-sigma", type=float, default=10.0, help="Lorenz sigma parameter")
    parser.add_argument("--lorenz-rho", type=float, default=28.0, help="Lorenz rho parameter")
    parser.add_argument("--lorenz-beta", type=float, default=8.0 / 3.0, help="Lorenz beta parameter")
    parser.add_argument(
        "--salience-channels",
        type=int,
        default=5,
        help="Number of salience channels (stress mode requires ≥5)",
    )
    parser.add_argument(
        "--salience-floor",
        type=float,
        default=0.0,
        help="Lower bound applied to all controller salience channels",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.lambda_values:
        try:
            lambda_list = [float(val.strip()) for val in args.lambda_values.split(",") if val.strip()]
        except ValueError as exc:
            raise ValueError("Failed to parse --lambda-values; ensure comma-separated floats") from exc
        if not lambda_list:
            raise ValueError("--lambda-values produced empty list")
    else:
        lambda_list = None

    if args.mode == "mass":
        lambdas = lambda_list or [0.0, 0.25, 0.5, 1.0, 2.0]
        template_cfg = ControllerConfig(
            salience_floor=args.salience_floor,
            salience_weight_novelty=args.salience_weight_novelty,
            salience_weight_retention=args.salience_weight_retention,
            salience_weight_payoff=args.salience_weight_payoff,
            salience_gate_bias=args.gate_bias,
            salience_gate_continuity=args.gate_continuity,
            salience_gate_fatigue=args.gate_fatigue,
            salience_gate_temp=args.gate_temp,
            salience_gate_floor=args.gate_floor,
            salience_credit_gain=args.credit_gain,
            salience_credit_bias=args.credit_bias,
            salience_credit_floor=args.credit_floor,
            salience_credit_alpha=args.credit_alpha,
            salience_continuity_scale=args.continuity_scale,
        )
        entries = run_mass_sweep(
            lambdas,
            seed=args.seed,
            salience_floor=args.salience_floor,
            controller_template=template_cfg,
        )
        artifact = write_artifact(entries)
        print("=== Experiment A: mass sweep ===")
        print_summary(entries)
        print(f"Results written to {artifact}")
    else:
        if args.salience_channels < 5:
            raise ValueError("Stress mode requires salience_channels ≥ 5")
        lambdas = lambda_list or [0.0, 5.0, 10.0, 20.0, 50.0]
        horizon = args.horizon if args.horizon is not None else 10.0
        dt = args.dt if args.dt is not None else 0.01
        plant_cfg = PlantConfig(
            dt=dt,
            horizon=horizon,
            noise_sigma=args.noise_sigma,
            noise_delay=max(0.0, args.noise_delay),
            breath_period=max(0.0, args.breath_period),
            breath_in_fraction=max(0.0, min(1.0, args.breath_in_fraction)),
            breath_in_scale=max(0.0, args.breath_in_scale),
            breath_out_scale=max(0.0, args.breath_out_scale),
            breath_pulse_count=max(1, args.breath_pulse_count),
            breath_pulse_width=float(np.clip(args.breath_pulse_width, 0.0, 1.0)),
            adaptive_breathing=bool(args.adaptive_breathing),
            adapt_authority_threshold=args.adapt_authority_threshold,
            adapt_salience_threshold=args.adapt_salience_threshold,
            adapt_scale_min=max(0.0, args.adapt_scale_min),
            adapt_scale_max=max(args.adapt_scale_min, args.adapt_scale_max),
            adapt_step_gain=max(0.0, args.adapt_step_gain),
            salience_breathing=bool(args.salience_breathing),
            salience_breath_floor=max(0.0, args.salience_breath_floor),
            salience_breath_gain=max(0.0, args.salience_breath_gain),
            salience_breath_slope_gain=max(0.0, args.salience_breath_slope_gain),
            salience_breath_hope_gain=max(0.0, args.salience_breath_hope_gain),
            salience_breath_smoothing=float(np.clip(args.salience_breath_smoothing, 0.0, 1.0)),
            salience_breath_recovery_scale=max(0.0, args.salience_breath_recovery_scale),
            salience_breath_attack_scale=max(args.salience_breath_recovery_scale, args.salience_breath_attack_scale),
            salience_osc_smoothing=float(np.clip(args.salience_osc_smoothing, 0.0, 1.0)),
            salience_osc_flip_gain=max(0.0, args.salience_osc_flip_gain),
            salience_osc_variance_gain=max(0.0, args.salience_osc_variance_gain),
            salience_lambda_gain=max(0.0, args.salience_lambda_gain),
            salience_gate_bias_gain=max(0.0, args.salience_gate_bias_gain),
            salience_gate_bias_damp=max(0.0, args.salience_gate_bias_damp),
            salience_gate_floor_gain=max(0.0, args.salience_gate_floor_gain),
            salience_assist_amp_floor=max(0.0, args.salience_assist_amp_floor),
            salience_assist_amp_gain=max(0.0, args.salience_assist_amp_gain),
            salience_assist_amp_max=max(args.salience_assist_amp_floor, args.salience_assist_amp_max),
            salience_assist_cooldown_gain=max(0.0, args.salience_assist_cooldown_gain),
            lorenz_gain=args.lorenz_gain,
            lorenz_sigma=args.lorenz_sigma,
            lorenz_rho=args.lorenz_rho,
            lorenz_beta=args.lorenz_beta,
            lorenz_gate_enabled=bool(args.lorenz_gate),
            lorenz_gate_threshold=args.lorenz_gate_threshold,
            lorenz_gate_release_steps=max(1, args.lorenz_gate_release_steps),
            lorenz_ramp_steps=max(1, args.lorenz_ramp_steps),
            lorenz_hot_tag_enabled=bool(args.lorenz_hot_tag),
            lorenz_hot_tag_window=max(2, args.lorenz_hot_tag_window),
            lorenz_hot_tag_slope_threshold=args.lorenz_hot_tag_slope_threshold,
            lorenz_hot_tag_pulse_steps=max(1, args.lorenz_hot_tag_pulse_steps),
            lorenz_hot_tag_cooldown_steps=max(1, args.lorenz_hot_tag_cooldown_steps),
            lorenz_hot_tag_control_boost=max(0.0, args.lorenz_hot_tag_control_boost),
            hot_tag_heat_steps=max(1, args.hot_tag_heat_steps),
            hot_tag_heat_authority=args.hot_tag_heat_authority,
            hot_tag_near_tag_threshold=args.hot_tag_near_tag_threshold,
            hot_tag_near_tag_upper=max(args.hot_tag_near_tag_threshold, args.hot_tag_near_tag_upper),
            hot_tag_near_tag_attempts=max(1, args.hot_tag_near_tag_attempts),
            hot_tag_partner_lambda_scale=max(0.0, args.hot_tag_partner_lambda_scale),
            hot_tag_partner_mass_scale=max(0.0, args.hot_tag_partner_mass_scale),
            hot_tag_partner_salience_floor=max(0.0, args.hot_tag_partner_salience_floor),
            hot_tag_assist_gain=max(0.0, args.hot_tag_assist_gain),
            hot_tag_auto_start_steps=max(0, args.hot_tag_auto_start_steps),
            assist_poincare_enabled=bool(args.assist_poincare),
            assist_poincare_axis=args.assist_axis,
            assist_poincare_threshold=args.assist_threshold,
            assist_poincare_direction=args.assist_direction,
            assist_poincare_window=max(0, args.assist_window),
            hope_spot_lower=args.hope_lower,
            hope_spot_upper=args.hope_upper,
            hope_spot_min_slope=args.hope_min_slope,
            authority_break_window=max(1, args.authority_break_window),
            assist_amplitude=max(0.0, args.assist_amplitude),
            assist_pulse_width=max(1, args.assist_width),
            assist_cooldown_steps=max(0, args.assist_cooldown),
            assist_salience_floor=max(0.0, args.assist_salience_floor),
            assist_authority_slope_min=args.assist_slope_min,
        )
        template_cfg = ControllerConfig(
            salience_channels=plant_cfg.salience_channels,
            salience_floor=args.salience_floor,
            salience_weight_novelty=args.salience_weight_novelty,
            salience_weight_retention=args.salience_weight_retention,
            salience_weight_payoff=args.salience_weight_payoff,
            salience_gate_bias=args.gate_bias,
            salience_gate_continuity=args.gate_continuity,
            salience_gate_fatigue=args.gate_fatigue,
            salience_gate_temp=args.gate_temp,
            salience_gate_floor=args.gate_floor,
            salience_credit_gain=args.credit_gain,
            salience_credit_bias=args.credit_bias,
            salience_credit_floor=args.credit_floor,
            salience_credit_alpha=args.credit_alpha,
            salience_continuity_scale=args.continuity_scale,
        )
        entries = run_stress_sweep(
            lambdas,
            plant_cfg,
            seed=args.seed,
            salience_floor=args.salience_floor,
            controller_template=template_cfg,
        )
        artifact = write_artifact(
            entries,
            experiment_name="experiment_p_stress",
            prefix="mass_stress",
        )
        print("=== Experiment P: stress sweep ===")
        print_summary(entries)
        print(f"Stress results written to {artifact}")


if __name__ == "__main__":
    main()
