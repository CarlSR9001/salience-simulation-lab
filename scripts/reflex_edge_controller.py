"""Experiment N: Reflex edge controller with low-pass merge and salience gating."""

from __future__ import annotations

import json
import math
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from collections import deque

import numpy as np

from salience_floor_gate import GateTelemetry, gate_subsidy

ARTIFACT_DIR = Path("artifacts/reflex_edge")
TRACE_DIR = ARTIFACT_DIR / "traces"
PROFILE_DIR = Path("profiles")
BASELINE_CACHE: Dict[str, Dict[str, float]] = {}
CALIBRATION_CACHE: Dict[str, CalibrationStats] = {}


@dataclass
class PlantConfig:
    dt: float = 0.01
    horizon: float = 5.0
    tau: float = 0.1
    target: float = 1.0


@dataclass
class ControllerConfig:
    kp: float = 0.36
    core_gain: float = 1.6
    edge_gain: float = 0.6
    salience_scale_core: float = 0.06
    salience_scale_edge: float = 0.12
    fatigue_decay: float = 0.9
    fatigue_gain: float = 0.08
    lambda_core: float = 0.24
    lambda_edge: float = 0.0
    mu_edge: float = 0.0
    merge_rate: float = 0.06
    salience_floor: float = 0.7
    recovery_tax: float = 0.2
    edge_feedforward: float = 0.35
    core_follow_gain: float = 0.858
    core_target_gain: float = 1.0
    control_target_gain: float = 0.5
    core_target_blend: float = 0.35
    core_integral_gain: float = 0.06
    core_integral_leak: float = 0.006
    core_integral_limit: float = 1.8
    control_integral_gain: float = 0.25
    control_integral_leak: float = 0.02
    merge_target_blend: float = 0.15
    core_high_salience_gain: float = 0.4
    core_high_salience_ctrl_kp: float = 0.4356
    core_high_salience_ctrl_ki: float = 0.06
    core_high_salience_ctrl_leak: float = 0.015
    core_feedforward_gain: float = 0.14
    degeneracy_tau_ms: float = 120.0
    degeneracy_rho_crit: float = 0.82
    degeneracy_kappa: float = 1.0
    degeneracy_gamma: float = 1.5
    degeneracy_beta: float = 0.35
    degeneracy_eta: float = 0.4
    curvature_tau_ms: float = 150.0
    curvature_target_init: float = 0.3
    curvature_anneal_ms: float = 600.0
    curvature_bias: float = 0.25
    profile_name: str = "default"
    plant_step_scale: float = 1.2
    gate_target_init: float = 0.62
    gate_target_final: float = 0.92
    gate_ramp_ms: float = 200.0
    edge_salience_seed: float = 0.78
    seed_duration_ms: float = 250.0
    traction_alpha: float = 0.22
    traction_decay_ms: float = 120.0
    credit_tau_ms: float = 90.0
    credit_leak_ms: float = 160.0
    credit_open_threshold: float = 0.85
    min_on_ms: float = 120.0
    min_off_ms: float = 120.0
    coupling_max: float = 1.08
    z_gate_thresh: float = 0.73
    gate_tau: float = 0.12
    traction_pulse_ms: float = 90.0
    traction_pulse_strength: float = 0.38
    traction_pulse_salience_low: float = 0.68
    traction_pulse_salience_high: float = 0.80
    traction_pulse_credit_ms: float = 20.0
    pulse_feedforward_gain: float = 0.12
    pulse_feedforward_clip: float = 0.6
    post_pulse_window: float = 0.12
    post_pulse_gain_bonus: float = 0.25
    post_pulse_merge_bonus: float = 0.35
    pulse_salience_floor: float = 0.78
    micro_burst_enabled: bool = True
    micro_burst_delay_ms: float = 50.0
    micro_burst_cooldown_ms: float = 300.0
    micro_burst_slope_baseline: float = 6.0
    micro_burst_salience_floor: float = 0.76
    micro_burst_coupling_max: float = 1.05
    stall_slope_min: float = 0.15
    stall_window_ms: float = 80.0
    stall_ladder_ms: float = 120.0
    stall_integral_eps: float = 1e-4
    stall_ladder_levels: Tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)
    pulse_cooldown_ms: float = 80.0
    pulse_cooldown_min_ms: float = 40.0
    pulse_cooldown_max_ms: float = 220.0


@dataclass
class IdentityProfile:
    name: str
    speed_affinity: float = 0.5
    salience_weights: Tuple[float, float, float] = (0.5, 0.25, 0.25)
    salience_on_z: float = 0.5
    salience_off_z: float = 0.0
    salience_floor_z: float = 0.0
    merge_preference: float = 0.06
    mu_edge_bounds: Tuple[float, float] = (0.8, 2.2)
    feedforward_default: float = 0.12
    pulse_fast: Tuple[int, int] = (24, 8)
    pulse_craft: Tuple[int, int] = (16, 16)

    @property
    def gamma(self) -> float:
        return 1.2 + 0.6 * float(np.clip(self.speed_affinity, 0.0, 1.0))


@dataclass
class CalibrationStats:
    mean_salience: float
    std_salience: float
    baseline_z: float


def _safe_tuple(values: Iterable[float], length: int, default: Tuple[float, ...]) -> Tuple[float, ...]:
    vals = list(values)
    if len(vals) != length:
        return default
    return tuple(float(v) for v in vals)


def load_identity_profile(name: str) -> IdentityProfile:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = PROFILE_DIR / f"identity_{name}.json"
    if not path.exists():
        return IdentityProfile(name=name)

    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    speed_affinity = float(np.clip(data.get("speed_affinity", 0.5), 0.0, 1.0))
    salience_weights = _safe_tuple(data.get("salience_weights", (0.5, 0.25, 0.25)), 3, (0.5, 0.25, 0.25))
    total = sum(salience_weights)
    if total <= 1e-6:
        salience_weights = (0.5, 0.25, 0.25)
    else:
        salience_weights = tuple(w / total for w in salience_weights)
    salience_floor_z = float(data.get("salience_floor_z", 0.0))
    salience_on_z = float(data.get("salience_on_z", 0.5))
    salience_off_z = float(data.get("salience_off_z", 0.0))
    merge_preference = float(data.get("merge_preference", 0.06))
    mu_edge_bounds = _safe_tuple(data.get("mu_edge_bounds", (0.8, 2.2)), 2, (0.8, 2.2))
    feedforward_default = float(data.get("k_ff", 0.12))
    pulse_fast = _safe_tuple(data.get("pulse_fast", (24, 8)), 2, (24, 8))
    pulse_craft = _safe_tuple(data.get("pulse_craft", (16, 16)), 2, (16, 16))

    return IdentityProfile(
        name=name,
        speed_affinity=speed_affinity,
        salience_weights=salience_weights,
        salience_on_z=salience_on_z,
        salience_off_z=salience_off_z,
        salience_floor_z=salience_floor_z,
        merge_preference=merge_preference,
        mu_edge_bounds=mu_edge_bounds,
        feedforward_default=feedforward_default,
        pulse_fast=pulse_fast,
        pulse_craft=pulse_craft,
    )


@dataclass
class SimulationResult:
    config: ControllerConfig
    edge_response_time_90: float
    core_response_time_90: float
    core_response_time_70: float
    core_response_time_50: float
    core_log_slope_50: float
    core_slope_50_70: float
    core_ladder_best: float
    core_ladder_rt: float
    core_area_error: float
    accel_score: float
    accel_score_alt: float
    accel_70_energy: float
    slope_gain_midband: float
    settle_time_5: float
    total_flux: float
    escape_time: float
    escape_flux: float
    escape_flux_window: float
    escape_fraction: float
    escape_time_gain: float
    escape_flux_gain: float
    degeneracy_pressure_mean: float
    degeneracy_pressure_peak: float
    curvature_bias_mean: float
    curvature_bias_std: float
    micro_burst_pending_pct: float
    micro_burst_active_pct: float
    micro_burst_cooldown_mean: float
    area_ratio: float
    utility_speed: float
    utility_quality: float
    utility_total: float
    core_salience_min: float
    edge_salience_min: float
    edge_salience_mean: float
    edge_salience_max: float
    edge_salience_coverage: float
    gate_open_pct: float
    gate_target_final: float
    gate_target_mean: float
    credit_peak: float
    credit_final: float
    traction_mean: float
    traction_peak: float
    mu_eff_mean: float
    mu_eff_peak: float
    pulse_activation_pct: float
    post_pulse_timer_final: float
    post_pulse_window_final: float
    seed_coverage: float
    slope_ema_mean: float
    cooldown_mean: float
    conditional_boost_pct: float
    pulse_active_pct: float
    ladder_progress_pct: float
    on_timer_final_ms: float
    off_timer_final_ms: float
    coupling_max_observed: float
    cross_coupling_error: float
    identity_violation: bool
    acceleration_pct: float
    blocked_pct: float
TRACE_PROBES: List[Tuple[str, ControllerConfig]] = [
    (
        "baseline_core_light",
        ControllerConfig(
            mu_edge=0.0,
            merge_rate=0.05,
            lambda_core=0.2,
            core_follow_gain=0.5,
            core_target_gain=1.0,
            control_target_gain=0.6,
            core_target_blend=0.45,
            core_integral_gain=0.6,
            core_integral_leak=0.04,
            control_integral_gain=0.3,
            control_integral_leak=0.02,
            merge_target_blend=0.25,
            core_high_salience_gain=0.5,
            core_high_salience_ctrl_kp=0.35,
            core_high_salience_ctrl_ki=0.06,
        ),
    ),
    (
        "mu2_salience_push",
        ControllerConfig(
            mu_edge=2.0,
            merge_rate=0.05,
            lambda_core=0.2,
            core_follow_gain=0.7,
            core_target_gain=1.0,
            control_target_gain=0.6,
            core_target_blend=0.45,
            core_integral_gain=0.6,
            core_integral_leak=0.04,
            control_integral_gain=0.3,
            control_integral_leak=0.02,
            merge_target_blend=0.25,
            core_high_salience_gain=0.5,
            core_high_salience_ctrl_kp=0.4,
            core_high_salience_ctrl_ki=0.08,
        ),
    ),
    (
        "heavy_lambda_core",
        ControllerConfig(
            mu_edge=4.0,
            merge_rate=0.1,
            lambda_core=0.5,
            core_follow_gain=0.7,
            core_target_gain=1.0,
            control_target_gain=0.3,
            core_target_blend=0.45,
            core_integral_gain=0.6,
            core_integral_leak=0.07,
            control_integral_gain=0.18,
            control_integral_leak=0.035,
            merge_target_blend=0.15,
            core_high_salience_gain=0.4,
            core_high_salience_ctrl_kp=0.3,
            core_high_salience_ctrl_ki=0.05,
        ),
    ),
]

class GateTimers:
    __slots__ = ("on_ms", "off_ms")

    def __init__(self) -> None:
        self.on_ms = 0.0
        self.off_ms = 0.0

    def reset(self) -> None:
        self.on_ms = 0.0
        self.off_ms = 0.0

    def step(self, gate_open: bool, dt_ms: float) -> None:
        if gate_open:
            self.on_ms += dt_ms
            self.off_ms = 0.0
        else:
            self.off_ms += dt_ms
            self.on_ms = 0.0


class CreditIntegrator:
    __slots__ = ("value", "tau_ms", "leak_ms", "threshold")

    def __init__(self, tau_ms: float, leak_ms: float, threshold: float) -> None:
        self.value = 0.0
        self.tau_ms = tau_ms
        self.leak_ms = leak_ms
        self.threshold = threshold

    def reset(self) -> None:
        self.value = 0.0

    def step(self, salience_edge: float, target: float, dt_ms: float) -> float:
        delta = salience_edge - target
        if delta > 0.0:
            self.value += delta * (dt_ms / max(self.tau_ms, 1.0))
        else:
            self.value += delta * (dt_ms / max(self.leak_ms, 1.0))
        self.value = float(np.clip(self.value, 0.0, 2.0))
        return self.value

    def satisfied(self) -> bool:
        return self.value >= self.threshold


class ReflexEdgeController:
    def __init__(
        self,
        cfg: ControllerConfig,
        plant_cfg: PlantConfig,
        profile: IdentityProfile | None = None,
        calibration: CalibrationStats | None = None,
        calibrating: bool = False,
    ) -> None:
        self.cfg = cfg
        self.plant_cfg = plant_cfg
        self.dt = plant_cfg.dt
        self.profile = profile or load_identity_profile(cfg.profile_name)
        self.calibrating = calibrating
        self.calibration = calibration
        if self.calibration is None and not self.calibrating:
            self.calibration = calibrate_profile(self.profile, plant_cfg)
        if self.calibration is None:
            self.calibration = CalibrationStats(mean_salience=0.72, std_salience=0.02, baseline_z=0.0)
        if self.cfg.core_feedforward_gain <= 0.0:
            self.cfg.core_feedforward_gain = self.profile.feedforward_default
        if self.cfg.merge_rate <= 0.0:
            self.cfg.merge_rate = self.profile.merge_preference
        if self.cfg.gate_target_final < 0.92:
            self.cfg.gate_target_final = 0.92
        self.reset()

    def reset(self) -> None:
        self.core_state = 0.0
        self.edge_state = 0.0
        self.prev_error = 0.0
        self.salience_core = 1.0
        self.salience_edge = 0.95
        self.retention_core = 0.9
        self.retention_edge = 0.85
        self.payoff_core = 0.9
        self.payoff_edge = 0.9
        self.fatigue_core = 0.0
        self.fatigue_edge = 0.0
        self.core_integral = 0.0
        self.control_integral = 0.0
        self.high_salience_integral = 0.0
        self.edge_salience_ema = max(0.78, self.salience_edge)
        self.edge_salience_prev = self.edge_salience_ema
        self.edge_salience_baseline = max(0.72, self.edge_salience_ema)
        self.edge_gate_on = False
        self.edge_gate_timer = 0.0
        self.edge_credit = 0.4
        self.blue_noise_phase = 0.0
        self.traction_timer = 0.0
        self.salience_count = 1
        self.salience_mean = self.calibration.mean_salience
        variance = max(self.calibration.std_salience ** 2, 1e-6)
        self.salience_M2 = variance
        self.salience_std = math.sqrt(variance)
        self.salience_z_prev = self.calibration.baseline_z
        self.salience_z_baseline = self.calibration.baseline_z
        self.current_step_ms = 0.0
        self.gate_timers = GateTimers()
        self.credit_int = CreditIntegrator(
            tau_ms=self.cfg.credit_tau_ms,
            leak_ms=self.cfg.credit_leak_ms,
            threshold=self.cfg.credit_open_threshold,
        )
        self.credit_peak = 0.0
        self.traction_peak = 0.0
        self.traction_sum = 0.0
        self.mu_eff_sum = 0.0
        self.mu_eff_energy = 0.0
        self.mu_eff_peak = 0.0
        self.mu_eff_samples = 0
        self.pulse_active_steps = 0
        self.post_pulse_timer = 0.0
        self.post_pulse_window_timer = 0.0
        self.gate_target_sum = 0.0
        self.seed_steps = 0
        self.gate_open_steps = 0
        self.coupling_max = 0.0
        self.total_steps = 0
        self.prev_credit = 0.0
        self.credit_rising_timer_ms = 0.0
        self.seed_phase_active = True
        self.seed_floor_mu = self.cfg.mu_edge
        self.degeneracy_rho = self.salience_core
        self.degeneracy_pressure = 0.0
        self.curvature_potential = 0.0
        self.curvature_bias_state = 0.0
        self.curvature_mean = 0.0
        self.curvature_var = 1e-6
        self.micro_burst_cooldown = 0.0
        self.micro_burst_pending = False
        self.micro_burst_timer = 0.0
        self.micro_burst_active_timer = 0.0
        self.cooldown_timer = 0.0
        cooldown_ms = float(np.clip(
            self.cfg.pulse_cooldown_ms,
            self.cfg.pulse_cooldown_min_ms,
            self.cfg.pulse_cooldown_max_ms,
        ))
        self.cooldown_current = cooldown_ms / 1000.0
        self.last_pulse_time = -1e6
        self.last_ladder_level = 0.0
        self.last_ladder_time = 0.0
        self.core_integral_prev = 0.0
        self.integral_delta_smoothed = 0.0
        self.slope_window_steps = max(1, int(round((self.cfg.stall_window_ms / 1000.0) / max(self.dt, 1e-6))))
        self.slope_buffer = deque(maxlen=self.slope_window_steps + 1)
        self.slope_ema = 0.0
        self.ladder_advanced_since_last_pulse = False
        self.ladder_stagnant_ms = 0.0

    def _update_salience(self, component: str, delta: float, error: float) -> None:
        w1, w2, w3 = self.profile.salience_weights
        if component == "core":
            scale = self.cfg.salience_scale_core
            novelty = math.exp(np.clip(-abs(delta) / scale, -10.0, 0.0))
            payoff = math.exp(np.clip(-abs(error), -10.0, 0.0))
            self.retention_core = 0.9 * self.retention_core + 0.1 * novelty
            self.payoff_core = 0.9 * self.payoff_core + 0.1 * payoff
            self.fatigue_core = self.cfg.fatigue_decay * self.fatigue_core + self.cfg.fatigue_gain * abs(delta) / scale
            phi = min(self.fatigue_core, 0.95)
            retention_term = np.clip(self.retention_core, 0.3, 1.2)
            payoff_term = np.clip(self.payoff_core, 0.3, 1.2)
            preference = w1 * novelty + w2 * retention_term + w3 * payoff_term
            salience = preference * (1.0 - phi)
            self.salience_core = float(np.clip(0.8 + 0.35 * salience, 0.75, 1.3))
        else:
            scale = self.cfg.salience_scale_edge
            novelty = math.exp(np.clip(-abs(delta) / (scale * 2.5), -10.0, 0.0))
            payoff = math.exp(np.clip(-abs(error), -10.0, 0.0))
            self.retention_edge = 0.88 * self.retention_edge + 0.12 * novelty
            self.payoff_edge = 0.9 * self.payoff_edge + 0.1 * payoff
            self.fatigue_edge = self.cfg.fatigue_decay * self.fatigue_edge + self.cfg.fatigue_gain * abs(delta) / scale
            phi = min(self.fatigue_edge, 0.95)
            retention_term = np.clip(self.retention_edge, 0.25, 1.2)
            payoff_term = np.clip(self.payoff_edge, 0.3, 1.2)
            preference = w1 * novelty + w2 * retention_term + w3 * payoff_term
            salience = preference * (1.0 - phi)
            self.salience_edge = float(np.clip(0.68 + 0.45 * salience, 0.65, 1.3))

    def step(self, error: float, measurement: float, telemetry: GateTelemetry, mux_enabled: bool = True) -> Dict[str, float]:
        error_clipped = float(np.clip(error, -5.0, 5.0))
        target_estimate = measurement + error

        raw_core_delta = error_clipped * self.dt
        mass_core = 1.0 + self.cfg.lambda_core * self.salience_core
        core_delta = raw_core_delta / mass_core
        self.core_state += core_delta
        self._update_salience("core", core_delta, error)

        tau_deg = max(self.cfg.degeneracy_tau_ms / 1000.0, 1e-3)
        alpha_deg = self.dt / (self.dt + tau_deg)
        self.degeneracy_rho = (1.0 - alpha_deg) * self.degeneracy_rho + alpha_deg * self.salience_core
        rho_excess = max(0.0, self.degeneracy_rho - self.cfg.degeneracy_rho_crit)
        self.degeneracy_pressure = self.cfg.degeneracy_kappa * (rho_excess ** self.cfg.degeneracy_gamma)

        tau_curv = max(self.cfg.curvature_tau_ms / 1000.0, 1e-3)
        alpha_curv = self.dt / (self.dt + tau_curv)
        curvature_sample = self.salience_core - self.salience_edge
        self.curvature_mean = (1.0 - alpha_curv) * self.curvature_mean + alpha_curv * curvature_sample
        delta_curv = curvature_sample - self.curvature_mean
        self.curvature_var = max((1.0 - alpha_curv) * self.curvature_var + alpha_curv * (delta_curv ** 2), 1e-6)
        curvature_std = math.sqrt(self.curvature_var)
        normalized_curvature = delta_curv / curvature_std

        target_curv = self.cfg.curvature_target_init
        if self.cfg.curvature_anneal_ms > 0:
            anneal_progress = min(self.current_step_ms / max(self.cfg.curvature_anneal_ms, 1.0), 1.0)
            target_curv = self.cfg.curvature_target_init * (1.0 - anneal_progress)
        curvature_error = normalized_curvature - target_curv
        self.curvature_bias_state = (1.0 - alpha_curv) * self.curvature_bias_state + alpha_curv * curvature_error
        curvature_bias = self.cfg.curvature_bias * self.curvature_bias_state

        gate_target = self.cfg.gate_target_final
        seed_active_flag = False
        traction = 0.0
        mu_base = float(np.clip(self.cfg.mu_edge, self.profile.mu_edge_bounds[0], self.profile.mu_edge_bounds[1]))
        mu_base += curvature_bias
        mu_base = float(np.clip(mu_base, self.profile.mu_edge_bounds[0], self.profile.mu_edge_bounds[1]))
        credit_val = self.prev_credit
        coupling = abs(self.edge_state - self.core_state)
        lane_open = False
        should_open = False
        raw_edge_delta = 0.0

        if mux_enabled:
            raw_edge_delta = (error_clipped - self.prev_error)
            raw_edge_delta = float(np.clip(raw_edge_delta, -1.5, 1.5))

            beta = 1.0 - math.exp(-self.dt / max(self.cfg.gate_tau, 1e-3))
            self.edge_salience_prev = self.edge_salience_ema
            self.edge_salience_ema = (1.0 - beta) * self.edge_salience_ema + beta * self.salience_edge

            dt_ms = self.dt * 1000.0
            self.current_step_ms += dt_ms
            if self.current_step_ms <= self.cfg.gate_ramp_ms:
                frac = self.current_step_ms / max(self.cfg.gate_ramp_ms, 1.0)
                gate_target = self.cfg.gate_target_init + frac * (self.cfg.gate_target_final - self.cfg.gate_target_init)
            self.gate_target_sum += gate_target

            seed_active_flag = self.current_step_ms <= self.cfg.seed_duration_ms
            if seed_active_flag:
                self.salience_edge = max(self.salience_edge, self.cfg.edge_salience_seed)
                self.seed_steps += 1
                self.seed_phase_active = True
            elif self.seed_phase_active:
                decay_ms = 200.0
                elapsed = self.current_step_ms - self.cfg.seed_duration_ms
                if decay_ms > 0:
                    decay_ratio = np.clip(elapsed / decay_ms, 0.0, 1.0)
                    self.seed_floor_mu = self.cfg.mu_edge * (1.0 - decay_ratio) + self.cfg.mu_edge
                if elapsed >= decay_ms:
                    self.seed_phase_active = False

            # Incremental statistics for z-space thresholds
            self.salience_count += 1
            delta = self.edge_salience_ema - self.salience_mean
            self.salience_mean += delta / self.salience_count
            delta2 = self.edge_salience_ema - self.salience_mean
            self.salience_M2 += delta * delta2
            variance = self.salience_M2 / max(self.salience_count - 1, 1)
            self.salience_std = math.sqrt(max(variance, 1e-6))

            z_value = (self.edge_salience_ema - self.salience_mean) / self.salience_std
            dz_dt = (z_value - self.salience_z_prev) / max(self.dt, 1e-6)
            self.salience_z_prev = z_value
            self.salience_z_baseline = max(self.salience_z_baseline, z_value)

            profile = self.profile
            z_gate = self.cfg.z_gate_thresh
            coupling = abs(self.edge_state - self.core_state)
            self.coupling_max = max(self.coupling_max, coupling)
            credit_val = self.credit_int.step(self.salience_edge, gate_target, dt_ms)
            self.credit_peak = max(self.credit_peak, credit_val)

            if credit_val > self.prev_credit:
                self.credit_rising_timer_ms += dt_ms
            else:
                self.credit_rising_timer_ms = max(self.credit_rising_timer_ms - dt_ms, 0.0)
            self.prev_credit = credit_val

            z_slack = z_value >= (z_gate - 0.15)
            credit_rising_ok = self.credit_rising_timer_ms >= 50.0

            z_ok = z_value >= z_gate
            near_open = (self.salience_edge + 0.01) >= gate_target or (z_slack and credit_rising_ok)
            credit_ok = credit_val >= self.cfg.credit_open_threshold
            should_open = (
                z_ok
                and (credit_ok or near_open)
                and coupling <= self.cfg.coupling_max
            )
            lane_open = self.edge_gate_on and z_value >= profile.salience_floor_z and core_gate_ok
        else:
            self.credit_rising_timer_ms = max(self.credit_rising_timer_ms - self.dt * 1000.0, 0.0)

        if self.micro_burst_active_timer > 0.0:
            self.micro_burst_active_timer = max(self.micro_burst_active_timer - self.dt, 0.0)

        base_mu = mu_base
        if seed_active_flag:
            base_mu = max(base_mu, 0.45)
        gamma = self.profile.gamma
        if mux_enabled and (lane_open or should_open):
            mu_eff = base_mu * (credit_val ** gamma)
        else:
            mu_eff = base_mu

        pulse_active = False
        pulse_contrib = 0.0

        if lane_open or should_open:
            decay_scale = max(0.0, 1.0 - self.current_step_ms / max(self.cfg.traction_decay_ms, 1.0))
            traction = self.cfg.traction_alpha * max(0.0, gate_target - self.salience_edge) * decay_scale
            mu_eff += traction
            self.traction_timer = max(self.traction_timer - self.dt, 0.0)
        elif self.traction_timer > 0.0:
            decay_scale = max(0.0, self.traction_timer / max(self.cfg.traction_decay_ms / 1000.0, 1e-3))
            traction = self.cfg.traction_alpha * decay_scale
            mu_eff = max(mu_eff, traction)
            self.traction_timer -= self.dt

        self.slope_buffer.append(self.core_state)
        slope_sample = 0.0
        if len(self.slope_buffer) >= 2:
            slope_sample = (self.slope_buffer[-1] - self.slope_buffer[0]) / max((len(self.slope_buffer) - 1) * self.dt, 1e-6)
        alpha_slope = 0.2
        self.slope_ema = (1.0 - alpha_slope) * self.slope_ema + alpha_slope * slope_sample

        # Ladder tracking
        advanced_this_step = False
        ladder_level = 0.0
        for rel in self.cfg.stall_ladder_levels:
            if self.core_state >= rel * self.cfg.gate_target_final:
                ladder_level = rel
        if ladder_level > self.last_ladder_level + 1e-6:
            advanced_this_step = True
            self.ladder_advanced_since_last_pulse = True
            self.last_ladder_time = self.current_step_ms
            self.last_ladder_level = ladder_level
        else:
            self.ladder_stagnant_ms = max(self.current_step_ms - self.last_ladder_time, 0.0)

        integral_delta = self.core_integral - self.core_integral_prev
        self.core_integral_prev = self.core_integral
        beta_integral = 0.2
        self.integral_delta_smoothed = (1.0 - beta_integral) * self.integral_delta_smoothed + beta_integral * integral_delta

        stalled_votes = 0
        slope_threshold = self.cfg.stall_slope_min
        if self.slope_ema < slope_threshold:
            stalled_votes += 1
        if self.ladder_stagnant_ms >= self.cfg.stall_ladder_ms:
            stalled_votes += 1
        if self.integral_delta_smoothed <= self.cfg.stall_integral_eps:
            stalled_votes += 1

        sal_low = min(self.cfg.traction_pulse_salience_low, self.cfg.traction_pulse_salience_high)
        sal_high = max(self.cfg.traction_pulse_salience_low, self.cfg.traction_pulse_salience_high)
        in_window = sal_low <= self.salience_edge <= sal_high
        time_since_last = (self.current_step_ms / 1000.0) - self.last_pulse_time
        cooldown_ok = time_since_last >= self.cooldown_current
        credit_ok = credit_val >= self.cfg.credit_open_threshold

        should_pulse = (
            not lane_open
            and in_window
            and cooldown_ok
            and stalled_votes >= 2
            and credit_ok
            and coupling <= self.cfg.micro_burst_coupling_max
        )

        if should_pulse:
            pulse_window = self.cfg.traction_pulse_ms / 1000.0
            self.traction_timer = max(self.traction_timer, pulse_window)
            self.post_pulse_timer = max(self.post_pulse_timer, pulse_window)
            self.post_pulse_window_timer = max(self.post_pulse_window_timer, self.cfg.post_pulse_window)
            pulse_active = True
            self.salience_edge = max(self.salience_edge, self.cfg.pulse_salience_floor)
            self.micro_burst_pending = False
            self.micro_burst_timer = 0.0
            self.micro_burst_cooldown = max(self.micro_burst_cooldown, self.cfg.micro_burst_cooldown_ms / 1000.0)
            self.micro_burst_active_timer = pulse_window
            if self.ladder_advanced_since_last_pulse:
                self.cooldown_current = min(
                    self.cfg.pulse_cooldown_max_ms / 1000.0,
                    self.cooldown_current * 1.25,
                )
            else:
                self.cooldown_current = max(
                    self.cfg.pulse_cooldown_min_ms / 1000.0,
                    self.cooldown_current * 0.8,
                )
            self.last_pulse_time = self.current_step_ms / 1000.0
            self.ladder_advanced_since_last_pulse = False

        if self.traction_timer > 0.0:
            tau = max(self.cfg.traction_pulse_ms / 1000.0, 1e-3)
            pulse_contrib = self.cfg.traction_pulse_strength * math.exp(- (tau - self.traction_timer) / tau)
            mu_eff += max(0.0, pulse_contrib)

        mu_eff -= self.cfg.degeneracy_beta * self.degeneracy_pressure
        mu_eff = max(mu_eff, 0.0)

        curvature_abs = abs(curvature_bias)
        if (
            self.degeneracy_pressure < 0.15
            and curvature_abs < 0.06
            and self.salience_edge >= self.cfg.salience_floor
        ):
            boost = 0.10 + 0.5 * (0.15 - self.degeneracy_pressure)
            boost = min(boost, 0.25)
            mu_eff *= 1.0 + boost
            self.post_pulse_window_timer = max(
                self.post_pulse_window_timer,
                min(self.cfg.post_pulse_window + 0.08, 0.35),
            )
            conditional_boost = True
        else:
            conditional_boost = False

        if pulse_active:
            self.micro_burst_pending = False
        if self.micro_burst_cooldown > 0.0:
            self.micro_burst_cooldown = max(self.micro_burst_cooldown - self.dt, 0.0)

        if self.cfg.micro_burst_enabled:
            if self.micro_burst_pending:
                self.micro_burst_timer += self.dt
                if (
                    self.micro_burst_timer >= self.cfg.micro_burst_delay_ms / 1000.0
                    and self.micro_burst_cooldown == 0.0
                    and self.salience_edge >= self.cfg.micro_burst_salience_floor
                    and coupling <= self.cfg.micro_burst_coupling_max
                ):
                    burst_window = self.cfg.traction_pulse_ms / 1000.0
                    self.traction_timer = max(self.traction_timer, burst_window)
                    self.post_pulse_timer = max(self.post_pulse_timer, burst_window)
                    self.post_pulse_window_timer = max(self.post_pulse_window_timer, self.cfg.post_pulse_window)
                    self.micro_burst_pending = False
                    self.micro_burst_timer = 0.0
                    self.micro_burst_cooldown = self.cfg.micro_burst_cooldown_ms / 1000.0
                    self.micro_burst_active_timer = burst_window
            else:
                if (
                    not pulse_active
                    and self.micro_burst_cooldown == 0.0
                    and self.salience_core >= 0.7 * self.cfg.gate_target_final
                    and self.salience_core < 0.8 * self.cfg.gate_target_final
                    and self.core_integral >= 0.0
                    and coupling <= self.cfg.micro_burst_coupling_max
                ):
                    self.micro_burst_pending = True
                    self.micro_burst_timer = 0.0

        if self.post_pulse_timer > 0.0:
            self.post_pulse_timer = max(self.post_pulse_timer - self.dt, 0.0)
        if self.post_pulse_window_timer > 0.0:
            self.post_pulse_window_timer = max(self.post_pulse_window_timer - self.dt, 0.0)

        total_traction = traction + max(0.0, pulse_contrib)
        self.traction_sum += total_traction
        self.traction_peak = max(self.traction_peak, total_traction)
        self.mu_eff_sum += mu_eff
        self.mu_eff_energy += mu_eff * self.dt
        self.mu_eff_peak = max(self.mu_eff_peak, mu_eff)
        self.mu_eff_samples += 1
        if pulse_active:
            self.pulse_active_steps += 1
        if lane_open:
            self.gate_open_steps += 1

        edge_gain = 1.0
        if mu_eff > 0.0:
            salience_excess = max(self.salience_edge - self.salience_mean, 0.0)
            desired = 1.0 + mu_eff * salience_excess
            desired = float(np.clip(desired, 1.0, 1.0 + 0.6 * mu_eff))
            multiplier, accelerated, blocked = gate_subsidy(
                self.salience_edge, self.cfg.salience_floor, desired, self.cfg.recovery_tax
            )
            edge_gain = multiplier
            telemetry.record(accelerated, blocked)
        else:
            multiplier, _, blocked = gate_subsidy(
                self.salience_edge, self.cfg.salience_floor, 1.0, self.cfg.recovery_tax
            )
            edge_gain = multiplier
            telemetry.record(False, blocked or not self.edge_gate_on)

        telemetry.record_flag("conditional_boost", conditional_boost)
        telemetry.record_flag("pulse_active", pulse_active)

        edge_delta = edge_gain * raw_edge_delta
        self.edge_state = 0.82 * self.edge_state + edge_delta
        self._update_salience("edge", edge_delta, error)

        # Low-pass merge
        merge_allowed = self.salience_core >= self.cfg.salience_floor
        post_pulse_boost = 1.0
        if self.post_pulse_window_timer > 0.0:
            post_pulse_boost += self.cfg.post_pulse_merge_bonus
        if merge_allowed:
            self.blue_noise_phase = (self.blue_noise_phase + 7.0 * self.dt) % 1.0
            noise = math.sin(2.0 * math.pi * self.blue_noise_phase) + 0.5 * math.sin(6.0 * math.pi * self.blue_noise_phase)
            noise *= 0.5
            err_state = abs(self.edge_state - self.core_state)
            sigmoid = 1.0 / (1.0 + math.exp(4.0 * err_state))
            merge_dynamic = self.cfg.merge_rate + 0.01 * noise + 0.04 * sigmoid
            merge_dynamic = float(np.clip(merge_dynamic * post_pulse_boost, 0.03, 0.12))
            merge_step = merge_dynamic * (self.salience_core - self.cfg.salience_floor + 0.15)
            merge_step = float(np.clip(merge_step, 0.0, merge_dynamic * 1.5))
            self.core_state += merge_step * (self.edge_state - self.core_state)
        else:
            # hold merge to protect identity
            self.core_state = self.core_state

        if merge_allowed and self.salience_core >= self.cfg.salience_floor + 0.12:
            blend = self.cfg.merge_target_blend * max(self.salience_core - (self.cfg.salience_floor + 0.12), 0.0)
            blend = float(np.clip(blend, 0.0, 0.5))
            self.core_state = (1.0 - blend) * self.core_state + blend * target_estimate

        if self.salience_core >= self.cfg.salience_floor + 0.05:
            follow_delta = self.cfg.core_follow_gain * error * self.dt
            self.core_state += follow_delta

        if self.salience_core >= self.cfg.salience_floor + 0.1:
            target_delta = self.cfg.core_target_gain * (target_estimate - self.core_state) * self.dt
            blend = self.cfg.core_target_blend * max(self.salience_core - (self.cfg.salience_floor + 0.1), 0.0)
            self.core_state += target_delta + blend * (target_estimate - self.core_state)

        high_salience_threshold = max(0.95, self.cfg.salience_floor + 0.25)
        if (
            self.cfg.core_high_salience_gain > 0.0
            and self.salience_core >= high_salience_threshold
        ):
            high_salience_delta = (
                self.cfg.core_high_salience_gain * (target_estimate - self.core_state) * self.dt
            )
            self.core_state += high_salience_delta

        plant_error = target_estimate - measurement
        leak_scale = 1.0 + self.cfg.degeneracy_eta * self.degeneracy_pressure
        leak_scale = max(leak_scale, 1.0)
        if self.salience_core >= self.cfg.salience_floor + 0.05:
            self.core_integral = (1.0 - self.cfg.core_integral_leak * leak_scale) * self.core_integral + error * self.dt
        else:
            self.core_integral *= (1.0 - 1.5 * self.cfg.core_integral_leak * leak_scale)
        if error * self.core_integral < 0.0:
            self.core_integral = 0.0
        limit = float(max(self.cfg.core_integral_limit, 0.1))
        self.core_integral = float(np.clip(self.core_integral, -limit, limit))

        if self.salience_core >= self.cfg.salience_floor + 0.08:
            self.control_integral = (1.0 - self.cfg.control_integral_leak) * self.control_integral + error * self.dt
        else:
            self.control_integral *= (1.0 - 2.0 * self.cfg.control_integral_leak)
        if error * self.control_integral < 0.0:
            self.control_integral = 0.0
        self.control_integral = float(np.clip(self.control_integral, -1.2, 1.2))

        core_output = np.clip(self.cfg.core_gain * self.core_state, -10.0, 10.0)
        edge_output = np.clip(self.cfg.edge_gain * self.edge_state, -10.0, 10.0)
        high_salience_ctrl_term = 0.0
        if self.salience_core >= high_salience_threshold:
            if self.cfg.core_high_salience_ctrl_kp > 0.0 or self.cfg.core_high_salience_ctrl_ki > 0.0:
                self.high_salience_integral = (
                    (1.0 - self.cfg.core_high_salience_ctrl_leak) * self.high_salience_integral
                    + plant_error * self.dt
                )
                if plant_error * self.high_salience_integral < 0.0:
                    self.high_salience_integral = 0.0
                self.high_salience_integral = float(np.clip(self.high_salience_integral, -1.5, 1.5))
                high_salience_ctrl_term = (
                    self.cfg.core_high_salience_ctrl_kp * plant_error
                    + self.cfg.core_high_salience_ctrl_ki * self.high_salience_integral
                )
        else:
            self.high_salience_integral *= (1.0 - 1.5 * self.cfg.core_high_salience_ctrl_leak)

        proportional = self.cfg.kp * error_clipped
        follow_gain = self.cfg.edge_feedforward * max(self.salience_core - self.cfg.salience_floor, 0.0)
        feedforward = follow_gain * (edge_output - core_output)
        anticipatory = self.cfg.core_feedforward_gain * (self.edge_state - self.core_state)
        control_target_term = 0.0
        if self.salience_core >= self.cfg.salience_floor + 0.1:
            control_target_term = self.cfg.control_target_gain * (target_estimate - measurement)
        integral_term = self.cfg.core_integral_gain * self.core_integral
        control_integral_term = self.cfg.control_integral_gain * self.control_integral

        if pulse_active:
            ff = self.cfg.pulse_feedforward_gain * mu_eff
            ff = float(np.clip(ff, -self.cfg.pulse_feedforward_clip, self.cfg.pulse_feedforward_clip))
            feedforward += ff

        post_pulse_active = self.traction_timer > 0.0 and self.traction_timer <= self.cfg.post_pulse_window
        bonus_multiplier = 1.0
        if post_pulse_active:
            bonus_multiplier = 1.0 + self.cfg.post_pulse_gain_bonus

        total_control = (
            core_output
            + edge_output
            + proportional
            + feedforward * bonus_multiplier
            + control_target_term
            + integral_term
            + control_integral_term
            + high_salience_ctrl_term
            + anticipatory
        )
        self.prev_error = error
        self.total_steps += 1

        return {
            "core": core_output,
            "edge": edge_output,
            "control": total_control,
            "merge_allowed": merge_allowed,
            "gate_target": gate_target,
            "credit": credit_val,
            "credit_value": credit_val,
            "lane_open": lane_open or should_open,
            "salience_core": self.salience_core,
            "salience_edge": self.salience_edge,
            "edge_salience": self.salience_edge,
            "core_salience": self.salience_core,
            "traction": total_traction,
            "traction_pulse_active": pulse_active,
            "traction_timer": self.traction_timer,
            "traction_total": total_traction,
            "seed_active": seed_active_flag,
            "coupling": coupling,
            "mu_eff": mu_eff,
            "degeneracy_pressure": self.degeneracy_pressure,
            "curvature_bias": curvature_bias,
            "micro_burst_pending": self.micro_burst_pending,
            "micro_burst_cooldown": self.micro_burst_cooldown,
            "micro_burst_active": self.micro_burst_active_timer > 0.0,
            "slope_ema": self.slope_ema,
            "cooldown_current": self.cooldown_current,
            "ladder_advanced": advanced_this_step,
            "stalled_votes": stalled_votes,
            "conditional_boost": conditional_boost,
        }


def calibrate_profile(profile: IdentityProfile, plant_cfg: PlantConfig, duration: float = 3.0) -> CalibrationStats:
    key = f"{profile.name}_dt{plant_cfg.dt}_tau{plant_cfg.tau}"
    if key in CALIBRATION_CACHE:
        return CALIBRATION_CACHE[key]

    base_stats = CalibrationStats(mean_salience=0.72, std_salience=0.02, baseline_z=0.0)
    calibrator_cfg = ControllerConfig(
        mu_edge=0.0,
        profile_name=profile.name,
        plant_step_scale=1.0,
        gate_target_init=0.65,
        gate_target_final=0.70,
    )
    calibrator = ReflexEdgeController(
        calibrator_cfg,
        plant_cfg,
        profile=profile,
        calibration=base_stats,
        calibrating=True,
    )
    telemetry = GateTelemetry(floor=calibrator_cfg.salience_floor)

    steps = max(1, int(duration / plant_cfg.dt))
    salience_samples = np.zeros(steps)
    state = 0.0
    target = plant_cfg.target
    for i in range(steps):
        error = target - state
        result = calibrator.step(error, state, telemetry)
        scaled_control = calibrator.cfg.plant_step_scale * result["control"]
        state += plant_cfg.dt * ((-state + scaled_control) / plant_cfg.tau)
        salience_samples[i] = result["edge_salience"]

    mean_salience = float(np.mean(salience_samples))
    std_salience = float(np.std(salience_samples) + 1e-6)
    stats = CalibrationStats(mean_salience=mean_salience, std_salience=std_salience, baseline_z=0.0)
    CALIBRATION_CACHE[key] = stats
    return stats


def simulate(cfg: ControllerConfig, plant_cfg: PlantConfig) -> Dict[str, np.ndarray | float]:
    profile = load_identity_profile(cfg.profile_name)
    calibration = calibrate_profile(profile, plant_cfg)
    controller = ReflexEdgeController(cfg, plant_cfg, profile=profile, calibration=calibration)
    controller.reset()
    telemetry = GateTelemetry(floor=cfg.salience_floor)

    steps = int(plant_cfg.horizon / plant_cfg.dt)
    target = np.ones(steps, dtype=float) * plant_cfg.target
    state = 0.0

    outputs = np.zeros(steps)
    core_trace = np.zeros(steps)
    edge_trace = np.zeros(steps)
    salience_core = np.zeros(steps)
    salience_edge = np.zeros(steps)
    control = np.zeros(steps)
    merge_mask = np.zeros(steps, dtype=bool)
    gate_target_arr = np.zeros(steps)
    credit_arr = np.zeros(steps)
    lane_open_arr = np.zeros(steps, dtype=bool)
    traction_arr = np.zeros(steps)
    seed_arr = np.zeros(steps, dtype=bool)
    coupling_arr = np.zeros(steps)
    pulse_arr = np.zeros(steps, dtype=bool)
    mu_eff_arr = np.zeros(steps)
    payoff_arr = np.zeros(steps)
    continuity_arr = np.zeros(steps)
    fatigue_arr = np.zeros(steps)
    degeneracy_arr = np.zeros(steps)
    curvature_arr = np.zeros(steps)
    micro_pending_arr = np.zeros(steps, dtype=bool)
    micro_active_arr = np.zeros(steps, dtype=bool)
    micro_cooldown_arr = np.zeros(steps)
    slope_ema_arr = np.zeros(steps)
    cooldown_arr = np.zeros(steps)
    conditional_boost_arr = np.zeros(steps, dtype=bool)
    ladder_advanced_arr = np.zeros(steps, dtype=bool)

    for i in range(steps):
        error = target[i] - state
        mux_enabled = cfg.mu_edge > 0.0 and controller.cfg.mu_edge > 0.0
        result = controller.step(error, state, telemetry, mux_enabled=mux_enabled)
        scaled_control = controller.cfg.plant_step_scale * result["control"]
        state += plant_cfg.dt * ((-state + scaled_control) / plant_cfg.tau)
        outputs[i] = state
        core_trace[i] = controller.core_state
        edge_trace[i] = controller.edge_state
        salience_core[i] = result["core_salience"]
        salience_edge[i] = result["edge_salience"]
        control[i] = result["control"]
        merge_mask[i] = result["merge_allowed"]
        gate_target_arr[i] = result["gate_target"]
        credit_arr[i] = result["credit_value"]
        lane_open_arr[i] = result["lane_open"]
        traction_arr[i] = result["traction"]
        seed_arr[i] = result["seed_active"]
        coupling_arr[i] = result["coupling"]
        pulse_arr[i] = result["traction_pulse_active"]
        mu_eff_arr[i] = result["mu_eff"]
        payoff_arr[i] = controller.payoff_core
        continuity_arr[i] = controller.salience_mean
        fatigue_arr[i] = controller.fatigue_core
        degeneracy_arr[i] = result.get("degeneracy_pressure", 0.0)
        curvature_arr[i] = result.get("curvature_bias", 0.0)
        micro_pending_arr[i] = result.get("micro_burst_pending", False)
        micro_cooldown_arr[i] = result.get("micro_burst_cooldown", 0.0)
        micro_active_arr[i] = result.get("micro_burst_active", False)
        slope_ema_arr[i] = float(result.get("slope_ema", 0.0))
        cooldown_arr[i] = float(result.get("cooldown_current", controller.cooldown_current))
        conditional_boost_arr[i] = bool(result.get("conditional_boost", False))
        ladder_advanced_arr[i] = bool(result.get("ladder_advanced", False))

    response_metrics = compute_response_times(
        core_trace,
        edge_trace,
        plant_cfg.dt,
        plant_cfg.target,
        controller.cfg.gate_target_final,
        controller.cfg.salience_floor,
        controller.mu_eff_energy,
        salience_core,
        payoff_arr,
        continuity_arr,
        fatigue_arr,
        traction_arr,
        credit_arr,
        gate_target_arr,
        pulse_arr,
        seed_arr,
    )
    response_metrics["core_area_error"] = float(np.sum(np.clip(plant_cfg.target - core_trace, 0.0, None)) * plant_cfg.dt)
    cross_coupling = float(np.max(np.abs(core_trace - edge_trace)))
    salience_edge_mean = float(np.mean(salience_edge))
    salience_edge_max = float(np.max(salience_edge))
    salience_floor = controller.cfg.salience_floor
    salience_edge_coverage = float(np.mean(salience_edge >= salience_floor))
    gate_open_pct = float(np.mean(lane_open_arr))
    gate_target_mean = float(np.mean(gate_target_arr))
    credit_peak = float(np.max(credit_arr))
    credit_final = float(credit_arr[-1])
    traction_mean = float(np.mean(traction_arr))
    traction_peak = float(np.max(traction_arr))
    mu_eff_mean = float(np.mean(mu_eff_arr))
    mu_eff_peak = float(np.max(mu_eff_arr))
    pulse_activation_pct = float(np.mean(pulse_arr))
    seed_coverage = float(np.mean(seed_arr))
    on_timer_final = float(controller.gate_timers.on_ms if lane_open_arr[-1] else 0.0)
    off_timer_final = float(controller.gate_timers.off_ms if not lane_open_arr[-1] else 0.0)
    coupling_max = float(np.max(coupling_arr))
    post_pulse_timer_final = float(controller.post_pulse_timer)
    post_pulse_window_timer_final = float(controller.post_pulse_window_timer)
    escape_time = float(response_metrics.get("escape_time", float("nan")))
    escape_flux = float(response_metrics.get("escape_flux", 0.0))
    escape_fraction = float(response_metrics.get("escape_fraction", float("nan")))
    total_flux = float(response_metrics.get("total_flux", float("nan")))
    degeneracy_mean = float(np.mean(degeneracy_arr))
    degeneracy_peak = float(np.max(degeneracy_arr))
    curvature_mean = float(np.mean(curvature_arr))
    curvature_std = float(np.std(curvature_arr))
    micro_pending_pct = float(np.mean(micro_pending_arr))
    micro_active_pct = float(np.mean(micro_active_arr))
    micro_cooldown_mean = float(np.mean(micro_cooldown_arr))
    slope_ema_mean = float(np.mean(slope_ema_arr))
    cooldown_mean = float(np.mean(cooldown_arr))
    conditional_boost_pct = telemetry.flag_ratio("conditional_boost")
    pulse_active_pct = telemetry.flag_ratio("pulse_active")
    ladder_progress_pct = float(np.mean(ladder_advanced_arr))

    identity_violation = bool(np.any((salience_core < cfg.salience_floor) & merge_mask))

    return {
        "outputs": outputs,
        "core": core_trace,
        "edge": edge_trace,
        "salience_core": salience_core,
        "salience_edge": salience_edge,
        "control": control,
        "response_metrics": response_metrics,
        "cross_coupling_error": cross_coupling,
        "identity_violation": identity_violation,
        "acceleration_pct": telemetry.acceleration_pct,
        "blocked_pct": telemetry.blocked_pct,
        "merge_mask": merge_mask.tolist(),
        "profile": profile,
        "calibration": calibration,
        "telemetry": {
            "gate_target": gate_target_arr.tolist(),
            "credit": credit_arr.tolist(),
            "lane_open": lane_open_arr.tolist(),
            "traction": traction_arr.tolist(),
            "seed_active": seed_arr.tolist(),
            "coupling": coupling_arr.tolist(),
            "salience_edge_mean": salience_edge_mean,
            "salience_edge_max": salience_edge_max,
            "salience_edge_coverage": salience_edge_coverage,
            "gate_open_pct": gate_open_pct,
            "gate_target_mean": gate_target_mean,
            "credit_peak": credit_peak,
            "credit_final": credit_final,
            "traction_mean": traction_mean,
            "traction_peak": traction_peak,
            "mu_eff_mean": mu_eff_mean,
            "mu_eff_peak": mu_eff_peak,
            "pulse_activation_pct": pulse_activation_pct,
            "mu_energy": controller.mu_eff_energy,
            "seed_coverage": seed_coverage,
            "on_timer_final": on_timer_final,
            "off_timer_final": off_timer_final,
            "coupling_max": coupling_max,
            "post_pulse_timer_final": post_pulse_timer_final,
            "post_pulse_window_final": post_pulse_window_timer_final,
            "escape_time": escape_time,
            "escape_flux": escape_flux,
            "escape_fraction": escape_fraction,
            "total_flux": total_flux,
            "escape_flux_window": response_metrics.get("escape_flux_window", float("nan")),
            "degeneracy_pressure_mean": degeneracy_mean,
            "degeneracy_pressure_peak": degeneracy_peak,
            "curvature_bias_mean": curvature_mean,
            "curvature_bias_std": curvature_std,
            "micro_burst_pending_pct": micro_pending_pct,
            "micro_burst_active_pct": micro_active_pct,
            "micro_burst_cooldown_mean": micro_cooldown_mean,
            "slope_ema": slope_ema_arr.tolist(),
            "cooldown_series": cooldown_arr.tolist(),
            "conditional_boost_flags": conditional_boost_arr.tolist(),
            "ladder_advanced_flags": ladder_advanced_arr.tolist(),
            "slope_ema_mean": slope_ema_mean,
            "cooldown_mean": cooldown_mean,
            "conditional_boost_pct": conditional_boost_pct,
            "pulse_active_pct": pulse_active_pct,
            "ladder_progress_pct": ladder_progress_pct,
        },
        "degeneracy_pressure": degeneracy_arr.tolist(),
        "curvature_bias": curvature_arr.tolist(),
        "micro_burst_pending": micro_pending_arr.tolist(),
        "micro_burst_active": micro_active_arr.tolist(),
        "micro_burst_cooldown": micro_cooldown_arr.tolist(),
    }


def _first_hysteretic_crossing(
    signal: np.ndarray,
    level: float,
    dt: float,
    pre_ms: float = 60.0,
    hold_ms: float = 50.0,
    eps: float = 0.01,
) -> int | None:
    if signal.size == 0:
        return None
    pre_steps = max(1, int(round((pre_ms / 1000.0) / dt)))
    hold_steps = max(1, int(round((hold_ms / 1000.0) / dt)))
    total = signal.size
    for idx in range(pre_steps, max(total - hold_steps + 1, pre_steps + 1)):
        pre_slice = signal[idx - pre_steps : idx]
        hold_slice = signal[idx : idx + hold_steps]
        if pre_slice.size < pre_steps or hold_slice.size < hold_steps:
            continue
        if np.all(pre_slice <= level - eps) and np.all(hold_slice >= level + eps):
            return idx
    return None


def compute_response_times(
    core: np.ndarray,
    edge: np.ndarray,
    dt: float,
    target: float,
    merge_target: float,
    salience_floor: float,
    mu_energy: float,
    salience_core: np.ndarray,
    payoff_arr: np.ndarray,
    continuity_arr: np.ndarray,
    fatigue_arr: np.ndarray,
    traction_arr: np.ndarray,
    credit_arr: np.ndarray,
    gate_target_arr: np.ndarray,
    pulse_arr: np.ndarray,
    seed_arr: np.ndarray,
) -> Dict[str, float]:
    def first_crossing(signal: np.ndarray, thresh: float) -> float:
        crossings = np.where(signal >= thresh)[0]
        if crossings.size == 0:
            return float("nan")
        return crossings[0] * dt

    merge_target = max(merge_target, 1e-6)
    mu_energy = float(mu_energy)
    tiers_rel = (0.6, 0.7, 0.8, 0.9)
    tiers_abs = tuple(merge_target * rel for rel in tiers_rel)

    edge_rt90 = first_crossing(edge, 0.9 * target)
    core_rt50 = first_crossing(core, 0.5 * target)
    core_rt70 = first_crossing(core, tiers_abs[1])
    core_rt90 = first_crossing(core, 0.9 * target)

    ladder_rts: Dict[str, float] = {}
    for rel, thresh in zip(tiers_rel, tiers_abs):
        ladder_rts[f"core_rt_{int(rel*100)}"] = first_crossing(core, thresh)

    ladder_highest = float("nan")
    ladder_rt = float("nan")
    for rel in reversed(tiers_rel):
        rt = ladder_rts[f"core_rt_{int(rel*100)}"]
        if not math.isnan(rt):
            ladder_highest = rel
            ladder_rt = rt
            break

    log_slope_50 = float("nan")
    if not math.isnan(core_rt50):
        idx = int(round(core_rt50 / dt))
        idx = min(max(idx, 1), len(core) - 2)
        slope = (core[idx + 1] - core[idx - 1]) / (2 * dt)
        log_slope_50 = float(math.log(max(slope, 1e-6)))

    slope_50_70 = float("nan")
    if not math.isnan(core_rt50) and not math.isnan(core_rt70) and core_rt70 > core_rt50:
        slope_50_70 = (tiers_abs[1] - 0.5 * target) / (core_rt70 - core_rt50)

    core_grad = np.gradient(core, dt)
    low_band = 0.4 * merge_target
    high_band = 0.6 * merge_target
    mask_mid = (core >= low_band) & (core <= high_band)
    slope_midband = float("nan")
    if np.any(mask_mid):
        slope_midband = float(np.max(core_grad[mask_mid]))

    def first_dwell_time(signal: np.ndarray, center: float, band: float, dwell_ms: float) -> float:
        if signal.size == 0:
            return float("nan")
        lo = center * (1.0 - band)
        hi = center * (1.0 + band)
        dwell_steps = max(1, int(round((dwell_ms / 1000.0) / dt)))
        if dwell_steps > signal.size:
            return float("nan")
        # Apply short moving average to suppress noise
        win = max(1, dwell_steps // 6)
        if win > 1:
            kernel = np.ones(win) / win
            smoothed = np.convolve(signal, kernel, mode="same")
        else:
            smoothed = signal
        for idx in range(0, signal.size - dwell_steps + 1):
            segment = smoothed[idx : idx + dwell_steps]
            if np.all((segment >= lo) & (segment <= hi)):
                return idx * dt
        return float("nan")

    settle_time_5 = first_dwell_time(core, target, band=0.05, dwell_ms=150.0)

    salience_horizon = max(0.9 * merge_target, salience_floor + 0.02)
    crossing_idx = _first_hysteretic_crossing(
        salience_core,
        salience_horizon,
        dt,
        pre_ms=60.0,
        hold_ms=50.0,
        eps=0.01,
    )

    fatigue_clamped = np.clip(fatigue_arr, 0.0, 0.95)
    flux_per_step = payoff_arr * continuity_arr * (1.0 - fatigue_clamped)
    total_flux = float(np.sum(flux_per_step) * dt)
    escape_flux = float("nan")
    escape_fraction = float("nan")
    escape_flux_window = float("nan")
    escape_time = float("nan")
    if crossing_idx is not None:
        escape_time = crossing_idx * dt
        win_seconds = 0.200
        win_steps = max(1, int(round(win_seconds / dt)))
        end_idx = min(crossing_idx + win_steps, flux_per_step.size)
        escape_flux_window = float(np.trapz(flux_per_step[crossing_idx:end_idx], dx=dt))
        escape_mask = salience_core >= salience_horizon
        escape_flux = float(np.sum(flux_per_step[escape_mask]) * dt)
        if total_flux > 0.0:
            escape_fraction = escape_flux / total_flux

    return {
        "edge_response_time_90": edge_rt90,
        "core_response_time_90": core_rt90,
        "core_response_time_70": core_rt70,
        "core_response_time_50": core_rt50,
        "core_log_slope_50": log_slope_50,
        "core_slope_50_70": slope_50_70,
        "core_ladder_best": ladder_highest,
        "core_ladder_rt": ladder_rt,
        "core_rt_60": ladder_rts["core_rt_60"],
        "core_rt_80": ladder_rts["core_rt_80"],
        "mu_energy": mu_energy,
        "slope_midband": slope_midband,
        "settle_time_5": settle_time_5,
        "escape_time": escape_time,
        "escape_flux": escape_flux,
        "escape_fraction": escape_fraction,
        "total_flux": total_flux,
        "escape_flux_window": escape_flux_window,
    }


def _baseline_key(cfg: ControllerConfig) -> str:
    return f"{cfg.profile_name}_mu0_lambda{cfg.lambda_core:.2f}_merge{cfg.merge_rate:.2f}"


def _baseline_metrics(cfg: ControllerConfig, plant_cfg: PlantConfig) -> Dict[str, float]:
    key = _baseline_key(cfg)
    if key in BASELINE_CACHE:
        return BASELINE_CACHE[key]
    baseline_cfg = replace(cfg, mu_edge=0.0)
    traces = simulate(baseline_cfg, plant_cfg)
    metrics = traces["response_metrics"]
    BASELINE_CACHE[key] = {
        "core_response_time_70": metrics.get("core_response_time_70", float("nan")),
        "core_response_time_50": metrics.get("core_response_time_50", float("nan")),
        "core_log_slope_50": metrics.get("core_log_slope_50", float("nan")),
        "core_area_error": traces["response_metrics"].get("core_area_error", float("nan")),
        "mu_energy": metrics.get("mu_energy", float("nan")),
        "slope_midband": metrics.get("slope_midband", float("nan")),
        "core_rt_80": metrics.get("core_rt_80", float("nan")),
        "settle_time_5": metrics.get("settle_time_5", float("nan")),
        "escape_time": metrics.get("escape_time", float("nan")),
        "escape_flux": metrics.get("escape_flux", float("nan")),
        "escape_flux_window": metrics.get("escape_flux_window", float("nan")),
        "escape_fraction": metrics.get("escape_fraction", float("nan")),
        "total_flux": metrics.get("total_flux", float("nan")),
    }
    return BASELINE_CACHE[key]


def build_result(
    cfg: ControllerConfig,
    traces: Dict[str, np.ndarray | float],
    plant_cfg: PlantConfig,
    profile: IdentityProfile,
) -> SimulationResult:
    res_metrics = traces["response_metrics"]
    baseline = _baseline_metrics(cfg, plant_cfg)
    accel_score = float("nan")
    accel_score_alt = float("nan")
    area_ratio = float("nan")
    accel_70_energy = float("nan")
    slope_gain_midband = float("nan")
    settle_time_5 = res_metrics.get("settle_time_5", float("nan"))
    total_flux = res_metrics.get("total_flux", float("nan"))
    escape_time = res_metrics.get("escape_time", float("nan"))
    escape_flux = res_metrics.get("escape_flux", float("nan"))
    escape_fraction = res_metrics.get("escape_fraction", float("nan"))
    escape_time_gain = float("nan")
    escape_flux_gain = float("nan")
    escape_flux_window = res_metrics.get("escape_flux_window", float("nan"))

    t70 = res_metrics.get("core_response_time_70", float("nan"))
    t50 = res_metrics.get("core_response_time_50", float("nan"))
    base_t70 = baseline.get("core_response_time_70", float("nan"))
    base_mu_energy = baseline.get("mu_energy", float("nan"))
    base_slope_mid = baseline.get("slope_midband", float("nan"))
    base_escape_time = baseline.get("escape_time", float("nan"))
    base_escape_flux = baseline.get("escape_flux", float("nan"))
    base_escape_flux_window = baseline.get("escape_flux_window", float("nan"))
    base_escape_fraction = baseline.get("escape_fraction", float("nan"))
    base_total_flux = baseline.get("total_flux", float("nan"))
    cfg_mu_energy = res_metrics.get("mu_energy", float("nan"))
    cfg_slope_mid = res_metrics.get("slope_midband", float("nan"))
    slope_log = res_metrics.get("core_log_slope_50", float("nan"))
    if math.isnan(t70) and not math.isnan(t50):
        t70 = t50

    if not math.isnan(t70) and not math.isnan(base_t70):
        accel_score = base_t70 / max(t70, 1e-6)
    if not math.isnan(res_metrics.get("core_rt_80", float("nan"))) and not math.isnan(baseline.get("core_rt_80", float("nan"))):
        accel_score_alt = baseline["core_rt_80"] / max(res_metrics["core_rt_80"], 1e-6)

    if not math.isnan(base_t70) and not math.isnan(t70) and not math.isnan(base_mu_energy) and not math.isnan(cfg_mu_energy):
        accel_70_energy = (base_t70 / max(t70, 1e-6)) * (base_mu_energy / (cfg_mu_energy + 1e-6))

    if not math.isnan(cfg_slope_mid) and not math.isnan(base_slope_mid) and base_slope_mid > 0.0:
        slope_gain_midband = cfg_slope_mid / base_slope_mid

    area = res_metrics.get("core_area_error", float("nan"))
    base_area = baseline.get("core_area_error", float("nan"))
    if not math.isnan(area) and not math.isnan(base_area) and area > 0.0:
        area_ratio = area / base_area if base_area > 0.0 else float("nan")

    if not math.isnan(escape_time) and not math.isnan(base_escape_time) and escape_time > 0.0:
        escape_time_gain = base_escape_time / escape_time
    if (
        not math.isnan(escape_flux_window)
        and not math.isnan(base_escape_flux_window)
        and base_escape_flux_window > 0.0
    ):
        escape_flux_gain = escape_flux_window / base_escape_flux_window
    elif not math.isnan(escape_flux) and not math.isnan(base_escape_flux) and base_escape_flux > 0.0:
        escape_flux_gain = escape_flux / base_escape_flux

    if math.isnan(accel_score):
        if not math.isnan(accel_score_alt):
            accel_score = accel_score_alt
        elif not math.isnan(area_ratio):
            accel_score = area_ratio

    telemetry_data = traces.get("telemetry", {})
    edge_salience_mean = float(telemetry_data.get("salience_edge_mean", np.mean(traces["salience_edge"])))
    edge_salience_max = float(telemetry_data.get("salience_edge_max", np.max(traces["salience_edge"])))
    edge_salience_coverage = float(telemetry_data.get("salience_edge_coverage", np.mean(traces["salience_edge"] >= cfg.salience_floor)))
    gate_open_pct = float(telemetry_data.get("gate_open_pct", 0.0))
    gate_target_mean = float(telemetry_data.get("gate_target_mean", cfg.gate_target_final))
    credit_peak = float(telemetry_data.get("credit_peak", 0.0))
    credit_final = float(telemetry_data.get("credit_final", 0.0))
    traction_mean = float(telemetry_data.get("traction_mean", 0.0))
    traction_peak = float(telemetry_data.get("traction_peak", 0.0))
    mu_eff_mean = float(telemetry_data.get("mu_eff_mean", 0.0))
    mu_eff_peak = float(telemetry_data.get("mu_eff_peak", 0.0))
    pulse_activation_pct = float(telemetry_data.get("pulse_activation_pct", 0.0))
    post_pulse_timer_final = float(telemetry_data.get("post_pulse_timer_final", 0.0))
    post_pulse_window_final = float(telemetry_data.get("post_pulse_window_final", 0.0))
    seed_coverage = float(telemetry_data.get("seed_coverage", 0.0))
    slope_ema_mean = float(telemetry_data.get("slope_ema_mean", 0.0))
    cooldown_mean = float(telemetry_data.get("cooldown_mean", 0.0))
    conditional_boost_pct = float(telemetry_data.get("conditional_boost_pct", 0.0))
    pulse_active_pct = float(telemetry_data.get("pulse_active_pct", 0.0))
    ladder_progress_pct = float(telemetry_data.get("ladder_progress_pct", 0.0))
    on_timer_final = float(telemetry_data.get("on_timer_final", 0.0))
    off_timer_final = float(telemetry_data.get("off_timer_final", 0.0))
    coupling_max_value = traces.get("cross_coupling_error", float("nan"))
    if isinstance(coupling_max_value, np.ndarray):
        coupling_max_value = float(np.max(coupling_max_value))
    coupling_max = float(telemetry_data.get("coupling_max", coupling_max_value))
    escape_time_metric = float(telemetry_data.get("escape_time", escape_time))
    escape_flux_metric = float(telemetry_data.get("escape_flux", escape_flux))
    escape_flux_window_metric = float(telemetry_data.get("escape_flux_window", escape_flux_window))
    escape_fraction_metric = float(telemetry_data.get("escape_fraction", escape_fraction))
    degeneracy_pressure_mean = float(telemetry_data.get("degeneracy_pressure_mean", np.mean(traces.get("degeneracy_pressure", [0.0]))))
    degeneracy_pressure_peak = float(telemetry_data.get("degeneracy_pressure_peak", np.max(traces.get("degeneracy_pressure", [0.0]))))
    curvature_bias_mean = float(telemetry_data.get("curvature_bias_mean", np.mean(traces.get("curvature_bias", [0.0]))))
    curvature_bias_std = float(telemetry_data.get("curvature_bias_std", np.std(traces.get("curvature_bias", [0.0]))))
    micro_burst_pending_pct = float(telemetry_data.get("micro_burst_pending_pct", np.mean(traces.get("micro_burst_pending", [0.0]))))
    micro_burst_active_pct = float(telemetry_data.get("micro_burst_active_pct", np.mean(traces.get("micro_burst_active", [0.0]))))
    micro_burst_cooldown_mean = float(telemetry_data.get("micro_burst_cooldown_mean", np.mean(traces.get("micro_burst_cooldown", [0.0]))))

    utility_speed = accel_score_alt if not math.isnan(accel_score_alt) else accel_score
    utility_quality = area_ratio if not math.isnan(area_ratio) else float("nan")
    utility_total = float("nan")
    if not math.isnan(utility_speed) and not math.isnan(utility_quality):
        weight = float(np.clip(profile.speed_affinity, 0.0, 1.0))
        utility_total = weight * utility_speed + (1.0 - weight) * utility_quality
    elif not math.isnan(utility_speed):
        utility_total = utility_speed
    elif not math.isnan(utility_quality):
        utility_total = utility_quality

    slope_50_70 = res_metrics.get("core_slope_50_70", float("nan"))
    core_rt70 = t70

    return SimulationResult(
        config=cfg,
        edge_response_time_90=res_metrics.get("edge_response_time_90", float("nan")),
        core_response_time_90=res_metrics.get("core_response_time_90", float("nan")),
        core_response_time_70=core_rt70,
        core_response_time_50=t50,
        core_log_slope_50=slope_log,
        core_slope_50_70=slope_50_70,
        core_ladder_best=res_metrics.get("core_ladder_best", float("nan")),
        core_ladder_rt=res_metrics.get("core_ladder_rt", float("nan")),
        core_area_error=res_metrics.get("core_area_error", float("nan")),
        accel_score=accel_score,
        accel_score_alt=accel_score_alt,
        accel_70_energy=accel_70_energy,
        slope_gain_midband=slope_gain_midband,
        settle_time_5=settle_time_5,
        total_flux=total_flux,
        escape_time=escape_time,
        escape_flux=escape_flux_metric,
        escape_flux_window=escape_flux_window_metric,
        escape_fraction=escape_fraction_metric,
        escape_time_gain=escape_time_gain,
        escape_flux_gain=escape_flux_gain,
        degeneracy_pressure_mean=degeneracy_pressure_mean,
        degeneracy_pressure_peak=degeneracy_pressure_peak,
        curvature_bias_mean=curvature_bias_mean,
        curvature_bias_std=curvature_bias_std,
        micro_burst_pending_pct=micro_burst_pending_pct,
        micro_burst_active_pct=micro_burst_active_pct,
        micro_burst_cooldown_mean=micro_burst_cooldown_mean,
        area_ratio=area_ratio,
        utility_speed=utility_speed,
        utility_quality=utility_quality,
        utility_total=utility_total,
        core_salience_min=float(np.min(traces["salience_core"])),
        edge_salience_min=float(np.min(traces["salience_edge"])),
        edge_salience_mean=edge_salience_mean,
        edge_salience_max=edge_salience_max,
        edge_salience_coverage=edge_salience_coverage,
        gate_open_pct=gate_open_pct,
        gate_target_final=cfg.gate_target_final,
        gate_target_mean=gate_target_mean,
        credit_peak=credit_peak,
        credit_final=credit_final,
        traction_mean=traction_mean,
        traction_peak=traction_peak,
        mu_eff_mean=mu_eff_mean,
        mu_eff_peak=mu_eff_peak,
        pulse_activation_pct=pulse_activation_pct,
        post_pulse_timer_final=post_pulse_timer_final,
        post_pulse_window_final=post_pulse_window_final,
        seed_coverage=seed_coverage,
        slope_ema_mean=slope_ema_mean,
        cooldown_mean=cooldown_mean,
        conditional_boost_pct=conditional_boost_pct,
        pulse_active_pct=pulse_active_pct,
        ladder_progress_pct=ladder_progress_pct,
        on_timer_final_ms=on_timer_final,
        off_timer_final_ms=off_timer_final,
        coupling_max_observed=coupling_max,
        cross_coupling_error=float(traces["cross_coupling_error"]),
        identity_violation=bool(traces["identity_violation"]),
        acceleration_pct=float(traces["acceleration_pct"]),
        blocked_pct=float(traces["blocked_pct"]),
    )


def _run_variant(args: Tuple[ControllerConfig, PlantConfig]) -> SimulationResult:
    cfg, plant_cfg = args
    traces = simulate(cfg, plant_cfg)
    profile = traces.get("profile") if isinstance(traces, dict) else None
    if profile is None:
        profile = load_identity_profile(cfg.profile_name)
    return build_result(cfg, traces, plant_cfg, profile)


def generate_parameter_grid() -> List[ControllerConfig]:
    merge_rate = 0.06
    mu_edge = 1.2
    base_cfg = ControllerConfig(
        merge_rate=merge_rate,
        mu_edge=mu_edge,
        core_feedforward_gain=0.12,
    )

    plant_scales = [1.0, 1.2]
    gate_ramps = [100.0, 200.0]
    seed_values = [0.76, 0.78]
    traction_alphas = [0.18, 0.22]
    credit_taus = [60.0, 90.0]

    configs: List[ControllerConfig] = []
    for plant_scale in plant_scales:
        for gate_ramp in gate_ramps:
            for seed in seed_values:
                for traction in traction_alphas:
                    for credit_tau in credit_taus:
                        cfg = replace(
                            base_cfg,
                            plant_step_scale=plant_scale,
                            gate_ramp_ms=gate_ramp,
                            edge_salience_seed=seed,
                            traction_alpha=traction,
                            credit_tau_ms=credit_tau,
                        )
                        configs.append(cfg)

    return configs


def save_trace(tag: str, cfg: ControllerConfig, traces: Dict[str, np.ndarray | float]) -> Path:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "tag": tag,
        "config": cfg.__dict__,
        "response_metrics": traces["response_metrics"],
        "core_salience_min": float(np.min(traces["salience_core"])),
        "edge_salience_min": float(np.min(traces["salience_edge"])),
        "acceleration_pct": float(traces["acceleration_pct"]),
        "blocked_pct": float(traces["blocked_pct"]),
        "trace": {
            "outputs": traces["outputs"].tolist(),
            "core": traces["core"].tolist(),
            "edge": traces["edge"].tolist(),
            "salience_core": traces["salience_core"].tolist(),
            "salience_edge": traces["salience_edge"].tolist(),
            "control": traces["control"].tolist(),
            "merge_allowed": traces.get("merge_mask"),
            "gate_target": traces["telemetry"]["gate_target"],
            "credit": traces["telemetry"]["credit"],
            "lane_open": traces["telemetry"]["lane_open"],
            "traction": traces["telemetry"]["traction"],
            "seed_active": traces["telemetry"]["seed_active"],
            "coupling": traces["telemetry"]["coupling"],
        },
        "telemetry_stats": {
            "salience_edge_mean": traces["telemetry"]["salience_edge_mean"],
            "salience_edge_max": traces["telemetry"]["salience_edge_max"],
            "salience_edge_coverage": traces["telemetry"]["salience_edge_coverage"],
            "gate_open_pct": traces["telemetry"]["gate_open_pct"],
            "gate_target_mean": traces["telemetry"]["gate_target_mean"],
            "credit_peak": traces["telemetry"]["credit_peak"],
            "credit_final": traces["telemetry"]["credit_final"],
            "traction_mean": traces["telemetry"]["traction_mean"],
            "traction_peak": traces["telemetry"]["traction_peak"],
            "mu_eff_mean": traces["telemetry"].get("mu_eff_mean", 0.0),
            "mu_eff_peak": traces["telemetry"].get("mu_eff_peak", 0.0),
            "pulse_activation_pct": traces["telemetry"].get("pulse_activation_pct", 0.0),
            "mu_energy": traces["telemetry"].get("mu_energy", 0.0),
            "seed_coverage": traces["telemetry"]["seed_coverage"],
            "on_timer_final": traces["telemetry"]["on_timer_final"],
            "off_timer_final": traces["telemetry"]["off_timer_final"],
            "coupling_max": traces["telemetry"]["coupling_max"],
        },
    }
    filename = TRACE_DIR / f"trace_{tag}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    with filename.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return filename


def run_trace_probes(plant_cfg: PlantConfig) -> List[SimulationResult]:
    results: List[SimulationResult] = []
    for tag, cfg in TRACE_PROBES:
        traces = simulate(cfg, plant_cfg)
        profile = traces.get("profile") if isinstance(traces, dict) else None
        if profile is None:
            profile = load_identity_profile(cfg.profile_name)
        save_trace(tag, cfg, traces)
        results.append(build_result(cfg, traces, plant_cfg, profile))
    return results


def run_experiment(parallel_workers: int | None = None) -> List[SimulationResult]:
    plant_cfg = PlantConfig()
    configs = generate_parameter_grid()

    if parallel_workers is None or parallel_workers <= 1:
        sweep_results = [_run_variant((cfg, plant_cfg)) for cfg in configs]
    else:
        tasks = [(cfg, plant_cfg) for cfg in configs]
        sweep_results = []
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            for result in executor.map(_run_variant, tasks):
                sweep_results.append(result)

    probe_results = run_trace_probes(plant_cfg)
    return probe_results + sweep_results


def write_artifact(results: List[SimulationResult]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())
    payload = []
    for result in results:
        payload.append(
            {
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "experiment_name": "experiment_n_reflex_edge",
                "run_id": run_id,
                "mu_edge": result.config.mu_edge,
                "merge_rate": result.config.merge_rate,
                "lambda_core": result.config.lambda_core,
                "core_follow_gain": result.config.core_follow_gain,
                "core_target_gain": result.config.core_target_gain,
                "control_target_gain": result.config.control_target_gain,
                "core_target_blend": result.config.core_target_blend,
                "core_integral_gain": result.config.core_integral_gain,
                "core_integral_leak": result.config.core_integral_leak,
                "control_integral_gain": result.config.control_integral_gain,
                "control_integral_leak": result.config.control_integral_leak,
                "merge_target_blend": result.config.merge_target_blend,
                "core_high_salience_gain": result.config.core_high_salience_gain,
                "core_high_salience_ctrl_kp": result.config.core_high_salience_ctrl_kp,
                "core_high_salience_ctrl_ki": result.config.core_high_salience_ctrl_ki,
                "core_high_salience_ctrl_leak": result.config.core_high_salience_ctrl_leak,
                "edge_response_time_90": result.edge_response_time_90,
                "core_response_time_90": result.core_response_time_90,
                "core_response_time_70": result.core_response_time_70,
                "core_response_time_50": result.core_response_time_50,
                "core_log_slope_50": result.core_log_slope_50,
                "core_slope_50_70": result.core_slope_50_70,
                "core_area_error": result.core_area_error,
                "accel_score": result.accel_score,
                "accel_score_alt": result.accel_score_alt,
                "area_ratio": result.area_ratio,
                "utility_speed": result.utility_speed,
                "utility_quality": result.utility_quality,
                "utility_total": result.utility_total,
                "core_salience_min": result.core_salience_min,
                "edge_salience_min": result.edge_salience_min,
                "edge_salience_mean": result.edge_salience_mean,
                "edge_salience_max": result.edge_salience_max,
                "edge_salience_coverage": result.edge_salience_coverage,
                "gate_open_pct": result.gate_open_pct,
                "gate_target_final": result.gate_target_final,
                "gate_target_mean": result.gate_target_mean,
                "credit_peak": result.credit_peak,
                "credit_final": result.credit_final,
                "traction_mean": result.traction_mean,
                "traction_peak": result.traction_peak,
                "mu_eff_mean": result.mu_eff_mean,
                "mu_eff_peak": result.mu_eff_peak,
                "pulse_activation_pct": result.pulse_activation_pct,
                "accel_70_energy": result.accel_70_energy,
                "slope_gain_midband": result.slope_gain_midband,
                "settle_time_5": result.settle_time_5,
                "total_flux": result.total_flux,
                "escape_time": result.escape_time,
                "escape_flux": result.escape_flux,
                "escape_flux_window": result.escape_flux_window,
                "escape_fraction": result.escape_fraction,
                "escape_time_gain": result.escape_time_gain,
                "escape_flux_gain": result.escape_flux_gain,
                "degeneracy_pressure_mean": result.degeneracy_pressure_mean,
                "degeneracy_pressure_peak": result.degeneracy_pressure_peak,
                "curvature_bias_mean": result.curvature_bias_mean,
                "curvature_bias_std": result.curvature_bias_std,
                "micro_burst_pending_pct": result.micro_burst_pending_pct,
                "micro_burst_active_pct": result.micro_burst_active_pct,
                "micro_burst_cooldown_mean": result.micro_burst_cooldown_mean,
                "seed_coverage": result.seed_coverage,
                "slope_ema_mean": result.slope_ema_mean,
                "cooldown_mean": result.cooldown_mean,
                "conditional_boost_pct": result.conditional_boost_pct,
                "pulse_active_pct": result.pulse_active_pct,
                "ladder_progress_pct": result.ladder_progress_pct,
                "on_timer_final_ms": result.on_timer_final_ms,
                "off_timer_final_ms": result.off_timer_final_ms,
                "coupling_max_observed": result.coupling_max_observed,
                "cross_coupling_error": result.cross_coupling_error,
                "identity_violation": result.identity_violation,
                "acceleration_pct": result.acceleration_pct,
                "blocked_pct": result.blocked_pct,
            }
        )
    path = ARTIFACT_DIR / f"reflex_edge_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def print_summary(results: List[SimulationResult]) -> None:
    from tabulate import tabulate

    rows = [
        (
            res.config.mu_edge,
            res.config.merge_rate,
            res.config.lambda_core,
            res.config.core_follow_gain,
            res.config.core_target_gain,
            res.config.control_target_gain,
            res.config.core_target_blend,
            res.config.core_integral_gain,
            res.config.core_integral_leak,
            res.config.control_integral_gain,
            res.config.control_integral_leak,
            res.config.merge_target_blend,
            res.config.core_high_salience_gain,
            res.config.core_high_salience_ctrl_kp,
            res.config.core_high_salience_ctrl_ki,
            res.config.core_high_salience_ctrl_leak,
            res.edge_response_time_90,
            res.core_response_time_90,
            res.core_salience_min,
            res.edge_salience_min,
            res.cross_coupling_error,
            res.identity_violation,
            res.acceleration_pct,
            res.blocked_pct,
        )
        for res in results
    ]
    print(
        tabulate(
            rows,
            headers=[
                "mu_edge",
                "merge_rate",
                "lambda_core",
                "follow_gain",
                "core_target",
                "ctrl_target",
                "blend",
                "int_gain",
                "int_leak",
                "ctrl_int_gain",
                "ctrl_int_leak",
                "merge_target_blend",
                "high_sal_gain",
                "hs_ctrl_kp",
                "hs_ctrl_ki",
                "hs_ctrl_leak",
                "edge_rt90",
                "core_rt90",
                "core_sal_min",
                "edge_sal_min",
                "cross_err",
                "identity_violation",
                "accel_pct",
                "blocked_pct",
            ],
            tablefmt="github",
            floatfmt=".6f",
        )
    )


def filter_success(results: Sequence[SimulationResult]) -> List[SimulationResult]:
    successes: List[SimulationResult] = []
    for res in results:
        if math.isnan(res.accel_score) and math.isnan(res.accel_score_alt) and math.isnan(res.area_ratio):
            continue
        metric = res.accel_score
        if math.isnan(metric) and not math.isnan(res.accel_score_alt):
            metric = res.accel_score_alt
        if math.isnan(metric) and not math.isnan(res.area_ratio):
            metric = res.area_ratio
        if math.isnan(metric) or metric < 1.05:
            continue
        if res.cross_coupling_error >= 1.1:
            continue
        if res.edge_salience_min < 0.75 or res.core_salience_min < 1.05:
            continue
        if res.identity_violation:
            continue
        successes.append(res)
    return successes


def main() -> None:
    results = run_experiment(parallel_workers=6)
    artifact = write_artifact(results)
    successes = filter_success(results)
    print("=== Experiment N: reflex edge controller ===")
    print_summary(results)
    print(f"Results written to {artifact}")
    if successes:
        print("--- Passing configurations ---")
        print_summary(successes)
    else:
        print("No configurations met the success criteria yet.")


if __name__ == "__main__":
    main()
