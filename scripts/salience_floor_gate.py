"""Utilities for Experiment K: Continuity Floor Gating.

Provides reusable gating helpers to ensure fast-mode accelerations only engage
when salience (S') remains above a configurable morale floor. When salience
falls below the floor, acceleration is disabled and a mild recovery tax is
applied instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

DEFAULT_FLOORS = (0.6, 0.7, 0.8)


@dataclass
class GateTelemetry:
    floor: float
    acceleration_steps: int = 0
    blocked_steps: int = 0
    total_steps: int = 0
    flag_hits: Dict[str, int] = field(default_factory=dict)
    flag_counts: Dict[str, int] = field(default_factory=dict)

    def record(self, accelerated: bool, blocked: bool) -> None:
        if accelerated:
            self.acceleration_steps += 1
        if blocked:
            self.blocked_steps += 1
        self.total_steps += 1

    def record_flag(self, name: str, active: bool) -> None:
        self.flag_counts[name] = self.flag_counts.get(name, 0) + 1
        if active:
            self.flag_hits[name] = self.flag_hits.get(name, 0) + 1

    def flag_ratio(self, name: str) -> float:
        total = self.flag_counts.get(name, 0)
        if total == 0:
            return 0.0
        return self.flag_hits.get(name, 0) / total

    @property
    def acceleration_pct(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.acceleration_steps / self.total_steps

    @property
    def blocked_pct(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.blocked_steps / self.total_steps


def gate_multiplier(
    salience: float,
    floor: float,
    accel_multiplier: float,
    recovery_tax: float = 0.2,
) -> Tuple[float, bool, bool]:
    """Return the multiplier after applying salience gating.

    Args:
        salience: Current scalar salience S'.
        floor: Minimum salience required to enable acceleration (>0 disables gate).
        accel_multiplier: Desired multiplier when acceleration is allowed (>=1).
        recovery_tax: Recovery penalty applied when salience is below the floor.

    Returns:
        (multiplier, accelerated_flag, blocked_flag)
    """

    if floor <= 0.0:
        accelerated = accel_multiplier > 1.0
        return accel_multiplier, accelerated, False

    if salience >= floor:
        accelerated = accel_multiplier > 1.0
        return accel_multiplier, accelerated, False

    safe_multiplier = 1.0 / (1.0 + max(recovery_tax, 0.0))
    return safe_multiplier, False, True


def gate_gradient_scale(
    salience: float,
    floor: float,
    boost_factor: float,
    recovery_tax: float = 0.2,
) -> Tuple[float, bool, bool]:
    return gate_multiplier(salience, floor, boost_factor, recovery_tax)


def gate_subsidy(
    salience: float,
    floor: float,
    subsidy_strength: float,
    recovery_tax: float = 0.2,
) -> Tuple[float, bool, bool]:
    return gate_multiplier(salience, floor, subsidy_strength, recovery_tax)
