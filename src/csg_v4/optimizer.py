"""Kappa scanning and refinement utilities for CSG V4."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from tqdm import tqdm

from .config import CSGConfig
from .galaxy import GalaxyData
from .model import CSGV4Model


def _get_verbosity() -> int:
    """Get verbosity level from environment: 0=quiet, 1=normal, 2=verbose."""
    return int(os.environ.get("CSG_VERBOSITY", "1"))


@dataclass(frozen=True)
class ScanResult:
    kappa_values: np.ndarray
    chisq_values: np.ndarray
    best_kappa: float
    best_index: int
    best_chisq: float
    per_galaxy_mse: np.ndarray | None = None
    residual_profiles: np.ndarray | None = None
    galaxy_names: tuple[str, ...] | None = None
    sample_counts: tuple[int, ...] | None = None
    radii_arrays: tuple[np.ndarray, ...] | None = None


def scan_kappa(
    galaxies: Iterable[GalaxyData],
    model: CSGV4Model | None = None,
    config: CSGConfig | None = None,
    kappa_grid: np.ndarray | None = None,
    store_profiles: bool = False,
) -> tuple[ScanResult, dict[str, np.ndarray]]:
    galaxies = list(galaxies)
    if model is None and config is None:
        config = CSGConfig()
    elif model is not None:
        config = model.config
    elif config is None:
        raise ValueError("Must provide either a model or a config.")

    model = model or CSGV4Model(config)
    grid = kappa_grid if kappa_grid is not None else config.kappa_grid

    chisq_values = np.zeros_like(grid)
    residual_map: dict[str, np.ndarray] = {}
    n_gal = len(galaxies)
    max_radii = max((galaxy.n_radii for galaxy in galaxies), default=0)
    per_galaxy_mse = np.zeros((n_gal, grid.size)) if store_profiles else None
    residual_profiles = (
        np.full((n_gal, grid.size, max_radii), np.nan)
        if store_profiles and max_radii > 0
        else None
    )
    galaxy_names = tuple(gal.name for gal in galaxies)
    sample_counts = tuple(gal.n_radii for gal in galaxies)
    radii_arrays = tuple(gal.radii_kpc for gal in galaxies)

    verbosity = _get_verbosity()
    disable_pbar = verbosity == 0

    # Outer loop: kappa values
    kappa_iter = tqdm(
        enumerate(grid),
        total=len(grid),
        desc="Scanning kappa values",
        disable=disable_pbar,
        leave=False,
    )

    for idx, kappa in kappa_iter:
        residuals: list[np.ndarray] = []
        for g_idx, galaxy in enumerate(galaxies):
            outputs = model.predict_velocity(galaxy, kappa)
            resid = model.residuals(galaxy, outputs)
            residuals.append(resid)
            if store_profiles and per_galaxy_mse is not None and residual_profiles is not None:
                per_galaxy_mse[g_idx, idx] = float(np.mean(np.square(resid)))
                residual_profiles[g_idx, idx, : galaxy.n_radii] = resid
        chisq_values[idx] = model.chisq(residuals)

        if verbosity >= 2 and idx % max(1, len(grid) // 20) == 0:
            # Update progress bar with current best
            current_best = float(grid[int(np.argmin(chisq_values[: idx + 1]))])
            kappa_iter.set_postfix({"current_best": f"{current_best:.5f}"})

    best_index = int(np.argmin(chisq_values))
    best_kappa = float(grid[best_index])
    best_chisq = float(chisq_values[best_index])

    for galaxy in galaxies:
        outputs = model.predict_velocity(galaxy, best_kappa)
        residual_map[galaxy.name] = model.residuals(galaxy, outputs)

    scan_result = ScanResult(
        kappa_values=grid,
        chisq_values=chisq_values,
        best_kappa=best_kappa,
        best_index=best_index,
        best_chisq=best_chisq,
        per_galaxy_mse=per_galaxy_mse,
        residual_profiles=residual_profiles,
        galaxy_names=galaxy_names,
        sample_counts=sample_counts,
        radii_arrays=radii_arrays,
    )

    return scan_result, residual_map


def _compute_chisq(
    galaxies: Iterable[GalaxyData],
    model: CSGV4Model,
    kappa: float,
) -> float:
    residuals: list[np.ndarray] = []
    for galaxy in galaxies:
        outputs = model.predict_velocity(galaxy, kappa)
        residuals.append(model.residuals(galaxy, outputs))
    return model.chisq(residuals)


@dataclass(frozen=True)
class QuadraticEstimate:
    kappa: float
    chisq: float
    curvature: float
    sigma: float
    support_points: np.ndarray


@dataclass(frozen=True)
class GoldenSectionEstimate:
    kappa: float
    chisq: float
    iterations: int
    bracket: tuple[float, float]


@dataclass(frozen=True)
class RefinementResult:
    quadratic: QuadraticEstimate | None
    golden: GoldenSectionEstimate | None
    zoom_scan: ScanResult | None
    zoom_residual_map: dict[str, np.ndarray]


def quadratic_refinement(scan: ScanResult, window: int | None = None) -> QuadraticEstimate | None:
    window = window or 5
    if scan.chisq_values.size < 3:
        return None
    half = max(window // 2, 1)
    start = max(scan.best_index - half, 0)
    end = min(scan.best_index + half + 1, scan.chisq_values.size)
    if end - start < 3:
        start = max(scan.best_index - 1, 0)
        end = min(scan.best_index + 2, scan.chisq_values.size)
    kappa_segment = scan.kappa_values[start:end]
    chisq_segment = scan.chisq_values[start:end]
    if kappa_segment.size < 3:
        return None
    coeffs = np.polyfit(kappa_segment, chisq_segment, 2)
    a, b, c = coeffs
    if a <= 0:
        return None
    kappa_star = -b / (2.0 * a)
    kappa_star = float(np.clip(kappa_star, kappa_segment.min(), kappa_segment.max()))
    chisq_star = float(a * kappa_star**2 + b * kappa_star + c)
    curvature = float(2.0 * a)
    sigma = float(math.sqrt(1.0 / max(curvature, 1e-12)))
    return QuadraticEstimate(
        kappa=kappa_star,
        chisq=chisq_star,
        curvature=curvature,
        sigma=sigma,
        support_points=kappa_segment,
    )


def _find_bracket(scan: ScanResult) -> tuple[float, float] | None:
    idx = scan.best_index
    if idx == 0 or idx == scan.kappa_values.size - 1:
        return None
    left = scan.kappa_values[idx - 1]
    right = scan.kappa_values[idx + 1]
    return float(left), float(right)


def golden_section_refinement(
    galaxies: Iterable[GalaxyData],
    model: CSGV4Model,
    bracket: tuple[float, float],
    tol: float,
    max_iter: int,
) -> GoldenSectionEstimate:
    phi = (1 + math.sqrt(5)) / 2.0
    a, b = bracket
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    fc = _compute_chisq(galaxies, model, c)
    fd = _compute_chisq(galaxies, model, d)
    iterations = 0
    while abs(b - a) > tol and iterations < max_iter:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / phi
            fc = _compute_chisq(galaxies, model, c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / phi
            fd = _compute_chisq(galaxies, model, d)
        iterations += 1
    kappa_star = (a + b) / 2.0
    chisq_star = _compute_chisq(galaxies, model, kappa_star)
    return GoldenSectionEstimate(
        kappa=float(kappa_star),
        chisq=float(chisq_star),
        iterations=iterations,
        bracket=(float(a), float(b)),
    )


def zoom_rescan(
    galaxies: Iterable[GalaxyData],
    model: CSGV4Model,
    scan: ScanResult,
    config: CSGConfig,
) -> tuple[ScanResult, dict[str, np.ndarray]]:
    if scan.kappa_values.size < 2:
        return scan, {}
    idx = scan.best_index
    lower_idx = max(idx - 2, 0)
    upper_idx = min(idx + 3, scan.kappa_values.size)
    local_values = scan.kappa_values[lower_idx:upper_idx]
    span = float(np.max(local_values) - np.min(local_values)) if local_values.size > 1 else 0.05
    half_width = max(span, 0.01)
    zoom_min = max(scan.best_kappa - 2.0 * half_width, config.kappa_min)
    zoom_max = min(scan.best_kappa + 2.0 * half_width, config.kappa_max)
    if zoom_min == zoom_max:
        zoom_max = min(zoom_min + 0.05, config.kappa_max)
    zoom_grid = np.linspace(zoom_min, zoom_max, config.zoom_samples)
    return scan_kappa(galaxies, model=model, config=config, kappa_grid=zoom_grid)


def refine_kappa(
    galaxies: Iterable[GalaxyData],
    model: CSGV4Model,
    scan: ScanResult,
    config: CSGConfig | None = None,
) -> RefinementResult:
    config = config or model.config
    quadratic = quadratic_refinement(scan, window=config.quadratic_window)

    golden: GoldenSectionEstimate | None = None
    bracket = _find_bracket(scan)
    if bracket is not None:
        golden = golden_section_refinement(
            galaxies,
            model,
            bracket,
            tol=config.gss_tol,
            max_iter=config.gss_max_iter,
        )

    zoom_scan, zoom_residual_map = zoom_rescan(galaxies, model, scan, config)

    return RefinementResult(
        quadratic=quadratic,
        golden=golden,
        zoom_scan=zoom_scan,
        zoom_residual_map=zoom_residual_map,
    )


def bootstrap_kappa(
    scan: ScanResult,
    config: CSGConfig,
) -> np.ndarray:
    if scan.residual_profiles is None or scan.sample_counts is None:
        raise ValueError("scan_kappa must be run with store_profiles=True to bootstrap.")
    profiles = scan.residual_profiles
    counts = np.array(scan.sample_counts)
    kappa_grid = scan.kappa_values
    rng = np.random.default_rng(config.bootstrap_seed)
    n_samples = config.bootstrap_samples
    estimates = np.zeros(n_samples, dtype=float)
    total_points = np.sum(counts)

    verbosity = _get_verbosity()
    disable_pbar = verbosity == 0

    bootstrap_iter = tqdm(
        range(n_samples),
        desc="Bootstrap resampling",
        disable=disable_pbar,
        leave=False,
    )

    for i in bootstrap_iter:
        chisq = np.zeros(kappa_grid.size, dtype=float)
        for g_idx, count in enumerate(counts):
            if count == 0:
                continue
            valid = profiles[g_idx, :, :count]
            resample_idx = rng.integers(0, count, size=count)
            sampled = valid[:, resample_idx]
            mse = np.mean(np.square(sampled), axis=1)
            chisq += mse * (count / total_points)
        estimates[i] = float(kappa_grid[np.argmin(chisq)])

    return estimates
