"""Analysis routines orchestrating the CSG V4 pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm

from .config import CSGConfig
from .galaxy import GalaxyData
from .model import CSGV4Model
from .optimizer import (
    RefinementResult,
    ScanResult,
    bootstrap_kappa,
    refine_kappa,
    scan_kappa,
)
from .reporting import GalaxyMetrics, compute_error_metrics, summarize_galaxy_metrics
from .synthetic_data import load_synthetic_galaxies


def _get_verbosity() -> int:
    """Get verbosity level from environment: 0=quiet, 1=normal, 2=verbose."""
    return int(os.environ.get("CSG_VERBOSITY", "1"))


@dataclass(slots=True)
class AnalysisArtifacts:
    config: CSGConfig
    scan_result: ScanResult
    metrics: Dict[str, GalaxyMetrics]
    tables: str
    outputs: Dict[str, Tuple[GalaxyData, np.ndarray, np.ndarray]]
    residual_map: Dict[str, np.ndarray]
    refinement: RefinementResult
    bootstrap_kappas: np.ndarray
    weighted_best_kappa: Optional[float]
    weighted_chisq: Optional[np.ndarray]
    plot_targets: List[str]


class CSGAnalysis:
    def __init__(self, config: CSGConfig | None = None) -> None:
        self.config = config or CSGConfig()
        self.model = CSGV4Model(self.config)

    def _plot_rotation_curve(self, galaxy: GalaxyData, outputs: np.ndarray, v_pred: np.ndarray, out_path: Path) -> None:
        plt.figure(figsize=(7.5, 5.0))
        plt.plot(galaxy.radii_kpc, galaxy.v_obs, "o-", label="Observed", linewidth=2.0)
        plt.plot(galaxy.radii_kpc, galaxy.v_bar, "s--", label="Baryonic", linewidth=2.0)
        plt.plot(galaxy.radii_kpc, v_pred, "^-", label="CSG V4", linewidth=2.0)
        plt.xlabel("Radius [kpc]")
        plt.ylabel("v_c [km/s]")
        plt.title(f"Rotation Curve: {galaxy.name}\n(kappa_c={self.best_kappa:.3f})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def _plot_kappa_scan(self, scan_result: ScanResult, out_path: Path, label: str = "primary") -> None:
        plt.figure(figsize=(7.5, 5.0))
        plt.plot(scan_result.kappa_values, scan_result.chisq_values, "-", linewidth=2.0)
        plt.scatter([scan_result.best_kappa], [scan_result.chisq_values[scan_result.best_index]], color="red", zorder=5)
        plt.xlabel("kappa_c")
        plt.ylabel("Mean squared fractional residual")
        plt.title(f"CSG V4 kappa scan ({label})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def _plot_residual_surface(self, scan_result: ScanResult, out_path: Path, max_galaxies: Optional[int] = None) -> None:
        if scan_result.residual_profiles is None or scan_result.sample_counts is None or scan_result.radii_arrays is None:
            return
        total = len(scan_result.sample_counts)
        if total == 0:
            return
        if max_galaxies is not None and total > max_galaxies:
            indices = np.linspace(0, total - 1, num=max_galaxies, dtype=int)
        else:
            indices = np.arange(total)
        fig = plt.figure(figsize=(9.0, 6.5))
        ax = fig.add_subplot(111, projection="3d")
        colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
        for color, g_idx in zip(colors, indices):
            count = scan_result.sample_counts[g_idx]
            if count == 0:
                continue
            radii = scan_result.radii_arrays[g_idx]
            residuals = scan_result.residual_profiles[g_idx, :, :count]
            kappa = scan_result.kappa_values[:, None]
            ax.plot_surface(
                kappa,
                np.broadcast_to(radii, residuals.shape),
                residuals,
                rstride=max(count // 20, 1),
                cstride=max(scan_result.kappa_values.size // 200, 1),
                alpha=0.4,
                color=color,
                edgecolor="none",
            )
        ax.set_xlabel("kappa_c")
        ax.set_ylabel("Radius [kpc]")
        ax.set_zlabel("Residual")
        ax.set_title("Residual surface across kappa and radius")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def _compute_radius_weighted_chisq(self, scan_result: ScanResult) -> Tuple[Optional[np.ndarray], Optional[float]]:
        profiles = scan_result.residual_profiles
        counts = scan_result.sample_counts
        radii = scan_result.radii_arrays
        if profiles is None or counts is None or radii is None:
            return None, None
        kappa_vals = scan_result.kappa_values
        weighted = np.zeros_like(kappa_vals)
        total_weight = 0.0
        for g_idx, count in enumerate(counts):
            if count == 0:
                continue
            r = radii[g_idx][:count]
            weights = r / (np.max(r) + 1.0e-6)
            weights = weights / (np.sum(weights) + 1.0e-6)
            residuals = profiles[g_idx, :, :count]
            mse = np.sum(np.square(residuals) * weights, axis=1)
            weighted += mse
            total_weight += 1.0
        if total_weight == 0:
            return None, None
        weighted /= total_weight
        idx = int(np.argmin(weighted))
        return weighted, float(kappa_vals[idx])

    def run(
        self,
        galaxies: Iterable[GalaxyData] | None = None,
        output_dir: str | Path | None = None,
        plot_limit: Optional[int] = None,
        surface_sample: Optional[int] = 12,
    ) -> AnalysisArtifacts:
        galaxies_dict = (
            {gal.name: gal for gal in galaxies}
            if galaxies is not None
            else load_synthetic_galaxies()
        )
        galaxies = list(galaxies_dict.values())

        verbosity = _get_verbosity()
        disable_pbar = verbosity == 0

        if verbosity >= 1:
            print("Running kappa scan...")

        scan_result, residual_map = scan_kappa(galaxies, model=self.model, store_profiles=True)
        best_kappa = scan_result.best_kappa
        self.best_kappa = best_kappa

        if verbosity >= 1:
            print("Computing galaxy metrics...")

        metrics: Dict[str, GalaxyMetrics] = {}
        outputs: Dict[str, Tuple[GalaxyData, np.ndarray, np.ndarray]] = {}

        galaxy_iter = tqdm(galaxies, desc="Computing metrics", disable=disable_pbar, leave=False)
        for galaxy in galaxy_iter:
            results = self.model.predict_velocity(galaxy, best_kappa)
            metrics[galaxy.name] = compute_error_metrics(galaxy, results)
            outputs[galaxy.name] = (galaxy, results.v_pred, results.a_eff)

        summary_table = summarize_galaxy_metrics(metrics.values())

        metric_items = list(metrics.items())
        if plot_limit is None:
            plot_targets = [name for name, _ in metric_items]
        else:
            sorted_items = sorted(metric_items, key=lambda item: item[1].mean_percent_error, reverse=True)
            plot_targets = [name for name, _ in sorted_items[:plot_limit]]
        plot_target_set = set(plot_targets)

        if verbosity >= 1:
            print("Refining kappa estimate...")

        refinement = refine_kappa(galaxies, self.model, scan_result, config=self.config)

        if verbosity >= 1:
            print("Running bootstrap analysis...")

        bootstrap_samples = bootstrap_kappa(scan_result, self.config)

        if verbosity >= 1:
            print("Computing radius-weighted chi-squared...")

        weighted_chisq, weighted_kappa = self._compute_radius_weighted_chisq(scan_result)

        artifact_dir = Path(output_dir) if output_dir else Path("artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)

        if verbosity >= 1:
            print(f"Generating plots (saving to {artifact_dir})...")

        self._plot_kappa_scan(scan_result, artifact_dir / "kappa_scan.png")
        if refinement.zoom_scan is not None:
            self._plot_kappa_scan(refinement.zoom_scan, artifact_dir / "kappa_scan_zoom.png", label="zoom")

        # Plot rotation curves with progress bar
        plot_items = [(name, outputs[name]) for name in plot_target_set if name in outputs]
        plot_iter = tqdm(plot_items, desc="Plotting rotation curves", disable=disable_pbar, leave=False)
        for galaxy_name, (galaxy, v_pred, _) in plot_iter:
            self._plot_rotation_curve(galaxy, galaxy.v_obs, v_pred, artifact_dir / f"rotation_curve_{galaxy_name}.png")

        self._plot_residual_surface(scan_result, artifact_dir / "residual_surface.png", max_galaxies=surface_sample)

        (artifact_dir / "metrics.txt").write_text(summary_table)

        analysis_report_lines = [
            f"Primary scan best kappa_c: {scan_result.best_kappa:.6f} (chi^2={scan_result.best_chisq:.6e})",
        ]
        if refinement.quadratic is not None:
            analysis_report_lines.append(
                "Quadratic refinement: "
                f"kappa_c={refinement.quadratic.kappa:.6f}, "
                f"chi^2={refinement.quadratic.chisq:.6e}, "
                f"sigma~{refinement.quadratic.sigma:.4f}"
            )
        if refinement.golden is not None:
            analysis_report_lines.append(
                "Golden-section: "
                f"kappa_c={refinement.golden.kappa:.6f}, "
                f"chi^2={refinement.golden.chisq:.6e}, "
                f"iters={refinement.golden.iterations}, "
                f"bracket={refinement.golden.bracket}"
            )
        if refinement.zoom_scan is not None:
            analysis_report_lines.append(
                f"Zoom scan best kappa_c: {refinement.zoom_scan.best_kappa:.6f} "
                f"(chi^2={refinement.zoom_scan.best_chisq:.6e})"
            )
        analysis_report_lines.append(
            f"Bootstrap mean: {np.mean(bootstrap_samples):.6f}, std: {np.std(bootstrap_samples):.6f}"
        )
        if weighted_kappa is not None:
            analysis_report_lines.append(
                f"Radius-weighted best kappa_c: {weighted_kappa:.6f}"
            )
        if plot_targets:
            analysis_report_lines.append(
                "Rotation curves plotted for: "
                + (", ".join(plot_targets[:10]) + ("..." if len(plot_targets) > 10 else ""))
            )
        (artifact_dir / "kappa_analysis.txt").write_text("\n".join(analysis_report_lines))

        return AnalysisArtifacts(
            config=self.config,
            scan_result=scan_result,
            metrics=metrics,
            tables=summary_table,
            outputs=outputs,
            residual_map=residual_map,
            refinement=refinement,
            bootstrap_kappas=bootstrap_samples,
            weighted_best_kappa=weighted_kappa,
            weighted_chisq=weighted_chisq,
            plot_targets=plot_targets,
        )


def run_synthetic_pipeline(output_dir: str | Path | None = None) -> AnalysisArtifacts:
    analysis = CSGAnalysis()
    return analysis.run(output_dir=output_dir)
