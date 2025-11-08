"""Detailed diagnostics for SPARC galaxy F571-8 using rotmod data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from csg_v4.config import CSGConfig
from csg_v4.data_io import SPARCParser
from csg_v4.model import CSGV4Model
from csg_v4.optimizer import scan_kappa
from csg_v4.reporting import compute_error_metrics

TARGET = "F571-8"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "sparc"
    parser = SPARCParser(data_dir / "table1.dat", data_dir / "table2.dat")
    config = CSGConfig()
    model = CSGV4Model(config)

    record = parser.load_record(TARGET)
    galaxy = parser.to_galaxy_data(record, config)
    scan, residual_map = scan_kappa([galaxy], model=model, store_profiles=True)
    best_kappa = scan.best_kappa
    outputs = model.predict_velocity(galaxy, best_kappa)
    metrics = compute_error_metrics(galaxy, outputs)

    residuals = residual_map[TARGET]
    radii = galaxy.radii_kpc
    median_r = np.median(radii)
    inner_mask = radii <= median_r
    outer_mask = radii > median_r
    inner_mean = float(np.mean(residuals[inner_mask]))
    outer_mean = float(np.mean(residuals[outer_mask]))
    inner_abs = float(np.mean(np.abs(residuals[inner_mask])))
    outer_abs = float(np.mean(np.abs(residuals[outer_mask])))

    rows = [
        (TARGET, galaxy.galaxy_type, best_kappa, metrics.mean_percent_error, metrics.max_percent_error),
    ]
    print(tabulate(rows, headers=["Galaxy", "Type", "Best kappa_c", "Mean |dv|/v [%]", "Max |dv|/v [%]"], tablefmt="github"))
    print(
        f"Inner residual mean={inner_mean:.3f} (|mean|={inner_abs:.3f}), "
        f"outer residual mean={outer_mean:.3f} (|mean|={outer_abs:.3f})"
    )

    v_baryonic = np.sqrt(record.v_gas**2 + record.v_disk**2 + record.v_bulge**2)
    frac_residuals = (outputs.v_pred - galaxy.v_obs) / np.maximum(galaxy.v_obs, 1e-3)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(radii, galaxy.v_obs, "o", label="Observed", linewidth=2.0)
    ax.plot(radii, outputs.v_pred, "-", label="Predicted", linewidth=2.0)
    ax.plot(radii, v_baryonic, "--", label="Baryonic", linewidth=1.5)
    ax.set_xlabel("Radius [kpc]")
    ax.set_ylabel("Velocity [km/s]")
    ax.set_title(f"F571-8 rotation curve (kappa_c={best_kappa:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_dir = root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "f5718_extended_rotation.png")
    plt.close(fig)

    plt.figure(figsize=(7.0, 3.5))
    plt.plot(radii, frac_residuals, "o-", label="Fractional residual")
    plt.axhline(0.0, color="k", linewidth=1.0, linestyle=":")
    plt.axvline(median_r, color="gray", linestyle="--", linewidth=1.0, label="Median radius")
    plt.xlabel("Radius [kpc]")
    plt.ylabel("(v_pred - v_obs)/v_obs")
    plt.title("F571-8 residual profile")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "f5718_residual_profile.png")
    plt.close()


if __name__ == "__main__":
    main()
