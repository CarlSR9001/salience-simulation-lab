"""Analyze low-surface-brightness dwarfs and high-SB spiral outliers."""

from __future__ import annotations

from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from csg_v4.config import CSGConfig
from csg_v4.data_io import SPARCParser
from csg_v4.analysis import load_synthetic_galaxies
from csg_v4.model import CSGV4Model, ModelOutputs
from csg_v4.optimizer import scan_kappa
from csg_v4.reporting import compute_error_metrics


def summarize(group):
    if not group:
        return {}
    keys = group[0].keys()
    return {k: mean(entry[k] for entry in group) for k in keys if isinstance(group[0][k], (int, float))}


def plot_rotation(galaxy, outputs: ModelOutputs, out_path: Path) -> None:
    plt.figure(figsize=(7.0, 5.0))
    plt.plot(galaxy.radii_kpc, galaxy.v_obs, "o", label="Observed", linewidth=2.0)
    plt.plot(galaxy.radii_kpc, outputs.v_pred, "-", label="Predicted", linewidth=2.0)
    plt.plot(galaxy.radii_kpc, np.sqrt(galaxy.v_bar**2), "--", label="Baryonic", linewidth=1.5)
    plt.xlabel("Radius [kpc]")
    plt.ylabel("Velocity [km/s]")
    plt.title(f"Rotation curve: {galaxy.name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "sparc"
    table1_path = data_dir / "table1.dat"
    table2_path = data_dir / "table2.dat"

    config = CSGConfig()
    model = CSGV4Model(config)

    parser = SPARCParser(table1_path, table2_path)
    table1 = parser.table1
    sparc_all = parser.load_galaxies(selection=None, config=config)
    q1_names = table1[table1["Q"] == 1].index.tolist()
    sparc_quality = [sparc_all[name] for name in q1_names if name in sparc_all]

    synthetic = list(load_synthetic_galaxies().values())

    records = []
    all_galaxies = sparc_quality + synthetic

    plot_dir = root / "artifacts" / "spiral_outliers"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for galaxy in all_galaxies:
        scan, residual_map = scan_kappa([galaxy], model=model, store_profiles=True)
        best_kappa = scan.best_kappa
        outputs = model.predict_velocity(galaxy, best_kappa)
        metrics = compute_error_metrics(galaxy, outputs)
        salience = model.compute_salience(galaxy)
        meta_row = table1.loc[galaxy.name] if galaxy.name in table1.index else None
        records.append(
            {
                "name": galaxy.name,
                "type": galaxy.galaxy_type,
                "n_points": metrics.n_points,
                "kappa": best_kappa,
                "mean_abs": metrics.mean_percent_error,
                "max_abs": metrics.max_percent_error,
                "core": float(np.mean(salience["CORE"])),
                "penalty": float(np.mean(salience["PENALTY"])),
                "hype": float(np.mean(salience["HYPE"])),
                "w_mean": float(np.mean(salience["W_local"])),
                "m_mean": float(np.mean(salience["M_local"])),
                "sb_disk": float(meta_row["SBdisk"]) if meta_row is not None else float("nan"),
                "v_flat": float(meta_row["Vflat"]) if meta_row is not None else float("nan"),
                "residuals": residual_map[galaxy.name],
            }
        )

        if galaxy.name in {"F568-1", "F568-3", "F571-8", "IC4202"}:
            plot_rotation(galaxy, outputs, plot_dir / f"rotation_{galaxy.name}.png")

    dwarf_branch = [r for r in records if "lsb_dwarf" in r["type"] and r["mean_abs"] < 80]
    outliers = [r for r in records if r["type"] in {"hsb_spiral", "spiral", "grand_design"} and r["mean_abs"] > 80]
    synthetic_branch = [r for r in records if r["name"].endswith("-like")]

    print("Low-surface-brightness dwarf branch (Q=1):")
    print(tabulate([
        (r["name"], r["kappa"], r["mean_abs"], r["sb_disk"], r["core"], r["penalty"], r["hype"], r["w_mean"], r["m_mean"])
        for r in dwarf_branch
    ], headers=["Galaxy", "kappa", "mean|dv/v| [%]", "SBdisk", "CORE", "Penalty", "HYPE", "W_mean", "M_mean"], tablefmt="github"))
    dwarf_summary = summarize([
        {
            "kappa": r["kappa"],
            "mean_abs": r["mean_abs"],
            "sb_disk": r["sb_disk"],
            "w_mean": r["w_mean"],
            "m_mean": r["m_mean"],
        }
        for r in dwarf_branch
    ])
    print("\nDwarf branch summary:")
    for key, value in dwarf_summary.items():
        print(f"  {key}: {value:.4f}")

    print("\nHigh-SB spiral outliers:")
    print(tabulate([
        (r["name"], r["type"], r["kappa"], r["mean_abs"], r["sb_disk"], r["v_flat"], r["core"], r["penalty"], r["hype"])
        for r in outliers
    ], headers=["Galaxy", "Type", "kappa", "mean|dv/v| [%]", "SBdisk", "Vflat", "CORE", "Penalty", "HYPE"], tablefmt="github"))

    for name in ["F568-1", "F568-3", "F571-8", "IC4202"]:
        record = next((r for r in records if r["name"] == name), None)
        if record is None:
            continue
        resid = record["residuals"]
        r_stats = (
            float(np.mean(resid)),
            float(np.max(np.abs(resid))),
            float(np.mean(resid[: len(resid)//2])),
            float(np.mean(resid[len(resid)//2 :])),
        )
        print(
            f"Residual summary for {name}: mean={r_stats[0]:+.3f}, max|resid|={r_stats[1]:.3f}, "
            f"inner mean={r_stats[2]:+.3f}, outer mean={r_stats[3]:+.3f}"
        )

    print("\nSynthetic benchmarks (for comparison):")
    print(tabulate([
        (r["name"], r["kappa"], r["mean_abs"], r["core"], r["penalty"], r["hype"], r["w_mean"], r["m_mean"])
        for r in synthetic_branch
    ], headers=["Galaxy", "kappa", "mean|dv/v| [%]", "CORE", "Penalty", "HYPE", "W_mean", "M_mean"], tablefmt="github"))


if __name__ == "__main__":
    main()
