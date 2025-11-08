"""Inspect high-quality SPARC galaxies for CSG V4 diagnostic metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tabulate import tabulate

from csg_v4.config import CSGConfig
from csg_v4.analysis import load_synthetic_galaxies
from csg_v4.data_io import SPARCParser
from csg_v4.model import CSGV4Model
from csg_v4.optimizer import scan_kappa


def filter_by_quality(galaxies, metadata_table, max_quality: int | None):
    if max_quality is None:
        return galaxies
    filtered = []
    for galaxy in galaxies:
        if galaxy.name not in metadata_table.index:
            continue
        quality = int(metadata_table.loc[galaxy.name, "Q"])
        if quality <= max_quality:
            filtered.append(galaxy)
    return filtered


def residual_stats(resid: np.ndarray) -> Tuple[float, float, float, float]:
    abs_resid = np.abs(resid)
    return (
        float(np.mean(resid)),
        float(np.mean(abs_resid)),
        float(np.median(resid)),
        float(np.max(abs_resid)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sparc", "synthetic"], default="sparc")
    parser.add_argument("--max-quality", type=int, default=1)
    parser.add_argument("--top", type=int, default=8, help="Show top-N galaxies by mean absolute residual.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "sparc"
    table1 = data_dir / "table1.dat"
    table2 = data_dir / "table2.dat"
    config = CSGConfig()

    if args.dataset == "sparc":
        sparc = SPARCParser(table1, table2)
        galaxies_dict = sparc.load_galaxies(selection=None, config=config, stellar_ml=0.5)
        metadata_table = sparc.table1
        galaxies_all = list(galaxies_dict.values())
        galaxies = filter_by_quality(galaxies_all, metadata_table, args.max_quality)
        print(f"Using {len(galaxies)} SPARC galaxies after quality filter (<= {args.max_quality}).")
    else:
        galaxies = list(load_synthetic_galaxies().values())
        print(f"Using {len(galaxies)} synthetic galaxies.")

    model = CSGV4Model(config)
    scan_result, _ = scan_kappa(galaxies, model=model, store_profiles=False)
    best_kappa = scan_result.best_kappa
    print(f"best_kappa = {best_kappa:.6f}")

    rows: List[Tuple] = []
    for galaxy in galaxies:
        outputs = model.predict_velocity(galaxy, best_kappa)
        resid = model.residuals(galaxy, outputs)
        salience = model.compute_salience(galaxy)
        mean_resid, mean_abs, median_resid, max_abs = residual_stats(resid)

        hype_bonus = float(salience.get("HYPE_workhorse", 0.0))
        workhorse_flag = bool(salience.get("Workhorse_flag", 0.0))
        row = (
            galaxy.name,
            galaxy.galaxy_type,
            galaxy.n_radii,
            f"{mean_resid:+.2f}",
            f"{mean_abs:.2f}",
            f"{median_resid:+.2f}",
            f"{max_abs:.2f}",
            f"{salience['phi_galaxy']:.2f}",
            f"{salience['K_galaxy']:.2f}",
            f"{salience['J_galaxy']:.2f}",
            f"{np.mean(salience['CORE']):.3f}",
            f"{np.mean(salience['HYPE']):.3f}",
            f"{np.mean(salience['PENALTY']):.3f}",
            "Y" if workhorse_flag else "N",
            f"{hype_bonus:.2f}",
            f"{np.mean(salience['W_local']):.2f}",
            f"{np.mean(salience['M_local']):.2f}",
        )
        rows.append(row)

    rows.sort(key=lambda r: float(r[4]), reverse=True)
    headers = [
        "Galaxy",
        "Type",
        "N",
        "mean resid",
        "mean|resid|",
        "median",
        "max|resid|",
        "phi",
        "K",
        "J",
        "CORE",
        "HYPE",
        "Penalty",
        "Workhorse",
        "HypeΔ",
        "W mean",
        "M mean",
    ]

    print(tabulate(rows[: args.top], headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
