"""Inspect SPARC table entries for debugging ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from csg_v4.data_io import SPARCParser


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SPARC rotation curve data for a galaxy.")
    parser.add_argument("galaxy", help="Galaxy name as listed in SPARC tables (case sensitive).")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root path.")
    args = parser.parse_args()

    data_dir = args.root / "data" / "sparc"
    table1 = data_dir / "table1.dat"
    table2 = data_dir / "table2.dat"

    raw_table2 = pd.read_fwf(table2, names=[
        "Galaxy",
        "D",
        "radius_kpc",
        "v_obs",
        "e_v_obs",
        "v_gas",
        "v_disk",
        "v_bulge",
        "sb_disk",
        "sb_bulge",
    ])
    subset = raw_table2[raw_table2["Galaxy"] == args.galaxy]
    print(f"Raw rows for {args.galaxy}: {len(subset)}")
    print(subset.head())

    parser = SPARCParser(table1, table2)
    try:
        record = parser.load_record(args.galaxy)
    except Exception as exc:  # noqa: BLE001
        print(f"load_record failed: {exc}")
        return

    print(f"load_record radii count: {record.radius_kpc.size}")
    print("Radii sample:", record.radius_kpc[:5])
    print("v_obs sample:", record.v_obs[:5])


if __name__ == "__main__":
    main()
