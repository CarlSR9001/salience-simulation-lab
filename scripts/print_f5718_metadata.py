"""Print SPARC table1 metadata for F571-8."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

columns = [
    "Galaxy",
    "T",
    "D",
    "e_D",
    "f_D",
    "Inc",
    "e_Inc",
    "L36",
    "e_L36",
    "Reff",
    "SBeff",
    "Rdisk",
    "SBdisk",
    "MHI",
    "RHI",
    "Vflat",
    "e_Vflat",
    "Q",
    "Ref",
]


def main() -> None:
    table1_path = Path("c:/SAL/data/sparc/table1.dat")
    df = pd.read_fwf(table1_path, names=columns)
    df["Galaxy"] = df["Galaxy"].astype(str).str.strip()
    record = df[df["Galaxy"] == "F571-8"]
    if record.empty:
        print("F571-8 not found")
        return
    print(record[["Galaxy", "D", "Inc", "Vflat", "e_Vflat", "Q", "Ref"]].to_string(index=False))


if __name__ == "__main__":
    main()
