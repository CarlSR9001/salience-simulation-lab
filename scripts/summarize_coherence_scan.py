from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from tabulate import tabulate


def main() -> None:
    latest = sorted(Path("artifacts/coherence_scanner").glob("coherence_scan_*.json"))[-1]
    data = json.loads(latest.read_text())
    results = data["results"]

    total = len(results)
    max_entry = max(results, key=lambda r: r["anomaly_score"])
    min_entry = min(results, key=lambda r: r["anomaly_score"])

    by_phase = defaultdict(list)
    by_damping = defaultdict(list)
    for r in results:
        by_phase[r["phase"]].append(r["anomaly_score"])
        by_damping[r["damping"]].append(r["anomaly_score"])

    phase_rows = []
    for phase, scores in sorted(by_phase.items()):
        phase_rows.append((phase, sum(scores) / len(scores), max(scores), min(scores)))

    damping_rows = []
    for damping, scores in sorted(by_damping.items()):
        damping_rows.append((damping, sum(scores) / len(scores), max(scores), min(scores)))

    print(f"Latest scan: {latest}")
    print(f"Total samples: {total}")
    print(f"Max anomaly: {max_entry}")
    print(f"Min anomaly: {min_entry}")

    print("\nPhase summary (mean / max / min anomaly):")
    print(tabulate(phase_rows, headers=["phase", "mean", "max", "min"], tablefmt="github", floatfmt=".6f"))

    print("\nDamping summary (mean / max / min anomaly):")
    print(tabulate(damping_rows, headers=["damping", "mean", "max", "min"], tablefmt="github", floatfmt=".6f"))


if __name__ == "__main__":
    main()
