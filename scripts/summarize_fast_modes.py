"""Summarize key metrics from fast-mode experiments (I–N)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

FAST_MODE_DIRECTORIES: Dict[str, Path] = {
    "experiment_i_time_overclock": Path("artifacts/time_overclock"),
    "experiment_l_pulsed_overclock": Path("artifacts/pulsed_overclock"),
    "experiment_m_soft_release": Path("artifacts/soft_release"),
    "experiment_n_reflex_edge": Path("artifacts/reflex_edge"),
}


@dataclass
class ExperimentSummary:
    experiment: str
    artifact_path: Path
    highlight: Dict[str, object]


def list_json_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"))


def load_json(path: Path) -> Optional[Iterable[dict]]:
    try:
        with path.open("r", encoding="utf-8") as fp:
            content = json.load(fp)
    except Exception:
        return None

    if isinstance(content, list):
        return content
    if isinstance(content, dict):
        return [content]
    return None


def summarize_time_overclock(path: Path) -> ExperimentSummary:
    payload = load_json(path) or []
    best = {
        "alpha_boost": "n/a",
        "converge_steps": float("inf"),
        "speedup": 0.0,
    }

    for item in payload:
        if not isinstance(item, dict):
            continue
        if item.get("experiment_name") != "experiment_i_time_overclock":
            continue
        for run in item.get("runs", []):
            summary = run.get("summary", {})
            converge = summary.get("t_converge_0.99")
            speedup = summary.get("speedup_0.99")
            if converge is None or speedup is None:
                continue
            if converge < best["converge_steps"]:
                best = {
                    "alpha_boost": run.get("alpha_boost", "?"),
                    "converge_steps": converge,
                    "speedup": speedup,
                }

    highlight = {
        "best_alpha_boost": best["alpha_boost"],
        "best_converge_steps": best["converge_steps"],
        "best_speedup": best["speedup"],
    }
    return ExperimentSummary("experiment_i_time_overclock", path, highlight)


def summarize_pulsed_overclock(path: Path) -> ExperimentSummary:
    payload = load_json(path) or []
    best = {
        "alpha_boost": "n/a",
        "t_converge_0.99": float("inf"),
        "speedup_0.99": 0.0,
    }

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        if entry.get("experiment_name") != "experiment_l_pulsed_overclock":
            continue
        converge = entry.get("t_converge_0.99")
        speedup = entry.get("speedup_0.99")
        if converge is None or speedup is None:
            continue
        if converge < best["t_converge_0.99"]:
            best = {
                "alpha_boost": entry.get("alpha_boost", "?"),
                "t_converge_0.99": converge,
                "speedup_0.99": speedup,
            }

    highlight = {
        "best_alpha_boost": best["alpha_boost"],
        "best_converge_steps": best["t_converge_0.99"],
        "best_speedup": best["speedup_0.99"],
    }
    return ExperimentSummary("experiment_l_pulsed_overclock", path, highlight)


def summarize_soft_release(path: Path) -> ExperimentSummary:
    payload = load_json(path) or []
    best = {
        "rise_time_90": float("inf"),
        "peak_rate": 0.0,
        "config": "n/a",
    }

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        if entry.get("experiment_name") != "experiment_m_soft_release":
            continue
        rise = entry.get("rise_time_90")
        if rise is None:
            continue
        if rise < best["rise_time_90"]:
            best = {
                "rise_time_90": rise,
                "peak_rate": entry.get("post_release_peak_rate", 0.0),
                "config": entry.get("configuration", "?"),
            }

    highlight = {
        "fastest_rise_time": best["rise_time_90"],
        "peak_rate": best["peak_rate"],
        "config": best["config"],
    }
    return ExperimentSummary("experiment_m_soft_release", path, highlight)


def summarize_reflex_edge(path: Path) -> ExperimentSummary:
    payload = load_json(path) or []
    passing: List[dict] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        if entry.get("experiment_name") != "experiment_n_reflex_edge":
            continue
        core_rt = entry.get("core_response_time_90")
        if core_rt is None or core_rt != core_rt:  # NaN check
            continue
        if entry.get("core_salience_min", 0.0) < 0.8:
            continue
        passing.append(entry)

    if passing:
        best_entry = min(passing, key=lambda e: e["core_response_time_90"])
        highlight = {
            "count_passing": len(passing),
            "best_core_rt90": best_entry["core_response_time_90"],
            "best_mu_edge": best_entry.get("mu_edge"),
            "best_lambda_core": best_entry.get("lambda_core"),
            "best_merge_rate": best_entry.get("merge_rate"),
        }
    else:
        highlight = {
            "count_passing": 0,
            "best_core_rt90": float("nan"),
        }

    return ExperimentSummary("experiment_n_reflex_edge", path, highlight)


SUMMARY_FUNCS = {
    "experiment_i_time_overclock": summarize_time_overclock,
    "experiment_l_pulsed_overclock": summarize_pulsed_overclock,
    "experiment_m_soft_release": summarize_soft_release,
    "experiment_n_reflex_edge": summarize_reflex_edge,
}


def find_latest_artifact(directory: Path) -> Optional[Path]:
    files = list_json_files(directory)
    return files[-1] if files else None


def main() -> None:
    summaries: List[ExperimentSummary] = []
    for experiment, directory in FAST_MODE_DIRECTORIES.items():
        latest = find_latest_artifact(directory)
        if not latest:
            continue
        summarize_func = SUMMARY_FUNCS[experiment]
        summaries.append(summarize_func(latest))

    report = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "summaries": [
            {
                "experiment": summary.experiment,
                "artifact": str(summary.artifact_path),
                "highlight": summary.highlight,
            }
            for summary in summaries
        ],
    }

    out_dir = Path("artifacts/fast_modes")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fast_modes_summary_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print("=== Fast-Mode Summary ===")
    for summary in summaries:
        print(f"[{summary.experiment}] -> {summary.artifact_path.name}")
        for key, value in summary.highlight.items():
            print(f"  {key}: {value}")
        print()
    print(f"Summary written to {out_path}")


if __name__ == "__main__":
    main()
