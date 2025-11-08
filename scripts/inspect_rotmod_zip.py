"""List contents of SPARC Rotmod zip and filter for target galaxies."""

from __future__ import annotations

from pathlib import Path
import zipfile

TARGETS = {"F568-1", "F568-3", "F571-8", "IC4202"}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    zip_path = root / "data" / "sparc" / "extended" / "Rotmod_LTG.zip"
    if not zip_path.exists():
        raise SystemExit(f"Missing {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            base = Path(name).stem
            if any(base.startswith(target) for target in TARGETS):
                print(name)


if __name__ == "__main__":
    main()
