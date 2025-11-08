"""List downloadable SPARC resource links from the main page."""

from __future__ import annotations

import re
import sys
import urllib.request

BASE_URL = "https://astroweb.cwru.edu/SPARC/"


def main() -> None:
    with urllib.request.urlopen(BASE_URL) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    pattern = re.compile(r"href=\"([^\"]+\.(?:zip|mrt|table1|table2))\"")
    matches = pattern.findall(html)
    for link in matches:
        if not link.startswith("http"):
            link = BASE_URL + link
        print(link)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
