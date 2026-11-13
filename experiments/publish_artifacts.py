"""Copy the finished outputs into the project artifacts directory.

    uv run python experiments/publish_artifacts.py

Creates <artifacts>/simulator-study/ with the report, the figures (and the scripts that
made them), the raw CSVs and the offline HTML viewer.
"""
from __future__ import annotations

import gzip
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path("/Users/vishruth/.local/share/openresearch/files/ai-exploration-exploitation")
OUT = ARTIFACTS / "simulator-study"


def main() -> None:
    if not ARTIFACTS.exists():
        print(f"artifacts directory {ARTIFACTS} does not exist", file=sys.stderr)
        raise SystemExit(1)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "viewer").mkdir(parents=True, exist_ok=True)

    for fig in sorted((ROOT / "figs").glob("fig*.pdf")) + sorted((ROOT / "figs").glob("fig*.svg")):
        shutil.copy2(fig, OUT / "figures" / fig.name)
    shutil.copy2(ROOT / "figs" / "orx_figstyle.py", OUT / "figures" / "orx_figstyle.py")
    for script in ("make_figures.py", "run_grid.py", "run_sweep.py", "summarize.py"):
        shutil.copy2(ROOT / "experiments" / script, OUT / "figures" / script)

    for name in ("grid.csv", "sweep.csv", "tables.md", "grid_meta.json", "sweep_meta.json"):
        src = ROOT / "results" / name
        if src.exists():
            shutil.copy2(src, OUT / "data" / name)
    # the per-step curves are large (1.6M rows); ship them compressed
    curves = ROOT / "results" / "curves.csv"
    if curves.exists():
        with open(curves, "rb") as src, gzip.open(OUT / "data" / "curves.csv.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)

    for src in sorted((ROOT / "results" / "viewer").glob("*")):
        if src.is_file():
            shutil.copy2(src, OUT / "viewer" / src.name)

    report = ROOT / "docs" / "07_results_report.md"
    if report.exists():
        # the published copy sits next to figures/, so fix the relative links
        text = report.read_text().replace("](../figs/", "](figures/")
        (OUT / "simulator-study-report.md").write_text(text)
    print(f"published to {OUT}")


if __name__ == "__main__":
    main()
