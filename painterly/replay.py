"""Re-render a run from its events.jsonl, e.g. after changing the brush engine."""

from __future__ import annotations

import json
from pathlib import Path

from .canvas import Canvas
from .tools import ToolError, apply_tool


def replay(run_dir: Path, out: Path | None = None) -> Path:
    cfg = json.loads((run_dir / "config.json").read_text())
    canvas = Canvas(cfg["width"], cfg["height"], paper=cfg["paper"], seed=cfg["seed"])
    for line in (run_dir / "events.jsonl").read_text().splitlines():
        ev = json.loads(line)
        if ev["kind"] != "tool" or ev["tool"] in ("look", "finish"):
            continue
        try:
            apply_tool(canvas, ev["tool"], ev["input"], ev["seed"])
        except ToolError as e:
            print(f"skipped {ev['tool']}: {e}")
    out = out or run_dir / "replay.png"
    canvas.image().save(out)
    return out
