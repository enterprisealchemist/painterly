"""Build a self-contained HTML page for stepping through a run, stroke by stroke.

Each step shows the canvas after that stroke, the path the brush took, and everything
the model thought or said on the way to it, including strokes the canvas refused.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image

_TEMPLATE = Path(__file__).with_name("viewer_template.html")


def _frame_uri(path: Path, max_side: int) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85, subsampling=0)
    return "data:image/jpeg;base64," + base64.standard_b64encode(buf.getvalue()).decode()


def build_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = [{"frame": 0, "kind": "start", "notes": []}]
    pending: list[dict[str, Any]] = []
    last_frame = 0
    finish: dict[str, Any] | None = None
    for ev in events:
        k = ev["kind"]
        if k in ("thinking", "text"):
            if ev.get("text", "").strip():
                pending.append({"type": k, "text": ev["text"].strip()})
        elif k == "tool_error":
            pending.append({"type": "refused", "tool": ev.get("tool"), "input": ev.get("input"), "error": ev.get("error")})
        elif k == "tool_crash":
            pending.append({"type": "system", "text": f"The canvas engine failed on a {ev.get('tool')} call: {ev.get('error')}"})
        elif k == "tool":
            tool = ev["tool"]
            if tool == "look":
                pending.append({"type": "look", "input": ev.get("input")})
            elif tool == "finish":
                finish = ev.get("input") or {}
            else:
                last_frame = ev.get("action", last_frame + 1)
                steps.append({"frame": last_frame, "kind": tool, "stroke": ev.get("input"), "notes": pending})
                pending = []
        elif k == "nudge":
            pending.append({"type": "system", "text": "It ended a turn without painting or finishing, and was told the canvas was still open."})
        elif k == "resumed":
            pending.append({"type": "system", "text": "The run stopped here and was resumed later."})
        elif k == "api_error":
            pending.append({"type": "system", "text": f"The run stopped on an API error: {ev.get('message')}"})
        elif k == "refusal":
            pending.append({"type": "system", "text": "The model declined to continue."})
        elif k == "ended_without_finish":
            pending.append({"type": "system", "text": "It stopped making strokes, even after being nudged, so the session ended."})
        elif k == "session_cap":
            pending.append({"type": "system", "text": "The session stopped here at its stroke cap, which the model wasn't told about. The painting can be continued with resume."})
    steps.append({"frame": last_frame, "kind": "end", "notes": pending, "finish": finish})
    return steps


def write_viewer(run_dir: Path, out: Path | None = None, max_side: int = 900) -> Path:
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines() if line.strip()]
    steps = build_steps(events)
    frames = sorted((run_dir / "frames").glob("*.png"), key=lambda f: int(f.stem))
    by_index = {int(f.stem): f for f in frames}
    used = sorted({s["frame"] for s in steps})
    data = {
        "meta": {
            "run": run_dir.name,
            "model": cfg.get("model"),
            "effort": cfg.get("effort"),
            "width": cfg.get("width"),
            "height": cfg.get("height"),
            "max_actions": cfg.get("max_actions"),
            "max_stroke_length": cfg.get("max_stroke_length"),
            "max_brush_size": cfg.get("max_brush_size"),
            "wobble": cfg.get("wobble", 0),
            "undo": cfg.get("allow_undo"),
            "system_prompt": cfg.get("system_prompt_rendered"),
            "finished": summary.get("finished"),
            "open_ended": summary.get("open_ended", False),
            "title": summary.get("title"),
            "note": summary.get("note"),
            "strokes": summary.get("actions"),
            "requests": summary.get("requests"),
            "cost": summary.get("estimated_cost_usd"),
            "coverage": summary.get("coverage"),
        },
        "frames": {str(i): _frame_uri(by_index[i], max_side) for i in used if i in by_index},
        "steps": steps,
    }
    blob = json.dumps(data).replace("</", "<\\/")
    title = data["meta"]["title"] or f"Painterly run {run_dir.name}"
    html = _TEMPLATE.read_text().replace("__TITLE__", _esc(title)).replace("__DATA__", blob)
    out = out or run_dir / "viewer.html"
    out.write_text(html, encoding="utf-8")
    return out


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
