"""Command line: `python -m painterly run` / `python -m painterly replay runs/<id>`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import RunConfig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="painterly", description="Give a Claude model a canvas and see what it does.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="start a painting session")
    r.add_argument("--model", default=RunConfig.model)
    r.add_argument("--effort", default=RunConfig.effort, choices=["low", "medium", "high", "xhigh", "max"])
    r.add_argument("--size", default=f"{RunConfig.width}x{RunConfig.height}", help="canvas WxH, e.g. 1024x768")
    r.add_argument("--paper", default=RunConfig.paper, help="paper colour (hex)")
    r.add_argument("--max-actions", type=int, default=RunConfig.max_actions,
                   help="safety cap on strokes (default 300); the model isn't told about it")
    r.add_argument("--keep-images", type=int, default=RunConfig.keep_images, help="recent canvas views kept in context")
    r.add_argument("--prompt", type=Path, help="file with a replacement system prompt ({width} {height} {max_actions} are filled in)")
    r.add_argument("--out", type=Path, default=RunConfig.out_dir)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--no-fallbacks", action="store_true", help="don't let the API retry a declined request on another model")
    r.add_argument("--runs", type=int, default=1, help="how many independent sessions to run, one after another")

    c = sub.add_parser("check", help="probe an endpoint for the API features the loop uses (tiny requests)")
    c.add_argument("--model", default="claude-haiku-5-5")
    c.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])

    for sp in (r, c):
        sp.add_argument("--base-url", help="API base URL, for a proxy or reseller (default: Anthropic)")
        sp.add_argument("--api-key-env", help="read the API key from this environment variable, e.g. COMET_API_KEY")

    r.add_argument("--inline-images", action="store_true", help="send canvas views as base64 (no Files API)")
    r.add_argument("--no-context-editing", action="store_true", help="don't ask the server to clear old canvas views")
    r.add_argument("--no-thinking-summaries", action="store_true", help="don't request thinking summaries")
    r.add_argument("--max-stroke-length", type=float, help="longest single stroke in px, along its path (default: no limit)")
    r.add_argument("--max-brush-size", type=float, help="widest brush in px (default 15%% of the longer side)")
    r.add_argument("--undo", action="store_true", help="give it an undo tool (off by default: every stroke is final)")
    r.add_argument("--wobble", type=float, default=RunConfig.wobble, help="natural hand wobble, 0 for perfectly geometric strokes (default 1)")
    r.add_argument("--no-look", action="store_true", help="no zooming in on the canvas")
    r.add_argument("--view-size", type=int, help="longest side of the image the model sees (default 1024; 768 with --inline-images)")

    rs = sub.add_parser("resume", help="carry on a run that stopped before the model finished")
    rs.add_argument("run_dir", type=Path)

    v = sub.add_parser("viewer", help="write a self-contained HTML page to step through a run's strokes")
    v.add_argument("run_dir", type=Path)
    v.add_argument("-o", "--output", type=Path, help="default: <run_dir>/viewer.html")

    p = sub.add_parser("replay", help="re-render a run from its event log")
    p.add_argument("run_dir", type=Path)
    p.add_argument("-o", "--output", type=Path)

    s = sub.add_parser("sampler", help="render a sheet showing every brush (no API calls)")
    s.add_argument("-o", "--output", type=Path, default=Path("brush_sampler.png"))

    a = ap.parse_args(argv)

    if a.cmd == "replay":
        from .replay import replay

        print(replay(a.run_dir, a.output))
        return 0

    if a.cmd == "resume":
        from .agent import PaintingSession

        session = PaintingSession.resume(a.run_dir)
        out = session.run(on_event=lambda s: print("  " + s, flush=True))
        print((out / "summary.json").read_text())
        return 0

    if a.cmd == "viewer":
        from .viewer import write_viewer

        print(write_viewer(a.run_dir, a.output))
        return 0

    if a.cmd == "sampler":
        from .sampler import render_sampler

        render_sampler(a.output)
        print(a.output)
        return 0

    if a.cmd == "check":
        from .check import run_checks

        res = run_checks(RunConfig(model=a.model, effort=a.effort, base_url=a.base_url, api_key_env=a.api_key_env))
        return 0 if res.get("basic request") else 1

    from .agent import PaintingSession

    w, h = (int(v) for v in a.size.lower().split("x"))
    for i in range(a.runs):
        cfg = RunConfig(
            model=a.model,
            effort=a.effort,
            width=w,
            height=h,
            paper=a.paper,
            max_actions=a.max_actions,
            keep_images=a.keep_images,
            out_dir=a.out,
            seed=a.seed + i,
            fallbacks=not a.no_fallbacks,
            base_url=a.base_url,
            api_key_env=a.api_key_env,
            inline_images=a.inline_images,
            context_editing=not a.no_context_editing,
            thinking_summaries=not a.no_thinking_summaries,
            max_stroke_length=a.max_stroke_length,
            max_brush_size=a.max_brush_size,
            allow_undo=a.undo,
            wobble=a.wobble,
            allow_look=not a.no_look,
            view_size=a.view_size or (768 if a.inline_images else 1024),
            **({"system_prompt": a.prompt.read_text()} if a.prompt else {}),
        )
        session = PaintingSession(cfg)
        print(f"[run {i + 1}/{a.runs}] → {session.dir}")
        out = session.run(on_event=lambda s: print("  " + s, flush=True))
        print((out / "summary.json").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
