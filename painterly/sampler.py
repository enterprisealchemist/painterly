"""A reference sheet showing every brush, drawn with the same `brush` tool the model uses.

Each row is one brush. From left to right: a straight stroke pressing in and lifting off;
a smooth wave through four points at steady pressure; a square (closed, sharp corners);
and a ring (closed, smooth) with little paint loaded, so it runs dry. The blender row
paints three colours, then blends across them.
"""

from __future__ import annotations

from pathlib import Path

from .canvas import BRUSHES, Canvas
from .tools import Limits, apply_tool


def _variants(x: float, y: float) -> list[dict]:
    return [
        dict(points=[[x, y], [x + 190, y]], pressure=[0.1, 1, 1, 0.1], color="#1d3557"),
        dict(points=[[x + 250, y + 20], [x + 310, y - 25], [x + 380, y + 25], [x + 440, y - 20]],
             pressure=[0.8], color="#c1121f"),
        dict(points=[[x + 520, y - 30], [x + 580, y - 30], [x + 580, y + 30], [x + 520, y + 30]],
             smooth=False, closed=True, pressure=[0.9], color="#e09f3e"),
        dict(points=[[x + 720, y], [x + 690, y + 30], [x + 660, y], [x + 690, y - 30]],
             closed=True, pressure=[1, 0.6], color="#2d6a4f", load=0.35),
    ]


def render_sampler(out: Path) -> None:
    names = list(BRUSHES)
    c = Canvas(1024, 110 + 100 * len(names), seed=1, mixing="pigment")
    seed = 0
    limits = Limits(None, 154, wobble=1.0)  # what a run uses on a 1024 canvas

    def brush(**kw):
        nonlocal seed
        seed += 1
        apply_tool(c, "brush", kw, seed, limits)

    for i, name in enumerate(names):
        y = 95 + i * 100
        if name == "blender":
            for k, col in enumerate(["#1d3557", "#e9c46a", "#c1121f"]):
                for x in range(60 + k * 300, 60 + k * 300 + 300, 60):
                    brush(brush="flat", points=[[x + 30, y - 35], [x + 30, y + 35]], pressure=[1], color=col, size=60)
            brush(brush="blender", points=[[150, y], [500, y + 10], [930, y]], pressure=[0.2, 1, 1, 0.2],
                  color="#000000", size=60)
            continue
        size = 50 if name == "spatter" else 30
        for v in _variants(60, y):
            brush(brush=name, size=size, **v)
    c.image().save(out)
