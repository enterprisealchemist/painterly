"""A reference sheet showing every brush, drawn with the same `brush` tool the model uses.

Each row is one brush. Left: straight, pressing in and lifting off. Middle: an arc at
steady pressure. Right: an arc the other way with fading pressure and little paint
loaded, so it runs dry. The blender row paints three colours, then blends across them.
"""

from __future__ import annotations

from pathlib import Path

from .canvas import BRUSHES, Canvas
from .tools import Limits, apply_tool

VARIANTS = [
    dict(curvature=0.0, pressure=[0.1, 1, 1, 0.1], color="#1d3557"),
    dict(curvature=0.45, pressure=[0.8], color="#c1121f"),
    dict(curvature=-0.45, pressure=[1, 0.7, 0.3], color="#2d6a4f", load=0.35),
]


def render_sampler(out: Path) -> None:
    names = list(BRUSHES)
    c = Canvas(1024, 110 + 100 * len(names), seed=1)
    seed = 0
    limits = Limits(410, 154, wobble=1.0)  # what a run uses on a 1024 canvas

    def brush(**kw):
        nonlocal seed
        seed += 1
        apply_tool(c, "brush", kw, seed, limits)

    for i, name in enumerate(names):
        y = 95 + i * 100
        if name == "blender":
            for k, col in enumerate(["#1d3557", "#e9c46a", "#c1121f"]):
                for x in range(60 + k * 300, 60 + k * 300 + 300, 60):
                    brush(brush="flat", start=[x + 30, y - 35], end=[x + 30, y + 35], pressure=[1], color=col, size=60)
            for x in (60, 360, 660):
                brush(brush="blender", start=[x + 150, y], end=[x + 450 if x < 660 else x + 300, y],
                      pressure=[0.2, 1, 1, 0.2], color="#000000", size=60)
            continue
        for j, v in enumerate(VARIANTS):
            x = 50 + j * 330
            size = 50 if name == "spatter" else 40
            brush(brush=name, start=[x, y], end=[x + 280, y], size=size, **v)
    c.image().save(out)
