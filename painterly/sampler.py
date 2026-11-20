"""A reference sheet showing every brush, with and without paint depletion (`fade`)."""

from __future__ import annotations

import math
from pathlib import Path

from .canvas import BRUSHES, Canvas


def render_sampler(out: Path) -> None:
    c = Canvas(1024, 1024, seed=1)
    for i, name in enumerate(BRUSHES):
        y = 60 + i * 105
        pts = [[60 + k * 45, y + 28 * math.sin(k * 0.7)] for k in range(21)]
        for j, (col, fade) in enumerate([("#1d3557", 0.0), ("#c1121f", 0.9)]):
            c.stroke(
                [[x, yy + j * 34] for x, yy in pts],
                brush=name,
                color=col,
                size=60 if name == "spatter" else 34,
                opacity=0.95,
                fade=fade,
                jitter=0.3 if name == "spatter" else 0.0,
                seed=i * 2 + j,
            )
    c.image().save(out)
