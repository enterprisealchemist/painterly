"""The canvas and the brush engine.

Strokes are rendered as *sweeps*, not as rows of stamps: for every pixel near a
stroke we find the nearest point on its centreline and get four fields:

    nd  normalised distance from the centreline (0 = centre, 1 = edge)
    v   signed normalised offset across the stroke (-1 .. 1)
    t   position along the stroke (0 = start, 1 = end)
    s   distance along the stroke in pixels

Each brush is just a function from those fields to coverage. That keeps tapering,
paint depletion and bristle streaks consistent along the whole stroke. Each stroke
builds its own coverage layer and is composited once, so a stroke doesn't darken
where it crosses itself, but separate strokes layer on top of each other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

# --------------------------------------------------------------------------- utils


def hex_to_rgb(color: str) -> np.ndarray:
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in c):
        raise ValueError(f"color must be a hex string like '#1a2b3c', got {color!r}")
    return np.array([int(c[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    if e1 <= e0:
        return (x >= e1).astype(np.float32)
    y = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return y * y * (3 - 2 * y)


def _box_blur_1d(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    if r <= 0:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    p = np.pad(a, pad, mode="edge")
    c = np.cumsum(p, axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Approximate gaussian blur (three box passes) for 2D or HxWxC float arrays."""
    if sigma < 0.5:
        return a
    r = max(1, int(round(sigma * 0.85)))
    out = a.astype(np.float32)
    for _ in range(3):
        out = _box_blur_1d(out, r, 0)
        out = _box_blur_1d(out, r, 1)
    return out


def value_noise(h: int, w: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth noise in [0, 1] with features roughly `scale` pixels across."""
    gh, gw = max(2, int(h / scale) + 2), max(2, int(w / scale) + 2)
    g = Image.fromarray((rng.random((gh, gw)) * 255).astype(np.uint8), "L")
    g = g.resize((w, h), Image.BICUBIC)
    return np.asarray(g, dtype=np.float32) / 255.0


# --------------------------------------------------------------------------- brushes


@dataclass(frozen=True)
class BrushSpec:
    name: str
    description: str
    hardness: float = 0.6  # default edge hardness (0 soft .. 1 crisp)
    blend: str = "normal"  # default blend mode
    taper_in: float = 0.0
    taper_out: float = 0.0


# Each description says what the brush actually does; `painterly/sampler.py` draws them all.
# A stroke is `size` wide at full pressure (at half strength, for the soft-edged ones).
BRUSHES: dict[str, BrushSpec] = {
    b.name: b
    for b in [
        BrushSpec("round", "Round brush: even, opaque colour with a slightly soft edge.", 0.6),
        BrushSpec(
            "ink",
            "Pointed brush with ink: crisp, dark, fully opaque line. Good for calligraphic "
            "marks that swell and thin with pressure.",
            0.9,
        ),
        BrushSpec(
            "bristle",
            "Hog-hair brush: the paint lies in streaks along the stroke, so a little of what's "
            "underneath shows through. With little paint loaded or light pressure it breaks up "
            "into dry-brush scratches.",
            0.75,
        ),
        BrushSpec(
            "flat",
            "Flat brush held square to the direction of travel: firm edges, faint streaks, "
            "square ends. Good for blocking in areas.",
            0.85,
        ),
        BrushSpec(
            "knife",
            "Palette knife: an opaque, hard-edged slab of paint that skips over the paper's "
            "grain, more so with light pressure or little paint loaded.",
            0.95,
        ),
        BrushSpec(
            "watercolor",
            "Transparent wash: pools darker at its edges and granulates in the paper grain. "
            "It darkens what's underneath like a glaze instead of covering it, so a light "
            "colour barely shows over a dark one.",
            0.3,
            blend="multiply",
        ),
        BrushSpec(
            "airbrush",
            "Soft spray with no edge: solid along the centre, fading to nothing at about the "
            "stroke's size. Light pressure or dilution gives a faint haze.",
            0.0,
        ),
        BrushSpec(
            "charcoal",
            "Charcoal: a dry, grainy mark that catches the raised tooth of the paper; light "
            "pressure leaves more paper showing.",
            0.5,
        ),
        BrushSpec(
            "spatter",
            "Droplets flicked off the brush, landing within about `size` of the path. "
            "Pressure sets how many land; a single point gives a small burst.",
            0.8,
        ),
    ]
}

BRUSHES["blender"] = BrushSpec(
    "blender",
    "Clean, dry soft brush: lays down no paint, but blends and softens the paint it passes "
    "over. Colour is ignored.",
    0.2,
)

BLEND_MODES = ("normal", "multiply", "screen", "erase")


# --------------------------------------------------------------------------- path geometry


def _catmull_rom(points: np.ndarray, step: float) -> np.ndarray:
    """Dense centripetal-ish Catmull-Rom curve through `points`, sampled every ~`step` px."""
    if len(points) < 2:
        return points.copy()
    p = np.vstack([points[0] * 2 - points[1], points, points[-1] * 2 - points[-2]])
    out = [points[0]]
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        seg = np.linalg.norm(p2 - p1)
        n = max(1, int(math.ceil(seg / step)))
        ts = np.linspace(0, 1, n + 1)[1:, None]
        t2, t3 = ts * ts, ts * ts * ts
        pts = 0.5 * (
            2 * p1
            + (-p0 + p2) * ts
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )
        out.extend(pts)
    return np.asarray(out, dtype=np.float64)


def arc_path(start, end, curvature: float = 0.0, step: float = 2.0) -> tuple[np.ndarray, float]:
    """Points along a constant-curvature arc from `start` to `end`, and its length.

    `curvature` is the bulge as a fraction of half the chord: 0 is straight, +-1 a half
    circle. Positive bows to the left of the direction of travel as seen on the canvas
    (y points down), negative to the right.
    """
    a, b = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    chord = b - a
    c2 = float(np.linalg.norm(chord))
    if c2 < 1e-6:
        return a[None, :].copy(), 0.0
    e = chord / c2
    c = c2 / 2.0
    k = float(np.clip(curvature, -1.0, 1.0))
    if abs(k) < 1e-4:
        n = max(1, int(math.ceil(c2 / step)))
        return a + np.linspace(0, 1, n + 1)[:, None] * chord, c2
    left = np.array([e[1], -e[0]])  # left of travel on screen, with y pointing down
    nrm = left * math.copysign(1.0, k)
    sag = abs(k) * c
    R = (c * c + sag * sag) / (2 * sag)
    theta = math.atan2(c, R - sag)  # half the angle the arc subtends
    length = 2 * R * theta
    centre = (a + b) / 2 - nrm * (R - sag)
    n = max(2, int(math.ceil(length / step)))
    phi = np.linspace(-theta, theta, n + 1)[:, None]
    return centre + R * (np.cos(phi) * nrm + np.sin(phi) * e), length


def node_path(points, smooth: bool = True, closed: bool = False, step: float = 2.0) -> tuple[np.ndarray, float]:
    """The path a brush takes through `points` (in order), sampled every ~`step` px, and its length.

    smooth: a flowing curve through every node (Catmull-Rom); otherwise straight lines
    with sharp corners at the nodes. closed: the stroke carries on back to the first node.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 1 or np.allclose(pts, pts[0]):
        return pts[:1].copy(), 0.0
    if closed and len(pts) >= 3:
        if smooth:
            ring = np.vstack([pts[-1], pts, pts[0], pts[1]])  # wrap so the join is smooth too
            out = [pts[0]]
            for i in range(1, len(ring) - 2):
                out.extend(_cr_segment(ring[i - 1], ring[i], ring[i + 1], ring[i + 2], step))
            path = np.asarray(out)
        else:
            path = _polyline(np.vstack([pts, pts[0]]), step)
    elif smooth and len(pts) >= 3:
        path = _catmull_rom(pts, step)
    else:
        path = _polyline(pts, step)
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    return path, length


def _cr_segment(p0, p1, p2, p3, step):
    n = max(1, int(math.ceil(np.linalg.norm(p2 - p1) / step)))
    ts = np.linspace(0, 1, n + 1)[1:, None]
    t2, t3 = ts * ts, ts * ts * ts
    return 0.5 * (2 * p1 + (-p0 + p2) * ts + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def _polyline(pts: np.ndarray, step: float) -> np.ndarray:
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        out.extend(a + (b - a) * np.linspace(0, 1, n + 1)[1:, None])
    return np.asarray(out, dtype=np.float64)


def hand_wobble(path: np.ndarray, length: float, rng: np.random.Generator, amount: float = 1.0) -> np.ndarray:
    """Displace a path sideways by a smooth, low-frequency drift, like an unsteady hand.

    Up to ~1.2% of the stroke's length (at most 5 px at amount=1), made of two slow sine
    waves with random phase. It is zero at both ends, so the stroke still starts and ends
    where it was asked to.
    """
    if amount <= 0 or len(path) < 3 or length < 4:
        return path
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)]) / max(seg.sum(), 1e-9)
    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    f1, f2 = rng.uniform(0.4, 1.3), rng.uniform(1.8, 3.5)
    p1, p2 = rng.uniform(0, 2 * np.pi, 2)
    wave = 0.65 * np.sin(2 * np.pi * f1 * t + p1) + 0.35 * np.sin(2 * np.pi * f2 * t + p2)
    envelope = np.clip(np.sin(np.pi * t), 0.0, 1.0) ** 0.7
    amp = amount * min(0.012 * length, 5.0)
    return path + normal * (amp * wave * envelope)[:, None]


def arc_length(start, end, curvature: float = 0.0) -> float:
    c2 = math.dist(start, end)
    k = abs(float(np.clip(curvature, -1.0, 1.0)))
    if c2 < 1e-6 or k < 1e-4:
        return c2
    c, sag = c2 / 2, k * c2 / 2
    R = (c * c + sag * sag) / (2 * sag)
    return 2 * R * math.atan2(c, R - sag)


def pressure_at(samples, t: np.ndarray) -> np.ndarray:
    """Pressure along the stroke: `samples` are evenly spaced from start (t=0) to end
    (t=1), joined by smooth (cosine) easing, so there is no overshoot between them."""
    p = np.clip(np.asarray(samples, dtype=np.float32), 0.0, 1.0)
    if len(p) == 1:
        return np.full_like(t, p[0], dtype=np.float32)
    x = np.clip(t, 0.0, 1.0) * (len(p) - 1)
    i = np.minimum(x.astype(np.int32), len(p) - 2)
    f = x - i
    f = 0.5 - 0.5 * np.cos(np.pi * f)
    return (p[i] * (1 - f) + p[i + 1] * f).astype(np.float32)


def _taper(t: np.ndarray, tin: float, tout: float) -> np.ndarray:
    w = np.ones_like(t)
    if tin > 0:
        w *= 0.08 + 0.92 * smoothstep(0.0, tin, t)
    if tout > 0:
        w *= 0.08 + 0.92 * smoothstep(0.0, tout, 1.0 - t)
    return w


# --------------------------------------------------------------------------- canvas


class Canvas:
    def __init__(self, width: int = 1024, height: int = 1024, paper: str = "#f4f1ea", seed: int = 0):
        self.w, self.h = width, height
        self.rng = np.random.default_rng(seed)
        self.paper = hex_to_rgb(paper)
        self.rgb = np.empty((height, width, 3), dtype=np.float32)
        self.rgb[:] = self.paper
        # Paper tooth, shared by every grain-sensitive brush so textures line up.
        g = 0.55 * value_noise(height, width, 2.0, self.rng) + 0.45 * value_noise(height, width, 7.0, self.rng)
        self.grain = (g - g.min()) / max(1e-6, g.max() - g.min())
        # Low-frequency pigment mottling for washes.
        m = 0.6 * value_noise(height, width, 22.0, self.rng) + 0.4 * value_noise(height, width, 6.0, self.rng)
        self.mottle = (m - m.min()) / max(1e-6, m.max() - m.min())
        self._history: list[np.ndarray] = []
        self.max_undo = 30

    # ---------------------------------------------------------------- history
    def checkpoint(self) -> None:
        self._history.append(self.rgb.copy())
        if len(self._history) > self.max_undo:
            self._history.pop(0)

    def undo(self, steps: int = 1) -> int:
        done = 0
        while done < steps and self._history:
            self.rgb = self._history.pop()
            done += 1
        return done

    # ---------------------------------------------------------------- compositing
    def _composite(self, alpha: np.ndarray, color: np.ndarray, x0: int, y0: int, blend: str) -> None:
        h, w = alpha.shape
        region = self.rgb[y0 : y0 + h, x0 : x0 + w]
        a = np.clip(alpha, 0, 1)[..., None]
        if blend == "normal":
            region[:] = region * (1 - a) + color * a
        elif blend == "multiply":
            region[:] = region * (1 - a + a * color)
        elif blend == "screen":
            region[:] = region + a * (color - region * color)
        elif blend == "erase":
            region[:] = region * (1 - a) + self.paper * a
        else:
            raise ValueError(f"unknown blend {blend!r}")
        np.clip(region, 0, 1, out=region)

    # ---------------------------------------------------------------- strokes
    def stroke(
        self,
        points,
        brush: str = "round",
        color: str = "#000000",
        size: float = 12.0,
        opacity: float = 1.0,
        hardness: float | None = None,
        taper_in: float | None = None,
        taper_out: float | None = None,
        fade: float = 0.0,
        jitter: float = 0.0,
        blend: str | None = None,
        seed: int | None = None,
        pressure=None,
        dense: bool = False,
    ) -> None:
        """`pressure`: samples (0-1) evenly spaced along the stroke; scales width and how much
        paint is laid down. `dense`: `points` is already a finished path (e.g. from
        `arc_path`) and is not smoothed again."""
        spec = BRUSHES[brush]
        hardness = spec.hardness if hardness is None else float(np.clip(hardness, 0, 1))
        tin = spec.taper_in if taper_in is None else float(np.clip(taper_in, 0, 1))
        tout = spec.taper_out if taper_out is None else float(np.clip(taper_out, 0, 1))
        blend = blend or spec.blend
        rgb = hex_to_rgb(color)
        radius = max(0.5, float(size) / 2.0)
        rng = np.random.default_rng(seed if seed is not None else self.rng.integers(1 << 31))

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if jitter > 0 and len(pts) >= 2 and brush != "spatter":
            # Hand wobble: low-frequency displacement perpendicular-ish to the path.
            pts = pts + rng.normal(0, jitter * radius * 0.6, pts.shape)

        if brush == "spatter":
            alpha, x0, y0 = self._spatter(pts, radius, hardness, jitter, fade, rng, pressure, dense)
        else:
            alpha, x0, y0 = self._sweep(pts, brush, radius, hardness, tin, tout, fade, rng, pressure, dense)
        if alpha is None:
            return
        alpha = alpha * float(np.clip(opacity, 0, 1))
        if brush == "blender":
            # A clean dry brush: pulls neighbouring paint together where it passes.
            h, w = alpha.shape
            region = self.rgb[y0 : y0 + h, x0 : x0 + w]
            soft = blur(region, max(1.0, radius * 0.7))
            a = np.clip(alpha, 0, 1)[..., None]
            region[:] = region * (1 - a) + soft * a
            return
        self._composite(alpha, rgb, x0, y0, blend)

    def _sweep(self, pts, brush, radius, hardness, tin, tout, fade, rng, pressure=None, dense=False):
        step = float(np.clip(radius * 0.35, 1.5, 10.0))
        path = _catmull_rom(pts, step) if len(pts) >= 2 and not dense else pts
        if len(path) == 1:
            path = np.vstack([path, path + [0.01, 0.0]])
        seglen = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seglen)])
        total = max(cum[-1], 1e-6)
        tpath = cum / total
        wpath = radius * _taper(tpath, tin, tout)
        if pressure is not None:
            wpath = wpath * (0.06 + 0.94 * pressure_at(pressure, tpath))

        pad = int(radius * 1.6) + 3
        x0 = max(0, int(math.floor(path[:, 0].min())) - pad)
        y0 = max(0, int(math.floor(path[:, 1].min())) - pad)
        x1 = min(self.w, int(math.ceil(path[:, 0].max())) + pad)
        y1 = min(self.h, int(math.ceil(path[:, 1].max())) + pad)
        if x1 <= x0 or y1 <= y0:
            return None, 0, 0
        H, W = y1 - y0, x1 - x0
        nd = np.full((H, W), np.inf, dtype=np.float32)
        V = np.zeros((H, W), dtype=np.float32)
        T = np.zeros((H, W), dtype=np.float32)
        S = np.zeros((H, W), dtype=np.float32)

        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            wmax = max(wpath[i], wpath[i + 1]) * 1.6 + 2
            bx0 = max(x0, int(min(a[0], b[0]) - wmax))
            bx1 = min(x1, int(max(a[0], b[0]) + wmax) + 1)
            by0 = max(y0, int(min(a[1], b[1]) - wmax))
            by1 = min(y1, int(max(a[1], b[1]) + wmax) + 1)
            if bx1 <= bx0 or by1 <= by0:
                continue
            ys, xs = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
            xs += 0.5
            ys += 0.5
            d = b - a
            L2 = float(d @ d)
            if L2 < 1e-9:
                u = np.zeros_like(xs)
            else:
                u = np.clip(((xs - a[0]) * d[0] + (ys - a[1]) * d[1]) / L2, 0, 1)
            px, py = a[0] + u * d[0], a[1] + u * d[1]
            dist = np.hypot(xs - px, ys - py)
            wl = np.maximum(wpath[i] + u * (wpath[i + 1] - wpath[i]), 0.35)
            n = dist / wl
            cross = d[0] * (ys - a[1]) - d[1] * (xs - a[0])
            sl = (slice(by0 - y0, by1 - y0), slice(bx0 - x0, bx1 - x0))
            better = n < nd[sl]
            nd[sl] = np.where(better, n, nd[sl])
            lat = cross / (math.sqrt(L2) * wl) if L2 > 1e-9 else np.zeros_like(n)
            V[sl] = np.where(better, np.clip(lat, -1, 1), V[sl])
            T[sl] = np.where(better, tpath[i] + u * (tpath[i + 1] - tpath[i]), T[sl])
            S[sl] = np.where(better, cum[i] + u * seglen[i], S[sl])

        inside = np.isfinite(nd)
        nd = np.where(inside, nd, 9.0)
        grain = self.grain[y0:y1, x0:x1]
        load = np.clip(1.0 - fade * T, 0.0, 1.0)  # paint remaining along the stroke
        dryness = fade * 1.1
        if pressure is not None:
            # light pressure lays down less paint and lets a bristle brush skip
            P = pressure_at(pressure, T)
            load = load * (0.5 + 0.5 * P)
            dryness = dryness + 0.5 * (1.0 - P)
        # Soft edges are centred on the stroke's nominal edge (nd = 1), so a stroke is `size`
        # wide at half strength whatever its hardness: softness blurs the edge, it doesn't shrink it.
        edge = lambda h: 1.0 - smoothstep(1.0 - max(0.03, (1.0 - h) * 0.45), 1.0 + max(0.03, (1.0 - h) * 0.45), nd)  # noqa: E731

        if brush in ("round", "ink", "blender"):
            alpha = edge(hardness)
            if brush == "ink":
                alpha *= 0.92 + 0.08 * grain
            alpha *= 0.35 + 0.65 * load if (fade > 0 or pressure is not None) else 1.0
        elif brush == "airbrush":
            # soft spray, no hard edge; half strength at ~0.6 of the size, fading out by ~0.9
            alpha = 1.0 - smoothstep(0.15, 1.1, nd)
            alpha *= load
        elif brush in ("bristle", "flat"):
            # One lane per bristle across the stroke; each lane has its own load and
            # runs out at its own point, so dry-brush breaks up into streaks, not dots.
            nb = int(np.clip(radius * (1.1 if brush == "bristle" else 0.5), 6, 140))
            lane_f = np.clip((V + 1) * 0.5, 0, 1) * (nb - 1)
            lane = np.clip(np.rint(lane_f).astype(np.int32), 0, nb - 1)
            ink = 0.55 + 0.45 * rng.random(nb).astype(np.float32)
            if brush == "flat":
                ink = 0.85 + 0.15 * ink
            thresh = rng.random(nb).astype(np.float32)
            # streak value interpolated between lanes so the edges are soft, not aliased
            lo = np.floor(lane_f).astype(np.int32)
            hi = np.minimum(lo + 1, nb - 1)
            fr = lane_f - lo
            streak = ink[lo] * (1 - fr) + ink[hi] * fr
            # skipping along each lane: 1D noise in arc length, offset per lane
            period = max(3.0, radius * 0.6)
            noise_len = int(S.max() / period) + 3
            lane_noise = blur(rng.random((nb, noise_len)).astype(np.float32), 0.0)
            sp = np.clip(S / period, 0, noise_len - 1.001)
            i0 = sp.astype(np.int32)
            f = sp - i0
            along = lane_noise[lane, i0] * (1 - f) + lane_noise[lane, i0 + 1] * f
            dry = smoothstep(-0.08, 0.08, load + 0.3 * (along - 0.5) - thresh[lane] * dryness)
            body = edge(hardness)
            if brush == "flat" and total > 1.0 and not np.allclose(path[0], path[-1]):  # dabs and closed loops have no ends
                body = body * ((T > 0.0) & (T < 1.0))  # square ends
            alpha = body * streak * dry
        elif brush == "knife":
            body = edge(hardness)
            skip = smoothstep(0.0, 0.18, grain - (0.25 + 0.6 * (1 - load)) + 0.25)
            alpha = body * skip
        elif brush == "charcoal":
            body = edge(hardness)
            alpha = smoothstep(0.0, 0.35, body * (0.55 + 0.45 * load) - (1.0 - grain) * 0.75 + 0.1)
        elif brush == "watercolor":
            body = edge(hardness) * (0.55 + 0.45 * load)
            body = blur(body, max(1.0, radius * 0.08))
            rim = np.clip(body - blur(body, max(1.5, radius * 0.25)), 0, 1)
            alpha = body * 0.6 + rim * 2.2
            mott = self.mottle[y0:y1, x0:x1]
            alpha *= (0.75 + 0.5 * (mott - 0.5)) * (0.96 + 0.08 * blur(grain, 1.2) - 0.04)
            alpha = np.clip(alpha, 0, 1)
        else:
            raise ValueError(f"unknown brush {brush!r}")

        alpha = np.where(inside, alpha, 0.0).astype(np.float32)
        return alpha, x0, y0

    def _spatter(self, pts, radius, hardness, jitter, fade, rng, pressure=None, dense=False):
        if len(pts) >= 2 and not dense:
            path = _catmull_rom(pts, 2.0)
        else:
            path = pts
        seglen = np.linalg.norm(np.diff(path, axis=0), axis=1) if len(path) > 1 else np.array([0.0])
        length = float(seglen.sum())
        count = int((20 + length / 3.0) * (0.3 + 2.0 * max(jitter, 0.15)))
        count = min(count, 4000)
        pad = int(radius * 2.5) + 4
        x0 = max(0, int(path[:, 0].min()) - pad)
        y0 = max(0, int(path[:, 1].min()) - pad)
        x1 = min(self.w, int(path[:, 0].max()) + pad)
        y1 = min(self.h, int(path[:, 1].max()) + pad)
        if x1 <= x0 or y1 <= y0:
            return None, 0, 0
        ss = 2  # supersample for antialiasing
        img = Image.new("L", ((x1 - x0) * ss, (y1 - y0) * ss), 0)
        dr = ImageDraw.Draw(img)
        idx = rng.integers(0, len(path), count)
        for k, i in enumerate(idx):
            t = i / max(1, len(path) - 1)
            if rng.random() < fade * t:
                continue
            if pressure is not None and rng.random() > float(pressure_at(pressure, np.array([t]))[0]):
                continue
            off = rng.normal(0, radius * 0.6, 2)
            cx, cy = path[i] + off
            r = abs(rng.lognormal(math.log(max(0.6, radius * 0.035)), 0.6))
            r = min(r, radius * 0.22)
            ex = 1.0 + (1.0 - hardness) * rng.random() * 1.2
            ang = rng.random() * math.pi
            # elongated droplets when hardness is low
            dx, dy = math.cos(ang) * r * ex, math.sin(ang) * r * ex
            bbox = [
                (cx - x0 - abs(dx) - r) * ss,
                (cy - y0 - abs(dy) - r) * ss,
                (cx - x0 + abs(dx) + r) * ss,
                (cy - y0 + abs(dy) + r) * ss,
            ]
            dr.ellipse(bbox, fill=int(200 + 55 * rng.random()))
        img = img.resize((x1 - x0, y1 - y0), Image.LANCZOS)
        return np.asarray(img, dtype=np.float32) / 255.0, x0, y0

    # ---------------------------------------------------------------- fills
    def fill(
        self,
        color: str,
        points=None,
        color_to: str | None = None,
        angle: float = 90.0,
        opacity: float = 1.0,
        softness: float = 0.0,
        texture: str = "none",
        blend: str = "normal",
    ) -> None:
        """Fill a polygon (or the whole canvas) with a flat colour or a linear gradient."""
        if points is None or len(points) < 3:
            mask = np.ones((self.h, self.w), dtype=np.float32)
        else:
            ss = 2
            img = Image.new("L", (self.w * ss, self.h * ss), 0)
            ImageDraw.Draw(img).polygon([(float(x) * ss, float(y) * ss) for x, y in points], fill=255)
            img = img.resize((self.w, self.h), Image.LANCZOS)
            mask = np.asarray(img, dtype=np.float32) / 255.0
        if softness > 0:
            mask = blur(mask, softness)

        if texture == "watercolor":
            mask = blur(mask, 1.5)
            rim = np.clip(mask - blur(mask, 8.0), 0, 1)
            mask = np.clip(mask * 0.65 + rim * 1.8, 0, 1)
            mask *= (0.75 + 0.5 * (self.mottle - 0.5)) * (0.96 + 0.08 * blur(self.grain, 1.2) - 0.04)
        elif texture == "charcoal":
            mask = smoothstep(0.0, 0.35, mask - (1.0 - self.grain) * 0.75 + 0.1)
        elif texture == "canvas":
            mask = mask * (0.85 + 0.3 * (self.grain - 0.5))
        elif texture != "none":
            raise ValueError(f"unknown texture {texture!r}")

        ys, xs = np.nonzero(mask > 1e-3)
        if len(xs) == 0:
            return
        bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        m = mask[by0:by1, bx0:bx1] * float(np.clip(opacity, 0, 1))
        c0 = hex_to_rgb(color)
        if color_to:
            c1 = hex_to_rgb(color_to)
            # angle in degrees, screen convention: 0 = left->right, 90 = top->bottom
            th = math.radians(angle)
            yy, xx = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
            proj = xx * math.cos(th) + yy * math.sin(th)
            g = (proj - proj.min()) / max(1e-6, proj.max() - proj.min())
            col = c0 * (1 - g[..., None]) + c1 * g[..., None]
            region = self.rgb[by0:by1, bx0:bx1]
            a = m[..., None]
            if blend == "normal":
                region[:] = region * (1 - a) + col * a
            elif blend == "multiply":
                region[:] = region * (1 - a + a * col)
            elif blend == "screen":
                region[:] = region + a * (col - region * col)
            elif blend == "erase":
                region[:] = region * (1 - a) + self.paper * a
            np.clip(region, 0, 1, out=region)
        else:
            self._composite(m, c0, int(bx0), int(by0), blend)

    # ---------------------------------------------------------------- whole-canvas
    def soften(self, region=None, amount: float = 2.0) -> None:
        """Blend wet paint together: blur a rectangle (or the whole canvas)."""
        x0, y0, x1, y1 = (0, 0, self.w, self.h) if region is None else self._clip_rect(region)
        if x1 <= x0 or y1 <= y0:
            return
        self.rgb[y0:y1, x0:x1] = blur(self.rgb[y0:y1, x0:x1], float(amount))

    def _clip_rect(self, r):
        x, y, w, h = (int(round(v)) for v in r)
        return max(0, x), max(0, y), min(self.w, x + w), min(self.h, y + h)

    # ---------------------------------------------------------------- io
    def image(self) -> Image.Image:
        return Image.fromarray((np.clip(self.rgb, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")

    def crop(self, x: int, y: int, w: int, h: int) -> Image.Image:
        x0, y0, x1, y1 = self._clip_rect((x, y, w, h))
        return self.image().crop((x0, y0, x1, y1))
