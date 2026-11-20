"""Tool definitions the agent sees, and the dispatcher that applies them to a Canvas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .canvas import BLEND_MODES, BRUSHES, Canvas, arc_path, hand_wobble, hex_to_rgb, node_path

_brush_list = "\n".join(f"- {b.name}: {b.description}" for b in BRUSHES.values())

POINT = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}


@dataclass(frozen=True)
class Limits:
    """Physical limits on a single stroke, in canvas pixels."""

    max_stroke_length: float | None  # None: no limit
    max_brush_size: float
    wobble: float = 0.0  # natural hand wobble (see canvas.hand_wobble); runs set 1.0 by default

    @classmethod
    def for_canvas(cls, width: int, height: int, stroke_frac: float | None = None, brush_frac: float = 0.15) -> "Limits":
        side = max(width, height)
        return cls(round(side * stroke_frac) if stroke_frac else None, round(side * brush_frac))


def tool_definitions(width: int, height: int, limits: Limits | None = None, undo: bool = True, look: bool = True) -> list[dict[str, Any]]:
    lim = limits or Limits.for_canvas(width, height)
    coord = f"Canvas pixels: x 0..{width} left to right, y 0..{height} top to bottom."
    tools: list[dict[str, Any]] = [
        {
            "name": "brush",
            "description": (
                "Make a brushstroke: put the brush down at the first point, move it through "
                "each point in order in one continuous motion, and lift it at the last. Up to "
                "7 points. Through them the stroke can flow as a smooth curve or turn sharp "
                "corners, and it can close back on itself. A single point makes a dab. Like a "
                "real hand, the brush wavers slightly from the exact path, by a few pixels at "
                "most, and always starts and ends exactly on the first and last points."
                + (f" A stroke can be at most {lim.max_stroke_length:g} px long; for anything "
                   "longer, use several strokes." if lim.max_stroke_length else "")
                + f" {coord}\n\nBrushes:\n{_brush_list}"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "brush": {"type": "string", "enum": list(BRUSHES)},
                    "points": {"type": "array", "items": POINT, "minItems": 1, "maxItems": 7,
                               "description": "[x, y] points the brush passes through, in order."},
                    "smooth": {"type": "boolean", "description": (
                        "true (default): a flowing curve through the points. false: straight lines "
                        "between them, with sharp corners.")},
                    "closed": {"type": "boolean", "description": (
                        "true: after the last point, carry on back to the first, closing the "
                        "stroke into a loop. Default false.")},
                    "pressure": {"type": "array", "items": {"type": "number"}, "minItems": 1, "maxItems": 12,
                                 "description": (
                        "Pressure along the stroke, 0-1, as values spaced evenly from start to end "
                        "and eased smoothly between, e.g. [0.2, 1, 1, 0.3] to press in and lift "
                        "off, or [0.6] for steady pressure. Pressure sets the stroke's width "
                        "(size is the width at full pressure) and how much paint it lays down.")},
                    "color": {"type": "string", "description": "Hex colour, e.g. '#2b4a6f'."},
                    "size": {"type": "number", "description": (
                        f"Width of the stroke at full pressure, in pixels (1-{lim.max_brush_size:g}). "
                        "Lighter pressure narrows it in proportion: at pressure 0.5 it is about half as wide.")},
                    "load": {"type": "number", "description": (
                        "How much paint is on the brush, 0-1. Default 1. With less, the stroke "
                        "runs dry toward its end.")},
                    "wet": {"type": "number", "description": (
                        "0-1. Default 0: the brush lays down its own colour cleanly. Higher, and "
                        "the brush is wet enough to pick up paint already on the canvas and drag "
                        "it along the stroke, mixing it with its own colour (at 1, mostly what it "
                        "picks up). Over bare paper there's nothing to pick up.")},
                    "angle": {"type": "number", "description": (
                        "For flat, ink and knife only. Degrees at which the tip's broad edge is "
                        "held for the whole stroke: 0 horizontal, 90 vertical, 45 running from "
                        "top-left to bottom-right. The mark is full width where the stroke moves "
                        "across the edge and thin where it moves along it, like a calligraphy "
                        "nib. Omit it to keep the tip square to the direction of travel.")},
                    "dilution": {"type": "number", "description": (
                        "How thinned the paint is, 0-0.95. Default 0 (straight from the tube); "
                        "higher is more transparent.")},
                },
                "required": ["brush", "points", "pressure", "color", "size"],
                "additionalProperties": False,
            },
            "eager_input_streaming": True,
        },
    ]
    if look:
        tools.append({
            "name": "look",
            "description": "Lean in and look closely at part of the canvas: returns that rectangle enlarged. Does not change the canvas.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "width": {"type": "number"},
                    "height": {"type": "number"},
                },
                "required": ["x", "y", "width", "height"],
                "additionalProperties": False,
            },
            "eager_input_streaming": True,
        })
    if undo:
        tools.append({
            "name": "undo",
            "description": "Take back your most recent strokes.",
            "input_schema": {
                "type": "object",
                "properties": {"steps": {"type": "integer", "description": "How many strokes to undo. Default 1."}},
                "additionalProperties": False,
            },
            "eager_input_streaming": True,
        })
    tools.append({
        "name": "finish",
        "description": "Call this when you feel the painting is finished. Title and note are optional.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "note": {"type": "string", "description": "Anything you want to say about it, or nothing."},
            },
            "additionalProperties": False,
        },
        "eager_input_streaming": True,
    })
    return tools


# Tools that change the canvas. stroke/fill/soften are no longer offered to the model;
# they stay here so runs made with them can still be replayed.
MARKING_TOOLS = {"brush", "stroke", "fill", "soften", "undo"}


class ToolError(Exception):
    pass


def _num(d: dict, key: str, lo: float, hi: float, default=None):
    v = d.get(key, default)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ToolError(f"`{key}` must be a number")
    return float(min(hi, max(lo, v)))


def _color(d: dict, key: str, required: bool = True):
    v = d.get(key)
    if v is None:
        if required:
            raise ToolError(f"`{key}` is required")
        return None
    try:
        hex_to_rgb(str(v))
    except ValueError as e:
        raise ToolError(str(e)) from None
    return str(v)


def _points(d: dict, key: str, min_n: int, required: bool = True):
    pts = d.get(key)
    if pts is None:
        if required:
            raise ToolError(f"`{key}` is required")
        return None
    if not isinstance(pts, list) or len(pts) < min_n:
        raise ToolError(f"`{key}` must be a list of at least {min_n} [x, y] pairs")
    out = []
    for p in pts:
        if not (isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(c, (int, float)) for c in p)):
            raise ToolError(f"every entry of `{key}` must be [x, y]")
        out.append([float(p[0]), float(p[1])])
    return out


def _enum(d: dict, key: str, allowed, default=None):
    v = d.get(key, default)
    if v is not None and v not in allowed:
        raise ToolError(f"`{key}` must be one of {list(allowed)}")
    return v


def _point(d: dict, key: str, w: int, h: int):
    v = d.get(key)
    if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in v)):
        raise ToolError(f"`{key}` must be [x, y]")
    x, y = float(v[0]), float(v[1])
    if not (-0.1 * w <= x <= 1.1 * w and -0.1 * h <= y <= 1.1 * h):
        raise ToolError(f"`{key}` {[x, y]} is off the canvas ({w} by {h})")
    return [x, y]


def apply_tool(canvas: Canvas, name: str, args: dict[str, Any], rng_seed: int, limits: Limits | None = None) -> dict[str, Any]:
    """Apply one tool call. Returns a dict describing what happened (for the log).

    Raises ToolError for bad input. The returned dict always has a 'summary' string;
    `look` additionally returns a PIL image under 'image'.
    """
    if not isinstance(args, dict):
        raise ToolError("input must be an object")

    if name == "brush":
        lim = limits or Limits.for_canvas(canvas.w, canvas.h)
        brush = _enum(args, "brush", BRUSHES)
        if brush is None:
            raise ToolError("`brush` is required")
        legacy = "points" not in args and "start" in args  # runs from before multi-point strokes
        if legacy:
            nodes = [_point(args, "start", canvas.w, canvas.h), _point(args, "end", canvas.w, canvas.h)]
            curvature = _num(args, "curvature", -1, 1, 0.0)
        else:
            raw = args.get("points")
            if not isinstance(raw, list) or not 1 <= len(raw) <= 7:
                raise ToolError("`points` must be a list of 1 to 7 [x, y] points")
            nodes = [_point({"point": q}, "point", canvas.w, canvas.h) for q in raw]
            smooth = args.get("smooth", True)
            closed = args.get("closed", False)
            if not isinstance(smooth, bool) or not isinstance(closed, bool):
                raise ToolError("`smooth` and `closed` must be true or false")
        pressure = args.get("pressure")
        if not (isinstance(pressure, list) and 1 <= len(pressure) <= 12
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in pressure)):
            raise ToolError("`pressure` must be a list of 1 to 12 numbers between 0 and 1")
        pressure = [float(min(1, max(0, v))) for v in pressure]
        color = _color(args, "color")
        size = _num(args, "size", 1, 10_000)
        if size is None:
            raise ToolError("`size` is required")
        if size > lim.max_brush_size:
            raise ToolError(f"size {size:g} is wider than the widest brush ({lim.max_brush_size:g} px)")
        step = float(min(10.0, max(1.5, size * 0.175)))
        if legacy:
            path, length = arc_path(nodes[0], nodes[1], curvature, step)
        else:
            path, length = node_path(nodes, smooth, closed, step)
        if lim.max_stroke_length and length > lim.max_stroke_length + 0.5:
            raise ToolError(
                f"that stroke would be {length:.0f} px long; one stroke can be at most "
                f"{lim.max_stroke_length:g} px. Nothing was painted. Use several shorter strokes.")
        load = _num(args, "load", 0, 1, 1.0)
        dilution = _num(args, "dilution", 0, 0.95, 0.0)
        wet = _num(args, "wet", 0, 1, 0.0)
        angle = _num(args, "angle", -3600, 3600)
        if lim.wobble > 0:
            path = hand_wobble(path, length, np.random.default_rng([rng_seed, 7919]), lim.wobble)
        kw = dict(points=path.tolist(), brush=brush, color=color, size=size, opacity=1.0 - dilution,
                  taper_in=0.0, taper_out=0.0, fade=1.0 - load, pressure=pressure, dense=True, seed=rng_seed,
                  wet=wet, angle=angle)
        canvas.checkpoint()
        canvas.stroke(**kw)
        shape = "dab" if length < 1 else f"{length:.0f} px through {len(nodes)} points"
        if not legacy and len(nodes) > 1:
            shape += (", smooth" if smooth else ", sharp corners") + (", closed" if closed else "")
        return {"summary": f"{brush} stroke, {shape}", "params": {k: v for k, v in kw.items() if k != "points"}}

    if name == "stroke":
        kw = dict(
            points=_points(args, "points", 1),
            brush=_enum(args, "brush", BRUSHES),
            color=_color(args, "color"),
            size=_num(args, "size", 1, 400),
            opacity=_num(args, "opacity", 0, 1, 1.0),
            hardness=_num(args, "hardness", 0, 1),
            taper_in=_num(args, "taper_in", 0, 1),
            taper_out=_num(args, "taper_out", 0, 1),
            fade=_num(args, "fade", 0, 1, 0.0),
            jitter=_num(args, "jitter", 0, 1, 0.0),
            blend=_enum(args, "blend", BLEND_MODES),
            seed=rng_seed,
        )
        if kw["brush"] is None or kw["size"] is None:
            raise ToolError("`brush` and `size` are required")
        canvas.checkpoint()
        canvas.stroke(**kw)
        return {"summary": f"{kw['brush']} stroke, {len(kw['points'])} points", "params": kw}

    if name == "fill":
        kw = dict(
            color=_color(args, "color"),
            points=_points(args, "points", 3, required=False),
            color_to=_color(args, "color_to", required=False),
            angle=_num(args, "angle", -3600, 3600, 90.0),
            opacity=_num(args, "opacity", 0, 1, 1.0),
            softness=_num(args, "softness", 0, 200, 0.0),
            texture=_enum(args, "texture", ["none", "watercolor", "charcoal", "canvas"], "none"),
            blend=_enum(args, "blend", BLEND_MODES, "normal"),
        )
        canvas.checkpoint()
        canvas.fill(**kw)
        where = "whole canvas" if kw["points"] is None else f"{len(kw['points'])}-gon"
        return {"summary": f"fill ({where})", "params": kw}

    if name == "soften":
        region = args.get("region")
        if region is not None and not (isinstance(region, list) and len(region) == 4 and all(isinstance(v, (int, float)) for v in region)):
            raise ToolError("`region` must be [x, y, width, height]")
        amount = _num(args, "amount", 0.5, 40, 2.0)
        canvas.checkpoint()
        canvas.soften(region, amount)
        return {"summary": "soften", "params": {"region": region, "amount": amount}}

    if name == "look":
        x = _num(args, "x", 0, canvas.w)
        y = _num(args, "y", 0, canvas.h)
        w = _num(args, "width", 8, canvas.w)
        h = _num(args, "height", 8, canvas.h)
        if None in (x, y, w, h):
            raise ToolError("x, y, width and height are required")
        img = canvas.crop(int(x), int(y), int(w), int(h))
        scale = min(4.0, 768 / max(img.width, img.height))
        if scale > 1:
            from PIL import Image

            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        return {"summary": f"look at ({x:.0f},{y:.0f},{w:.0f}x{h:.0f})", "image": img, "params": args}

    if name == "undo":
        steps = int(_num(args, "steps", 1, 30, 1))
        n = canvas.undo(steps)
        return {"summary": f"undid {n} action(s)", "params": {"steps": steps}, "undone": n}

    if name == "finish":
        return {"summary": "finished", "params": args}

    raise ToolError(f"unknown tool {name!r}")
