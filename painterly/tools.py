"""Tool definitions the agent sees, and the dispatcher that applies them to a Canvas."""

from __future__ import annotations

from typing import Any

from .canvas import BLEND_MODES, BRUSHES, Canvas, hex_to_rgb

_brush_list = "\n".join(f"- {b.name}: {b.description}" for b in BRUSHES.values())

POINT = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}


def tool_definitions(width: int, height: int) -> list[dict[str, Any]]:
    coord = f"Canvas pixels: x 0..{width} left to right, y 0..{height} top to bottom."
    return [
        {
            "name": "stroke",
            "description": (
                "Drag a brush through a sequence of points. The path is smoothed into a curve "
                "through the points, so a few well-placed points make a fluid stroke; one point "
                f"makes a single dab. {coord}\n\nBrushes:\n{_brush_list}"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "brush": {"type": "string", "enum": list(BRUSHES)},
                    "points": {"type": "array", "items": POINT, "minItems": 1, "maxItems": 400},
                    "color": {"type": "string", "description": "Hex colour, e.g. '#2b4a6f'."},
                    "size": {"type": "number", "description": "Brush width in pixels (1-400)."},
                    "opacity": {"type": "number", "description": "0-1. Default 1."},
                    "hardness": {"type": "number", "description": "Edge crispness 0-1. Default depends on the brush."},
                    "taper_in": {"type": "number", "description": "0-1: fraction of the stroke over which width swells from a point at the start."},
                    "taper_out": {"type": "number", "description": "0-1: same, thinning to a point at the end."},
                    "fade": {"type": "number", "description": "0-1: how much the brush runs out of paint by the end of the stroke."},
                    "jitter": {"type": "number", "description": "0-1: hand wobble applied to the path (for spatter: droplet density)."},
                    "blend": {"type": "string", "enum": list(BLEND_MODES), "description": "How the paint mixes with what's beneath. 'erase' lifts paint back to bare paper."},
                },
                "required": ["brush", "points", "color", "size"],
                "additionalProperties": False,
            },
            "eager_input_streaming": True,
        },
        {
            "name": "fill",
            "description": (
                "Lay paint over a polygon, or over the whole canvas when `points` is omitted. "
                f"Optionally a linear gradient from `color` to `color_to`. {coord}"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "color": {"type": "string"},
                    "points": {"type": "array", "items": POINT, "minItems": 3, "maxItems": 400, "description": "Polygon vertices. Omit to cover the whole canvas."},
                    "color_to": {"type": "string", "description": "End colour for a gradient."},
                    "angle": {"type": "number", "description": "Gradient direction in degrees: 0 = left to right, 90 = top to bottom."},
                    "opacity": {"type": "number"},
                    "softness": {"type": "number", "description": "Edge blur radius in pixels."},
                    "texture": {"type": "string", "enum": ["none", "watercolor", "charcoal", "canvas"]},
                    "blend": {"type": "string", "enum": list(BLEND_MODES)},
                },
                "required": ["color"],
                "additionalProperties": False,
            },
            "eager_input_streaming": True,
        },
        {
            "name": "soften",
            "description": "Blend wet paint together by blurring a rectangle (or the whole canvas if `region` is omitted).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "region": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": "[x, y, width, height]"},
                    "amount": {"type": "number", "description": "Blur radius in pixels (0.5-40)."},
                },
                "additionalProperties": False,
            },
            "eager_input_streaming": True,
        },
        {
            "name": "look",
            "description": "Look closely at part of the canvas: returns that rectangle enlarged. Does not change the canvas.",
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
        },
        {
            "name": "undo",
            "description": "Take back the most recent marks (strokes, fills, softening).",
            "input_schema": {
                "type": "object",
                "properties": {"steps": {"type": "integer", "description": "How many actions to undo. Default 1."}},
                "additionalProperties": False,
            },
            "eager_input_streaming": True,
        },
        {
            "name": "finish",
            "description": "Call this when you are done with the canvas. Title and note are optional.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "note": {"type": "string", "description": "Anything you want to say about it, or nothing."},
                },
                "additionalProperties": False,
            },
            "eager_input_streaming": True,
        },
    ]


MARKING_TOOLS = {"stroke", "fill", "soften", "undo"}


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


def apply_tool(canvas: Canvas, name: str, args: dict[str, Any], rng_seed: int) -> dict[str, Any]:
    """Apply one tool call. Returns a dict describing what happened (for the log).

    Raises ToolError for bad input. The returned dict always has a 'summary' string;
    `look` additionally returns a PIL image under 'image'.
    """
    if not isinstance(args, dict):
        raise ToolError("input must be an object")

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
