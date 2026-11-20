"""Probe an endpoint for each API feature the painting loop uses, one tiny request each.

Run this before the first session on a new provider (e.g. a reseller/proxy): it tells you
which `run` flags you need. Every probe uses a small max_tokens, so the whole check costs
a fraction of a cent on Haiku.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Callable

import anthropic
from PIL import Image

from .agent import RunConfig, is_empty_refusal, make_client


def _png_b64(color=(200, 40, 40)) -> str:
    img = Image.new("RGB", (64, 64), (245, 241, 234))
    img.paste(color, (16, 16, 48, 48))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def _text(msg) -> str:
    return " ".join(b.text for b in msg.content if b.type == "text").strip()


class _Retrying:
    """Wraps create() to resend empty refusals, as a run would, and counts them."""

    def __init__(self, create, tries):
        self.create, self.tries, self.empty, self.total = create, tries, 0, 0

    def __call__(self, **kw):
        for i in range(self.tries + 1):
            r = self.create(**kw)
            self.total += 1
            if not is_empty_refusal(r) or i == self.tries:
                return r
            self.empty += 1
        return r


def run_checks(cfg: RunConfig) -> dict[str, bool]:
    raw = make_client(cfg)
    create = _Retrying(raw.messages.create, cfg.empty_refusal_retries)
    beta_create = _Retrying(raw.beta.messages.create, cfg.empty_refusal_retries)
    m = cfg.model
    hi = [{"role": "user", "content": "Reply with the single word: ready"}]
    tool = {
        "name": "dab",
        "description": "Put a dot of colour on the canvas.",
        "input_schema": {
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "color": {"type": "string"}},
            "required": ["x", "y", "color"],
        },
    }
    image_msg = lambda src: [{"role": "user", "content": [  # noqa: E731
        {"type": "image", "source": src},
        {"type": "text", "text": "What colour is the square in the middle? One word."},
    ]}]

    def basic():
        return _text(create(model=m, max_tokens=50, messages=hi))

    def streaming():
        with raw.beta.messages.stream(model=m, max_tokens=50, messages=hi) as s:
            return _text(s.get_final_message())

    def effort_thinking():
        r = beta_create(
            model=m, max_tokens=2000, messages=hi,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": cfg.effort},
        )
        kinds = [b.type for b in r.content]
        return f"blocks={kinds}"

    def tools():
        r = create(
            model=m, max_tokens=500, tools=[tool],
            messages=[{"role": "user", "content": "Use the dab tool once, anywhere, any colour."}],
        )
        uses = [b for b in r.content if b.type == "tool_use"]
        if not uses:
            raise RuntimeError(f"no tool call came back (stop_reason={r.stop_reason})")
        return f"dab({uses[0].input})"

    def image_base64():
        r = create(model=m, max_tokens=50, messages=image_msg(
            {"type": "base64", "media_type": "image/png", "data": _png_b64()}))
        return _text(r)

    def files_api():
        meta = raw.files.upload(file=("probe.png", base64.b64decode(_png_b64((40, 60, 200))), "image/png"))
        try:
            r = create(model=m, max_tokens=50, messages=image_msg({"type": "file", "file_id": meta.id}))
            return _text(r)
        finally:
            try:
                raw.files.delete(meta.id)
            except anthropic.APIError:
                pass

    def context_editing():
        r = beta_create(
            model=m, max_tokens=50, messages=hi,
            betas=["context-management-2025-06-27"],
            context_management={"edits": [{
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 40000},
                "keep": {"type": "tool_uses", "value": 4},
            }]},
        )
        return _text(r)

    def caching():
        r = create(model=m, max_tokens=50, messages=hi, cache_control={"type": "ephemeral"})
        return _text(r)

    probes: list[tuple[str, Callable[[], Any], str]] = [
        ("basic request", basic, "nothing will work: check base URL, key and model name"),
        ("streaming", streaming, "required"),
        ("tool use", tools, "required"),
        ("image (base64)", image_base64, "required: the model must see its canvas"),
        ("effort + thinking summaries", effort_thinking, "use --no-thinking-summaries (and/or --effort default)"),
        ("prompt caching", caching, "costs more without it; no flag yet"),
        ("Files API", files_api, "use --inline-images"),
        ("context editing", context_editing, "use --no-context-editing"),
    ]
    results: dict[str, bool] = {}
    print(f"endpoint: {cfg.base_url or 'api.anthropic.com (default)'}   model: {m}\n")
    for name, fn, if_fail in probes:
        try:
            detail = fn()
            results[name] = True
            print(f"  ok    {name:28s} {str(detail)[:70]}")
        except Exception as e:  # diagnostic: report anything, keep probing
            results[name] = False
            msg = getattr(e, "message", None) or str(e)
            print(f"  FAIL  {name:28s} {str(msg)[:110]}\n        -> {if_fail}")
            if name == "basic request":
                break
    total = create.total + beta_create.total
    empty = create.empty + beta_create.empty
    if empty:
        print(f"\nnote: {empty} of {total} requests came back as an empty refusal (a bad backend at the "
              f"provider) and were resent; runs do the same automatically, up to {cfg.empty_refusal_retries} times.")
    return results
