"""The painting loop: give the model a canvas and brushes, show it the result after every action."""

from __future__ import annotations

import base64
import io
import json
import os
import hashlib
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import anthropic
from PIL import Image

from .canvas import Canvas
from .tools import MARKING_TOOLS, Limits, ToolError, apply_tool, tool_definitions

DEFAULT_SYSTEM_PROMPT = """\
You have a blank canvas, {width} by {height} pixels, and a set of brushes.

This painting is an exercise for yourself, not a task to complete. There's nothing to \
deliver.

Paint without a plan. Don't decide on a subject or picture what it will become. Begin \
with an arbitrary stroke, anywhere, and see what it suggests. After each stroke you'll \
see the canvas as it is now: look at what's there, respond to it, and let the painting \
find its own way as you improvise.

Still, compose something. It doesn't have to depict anything, and abstraction is \
welcome, but art is expression, not random squiggles. Let the strokes come to work \
together and say something.

A stroke can be as large, long or intricate as you like. You simply see each one land \
before you make the next.

Most of the canvas will probably end up covered in paint, over many strokes.

When you feel it's finished, call `finish`."""

FIRST_MESSAGE = "The canvas is ready."

NUDGE = "(The canvas is still open. If you're done, call `finish`; otherwise carry on.)"

# Rough per-million-token prices for the cost estimate.
PRICES = {
    "claude-opus-5-5": dict(input=4.0, output=20.0, cache_read=0.20, cache_write=5.0),
    # Haiku 5.5: prompt rates for requests under 100K tokens; cache rates approximate.
    "claude-haiku-5-5": dict(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125),
}

# Models that accept the server-side refusal fallback parameter.
FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


@dataclass
class RunConfig:
    model: str = "claude-opus-5-5"
    effort: str = "high"
    width: int = 1024
    height: int = 1024
    paper: str = "#f4f1ea"
    max_actions: int = 300  # a safety cap on strokes; the model isn't told about it
    max_nudges: int = 1
    keep_images: int = 4  # how many recent tool results (canvas views) stay in context
    clear_trigger_tokens: int = 40_000
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    out_dir: Path = Path("runs")
    seed: int = 0
    fallbacks: bool = True
    file_ttl_seconds: int = 2 * 24 * 3600
    # Provider plumbing, for going through a proxy/reseller instead of api.anthropic.com.
    base_url: str | None = None
    api_key_env: str | None = None  # read the key from this env var instead of ANTHROPIC_API_KEY
    inline_images: bool = False  # send canvas views as base64 instead of via the Files API
    context_editing: bool = True  # server-side clearing of old canvas views
    thinking_summaries: bool = True  # ask for readable thinking summaries for the journal
    view_size: int = 1024  # longest side of the canvas image the model sees
    max_stroke_length: float | None = None  # px along the path; None = no limit
    max_brush_size: float | None = None  # px; default 15% of the longer side
    allow_undo: bool = False  # every stroke is final, as on a real canvas
    allow_look: bool = True
    one_call_per_turn: bool = True  # brush, see the canvas, repeat
    empty_refusal_retries: int = 6
    transient_retries: int = 8  # resend after mid-stream overload/unavailable errors
    mixing: str = "pigment"  # "pigment": paint mixes like pigment (blue + yellow = green); "rgb": like light
    wobble: float = 1.0  # natural hand wobble on each stroke; 0 = perfectly geometric


def is_empty_refusal(msg) -> bool:
    """A "refusal" with no content and no output tokens.

    Some resellers load-balance across backends, and a bad backend answers every request
    this way, even a harmless one. Nothing was generated, so the identical request can
    simply be sent again. A real refusal from the model is left alone.
    """
    return msg.stop_reason == "refusal" and not msg.content and not (msg.usage.output_tokens or 0)


def _is_transient(e: Exception) -> bool:
    if isinstance(e, anthropic.APIConnectionError):
        return True
    status = getattr(e, "status_code", None)
    text = str(getattr(e, "message", "") or e).lower()
    # status 200: the error arrived as an event after the response had started, i.e. the
    # server or an upstream backend failed partway; nothing was committed, so resend.
    return (status in (200, 408, 409, 429) or (status or 0) >= 500 or "overloaded" in text
            or "temporarily unavailable" in text or "try again" in text or "upstream" in text)


def make_client(cfg: RunConfig) -> anthropic.Anthropic:
    kw: dict[str, Any] = {"max_retries": 5}
    if cfg.base_url:
        kw["base_url"] = cfg.base_url
    if cfg.api_key_env:
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise SystemExit(f"environment variable {cfg.api_key_env} is not set")
        kw["api_key"] = key
    return anthropic.Anthropic(**kw)


@dataclass
class RunStats:
    actions: int = 0
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    uploaded_files: list[str] = field(default_factory=list)
    empty_refusals: int = 0
    transient_errors: int = 0

    def cost(self, model: str) -> float | None:
        p = PRICES.get(model)
        if not p:
            return None
        return (
            self.input_tokens * p["input"]
            + self.output_tokens * p["output"]
            + self.cache_read * p["cache_read"]
            + self.cache_write * p["cache_write"]
        ) / 1e6


def _jpeg(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


try:  # exactly how the SDK serialises a response block when it is sent back in a request
    from anthropic._utils._json import openapi_model_dump as _dump_model
except ImportError:  # pragma: no cover - private helper moved; to_dict is the documented near-equivalent
    def _dump_model(b: Any) -> dict[str, Any]:
        return b.to_dict(mode="json")


def _block_dict(b: Any) -> Any:
    return _dump_model(b) if hasattr(b, "model_dump") else b


class PaintingSession:
    """One painting session. Everything needed to pick it up again is written to its
    directory after every turn: the conversation (`conversation.jsonl`, with inline canvas
    views stored as files under `views/`) and the loop state (`state.json`). See `resume`."""

    def __init__(self, cfg: RunConfig, client: anthropic.Anthropic | None = None, *, _resume_dir: Path | None = None):
        self.cfg = cfg
        self.client = client or make_client(cfg)
        self.canvas = Canvas(cfg.width, cfg.height, paper=cfg.paper, seed=cfg.seed, mixing=cfg.mixing)
        lim = Limits.for_canvas(cfg.width, cfg.height)
        self.limits = Limits(cfg.max_stroke_length or lim.max_stroke_length, cfg.max_brush_size or lim.max_brush_size,
                             wobble=cfg.wobble)
        cfg.max_stroke_length, cfg.max_brush_size = self.limits.max_stroke_length, self.limits.max_brush_size
        self.tools = tool_definitions(cfg.width, cfg.height, self.limits, undo=cfg.allow_undo, look=cfg.allow_look)
        self.tool_names = {t["name"] for t in self.tools}
        self.system = cfg.system_prompt.format(width=cfg.width, height=cfg.height, max_actions=cfg.max_actions)
        if max(cfg.width, cfg.height) > cfg.view_size:
            self.system += (f"\n\n(The canvas image you see is scaled down to fit {cfg.view_size} pixels. "
                            f"Tool coordinates are always in canvas pixels, {cfg.width} by {cfg.height}.)")
        self.stats = RunStats()
        self.messages: list[Any] = []
        self.finished: dict[str, Any] | None = None
        self._seed_counter = cfg.seed * 1_000_003
        self._view_paths: dict[str, str] = {}  # sha1 of base64 data -> file under views/
        self._committed = 0  # messages already in conversation.jsonl
        self._event_count = 0
        self.resumed = _resume_dir is not None
        if self.resumed:
            self.dir = _resume_dir
            self._load()
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = cfg.out_dir / stamp
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.dir / "views").mkdir(exist_ok=True)
        self._journal = open(self.dir / "journal.md", "w", encoding="utf-8")
        self._events = open(self.dir / "events.jsonl", "w", encoding="utf-8")
        (self.dir / "conversation.jsonl").write_text("")
        meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()}
        meta["system_prompt_rendered"] = self.system
        (self.dir / "config.json").write_text(json.dumps(meta, indent=2))

    # ------------------------------------------------------------------ resume
    @classmethod
    def resume(cls, run_dir: Path, client: anthropic.Anthropic | None = None) -> "PaintingSession":
        """Pick up a run where it stopped (an API error, Ctrl-C, a crash, or the model ending
        its turn without finishing). The canvas is rebuilt exactly by replaying the strokes
        it made, and the model gets back the same conversation, so it carries on as if
        nothing happened."""
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "config.json").read_text())
        state_path = run_dir / "state.json"
        if not state_path.exists():
            raise SystemExit(f"{run_dir} has no state.json (made before runs were resumable)")
        if json.loads(state_path.read_text()).get("finished") is not None:
            raise SystemExit(f"{run_dir} is already finished")
        known = {f.name for f in fields(RunConfig)}
        cfg = RunConfig(**{k: v for k, v in meta.items() if k in known})
        cfg.mixing = meta.get("mixing", "rgb")  # runs from before pigment mixing mixed like light
        cfg.out_dir = Path(cfg.out_dir)
        sess = cls(cfg, client, _resume_dir=run_dir)
        sess.system = meta["system_prompt_rendered"]  # exactly what the model saw before
        return sess

    def _load(self) -> None:
        state = json.loads((self.dir / "state.json").read_text())
        self.stats = RunStats(**state["stats"])
        self._seed_counter = state["seed_counter"]
        self._view_paths = state.get("view_paths", {})
        # Keep only the events of completed turns, and rebuild the canvas from them.
        lines = (self.dir / "events.jsonl").read_text().splitlines()[: state["events"]]
        (self.dir / "events.jsonl").write_text("".join(line + "\n" for line in lines))
        self._event_count = len(lines)
        for line in lines:
            ev = json.loads(line)
            if ev["kind"] == "tool" and ev["tool"] in MARKING_TOOLS:
                apply_tool(self.canvas, ev["tool"], ev["input"], ev["seed"], self.limits)
        for f in (self.dir / "frames").glob("*.png"):
            if int(f.stem) > self.stats.actions:
                f.unlink()
        for line in (self.dir / "conversation.jsonl").read_text().splitlines()[: state["messages"]]:
            self.messages.append(self._restore_message(json.loads(line)))
        self._committed = len(self.messages)
        self._journal = open(self.dir / "journal.md", "a", encoding="utf-8")
        self._events = open(self.dir / "events.jsonl", "a", encoding="utf-8")

    def _store_message(self, m: dict[str, Any]) -> dict[str, Any]:
        content = m["content"]
        if isinstance(content, str):
            return dict(m)
        return {"role": m["role"], "content": [self._store_block(_block_dict(b)) for b in content]}

    def _store_block(self, b: dict[str, Any]) -> dict[str, Any]:
        if b.get("type") == "image" and b.get("source", {}).get("type") == "base64":
            path = self._view_paths[hashlib.sha1(b["source"]["data"].encode()).hexdigest()]
            return {**b, "source": {**b["source"], "data": None, "data_file": path}}
        if b.get("type") == "tool_result" and isinstance(b.get("content"), list):
            return {**b, "content": [self._store_block(_block_dict(c)) for c in b["content"]]}
        return b

    def _restore_message(self, m: dict[str, Any]) -> dict[str, Any]:
        if isinstance(m["content"], str):
            return m
        return {"role": m["role"], "content": [self._restore_block(b) for b in m["content"]]}

    def _restore_block(self, b: dict[str, Any]) -> dict[str, Any]:
        src = b.get("source") or {}
        if b.get("type") == "image" and "data_file" in src:
            data = base64.standard_b64encode((self.dir / src["data_file"]).read_bytes()).decode()
            return {**b, "source": {"type": "base64", "media_type": src["media_type"], "data": data}}
        if b.get("type") == "tool_result" and isinstance(b.get("content"), list):
            return {**b, "content": [self._restore_block(c) for c in b["content"]]}
        return b

    def _commit(self) -> None:
        """Write completed turns to disk. A model turn whose tool calls haven't been answered
        yet is held back, together with any strokes it made, so a resume redoes that turn."""
        n = len(self.messages)
        last = self.messages[-1] if self.messages else None
        if last and last["role"] == "assistant" and any(
            _block_dict(b).get("type") == "tool_use" for b in last["content"]
        ):
            n -= 1
            if self.finished is not None:
                n += 1  # finish needs no answer
        if n > self._committed or not (self.dir / "state.json").exists():
            with open(self.dir / "conversation.jsonl", "a", encoding="utf-8") as f:
                for m in self.messages[self._committed : n]:
                    f.write(json.dumps(self._store_message(m), default=str) + "\n")
            self._committed = n
            self._committed_events = self._event_count
            self._committed_stats = asdict(self.stats)
            self._committed_seed = self._seed_counter
        state = {
            "messages": self._committed,
            "events": self._committed_events,
            "stats": self._committed_stats,
            "seed_counter": self._committed_seed,
            "view_paths": self._view_paths,
            "finished": self.finished,
        }
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(json.dumps(state))
        tmp.replace(self.dir / "state.json")

    # ------------------------------------------------------------------ logging
    def _log(self, kind: str, **data: Any) -> None:
        self._events.write(json.dumps({"t": time.time(), "kind": kind, **data}, default=str) + "\n")
        self._events.flush()
        self._event_count += 1

    def _journal_write(self, text: str) -> None:
        self._journal.write(text)
        self._journal.flush()

    def _save_frame(self) -> None:
        img = self.canvas.image()
        img.save(self.dir / "frames" / f"{self.stats.actions:04d}.png")

    # ------------------------------------------------------------------ images
    def _image_block(self, img: Image.Image, name: str) -> dict[str, Any]:
        """Upload an image to the Files API and reference it by id.

        Sending base64 would make every request carry every earlier canvas view (the client
        resends the whole conversation), which passes the request size limit in a long run.
        """
        if img.width > self.cfg.view_size or img.height > self.cfg.view_size:
            img = img.copy()
            img.thumbnail((self.cfg.view_size, self.cfg.view_size), Image.LANCZOS)
        if self.cfg.inline_images:
            data = _jpeg(img, quality=82)
            b64 = base64.standard_b64encode(data).decode()
            rel = f"views/{name}.jpg"
            (self.dir / rel).write_bytes(data)
            self._view_paths[hashlib.sha1(b64.encode()).hexdigest()] = rel
            return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
        data = _jpeg(img)
        try:
            meta = self.client.files.upload(
                file=(f"{name}.jpg", data, "image/jpeg"), expires_in_seconds=self.cfg.file_ttl_seconds
            )
        except anthropic.BadRequestError:
            meta = self.client.files.upload(file=(f"{name}.jpg", data, "image/jpeg"))
        self.stats.uploaded_files.append(meta.id)
        return {"type": "image", "source": {"type": "file", "file_id": meta.id}}

    # ------------------------------------------------------------------ api
    def _request(self):
        kwargs: dict[str, Any] = dict(
            model=self.cfg.model,
            max_tokens=64_000,
            system=self.system,
            tools=self.tools,
            messages=self.messages,
            output_config={"effort": self.cfg.effort},
            cache_control={"type": "ephemeral"},
            betas=[],
        )
        if self.cfg.one_call_per_turn:
            kwargs["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        if self.cfg.thinking_summaries:
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        if self.cfg.context_editing:
            kwargs["betas"].append("context-management-2025-06-27")
            kwargs["context_management"] = {
                "edits": [
                    {
                        "type": "clear_tool_uses_20250919",
                        "trigger": {"type": "input_tokens", "value": self.cfg.clear_trigger_tokens},
                        "keep": {"type": "tool_uses", "value": self.cfg.keep_images},
                        "exclude_tools": ["finish"],
                    }
                ]
            }
        if self.cfg.fallbacks and self.cfg.model in FALLBACK_MODELS:
            kwargs["betas"].append("server-side-fallback-2026-07-01")
            kwargs["fallbacks"] = "default"
        if not kwargs["betas"]:
            del kwargs["betas"]
        empty = transient = 0
        while True:
            try:
                with self.client.beta.messages.stream(**kwargs) as stream:
                    msg = stream.get_final_message()
            except (anthropic.APIConnectionError, anthropic.APIStatusError) as e:
                # The SDK already retries errors that arrive before the response starts; this
                # catches ones that arrive mid-stream (some providers send "model temporarily
                # unavailable" that way). Nothing was added to the conversation, so resend.
                if not _is_transient(e) or transient >= self.cfg.transient_retries:
                    raise
                transient += 1
                self.stats.transient_errors += 1
                self._log("transient_error_retry", attempt=transient, error=str(e)[:300])
                time.sleep(min(2 ** transient, 60))
                continue
            u = msg.usage
            self.stats.requests += 1
            self.stats.input_tokens += u.input_tokens or 0
            self.stats.output_tokens += u.output_tokens or 0
            self.stats.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
            self.stats.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0
            if not is_empty_refusal(msg) or empty >= self.cfg.empty_refusal_retries:
                return msg
            empty += 1
            self.stats.empty_refusals += 1
            self._log("empty_refusal_retry", attempt=empty)
            time.sleep(min(2 ** (empty - 1), 20))

    # ------------------------------------------------------------------ loop
    def run(self, on_event=None) -> Path:
        say = on_event or (lambda s: None)
        if self.resumed:
            self._journal_write(f"\n*(Resumed at stroke {self.stats.actions}.)*\n\n")
            self._log("resumed", actions=self.stats.actions)
            say(f"resuming at stroke {self.stats.actions}")
            if self.messages and self.messages[-1]["role"] == "assistant":
                # it had ended its turn without finishing; give it the same nudge
                self.messages.append({"role": "user", "content": NUDGE})
                self._log("nudge")
        else:
            self._journal_write(f"# Painterly run {self.dir.name}\n\nModel: `{self.cfg.model}` · effort `{self.cfg.effort}`\n\n")
            self._save_frame()
            self.messages.append(
                {"role": "user", "content": [self._image_block(self.canvas.image(), "frame-0000"), {"type": "text", "text": FIRST_MESSAGE}]}
            )
        self._committed_events, self._committed_stats, self._committed_seed = (
            self._event_count, asdict(self.stats), self._seed_counter)
        self._commit()
        nudges = 0
        try:
            while self.finished is None:
                msg = self._request()
                self.messages.append({"role": "assistant", "content": msg.content})
                self._record_assistant(msg, say)

                if msg.stop_reason == "refusal":
                    self._log("refusal", details=getattr(msg, "stop_details", None))
                    say("model declined; stopping")
                    break

                uses = [b for b in msg.content if b.type == "tool_use"]
                if not uses:
                    if msg.stop_reason == "pause_turn":
                        continue
                    if nudges >= self.cfg.max_nudges:
                        self._log("ended_without_finish")
                        self._journal_write("\n*(Ended its turn without calling `finish`.)*\n")
                        break
                    nudges += 1
                    self.messages.append({"role": "user", "content": NUDGE})
                    self._log("nudge")
                    self._commit()
                    continue

                truncated = msg.stop_reason == "max_tokens"
                results = self._run_tools(uses, truncated, say)
                if self.finished is None:
                    self.messages.append({"role": "user", "content": results})
                self._commit()
        except KeyboardInterrupt:
            say("interrupted; saving what's there")
            self._log("interrupted")
        except anthropic.APIStatusError as e:
            say(f"API error {e.status_code}: {e.message}")
            self._log("api_error", status=e.status_code, message=str(e.message))
        except anthropic.APIConnectionError as e:
            say(f"connection error: {e}")
            self._log("api_error", status=None, message=str(e))
        finally:
            self._wrap_up(say)
        return self.dir

    def _record_assistant(self, msg, say) -> None:
        for b in msg.content:
            if b.type == "thinking" and getattr(b, "thinking", ""):
                quoted = "\n".join(f"> {line}" if line else ">" for line in b.thinking.strip().splitlines())
                self._journal_write(f"> *(thinking)*\n{quoted}\n\n")
                self._log("thinking", text=b.thinking)
            elif b.type == "text" and b.text.strip():
                self._journal_write(b.text.strip() + "\n\n")
                self._log("text", text=b.text)
                say(f"says: {b.text.strip()[:200]}")
            elif b.type == "fallback":
                self._log("fallback", block=b.model_dump() if hasattr(b, "model_dump") else str(b))

    def _run_tools(self, uses, truncated: bool, say) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        changed = False
        acted = False
        for i, tu in enumerate(uses):
            if truncated:
                results.append({"type": "tool_result", "tool_use_id": tu.id, "is_error": True,
                                "content": "Your output was cut off before this call was complete; nothing was applied."})
                continue
            if tu.name not in self.tool_names:
                results.append({"type": "tool_result", "tool_use_id": tu.id, "is_error": True,
                                "content": f"There is no `{tu.name}` tool."})
                continue
            if self.cfg.one_call_per_turn and acted and tu.name != "finish":
                results.append({"type": "tool_result", "tool_use_id": tu.id, "is_error": True,
                                "content": "Not carried out: only the first call in a turn is made, so you can see how it landed before the next."})
                continue
            if self.stats.actions >= self.cfg.max_actions and tu.name in MARKING_TOOLS:
                results.append({"type": "tool_result", "tool_use_id": tu.id, "is_error": True,
                                "content": "This session can't take any more strokes. Call `finish` when you're ready."})
                continue
            acted = True
            self._seed_counter += 1
            try:
                out = apply_tool(self.canvas, tu.name, tu.input, self._seed_counter, self.limits)
            except ToolError as e:
                results.append({"type": "tool_result", "tool_use_id": tu.id, "is_error": True, "content": f"Error: {e}"})
                self._log("tool_error", tool=tu.name, input=tu.input, error=str(e))
                continue
            except Exception as e:  # engine bug: tell the model, keep going
                results.append({"type": "tool_result", "tool_use_id": tu.id, "is_error": True, "content": f"Internal error: {e}"})
                self._log("tool_crash", tool=tu.name, input=tu.input, error=repr(e))
                continue

            self._log("tool", tool=tu.name, input=tu.input, seed=self._seed_counter, action=self.stats.actions + 1)
            self._journal_write(f"`{tu.name}` — {out['summary']}\n\n")
            say(f"{tu.name}: {out['summary']}")

            if tu.name == "finish":
                self.finished = dict(tu.input or {})
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": "Done. Thank you."})
                continue
            if tu.name == "look":
                block = self._image_block(out["image"], f"look-{self.stats.actions:04d}-{i}")
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": [block]})
                continue

            self.stats.actions += 1
            changed = True
            self._save_frame()
            view_at = len(results)
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": [{"type": "text", "text": "ok"}]})

        if changed and self.finished is None:
            # Attach the current canvas once, to the result of the last call that changed it.
            view = self._image_block(self.canvas.image(), f"frame-{self.stats.actions:04d}")
            last = results[view_at]
            content = last["content"] if isinstance(last["content"], list) else [{"type": "text", "text": last["content"]}]
            last["content"] = content + [view]
        return results

    def _wrap_up(self, say) -> None:
        self._commit()
        img = self.canvas.image()
        img.save(self.dir / "final.png")
        frames = sorted((self.dir / "frames").glob("*.png"))
        if len(frames) > 1:
            pics = [Image.open(f).convert("RGB").resize((512, round(512 * self.cfg.height / self.cfg.width))) for f in frames]
            pics += [pics[-1]] * 8  # linger on the result
            pics[0].save(self.dir / "timelapse.gif", save_all=True, append_images=pics[1:], duration=120, loop=0)
        summary = {
            "finished": self.finished is not None,
            "title": (self.finished or {}).get("title"),
            "note": (self.finished or {}).get("note"),
            "actions": self.stats.actions,
            "requests": self.stats.requests,
            "empty_refusal_retries": self.stats.empty_refusals,
            "transient_error_retries": self.stats.transient_errors,
            "tokens": {
                "input": self.stats.input_tokens,
                "output": self.stats.output_tokens,
                "cache_read": self.stats.cache_read,
                "cache_write": self.stats.cache_write,
            },
            "estimated_cost_usd": self.stats.cost(self.cfg.model),
            "coverage": round(self.canvas.coverage(), 3),
        }
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        if self.finished:
            t, n = self.finished.get("title"), self.finished.get("note")
            self._journal_write("\n---\n\n" + (f"## {t}\n\n" if t else "") + (f"{n}\n" if n else ""))
        self._journal.close()
        self._events.close()
        say(f"saved to {self.dir}" + ("" if self.finished else f"  (resume with: python -m painterly resume {self.dir})"))
