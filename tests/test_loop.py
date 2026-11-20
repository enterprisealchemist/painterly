"""End-to-end test of the painting loop with a scripted stand-in for the API client."""

import json
from types import SimpleNamespace

from anthropic.types.beta import BetaMessage

from painterly.agent import PaintingSession, RunConfig
from painterly.replay import replay


def _msg(content, stop="tool_use"):
    return BetaMessage.model_validate({
        "id": "msg_x", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": content, "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": 1000, "output_tokens": 200},
    })


def _brush(i, **kw):
    args = {"brush": "bristle", "start": [100, 600], "end": [400, 560], "pressure": [0.3, 1, 0.4],
            "color": "#2d1e2f", "size": 40, **kw}
    return {"type": "tool_use", "id": f"t{i}", "name": "brush", "input": args}


SCRIPT = [
    _msg([
        {"type": "thinking", "thinking": "A wide quiet field, then one line.", "signature": "s"},
        {"type": "text", "text": "Starting with a wash."},
        _brush(1, brush="watercolor", start=[50, 200], end=[420, 200], curvature=0.3, color="#5b6c8f", size=150),
        _brush(2),  # a second call in the same turn is refused
    ]),
    _msg([_brush(3, brush="nope")]),  # unknown brush -> error
    _msg([{"type": "tool_use", "id": "t4", "name": "look", "input": {"x": 0, "y": 500, "width": 300, "height": 200}}]),
    _msg([_brush(5, start=[0, 0], end=[900, 900])]),  # too long -> error
    _msg([{"type": "tool_use", "id": "t6", "name": "fill", "input": {"color": "#000"}}]),  # not offered
    _msg([_brush(7)]),
    _msg([_brush(8, brush="spatter", start=[500, 300], end=[500, 300], pressure=[1], color="#c44536", size=120)]),
    _msg([{"type": "tool_use", "id": "t9", "name": "undo", "input": {}}]),
    _msg([_brush(10, brush="blender", start=[100, 150], end=[400, 250], pressure=[0.8], size=60)]),
    _msg([{"type": "text", "text": "Hm."}], stop="end_turn"),
    _msg([{"type": "tool_use", "id": "t11", "name": "finish", "input": {"title": "Test", "note": "fine"}}]),
]


class FakeClient:
    def __init__(self, script=None):
        self.script = script or SCRIPT
        self.requests = []
        self.n = 0
        self.files = SimpleNamespace(upload=self._upload)
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _upload(self, file, **kw):
        self.n += 1
        return SimpleNamespace(id=f"file_{self.n}")

    def _stream(self, **kw):
        self.requests.append({**kw, "messages": list(kw["messages"])})
        msg = self.script[len(self.requests) - 1]
        if isinstance(msg, Exception):
            raise msg

        class Ctx:
            def __enter__(s):
                return SimpleNamespace(get_final_message=lambda: msg)

            def __exit__(s, *a):
                return False

        return Ctx()


def test_full_loop(tmp_path):
    client = FakeClient()
    s = PaintingSession(RunConfig(out_dir=tmp_path, max_actions=20), client=client)
    run_dir = s.run()

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["finished"] and summary["title"] == "Test"
    assert summary["actions"] == 5  # watercolor, bristle, spatter dab, undo, blender
    assert (run_dir / "final.png").exists() and (run_dir / "timelapse.gif").exists()

    # one call per turn, asked of the API and enforced locally
    assert client.requests[0]["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert [t["name"] for t in client.requests[0]["tools"]] == ["brush", "look", "undo", "finish"]
    first = client.requests[1]["messages"][-1]["content"]
    assert first[1]["is_error"] and "one stroke at a time" in first[1]["content"]
    assert [b["type"] for b in first[0]["content"]] == ["text", "image", "text"]  # view after the stroke
    assert first[0]["content"][-1]["text"] == "19 strokes left."

    def result(n):
        return client.requests[n]["messages"][-1]["content"][0]

    assert result(2)["is_error"]  # unknown brush
    assert result(3)["content"][0]["type"] == "image"  # look
    assert result(4)["is_error"] and "at most 410 px" in result(4)["content"]
    assert result(5)["is_error"] and "no `fill` tool" in result(5)["content"]
    # nudge after a turn with no tool call
    assert client.requests[10]["messages"][-1]["content"].startswith("(The canvas")
    # replay reproduces the final image exactly
    from PIL import Image, ImageChops
    rp = replay(run_dir)
    assert ImageChops.difference(Image.open(rp), Image.open(run_dir / "final.png")).getbbox() is None
    assert "thinking" in (run_dir / "journal.md").read_text()


def test_inline_images_without_optional_features(tmp_path):
    client = FakeClient()
    cfg = RunConfig(out_dir=tmp_path, max_actions=10, model="claude-haiku-5-5", inline_images=True,
                    context_editing=False, thinking_summaries=False, view_size=768)
    PaintingSession(cfg, client=client).run()
    req = client.requests[1]
    assert "context_management" not in req and "thinking" not in req and "betas" not in req
    view = req["messages"][-1]["content"][0]["content"][1]
    assert view["source"]["type"] == "base64"
    assert client.n == 0  # nothing went through the Files API


def test_no_undo_no_look(tmp_path):
    client = FakeClient([SCRIPT[-1]])
    cfg = RunConfig(out_dir=tmp_path, allow_undo=False, allow_look=False, max_stroke_length=200)
    PaintingSession(cfg, client=client).run()
    tools = client.requests[0]["tools"]
    assert [t["name"] for t in tools] == ["brush", "finish"]
    assert "at most 200 px" in tools[0]["description"]


def test_empty_refusals_are_resent(tmp_path, monkeypatch):
    monkeypatch.setattr("painterly.agent.time.sleep", lambda s: None)
    empty = BetaMessage.model_validate({
        "id": "msg_r", "type": "message", "role": "assistant", "model": "x", "content": [],
        "stop_reason": "refusal", "stop_sequence": None, "usage": {"input_tokens": 400, "output_tokens": 0},
    })
    client = FakeClient([empty, empty, SCRIPT[0], empty, SCRIPT[-1]])
    run_dir = PaintingSession(RunConfig(out_dir=tmp_path, max_actions=10), client=client).run()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["finished"] and summary["actions"] == 1
    assert summary["empty_refusal_retries"] == 3
    # a resend is the identical request
    assert client.requests[0]["messages"] == client.requests[2]["messages"]


def test_mid_stream_unavailable_is_resent(tmp_path, monkeypatch):
    import anthropic
    import httpx2 as httpx

    monkeypatch.setattr("painterly.agent.time.sleep", lambda s: None)
    req = httpx.Request("POST", "https://example.invalid/v1/messages")
    err = anthropic.APIStatusError(
        "The requested model is temporarily unavailable. Please try again later.",
        response=httpx.Response(200, request=req), body=None)
    client = FakeClient([SCRIPT[0], err, SCRIPT[-1]])
    run_dir = PaintingSession(RunConfig(out_dir=tmp_path), client=client).run()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["finished"] and summary["transient_error_retries"] == 1
    assert client.requests[1]["messages"] == client.requests[2]["messages"]
