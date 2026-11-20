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


SCRIPT = [
    _msg([
        {"type": "thinking", "thinking": "A wide quiet field, then one line.", "signature": "s"},
        {"type": "text", "text": "Starting with a wash."},
        {"type": "tool_use", "id": "t1", "name": "fill", "input": {"color": "#d8e2dc", "color_to": "#5b6c8f", "angle": 90}},
        {"type": "tool_use", "id": "t2", "name": "stroke", "input": {"brush": "bristle", "points": [[100, 600], [500, 560], [900, 640]], "color": "#2d1e2f", "size": 40, "fade": 0.8}},
    ]),
    _msg([
        {"type": "tool_use", "id": "t3", "name": "stroke", "input": {"brush": "nope", "points": [[1, 1]], "color": "#000", "size": 3}},
        {"type": "tool_use", "id": "t4", "name": "look", "input": {"x": 0, "y": 500, "width": 300, "height": 200}},
    ]),
    _msg([
        {"type": "tool_use", "id": "t5", "name": "stroke", "input": {"brush": "spatter", "points": [[400, 300], [600, 320]], "color": "#c44536", "size": 120, "jitter": 0.5}},
        {"type": "tool_use", "id": "t6", "name": "undo", "input": {}},
        {"type": "tool_use", "id": "t7", "name": "stroke", "input": {"brush": "watercolor", "points": [[300, 300], [700, 280]], "color": "#c44536", "size": 90, "opacity": 0.7}},
    ]),
    _msg([{"type": "text", "text": "Hm."}], stop="end_turn"),
    _msg([{"type": "tool_use", "id": "t8", "name": "finish", "input": {"title": "Test", "note": "fine"}}]),
]


class FakeClient:
    def __init__(self):
        self.requests = []
        self.n = 0
        self.files = SimpleNamespace(upload=self._upload)
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _upload(self, file, **kw):
        self.n += 1
        return SimpleNamespace(id=f"file_{self.n}")

    def _stream(self, **kw):
        self.requests.append({**kw, "messages": list(kw["messages"])})
        msg = SCRIPT[len(self.requests) - 1]

        class Ctx:
            def __enter__(s):
                return SimpleNamespace(get_final_message=lambda: msg)

            def __exit__(s, *a):
                return False

        return Ctx()


def test_full_loop(tmp_path):
    client = FakeClient()
    s = PaintingSession(RunConfig(out_dir=tmp_path, max_actions=10), client=client)
    run_dir = s.run()

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["finished"] and summary["title"] == "Test"
    assert summary["actions"] == 5  # fill, stroke, spatter, undo, watercolor
    assert (run_dir / "final.png").exists() and (run_dir / "timelapse.gif").exists()

    # bad brush came back as an error result; look returned an image
    second_results = client.requests[2]["messages"][-1]["content"]
    assert second_results[0]["is_error"] is True
    assert second_results[1]["content"][0]["type"] == "image"
    # the canvas view is attached once per batch, on the last result
    first_results = client.requests[1]["messages"][-1]["content"]
    assert [b["type"] for b in first_results[-1]["content"]] == ["text", "image", "text"]
    # nudge after a turn with no tool call
    assert client.requests[4]["messages"][-1]["content"].startswith("(The canvas")
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
    view = req["messages"][-1]["content"][-1]["content"][1]
    assert view["source"]["type"] == "base64"
    assert client.n == 0  # nothing went through the Files API
