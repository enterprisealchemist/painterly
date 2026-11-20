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
    args = {"brush": "bristle", "points": [[100, 600], [250, 540], [400, 560]], "pressure": [0.3, 1, 0.4],
            "color": "#2d1e2f", "size": 40, **kw}
    return {"type": "tool_use", "id": f"t{i}", "name": "brush", "input": args}


SCRIPT = [
    _msg([
        {"type": "thinking", "thinking": "A wide quiet field, then one line.", "signature": "s"},
        {"type": "text", "text": "Starting with a wash."},
        _brush(1, brush="watercolor", points=[[50, 200], [420, 200]], color="#5b6c8f", size=150),
        _brush(2),  # a second call in the same turn is refused
    ]),
    _msg([_brush(3, brush="nope")]),  # unknown brush -> error
    _msg([{"type": "tool_use", "id": "t4", "name": "look", "input": {"x": 0, "y": 500, "width": 300, "height": 200}}]),
    _msg([_brush(5, points=[[0, 0]] * 8)]),  # too many points -> error
    _msg([{"type": "tool_use", "id": "t6", "name": "fill", "input": {"color": "#000"}}]),  # not offered
    _msg([_brush(7)]),
    _msg([_brush(8, brush="spatter", points=[[500, 300]], pressure=[1], color="#c44536", size=120)]),
    _msg([{"type": "tool_use", "id": "t9", "name": "undo", "input": {}}]),
    _msg([_brush(10, brush="blender", points=[[100, 150], [400, 150], [400, 450], [100, 450]], smooth=False, closed=True,
                 pressure=[0.8], size=60)]),
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
    s = PaintingSession(RunConfig(open_ended=False, out_dir=tmp_path, max_actions=20, allow_undo=True), client=client)
    run_dir = s.run()

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["finished"] and summary["title"] == "Test"
    assert 0 < summary["coverage"] < 1
    assert summary["actions"] == 5  # watercolor, bristle, spatter dab, undo, blender
    assert (run_dir / "final.png").exists() and (run_dir / "timelapse.gif").exists()

    # one call per turn, asked of the API and enforced locally
    assert client.requests[0]["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert [t["name"] for t in client.requests[0]["tools"]] == ["brush", "look", "undo", "show", "finish"]
    first = client.requests[1]["messages"][-1]["content"]
    assert first[1]["is_error"] and "only the first call" in first[1]["content"]
    assert [b["type"] for b in first[0]["content"]] == ["text", "image"]  # view after the stroke, no countdown

    def result(n):
        return client.requests[n]["messages"][-1]["content"][0]

    assert result(2)["is_error"]  # unknown brush
    assert result(3)["content"][0]["type"] == "image"  # look
    assert result(4)["is_error"] and "1 to 7" in result(4)["content"]
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
    cfg = RunConfig(open_ended=False, out_dir=tmp_path, max_actions=10, model="claude-haiku-5-5", inline_images=True,
                    context_editing=False, thinking_summaries=False, view_size=768)
    PaintingSession(cfg, client=client).run()
    req = client.requests[1]
    assert "context_management" not in req and "thinking" not in req and "betas" not in req
    view = req["messages"][-1]["content"][0]["content"][1]
    assert view["source"]["type"] == "base64"
    assert client.n == 0  # nothing went through the Files API


def test_no_undo_no_look(tmp_path):
    client = FakeClient([SCRIPT[-1]])
    cfg = RunConfig(open_ended=False, out_dir=tmp_path, allow_look=False)  # undo is off by default
    PaintingSession(cfg, client=client).run()
    tools = client.requests[0]["tools"]
    assert [t["name"] for t in tools] == ["brush", "show", "finish"]
    assert "px long" not in tools[0]["description"]  # no length limit by default
    assert tools[0]["input_schema"]["properties"]["points"]["maxItems"] == 7


def test_empty_refusals_are_resent(tmp_path, monkeypatch):
    monkeypatch.setattr("painterly.agent.time.sleep", lambda s: None)
    empty = BetaMessage.model_validate({
        "id": "msg_r", "type": "message", "role": "assistant", "model": "x", "content": [],
        "stop_reason": "refusal", "stop_sequence": None, "usage": {"input_tokens": 400, "output_tokens": 0},
    })
    client = FakeClient([empty, empty, SCRIPT[0], empty, SCRIPT[-1]])
    run_dir = PaintingSession(RunConfig(open_ended=False, out_dir=tmp_path, max_actions=10), client=client).run()
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
    run_dir = PaintingSession(RunConfig(open_ended=False, out_dir=tmp_path), client=client).run()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["finished"] and summary["transient_error_retries"] == 1
    assert client.requests[1]["messages"] == client.requests[2]["messages"]


def _status_error(msg, status=400):
    import anthropic
    import httpx2 as httpx

    req = httpx.Request("POST", "https://example.invalid/v1/messages")
    return anthropic.APIStatusError(msg, response=httpx.Response(status, request=req), body=None)


RESUME_SCRIPT = [
    _msg([{"type": "thinking", "thinking": "first", "signature": "s1"}, _brush(1, brush="watercolor", color="#5b6c8f")]),
    _msg([_brush(2, points=[[300, 300], [400, 250], [500, 420]], smooth=True)]),
    _msg([{"type": "text", "text": "Pausing to think."}], stop="end_turn"),
    _msg([_brush(3, brush="ink", points=[[700, 600], [800, 700], [700, 800], [600, 700]], closed=True,
                 pressure=[0.1, 1, 0.2], color="#111")]),
    _msg([_brush(4, brush="spatter", points=[[200, 800], [260, 780]], color="#c44536", size=60)]),
    _msg([{"type": "tool_use", "id": "t9", "name": "finish", "input": {"title": "Resumed"}}]),
]


def test_resume_matches_an_uninterrupted_run(tmp_path, monkeypatch):
    from PIL import Image, ImageChops

    monkeypatch.setattr("painterly.agent.time.sleep", lambda s: None)
    cfg = lambda d: RunConfig(open_ended=False, out_dir=d, inline_images=True, seed=4)  # noqa: E731

    straight = FakeClient(RESUME_SCRIPT)
    ref_dir = PaintingSession(cfg(tmp_path / "a"), client=straight).run()

    # Same run, but the API fails for good on the 5th request.
    broken = FakeClient(RESUME_SCRIPT[:4] + [_status_error("bad request")])
    run_dir = PaintingSession(cfg(tmp_path / "b"), client=broken).run()
    assert not json.loads((run_dir / "summary.json").read_text())["finished"]

    rest = FakeClient(RESUME_SCRIPT[4:])
    resumed = PaintingSession.resume(run_dir, client=rest)
    resumed.run()

    # the model got back exactly the conversation it would have had
    from painterly.agent import _block_dict

    def wire(msgs):
        return [m if isinstance(m["content"], str) else {**m, "content": [_block_dict(b) for b in m["content"]]} for m in msgs]

    assert wire(rest.requests[0]["messages"]) == wire(straight.requests[4]["messages"])
    assert rest.requests[0]["system"] == straight.requests[4]["system"]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["finished"] and summary["title"] == "Resumed" and summary["actions"] == 4
    a, b = Image.open(ref_dir / "final.png"), Image.open(run_dir / "final.png")
    assert ImageChops.difference(a, b).getbbox() is None
    assert ImageChops.difference(Image.open(replay(run_dir)), b).getbbox() is None
    import pytest
    with pytest.raises(SystemExit):
        PaintingSession.resume(run_dir, client=FakeClient([]))


def test_wobble_is_small_and_keeps_the_ends():
    import numpy as np
    from painterly.canvas import arc_path, hand_wobble

    path, length = arc_path([100, 100], [500, 100], 0.0, 2.0)
    w = hand_wobble(path, length, np.random.default_rng(1))
    assert np.allclose(w[0], path[0]) and np.allclose(w[-1], path[-1])
    off = np.abs(w[:, 1] - 100)
    assert 0.5 < off.max() <= 5.0


def test_node_paths():
    import numpy as np
    from painterly.canvas import node_path

    sq, n = node_path([[0, 0], [100, 0], [100, 100], [0, 100]], smooth=False, closed=True, step=2)
    assert abs(n - 400) < 1e-6 and np.allclose(sq[0], sq[-1])
    ring, n = node_path([[100, 0], [0, 100], [-100, 0], [0, -100]], smooth=True, closed=True, step=2)
    r = np.linalg.norm(ring, axis=1)
    assert np.allclose(ring[0], ring[-1]) and r.min() > 85 and r.max() < 115 and 560 < n < 700  # close to a circle
    dab, n = node_path([[5, 5]])
    assert n == 0 and len(dab) == 1


def test_length_limit_when_set(tmp_path):
    client = FakeClient([_msg([_brush(1, points=[[0, 0], [900, 900]])]), SCRIPT[-1]])
    PaintingSession(RunConfig(open_ended=False, out_dir=tmp_path, max_stroke_length=410), client=client).run()
    res = client.requests[1]["messages"][-1]["content"][0]
    assert res["is_error"] and "at most 410 px" in res["content"]


def test_nothing_mentions_a_budget(tmp_path):
    client = FakeClient([SCRIPT[0], SCRIPT[-1]])
    PaintingSession(RunConfig(open_ended=False, out_dir=tmp_path), client=client).run()
    sent = json.dumps([client.requests[1]["system"], client.requests[1]["tools"],
                       client.requests[1]["messages"]], default=str).lower()
    for word in ("300", "strokes left", "budget", "remaining", "up to 300"):
        assert word not in sent, word


def test_size_is_the_width_at_full_pressure():
    import numpy as np
    from painterly.canvas import Canvas
    from painterly.tools import Limits, apply_tool

    for brush in ("round", "ink", "bristle", "flat", "knife", "charcoal"):
        for pts in ([[50, 200], [350, 200]], [[200, 200]]):
            c = Canvas(400, 400, seed=2)
            apply_tool(c, "brush", {"brush": brush, "points": pts, "pressure": [1], "color": "#000000", "size": 100},
                       3, Limits(None, 154, 0.0))
            cov = 1 - c.rgb.mean(axis=2) / c.paper.mean()
            core = cov[195:205, 195:205].mean()
            width = (cov[:, 195:205].mean(axis=1) > core / 2).sum()
            assert 92 <= width <= 108, (brush, len(pts), width)


def test_pigment_mixing_makes_green():
    from painterly.canvas import hex_to_rgb
    from painterly.pigment import mix

    g = mix(hex_to_rgb("#1d4ed8"), hex_to_rgb("#facc15"), 0.5)
    assert g[1] > g[0] + 0.1 and g[1] > g[2] + 0.05  # green, not grey
    for c in ("#1d4ed8", "#f4f1ea", "#000000"):
        assert abs(mix(hex_to_rgb(c), hex_to_rgb(c), 0.3) - hex_to_rgb(c)).max() < 2 / 255


def test_wet_brush_drags_paint_and_angle_shapes_the_mark():
    import numpy as np
    from painterly.canvas import Canvas
    from painterly.tools import Limits, apply_tool

    L = Limits(None, 154, 0.0)
    c = Canvas(400, 200, seed=1, mixing="pigment")
    apply_tool(c, "brush", {"brush": "flat", "points": [[60, 20], [60, 180]], "pressure": [1], "color": "#1f4fbf", "size": 60}, 1, L)
    dry, wet = c.rgb.copy(), None
    apply_tool(c, "brush", {"brush": "round", "points": [[40, 100], [360, 100]], "pressure": [1], "color": "#f2c014",
                            "size": 40, "wet": 0.9}, 2, L)
    wet = c.rgb
    # just past the blue stripe the wet yellow stroke carries blue: bluer than plain yellow
    assert wet[100, 140, 2] > 0.25 and wet[100, 140, 2] > wet[100, 350, 2] + 0.1
    # the carried paint thins out with distance, back toward the brush's own colour
    yellow = np.array([0xf2, 0xc0, 0x14]) / 255
    assert abs(wet[100, 350] - yellow).max() < abs(wet[100, 140] - yellow).max() / 3
    assert np.allclose(dry[20:60, 200:], wet[20:60, 200:])  # nothing outside the stroke changed

    def width(angle, pts):
        c = Canvas(400, 400, seed=1)
        apply_tool(c, "brush", {"brush": "flat", "points": pts, "pressure": [1], "color": "#000000", "size": 100,
                                "angle": angle}, 1, L)
        cov = (1 - c.rgb.mean(axis=2) / c.paper.mean()) > 0.5
        return cov[:, 195:205].any(axis=1).sum() if pts[0][1] == pts[1][1] else cov[195:205].any(axis=0).sum()

    across = width(90, [[50, 200], [350, 200]])  # vertical edge moving sideways: full width
    along = width(0, [[50, 200], [350, 200]])  # horizontal edge moving sideways: thin
    assert across > 90 and along < 20


def test_open_ended_has_no_finish_and_stops_quietly_at_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr("painterly.agent.time.sleep", lambda s: None)
    strokes = [_msg([_brush(i, points=[[50 + 40 * i, 100], [50 + 40 * i, 400]], size=20)]) for i in range(1, 7)]
    client = FakeClient(strokes[:3])
    run_dir = PaintingSession(RunConfig(out_dir=tmp_path, max_actions=3, inline_images=True), client=client).run()
    req = client.requests[0]
    assert [t["name"] for t in req["tools"]] == ["brush", "look", "show"]
    assert "finish" not in req["system"] and "indefinitely" in req["system"]
    sent = json.dumps([r["messages"] for r in client.requests], default=str).lower()
    assert "finish" not in sent and "left" not in sent
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["actions"] == 3 and not summary["finished"] and summary["open_ended"]
    assert len(client.requests) == 3  # stopped right after the third stroke, without telling it

    # carrying on: the model simply gets its next turn
    more = FakeClient(strokes[3:])
    PaintingSession.resume(run_dir, client=more, max_actions=6).run()
    assert json.loads((run_dir / "summary.json").read_text())["actions"] == 6
    assert len(more.requests[0]["messages"]) == len(client.requests[2]["messages"]) + 2


def test_open_ended_nudges_then_stops_when_it_stops_painting(tmp_path):
    quiet = _msg([{"type": "text", "text": "Looking."}], stop="end_turn")
    client = FakeClient([SCRIPT[0], quiet, _msg([_brush(9)]), quiet, quiet, quiet])
    run_dir = PaintingSession(RunConfig(out_dir=tmp_path), client=client).run()
    nudges = [m for r in client.requests for m in r["messages"][-1:] if m["content"] == "(The painting continues.)"]
    assert nudges  # neutral nudge, no mention of finishing
    assert json.loads((run_dir / "summary.json").read_text())["actions"] == 2
    assert len(client.requests) == 6  # one pause forgiven; then two nudges in a row and it ends


def test_show_asks_a_separate_critic(tmp_path):
    critique = _msg([{"type": "thinking", "thinking": "critic thoughts", "signature": "c"},
                     {"type": "text", "text": "The blue diagonal anchors it; the lower right feels empty."}], stop="end_turn")
    critique2 = _msg([{"type": "text", "text": "Better: the lower right now answers the diagonal."}], stop="end_turn")
    show = lambda i, m: _msg([{"type": "tool_use", "id": f"s{i}", "name": "show", "input": {"message": m}}])  # noqa: E731
    script = [_msg([_brush(1)]), show(1, "Does this hold together?"), critique,
              _msg([_brush(2, points=[[700, 800], [900, 900]])]), show(2, ""), critique2,
              _msg([_brush(3)])]
    client = FakeClient(script)
    run_dir = PaintingSession(RunConfig(out_dir=tmp_path, max_actions=3, inline_images=True), client=client).run()

    critic_req = client.requests[2]
    assert "tools" not in critic_req and "critic" in critic_req["system"]
    content = critic_req["messages"][0]["content"]
    assert content[0]["type"] == "image" and 'Does this hold together?' in content[1]["text"]
    assert "Starting with a wash" not in json.dumps(critic_req["messages"], default=str)  # never sees the painter's words to itself
    # the painter gets the critique as the result of its show call, on its next turn
    back = client.requests[3]["messages"][-1]["content"][0]
    assert back["tool_use_id"] == "s1" and "lower right feels empty" in back["content"]
    # second critique remembers the first
    second = json.dumps(client.requests[5]["messages"], default=str)
    assert "lower right feels empty" in second and "doesn't say anything" in second
    assert json.loads((run_dir / "summary.json").read_text())["actions"] == 3
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert [e["stroke"] for e in events if e["kind"] == "critique"] == [1, 2]
