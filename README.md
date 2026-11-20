# painterly

A blank canvas for an AI agent. The model paints the way a person does: one brushstroke
at a time, looking at the canvas after each one. It isn't given a subject, a style or a goal, and it
decides for itself when the work is finished. The question is what it makes when the
choice is entirely its own.

![every brush: a pressure swell, a wave, a square and a ring](docs/brush_sampler.png)

## Quick start

```bash
pip install -e .            # or: pip install -r requirements.txt
export ANTHROPIC_API_KEY=... # or `ant auth login`

python -m painterly run                     # one session (safety cap: 300 strokes, never shown to the model)
python -m painterly run --runs 5 --seed 10  # five independent sessions
python -m painterly sampler -o brushes.png  # brush reference sheet, no API calls
python -m painterly resume runs/<id>        # carry on a run that stopped early
python -m painterly viewer runs/<id>        # HTML page to step through it, stroke by stroke
python -m painterly replay runs/<id>        # re-render a run from its log
```

Runs are resumable. After every turn the conversation (`conversation.jsonl`, with inline
canvas views under `views/`) and the loop state (`state.json`) are saved. If a run stops
because of an API error, Ctrl-C, a crash, or the model ending its turn without finishing,
`resume` picks it up:
- The canvas is rebuilt exactly from the strokes made so far.
- The model gets back exactly the conversation it had.
- A turn interrupted halfway is redone.

With the Files API, uploaded canvas views expire after two days, so resume within that
window; `--inline-images` runs have no such limit.

### Through a proxy or reseller

Some providers don't pass through every Anthropic feature. Probe yours first; it makes a
handful of tiny requests and tells you which flags you need:

```bash
python -m painterly check --base-url https://api.example.com --api-key-env MY_KEY_VAR
python -m painterly run --model claude-haiku-5-5 --base-url ... --api-key-env MY_KEY_VAR \
    --max-actions 5                # smoke test
    # plus, if `check` says so: --inline-images --no-context-editing --no-thinking-summaries
```

Some providers spread requests across several backends, and a bad backend may return an
empty `refusal` (no content, zero output tokens) even for harmless requests. Requests that
come back that way are resent automatically (`empty_refusal_retries` in `summary.json`).
A real refusal from the model is not resent. On CometAPI with Haiku 5.5 (tested Nov 2026)
everything works except the Files API, so add `--inline-images`.

Each session writes to `runs/<timestamp>/`:

| file | what |
|---|---|
| `final.png` | the canvas at the end |
| `timelapse.gif` | every action, in order |
| `frames/` | full-resolution canvas after each action |
| `journal.md` | the model's (summarised) thinking, anything it said, and each action |
| `events.jsonl` | every tool call with its exact input and random seed (used by `replay`) |
| `summary.json` | title and note (if it gave any), action count, token use, estimated cost |
| `config.json` | settings plus the exact system prompt the model received |

## The loop

```
system prompt (no brief)  ─▶  model  ─▶ tool calls ─▶ canvas engine
        ▲                                                  │
        └──────── tool results + current canvas image ◀────┘
```

The run ends when the model calls `finish`, uses up its action budget, or ends a turn
without acting twice in a row (it gets one neutral nudge first).

### Tools

| tool | does |
|---|---|
| `brush` | one brushstroke: touch down, move, lift off |
| `look` | lean in and look closely at a region (doesn't change the canvas; `--no-look` removes it) |
| `undo` | off by default, so every stroke is final; `--undo` adds it |
| `finish` | end the session, with an optional title and note |

There's no fill. Everything is made of brushstrokes. A `brush` call takes:

| parameter | |
|---|---|
| `brush` | which brush (below) |
| `points` | 1–7 points the brush passes through, in order: touch down at the first, lift off at the last. One point is a dab |
| `smooth` | `true` (default): a flowing curve through the points; `false`: straight lines with sharp corners |
| `closed` | `true`: carry on back to the first point, closing a loop (a ring, a square outline) |
| `pressure` | 1–12 values spaced evenly along the stroke, eased smoothly between. Sets the width and how much paint goes down. `[0.2, 1, 0.3]` presses in and lifts off |
| `size` | width at full pressure |
| `color` | hex |
| `load` | optional: how much paint is on the brush; less, and it runs dry toward the end |
| `wet` | optional: how much paint already on the canvas the brush picks up and drags along, mixed into its own colour |
| `angle` | optional, for `flat`, `ink` and `knife`: hold the tip's broad edge at a fixed angle, so the mark goes thick and thin with direction like a calligraphy nib |
| `dilution` | optional: how thinned the paint is (more transparent) |

**Limits.**
- **Turns.** The model makes one call per turn. This is asked of the API with
  `disable_parallel_tool_use` and also enforced by the loop, and it sees the canvas after
  every stroke.
- **Brush size.** The widest brush is 15% of the canvas's longer side (154 px on 1024).
- **Stroke length.** There's no limit by default; `--max-stroke-length` sets one.
- **Errors.** A call that breaks a limit is refused with an explanation, and nothing is
  painted.

### Brushes

`round`, `ink`, `bristle`, `flat`, `knife`, `watercolor`, `airbrush`, `charcoal`, `spatter`,
and `blender`, a clean dry brush that softens the paint it passes over. Each row of the
sheet above shows one brush. From left to right:
- a straight stroke pressing in and lifting off
- a smooth wave through four points
- a square (closed, sharp corners)
- a ring (closed, smooth), with little paint loaded

Strokes are rendered as continuous sweeps, not rows of stamps. Each pixel near a stroke
knows how far along the stroke it is, how far across, and how much paint is left. Each
brush is a function of those values. Bristle and flat brushes give each bristle its own
lane and its own paint supply, so a dry brush breaks into streaks. Grain-sensitive brushes
(charcoal, knife, watercolor) all read the same paper texture, so their marks line up. To
add a brush, add a `BrushSpec` and one branch in `Canvas._sweep` (`painterly/canvas.py`).

**Paint mixes like pigment.** Overlapping paint mixes subtractively: blue over yellow
gives green, not grey. Each colour becomes a 10-band reflectance spectrum, and the spectra
are mixed by weighted geometric mean, using libmypaint's method and tables (ISC licence,
see `painterly/pigment.py`). `--mixing rgb` switches back to mixing like light. A `wet`
brush carries paint the way MyPaint's smudge and Fresh Paint's pickup map do. The canvas
under the stroke is sampled before the stroke lands, in lanes across it. Along each lane,
the brush keeps a running mix of the paint it has passed over, weighted by how much paint
was there, and blends that into its own colour. The angled nib narrows the stroke's width
by how far the tip's edge is turned toward the direction of travel.
The engine still has free-form paths, polygon fills and region blur. The model can't reach
them; they're kept so older runs can be replayed.

## Design choices that affect the result

The experiment depends heavily on how the setup is framed, so these choices are deliberate
and easy to change:

- **The prompt suggests nothing.** No subject, mood or style, and it doesn't call the
  model an artist. Even naming a brush "watercolor" nudges it, but some vocabulary is
  unavoidable. Use `--prompt my_prompt.txt` to compare framings, and see `config.json` for
  the exact text each run got.
- **An exercise, not a task, painted without intention.** The prompt calls the painting an
  exercise for the model itself, with nothing to deliver. It asks it not to choose a
  subject, but to start with an arbitrary stroke and respond to what's on the canvas
  after each one. It's told that a stroke can be as large or intricate as it likes.
- **Compose something.** Abstraction is welcome, but the prompt asks for expression, with
  the strokes working together, rather than random marks.
- **A coverage expectation, not a target.** The prompt says most of the canvas will
  probably end up covered, over many strokes. An earlier version gave a percentage, and the
  model spent its thinking estimating coverage instead of painting. `summary.json` records
  the actual `coverage`: the fraction of the canvas visibly different from bare paper.
- **Nothing about an audience.** The prompt doesn't say anyone will look at the painting,
  and it doesn't say nobody will.
- **No budget in sight.** The model isn't told how many strokes it has, and there's no
  countdown. A safety cap (`--max-actions`, default 300) exists, and it's only mentioned
  if the model reaches it. Earlier runs that showed the budget used about half of it and
  planned to that size from the first thought.
- **The tools do what they say.** Every brush is `size` wide at full pressure. The
  descriptions match the rendering, including the slight hand wobble, so the model
  doesn't trip over results it didn't ask for. `tests/` checks the widths.
- **Memory is short-term.** Only the last few canvas views stay in context (`--keep-images`,
  via server-side context editing), along with all of its own words and tool calls. Like a
  painter, it sees the canvas as it is now, not every earlier state.
- **It paints like a person.** One stroke per turn and no fills, so it has to build up
  every area mark by mark. A stroke can pass through up to 7 points, enough for a
  gesture, a square or a ring, but not for tracing arbitrary outlines.
- **No undo.** Every stroke is final, as on a real canvas (`--undo` to change that).
- **An unsteady hand.** Each stroke drifts slightly off its ideal path, by about 1% of its
  length and never more than 5 px. The drift is smooth, random and seeded, so replays are
  exact, and it's zero at both ends. `--wobble 0` turns it off.
- **Run it many times.** One painting is an anecdote. With `--runs N` you can look for what
  recurs across sessions: subjects, palettes, compositions, and when it decides to stop.

## Notes

- Defaults to `claude-opus-5-5` at `high` effort, with adaptive thinking shown as summaries
  so the journal shows its reasoning. Use `--model` and `--effort` to compare.
- Server-side refusal fallbacks are on: if a request is declined, the API retries it on
  another model and the event is logged. Turn this off with `--no-fallbacks`.
- Canvas views are uploaded through the Files API and referenced by id, with a two-day
  expiry, so request size stays flat over a long session.
- Cost: each canvas view is roughly 1.4k input tokens, and every stroke is a request. A rough estimate, not yet measured,
  is $3–15 for a 150-stroke session at `high` effort, mostly thinking output and
  resent history. `summary.json` gives the figure for each run, so run one session before
  starting a batch. `--max-actions` and `--effort` are the main controls.
- Tests: `pip install pytest && pytest` runs the whole loop against a scripted stand-in
  for the API.
