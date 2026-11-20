"""Follow a run's events.jsonl and print only what's worth a look, one line per event.

Meant to be piped into a notifier: progress every N strokes, every critique, errors,
loops (several turns in a row without a stroke), stalls, and the end of the session.
It exits when the session ends.

    python -m painterly.watch runs/<id> [--every 10] [--stall 180]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

END_KINDS = {"session_cap", "ended_without_finish", "interrupted", "refusal"}
ERROR_KINDS = {"api_error", "tool_crash", "critic_error", "force_refused"}


def watch(run_dir: Path, every: int = 10, stall: float = 180.0, from_start: bool = False) -> None:
    path = run_dir / "events.jsonl"
    while not path.exists():
        time.sleep(1)
    strokes = reminders = streak = critiques = refused = retries = 0
    last_event = time.time()
    stalled = False

    def say(line: str) -> None:
        print(line, flush=True)

    with open(path, encoding="utf-8") as f:
        if not from_start:  # catch up silently on what's already there
            for line in f:
                ev = json.loads(line)
                k = ev["kind"]
                strokes += k == "tool" and ev.get("tool") in ("brush", "undo")
                reminders += k == "reminder"
                critiques += k == "critique"
        say(f"watching {run_dir.name}: {strokes} strokes so far, {reminders} reminders, {critiques} critiques")
        while True:
            line = f.readline()
            if not line:
                if not stalled and time.time() - last_event > stall:
                    say(f"STALLED: nothing new for {stall:.0f}s (at stroke {strokes})")
                    stalled = True
                time.sleep(1)
                continue
            if not line.endswith("\n"):  # a partly written line; wait for the rest
                time.sleep(0.2)
                line += f.readline()
            ev = json.loads(line)
            last_event, stalled = time.time(), False
            k = ev["kind"]
            if k == "tool" and ev.get("tool") in ("brush", "undo"):
                strokes += 1
                streak = 0
                if strokes % every == 0:
                    say(f"stroke {strokes}: {reminders} reminders, {critiques} critiques, {refused} refused, {retries} retries")
            elif k == "reminder":
                reminders += 1
                streak += 1
                if streak >= 3:
                    say(f"LOOP? {streak} reminders in a row without a stroke (at stroke {strokes})")
            elif k == "critique":
                critiques += 1
                who = "asked" if ev.get("asked", True) else "unasked"
                msg = f' painter: "{ev["message"].strip()[:100]}"' if ev.get("message") else ""
                say(f"critic @{ev.get('stroke')} ({who}){msg} :: {ev.get('text', '').strip()[:220]}")
            elif k == "tool_error":
                refused += 1
            elif k in ("transient_error_retry", "empty_refusal_retry"):
                retries += 1
            elif k in ERROR_KINDS:
                say(f"ERROR {k}: {str(ev.get('message') or ev.get('error'))[:200]}")
            if k in END_KINDS:
                say(f"ENDED ({k}) at stroke {strokes}: {reminders} reminders, {critiques} critiques")
                return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--stall", type=float, default=180.0)
    ap.add_argument("--from-start", action="store_true")
    a = ap.parse_args()
    watch(a.run_dir, a.every, a.stall, a.from_start)


if __name__ == "__main__":
    main()
