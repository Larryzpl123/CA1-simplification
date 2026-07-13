#!/usr/bin/env python3
"""
progress.py — a heartbeat, not a progress bar.
Larry (Peilin) Zhong

WHY NOT tqdm
------------
A tqdm-style bar redraws one line with carriage returns. That is fine on a
terminal and it is garbage in a log file, and every run in this project is
piped into a log that gets read, archived and pasted. A 60-minute job that
writes 4000 lines of "\r  37%|####      |" is not a log, it is a hazard.

What actually went wrong was subtler and worth recording. The runner piped
Python through `grep`, and grep BLOCK-buffers when its stdout is a pipe. Every
flush=True in the Python source was therefore swallowed: the job printed nothing
for its entire 80-minute runtime. A long job that prints nothing is
indistinguishable from a hung one, and the only honest way to tell them apart
was `ps aux`.

(The fix for that was `grep --line-buffered` in run_all.sh. This module is the
other half: something worth printing.)

WHAT A HEARTBEAT MUST ANSWER
----------------------------
    1. is it alive?          -> a line appeared
    2. how far in?           -> done / total
    3. when will it finish?  -> ETA from measured throughput, not a guess

Plain lines. Append-only. Safe to tee, grep, and paste.

USAGE
-----
    from progress import Heartbeat
    hb = Heartbeat(total=len(TAU_GRID) * len(SEEDS), label="PING sweep")
    for tg in TAU_GRID:
        for sd in SEEDS:
            ...run one simulation...
            hb.tick(f"tau={tg}ms seed={sd}")
    hb.done()

Output:
    [PING sweep]   3/80   4%  elapsed 0:02:31  eta 0:59:12  (50.3 s/it)  tau=2ms seed=202
"""

import sys
import time


def _hms(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


class Heartbeat:
    """Append-only progress lines with a measured ETA.

    every: print at most one line per `every` ticks (1 = every tick).
    min_interval_s: never print more often than this, whatever `every` says.
                    Keeps a fast inner loop from flooding the log.
    """

    def __init__(self, total, label="progress", every=1, min_interval_s=0.0,
                 stream=None):
        self.total = int(total)
        self.label = label
        self.every = max(1, int(every))
        self.min_interval_s = float(min_interval_s)
        self.stream = stream or sys.stdout
        self.n = 0
        self.t0 = time.time()
        self._last_print = 0.0
        self._emit(f"[{self.label}] starting: {self.total} items")

    def _emit(self, line):
        self.stream.write(line + "\n")
        self.stream.flush()          # the whole point; see the note above

    def tick(self, note=""):
        self.n += 1
        now = time.time()
        due = (self.n % self.every == 0) or (self.n == self.total)
        if not due:
            return
        if (now - self._last_print) < self.min_interval_s and self.n != self.total:
            return
        self._last_print = now

        elapsed = now - self.t0
        per_item = elapsed / max(self.n, 1)
        remaining = per_item * (self.total - self.n)
        pct = 100.0 * self.n / max(self.total, 1)

        self._emit(
            f"[{self.label}] {self.n:>4}/{self.total:<4} {pct:>3.0f}%  "
            f"elapsed {_hms(elapsed)}  eta {_hms(remaining)}  "
            f"({per_item:.1f} s/it)  {note}"
        )

    def done(self, note=""):
        elapsed = time.time() - self.t0
        self._emit(
            f"[{self.label}] finished {self.n}/{self.total} in {_hms(elapsed)}"
            f"  ({elapsed / max(self.n, 1):.1f} s/it)  {note}"
        )


if __name__ == "__main__":
    # A 12-item job that takes ~0.15 s each. The ETA should converge quickly.
    hb = Heartbeat(total=12, label="demo")
    for i in range(12):
        time.sleep(0.15)
        hb.tick(f"item {i}")
    hb.done()
    print()
    print("Note the ETA: it is measured from actual throughput, not assumed.")
    print("Also note there are no carriage returns. This is safe to tee to a log.")
