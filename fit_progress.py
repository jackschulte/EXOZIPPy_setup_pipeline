#!/usr/bin/env python
"""Summarize the sampling progress in an EXOZIPPy log written by plain `exozippy`.

    python fit_progress.py FITLOG_18604236.out [more logs ...]

numpyro writes one terminal progress bar per chain, so the log is a pile of
carriage returns and cursor-movement codes.  This pulls out each chain's
latest bar and prints a table plus an estimate of when the slowest chain
will finish.  (Fits started with run_exozippy.py print readable progress
themselves.)

The warmup length is taken from the fit's yaml in the same directory when
one matches the iteration count; pass --tune to give it explicitly.
"""

import argparse
import datetime
import glob
import os
import re
import time

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
BAR = re.compile(
    r"Running chain (\d+):\s*(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*"
    r"\[([^<\]]+)<([^,\]]+),\s*([\d.]+|\?)\s*(s/it|it/s)"
)


def _seconds(clock):
    """'34:44:07' or '4:24' -> seconds; None for '?'."""
    if "?" in clock:
        return None
    parts = [int(p) for p in clock.strip().split(":")]
    s = 0
    for p in parts:
        s = 60 * s + p
    return s


def _guess_sampler(logpath, total):
    """(tune, chains) from a yaml next to the log whose tune + draws == total."""
    import yaml

    for path in glob.glob(os.path.join(os.path.dirname(os.path.abspath(logpath)), "*.yaml")):
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:  # not every yaml there is a config
            continue
        s = cfg.get("sampler") if isinstance(cfg, dict) else None
        if isinstance(s, dict) and "draws" in s:
            tune, draws = int(s.get("tune", 2000)), int(s["draws"])
            if tune + draws == total:
                return tune, int(s.get("chains", 4))  # EXOZIPPy's defaults
    return None, None


def summarize(path, tune=None):
    with open(path, "rb") as f:
        text = f.read().decode("utf-8", "replace")
    latest = {}
    for piece in re.split(r"[\r\n]", ANSI.sub("", text)):
        m = BAR.search(piece)
        if m:
            latest[int(m.group(1))] = m
    as_of = os.path.getmtime(path)
    age = time.time() - as_of
    print(f"{path}  (last written {age / 60:.0f} min ago)")
    if not latest:
        print("  no 'Running chain' progress bars yet (still compiling, or not a numpyro fit)\n")
        return

    total = int(next(iter(latest.values())).group(4))
    yaml_tune, n_chains = _guess_sampler(path, total)
    tune = tune if tune is not None else yaml_tune
    phase_note = f", warmup {tune}" if tune is not None else ""
    print(f"  {total} iterations per chain{phase_note}")
    print(f"  {'chain':>5} {'iter':>11} {'%':>4}  {'phase':6} {'elapsed':>10} {'s/it':>7} {'left':>10}")
    # Each chain's last bar was drawn at a different time, but every bar's
    # elapsed clock started together; the most advanced clock was drawn
    # about when the file was last written, which pins the shared start.
    elapsed = {c: _seconds(m.group(5)) or 0 for c, m in latest.items()}
    t_start = as_of - max(elapsed.values())
    finish = {}
    for c in sorted(latest):
        m = latest[c]
        done = int(m.group(3))
        rate = m.group(7)
        if rate != "?" and m.group(8) == "it/s":
            rate = f"{1 / float(rate):.2f}"
        left = _seconds(m.group(6))
        if left is not None and done < total:
            finish[c] = t_start + elapsed[c] + left
        phase = "" if tune is None else ("warmup" if done < tune else "draws")
        if done >= total:
            phase = "done"
        print(f"  {c:5d} {done:>5}/{total:<5} {m.group(2):>3}%  {phase:6} {m.group(5).strip():>10} "
              f"{rate:>7} {m.group(6).strip():>10}")

    missing = sorted(set(range(n_chains or 0)) - set(latest))
    if missing:
        which = (f"chain {missing[0]} is" if len(missing) == 1
                 else f"chains {', '.join(map(str, missing))} are")
        print(f"  {which} not in the log at all "
              f"(tqdm hides bars past ~20 rows, or not started yet)")
    elif "(more hidden)" in text:
        print("  note: tqdm hid some chains' bars ('... (more hidden) ...'); "
              "chains not listed above are not in the log")
    if finish:
        c = max(finish, key=finish.get)
        n_left = sum(1 for m in latest.values() if int(m.group(3)) < total)
        print(f"  {n_left} listed chain(s) still running; the last to finish should be "
              f"chain {c}, ~{(finish[c] - time.time()) / 3600:.1f} h from now, about "
              f"{datetime.datetime.fromtimestamp(finish[c]):%Y-%m-%d %H:%M} "
              f"(tqdm's own estimate)")
    elif all(int(m.group(3)) >= total for m in latest.values()):
        print("  every listed chain has finished sampling")
    print()


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("logs", nargs="+", help="FITLOG_*.out files")
    p.add_argument("--tune", type=int, default=None, help="warmup iterations per chain")
    args = p.parse_args(argv)
    for path in args.logs:
        summarize(path, args.tune)


if __name__ == "__main__":
    main()
