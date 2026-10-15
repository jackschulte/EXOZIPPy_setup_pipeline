#!/usr/bin/env python
"""Run an EXOZIPPy fit with log-friendly sampling progress.

    python run_exozippy.py toi7792.yaml [--report-minutes 30] [--steps 200]

This does exactly what `exozippy toi7792.yaml` does, except for how numpyro's
sampling progress is written.  numpyro draws one tqdm bar per chain, built
for a terminal: in a SLURM log they become carriage returns and cursor-up
escape codes, bars past the ~20th row turn into "... (more hidden) ...", and
each chain reports only every 5% of its iterations (hours apart on a slow
fit).  Here those bars are replaced by plain timestamped lines:

  [2026-10-09 03:12] sampling 2h41m | warmup: 12 chains, draws: 8 chains, done: 0/20
      progress: min 38% (chain 4), median 55%, max 72% (chain 14)
      slowest: chain 4 at 55.7 s/it, ~1d22h left -> about 2026-10-11 01:36
      % by chain: 0:41 1:55 2:38 ...

one report every --report-minutes, a line as each chain finishes, and a
per-chain timing table at the end.  Progress is updated every 1/--steps of
each chain's iterations (numpyro's own is 1/20).

Only numpyro's multi-chain progress is changed; every other sampler, and
everything else EXOZIPPy prints, is untouched.
"""

import argparse
import collections
import datetime
import os
import re
import statistics
import sys
import threading
import time

# pymc.sampling.jax sets this when EXOZIPPy imports it, and it must be set
# before JAX starts its backend -- which importing numpyro below could
# trigger.  Same value pymc uses: up to 100 CPU devices, so parallel chains
# each get their own.
_flags = re.sub(r"--xla_force_host_platform_device_count=\S+", "",
                os.environ.get("XLA_FLAGS", "")).split()
os.environ["XLA_FLAGS"] = " ".join(["--xla_force_host_platform_device_count=100", *_flags])


def _stamp(t=None):
    return datetime.datetime.fromtimestamp(t or time.time()).strftime("%Y-%m-%d %H:%M")


def _dur(seconds):
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d{h:02d}h"
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


class ProgressReport:
    """Collects per-chain progress from numpyro's callbacks and logs it."""

    def __init__(self, n_iter, n_chains, tune, report_every):
        self.n_iter, self.n_chains, self.tune = n_iter, n_chains, tune
        self.report_every = report_every
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.t_start = None  # first iteration of the first chain (compile done)
        self.done = [0] * n_chains
        self.started = [None] * n_chains
        self.t_tune = [None] * n_chains  # when warmup ended
        self.finished = [None] * n_chains
        self.history = [collections.deque(maxlen=6) for _ in range(n_chains)]
        self.last_report = 0.0
        self._print(f"compiling the sampler for {n_chains} chains "
                    f"({tune} warmup + {n_iter - tune} draws each) ...")

    def _print(self, msg):
        print(f"[{_stamp()}] {msg}", flush=True)

    def bar(self, iterable=None, position=0, **kwargs):
        return _Bar(self, position)

    # -- callbacks (numpyro calls these from its own threads) ----------------

    def update(self, chain, n):
        with self.lock:
            now = time.time()
            if self.started[chain] is None:
                self.started[chain] = now
                if self.t_start is None:
                    self.t_start = now
                    self._print(f"compiled in {_dur(now - self.t0)}; sampling started")
                self.history[chain].append((now, self.done[chain]))
            if n:
                before = self.done[chain]
                self.done[chain] = min(before + n, self.n_iter)
                self.history[chain].append((now, self.done[chain]))
                if before < self.tune <= self.done[chain]:
                    self.t_tune[chain] = now
            if now - self.last_report >= self.report_every:
                self.report(now)

    def close(self, chain):
        with self.lock:
            now = time.time()
            self.done[chain] = self.n_iter
            self.finished[chain] = now
            n_done = sum(f is not None for f in self.finished)
            began = self.started[chain] or self.t_start or self.t0
            self._print(f"chain {chain} finished after {_dur(now - began)} "
                        f"({n_done}/{self.n_chains} chains done)")
            if n_done == self.n_chains:
                self.summary()

    # -- reporting -------------------------------------------------------------

    def _rate(self, chain):
        """Seconds per iteration over the last few updates."""
        h = self.history[chain]
        if len(h) < 2 or h[-1][1] == h[0][1]:
            return None
        return (h[-1][0] - h[0][0]) / (h[-1][1] - h[0][1])

    def report(self, now):
        self.last_report = now
        pct = [100.0 * d / self.n_iter for d in self.done]
        running = [c for c in range(self.n_chains) if self.finished[c] is None]
        warm = sum(1 for c in running if self.done[c] < self.tune)
        lines = [
            f"sampling {_dur(now - (self.t_start or self.t0))} | warmup: {warm} chains, "
            f"draws: {len(running) - warm} chains, done: "
            f"{self.n_chains - len(running)}/{self.n_chains}",
        ]
        lo = min(range(self.n_chains), key=lambda c: pct[c])
        hi = max(range(self.n_chains), key=lambda c: pct[c])
        lines.append(f"    progress: min {pct[lo]:.0f}% (chain {lo}), median "
                     f"{statistics.median(pct):.0f}%, max {pct[hi]:.0f}% (chain {hi})")
        left = {}
        for c in running:
            r = self._rate(c)
            if r is not None:
                left[c] = (self.n_iter - self.done[c]) * r
        if left:
            c = max(left, key=left.get)
            rough = " (rough: still in warmup)" if self.done[c] < self.tune else ""
            lines.append(f"    slowest: chain {c} at {self._rate(c):.2f} s/it, "
                         f"~{_dur(left[c])} left -> about {_stamp(now + left[c])}{rough}")
        lines.append("    % by chain: " + " ".join(f"{c}:{p:.0f}" for c, p in enumerate(pct)))
        self._print("\n".join(lines))

    def summary(self):
        end = max(self.finished)
        rows = [f"all {self.n_chains} chains finished; sampling took "
                f"{_dur(end - (self.t_start or self.t0))}",
                "    chain   warmup   draws    total   s/it (warmup, draws)"]
        n_draws = self.n_iter - self.tune
        for c in range(self.n_chains):
            s = self.started[c] or self.t_start or self.t0
            f = self.finished[c]
            t = self.t_tune[c] or f
            w_rate = (t - s) / self.tune if self.tune else 0.0
            d_rate = (f - t) / n_draws if n_draws else 0.0
            rows.append(f"    {c:5d} {_dur(t - s):>8s} {_dur(f - t):>7s} {_dur(f - s):>8s}"
                        f"   {w_rate:6.2f}, {d_rate:6.2f}")
        self._print("\n".join(rows))


class _Bar:
    """Stands in for the tqdm bar numpyro makes for each chain."""

    def __init__(self, report, chain):
        self.report, self.chain = report, chain

    def set_description(self, desc=None, refresh=True):
        pass

    def update(self, n=1):
        self.report.update(self.chain, int(n))

    def close(self):
        self.report.close(self.chain)


def install(tune, report_every=1800.0, steps=200):
    """Swap numpyro's multi-chain progress bars for ProgressReport."""
    # This hooks numpyro internals (written against numpyro 0.22).  If a later
    # numpyro moves them, fall back to its own bars rather than fail a fit.
    try:
        import numpyro.util as nu

        original = nu.progress_bar_factory
        nu.tqdm_auto  # noqa: B018 -- the factory must still build bars from this
    except (ImportError, AttributeError) as e:
        print(f"run_exozippy: cannot hook numpyro's progress bars ({e}); "
              f"using numpyro's own", flush=True)
        return

    def factory(num_samples, num_chains, progress_rate):
        try:
            report = ProgressReport(num_samples, num_chains, min(tune, num_samples),
                                    report_every)
            rate = max(1, num_samples // steps) if steps else progress_rate
            saved = nu.tqdm_auto
            nu.tqdm_auto = report.bar  # the original factory builds its bars from this
            try:
                return original(num_samples, num_chains, rate)
            finally:
                nu.tqdm_auto = saved
        except Exception as e:  # never let progress reporting cost a fit
            print(f"run_exozippy: progress hook failed ({type(e).__name__}: {e}); "
                  f"using numpyro's own bars", flush=True)
            return original(num_samples, num_chains, progress_rate)

    nu.progress_bar_factory = factory


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("config", help="EXOZIPPy config file, e.g. toi7792.yaml")
    p.add_argument("--logger-level", choices=("DEBUG", "INFO", "WARNING"),
                   help="passed to exozippy")
    p.add_argument("--report-minutes", type=float, default=30.0,
                   help="minutes between progress reports (default: 30)")
    p.add_argument("--steps", type=int, default=200,
                   help="progress updates per chain (default: 200; numpyro's own is 20)")
    args = p.parse_args(argv)

    import yaml

    with open(args.config) as f:
        sampler = (yaml.safe_load(f) or {}).get("sampler") or {}
    tune = int(sampler.get("tune", 2000))  # EXOZIPPy's default
    # harmless for any other sampler: it only touches numpyro's multi-chain bars
    install(tune, args.report_minutes * 60.0, args.steps)

    from exozippy.cli import main as exozippy_main

    cli = [args.config] + (["--logger-level", args.logger_level] if args.logger_level else [])
    exozippy_main.main(args=cli, prog_name="exozippy")


if __name__ == "__main__":
    main()
