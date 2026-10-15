#!/usr/bin/env python
"""Download, flatten, clean and chop TESS light curves for an EXOZIPPy fit.

For every TESS sector of the target, one product is chosen (by cadence, then
pipeline: SPOC > TESS-SPOC > QLP), downloaded with lightkurve, and:

  1. normalized to its median;
  2. flattened with keplersplinev2 with every known transit masked
     (all TOIs on the star, windows of `window` x T14);
  3. sigma-clipped: both sides out of transit, upward outliers only in
     transit;
  4. given a single per-point error, the robust (1.4826 x MAD) scatter of the
     flattened out-of-transit flux;
  5. chopped to the transit windows and written in EXOFASTv2/EXOZIPPy format:
     nYYYYMMDD.TESS.TESS.<TARGET>SlimFlat.S<sector>.<exptime>s<binning>.dat
     with columns BJD_TDB, normalized flux, error.

A diagnostic PNG per sector goes to <outdir>/lc_diagnostics/.

Adapted from scripts/lightcurve_pipeline/lightcurve_pipeline.py (E. Thygesen,
J. Ranshaw) and setup_fit.ipynb in https://github.com/jackschulte/exofast_tools.
Requires Andrew Vanderburg's
keplersplinev2.py (https://github.com/avanderburg/keplersplinev2; put it next
to this script or use --keplerspline-path) and pydl (pip install pydl).

Examples
--------
  # see what is available and what would be chosen, without downloading
  python get_tess_data.py TOI-7475 --list

  # download everything into a fit directory
  python get_tess_data.py TOI-7475 --outdir ~/fits/toi7475_exozippy

  # only sectors 14 and 41, prefer 120 s over 20 s, bin to 10 minutes
  python get_tess_data.py TOI-7475 --sectors 14 41 --cadences 120,20,200,600,1800 --bin-minutes 10
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.time import Time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exofop_utils  # noqa: E402
import user_defaults  # noqa: E402

KEPLERSPLINE_URL = "https://github.com/avanderburg/keplersplinev2"

# Highest cadence first, as setup_fit.ipynb recommends.  Pass --cadences to
# prefer 120 s (6x fewer points than 20 s, so a cheaper fit).
DEFAULT_CADENCES = (20, 120, 200, 600, 1800)
DEFAULT_AUTHORS = ("SPOC", "TESS-SPOC", "QLP")

# TESS times are BTJD = BJD_TDB - 2457000
BTJD_OFFSET = 2457000.0


def _import_keplerspline(path=None):
    """Import choosekeplersplinev2.

    keplersplinev2.py is not on PyPI, so it is looked for, in order, in the
    directory `path` (--keplerspline-path, or $KEPLERSPLINE_PATH), wherever a
    plain import finds it (e.g. on PYTHONPATH), and next to these scripts.
    """
    path = path or os.environ.get("KEPLERSPLINE_PATH")
    if path:
        path = os.path.expanduser(path)
        if not os.path.exists(os.path.join(path, "keplersplinev2.py")):
            raise ImportError(f"no keplersplinev2.py in {path} (--keplerspline-path / "
                              f"$KEPLERSPLINE_PATH)")
        sys.path.insert(0, path)
    elif HERE not in sys.path:
        sys.path.append(HERE)
    try:
        from keplersplinev2 import choosekeplersplinev2
    except ImportError as e:
        if e.name == "keplersplinev2":
            raise ImportError(
                f"keplersplinev2.py not found. Download it from {KEPLERSPLINE_URL} "
                f"and put it in {HERE}, or point --keplerspline-path (or "
                f"$KEPLERSPLINE_PATH) at the directory that holds it."
            ) from e
        raise ImportError(f"keplersplinev2 needs {e.name}: pip install {e.name}") from e
    return choosekeplersplinev2


# ---------------------------------------------------------------------------
# Product search and selection
# ---------------------------------------------------------------------------


def with_retries(fn, what, timeout, tries=3):
    """Call fn() with a hard per-attempt timeout, retrying on failure.

    astroquery sends MAST requests with no socket timeout, so a stalled
    connection blocks forever; each attempt therefore runs in a daemon thread
    that is abandoned if it has not answered within `timeout` seconds.
    """
    import queue
    import threading

    for attempt in range(1, tries + 1):
        q = queue.Queue()

        def run(q=q):  # bind this attempt's queue; an abandoned thread must not answer the next
            try:
                q.put((True, fn()))
            except BaseException as e:  # handed back to the caller below
                q.put((False, e))

        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(timeout)
        if th.is_alive():
            err = TimeoutError(f"no answer from MAST in {timeout:g} s")
        else:
            ok, val = q.get()
            if ok:
                return val
            err = val
        if attempt == tries:
            raise err
        print(f"{what} failed ({type(err).__name__}: {err}); retrying ({attempt}/{tries}) ...")


def search_products(ticid, authors=DEFAULT_AUTHORS, timeout=120, tries=3):
    """Return the lightkurve SearchResult for every TESS LC of this TIC."""
    import lightkurve as lk

    sr = with_retries(
        lambda: lk.search_lightcurve(f"TIC {ticid}", mission="TESS", author=list(authors)),
        "MAST search", timeout, tries,
    )
    if len(sr) == 0:
        return sr
    # a cone search can return neighbors; keep this TIC only
    keep = np.array([str(t).strip() == str(ticid) for t in sr.target_name])
    return sr[keep]


def _sector(mission):
    return int(str(mission).split()[-1])


def select_products(
    sr, cadences=DEFAULT_CADENCES, authors=DEFAULT_AUTHORS, sectors=None
):
    """Pick one product per sector: first cadence in `cadences` present, then
    first author in `authors`.  Returns a list of (sector, index into sr)."""
    exptimes = np.array([float(e) for e in sr.exptime.value])
    author = np.array([str(a) for a in sr.author])
    sector = np.array([_sector(m) for m in sr.mission])

    chosen = []
    for sec in sorted(set(sector)):
        if sectors and sec not in sectors:
            continue
        idx = np.where(sector == sec)[0]
        pick = None
        for cad in cadences:
            at_cad = idx[np.isclose(exptimes[idx], cad)]
            for auth in authors:
                hit = at_cad[author[at_cad] == auth]
                if len(hit):
                    pick = hit[0]
                    break
            if pick is not None:
                break
        if pick is None:
            print(
                f"WARNING: sector {sec}: no product at cadences {list(cadences)} "
                f"from {list(authors)}; offered: "
                f"{sorted(set(zip(author[idx], exptimes[idx].astype(int))))}"
            )
            continue
        chosen.append((sec, pick))
    return chosen


def print_products(sr, chosen):
    picked = {i for _, i in chosen}
    print(f"\n{'':2s}{'sector':>6s}  {'author':<10s} {'exptime':>8s}")
    order = sorted(
        range(len(sr)), key=lambda i: (_sector(sr.mission[i]), float(sr.exptime[i].value))
    )
    for i in order:
        mark = "->" if i in picked else "  "
        print(
            f"{mark}{_sector(sr.mission[i]):6d}  {str(sr.author[i]):<10s} "
            f"{int(sr.exptime[i].value):6d} s"
        )
    print("(-> = selected)\n")


# ---------------------------------------------------------------------------
# Light-curve processing
# ---------------------------------------------------------------------------


def transit_windows(time, planets, window=3.0):
    """Boolean mask of points within window*T14/2 (+1% buffer) of any transit,
    plus a per-planet list of the predicted mid-transit times inside `time`."""
    mask = np.zeros(len(time), dtype=bool)
    epochs = {}
    for p in planets:
        half = 0.5 * window * 1.01 * p["duration_hr"] / 24.0
        if p["period"]:
            P = p["period"]
            dt = (time - p["tc"] + 0.5 * P) % P - 0.5 * P
            n0 = np.ceil((time.min() - half - p["tc"]) / P)
            n1 = np.floor((time.max() + half - p["tc"]) / P)
            tcs = p["tc"] + P * np.arange(n0, n1 + 1)
        else:
            dt = time - p["tc"]
            tcs = np.array([p["tc"]])
        mask |= np.abs(dt) < half
        # only epochs that actually have data
        epochs[p["letter"]] = [
            t for t in tcs if np.any(np.abs(time - t) < half)
        ]
    return mask, epochs


def robust_std(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def running_median(x, n=5):
    """Centered running median over n consecutive points (edges shrink)."""
    h = n // 2
    pad = np.pad(x, h, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, n), axis=1)


def sigma_clip(flux, in_window, nsigma=5.0, maxiter=10):
    """Iteratively flag outliers.

    Out of transit: both sides.  In a transit window: only isolated HIGH
    points (cosmic rays, momentum-dump spikes) -- a point must sit nsigma
    above both the baseline and the 5-point running median around it.  A
    transit is a legitimate low excursion, and a run of high points (a
    ramp the spline could not follow) is a systematic to look at, not
    something to clip half of; check_transits flags those.
    """
    bad = ~np.isfinite(flux)
    if nsigma <= 0:
        return bad
    local = running_median(np.where(bad, np.nanmedian(flux), flux))
    for _ in range(maxiter):
        oot = ~in_window & ~bad
        med = np.median(flux[oot])
        sig = robust_std(flux[oot])
        new = (oot & (np.abs(flux - med) > nsigma * sig)) | (
            in_window & ~bad
            & (flux - med > nsigma * sig)
            & (flux - local > nsigma * sig)
        )
        if not new.any():
            break
        bad |= new
    return bad


def check_transits(time, flux, planets, epochs, sigma, cadence_days, window=3.0):
    """Flag transits that are partial or whose baseline is off or missing.

    Compares the median flattened flux before ingress and after egress
    (inside the kept window) with 1; an offset bigger than 5x its own error
    and 10% of the transit depth (300 ppm if the depth is unknown) usually
    means the spline could not follow a ramp or a gap edge, and would bias
    that transit's depth.  Returns a list of (letter, tc, reason) tuples.
    """
    flags = []
    for p in planets:
        T14 = p["duration_hr"] / 24.0
        floor = 0.1 * p["depth_ppm"] / 1e6 if p["depth_ppm"] else 3e-4
        # each baseline side runs from 0.6 T14 to the edge of the kept window
        edge = 0.5 * window * 1.01
        side_expected = max(edge - 0.6, 0.0) * T14 / cadence_days
        for tc in epochs[p["letter"]]:
            dt = time - tc
            reasons = []
            n_in = np.sum(np.abs(dt) < 0.5 * T14)
            expected = T14 / cadence_days
            if n_in < 0.5 * expected:
                reasons.append(f"partial: {100 * n_in / expected:.0f}% of the transit covered")
            if side_expected < 2:
                continue  # window too narrow for a baseline check
            for side, m in (("pre-ingress", (dt < -0.6 * T14) & (dt > -edge * T14)),
                            ("post-egress", (dt > 0.6 * T14) & (dt < edge * T14))):
                if m.sum() < max(2, 0.3 * side_expected):
                    have = "no" if m.sum() == 0 else f"only {m.sum()} points of"
                    reasons.append(f"{have} {side} baseline")
                    continue
                off = np.median(flux[m]) - 1.0
                err = 1.2533 * sigma / np.sqrt(m.sum())
                if abs(off) > max(5 * err, floor):
                    reasons.append(f"{side} baseline {off * 1e6:+.0f} ppm")
            if reasons:
                flags.append((p["letter"], tc, "; ".join(reasons)))
    return flags


def bin_lightcurve(time, flux, width_days):
    """Mean-bin (time, flux) in bins of width_days; empty bins dropped."""
    idx = np.floor((time - time.min()) / width_days).astype(int)
    uniq, inv, counts = np.unique(idx, return_inverse=True, return_counts=True)
    tb = np.bincount(inv, weights=time) / counts
    fb = np.bincount(inv, weights=flux) / counts
    return tb, fb


def process_product(
    lcfile_result,
    planets,
    outdir,
    targetname,
    window=3.0,
    nsigma=5.0,
    bin_minutes=None,
    quality_zero=True,
    undeblend=False,
    omit_tc=(),
    plot=True,
    overwrite=False,
    keplerspline_path=None,
):
    """Download, flatten, clean and chop one TESS product.

    Returns a dict describing the written file, or None if the sector has
    no usable in-transit data.
    """
    choosekeplersplinev2 = _import_keplerspline(keplerspline_path)

    sector = _sector(lcfile_result.mission[0])
    author = str(lcfile_result.author[0])
    exptime_s = int(round(float(lcfile_result.exptime[0].value)))
    tag = f"S{sector:02d} {author} {exptime_s}s"

    lc = with_retries(
        lambda: lcfile_result.download(quality_bitmask="default"),
        f"{tag} download", timeout=600,
    )
    if lc is None:
        print(f"WARNING: {tag}: download failed; skipping")
        return None
    if quality_zero:
        lc = lc[np.asarray(lc.quality) == 0]
    lc = lc.remove_nans()

    if undeblend:
        # put back the contaminating flux SPOC removed (EXOZIPPy's getdata
        # logic: CROWDSAP from the header, TIC Rcont as a fallback)
        from exozippy.utilities.getdata import (
            crowding_fraction,
            reblend_lightcurve,
            tic_contamination_ratio,
        )

        crowdsap, source = crowding_fraction(
            lc, tic_contamination_ratio(f"TIC {lcfile_result.target_name[0]}")
        )
        lc = reblend_lightcurve(lc, crowdsap)
        print(f"{tag}: undeblended with CROWDSAP = {crowdsap:.5f} ({source})")

    time = np.asarray(lc.time.value, dtype=float) + BTJD_OFFSET
    flux = np.asarray(lc.flux.value, dtype=float)
    good = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[good], flux[good]
    order = np.argsort(time)
    time, flux = time[order], flux[order]
    flux = flux / np.median(flux)

    in_win, epochs = transit_windows(time, planets, window)

    # drop whole transit windows the user asked to omit
    if len(omit_tc):
        for p in planets:
            half = 0.5 * window * 1.01 * p["duration_hr"] / 24.0
            for t in epochs[p["letter"]]:
                if np.any(np.abs(np.asarray(omit_tc) - t) < half):
                    drop = np.abs(time - t) < half
                    print(f"{tag}: omitting the {p['letter']} transit at {t:.4f}")
                    time, flux, in_win = time[~drop], flux[~drop], in_win[~drop]
        in_win, epochs = transit_windows(time, planets, window)

    if not in_win.any():
        print(f"{tag}: no transits fall in this sector; skipping")
        return None

    # flatten with the transits masked (input_mask: True = use in the fit)
    trend = choosekeplersplinev2(time, flux, input_mask=~in_win, verbose=False)
    flat = flux / trend

    bad = sigma_clip(flat, in_win, nsigma=nsigma)
    n_clipped = int(bad.sum())

    t_out, f_out = time[~bad], flat[~bad]
    if bin_minutes:
        t_out, f_out = bin_lightcurve(t_out, f_out, bin_minutes / 1440.0)
        binflag = f"binned-{bin_minutes:g}min"
    else:
        binflag = "unbinned"
    win_out, epochs_out = transit_windows(t_out, planets, window)

    # one error for every point: the robust out-of-transit scatter
    err = robust_std(f_out[~win_out])
    keep = win_out
    t_keep, f_keep = t_out[keep], f_out[keep]

    cadence = np.median(np.diff(t_out))
    flags = check_transits(t_keep, f_keep, planets, epochs_out, err, cadence, window)

    date = Time(t_keep[0], format="jd", scale="tdb").utc.strftime("%Y%m%d")
    fname = (
        f"n{date}.TESS.TESS.{targetname}SlimFlat.S{sector}."
        f"{exptime_s}s{binflag}.dat"
    )
    path = os.path.join(outdir, fname)
    if os.path.exists(path) and not overwrite:
        print(f"{tag}: {fname} exists; not overwriting (use --overwrite)")
    else:
        np.savetxt(
            path,
            np.column_stack([t_keep, f_keep, np.full(len(t_keep), err)]),
            fmt=["%.8f", "%.8f", "%.4e"],
            header=f"BJD_TDB flux flux_err  (TESS S{sector} {author} {exptime_s}s, "
            f"keplerspline-flattened, {binflag})",
        )

    ntr = sum(len(v) for v in epochs.values())
    print(
        f"{tag}: {len(t_keep)} points in {ntr} transit window(s), "
        f"{n_clipped} clipped, per-point error {err * 1e6:.0f} ppm -> {fname}"
    )
    for letter, tc, reason in flags:
        print(f"  CHECK {letter} transit at {tc:.4f}: {reason}  "
              f"(drop it with --omit-tc {tc:.4f})")

    if plot:
        _plot_diagnostics(
            time, flux, trend, flat, bad, in_win, t_keep, f_keep, planets,
            epochs, flags, window,
            f"{targetname} S{sector} {author} {exptime_s}s {binflag}",
            os.path.join(
                outdir, "lc_diagnostics",
                f"{targetname}_S{sector}_{exptime_s}s_{binflag}.png",
            ),
        )

    return {
        "file": fname,
        "sector": sector,
        "author": author,
        "exptime_s": exptime_s,
        "bin_minutes": bin_minutes,
        "npoints": len(t_keep),
        "ntransits": ntr,
        "error": err,
        "tmin": float(t_keep.min()),
        "tmax": float(t_keep.max()),
        "flags": flags,
    }


def _plot_diagnostics(
    time, flux, trend, flat, bad, in_win, t_keep, f_keep, planets, epochs,
    flags, window, title, path,
):
    flagged = {(letter, round(tc, 6)) for letter, tc, _ in flags}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nplan = len(planets)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, max(nplan, 1), height_ratios=[1, 1, 1.1])

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(time, flux, ".", ms=1, color="0.6", label="normalized flux")
    ax0.plot(time, trend, "-", lw=1, color="C3", label="keplerspline")
    ax0.plot(time[in_win], flux[in_win], ".", ms=1, color="C0", label="masked (transit)")
    ax0.set_ylabel("flux")
    ax0.legend(loc="upper right", markerscale=6, fontsize=8)
    ax0.set_title(title)

    ax1 = fig.add_subplot(gs[1, :], sharex=ax0)
    ax1.plot(time, flat, ".", ms=1, color="0.6", label="flattened")
    ax1.plot(t_keep, f_keep, ".", ms=1.5, color="C0", label="kept (written)")
    ax1.plot(time[bad], flat[bad], "x", ms=4, color="C3", label="clipped")
    for p in planets:
        for t in epochs[p["letter"]]:
            if (p["letter"], round(t, 6)) in flagged:
                ax1.axvline(t, color="C3", lw=1.5, alpha=0.8)
                ax1.annotate("CHECK", (t, 1), xycoords=("data", "axes fraction"),
                             color="C3", fontsize=8, ha="center", va="bottom")
            else:
                ax1.axvline(t, color="C2", lw=0.5, alpha=0.6)
    ax1.set_ylabel("flattened flux")
    ax1.set_xlabel("BJD_TDB")
    ax1.legend(loc="upper right", markerscale=6, fontsize=8)

    for j, p in enumerate(planets):
        ax = fig.add_subplot(gs[2, j])
        half = 0.5 * window * 1.01 * p["duration_hr"]
        if p["period"]:
            P = p["period"]
            dt = ((t_keep - p["tc"] + 0.5 * P) % P - 0.5 * P) * 24.0
        else:
            dt = (t_keep - p["tc"]) * 24.0
        sel = np.abs(dt) < half
        bad_ep = np.zeros(len(t_keep), dtype=bool)
        for letter, tc, _ in flags:
            if letter == p["letter"]:
                bad_ep |= np.abs(t_keep - tc) < half / 24.0
        ax.plot(dt[sel & ~bad_ep], f_keep[sel & ~bad_ep], ".", ms=2, color="C0")
        if np.any(sel & bad_ep):
            ax.plot(dt[sel & bad_ep], f_keep[sel & bad_ep], ".", ms=2, color="C3",
                    alpha=0.5, label="flagged transit(s)")
            ax.legend(loc="lower left", fontsize=8, markerscale=4)
        if sel.sum() > 20:
            nb = max(int(2 * half / 0.25), 10)  # ~15-min bins
            edges = np.linspace(-half, half, nb + 1)
            k = np.digitize(dt[sel], edges)
            xb = [dt[sel][k == i].mean() for i in range(1, nb + 1) if np.any(k == i)]
            yb = [f_keep[sel][k == i].mean() for i in range(1, nb + 1) if np.any(k == i)]
            ax.plot(xb, yb, "o", ms=4, color="C1")
        for s in (-0.5, 0.5):
            ax.axvline(s * p["duration_hr"], color="C2", lw=0.8, ls="--")
        lab = f"TOI-{p['toi']}" if p["toi"] else p["letter"]
        ax.set_title(f"{lab} ({len(epochs[p['letter']])} transit(s))", fontsize=9)
        ax.set_xlabel("hours from mid-transit")
        if j == 0:
            ax.set_ylabel("flattened flux")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def get_tess_data(
    ticid,
    planets,
    outdir=".",
    targetname=None,
    sectors=None,
    cadences=DEFAULT_CADENCES,
    authors=DEFAULT_AUTHORS,
    list_only=False,
    **process_kwargs,
):
    """Search, select and process every TESS sector for a target.

    Returns a list of dicts (see process_product), one per file written.
    """
    targetname = targetname or f"TIC{ticid}"
    if not list_only:  # fail before downloading anything, not once per sector
        _import_keplerspline(process_kwargs.get("keplerspline_path"))
    print(f"Searching MAST for TESS light curves of TIC {ticid} ...")
    sr = search_products(ticid, authors)
    if len(sr) == 0:
        print("No TESS light curves found.")
        return []
    chosen = select_products(sr, cadences, authors, sectors)
    print_products(sr, chosen)
    if list_only:
        return []

    os.makedirs(outdir, exist_ok=True)
    results = []
    for sec, i in chosen:
        try:
            res = process_product(sr[i], planets, outdir, targetname, **process_kwargs)
        except Exception as e:  # keep going: one bad sector should not stop the rest
            print(f"WARNING: sector {sec} failed: {type(e).__name__}: {e}")
            continue
        if res:
            results.append(res)
    return results


def add_tess_arguments(p):
    """Light-curve options shared with setup_exozippy_fit.py."""
    g = p.add_argument_group("TESS light curves")
    g.add_argument("--sectors", type=int, nargs="+", help="only these sectors (default: all)")
    g.add_argument(
        "--cadences", default=",".join(str(c) for c in DEFAULT_CADENCES),
        help="cadence preference order in seconds (default: %(default)s)",
    )
    g.add_argument(
        "--authors", default=",".join(DEFAULT_AUTHORS),
        help="pipeline preference order (default: %(default)s)",
    )
    g.add_argument("--window", type=float, default=3.0,
                   help="width of the kept/masked window around each transit, in units of T14 (default: 3)")
    g.add_argument("--nsigma", type=float, default=5.0,
                   help="outlier clipping threshold; <= 0 disables clipping (default: 5)")
    g.add_argument("--bin-minutes", type=float, default=None,
                   help="bin the flattened light curve to this many minutes (default: no binning)")
    g.add_argument("--keep-flagged", action="store_true",
                   help="keep cadences with nonzero quality flags (default: quality == 0 only)")
    g.add_argument("--undeblend", action="store_true",
                   help="undo the SPOC crowding correction (re-dilute the transits)")
    g.add_argument("--omit-tc", type=float, nargs="+", default=(),
                   help="BJD mid-transit times of individual transits to leave out")
    g.add_argument("--no-plots", action="store_true", help="skip the diagnostic PNGs")
    g.add_argument("--keplerspline-path", default=None,
                   help="directory holding keplersplinev2.py (default: $KEPLERSPLINE_PATH, "
                        "the Python path, or this script's directory)")
    return g


def _as_list(value):
    """'120,20,200' or [120, 20, 200] (from a defaults file) -> ['120', '20', '200']."""
    items = value.split(",") if isinstance(value, str) else value
    return [str(v).strip() for v in items]


def tess_kwargs(args):
    return dict(
        sectors=args.sectors,
        cadences=tuple(int(c) for c in _as_list(args.cadences)),
        authors=tuple(_as_list(args.authors)),
        window=args.window,
        nsigma=args.nsigma,
        bin_minutes=args.bin_minutes,
        quality_zero=not args.keep_flagged,
        undeblend=args.undeblend,
        omit_tc=args.omit_tc,
        plot=not args.no_plots,
        keplerspline_path=args.keplerspline_path,
    )


def build_parser(apply_defaults=True):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("target", help="TOI (e.g. TOI-7475 or 7475) or TIC (e.g. 'TIC 376866659')")
    p.add_argument("--outdir", default=".", help="where to write the light curves (default: .)")
    p.add_argument("--targetname", default=None,
                   help="name used in the file names (default: TOI-<N>, or TIC<id>)")
    p.add_argument("--planets", nargs="+",
                   help="only these TOIs, e.g. 7475.01 (default: all non-FP TOIs on the star)")
    p.add_argument("--ephem", action="append",
                   help="manual ephemeris 'P,TC,T14_HOURS[,DEPTH_PPM]' (repeat per planet); "
                        "replaces the ExoFOP ephemerides")
    p.add_argument("--list", action="store_true",
                   help="only list available products and the selection")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing light-curve files")
    add_tess_arguments(p)
    if apply_defaults:
        user_defaults.apply(p)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    kind, value = exofop_utils.parse_target(args.target)
    if args.ephem:
        planets = exofop_utils.planets_from_ephem(args.ephem)
        if kind == "tic":
            ticid, toi = value, None
        else:
            info = exofop_utils.get_target_info(args.target)
            ticid, toi = info["ticid"], info["toi"]
    else:
        info = exofop_utils.get_target_info(args.target, planets=args.planets)
        ticid, toi, planets = info["ticid"], info["toi"], info["planets"]
    targetname = args.targetname or (f"TOI-{toi}" if toi else f"TIC{ticid}")

    print(f"{targetname} = TIC {ticid}")
    exofop_utils.print_planets(planets)
    kw = tess_kwargs(args)
    get_tess_data(
        ticid, planets, outdir=args.outdir, targetname=targetname,
        list_only=args.list, overwrite=args.overwrite, **kw,
    )


if __name__ == "__main__":
    main()
