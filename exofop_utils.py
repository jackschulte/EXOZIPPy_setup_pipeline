"""Shared ExoFOP-TESS lookups for the EXOZIPPy pipeline scripts.

get_target_info()   -> TIC ID and per-planet ephemerides for a TOI (or a TIC)
spectroscopic_feh() -> SNR-weighted spectroscopic [Fe/H] prior from ExoFOP

Adapted from scripts/generate_priors_TOI.py and the ExoFOP branch of
lightcurve_pipeline.get_params in https://github.com/jackschulte/exofast_tools.
"""

import io
import string

import numpy as np
import pandas as pd
import requests

TOI_TABLE_URL = (
    "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=pipe"
)
TARGET_JSON_URL = "https://exofop.ipac.caltech.edu/tess/target.php?id={ticid}&json"

# TFOPWG dispositions that mean "not a planet": such TOIs are left out of the
# fit by default (pass the TOI explicitly with --planets to keep one).
FALSE_POSITIVE_DISPOSITIONS = ("FP", "FA")

_toi_table_cache = None


def _get(url, timeout=120):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r


def parse_target(target):
    """Split a target string into ('toi', 7475) or ('tic', '376866659').

    Accepts 'TOI-7475', 'toi7475', '7475', '7475.01', 'TIC 376866659',
    'TIC376866659'.  A bare number is taken to be a TOI.
    """
    s = str(target).strip().upper().replace("_", "-")
    if s.startswith("TIC"):
        return "tic", s[3:].strip(" -")
    s = s.replace("TOI", "").strip(" -")
    return "toi", int(s.split(".")[0])


def load_toi_table():
    """Return the full ExoFOP TOI table (downloaded once per process)."""
    global _toi_table_cache
    if _toi_table_cache is None:
        text = _get(TOI_TABLE_URL).text
        _toi_table_cache = pd.read_csv(io.StringIO(text), sep="|")
    return _toi_table_cache


def _finite(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def get_target_info(target, planets=None, include_false_positives=False):
    """Look up a target's TIC ID and its TOIs' ephemerides on ExoFOP.

    Parameters
    ----------
    target : str or int
        TOI ('TOI-7475', '7475') or TIC ('TIC 376866659').
    planets : list of str, optional
        TOI numbers to keep (e.g. ['7475.01']).  Default: every TOI on the
        star except those dispositioned FP/FA by TFOPWG.
    include_false_positives : bool
        Keep FP/FA TOIs too.

    Returns
    -------
    dict with keys 'toi' (host TOI number or None), 'ticid' (str),
    'tmag', 'teff', 'logg', 'rstar', 'mstar', 'feh' (TOI-table stellar values,
    may be None) and 'planets': a list of dicts, one per TOI, ordered by TOI
    number and lettered b, c, d, ... with keys
    'letter', 'toi', 'period', 'period_err', 'tc', 'tc_err', 'duration_hr',
    'depth_ppm', 'disposition'.  'period' is None for a single-transit TOI.
    """
    kind, value = parse_target(target)
    df = load_toi_table()
    if kind == "toi":
        rows = df[np.floor(df["TOI"]).astype(int) == value]
    else:
        rows = df[df["TIC ID"].astype(str) == str(int(value))]
    rows = rows.drop_duplicates(subset="TOI").sort_values("TOI")
    if len(rows) == 0:
        raise ValueError(f"{target}: no TOI found in the ExoFOP TOI table")

    if planets:
        want = {f"{float(p):.2f}" for p in planets}
        rows = rows[rows["TOI"].map(lambda t: f"{t:.2f}").isin(want)]
        if len(rows) == 0:
            raise ValueError(f"{target}: none of the TOIs {planets} were found")
    elif not include_false_positives:
        disp = rows["TFOPWG Disposition"].fillna("").astype(str).str.strip()
        dropped = rows[disp.isin(FALSE_POSITIVE_DISPOSITIONS)]
        for _, r in dropped.iterrows():
            print(
                f"NOTE: skipping TOI-{r['TOI']:.2f} (TFOPWG disposition "
                f"{r['TFOPWG Disposition']}); pass --planets to keep it"
            )
        rows = rows[~disp.isin(FALSE_POSITIVE_DISPOSITIONS)]
        if len(rows) == 0:
            raise ValueError(f"{target}: every TOI on this star is FP/FA")

    first = rows.iloc[0]
    info = {
        "toi": int(np.floor(first["TOI"])),
        "ticid": str(int(first["TIC ID"])),
        "tmag": _finite(first["TESS Mag"]),
        "teff": _finite(first["Stellar Eff Temp (K)"]),
        "logg": _finite(first["Stellar log(g) (cm/s^2)"]),
        "rstar": _finite(first["Stellar Radius (R_Sun)"]),
        "mstar": _finite(first["Stellar Mass (M_Sun)"]),
        "feh": _finite(first["Stellar Metallicity"]),
        "planets": [],
    }
    letters = string.ascii_lowercase[1:]
    for letter, (_, r) in zip(letters, rows.iterrows()):
        period = _finite(r["Period (days)"])
        if period is not None and period <= 0:
            period = None
        info["planets"].append(
            {
                "letter": letter,
                "toi": f"{r['TOI']:.2f}",
                "period": period,
                "period_err": _finite(r["Period (days) err"]),
                "tc": _finite(r["Epoch (BJD)"]),
                "tc_err": _finite(r["Epoch (BJD) err"]),
                "duration_hr": _finite(r["Duration (hours)"]),
                "depth_ppm": _finite(r["Depth (ppm)"]),
                "disposition": str(r["TFOPWG Disposition"]),
            }
        )
    return info


def planets_from_ephem(ephem_strings):
    """Build planet dicts from --ephem strings 'P,TC,T14_HOURS[,DEPTH_PPM]'.

    Use P = 0 for a single-transit candidate (no period).
    """
    out = []
    for letter, s in zip(string.ascii_lowercase[1:], ephem_strings):
        vals = [float(v) for v in s.split(",")]
        if len(vals) not in (3, 4):
            raise ValueError(
                f"--ephem '{s}': expected P,TC,T14_HOURS[,DEPTH_PPM]"
            )
        out.append(
            {
                "letter": letter,
                "toi": None,
                "period": vals[0] if vals[0] > 0 else None,
                "period_err": None,
                "tc": vals[1],
                "tc_err": None,
                "duration_hr": vals[2],
                "depth_ppm": vals[3] if len(vals) == 4 else None,
                "disposition": "manual",
            }
        )
    return out


def print_planets(planets):
    for p in planets:
        label = f"TOI-{p['toi']}" if p["toi"] else "manual"
        period = f"{p['period']:.6f} d" if p["period"] else "single transit"
        depth = f"{p['depth_ppm']:.0f} ppm" if p["depth_ppm"] else "? ppm"
        print(
            f"  {p['letter']}: {label:<12s} P = {period:<16s} "
            f"Tc = {p['tc']:.5f}  T14 = {p['duration_hr']:.3f} h  "
            f"depth = {depth}  ({p['disposition']})"
        )


def count_spectra(ticid, instrument):
    """Number of spectra from `instrument` (e.g. 'TRES') listed on ExoFOP.

    Returns 0 if ExoFOP cannot be reached: this is only a cross-check.
    """
    try:
        data = _get(TARGET_JSON_URL.format(ticid=ticid)).json()
    except Exception:
        return 0
    return sum(
        1 for s in data.get("spectroscopy") or []
        if str(s.get("sinst", "")).strip().upper() == instrument.upper()
    )


def spectroscopic_feh(ticid, min_sigma=0.0):
    """SNR-weighted mean of the spectroscopic [Fe/H] values on ExoFOP.

    Uses the 'stellar' (user-uploaded spectroscopic) rows of the target's
    ExoFOP stellar-parameter table -- typically TRES/SPC.  The prior width is
    the larger of the scatter between spectra and the median quoted error,
    so a single spectrum does not produce a zero width (which EXOZIPPy
    would read as "fixed").

    Returns (mean, sigma, n_spectra), or None if there are no spectroscopic
    metallicities.
    """
    data = _get(TARGET_JSON_URL.format(ticid=ticid)).json()
    rows = data.get("stellar_parameters", [])

    # The table is a flat list: a {'prov', 'prov_num'} header followed by
    # prov_num rows from that provenance.
    feh, err, snr = [], [], []
    i = 0
    while i < len(rows):
        head = rows[i]
        if "prov" not in head:
            i += 1
            continue
        n = int(head.get("prov_num") or 0)
        if head["prov"] == "stellar":
            for row in rows[i + 1 : i + 1 + n]:
                met = _finite(row.get("met"))
                if met is None:
                    continue
                feh.append(met)
                err.append(_finite(row.get("met_e")))
                snr.append(_finite(row.get("snr")) or 1.0)
        i += 1 + n

    if not feh:
        return None
    feh = np.array(feh)
    mean = float(np.average(feh, weights=np.array(snr)))
    scatter = float(np.std(feh)) if len(feh) > 1 else 0.0
    errs = [e for e in err if e is not None and e > 0]
    quoted = float(np.median(errs)) if errs else 0.0
    sigma = max(scatter, quoted, min_sigma)
    if sigma <= 0:
        sigma = 0.1
    return mean, sigma, len(feh)
