#!/usr/bin/env python
"""Set up an EXOZIPPy global fit for a TOI, from scratch.

Steps (each can be skipped to reuse what is already in the fit directory):

  1. ExoFOP: TIC ID, and period / Tc / T14 / depth for every TOI on the star
  2. RVs:    fetch TRES (or CHIRON) RVs from tess.exoplanets.dk (get_tres_rvs.py);
             a failed download stops the setup (--skip-rvs to go without)
  3. TESS:   download, flatten, clean and chop every sector (get_tess_data.py)
  4. SED:    exozippy-mkticsed -> <name>.sed.yaml and the stellar priors
             (Gaia DR3 parallax, Schlegel Av upper limit, TIC starting values)
  5. [Fe/H]: spectroscopic prior from ExoFOP (SNR-weighted SPC values)
  6. Writes  <name>.yaml (the EXOZIPPy config), <name>.params.yaml and
             SLURM_TOI<N>.sb

Every n2*.dat light curve and *.rv file in the fit directory goes into the
config, so ground-based light curves named in the EXOFASTv2 convention
(nYYYYMMDD.<band>.<telescope>.<anything>.dat, e.g.
n20260310.Sloani.MuSCAT2.TOI7475detrend.dat) can be dropped in and picked up
by re-running with --skip-tess --skip-rvs --skip-sed --overwrite.

Run it in the exozippy conda environment:
    conda activate exozippy
    python setup_exozippy_fit.py TOI-7475 --outdir ~/fits/toi7475_exozippy --username schulte --check

Then submit:
    cd ~/fits/toi7475_exozippy && sbatch SLURM_TOI7475.sb
"""

import argparse
import datetime
import glob
import os
import re
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exofop_utils  # noqa: E402
import get_tess_data  # noqa: E402
import get_tres_rvs  # noqa: E402

STAR = "A"

SLURM_TEMPLATE = """\
#!/bin/bash --login

#SBATCH --account=exoplanet_lab # Set this to charge cpu-hours to buy-in account
##SBATCH --nodelist="acm-007"    # Set this job to run on the buy-in node
#SBATCH --time={minutes}              # Set the runtime limit for this (minutes)
#SBATCH --nodes=1               # Set to one node
#SBATCH --ntasks=1             # Number of tasks this job will run
#SBATCH --cpus-per-task={cpus}       # Number of cores needed for each task
#SBATCH --mem-per-cpu={mem}        # Memory required for each core
{mail}#SBATCH --job-name=TOI-{toi}     # Set a job name to easily track status
#SBATCH --open-mode=append      # Set the output file to append mode
#SBATCH --output=FITLOG_%j.out # Standard output goes to this file

module purge
module load Miniforge3/25.11.0-1

# Setup the EXOFASTv2 environment
source /mnt/research/Exoplanet_Lab/source.sh

export TQDM_MININTERVAL=1800   # each chain's bar redraws at most every 30 min

# run the fit: run_exozippy.py is `exozippy {config}` with readable progress
# lines in this log (a report every --report-minutes) instead of tqdm bars

conda activate exozippy
python {wrapper} {config} --report-minutes 30

# {days:g} days for SLURM

wait
"""


# ---------------------------------------------------------------------------
# Data-file discovery
# ---------------------------------------------------------------------------

_TESS_NAME = re.compile(r"\.S(\d+)\.(\d+)s(unbinned|binned-([\d.]+)min)\.dat$")


def ninterp_for(exptime_min):
    """Jason's recommendation (generate_supporting_files.generate_pro_file):
    30 min -> 10, 10 min -> 4, shorter -> 1."""
    if exptime_min >= 25:
        return 10
    if exptime_min >= 8:
        return 4
    return 1


def _load_lc(path):
    return np.loadtxt(path, comments="#", usecols=(0, 1, 2), ndmin=2)


def scan_transits(outdir):
    """Describe every n2*.dat light curve in the fit directory."""
    entries = []
    for path in sorted(glob.glob(os.path.join(outdir, "n2*.dat"))):
        fname = os.path.basename(path)
        parts = fname.split(".")
        if len(parts) < 4:
            print(f"WARNING: {fname} is not nYYYYMMDD.BAND.TELESCOPE.*.dat; skipped")
            continue
        date, band, telescope = parts[0][1:], parts[1], parts[2]
        data = _load_lc(path)
        m = _TESS_NAME.search(fname)
        if band == "TESS" and m:
            exptime = float(m.group(4)) if m.group(4) else int(m.group(2)) / 60.0
            sector = int(m.group(1))
        else:
            # ground-based: the median cadence is an upper limit on the exposure time
            exptime = float(np.median(np.diff(np.sort(data[:, 0])))) * 1440.0
            exptime = max(round(exptime, 2), 0.01)
            sector = None
        entries.append(
            {
                "file": fname,
                "name": f"{telescope}_UT{date}",
                "band": band,
                "telescope": telescope,
                "exptime": round(exptime, 4),
                "ninterp": ninterp_for(exptime),
                "sector": sector,
                "tmin": float(data[:, 0].min()),
                "tmax": float(data[:, 0].max()),
                "npoints": len(data),
            }
        )

    # EXOZIPPy needs unique names: add the band, then a counter, on collisions
    for key in ("band", "count"):
        names = [e["name"] for e in entries]
        dups = {n for n in names if names.count(n) > 1}
        if not dups:
            break
        for i, e in enumerate(entries):
            if e["name"] in dups:
                e["name"] += f"_{e['band']}" if key == "band" else f"_{i}"

    sectors = [e["sector"] for e in entries if e["sector"] is not None]
    for s in sorted({s for s in sectors if sectors.count(s) > 1}):
        files = [e["file"] for e in entries if e["sector"] == s]
        print(f"WARNING: several light curves for TESS sector {s}: {files}. "
              f"Delete the ones you do not want and re-run with --overwrite.")
    return entries


def scan_rvs(outdir):
    """Describe every *.rv file in the fit directory (<target>.<INST>.rv)."""
    entries = []
    for path in sorted(glob.glob(os.path.join(outdir, "*.rv"))):
        fname = os.path.basename(path)
        parts = fname.split(".")
        inst = parts[-2] if len(parts) >= 3 else parts[0]
        with open(path) as f:
            first = f.readline()
        scale = "utc" if "UTC" in first.upper() else None
        data = np.loadtxt(path, comments="#", usecols=(0, 1, 2), ndmin=2)
        entries.append(
            {
                "file": fname,
                "name": inst,
                "time_scale": scale,
                "tmin": float(data[:, 0].min()),
                "tmax": float(data[:, 0].max()),
                "npoints": len(data),
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Params file
# ---------------------------------------------------------------------------


def _fmt(v):
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(f'"{x}"' for x in v) + "]"
    if isinstance(v, float):
        return repr(round(v, 10))
    return str(v)


def emit_block(lines, title, entries):
    """Append a '# ---- title ----' block of key: {field: value} entries."""
    if not entries:
        return
    lines.append(f"# ---- {title} ----")
    for key, fields in entries:
        comment = fields.pop("_comment", None)
        if comment:
            lines.append(f"# {comment}")
        lines.append(f"{key}:")
        for f, v in fields.items():
            lines.append(f"  {f}: {_fmt(v)}")
    lines.append("")


def read_mkticsed_params(path):
    """Return (header_comment_lines, {key: fields}) from mkticsed's output."""
    with open(path) as f:
        text = f.read()
    header = []
    for line in text.splitlines():
        if line.startswith("#"):
            header.append(line)
        elif line.strip():
            break
    return header, (yaml.safe_load(text) or {})


def build_params(info, planets, mkticsed_path, feh_prior, circular, tc_ref, has_rvs):
    """Assemble the params file text."""
    today = datetime.date.today().isoformat()
    toi = info["toi"]
    lines = [
        f"# EXOZIPPy parameters for TOI-{toi} (TIC {info['ticid']})",
        f"# Generated by exozippy_pipeline/setup_exozippy_fit.py on {today}.",
        "# initval = starting value; mu/sigma = Gaussian prior (sigma: 0 = fixed);",
        "# lower/upper = hard bounds.",
        "#",
    ]

    star = []
    if mkticsed_path and os.path.exists(mkticsed_path):
        header, params = read_mkticsed_params(mkticsed_path)
        lines.append("# From exozippy-mkticsed:")
        lines += [h if h.strip() != "#" else "#" for h in header]
        lines.append("")
        for key, fields in params.items():
            fields = dict(fields or {})
            if key == f"star.{STAR}.appks":
                # only read by the Mann (M-dwarf) relations, which this config
                # does not use; an unused key is silently ignored by EXOZIPPy
                continue
            if key == f"star.{STAR}.feh" and feh_prior is not None:
                continue
            star.append((key, fields))
    else:
        print("WARNING: no mkticsed params; the star has no parallax/Av priors")

    if feh_prior is not None:
        mu, sigma, label = feh_prior
        star.append(
            (
                f"star.{STAR}.feh",
                {"_comment": label, "initval": round(mu, 4), "mu": round(mu, 4),
                 "sigma": round(sigma, 4)},
            )
        )
    emit_block(lines, f"Star {STAR}", star)

    for p in planets:
        L = p["letter"]
        src = f"TOI-{p['toi']}" if p["toi"] else "manual ephemeris"
        ent = []
        if p["period"]:
            tc = p["tc"]
            if tc_ref is not None:
                # move Tc to the epoch nearest the middle of the data
                tc = tc + np.round((tc_ref - tc) / p["period"]) * p["period"]
            ent.append((f"orbit.{L}.period", {"initval": p["period"]}))
            ent.append((f"orbit.{L}.tc", {"initval": round(float(tc), 6)}))
        else:
            print(f"WARNING: planet {L} has no period (single transit); "
                  f"set orbit.{L}.period by hand")
            ent.append((f"orbit.{L}.tc", {"initval": p["tc"]}))
        if p["depth_ppm"]:
            ent.append((f"planet.{L}.p", {"initval": round(float(np.sqrt(p["depth_ppm"] / 1e6)), 5)}))
        ent.append((f"orbit.{L}.cosi", {"initval": 0.001}))
        if circular:
            why = "circular orbit" + ("" if has_rvs else " (no RVs to constrain e)")
            ent.append((f"orbit.{L}.secosw", {"_comment": why, "initval": 0.0, "sigma": 0}))
            ent.append((f"orbit.{L}.sesinw", {"initval": 0.0, "sigma": 0}))
        emit_block(lines, f"Planet {L} ({src})", ent)

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------


def build_config(info, name, planets, transits, rvs, args, notes):
    today = datetime.date.today().isoformat()
    toi, ticid = info["toi"], info["ticid"]
    L = [
        f"# TOI-{toi} (TIC {ticid}): EXOZIPPy global fit",
        f"# Generated by exozippy_pipeline/setup_exozippy_fit.py on {today}.",
        "#",
    ]
    for n in notes:
        L.append(f"#   - {n}")
    L += [
        "",
        "run:",
        f'  name: "{name}"',
        "",
        f'prefix: "fitresults/{ticid}"',
        "logger_level: INFO   # DEBUG, INFO, or WARNING",
        "",
        "star:",
        f'  - name: "{STAR}"',
        "    mist: True",
        "",
        "# MIST tracks interpolated at (logmass, initfeh, eep); each",
        "# constrained quantity gets a Gaussian potential toward the",
        "# track value with EXOFASTv2's systematic floor as sigma.",
        "evolutionarymodel:",
        f'  - star: "{STAR}"',
        "    constrain: [feh, radius, teff, age]",
        "",
        "sed:",
        f'  file: "{name}.sed.yaml"',
        "",
        "planet:",
    ]
    # orbit_ndx defaults to 0, so with several planets each must point at its
    # own orbit or its transit geometry follows planet b's
    for i, p in enumerate(planets):
        L.append(f'  - name: "{p["letter"]}"')
        if len(planets) > 1:
            L.append(f"    orbit_ndx: {i}   # orbit \"{p['letter']}\" below")
    L += ["", "orbit:"]
    for p in planets:
        L += [
            f'  - name: "{p["letter"]}"',
            f'    primary: ["{STAR}"]',
            f'    companion: ["{p["letter"]}"]',
        ]
    if args.rv_trend and rvs:
        L += [
            "  # linear RV trend (EXOFASTv2's /fitslope): orbit.trend.gammadot",
            '  - name: "trend"',
            "    type: linear",
            f'    primary: ["{STAR}"]',
        ]

    if transits:
        L += ["", "transit:"]
        for t in transits:
            L += [
                f'  - name: "{t["name"]}"',
                f'    file: "{t["file"]}"',
                f'    band: "{t["band"]}"',
                f"    exptime: {t['exptime']:g} # minutes",
                f"    ninterp: {t['ninterp']}",
            ]
        L += ["", "band:"]
        for b in sorted({t["band"] for t in transits}):
            L += [f'  - name: "{b}"', f'    filter: "{b}"']

    if rvs:
        L += ["", "rvinstrument:"]
        for r in rvs:
            L += [f'  - name: "{r["name"]}"', f'    file: "{r["file"]}"']
            if r["time_scale"]:
                L.append(f'    time_scale: "{r["time_scale"]}"   # file times are BJD_UTC; '
                         f'EXOZIPPy converts to BJD_TDB')

    L += [
        "",
        f'parameter_file: "{name}.params.yaml"',
        "",
        "sampler:",
        f"  method: {args.method_sampler}      # nuts, numpyro, blackjax, nutpie, ptde, ptde_async",
        f"  tune: {args.tune}",
        f"  draws: {args.draws}",
        f"  chains: {args.chains}",
        f"  cores: {args.cpus or args.chains}",
        "  target_accept: 0.95",
        "  recompute_trace: True   # if False and a prefix.nc file exists, load it instead of re-sampling",
        "",
    ]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def check_fit(outdir, config_name):
    """Build the EXOZIPPy model (no sampling) and evaluate the start logp."""
    print(f"\nChecking {config_name}: System.prepare() + build_model() ...")
    cwd = os.getcwd()
    os.chdir(outdir)
    try:
        from exozippy.system import System
        from exozippy.yamlio import load_system_config

        config = load_system_config(config_name)
        system = System(config)
        system.prepare()
        model = system.build_model()
        logp = float(model.compile_logp()(model.initial_point()))
    finally:
        os.chdir(cwd)
    if not np.isfinite(logp):
        raise RuntimeError(f"the start logp is {logp}: fix the starting values")
    print(f"OK: model builds; start logp = {logp:.2f}")
    return logp


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("target", help="TOI, e.g. TOI-7475 or 7475")
    p.add_argument("--outdir", default=None,
                   help="fit directory (default: ./toi<N>_exozippy)")
    p.add_argument("--name", default=None, help="base name of the yaml files (default: toi<N>)")
    p.add_argument("--planets", nargs="+",
                   help="only these TOIs, e.g. 7475.01 (default: all non-FP TOIs on the star)")
    p.add_argument("--ephem", action="append",
                   help="manual ephemeris 'P,TC,T14_HOURS[,DEPTH_PPM]' (repeat per planet, "
                        "lettered b, c, ...); replaces the ExoFOP ephemerides")
    p.add_argument("--overwrite", action="store_true",
                   help="overwrite an existing config/params/SLURM file (and light curves)")

    s = p.add_argument_group("steps")
    s.add_argument("--skip-tess", action="store_true", help="use the n2*.dat files already there")
    s.add_argument("--skip-rvs", action="store_true", help="use the *.rv files already there")
    s.add_argument("--skip-sed", action="store_true",
                   help="reuse <name>.sed.yaml and <name>.mkticsed.params.yaml")
    s.add_argument("--check", action="store_true",
                   help="build the EXOZIPPy model and evaluate the start logp at the end")

    get_tess_data.add_tess_arguments(p)
    get_tres_rvs.add_rv_arguments(p)

    m = p.add_argument_group("model")
    m.add_argument("--feh", nargs=2, type=float, metavar=("MU", "SIGMA"),
                   help="[Fe/H] prior (default: ExoFOP spectroscopy, else the TIC value)")
    orb = m.add_mutually_exclusive_group()
    orb.add_argument("--circular", action="store_true",
                     help="fix e = 0 (default when there are no RVs)")
    orb.add_argument("--eccentric", action="store_true",
                     help="fit e even without RVs (default when there are RVs)")
    m.add_argument("--rv-trend", action="store_true", help="fit a linear RV trend")

    sm = p.add_argument_group("sampler / SLURM")
    sm.add_argument("--method-sampler", default="numpyro", help="EXOZIPPy sampler (default: numpyro)")
    sm.add_argument("--chains", type=int, default=20, help="(default: 20)")
    sm.add_argument("--tune", type=int, default=2000, help="(default: 2000)")
    sm.add_argument("--draws", type=int, default=4000, help="(default: 4000)")
    sm.add_argument("--cpus", type=int, default=None, help="cores (default: = chains)")
    sm.add_argument("--days", type=float, default=6.5, help="SLURM time limit in days (default: 6.5)")
    sm.add_argument("--mem-per-cpu", default="1G", help="(default: 1G)")
    sm.add_argument("--email", default="jschulte@msu.edu",
                    help="SLURM notification address; '' for none (default: %(default)s)")
    args = p.parse_args(argv)

    # ---- 1. ExoFOP ------------------------------------------------------------
    kind, _ = exofop_utils.parse_target(args.target)
    if kind != "toi":
        p.error("give a TOI (setup needs ExoFOP's TOI entry); use get_tess_data.py for TICs")
    print(f"Looking up {args.target} on ExoFOP ...")
    info = exofop_utils.get_target_info(args.target, planets=args.planets)
    planets = exofop_utils.planets_from_ephem(args.ephem) if args.ephem else info["planets"]
    toi, ticid = info["toi"], info["ticid"]
    name = args.name or f"toi{toi}"
    outdir = os.path.abspath(args.outdir or f"toi{toi}_exozippy")
    print(f"TOI-{toi} = TIC {ticid}; fit directory {outdir}")
    exofop_utils.print_planets(planets)

    config_path = os.path.join(outdir, f"{name}.yaml")
    params_path = os.path.join(outdir, f"{name}.params.yaml")
    slurm_path = os.path.join(outdir, f"SLURM_TOI{toi}.sb")
    for path in (config_path, params_path, slurm_path):
        if os.path.exists(path) and not args.overwrite:
            p.error(f"{path} exists; use --overwrite to replace it")
    os.makedirs(os.path.join(outdir, "fitresults"), exist_ok=True)

    notes = []

    # ---- 2. RVs (before TESS, so a broken download fails fast) ------------------
    if not args.skip_rvs:
        inst = args.instrument.upper()
        print(f"\n=== {inst} RVs ===")
        n_rv = 0
        try:
            rv_path = get_tres_rvs.get_rvs(
                ticid, outdir=outdir, instrument=args.instrument,
                username=args.username, password_file=args.password_file,
                raw_file=args.raw_file, method=args.method, browser=args.browser,
            )
            if rv_path:
                n_rv = len(np.loadtxt(rv_path, comments="#", ndmin=2))
        except get_tres_rvs.TargetNotOnSite as e:
            print(f"NOTE: {e}; setting up without {inst} RVs")
        except Exception as e:
            sys.exit(
                f"ERROR: the {inst} RV download failed: {type(e).__name__}: {e}\n"
                f"Fix that, or re-run with --skip-rvs to set up without RVs, or with "
                f"--raw-file <saved table> to use a table saved from the site."
            )
        n_spec = exofop_utils.count_spectra(ticid, inst)
        if n_rv < n_spec:
            print(
                f"\nWARNING: ExoFOP lists {n_spec} {inst} spectra but only {n_rv} "
                f"{inst} RVs were retrieved. Check the target's page on "
                f"tess.exoplanets.dk before trusting this fit.\n"
            )
            notes.append(f"only {n_rv} {inst} RVs retrieved, but ExoFOP lists {n_spec} "
                         f"{inst} spectra")

    # ---- 3. TESS ---------------------------------------------------------------
    if not args.skip_tess:
        print("\n=== TESS light curves ===")
        kw = get_tess_data.tess_kwargs(args)
        res = get_tess_data.get_tess_data(
            ticid, planets, outdir=outdir, targetname=f"TOI-{toi}",
            overwrite=args.overwrite, **kw,
        )
        if res:
            secs = ", ".join(f"S{r['sector']} ({r['author']} {r['exptime_s']}s)" for r in res)
            notes.append(f"TESS: {secs}; keplerspline-flattened, chopped to "
                         f"{args.window:g} x T14 around each transit")

    # ---- 4. SED + stellar priors ----------------------------------------------
    sed_path = os.path.join(outdir, f"{name}.sed.yaml")
    mkticsed_params = os.path.join(outdir, f"{name}.mkticsed.params.yaml")
    if not args.skip_sed:
        print("\n=== SED and stellar priors (exozippy-mkticsed) ===")
        from exozippy.utilities.mkticsed import mkticsed

        mkticsed(ticid, star_name=STAR, outpath=outdir, name=name,
                 priorfile=mkticsed_params, sedfile=sed_path)
    elif not os.path.exists(sed_path):
        p.error(f"--skip-sed, but {sed_path} does not exist")

    # ---- 5. [Fe/H] prior --------------------------------------------------------
    if args.feh:
        feh_prior = (args.feh[0], args.feh[1], "[Fe/H] prior given on the command line")
    else:
        spec = exofop_utils.spectroscopic_feh(ticid)
        if spec:
            mu, sig, n = spec
            feh_prior = (mu, sig, f"spectroscopic [Fe/H]: SNR-weighted mean of {n} ExoFOP "
                                  f"value(s); sigma = max(scatter, median error)")
        else:
            feh_prior = None
            print("No spectroscopic [Fe/H] on ExoFOP; keeping mkticsed's TIC value")
    if feh_prior:
        print(f"[Fe/H] prior: {feh_prior[0]:.4f} +/- {feh_prior[1]:.4f}")

    # ---- 6. Write the fit files -------------------------------------------------
    transits = scan_transits(outdir)
    rvs = scan_rvs(outdir)
    if not transits:
        print("WARNING: no light curves in the fit directory")
    circular = args.circular or (not rvs and not args.eccentric)

    tspan = [t["tmin"] for t in transits + rvs] + [t["tmax"] for t in transits + rvs]
    tc_ref = 0.5 * (min(tspan) + max(tspan)) if tspan else None

    if any(t["band"] != "TESS" for t in transits):
        notes.append("ground-based light curves picked up from the fit directory: "
                     + ", ".join(t["name"] for t in transits if t["band"] != "TESS")
                     + " (check their exptime/ninterp)")
    if rvs:
        notes.append("RVs: " + ", ".join(f"{r['name']} ({r['npoints']})" for r in rvs))
    notes.append("no dilution term: EXOZIPPy has no explicit dilution; SPOC/QLP fluxes "
                 "are already deblended")
    notes.append("limb darkening is fit per band with no Claret prior")
    if circular:
        notes.append("circular orbit(s) (secosw = sesinw = 0 fixed in the params file)")

    with open(params_path, "w") as f:
        f.write(build_params(info, planets, mkticsed_params, feh_prior, circular,
                             tc_ref, bool(rvs)))
    with open(config_path, "w") as f:
        f.write(build_config(info, name, planets, transits, rvs, args, notes))

    mail = ""
    if args.email:
        mail = (f"#SBATCH --mail-type=ALL         # SLURM automated messages to email\n"
                f"#SBATCH --mail-user={args.email}\n")
    with open(slurm_path, "w") as f:
        f.write(SLURM_TEMPLATE.format(
            minutes=int(round(args.days * 1440)), cpus=args.cpus or args.chains,
            mem=args.mem_per_cpu, mail=mail, toi=toi, config=f"{name}.yaml", days=args.days,
            wrapper=os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_exozippy.py"),
        ))

    print(f"\nWrote {config_path}\n      {params_path}\n      {slurm_path}")
    print(f"  {len(transits)} light curve(s), {len(rvs)} RV file(s), "
          f"{len(planets)} planet(s), {'circular' if circular else 'eccentric'}")

    if args.check:
        check_fit(outdir, f"{name}.yaml")

    print(f"\nReview the files, then:  cd {outdir} && sbatch {os.path.basename(slurm_path)}")


if __name__ == "__main__":
    main()
