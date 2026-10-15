# exozippy_pipeline

Scripts that set up an EXOZIPPy global fit for a TOI from scratch: download and
clean the TESS light curves, fetch TRES RVs, build the SED and stellar priors,
and write the EXOZIPPy config, params file and SLURM script. It is the EXOZIPPy
counterpart of `../exofast_tools/scripts/setup_fit.ipynb`, and the output
follows the layout of `Global_Fits/meep4/toi7475_exozippy`.

Run everything in the `exozippy` conda environment:

```bash
module load Miniforge3/25.11.0-1
conda activate exozippy
cd /mnt/research/Exoplanet_Lab/jack/pipelines/exozippy_pipeline
```

## Quick start

```bash
# one-time: store the tess.exoplanets.dk password (only the password, one line)
echo 'MY_PASSWORD' > ~/.tres_password && chmod 600 ~/.tres_password

python setup_exozippy_fit.py TOI-7475 --outdir /mnt/research/Exoplanet_Lab/jack/Global_Fits/meep4/toi7475_exozippy \
       --username schulte --check
```

Then look at `lc_diagnostics/*.png` and the generated files, and submit:

```bash
cd <outdir> && sbatch SLURM_TOI7475.sb
```

`--check` builds the EXOZIPPy model, without sampling, and prints the starting
log-probability. It catches config mistakes in a minute or two instead of in
the SLURM queue.

## What gets made

| File | From |
|------|------|
| `nYYYYMMDD.TESS.TESS.TOI-<N>SlimFlat.S<sec>.<exp>s<bin>.dat` | `get_tess_data.py`: one per sector |
| `lc_diagnostics/*.png` | raw + spline, flattened + clipped, phase-folded, per sector |
| `<TIC>.TRES.rv` (+ `<TIC>.TRES.raw.txt`) | `get_tres_rvs.py` |
| `toi<N>.sed.yaml`, `toi<N>.mkticsed.params.yaml` | `exozippy-mkticsed` (TICv8.2, Gaia DR3, 2MASS, WISE, Schlegel Av) |
| `toi<N>.params.yaml` | stellar priors from mkticsed, a spectroscopic [Fe/H] prior from ExoFOP, and each planet's period, Tc, Rp/R* starting values |
| `toi<N>.yaml` | the EXOZIPPy config: star + MIST, SED, planets/orbits, every `n2*.dat` and `*.rv` in the directory, sampler |
| `SLURM_TOI<N>.sb` | the SLURM script (20 cores, 6.5 days by default). It runs the fit through `run_exozippy.py`, for readable progress in the log |

## The scripts

### `setup_exozippy_fit.py`: the whole thing

```
python setup_exozippy_fit.py TOI-<N> [--outdir DIR] [--username USER] [--check]
```

Steps, each of which can be skipped to reuse what is already in the directory:

1. **ExoFOP**: the TIC ID, and period, Tc, T14 and depth for every TOI on the
   star. TOIs dispositioned FP/FA are left out; pick TOIs with
   `--planets 7475.01 7475.02`, or give ephemerides by hand with
   `--ephem P,TC,T14_HOURS,DEPTH_PPM` (repeat per planet). Planets are lettered
   b, c, ... in TOI order.
2. **RVs** (`--skip-rvs`): see `get_tres_rvs.py`. This runs before the slow
   TESS step, so problems show up early. If the download fails (missing
   credentials, a failed login, or site changes), the setup stops. Re-run with
   `--skip-rvs` to set up without RVs, or with `--raw-file` to use a table you
   saved. A target that is not on the TRES site, or has no RVs there yet, is
   set up without RVs. The setup also warns if fewer RVs came back than ExoFOP
   lists TRES spectra.
3. **TESS** (`--skip-tess`): see `get_tess_data.py`.
4. **SED** (`--skip-sed`): `exozippy-mkticsed` with star name `A`.
5. **[Fe/H]**: the SNR-weighted mean of the spectroscopic (SPC) metallicities
   on ExoFOP. The width is the larger of the scatter between spectra and the
   median quoted error. Without spectra, mkticsed's TIC value is kept.
   Override it with `--feh MU SIGMA`.
6. Writes `toi<N>.yaml`, `toi<N>.params.yaml` and `SLURM_TOI<N>.sb`. It will
   not overwrite them without `--overwrite`.

Model choices:
- Orbits are eccentric when there are RVs and circular when there are none.
  Override with `--circular` or `--eccentric`. Add `--rv-trend` for a linear
  RV slope.
- Tc is moved to the epoch nearest the middle of the data.
- TRES/CHIRON times are BJD_UTC, so their `rvinstrument` entries get
  `time_scale: "utc"` and EXOZIPPy converts them to BJD_TDB. The time scale
  is read from each `.rv` file's header line, so a file headed `#BJD_TDB` is
  left alone.
- There is no dilution term. EXOZIPPy has no explicit dilution parameter, and
  SPOC/QLP fluxes are already deblended. Use `--undeblend` and a multi-star
  model if you need it.
- Limb darkening is fit per band, with no Claret prior. EXOZIPPy does not have
  that prior yet.
- Sampler: numpyro with 20 chains, tune 2000, draws 4000 and target_accept
  0.95, the same as the toi7475 fit. Change these with `--chains`, `--tune`,
  `--draws` and `--method-sampler`.

**Adding ground-based light curves.** Every `n2*.dat` file in the fit
directory goes into the config. Drop in files named in the EXOFASTv2
convention, `nYYYYMMDD.<band>.<telescope>.<anything>.dat` (for example
`n20260310.Sloani.MuSCAT2.TOI7475detrend.dat`), and regenerate:

```bash
python setup_exozippy_fit.py TOI-7475 --outdir <outdir> --skip-tess --skip-rvs --skip-sed --overwrite --check
```

Ground-based exposure times are estimated from the median cadence, and
`ninterp` is set by Jason's rule (30 min -> 10, 10 min -> 4, shorter -> 1).
Check both in the config. The band name in the filename is also used as the
filter name, so it must be a name EXOZIPPy knows: `Sloang/r/i/z`, `TESS`,
`V`, `R`, `I` (`R`/`I` mean Cousins), `B`, `J`, `H`, `K`, ...

### `get_tess_data.py`: TESS light curves

```
python get_tess_data.py TOI-<N> --list           # what is available, and what would be chosen
python get_tess_data.py TOI-<N> --outdir DIR     # download + process every sector
```

For each sector, it picks one product: the first cadence in `--cadences`
(default `20,120,200,600,1800`, the highest cadence first, as in
`setup_fit.ipynb`), then the first pipeline in `--authors` (default
`SPOC,TESS-SPOC,QLP`). Then, as in `lightcurve_pipeline.create_light_curve`,
it:

- keeps only cadences with quality == 0 (`--keep-flagged` to turn this off)
  and normalizes the flux;
- flattens it with `keplersplinev2`, masking a `--window` x T14 window (default
  3) around every transit of every TOI on the star;
- clips outliers at `--nsigma` (default 5): both sides out of transit, but only
  isolated high points in transit. A run of high points, such as a ramp the
  spline could not follow, is left in place and flagged instead, so that half
  of a systematic is not clipped away;
- sets one per-point error for the sector, the robust (1.4826 x MAD) scatter of
  the flattened out-of-transit flux;
- optionally bins the light curve (`--bin-minutes`);
- keeps only the transit windows and writes them in the EXOFASTv2/EXOZIPPy
  format.

It also checks every transit and prints a `CHECK` line, and marks the transit
in red in the PNG, if the transit is less than half covered, has no
pre-ingress or post-egress baseline, or has a baseline offset of more than 5
sigma and 10% of the depth. Look at those transits; to drop one, re-run with
`--omit-tc <BJD> [<BJD> ...] --overwrite`.

For every option, see the [option reference](#option-reference) at the end.
setup_exozippy_fit.py accepts all of the light-curve options too.

MAST sometimes stalls without answering, and astroquery has no socket timeout
for that case. Each MAST search is therefore abandoned after 2 minutes, and
each download after 10 minutes, then retried up to 3 times. If MAST is down,
the script stops with a `TimeoutError` instead of hanging.

`keplersplinev2` is imported from
`../exofast_tools/scripts/lightcurve_pipeline`; set `KEPLERSPLINE_PATH` if it
moves. It needs `pydl`, which has been installed in the `exozippy` env.

### `get_tres_rvs.py`: TRES/CHIRON RVs

```
python get_tres_rvs.py <TIC or TOI> --username schulte --outdir DIR [--instrument chiron]
python get_tres_rvs.py <TIC> --raw-file table.txt --outdir DIR   # only reformat a table you saved
```

Logs in to tess.exoplanets.dk, opens the target's candidate page, presses the
multi-order ("high precision") RV button, and writes `<TIC>.TRES.rv` with the
columns BJD_UTC, RV and error, in m/s. CHIRON's km/s are converted to m/s, and
rows from other spectrographs are dropped.

- Credentials: `--username` or `$TRES_USERNAME`; the password in
  `~/.tres_password`, a file given with `--password-file`, or `$TRES_PASSWORD`.
  Never put the password on the command line.
- `--method requests` (default) posts the site's ASP.NET forms directly, so it
  needs no browser and runs on the HPCC. It runs the search through the
  Search link-button's postback (`__EVENTTARGET`), opens the candidate page
  whose results row matches `T0<TIC>`, and follows the export's
  `window.open(...)` to the `T0<TIC>.RV.txt` file. `--method selenium` uses
  the old RV_mini_pipeline browser flow, which needs `pip install selenium`,
  plus Firefox (on the HPCC) or Chrome. Try it if the site's pages change and
  the requests method stops working.
- Both methods refuse to download unless the candidate page's title and the
  export file are for the requested `T0<TIC>`, so another star's RVs cannot
  end up in a fit. Only the export buttons are pressed; the candidate page's
  "Update" and "Recalc" buttons are never touched.
- If the downloaded page cannot be parsed, it is saved as
  `<TIC>.TRES.raw.txt.unparsed` for inspection.

### `run_exozippy.py`: readable progress in the SLURM log

```
python run_exozippy.py toi7792.yaml [--report-minutes 30] [--steps 200]
```

This runs the fit exactly as `exozippy toi7792.yaml` does; the generated SLURM
scripts use it. The only difference is the sampling progress. numpyro draws
one terminal progress bar per chain. In a log file those become carriage
returns and cursor-movement codes, bars past about the 20th become
`... (more hidden) ...`, and each chain reports only every 5% of its
iterations. The wrapper writes plain lines instead:

```
[2026-10-09 03:12] sampling 2h41m | warmup: 12 chains, draws: 8 chains, done: 0/20
    progress: min 38% (chain 4), median 55%, max 72% (chain 14)
    slowest: chain 4 at 55.7 s/it, ~1d22h left -> about 2026-10-11 01:36
    % by chain: 0:41 1:55 2:38 ...
```

It writes one report every `--report-minutes` (default 30), a line as each
chain finishes, and a per-chain table of warmup and draw times at the end.
Progress is updated every 1/`--steps` of each chain (default 200 steps,
against numpyro's 20), which costs nothing noticeable. Only numpyro's
multi-chain progress is changed; the rest of EXOZIPPy's output is untouched.

### `fit_progress.py`: progress of a fit started with plain `exozippy`

```
python fit_progress.py FITLOG_18604236.out [more logs ...]
```

Reads a log full of tqdm bars and prints each chain's latest iteration,
phase (warmup or draws, read from the fit's yaml), elapsed time, s/it and time
left. It also says which chains tqdm hid from the log, and when the slowest
chain should finish.

### `exofop_utils.py`

Shared ExoFOP lookups (TOI table, target JSON) used by the other scripts.

## Option reference

Every script also prints its options with `--help`.

### Target and output

| Option | Scripts | Default | What it does |
|---|---|---|---|
| `target` (positional) | all | | `setup`: a TOI (`TOI-7475` or `7475`). `get_tess_data`: a TOI or a TIC (`'TIC 376866659'`). `get_tres_rvs`: a TIC (`376866659`) or a TOI. |
| `--outdir DIR` | all | `setup`: `./toi<N>_exozippy`; others: `.` | Where the files are written. |
| `--name NAME` | setup | `toi<N>` | Base name of `<name>.yaml`, `<name>.params.yaml` and `<name>.sed.yaml`. |
| `--targetname NAME` | get_tess_data | `TOI-<N>`, or `TIC<id>` | The target name in the light-curve file names. |
| `--planets TOI [TOI ...]` | setup, get_tess_data | every TOI on the star not dispositioned FP/FA | Only these TOIs, e.g. `7475.01 7475.02`. |
| `--ephem P,TC,T14_HOURS[,DEPTH_PPM]` | setup, get_tess_data | ExoFOP ephemerides | A hand-entered ephemeris; repeat once per planet (lettered b, c, ...). It replaces ExoFOP's. Use `P = 0` for a single transit. |
| `--overwrite` | setup, get_tess_data | off | `setup`: replace an existing config, params and SLURM file, and any light curves it writes. `get_tess_data`: replace existing light-curve files. |
| `--list` | get_tess_data | off | Only list the TESS products available and which would be chosen; download nothing. |

### Setup steps (setup_exozippy_fit.py)

| Option | Default | What it does |
|---|---|---|
| `--skip-tess` | off | Don't download TESS data; use the `n2*.dat` files already in the directory. |
| `--skip-rvs` | off | Don't download RVs; use the `*.rv` files already in the directory, if any. |
| `--skip-sed` | off | Don't run mkticsed; reuse `<name>.sed.yaml` and `<name>.mkticsed.params.yaml`. |
| `--check` | off | At the end, build the EXOZIPPy model (no sampling) and print the starting log-probability. |

### TESS light curves (setup_exozippy_fit.py, get_tess_data.py)

| Option | Default | What it does |
|---|---|---|
| `--sectors N [N ...]` | all | Only these sectors. |
| `--cadences LIST` | `20,120,200,600,1800` | Cadence preference, in seconds, highest priority first. For each sector, the first cadence available is used. |
| `--authors LIST` | `SPOC,TESS-SPOC,QLP` | Pipeline preference among products at the chosen cadence. |
| `--window W` | `3` | Width, in units of T14, of the window around each transit that is masked during flattening and kept in the output. |
| `--nsigma N` | `5` | Outlier-clipping threshold; `0` or less turns clipping off. |
| `--bin-minutes M` | no binning | Bin the flattened light curve to M minutes; the config's `exptime` then becomes M. |
| `--keep-flagged` | off (quality == 0 only) | Keep cadences with nonzero quality flags. |
| `--undeblend` | off | Put back the contaminating flux SPOC removed, which dilutes the transits again. |
| `--omit-tc BJD [BJD ...]` | none | Leave out the transits at these mid-transit times. The `CHECK` lines print the value to use. |
| `--no-plots` | off | Skip the diagnostic PNGs in `lc_diagnostics/`. |

### TRES/CHIRON RVs (setup_exozippy_fit.py, get_tres_rvs.py)

| Option | Default | What it does |
|---|---|---|
| `--instrument {tres,chiron}` | `tres` | Which spectrograph's RVs to fetch. CHIRON km/s are converted to m/s. |
| `--username USER` | `$TRES_USERNAME` | tess.exoplanets.dk username. |
| `--password-file FILE` | `$TRES_PASSWORD`, else `~/.tres_password` | File holding only the site password. |
| `--raw-file FILE` | download | Reformat an RV table you saved from the site instead of downloading it. |
| `--method {requests,selenium}` | `requests` | Plain web requests, or a headless browser (needs `pip install selenium`). |
| `--browser {firefox,chrome}` | `firefox` | Browser for `--method selenium`. |

### Model (setup_exozippy_fit.py)

| Option | Default | What it does |
|---|---|---|
| `--feh MU SIGMA` | ExoFOP spectroscopy, else the TIC value | The [Fe/H] prior. |
| `--circular` | when there are no RVs | Fix e = 0 (`secosw = sesinw = 0`, `sigma: 0`). |
| `--eccentric` | when there are RVs | Fit the eccentricity even without RVs. |
| `--rv-trend` | off | Add a linear RV trend (`orbit.trend.gammadot`). |

### Sampler and SLURM (setup_exozippy_fit.py)

| Option | Default | What it does |
|---|---|---|
| `--method-sampler M` | `numpyro` | EXOZIPPy sampler: `numpyro`, `nuts`, `blackjax`, `nutpie`, `ptde`, `ptde_async`, `demc`, `demcz` or `nested`. Only `ptde`/`ptde_async` stop on convergence; `nuts`, `demcz` and the PTDEs honor a `maxtime:` added to the config. |
| `--chains N` | `20` | Number of chains. For numpyro, keep this equal to `--cpus`: each chain runs on its own CPU. |
| `--tune N` | `2000` | Tuning (warm-up) iterations per chain. |
| `--draws N` | `4000` | Kept iterations per chain. With 20 chains, 1000 is usually enough and finishes much sooner. |
| `--cpus N` | equal to `--chains` | Sets both `cores:` in the config and `--cpus-per-task` in the SLURM script. |
| `--days D` | `6.5` | SLURM time limit, in days. numpyro cannot stop itself early, so the run has to finish inside this. |
| `--mem-per-cpu MEM` | `1G` | SLURM memory per CPU. |
| `--email ADDR` | `jschulte@msu.edu` | SLURM notification address; `''` for none. |
