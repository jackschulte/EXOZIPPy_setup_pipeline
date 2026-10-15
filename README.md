# EXOZIPPy_setup_pipeline

Scripts that set up an [EXOZIPPy](https://github.com/jdeast/EXOZIPPy) global
fit of a TESS planet candidate from scratch. They:
- download and clean the TESS light curves;
- fetch TRES (or CHIRON) RVs;
- build the SED and the stellar priors;
- write the EXOZIPPy config and parameter files, ready to run.

They run anywhere EXOZIPPy runs, on a laptop or on a SLURM cluster.

## Installation

1. **Install EXOZIPPy** by following [its README](https://github.com/jdeast/EXOZIPPy).
   It brings numpy, pandas, astropy, lightkurve, astroquery, matplotlib and
   PyYAML.
2. **Install the extra packages**, in the same Python environment:
   ```bash
   pip install -r requirements.txt
   ```
3. **Get `keplersplinev2.py`** from
   [avanderburg/keplersplinev2](https://github.com/avanderburg/keplersplinev2).
   The light curves are flattened with it, and it is not on PyPI. Put the file
   in this folder. If it lives somewhere else, give its directory with
   `--keplerspline-path`, `$KEPLERSPLINE_PATH`, or the
   [defaults file](#personal-defaults).
4. **Store your tess.exoplanets.dk password** if you want TRES RVs: one line
   in a file only you can read.
   ```bash
   echo 'MY_PASSWORD' > ~/.tres_password && chmod 600 ~/.tres_password
   ```

## Quick start

```bash
python setup_exozippy_fit.py TOI-7475 --username myname --check
cd toi7475_exozippy && exozippy toi7475.yaml
```

This writes the fit into `./toi7475_exozippy` (choose another place with
`--outdir`). Look at `lc_diagnostics/*.png` and the generated files before you
start the fit.

`--check` builds the EXOZIPPy model, without sampling, and prints the starting
log-probability. That catches config mistakes in a minute or two instead of
after the fit has started.

**On a SLURM cluster**, add `--slurm` to also write `SLURM_TOI7475.sb`, then
submit it with `sbatch`. The cluster-specific parts (account, partition,
module and environment lines, email) are options. They are most convenient
in your defaults file.

## Personal defaults

Any option of any script can be given a new default in
`~/.exozippy_pipeline.yaml`, or in the file named by
`$EXOZIPPY_PIPELINE_CONFIG`. This keeps the settings that belong to you or to
your machine off the command line. Keys are option names without their
leading dashes, and an option typed on the command line still wins. For
example, on a cluster:

```yaml
username: myname
keplerspline-path: ~/code/keplersplinev2
slurm: true
chains: 20
slurm-account: my_group
email: me@example.edu
slurm-setup:
  - module load Miniforge3
  - conda activate exozippy
```

[`exozippy_pipeline.example.yaml`](exozippy_pipeline.example.yaml) lists every
useful key. A key that no script knows (usually a typo) is reported as a
warning.

## What gets made

| File | From |
|------|------|
| `nYYYYMMDD.TESS.TESS.TOI-<N>SlimFlat.S<sec>.<exp>s<bin>.dat` | `get_tess_data.py`: one per sector |
| `lc_diagnostics/*.png` | raw + spline, flattened + clipped, phase-folded, per sector |
| `<TIC>.TRES.rv` (+ `<TIC>.TRES.raw.txt`) | `get_tres_rvs.py` |
| `toi<N>.sed.yaml`, `toi<N>.mkticsed.params.yaml` | `exozippy-mkticsed` (TICv8.2, Gaia DR3, 2MASS, WISE, Schlegel Av) |
| `toi<N>.params.yaml` | stellar priors from mkticsed, a spectroscopic [Fe/H] prior from ExoFOP, and each planet's period, Tc and Rp/R* starting values |
| `toi<N>.yaml` | the EXOZIPPy config: star + MIST, SED, planets and orbits, every `n2*.dat` and `*.rv` in the directory, and the sampler |
| `SLURM_TOI<N>.sb` | only with `--slurm`: a batch script that runs the fit |

## The scripts

### `setup_exozippy_fit.py`: the whole thing

```
python setup_exozippy_fit.py TOI-<N> [--outdir DIR] [--username USER] [--check] [--slurm]
```

Steps, each of which can be skipped to reuse what is already in the directory:

1. **ExoFOP**: the TIC ID, and period, Tc, T14 and depth for every TOI on the
   star. TOIs dispositioned FP/FA are left out. Pick TOIs with
   `--planets 7475.01 7475.02`, or give ephemerides by hand with
   `--ephem P,TC,T14_HOURS,DEPTH_PPM` (repeat per planet). Planets are lettered
   b, c, ... in TOI order.
2. **RVs** (`--skip-rvs`): see `get_tres_rvs.py`. This runs before the slow
   TESS step, so problems show up early. If the download fails (missing
   credentials, a failed login, or a change to the site), the setup stops.
   Re-run with `--skip-rvs` to set up without RVs, or with `--raw-file` to use
   a table you saved. A target that is not on the TRES site, or has no RVs
   there yet, is set up without RVs. The setup also warns if fewer RVs came
   back than the number of TRES spectra ExoFOP lists.
3. **TESS** (`--skip-tess`): see `get_tess_data.py`.
4. **SED** (`--skip-sed`): `exozippy-mkticsed`, with star name `A`.
5. **[Fe/H]**: the SNR-weighted mean of the spectroscopic (SPC) metallicities
   on ExoFOP. The width is the larger of the scatter between spectra and the
   median quoted error. Without spectra, mkticsed's TIC value is kept.
   Override it with `--feh MU SIGMA`.
6. **Write** `toi<N>.yaml` and `toi<N>.params.yaml`, and with `--slurm` also
   `SLURM_TOI<N>.sb`. Existing files are not replaced without `--overwrite`.

Model choices:
- **Orbits** are eccentric when there are RVs and circular when there are
  none. Override with `--circular` or `--eccentric`. Add `--rv-trend` for a
  linear RV slope.
- **Tc** is moved to the epoch nearest the middle of the data.
- **TRES/CHIRON times** are BJD_UTC, so their `rvinstrument` entries get
  `time_scale: "utc"` and EXOZIPPy converts them to BJD_TDB. The time scale
  is read from each `.rv` file's header line, so a file headed `#BJD_TDB` is
  left alone.
- **No dilution term.** EXOZIPPy has no explicit dilution parameter, and the
  SPOC and QLP fluxes are already deblended. Use `--undeblend` and a
  multi-star model if you need one.
- **Limb darkening** is fit per band, with no Claret prior; EXOZIPPy does not
  have that prior yet.
- **Sampler**: numpyro with tune 2000, draws 4000 and target_accept 0.95.
  numpyro runs the chains in parallel, one per core, so set `--chains` to the
  number of cores the fit will have. The default is EXOZIPPy's 4; 20 suits a
  20-core cluster job. `cores:` is left for EXOZIPPy to choose, except with
  `--cpus` or `--slurm`. With `--slurm` it is the job's CPU count, because
  EXOZIPPy's own choice would count the whole node.

**Adding ground-based light curves.** Every `n2*.dat` file in the fit
directory goes into the config. Drop in files named in the EXOFASTv2
convention, `nYYYYMMDD.<band>.<telescope>.<anything>.dat` (for example
`n20260310.Sloani.MuSCAT2.TOI7475detrend.dat`), and regenerate:

```bash
python setup_exozippy_fit.py TOI-7475 --skip-tess --skip-rvs --skip-sed --overwrite --check
```

Ground-based exposure times are estimated from the median cadence, and
`ninterp` follows the usual EXOFASTv2 rule: 30 min -> 10, 10 min -> 4,
shorter -> 1. Check both in the config. The band name in the file name is
also used as the filter name, so it must be one EXOZIPPy knows: `Sloang/r/i/z`,
`TESS`, `V`, `R`, `I` (`R`/`I` mean Cousins), `B`, `J`, `H`, `K`, ...

### `get_tess_data.py`: TESS light curves

```
python get_tess_data.py TOI-<N> --list           # what is available, and what would be chosen
python get_tess_data.py TOI-<N> --outdir DIR     # download and process every sector
```

For each sector, it picks one product: the first cadence in `--cadences`
(default `20,120,200,600,1800`, highest cadence first), then the first
pipeline in `--authors` (default `SPOC,TESS-SPOC,QLP`). Then it:

- keeps only cadences with quality == 0 (`--keep-flagged` turns this off) and
  normalizes the flux;
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

It also checks every transit. It prints a `CHECK` line, and marks the transit
in red in the PNG, if the transit is less than half covered, has no
pre-ingress or post-egress baseline, or has a baseline offset of more than 5
sigma and 10% of the depth. Look at those transits; to drop one, re-run with
`--omit-tc <BJD> [<BJD> ...] --overwrite`.

MAST sometimes stalls without answering, and astroquery has no socket timeout
for that case. Each MAST search is therefore abandoned after 2 minutes, and
each download after 10 minutes, then retried up to 3 times. If MAST is down,
the script stops with a `TimeoutError` instead of hanging.

The flattening and chopping follow the EXOFASTv2 light-curve pipeline by
E. Thygesen and J. Ranshaw in
[exofast_tools](https://github.com/jackschulte/exofast_tools).

### `get_tres_rvs.py`: TRES/CHIRON RVs

```
python get_tres_rvs.py <TIC or TOI> --username USER --outdir DIR [--instrument chiron]
python get_tres_rvs.py <TIC> --raw-file table.txt --outdir DIR   # only reformat a table you saved
```

Logs in to tess.exoplanets.dk, opens the target's candidate page, presses the
multi-order ("high precision") RV button, and writes `<TIC>.TRES.rv` with the
columns BJD_UTC, RV and error, in m/s. CHIRON's km/s are converted to m/s, and
rows from other spectrographs are dropped.

- **Credentials:** the username comes from `--username`, `$TRES_USERNAME`, or
  `username:` in the defaults file. The password comes from `--password-file`,
  `$TRES_PASSWORD`, or `~/.tres_password`. Never put the password on the
  command line.
- **`--method requests`** (the default) posts the site's ASP.NET forms
  directly, so it needs no browser and also works on a cluster node. It runs
  the search through the Search link-button's postback (`__EVENTTARGET`),
  opens the candidate page whose results row matches `T0<TIC>`, and follows
  the export's `window.open(...)` to the `T0<TIC>.RV.txt` file.
- **`--method selenium`** drives a headless Firefox or Chrome instead
  (`pip install selenium`). Try it if the site's pages change and the requests
  method stops working.
- **Target check:** both methods refuse to download unless the candidate
  page's title and the export file are for the requested `T0<TIC>`, so another
  star's RVs cannot end up in a fit. Only the export buttons are pressed; the
  candidate page's "Update" and "Recalc" buttons are never touched.
- **Debugging:** if the downloaded page cannot be parsed, it is saved as
  `<TIC>.TRES.raw.txt.unparsed` for inspection.

### `fit_progress.py`: progress of a running fit

```
python fit_progress.py FITLOG_12345678.out [more logs ...]
```

numpyro draws one terminal progress bar per chain. In a log file, such as a
SLURM job's output, these become carriage returns and cursor-movement codes.
This script reads such a log and prints, for each chain, the latest iteration,
the phase (warmup or draws, read from the fit's yaml), the elapsed time, the
s/it and the time left. It also names the chains whose bars tqdm hid, and
estimates when the slowest chain will finish. It only reads the log, so
nothing breaks if EXOZIPPy or numpyro change their output; it then just finds
nothing.

### `exofop_utils.py`, `user_defaults.py`

Shared helpers: ExoFOP lookups (the TOI table and the target JSON), and the
defaults file.

## Option reference

Every script also prints its options with `--help`. That output shows the
defaults in effect, including any from your defaults file.

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
| `--check` / `--no-check` | off | At the end, build the EXOZIPPy model (no sampling) and print the starting log-probability. |

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
| `--keplerspline-path DIR` | `$KEPLERSPLINE_PATH`, the Python path, or this folder | Directory holding `keplersplinev2.py`. |

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

### Sampler (setup_exozippy_fit.py)

| Option | Default | What it does |
|---|---|---|
| `--method-sampler M` | `numpyro` | EXOZIPPy sampler: `numpyro`, `nuts`, `blackjax`, `nutpie`, `ptde`, `ptde_async`, `demc`, `demcz` or `nested`. Only `ptde`/`ptde_async` stop on convergence; `nuts`, `demcz` and the PTDEs honor a `maxtime:` added to the config. |
| `--chains N` | `4` | Number of chains. numpyro runs one per core, so match the cores the fit will have. |
| `--tune N` | `2000` | Tuning (warmup) iterations per chain. |
| `--draws N` | `4000` | Kept iterations per chain. With many chains, fewer draws each is usually enough and finishes sooner. |
| `--cpus N` | left to EXOZIPPy; equal to `--chains` with `--slurm` | `cores:` in the config, and `--cpus-per-task` in the SLURM script. |
| `--exozippy-cmd CMD` | `exozippy` | The command that runs EXOZIPPy, e.g. `poetry run exozippy`. Used in the SLURM script and the printed run command. |

### SLURM (setup_exozippy_fit.py, only with `--slurm`)

| Option | Default | What it does |
|---|---|---|
| `--slurm` / `--no-slurm` | off | Also write the batch script `SLURM_TOI<N>.sb`. |
| `--slurm-days D` | `6.5` | Time limit, in days. numpyro cannot stop itself early, so the fit has to finish inside this. |
| `--slurm-mem-per-cpu MEM` | `1G` | Memory per CPU. |
| `--slurm-account NAME` | none | Account to charge. |
| `--slurm-partition NAME` | none | Partition to run in. |
| `--email ADDR` | none | Address for SLURM job notifications. |
| `--slurm-setup LINE` | none | A shell line run before the fit, e.g. `'module load Miniforge3'` or `'conda activate exozippy'`. Repeat for several. |
| `--slurm-directive OPTION` | none | An extra `#SBATCH` option without its dashes, e.g. `qos=normal`. Repeat for several. |
