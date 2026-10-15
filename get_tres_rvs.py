#!/usr/bin/env python
"""Fetch TRES (or CHIRON) RVs from tess.exoplanets.dk and write an .rv file.

Logs in to Lars Buchhave's TFOP spectroscopy site, finds the target's
candidate page, presses the multi-order ("high precision") RV button, and
converts the table to the EXOZIPPy/EXOFASTv2 format:

    #BJD_UTC vrad(m/s) svrad(m/s)
    2461017.051839 -1397.40 49.60
    ...

written as <outdir>/<TICID>.TRES.rv (or .CHIRON.rv).  The raw table is kept
next to it as <TICID>.TRES.raw.txt.  The site's times are BJD_UTC; the setup
script reads that from the header and tells EXOZIPPy (time_scale: utc).

Credentials (never put the password on the command line):
  username: --username, $TRES_USERNAME, or `username:` in ~/.exozippy_pipeline.yaml
  password: the file given by --password-file, else $TRES_PASSWORD, else
            ~/.tres_password (a file holding only the password;
            `chmod 600 ~/.tres_password`)

Two download methods:
  requests  (default) plain HTTP form posts -- no browser needed, so it also
            works on a cluster node
  selenium  drives a headless browser like the old RV_mini_pipeline.py;
            needs `pip install selenium` and Firefox or Chrome

Adapted from scripts/RV_mini_pipeline.py in https://github.com/jackschulte/exofast_tools.

Examples
--------
  python get_tres_rvs.py 376866659 --username myname --outdir toi7475_exozippy
  python get_tres_rvs.py TOI-7475 --username myname             # TOIs are resolved on ExoFOP
  python get_tres_rvs.py 376866659 --raw-file saved_table.txt   # just reformat a saved table
"""

import argparse
import io
import os
import re
import sys
import time
from urllib.parse import urljoin

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exofop_utils  # noqa: E402
import user_defaults  # noqa: E402

BASE_URL = "http://tess.exoplanets.dk/"
LOGIN_USER = "ctl00$cpMainContent$tbUserName"
LOGIN_PASS = "ctl00$cpMainContent$tbPassword"
LOGIN_BUTTON = "ctl00$cpMainContent$btnLogin"
SEARCH_FIELD = "ctl00$cpMainContent$tbSearch"

# button on the candidate page that opens each instrument's multi-order RVs
RV_BUTTONS = {
    "tres": "ctl00$cpMainContent$Button10",
    "chiron": "ctl00$cpMainContent$Button12",
}
# units of the site's vrad/svrad columns, as a factor to m/s
TO_MS = {"tres": 1.0, "chiron": 1000.0}

DEFAULT_PASSWORD_FILE = os.path.expanduser("~/.tres_password")


def get_credentials(username=None, password_file=None):
    """Password from --password-file, else $TRES_PASSWORD, else ~/.tres_password."""
    username = username or os.environ.get("TRES_USERNAME")
    password = None
    if password_file:
        path = os.path.expanduser(password_file)
        if not os.path.exists(path):
            raise RuntimeError(f"--password-file {password_file} does not exist")
        with open(path) as f:
            password = f.read().strip()
    else:
        password = os.environ.get("TRES_PASSWORD")
        if password is None and os.path.exists(DEFAULT_PASSWORD_FILE):
            with open(DEFAULT_PASSWORD_FILE) as f:
                password = f.read().strip()
    if not username or not password:
        raise RuntimeError(
            "TRES site credentials not found: give --username (or set "
            "$TRES_USERNAME) and put the password in ~/.tres_password, a file "
            "given with --password-file, or $TRES_PASSWORD."
        )
    return username, password


def tres_target_id(ticid):
    """The site indexes targets as T0 + the zero-padded TIC."""
    return "T0" + str(ticid).zfill(9)


# ---------------------------------------------------------------------------
# requests method (ASP.NET WebForms postbacks)
# ---------------------------------------------------------------------------


def _form_state(html, page_url):
    """Return (action_url, fields, soup) for the page's main form: every named
    input a browser would submit, without any submit buttons."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    form = soup.find("form")
    if form is None:
        raise RuntimeError(f"no <form> on {page_url}")
    fields = {}
    for inp in form.find_all("input"):
        name = inp.get("name")
        typ = (inp.get("type") or "text").lower()
        if not name or typ in ("submit", "image", "button", "reset", "file"):
            continue
        if typ in ("checkbox", "radio") and not inp.has_attr("checked"):
            continue
        fields[name] = inp.get("value", "")
    for sel in form.find_all("select"):
        opt = sel.find("option", selected=True) or sel.find("option")
        if sel.get("name") and opt is not None:
            fields[sel["name"]] = opt.get("value", opt.text)
    for ta in form.find_all("textarea"):
        if ta.get("name"):
            fields[ta["name"]] = ta.text
    action = urljoin(page_url, form.get("action") or page_url)
    return action, fields, soup


def _button_value(soup, name, default=""):
    b = soup.find("input", attrs={"name": name})
    return b.get("value", default) if b is not None else None


class TargetNotOnSite(RuntimeError):
    """The search found no candidate page for this target."""


def _search_event_target(soup):
    """The postback target of the site's Search link-button.

    It is an <a href="javascript:__doPostBack('ctl00$cpMainContent$lnkbSearch','')">,
    not a submit button, so a plain form post has to name it in
    __EVENTTARGET.  Posting without it just re-renders the default list of
    ~100 candidates -- the bug that once fetched another star's RVs.
    """
    for a in soup.find_all("a", href=True):
        m = re.search(r"__doPostBack\('([^']+)'", a["href"])
        if m and a.get_text(strip=True).lower() == "search":
            return m.group(1)
    raise RuntimeError("no Search button on Default.aspx -- has the site changed?")


def _candidate_url(html, page_url, tid):
    """The Candidate_Edit link whose results-table row is this target."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    hits = []
    for a in soup.find_all("a", href=True):
        if "Candidate_Edit" not in a["href"]:
            continue
        row = a.find_parent("tr")
        text = row.get_text(" ", strip=True) if row else a.get_text(strip=True)
        if re.search(rf"\b{tid}\b", text):
            hits.append(urljoin(page_url, a["href"]))
    if not hits:
        raise TargetNotOnSite(f"{tid} is not on the TRES site (the search found no candidate)")
    if len(set(hits)) > 1:
        print(f"WARNING: {len(set(hits))} candidate pages match {tid}; using the first")
    return hits[0]


def _check_candidate_page(html, tid):
    """Refuse to go on unless the candidate page is this target."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    if title != tid:
        raise RuntimeError(
            f"the candidate page is for {title or 'an unknown target'!r}, not {tid}; "
            f"refusing to download its RVs"
        )


def _extract_table(resp, session, timeout, tid):
    """Pull the plain-text RV table out of the button's response.

    The export button answers with a page whose script opens the table, e.g.
    window.open('http://exoplanets.dk/uploads/temp/T0033400193.RV.txt').
    """
    ctype = resp.headers.get("Content-Type", "")
    text = resp.text
    if "text/plain" in ctype or not text.lstrip().startswith("<"):
        return text
    # a page that opens the table in a new window
    m = re.search(r"window\.open\(\s*['\"]([^'\"]+)['\"]", text)
    if m:
        url = urljoin(resp.url, m.group(1))
        if tid not in url:
            raise RuntimeError(f"the export points at {url}, not a {tid} file; refusing it")
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        return _extract_table(r, session, timeout, tid)
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(text, "html.parser")
    pre = soup.find("pre")
    if pre is not None:
        return pre.get_text()
    body = soup.body.get_text("\n") if soup.body else soup.get_text("\n")
    return body


def download_requests(ticid, username, password, instrument="tres", timeout=60):
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64) exozippy_pipeline"

    # 1. log in
    r = s.get(urljoin(BASE_URL, "Login.aspx"), timeout=timeout)
    r.raise_for_status()
    action, data, soup = _form_state(r.text, r.url)
    data[LOGIN_USER] = username
    data[LOGIN_PASS] = password
    data[LOGIN_BUTTON] = _button_value(soup, LOGIN_BUTTON, "Login") or "Login"
    r = s.post(action, data=data, timeout=timeout)
    r.raise_for_status()
    if LOGIN_PASS in r.text:
        raise RuntimeError("TRES site login failed (check the username/password)")

    # 2. search for the target (the Search link-button's postback)
    tid = tres_target_id(ticid)
    r = s.get(urljoin(BASE_URL, "Default.aspx"), timeout=timeout)
    r.raise_for_status()
    action, data, soup = _form_state(r.text, r.url)
    if SEARCH_FIELD not in data:
        raise RuntimeError("no search box on Default.aspx -- has the site changed?")
    data[SEARCH_FIELD] = tid
    data["__EVENTTARGET"] = _search_event_target(soup)
    data["__EVENTARGUMENT"] = ""
    r = s.post(action, data=data, timeout=timeout)
    r.raise_for_status()

    # 3. open this target's candidate page, and make sure it is this target
    cand_url = r.url if "Candidate_Edit" in r.url else _candidate_url(r.text, r.url, tid)
    r = s.get(cand_url, timeout=timeout)
    r.raise_for_status()
    _check_candidate_page(r.text, tid)

    # 4. press the RV button (an export: it changes nothing on the site)
    action, data, soup = _form_state(r.text, r.url)
    button = RV_BUTTONS[instrument]
    value = _button_value(soup, button)
    if value is None:
        raise RuntimeError(
            f"no {instrument.upper()} RV button ({button}) on the candidate page"
        )
    data[button] = value
    r = s.post(action, data=data, timeout=timeout)
    r.raise_for_status()
    return _extract_table(r, s, timeout, tid)


# ---------------------------------------------------------------------------
# selenium method (the original RV_mini_pipeline.py flow)
# ---------------------------------------------------------------------------


def download_selenium(ticid, username, password, instrument="tres", browser="firefox",
                      show=False):
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
    except ImportError as e:
        raise ImportError("the selenium method needs `pip install selenium`") from e

    if browser == "firefox":
        opts = webdriver.FirefoxOptions()
        if not show:
            opts.add_argument("--headless")
        driver = webdriver.Firefox(options=opts)
    else:
        opts = webdriver.ChromeOptions()
        if not show:
            opts.add_argument("--headless")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=opts)

    try:
        driver.get(urljoin(BASE_URL, "Login.aspx"))
        driver.find_element(By.NAME, LOGIN_USER).send_keys(username)
        pw = driver.find_element(By.NAME, LOGIN_PASS)
        pw.send_keys(password)
        pw.send_keys(Keys.RETURN)
        time.sleep(5)

        tid = tres_target_id(ticid)
        driver.get(urljoin(BASE_URL, "Default.aspx"))
        driver.find_element(By.NAME, SEARCH_FIELD).send_keys(tid)
        driver.find_element(By.LINK_TEXT, "Search").click()
        time.sleep(5)

        driver.get(_candidate_url(driver.page_source, driver.current_url, tid))
        time.sleep(5)
        _check_candidate_page(driver.page_source, tid)

        driver.find_element(By.NAME, RV_BUTTONS[instrument]).click()  # opens a new tab
        time.sleep(5)
        driver.switch_to.window(driver.window_handles[-1])
        if tid not in driver.current_url:
            raise RuntimeError(f"the export opened {driver.current_url}, not a {tid} file")
        return driver.find_element(By.TAG_NAME, "body").text
    finally:
        driver.quit()


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def parse_rv_table(raw_text, instrument="tres"):
    """Parse the site's RV table into a DataFrame (time, rv, err in m/s).

    Returns (df, time_column_name).  The time column is named e.g. BJD_UTC.
    """
    lines = [ln for ln in raw_text.splitlines() if ln.strip()]
    hdr = next(
        (i for i, ln in enumerate(lines) if "vrad" in ln and "BJD" in ln.upper()),
        None,
    )
    if hdr is None:
        raise ValueError(
            "could not find the RV table header (a line with BJD... and vrad) "
            "in the downloaded text"
        )
    lines[hdr] = lines[hdr].lstrip("#")
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), sep=r"\s+", comment="#")
    for col in ("spectrograph", "spec."):  # keep only this instrument's spectra
        if col in df.columns:
            df = df[df[col].astype(str).str.lower() == instrument]
            break
    tcol = next(c for c in df.columns if c.upper().startswith("BJD"))
    factor = TO_MS[instrument]
    out = pd.DataFrame(
        {
            "time": pd.to_numeric(df[tcol], errors="coerce"),
            "rv": pd.to_numeric(df["vrad"], errors="coerce") * factor,
            "err": pd.to_numeric(df["svrad"], errors="coerce") * factor,
        }
    )
    out = out[np.isfinite(out).all(axis=1)].sort_values("time")
    return out, tcol


def write_rv_file(df, tcol, path):
    with open(path, "w") as f:
        f.write(f"#{tcol} vrad(m/s) svrad(m/s)\n")
        for t, v, e in zip(df["time"], df["rv"], df["err"]):
            f.write(f"{t:.6f} {v:.2f} {e:.2f}\n")


def get_rvs(
    ticid,
    outdir=".",
    instrument="tres",
    username=None,
    password_file=None,
    raw_file=None,
    method="requests",
    browser="firefox",
):
    """Download (unless raw_file is given) and format the RVs.

    Returns the path of the .rv file, or None if the target has no RVs.
    """
    instrument = instrument.lower()
    os.makedirs(outdir, exist_ok=True)
    inst = instrument.upper()
    raw_path = os.path.join(outdir, f"{ticid}.{inst}.raw.txt")
    rv_path = os.path.join(outdir, f"{ticid}.{inst}.rv")

    if raw_file:
        with open(raw_file) as f:
            raw = f.read()
    else:
        user, pw = get_credentials(username, password_file)
        print(f"Downloading {inst} RVs for {tres_target_id(ticid)} ({method}) ...")
        if method == "requests":
            raw = download_requests(ticid, user, pw, instrument)
        else:
            raw = download_selenium(ticid, user, pw, instrument, browser)
        with open(raw_path, "w") as f:
            f.write(raw)

    try:
        df, tcol = parse_rv_table(raw, instrument)
    except ValueError as e:
        dbg = raw_path + ".unparsed"
        with open(dbg, "w") as f:
            f.write(raw)
        raise ValueError(f"{e}. The downloaded text is in {dbg}") from e

    if len(df) == 0:
        print(f"No {inst} RVs for TIC {ticid}.")
        return None
    write_rv_file(df, tcol, rv_path)
    print(
        f"Wrote {len(df)} {inst} RVs ({tcol}, median error "
        f"{np.median(df['err']):.1f} m/s, rms {np.std(df['rv']):.1f} m/s) -> {rv_path}"
    )
    return rv_path


def add_rv_arguments(p):
    """RV options shared with setup_exozippy_fit.py."""
    g = p.add_argument_group("TRES/CHIRON RVs")
    g.add_argument("--instrument", default="tres", choices=sorted(RV_BUTTONS),
                   help="which spectrograph's RVs to fetch (default: tres)")
    g.add_argument("--username", default=None, help="tess.exoplanets.dk username (or $TRES_USERNAME)")
    g.add_argument("--password-file", default=None,
                   help="file holding only the site password (default: $TRES_PASSWORD, "
                        "else ~/.tres_password)")
    g.add_argument("--raw-file", default=None,
                   help="reformat this saved RV table instead of downloading")
    g.add_argument("--method", default="requests", choices=("requests", "selenium"),
                   help="download method (default: requests)")
    g.add_argument("--browser", default="firefox", choices=("firefox", "chrome"),
                   help="browser for --method selenium (default: firefox)")
    return g


def build_parser(apply_defaults=True):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("target", help="TIC ID (e.g. 376866659) or TOI (e.g. TOI-7475)")
    p.add_argument("--outdir", default=".", help="where to write the .rv file (default: .)")
    add_rv_arguments(p)
    if apply_defaults:
        user_defaults.apply(p)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    t = str(args.target).strip()
    if t.isdigit() and len(t) > 5:  # a bare long number is a TIC
        ticid = t
    else:
        kind, value = exofop_utils.parse_target(t)
        ticid = value if kind == "tic" else exofop_utils.get_target_info(t)["ticid"]

    get_rvs(
        ticid, outdir=args.outdir, instrument=args.instrument,
        username=args.username, password_file=args.password_file,
        raw_file=args.raw_file, method=args.method, browser=args.browser,
    )


if __name__ == "__main__":
    main()
