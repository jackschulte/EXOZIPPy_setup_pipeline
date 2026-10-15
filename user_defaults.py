"""Personal or site defaults for the pipeline's command-line options.

Any option of setup_exozippy_fit.py, get_tess_data.py or get_tres_rvs.py can
be given a new default in a YAML file, so that settings belonging to a person
or a machine (a TRES username, where keplersplinev2.py lives, a cluster
account, module lines, an email address) stay out of the code and off the
command line.  Keys are the option names without the leading dashes:

    # ~/.exozippy_pipeline.yaml
    username: myname
    keplerspline-path: ~/code/keplersplinev2
    chains: 8

The file is ~/.exozippy_pipeline.yaml, or the path in
$EXOZIPPY_PIPELINE_CONFIG.  An option given on the command line always wins.
See exozippy_pipeline.example.yaml for every key.
"""

import os

DEFAULT_PATH = "~/.exozippy_pipeline.yaml"


def config_path():
    return os.path.expanduser(os.environ.get("EXOZIPPY_PIPELINE_CONFIG") or DEFAULT_PATH)


def _all_option_names():
    """Every option name any of the three scripts accepts."""
    import get_tess_data
    import get_tres_rvs
    import setup_exozippy_fit

    names = set()
    for build in (setup_exozippy_fit.build_parser, get_tess_data.build_parser,
                  get_tres_rvs.build_parser):
        names |= {a.dest for a in build(apply_defaults=False)._actions}
    return names


def apply(parser):
    """Set `parser`'s defaults from the user's file, if there is one.

    One file serves all three scripts, so a key another script owns is
    skipped quietly; a key no script knows (usually a typo) is warned about.
    """
    path = config_path()
    if not os.path.exists(path):
        if os.environ.get("EXOZIPPY_PIPELINE_CONFIG"):
            parser.error(f"$EXOZIPPY_PIPELINE_CONFIG points at {path}, which does not exist")
        return
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        parser.error(f"{path} must be a mapping of option: value")

    mine = {a.dest for a in parser._actions}
    defaults = {}
    for key, value in cfg.items():
        dest = str(key).lstrip("-").replace("-", "_")
        if dest in mine:
            defaults[dest] = value
    unknown = [k for k in cfg if str(k).lstrip("-").replace("-", "_") not in _all_option_names()]
    for key in unknown:
        print(f"WARNING: {path}: '{key}' is not an option of any pipeline script; ignored")
    parser.set_defaults(**defaults)
