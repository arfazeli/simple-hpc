"""Load solver run directories."""

import json
import os

import numpy as np

FIELDS = ("u", "v", "p", "ux_centre", "uy_centre")


def _load_array(directory, name):
    """Return the array for `name`, preferring `.csv` over `.npy`; None if neither exists."""
    csv_path = os.path.join(directory, name + ".csv")
    if os.path.isfile(csv_path):
        return np.loadtxt(csv_path, delimiter=",", ndmin=2)
    npy_path = os.path.join(directory, name + ".npy")
    if os.path.isfile(npy_path):
        return np.load(npy_path)
    return None


def load_run(path):
    """Load a run directory.

    Returns a dict with keys `info`, `u`, `v`, `p`, `ux_centre`, `uy_centre`, `alog`.
    `info` is the parsed run_info.json, or {} if absent. Missing arrays are None.
    `alog` has shape (n, 4) without the header row.
    """
    info = {}
    info_path = os.path.join(path, "run_info.json")
    if os.path.isfile(info_path):
        with open(info_path, encoding="utf-8") as fh:
            info = json.load(fh)

    run = {"info": info}
    for name in FIELDS:
        run[name] = _load_array(path, name)

    alog_csv = os.path.join(path, "alog.csv")
    if os.path.isfile(alog_csv):
        run["alog"] = np.loadtxt(alog_csv, delimiter=",", skiprows=1, ndmin=2)
    else:
        run["alog"] = _load_array(path, "alog")
    return run
