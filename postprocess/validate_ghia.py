"""Compare run centrelines with Ghia et al. (1982) at the 15 interior points.

Usage: validate_ghia.py RUN_DIR ... [--markdown] [--keep-suspect]
                        [--check --max-rms-u X --max-rms-v Y]

Points listed in ghia.SUSPECT_POINTS are excluded unless --keep-suspect is given. With --check
the exit status is 1 if any run has an rms error above the limits.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from postprocess.fields import centrelines  # noqa: E402
from postprocess.ghia import SUSPECT_POINTS, ghia_u, ghia_v  # noqa: E402
from postprocess.io import load_run  # noqa: E402

HEADER = ["name", "Re", "NX", "iterations", "converged", "rms_u", "max_u", "rms_v", "max_v"]


def _stats(coord, vals, gcoord, gvals, skip=()):
    """RMS and max-abs error of the interpolated centreline at interior Ghia points."""
    gc, gv = gcoord[1:-1], gvals[1:-1]
    keep = np.array([not any(abs(x - s) < 1e-12 for s in skip) for x in gc], dtype=bool)
    err = (np.interp(gc, coord, vals) - gv)[keep]
    return float(np.sqrt(np.mean(err**2))), float(np.max(np.abs(err)))


def evaluate(path, keep_suspect=False):
    run = load_run(path)
    info = run["info"]
    Re = int(round(float(info["Re"])))
    c = centrelines(run)
    gy, gu = ghia_u(Re)
    gx, gv = ghia_v(Re)
    skip_u = [] if keep_suspect else SUSPECT_POINTS.get((Re, "u"), [])
    skip_v = [] if keep_suspect else SUSPECT_POINTS.get((Re, "v"), [])
    rms_u, max_u = _stats(c["y"], c["u"], gy, gu, skip_u)
    rms_v, max_v = _stats(c["x"], c["v"], gx, gv, skip_v)
    return [
        os.path.basename(os.path.normpath(path)),
        Re,
        int(info["N_cells_x"]),
        int(info["iterations"]),
        "yes" if info["converged"] else "no",
        rms_u,
        max_u,
        rms_v,
        max_v,
    ]


def _fmt(row):
    return [f"{x:.5f}" if isinstance(x, float) else str(x) for x in row]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--keep-suspect", action="store_true", help="include suspect Ghia points")
    ap.add_argument("--check", action="store_true", help="exit 1 if an rms limit is exceeded")
    ap.add_argument("--max-rms-u", type=float, default=None)
    ap.add_argument("--max-rms-v", type=float, default=None)
    args = ap.parse_args(argv)
    if args.check and (args.max_rms_u is None or args.max_rms_v is None):
        ap.error("--check requires --max-rms-u and --max-rms-v")
    rows = [
        evaluate(r, args.keep_suspect)
        for r in args.runs
        if os.path.isfile(os.path.join(r, "run_info.json"))
    ]
    if args.check and not rows:
        print("no runs found")
        return 1
    if not args.keep_suspect:
        skipped = {k: v for k, v in SUSPECT_POINTS.items()}
        print(f"excluded suspect Ghia points: {skipped} (use --keep-suspect to include)")
    if args.markdown:
        print("| " + " | ".join(HEADER) + " |")
        print("|" + "|".join("---" for _ in HEADER) + "|")
        for row in rows:
            print("| " + " | ".join(_fmt(row)) + " |")
    else:
        for row in rows:
            print("  ".join(_fmt(row)))
    status = 0
    if args.check:
        for row in rows:
            rms_u, rms_v = row[HEADER.index("rms_u")], row[HEADER.index("rms_v")]
            if rms_u > args.max_rms_u or rms_v > args.max_rms_v:
                print(
                    f"FAIL {row[0]}: rms_u {rms_u:.5f} (max {args.max_rms_u}), "
                    f"rms_v {rms_v:.5f} (max {args.max_rms_v})"
                )
                status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
