"""Figure scripts for SIMPLE run directories."""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from postprocess.fields import cell_coords, centrelines  # noqa: E402
from postprocess.ghia import ghia_u, ghia_v  # noqa: E402
from postprocess.io import load_run  # noqa: E402

DPI = 200


def _dims(run):
    info = run["info"]
    return int(info.get("N_cells_x", info.get("NX"))), int(info.get("N_cells_y", info.get("NY"))), float(info["Re"])


def fig_streamlines(run, path):
    NX, NY, Re = _dims(run)
    x, y = cell_coords(NX, NY)
    u = run["ux_centre"][::-1]
    v = run["uy_centre"][::-1]
    speed = np.hypot(u, v)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    sp = ax.streamplot(x, y, u, v, color=speed, cmap="viridis", density=1.5, linewidth=1)
    fig.colorbar(sp.lines, ax=ax, label="speed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Streamlines, Re={Re:g}, {NX}x{NY}")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_pressure(run, path):
    NX, NY, Re = _dims(run)
    H = int(run["info"]["N_halo_x"])
    p = np.asarray(run["p"][H:NX + H, H + 1:NY + H + 1], dtype=float)
    p = p - p.mean()
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(p, cmap="cividis", origin="upper", extent=(0, 1, 0, 1), aspect="equal")
    fig.colorbar(im, ax=ax, label="p - mean(p)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Pressure, Re={Re:g}, {NX}x{NY}")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _centreline_plot(runs, labels, path):
    Re = _dims(runs[0])[2]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.5))
    for run, lab in zip(runs, labels):
        c = centrelines(run)
        a1.plot(c["u"], c["y"], "-", label=lab)
        a2.plot(c["x"], c["v"], "-", label=lab)
    note = ""
    if Re in (400, 1000):
        gy, gu = ghia_u(Re)
        gx, gv = ghia_v(Re)
        a1.plot(gu, gy, "ks", mfc="none", label="Ghia et al.")
        a2.plot(gx, gv, "ks", mfc="none", label="Ghia et al.")
    else:
        note = " (no Ghia data for this Re)"
    a1.set_xlabel("u")
    a1.set_ylabel("y")
    a1.set_title("u on x = 0.5")
    a2.set_xlabel("x")
    a2.set_ylabel("v")
    a2.set_title("v on y = 0.5")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(f"Centrelines, Re={Re:g}{note}")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_centrelines(run, path):
    _centreline_plot([run], ["computed"], path)


def _residual_plot(runs, labels, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    multi = len(runs) > 1
    for run, lab in zip(runs, labels):
        a = run["alog"]
        if a is None:
            continue
        for col, name in ((1, "u"), (2, "v"), (3, "p")):
            ax.semilogy(a[:, 0], a[:, col], label=f"{lab} {name}" if multi else name)
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    ax.set_title("Residual history")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_residuals(run, path):
    _residual_plot([run], ["run"], path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    for r in args.runs:
        if not os.path.isfile(os.path.join(r, "run_info.json")):
            ap.error(f"{r} is not a run directory (no run_info.json)")
    runs =[load_run(r) for r in args.runs]
    if len(runs) == 1:
        out = args.out or os.path.join(args.runs[0], "figures")
        os.makedirs(out, exist_ok=True)
        r = runs[0]
        fig_streamlines(r, os.path.join(out, "streamlines.png"))
        fig_pressure(r, os.path.join(out, "pressure.png"))
        fig_centrelines(r, os.path.join(out, "centrelines.png"))
        fig_residuals(r, os.path.join(out, "residuals.png"))
    else:
        if not args.out:
            ap.error("--out is required with several runs")
        labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.runs]
        if len(labels) != len(runs):
            ap.error("number of --labels must match number of runs")
        os.makedirs(args.out, exist_ok=True)
        _centreline_plot(runs, labels, os.path.join(args.out, "centrelines.png"))
        _residual_plot(runs, labels, os.path.join(args.out, "residuals.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
