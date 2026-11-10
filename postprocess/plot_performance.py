#!/usr/bin/env python3
"""Plot benchmark CSVs from bench/run_bench.py: phase breakdown, throughput, speedup."""
import argparse
import csv
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GROUPS = [
    ("u-momentum", ["assemble_u", "bc_u", "residual_u"]),
    ("u sweeps", ["sweep_u"]),
    ("v-momentum", ["assemble_v", "bc_v", "residual_v"]),
    ("v sweeps", ["sweep_v"]),
    ("p' assembly + corrections", ["assemble_p", "correct"]),
    ("p' solve", ["solve_p"]),
]


def load(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def med(rows, key):
    return statistics.median(float(r[key]) for r in rows)


def group_per_iter(rows):
    """Median seconds per iteration for each phase group."""
    out = {}
    for name, cols in GROUPS:
        out[name] = sum(med(rows, c) for c in cols) / float(rows[0]["iters"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.csv)
    by = defaultdict(list)
    for r in rows:
        by[(r["tag"], int(r["grid"]), int(r["threads"]))].append(r)

    # phase breakdown (threads = 1)
    grids = sorted({k[1] for k in by if k[2] == 1})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Shares in percent on a linear axis: stacking on a log axis distorts the segments.
    per_grid = []
    for g in grids:
        rs = [r for k, v in by.items() if k[1] == g and k[2] == 1 for r in v]
        per_grid.append(group_per_iter(rs))
    bottom = [0.0] * len(grids)
    for name, _ in GROUPS:
        vals = [100 * d[name] / sum(d.values()) for d in per_grid]
        ax.bar([str(g) for g in grids], vals, bottom=bottom, label=name)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xlabel("grid (N x N)")
    ax.set_ylabel("share of time per outer iteration (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "phase_breakdown.png"), dpi=200)
    plt.close(fig)

    # throughput
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lines = sorted({(k[0], k[2]) for k in by})
    for tag, th in lines:
        gs = sorted(k[1] for k in by if k[0] == tag and k[2] == th)
        ys = [med(by[(tag, g, th)], "cell_updates_per_s") for g in gs]
        ax.plot([g * g for g in gs], ys, "o-", label=f"{tag}, {th} thr")
    ax.set_xscale("log")
    ax.set_xlabel("grid size (cells)")
    ax.set_ylabel("cell updates per second")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "throughput.png"), dpi=200)
    plt.close(fig)

    # speedup
    # Speedup per (tag, grid) against the same tag's lowest thread count, so builds are never mixed.
    series = {}
    for tag, g, t in by:
        series.setdefault((tag, g), []).append(t)
    series = {k: sorted(v) for k, v in series.items() if len(v) > 1}
    if series:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (tag, g), ths in sorted(series.items()):
            base = med(by[(tag, g, ths[0])], "seconds_per_iter")
            ax.plot(ths, [base / med(by[(tag, g, t)], "seconds_per_iter") for t in ths], "o-",
                    label=f"{tag} {g}^2")
        mx = max(max(v) for v in series.values())
        ax.plot([1, mx], [1, mx], "k--", label="ideal")
        ax.set_xlabel("threads")
        ax.set_ylabel("speedup")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "speedup.png"), dpi=200)
        plt.close(fig)
    else:
        print("skipped speedup.png: only one thread count")


if __name__ == "__main__":
    main()
