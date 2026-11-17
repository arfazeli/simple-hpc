#!/usr/bin/env python3
"""MPI scaling figures from bench/slurm/mpi_scaling.sbatch output.

Usage: python3 postprocess/plot_mpi.py [--results bench/results/mpi_scaling] [--out docs/figures/performance/mpi]

Reads n<nodes>_<job>/summary.csv (latest job per node count) and pingpong.csv. Writes
mpi_strong.png (speedup vs nodes), mpi_weak.png (weak-scaling efficiency), mpi_comm.png (time split
into compute, halo and collectives) and prints the alpha-beta fit and a table of medians.
"""
import argparse
import csv
import glob
import os
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_hpc import DPI, GRID, MARKERS, MUTED, SERIES, style  # noqa: E402


def load(results):
    """Rows of the latest job per node count, plus the latest ping-pong table."""
    latest, ping = {}, None
    for d in sorted(glob.glob(os.path.join(results, "n*_*"))):
        m = re.fullmatch(r"n(\d+)_(\w+)", os.path.basename(d))
        if m and os.path.exists(os.path.join(d, "summary.csv")):
            latest[int(m.group(1))] = d  # sorted, so the last job id wins
            if os.path.exists(os.path.join(d, "pingpong.csv")):
                ping = os.path.join(d, "pingpong.csv")
    rows = []
    for d in latest.values():
        with open(os.path.join(d, "summary.csv"), newline="") as fh:
            rows += [r for r in csv.DictReader(fh) if r["seconds"] != "FAILED" and r["seconds"]]
    if not rows:
        raise SystemExit(f"no summary.csv rows under {results}")
    return rows, ping


def medians(rows):
    """{(study, layout, grid, nodes): {key: median}}; layout = 'hybrid' or 'pure MPI'."""
    groups = {}
    for r in rows:
        layout = "pure MPI" if r["threads"] == "1" else "hybrid"
        key = (r["study"], layout, f'{r["grid_x"]}x{r["grid_y"]}', int(r["nodes"]))
        groups.setdefault(key, []).append(r)
    out = {}
    for key, rs in groups.items():
        out[key] = {k: statistics.median(float(r[k]) for r in rs)
                    for k in ("seconds", "halo_mean", "halo_max", "collective_mean")}
        out[key]["ranks"], out[key]["threads"] = rs[0]["ranks"], rs[0]["threads"]
    return out


def series(med, study):
    """{(layout, grid): [(nodes, stats), ...]} sorted by nodes."""
    s = {}
    for (st, layout, grid, n), v in med.items():
        if st == study:
            s.setdefault((layout, grid if study == "strong" else "weak"), []).append((n, v))
    return {k: sorted(v) for k, v in sorted(s.items())}


def fig_strong(med, path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    grids = sorted({g for (_, g) in series(med, "strong")}, key=lambda g: int(g.split("x")[0]))
    nmax = 1
    for (layout, grid), pts in series(med, "strong").items():
        base = dict(pts).get(1)
        if base is None:
            continue
        ns = [n for n, _ in pts]
        nmax = max(nmax, max(ns))
        ax.plot(ns, [base["seconds"] / v["seconds"] for _, v in pts],
                color=SERIES[grids.index(grid) % len(SERIES)],
                marker=MARKERS[0 if layout == "hybrid" else 1],
                linestyle="-" if layout == "hybrid" else "--", label=f"{grid.split('x')[0]}², {layout}")
    ax.plot([1, nmax], [1, nmax], color=MUTED, linewidth=1, linestyle=":", label="ideal")
    ax.set(xscale="log", yscale="log", xlabel="nodes", ylabel="speedup over 1 node",
           title="Strong scaling (fixed grid)")
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_yticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.minorticks_off()
    ax.grid(color=GRID, linewidth=0.6)
    style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")


def fig_weak(med, path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for i, ((layout, _), pts) in enumerate(series(med, "weak").items()):
        base = dict(pts).get(1)
        if base is None:
            continue
        ax.plot([n for n, _ in pts], [100 * base["seconds"] / v["seconds"] for _, v in pts],
                color=SERIES[i % len(SERIES)], marker=MARKERS[i % len(MARKERS)], label=layout)
    ax.axhline(100, color=MUTED, linewidth=1, linestyle=":")
    ax.set(xscale="log", xlabel="nodes (2048² cells per node)", ylabel="weak-scaling efficiency (%)",
           title="Weak scaling (fixed work per node)", ylim=(0, 110))
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.minorticks_off()
    ax.grid(color=GRID, linewidth=0.6)
    style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")


def fig_comm(med, path, grid):
    """Stacked fractions of solver time for the strong-scaling runs on `grid`, both layouts."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels, comp, halo, coll = [], [], [], []
    for (layout, g), pts in series(med, "strong").items():
        if g != grid:
            continue
        for n, v in pts:
            t = v["seconds"]
            labels.append(f"{n}N\n{layout}\n{v['ranks']}×{v['threads']}")
            halo.append(v["halo_mean"] / t)
            coll.append(v["collective_mean"] / t)
            comp.append(1 - halo[-1] - coll[-1])
    x = range(len(labels))
    ax.bar(x, comp, color=SERIES[0], label="compute")
    ax.bar(x, halo, bottom=comp, color=SERIES[1], label="halo exchange (not hidden)")
    ax.bar(x, coll, bottom=[a + b for a, b in zip(comp, halo)], color=SERIES[2],
           label="reductions + coarse-level gathers")
    ax.set_xticks(list(x), labels, fontsize=7)
    ax.set(ylabel="fraction of solver time (rank mean)", ylim=(0, 1),
           title=f"Where the time goes, {grid.split('x')[0]}² strong scaling")
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")


def alpha_beta(path):
    """Least-squares fit of one-way time = alpha + bytes / bandwidth."""
    with open(path, newline="") as fh:
        pts = [(float(r["bytes"]), float(r["one_way_us"]) * 1e-6) for r in csv.DictReader(fh)]
    n = len(pts)
    mx, my = sum(b for b, _ in pts) / n, sum(t for _, t in pts) / n
    beta = sum((b - mx) * (t - my) for b, t in pts) / sum((b - mx) ** 2 for b, _ in pts)
    alpha = pts[0][1]  # ponytail: smallest message as latency; the fitted intercept is noisy
    return alpha, 1 / beta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="bench/results/mpi_scaling")
    ap.add_argument("--out", default="docs/figures/performance/mpi")
    args = ap.parse_args()
    rows, ping = load(args.results)
    med = medians(rows)
    os.makedirs(args.out, exist_ok=True)
    print("study,layout,grid,nodes,ranks,threads,seconds,halo_mean,halo_max,collective_mean")
    for (st, layout, grid, n), v in sorted(med.items()):
        print(f"{st},{layout},{grid},{n},{v['ranks']},{v['threads']},{v['seconds']:.3f},"
              f"{v['halo_mean']:.3f},{v['halo_max']:.3f},{v['collective_mean']:.3f}")
    fig_strong(med, os.path.join(args.out, "mpi_strong.png"))
    fig_weak(med, os.path.join(args.out, "mpi_weak.png"))
    fig_comm(med, os.path.join(args.out, "mpi_comm.png"), "8192x8192")
    if ping:
        a, bw = alpha_beta(ping)
        print(f"alpha-beta: latency {a * 1e6:.2f} us, bandwidth {bw / 1e9:.2f} GB/s ({ping})")
    print(f"figures in {args.out}")


if __name__ == "__main__":
    main()
