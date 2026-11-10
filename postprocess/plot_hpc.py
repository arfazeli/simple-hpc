#!/usr/bin/env python3
"""HPC figures from bench/results CSVs: time to solution, kernel bandwidth vs STREAM, grid scaling.

Usage: python3 postprocess/plot_hpc.py [--results bench/results] [--out docs/figures/performance/hpc]

Inputs (latest file per prefix): conv_{gs_lex,multicolour,mg1}_*.csv (128^2 runs to convergence),
par_{jacobi,gs_lex}_*.csv (200-iteration timings), stream_*.csv (bench/stream_triad.cpp),
scale_{gs_lex,multicolour,mg}_*.csv (runs to convergence on several grids, 1 thread),
cluster_<job>_g<grid>_*.csv and cluster_<job>_stream.csv (T36 strong scaling on the cluster, optional).
"""
import argparse
import csv
import glob
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

DPI = 200
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated: dataviz validate_palette.js
MARKERS = ["o", "s", "^", "D"]
CONFIGS = [  # (file suffix in conv_/scale_, label)
    ("gs_lex", "lexicographic GS (baseline)"),
    ("multicolour", "multicolour GS + red-black p'"),
    ("mg", "multicolour GS + multigrid p'"),
]

# Bytes moved per interior cell per inner sweep, counting each array once (no write-allocate),
# from the loops in src/pressure.cpp and src/momentum_u.cpp. ponytail: a streaming model, not a
# hardware counter; it ignores halo cells and cache reuse, so it is a lower bound on traffic.
BYTES = {
    "pc_jacobi": 16 + 8 * 8,  # copy pc->old (r+w), then read b, APP, AEP, AWP, ASP, ANP, old; write pc
    "u_jacobi": 16 + 12 * 8,  # copy u->old, then read 9 coefficients, old, p; write u
    "u_gs_lex": 11 * 8,  # read 9 coefficients, p, u; write u in place (counted with the read)
}


def latest(results, prefix):
    files = sorted(glob.glob(os.path.join(results, f"{prefix}_2*.csv")))
    if not files:
        raise SystemExit(f"no {prefix}_*.csv in {results}")
    with open(files[-1], newline="") as fh:
        return list(csv.DictReader(fh))


def med(rows, key, **match):
    vals = [float(r[key]) for r in rows if all(r[k] == str(v) for k, v in match.items())]
    if not vals:
        raise SystemExit(f"no rows for {match}")
    return statistics.median(vals)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_time_to_solution(results, path):
    data = {}
    for key, _ in CONFIGS:
        rows = latest(results, "conv_" + ("mg1" if key == "mg" else key))
        data[key] = [med(rows, "total_seconds", grid=128, threads=t) for t in (1, 4)]
    base = data["gs_lex"][0]
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.36
    for k, t in enumerate((1, 4)):
        xs = [i + (k - 0.5) * w for i in range(len(CONFIGS))]
        ys = [data[key][k] for key, _ in CONFIGS]
        ax.bar(xs, ys, w - 0.02, color=SERIES[k], label=f"{t} thread{'s' if t > 1 else ''}")
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.2, f"{y:.1f} s\n{base / y:.2f}×", ha="center", va="bottom",
                    fontsize=8, color=INK)
    ax.set_xticks(range(len(CONFIGS)), [lbl.replace(" + ", "\n+ ") for _, lbl in CONFIGS])
    ax.set_ylabel("time to converged solution (s)", color=INK)
    ax.set_ylim(0, base * 1.25)
    ax.set_title("Time to solution, 128², Re 1000 (speedup vs baseline on 1 thread)",
                 color=INK, fontsize=10)
    style(ax)
    ax.grid(False, axis="x")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_bandwidth(results, path, grid=512):
    stream = latest(results, "stream")
    st_t = [int(r["threads"]) for r in stream]
    st_bw = [float(r["GB_per_s"]) for r in stream]
    jac, lex = latest(results, "par_jacobi"), latest(results, "par_gs_lex")
    sweeps_u, sweeps_p, cells = 4, 8, grid * grid  # maxit_inner_u / _p in configs/cavity_re1000_80
    kernels = [  # (label, rows, phase column, bytes per iteration)
        ("p' Jacobi sweep", jac, "solve_p", BYTES["pc_jacobi"] * sweeps_p * cells),
        ("u Jacobi sweep", jac, "sweep_u", BYTES["u_jacobi"] * sweeps_u * cells),
        ("u lexicographic GS sweep", lex, "sweep_u", BYTES["u_gs_lex"] * sweeps_u * cells),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(st_t, st_bw, color=MUTED, linestyle="--", linewidth=2)
    ax.text(st_t[-1] + 0.15, st_bw[-1], "STREAM triad\n(DRAM limit)", va="center", fontsize=8,
            color=MUTED)
    for k, (label, rows, col, nbytes) in enumerate(kernels):
        ts = sorted({int(r["threads"]) for r in rows})
        bw = [nbytes / (med(rows, col, grid=grid, threads=t) / med(rows, "iters", grid=grid,
                                                                   threads=t)) / 1e9 for t in ts]
        ax.plot(ts, bw, color=SERIES[k], marker=MARKERS[k], markersize=7, linewidth=2,
                label=label)
        ax.text(ts[-1] + 0.15, bw[-1], f"{label}\n{bw[-1]:.0f} GB/s", va="center", fontsize=8,
                color=INK)
    ax.set_xlim(0.7, 10.2)
    ax.set_ylim(0, max(st_bw + [ax.get_ylim()[1]]) * 1.1)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xlabel("OpenMP threads", color=INK)
    ax.set_ylabel("achieved bandwidth (GB/s)", color=INK)
    ax.set_title(f"Kernel memory bandwidth vs STREAM, {grid}² (streaming byte model)",
                 color=INK, fontsize=10)
    style(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


SCALING = [(f"scale_{key}", label) for key, label in CONFIGS] + [  # (CSV prefix, label)
    ("t34s3_tuned", "multicolour GS + multigrid p', tuned relaxation (T34)"),
]


def fig_grid_scaling(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for k, (prefix, label) in enumerate(SCALING):
        # Non-converged points are dropped (T34 tuned diverges at 32^2).
        rows = [r for r in latest(results, prefix) if r["converged"] == "True"]
        grids = sorted({int(r["grid"]) for r in rows})
        cells = [g * g for g in grids]
        its = [med(rows, "iters", grid=g) for g in grids]
        secs = [med(rows, "total_seconds", grid=g) for g in grids]
        for ax, ys in zip(axes, (its, secs)):
            ax.plot(cells, ys, color=SERIES[k], marker=MARKERS[k], markersize=7, linewidth=2,
                    label=label)
    for ax, ylabel in zip(axes, ("outer SIMPLE iterations", "time to solution (s), 1 thread*")):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("cells", color=INK)
        ax.set_ylabel(ylabel, color=INK)
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 0.02))
    fig.text(0.99, -0.1, "* first 3 series provisional (machine was shared); tuned diverges at 32²",
             ha="right", fontsize=7, color=MUTED)
    fig.suptitle("Cost to convergence vs grid size, Re 1000 (running-max residual < 1e-5)",
                 color=INK, fontsize=10)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def latest_glob(results, pattern):
    files = sorted(glob.glob(os.path.join(results, pattern)), key=os.path.getmtime)
    if not files:
        return None
    with open(files[-1], newline="") as fh:
        return list(csv.DictReader(fh))


def fig_cpu_scaling(results, path, grids=(512, 1024, 2048, 4096)):
    """T36: strong scaling on 32 cores, one line per grid, STREAM as the bandwidth ceiling."""
    stream = latest_glob(results, "cluster_*_stream.csv")
    runs = {g: latest_glob(results, f"cluster_*_g{g}_2*.csv") for g in grids}
    if stream is None or any(r is None for r in runs.values()):
        print("skip cpu_scaling.png: no cluster_* results")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    tmax = 32
    ax.plot([1, tmax], [1, tmax], color=GRID, linewidth=2, linestyle="--", zorder=1)
    ax.text(tmax - 1, tmax - 1.2, "ideal", color=MUTED, fontsize=8, ha="right", va="top", rotation=0)
    st = {int(r["threads"]): float(r["GB_per_s"]) for r in stream}
    ts = sorted(st)
    ax.plot(ts, [st[t] / st[1] for t in ts], color=MUTED, linewidth=1.5, linestyle=":", zorder=2)
    ax.text(ts[-1] + 0.4, st[ts[-1]] / st[1], "STREAM\nbandwidth", color=MUTED, fontsize=8,
            va="center")
    for k, g in enumerate(grids):
        rows = runs[g]
        threads = sorted({int(r["threads"]) for r in rows})
        t1 = med(rows, "seconds_per_iter", threads=1)
        sp = [t1 / med(rows, "seconds_per_iter", threads=t) for t in threads]
        ax.plot(threads, sp, color=SERIES[k], marker=MARKERS[k], markersize=7, linewidth=2,
                label=f"{g}²", zorder=3,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.text(threads[-1] + 0.4, sp[-1], f"{sp[-1]:.1f}×", color=INK, fontsize=8,
                va="center")
    ax.set_xlim(0, tmax + 6)
    ax.set_ylim(0, tmax + 1)
    ax.set_xticks([1, 8, 16, 24, 32])
    ax.set_xlabel("OpenMP threads (Xeon 6972P, 2 NUMA domains)", color=INK)
    ax.set_ylabel("speedup vs 1 thread", color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left", title="grid", title_fontsize=8)
    ax.set_title("Strong scaling on 32 cores: multicolour GS + multigrid p', fixed work",
                 color=INK, fontsize=10)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_numa(results, path):
    """T37 + T38: NUMA page placement on the cluster. Left: 4096² speedup per variant (one job, same
    cores). Right: multigrid solve_p time before/after T38 (another job)."""
    t37 = {v: latest_glob(results, f"numa_*_{v}_g4096_*.csv") for v in ("old", "ilv", "new")}
    t38 = {(v, g): latest_glob(results, f"mgnuma_*_{v}_g{g}_*.csv")
           for v in ("ft", "mg") for g in (4096, 2048)}
    if any(r is None for r in list(t37.values()) + list(t38.values())):
        print("skip numa.png: no numa_*/mgnuma_* results")
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    a.plot([1, 32], [1, 32], color=GRID, linewidth=2, linestyle="--", zorder=1)
    a.text(31, 30.5, "ideal", color=MUTED, fontsize=8, ha="right", va="top")
    t1 = med(t37["old"], "seconds_per_iter", threads=1)
    labels = {"old": "serial first touch (before)", "ilv": "numactl --interleave=all",
              "new": "parallel first touch (T37)"}
    for k, v in enumerate(("old", "ilv", "new")):
        rows = t37[v]
        th = sorted({int(r["threads"]) for r in rows})
        sp = [t1 / med(rows, "seconds_per_iter", threads=t) for t in th]
        a.plot(th, sp, color=SERIES[k], marker=MARKERS[k], markersize=7, linewidth=2,
               label=labels[v], markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        a.text(th[-1] + 0.6, sp[-1], f"{sp[-1]:.1f}×", color=INK, fontsize=8, va="center")
    a.set_xlim(0, 36)
    a.set_ylim(0, 33)
    a.set_xticks([1, 8, 16, 24, 32])
    a.set_xlabel("OpenMP threads (25 in one NUMA domain, then 7 in another)", color=INK)
    a.set_ylabel("speedup vs serial first touch, 1 thread", color=INK)
    a.set_title("Page placement, 4096² (T37)", color=INK, fontsize=10)
    a.legend(frameon=False, fontsize=8, loc="upper left")
    style(a)

    threads = (16, 24, 32)
    w = 0.38
    xs, ticks = [], []
    for gi, g in enumerate((4096, 2048)):
        for ti, t in enumerate(threads):
            x = gi * (len(threads) + 1) + ti
            ft = med(t38[("ft", g)], "solve_p", threads=t)
            mg = med(t38[("mg", g)], "solve_p", threads=t)
            for k, (val, off) in enumerate(((1.0, -w / 2), (mg / ft, w / 2))):
                b.bar(x + off, val, width=w - 0.04, color=SERIES[k], zorder=3)
            b.text(x + w / 2, mg / ft + 0.015, f"{ft / mg:.2f}×", ha="center", fontsize=8,
                   color=INK)
            xs.append(x)
            ticks.append(f"{t}\n{g}²" if ti == 1 else f"{t}")
    b.set_xticks(xs, ticks)
    b.set_ylim(0, 1.15)
    b.set_xlabel("threads / grid", color=INK)
    b.set_ylabel("multigrid solve_p time (T37 = 1)", color=INK)
    b.set_title("Multigrid first touch (T38), 8 + 24 cores", color=INK, fontsize=10)
    b.legend(handles=[Patch(color=SERIES[0], label="Field2D first touch only (T37)"),
                      Patch(color=SERIES[1], label="+ multigrid arrays and loops (T38)")],
             frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    style(b)
    b.grid(False, axis="x")
    fig.suptitle("NUMA-aware memory placement on a dual-socket node (Xeon 6972P, Linux first touch)",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_gpu_kernels(results, path):
    """T24 baseline: achieved bandwidth per kernel (rocprofv3 counters) against device STREAM."""
    rows = latest_glob(results, "gpu_prof_summary_*_g2048.csv")
    stream = latest_glob(results, "gpu_prof_*/stream_hip.csv")
    if rows is None or stream is None:
        print("skip gpu_kernels.png: no gpu_prof_summary_*/stream_hip results")
        return
    copy = next(float(r["GB_per_s"]) for r in stream if r["kernel"] == "copy")
    top = rows[:7]
    names = [r["kernel"].removeprefix("k_") for r in top][::-1]
    gbs = [float(r["GB_per_s"]) / 1e3 for r in top][::-1]
    pct = [float(r["time_pct"]) for r in top][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(names, gbs, height=0.6, color=SERIES[0], zorder=3)
    ax.axvline(copy / 1e3, color=MUTED, linestyle="--", linewidth=1.5, zorder=4)
    ax.text(copy / 1e3, len(names) - 0.35, f" device STREAM copy {copy / 1e3:.2f} TB/s",
            color=MUTED, fontsize=8, va="bottom")
    for y, (g, p) in enumerate(zip(gbs, pct)):
        ax.text(g - 0.06, y, f"{g:.2f} TB/s", va="center", ha="right", fontsize=8, color="white")
        ax.text(copy / 1e3 + 0.08, y, f"{p:.0f}% of GPU time", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, copy / 1e3 * 1.45)
    ax.set_ylim(-0.6, len(names) + 0.2)
    ax.set_xlabel("achieved bandwidth, L2-miss bytes (rocprofv3 FETCH_SIZE + WRITE_SIZE)",
                  color=INK)
    ax.set_title("MI300A kernels at 2048²: the sweeps run at 91–95 % of STREAM (T24 baseline)",
                 color=INK, fontsize=10)
    style(ax)
    ax.grid(False, axis="y")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# Ghia Re1000 128² to convergence (5919 outer iterations), multicolour GS + red-black p',
# identical residual history: bench/slurm/logs/simple-gpu-test-26745839.out (T23).
GPU_VS_CPU = [("MI300A node: 8 CPU cores (OpenMP)", 5.83), ("same node: 1 MI300A GPU (HIP)", 2.19)]


def fig_gpu_vs_cpu(path):
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    names = [n for n, _ in GPU_VS_CPU][::-1]
    secs = [s for _, s in GPU_VS_CPU][::-1]
    ax.barh(names, secs, height=0.55, color=[SERIES[1], SERIES[0]], zorder=3)
    ax.text(secs[1] + 0.08, 1, f"{secs[1]:.2f} s", va="center", fontsize=9, color=INK)
    ax.text(secs[0] + 0.08, 0, f"{secs[0]:.2f} s  ({secs[1] / secs[0]:.1f}× faster)",
            va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(secs) * 1.25)
    ax.set_xlabel("time to converged solution, Re 1000, 128² (seconds)", color=INK)
    ax.set_title("GPU port (T23): same algorithm, same 5919 iterations", color=INK, fontsize=10)
    style(ax)
    ax.grid(False, axis="y")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="bench/results")
    ap.add_argument("--out", default="docs/figures/performance/hpc")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_time_to_solution(args.results, os.path.join(args.out, "time_to_solution.png"))
    fig_bandwidth(args.results, os.path.join(args.out, "bandwidth.png"))
    fig_grid_scaling(args.results, os.path.join(args.out, "grid_scaling.png"))
    fig_cpu_scaling(args.results, os.path.join(args.out, "cpu_scaling.png"))
    fig_numa(args.results, os.path.join(args.out, "numa.png"))
    fig_gpu_kernels(args.results, os.path.join(args.out, "gpu_kernels.png"))
    fig_gpu_vs_cpu(os.path.join(args.out, "gpu_vs_cpu.png"))


if __name__ == "__main__":
    main()
