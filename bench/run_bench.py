#!/usr/bin/env python3
"""Fixed-work benchmark runner for the SIMPLE solver.

Runs QUICK / diffusion_order 4 / Re 1000 square cavities for a fixed number
of outer iterations (error_tolerance 0), records per-phase times from
timing.csv, and appends one row per run to bench/results/<tag>_<stamp>.csv.
"""
import argparse
import csv
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "cavity_re1000_80.cfg"
RESULTS = REPO / "bench" / "results"

OVERRIDES = {
    "convection_scheme": "QUICK",
    "diffusion_order": "4",
    "Re": "1000",
    "error_tolerance": "0",
}


def make_config(grid, iters, out_dir, extra=None):
    """Return config text: the template with the benchmark keys overridden."""
    values = dict(OVERRIDES)
    values.update(extra or {})
    values["N_cells_x"] = str(grid)
    values["N_cells_y"] = str(grid)
    values["maxit"] = str(iters)
    values["output_dir"] = str(out_dir)
    seen = set()
    lines = []
    for line in TEMPLATE.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in values:
                lines.append(f"{key} = {values[key]}")
                seen.add(key)
                continue
        lines.append(line)
    for key, val in values.items():
        if key not in seen:
            lines.append(f"{key} = {val}")
    return "\n".join(lines) + "\n"


def read_timing(path):
    """Return ordered list of (phase, seconds) from timing.csv, including total."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [(r["phase"], float(r["seconds"])) for r in rows]


def cpu_name():
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True, check=True).stdout.strip()
        if out:
            return out
    except (OSError, subprocess.CalledProcessError):
        pass
    try:  # Linux
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def read_run_info(path):
    """Parse run_info.json with the json module; empty dict if absent."""
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


def run_one(exe, grid, threads, iters, extra=None):
    """Run one benchmark; return (phase_list, run_info dict)."""
    with tempfile.TemporaryDirectory(prefix="bench_") as tmp:
        tmp_path = Path(tmp)
        out_dir = tmp_path / "out"
        cfg = tmp_path / "bench.cfg"
        cfg.write_text(make_config(grid, iters, out_dir, extra))
        env = dict(os.environ, OMP_NUM_THREADS=str(threads))
        proc = subprocess.run([str(exe), str(cfg)], env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"run failed (grid={grid}, threads={threads}):\n"
                               + proc.stdout[-2000:] + proc.stderr[-2000:])
        phases = read_timing(out_dir / "timing.csv")
        info = read_run_info(out_dir / "run_info.json")
        return phases, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grids", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--exe", default=str(REPO / "build" / "simple"))
    ap.add_argument("--tag", default="serial")
    ap.add_argument("--cooldown", type=float, default=0.0,
                    help="seconds to sleep before each run (thermal settling)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config key (repeatable)")
    args = ap.parse_args()

    extra = {}
    for item in args.set:
        if "=" not in item:
            sys.exit(f"--set expects KEY=VALUE, got: {item}")
        key, val = item.split("=", 1)
        extra[key.strip()] = val.strip()
    exe = Path(args.exe).resolve()
    if not exe.exists():
        sys.exit(f"executable not found: {exe}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out_path = RESULTS / f"{args.tag}_{stamp}.csv"
    cpu = cpu_name()
    fallback_hash = git_hash()

    records = []
    phase_names = None
    # Repeat-major order with the thread list rotated each repeat: on a laptop that throttles
    # under sustained load, running all 1-thread runs first (cool) and 8-thread runs last (hot)
    # biases the comparison.
    for rep in range(1, args.repeats + 1):
        k = (rep - 1) % len(args.threads)
        order = args.threads[k:] + args.threads[:k]
        for grid in args.grids:
            for threads in order:
                if args.cooldown:
                    time.sleep(args.cooldown)
                phases, info = run_one(exe, grid, threads, args.iters, extra)
                if phase_names is None:
                    phase_names = [p for p, _ in phases if p not in ("total", "unaccounted")]
                times = dict(phases)
                # Solver wall time from run_info.json (stops before the serial field output, which
                # stays in the `output` column); timing.csv's `total` includes the output.
                total = float(info.get("seconds", times["total"]))
                # Outer iterations actually run (fewer than --iters if the run converged).
                iters = int(info.get("iterations", args.iters))
                cell_rate = grid * grid * iters / total
                records.append({
                    "tag": args.tag, "grid": grid, "threads": threads,
                    "repeat": rep, "iters": iters,
                    "total_seconds": f"{total:.9f}",
                    "seconds_per_iter": f"{total / iters:.9f}",
                    "cell_updates_per_s": f"{cell_rate:.6e}",
                    **{p: f"{times[p]:.9f}" for p in phase_names},
                    "cpu": cpu,
                    "compiler": info.get("compiler", "unknown"),
                    "git_hash": info.get("git_hash", fallback_hash),
                    "converged": info.get("converged", ""),
                    "cg_iters_mean": info.get("cg_iterations_mean", ""),
                })
                print(f"grid={grid} threads={threads} repeat={rep} iters={iters} "
                      f"seconds_per_iter={total / iters:.6f}", file=sys.stderr)

    fields = (["tag", "grid", "threads", "repeat", "iters", "total_seconds",
               "seconds_per_iter", "cell_updates_per_s"] + phase_names
              + ["cpu", "compiler", "git_hash", "converged", "cg_iters_mean"])
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    for grid in args.grids:
        for threads in args.threads:
            vals = [float(r["seconds_per_iter"]) for r in records
                    if r["grid"] == grid and r["threads"] == threads]
            print(f"grid={grid} threads={threads} "
                  f"median_seconds_per_iter={statistics.median(vals):.6f} "
                  f"(n={len(vals)})")
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
