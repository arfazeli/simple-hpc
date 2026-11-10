#!/usr/bin/env python3
"""Per-kernel summary of a T24 rocprofv3 profile: time share and launch time (kernel trace) plus
bytes read/written per launch (FETCH_SIZE / WRITE_SIZE passes), achieved GB/s.

Usage: python3 bench/gpu_prof_summary.py <trace dir> <counter dir> [grid] > summary.csv
  trace dir:   holds trace_<grid>_kernel_trace.csv     (gpu_profile.sbatch, step trace)
  counter dir: holds counters_{1,2}_counter_collection.csv (step pmc: FETCH_SIZE, WRITE_SIZE)
"""
import collections
import csv
import re
import sys


def name(s):
    return (re.search(r"::(k_\w+)", s) or re.search(r"(__amd_rocclr_\w+)", s)).group(1)


def main():
    tdir, pdir = sys.argv[1], sys.argv[2]
    grid = sys.argv[3] if len(sys.argv) > 3 else "2048"
    dur = collections.defaultdict(list)
    with open(f"{tdir}/trace_{grid}_kernel_trace.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            dur[name(r["Kernel_Name"])].append(int(r["End_Timestamp"]) - int(r["Start_Timestamp"]))
    busy = sum(sum(v) for v in dur.values())
    byts = collections.defaultdict(lambda: [0.0, 0.0])
    calls = collections.Counter()
    for k, f in ((0, 1), (1, 2)):  # FETCH_SIZE, WRITE_SIZE in KiB per dispatch
        with open(f"{pdir}/counters_{f}_counter_collection.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                byts[name(r["Kernel_Name"])][k] += float(r["Counter_Value"]) * 1024
                calls[name(r["Kernel_Name"])] += k == 0
    w = csv.writer(sys.stdout)
    w.writerow(["kernel", "time_pct", "us_per_launch", "MB_read", "MB_written", "GB_per_s"])
    for k, v in sorted(dur.items(), key=lambda kv: -sum(kv[1])):
        if not calls[k]:
            continue
        us = sum(v) / len(v) / 1e3
        rd, wr = (x / calls[k] for x in byts[k])
        w.writerow([k, f"{sum(v) / busy * 100:.1f}", f"{us:.1f}", f"{rd / 1e6:.1f}",
                    f"{wr / 1e6:.1f}", f"{(rd + wr) / us / 1e3:.0f}"])


if __name__ == "__main__":
    main()
