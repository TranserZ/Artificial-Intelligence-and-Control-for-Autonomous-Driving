#!/usr/bin/env python3
"""Low-speed stall patterns in final_s5 step250k traces."""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_TRACE_DIR = (
    Path(ROOT)
    / "results"
    / "overnight_s6_final"
    / "final_s5_step250k_scan0"
    / "traces"
)

SEEDS = [
    900304,
    900317,
    900322,
    900331,
    900342,
    900351,
    900365,
    900369,
    900386,
    900388,
    900389,
    900399,
    900310,
    900325,
    900338,
    900345,
    900346,
    900347,
    900358,
    900384,
]

MIN_STALL_STEPS = 120
SPEED_THRESH = 0.1
JUMP_M = 20.0
POST_STEPS = 5

SUMO_ENV = Path(ROOT) / "env" / "sumo_env.py"
SUMO_ENV_PATTERNS = ("teleport", "time-to-teleport", "getStartingTeleportIDList")


def find_stall_runs(speed: np.ndarray, min_steps: int) -> list[tuple[int, int]]:
    """Inclusive index ranges where speed < SPEED_THRESH for at least min_steps."""
    runs: list[tuple[int, int]] = []
    n = len(speed)
    i = 0
    while i < n:
        if speed[i] >= SPEED_THRESH:
            i += 1
            continue
        j = i
        while j < n and speed[j] < SPEED_THRESH:
            j += 1
        if j - i >= min_steps:
            runs.append((i, j - 1))
        i = j
    return runs


def dist_xy(x0: float, y0: float, x1: float, y1: float) -> float:
    if math.isnan(x0) or math.isnan(y0) or math.isnan(x1) or math.isnan(y1):
        return 0.0
    return math.hypot(x1 - x0, y1 - y0)


def analyze_seed(trace_dir: Path, seed: int) -> None:
    path = trace_dir / f"{seed}.npz"
    if not path.is_file():
        print(f"MISSING\t{seed}\t{path}")
        return

    with np.load(path) as z:
        speed = z["speed"]
        time = z["time"]
        edge = z["edge"]
        lane_cmd = z["lane_cmd"]
        accel_cmd = z["accel_cmd"]
        bus_stop = z["bus_stop"]
        neighbors = z["neighbors_50m"]
        end_reason = str(z["end_reason"].item())
        x = z["x"]
        y = z["y"]

    runs = find_stall_runs(speed, MIN_STALL_STEPS)
    if not runs:
        print(f"STALL_NONE\t{seed}\tend_reason={end_reason}")
        return

    for run_i, (a, b) in enumerate(runs, 1):
        t0 = float(time[a])
        t1 = float(time[b])
        length_sec = t1 - t0 + (float(time[1] - time[0]) if len(time) > 1 else 0.5)
        lc_nz = int(np.count_nonzero(lane_cmd[a : b + 1]))
        accel_mean = float(np.mean(accel_cmd[a : b + 1]))
        bs1 = float(bus_stop[a, 1]) if bus_stop.ndim == 2 else float(bus_stop[a])
        neigh_max = int(np.max(neighbors[a : b + 1]))
        edge0 = str(edge[a])

        print(
            f"STALL\t{seed}\t{t0:.1f}\t{t1:.1f}\t{length_sec:.1f}\t{edge0}\t"
            f"{lc_nz}\t{accel_mean:.6f}\t{bs1:.6f}\t{neigh_max}\t{end_reason}"
        )

        post_start = b + 1
        post_end = min(post_start + POST_STEPS, len(time))
        jump = False
        for k in range(post_start, post_end):
            prev = k - 1
            d = dist_xy(float(x[prev]), float(y[prev]), float(x[k]), float(y[k]))
            if d >= JUMP_M:
                jump = True
            print(
                f"POST\t{seed}\tstall#{run_i}\t{float(time[k]):.1f}\t{edge[k]}\t"
                f"{float(x[k]):.6f}\t{float(y[k]):.6f}\td={d:.3f}m"
            )
        if post_start >= len(time):
            print(f"POST\t{seed}\tstall#{run_i}\t(end of trace, no post steps)")
        elif jump:
            print(f"JUMP\t{seed}\tstall#{run_i}")


def dump_sumo_env_teleport_lines() -> None:
    print("=== env/sumo_env.py (teleport | time-to-teleport | getStartingTeleportIDList) ===")
    text = SUMO_ENV.read_text(encoding="utf-8")
    needles = SUMO_ENV_PATTERNS
    for lineno, line in enumerate(text.splitlines(), 1):
        low = line.lower()
        if any(n.lower() in low for n in needles):
            print(f"{lineno}\t{line.rstrip()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--seeds", type=str, default="")
    args = parser.parse_args()

    seeds = SEEDS
    if args.seeds.strip():
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    for seed in seeds:
        analyze_seed(args.trace_dir, seed)

    print()
    dump_sumo_env_teleport_lines()


if __name__ == "__main__":
    main()