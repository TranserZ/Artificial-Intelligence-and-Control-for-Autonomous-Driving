"""collect_pomdp_data 20판 스모크: 용량·속도·10^7 스텝 추정."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    os.environ.setdefault("CGC_STAGE", "6")
    os.environ.setdefault("OBS_SCAN", "1")
    out_dir = os.path.join(ROOT, "results", "pomdp_smoke")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(ROOT, "collect_pomdp_data.py"),
        "--episodes", "20", "--out", out_dir,
    ]
    t0 = time.perf_counter()
    subprocess.check_call(cmd, cwd=ROOT)
    elapsed = time.perf_counter() - t0
    sizes = []
    steps = 0
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            sizes.append(os.path.getsize(path))
        if name.endswith(".json") and name != "manifest.json":
            try:
                d = json.load(open(path, encoding="utf-8"))
                steps += int(d.get("steps", d.get("length", 0)))
            except Exception:
                pass
    per_ep = steps / 20 if steps else 0
    bytes_total = sum(sizes)
    per_step = bytes_total / max(steps, 1)
    est_10m_steps = per_step * 10_000_000
    est_sec = (elapsed / 20) * 10_000_000 / max(per_ep, 1)
    print(json.dumps({
        "episodes": 20,
        "wall_sec": round(elapsed, 1),
        "steps_recorded": steps,
        "bytes_total": bytes_total,
        "bytes_per_step_est": per_step,
        "est_10M_step_bytes": est_10m_steps,
        "est_10M_step_hours": est_sec / 3600,
        "steps_per_sec_collect": steps / max(elapsed, 1e-6),
    }, indent=2))


if __name__ == "__main__":
    main()