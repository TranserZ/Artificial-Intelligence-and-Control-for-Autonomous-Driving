"""klppo2_s6 학습 종료 후 체크포인트 A~H 평가 + 100판 재평가."""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.official_metrics import checkpoint_sort_key  # noqa: E402

REPORT = os.path.join(
    r"C:\Users\USER\Desktop\청계로드\docs\reports",
    "2026-11-25_s6_2차_결과.md",
)
EVAL_DIR = os.path.join(ROOT, "results", "overnight_eval_klppo2_s6")
SEEDS_DEV = "900100-900129"
SEEDS_HOLD = "900130-900189,900000-900019"
BASELINE_OFF = 0.567
WORKERS = 7


def wait_runs():
    logs = ["klppo2_s6_s0.err", "klppo2_s6_s1.err"]
    while True:
        alive = False
        for name in logs:
            try:
                out = subprocess.check_output(
                    ["powershell", "-NoProfile", "-Command",
                     f"Get-Content -Tail 3 '{os.path.join(ROOT, 'results', 'logs', name)}'"],
                    text=True, timeout=10,
                )
                if "학습 속도" not in out and "저장 완료" not in out:
                    alive = True
            except Exception:
                pass
        try:
            n = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-Process python -ErrorAction SilentlyContinue).Count"],
                text=True, timeout=10,
            ).strip()
            if int(n or 0) >= 2:
                alive = True
        except Exception:
            pass
        if not alive:
            return
        time.sleep(120)


def eval_ckpt(run, step_path, seeds, tag):
    os.environ["CGC_STAGE"] = "6"
    os.environ["OBS_SCAN"] = "1"
    jout = os.path.join(EVAL_DIR, f"{run}_{os.path.basename(step_path).replace('.pt','')}.json")
    subprocess.check_call([
        sys.executable, os.path.join(ROOT, "scripts", "drive_parallel.py"),
        "--model", step_path, "--seeds", seeds, "--shield", "shadow",
        "--workers", str(WORKERS), "--json-out", jout, "--title", tag,
    ], cwd=ROOT)
    return json.load(open(jout, encoding="utf-8"))


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    wait_runs()
    records = []
    for run in ("klppo2_s6_s0", "klppo2_s6_s1"):
        for ck in sorted(glob.glob(os.path.join(ROOT, "results", run, "step*.pt"))):
            data = eval_ckpt(run, ck, SEEDS_DEV, f"{run} dev")
            stats = data["stats"]
            stats["run"] = run
            stats["ckpt"] = os.path.basename(ck)
            records.append(stats)
    records.sort(key=lambda s: checkpoint_sort_key(s, BASELINE_OFF))
    best = records[0] if records else None
    hold = None
    if best and best.get("collision_count", 99) <= 1:
        hold = eval_ckpt(
            best["run"],
            os.path.join(ROOT, "results", best["run"], best["ckpt"]),
            SEEDS_HOLD, "holdout 100",
        )
    lines = [
        "# 6단계 KL-PPO 2차 결과",
        "",
        f"평가 JSON: `{EVAL_DIR}`",
        "",
        "## 체크포인트 (개발 30판, A~H)",
        "",
        "| run | ckpt | 공식 완주 | 완벽 완주 | 충돌 | 무한대기+TO | wrong_lane |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for s in records:
        n = s["episodes"]
        lines.append(
            f"| {s['run']} | {s['ckpt']} | {s['official_complete']*n:.0f}/{n} "
            f"({s['official_complete']:.1%}) | {s['perfect_complete']*n:.0f}/{n} "
            f"({s['perfect_complete']:.1%}) | {s['collision_count']}/{n} "
            f"({s['collision_rate']:.1%}) | {s['inf_wait_episodes']} | {s['wrong_lane_count']} |"
        )
    lines.append("")
    lines.append("<!-- SUMMARY_START -->")
    if best is None:
        lines.append("불가: 체크포인트 평가 없음.")
    elif hold and hold["stats"]["collision_count"] <= 4:
        hs = hold["stats"]
        n = hs["episodes"]
        lines.append(
            f"6단계 최종 가능: {best['run']} {best['ckpt']}, 100판 충돌 {hs['collision_count']}판"
            f" ({hs['collision_rate']:.0%}), 공식 완주 {hs['official_complete']:.0%}, "
            f"완벽 {hs['perfect_complete']:.0%}, 무한대기+TO {hs['inf_wait_episodes']}판."
        )
    else:
        lines.append(
            f"불가: 최선 {best['run']} {best['ckpt']} 충돌 {best['collision_rate']:.1%} "
            f"({best['collision_count']}/{best['episodes']}), 공식 {best['official_complete']:.1%}, "
            f"완벽 {best['perfect_complete']:.1%}."
        )
    lines.append("<!-- SUMMARY_END -->")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote", REPORT)


if __name__ == "__main__":
    main()