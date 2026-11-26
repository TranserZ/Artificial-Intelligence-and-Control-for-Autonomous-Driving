"""5·6단계 KL-PPO 보고 80판 짝 비교 후 docs/reports 마크다운 생성."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.drive_parallel import (  # noqa: E402
    evaluate,
    format_table,
    parse_seeds,
)

DONGYEON = r"C:\Users\USER\Desktop\청계로드-dongyeon-d54a761\models\dongyeon_bc"
REPORT_SEEDS = "900130-900189,900000-900019"
REPORT_DIR_MAIN = r"C:\Users\USER\Desktop\청계로드\docs\reports"
REPORT_DIR_LOCAL = os.path.join(ROOT, "docs", "reports")
RESULTS = os.path.join(ROOT, "results", "report_eval_s56")
RESUME_LOG = os.path.join(ROOT, "results", "logs", "resume_eval_after_s56.out")

# 개발 30판 기준 선정 (공식 완주 → 충돌 ↓ → 완벽 완주 ↓)
CKPT = {
    "klppo1_s6_s0": (300000, "300k: 개발 공식 73.3%로 최고, 충돌 20%, 완벽 73.3%"),
    "klppo1_s6_s1": (200000, "200k: 공식 66.7%로 최고(300k 60%), 충돌 13.3%로 최저"),
    "klppo1_s5_s0": (250000, "250k: 공식 93.3%로 최고, 충돌 3.3%, 완벽 86.7%"),
}


def ckpt_path(run: str, step: int) -> str:
    return os.path.join(ROOT, "results", run, f"step{step // 1000}k.pt")


def wait_resume_done(timeout_h: float = 12.0):
    markers = ("4단계 이어하기 끝", "resume_klppo_eval 끝")
    deadline = time.time() + timeout_h * 3600
    while time.time() < deadline:
        if os.path.isfile(RESUME_LOG):
            tail = open(RESUME_LOG, encoding="utf-8", errors="replace").read()[-8000:]
            if any(m in tail for m in markers):
                return
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                 "Where-Object { $_.CommandLine -match 'resume_klppo_eval' } | Measure-Object | "
                 "Select-Object -ExpandProperty Count"],
                text=True,
                timeout=30,
            ).strip()
            if out == "0":
                time.sleep(5)
                return
        except Exception:
            pass
        time.sleep(60)
    print("resume 대기 시간 초과 — 80판 평가를 계속한다.", flush=True)


def official_ok(reason: str) -> bool:
    return reason in ("completed", "missed_stop")


def paired_block(label: str, rows_a, rows_b) -> list[str]:
    by_a = {r["seed"]: r for r in rows_a}
    fixed, broken = [], []
    for r in rows_b:
        a = by_a[r["seed"]]
        if (not official_ok(a["end_reason"])) and official_ok(r["end_reason"]):
            fixed.append(r["seed"])
        if official_ok(a["end_reason"]) and (not official_ok(r["end_reason"])):
            broken.append(r["seed"])
    lines = [
        f"#### {label}",
        "",
        f"- BC 실패 → PPO 성공: {len(fixed)}판 {fixed}",
        f"- BC 성공 → PPO 실패: {len(broken)}판 {broken}",
        "",
        "실패 위치 (PPO가 공식 완주가 아닌 판, `place`=불법주정차 핫스폿 또는 정류장):",
        "",
        "| 시드 | PPO 종료 | place | BC 종료 | BC place |",
        "|---:|---|---|---|---|",
    ]
    for r in rows_b:
        if official_ok(r["end_reason"]):
            continue
        a = by_a[r["seed"]]
        lines.append(
            f"| {r['seed']} | {r['end_reason']} | {r.get('place') or r['nearest_stop_id']} "
            f"| {a['end_reason']} | {a.get('place') or a['nearest_stop_id']} |"
        )
    lines.append("")
    return lines


def stats_row(name: str, stats: dict) -> str:
    n = stats["episodes"]
    off = stats["official_complete"]
    perf = stats["perfect_complete"]
    return (
        f"| {name} | {off * n:.0f}/{n} ({off:.1%}) | {perf * n:.0f}/{n} ({perf:.1%}) | "
        f"{stats['stop_success_mean']:.2f} | {stats['collision'] * n:.0f}/{n} ({stats['collision']:.1%}) | "
        f"{stats['red_light'] * n:.0f}/{n} ({stats['red_light']:.1%}) | "
        f"{stats['timeout'] * n:.0f}/{n} ({stats['timeout']:.1%}) | "
        f"{stats['yield_steps']} | {stats['intervention']:.1%} | {stats['avg_speed']:.2f} m/s |"
    )


def run_stage(stage: int, obs_scan: str, student_pt: str, ppo_runs: list[str], workers: int):
    os.environ["CGC_STAGE"] = str(stage)
    os.environ["OBS_SCAN"] = obs_scan
    seeds = parse_seeds(REPORT_SEEDS)
    print(f"=== stage {stage} OBS_SCAN={obs_scan} seeds={len(seeds)} ===", flush=True)
    rows_stu, stu = evaluate(
        student_pt, seeds, "shadow", workers, title=f"s{stage} 학생",
        json_out=os.path.join(RESULTS, f"s{stage}_student_report80.json"),
    )
    ppo_out = []
    for run in ppo_runs:
        step, _why = CKPT[run]
        path = ckpt_path(run, step)
        title = f"{run} @{step}"
        jpath = os.path.join(RESULTS, f"{run}_step{step}_report80.json")
        rows, stats = evaluate(path, seeds, "shadow", workers, title=title, json_out=jpath)
        ppo_out.append((run, step, rows, stats))
    return rows_stu, stu, ppo_out


def write_report(rows_stu6, stu6, ppo6, rows_stu5, stu5, ppo5):
    os.makedirs(REPORT_DIR_MAIN, exist_ok=True)
    os.makedirs(REPORT_DIR_LOCAL, exist_ok=True)
    path = os.path.join(REPORT_DIR_MAIN, "2026-11-24_kl_ppo_56단계_결과.md")
    lines = [
        "# KL-PPO 5·6단계 보고 80판 짝 비교 (2026-11-24)",
        "",
        f"작업 트리 `{ROOT}`. 출발 학생 `{DONGYEON}`.",
        "",
        "보고 판: 900130–900189(60) + 900000–900019(20) = 80판. 결정적, shield shadow.",
        "",
        "## 개발 30판에서 고른 체크포인트",
        "",
        "기준: 공식 완주율 ↓, 동률이면 충돌률 ↑, 그다음 완벽 완주율.",
        "",
    ]
    for run, (step, why) in CKPT.items():
        lines.append(f"- **{run}** → `step{step // 1000}k.pt`: {why}")
    lines.extend(["", "## 6단계 (OBS_SCAN=1)", ""])

    lines.extend([
        "| 모델 | 공식 완주 | 완벽 완주 | 정차 성공 | 충돌 | 적색 | 시간초과 | 양보 위반 스텝 | 수동개입률 | 평균 속도 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        stats_row("s6_student_obsfix", stu6),
    ])
    for run, step, _rows, stats in ppo6:
        lines.append(stats_row(f"{run} @ {step}", stats))
    lines.append("")

    for run, step, rows, stats in ppo6:
        lines.append(format_table(f"{run} @ {step}", stats))
        lines.extend(paired_block(f"s6_student_obsfix vs {run} @ {step}", rows_stu6, rows))

    lines.extend(["## 5단계 (OBS_SCAN=0)", ""])
    lines.extend([
        "| 모델 | 공식 완주 | 완벽 완주 | 정차 성공 | 충돌 | 적색 | 시간초과 | 양보 위반 스텝 | 수동개입률 | 평균 속도 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        stats_row("s5_student", stu5),
    ])
    run, step, rows, stats = ppo5[0]
    lines.append(stats_row(f"{run} @ {step}", stats))
    lines.append("")
    lines.append(format_table(f"{run} @ {step}", stats))
    lines.extend(paired_block(f"s5_student vs {run} @ {step}", rows_stu5, rows))

    # 개발 vs 보고 한 줄
    dev_s6_stu = 56.7  # from overnight s6 student 30판
    dev_s6_s0_300 = 73.3
    dev_s6_s1_best = 66.7  # 200k
    rep_s6_stu = stu6["official_complete"] * 100
    rep_s6_s0 = ppo6[0][3]["official_complete"] * 100
    rep_s6_s1 = ppo6[1][3]["official_complete"] * 100
    dev_s5_stu = None  # load from json if needed
    dev_path = os.path.join(ROOT, "results", "overnight_eval_s56", "s5_student.json")
    if os.path.isfile(dev_path):
        dev_s5_stu = json.load(open(dev_path, encoding="utf-8"))["stats"]["official_complete"] * 100
    dev_s5_250 = 93.3
    rep_s5_stu = stu5["official_complete"] * 100
    rep_s5_ppo = ppo5[0][3]["official_complete"] * 100

    verdict = (
        f"6단계: 개발에서 학생 {dev_s6_stu:.1f}% → KL s0 300k {dev_s6_s0_300:.1f}%·s1 200k {dev_s6_s1_best:.1f}%였고, "
        f"보고 80판에서는 학생 {rep_s6_stu:.1f}%, s0 {rep_s6_s0:.1f}%, s1 {rep_s6_s1:.1f}%다. "
    )
    if dev_s5_stu is not None:
        verdict += (
            f"5단계: 개발 학생 {dev_s5_stu:.1f}%·KL 250k {dev_s5_250:.1f}% → 보고 학생 {rep_s5_stu:.1f}%·KL {rep_s5_ppo:.1f}%. "
        )
    better_6 = rep_s6_s0 > rep_s6_stu or rep_s6_s1 > rep_s6_stu
    better_5 = rep_s5_ppo > rep_s5_stu
    if better_6 and better_5:
        verdict += "개발에서 본 KL-PPO 대비 학생 우위가 보고 판에서도 대체로 유지된다."
    elif better_6 or better_5:
        verdict += "개발 개선이 보고 판에서는 단계마다 다르게 나타난다(한쪽만 유지)."
    else:
        verdict += "보고 판에서는 개발 대비 개선이 약해지거나 뒤집힌다."

    lines.extend([
        "## 한 줄",
        "",
        "<!-- SUMMARY_START -->",
        verdict,
        "<!-- SUMMARY_END -->",
        "",
        f"JSON: `{RESULTS}`. 생성 시각 {datetime.now().isoformat(timespec='seconds')}.",
    ])
    text = "\n".join(lines)
    for d in (REPORT_DIR_MAIN, REPORT_DIR_LOCAL):
        with open(os.path.join(d, "2026-11-24_kl_ppo_56단계_결과.md"), "w", encoding="utf-8") as f:
            f.write(text)
    print(f"report -> {path}", flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-wait", action="store_true", help="resume 종료 대기 생략")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if not args.skip_wait:
        print("resume_klppo_eval 종료 대기...", flush=True)
        wait_resume_done()
    workers = min(args.workers, 7)
    s6_stu = os.path.join(DONGYEON, "s6_student_obsfix.pt")
    s5_stu = os.path.join(DONGYEON, "s5_student.pt")
    rows_stu6, stu6, ppo6 = run_stage(
        6, "1", s6_stu, ["klppo1_s6_s0", "klppo1_s6_s1"], workers)
    rows_stu5, stu5, ppo5 = run_stage(5, "0", s5_stu, ["klppo1_s5_s0"], workers)
    write_report(rows_stu6, stu6, ppo6, rows_stu5, stu5, ppo5)


if __name__ == "__main__":
    main()