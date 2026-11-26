"""5·6단계 체크포인트 평가가 끝난 뒤 4단계 남은 작업을 이어서 돌린다.

순서: klppo_s4_s1 30만, 25만, ctrl 20만, klppo_s4_s1 20만,
그다음 결정적 가속 RMS, bs_02 사례.
30만·25만 공식 평가는 결과 문서에 이미 있으면 다시 달리지 않는다.
결과는 docs/reports/2026-11-23_kl_ppo_4단계_결과.md 에만 잇는다.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import sys
import time

os.environ["CGC_STAGE"] = "4"
os.environ["OBS_SCAN"] = "0"

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from scripts.klppo_result_extra import (  # noqa: E402
    FOCUS, KL_RUNS, STUDENT, drive_model, fmt_dist, fmt_gap, fmt_speed)
from scripts.overnight_watch import plot, replace_block  # noqa: E402
from scripts.resume_klppo_eval import (  # noqa: E402
    REPORTS, ckpt, collected_from, record_of, row_line)

REPORT = r"C:\Users\USER\Desktop\청계로드\docs\reports\2026-11-23_kl_ppo_4단계_결과.md"
ORDER = (
    ("klppo_s4_s1", 300_000),
    ("klppo_s4_s1", 250_000),
    ("ctrl_s4_s0", 200_000),
    ("klppo_s4_s1", 200_000),
)
STILL_ACTIVE = 259
SEEDS = list(range(900100, 900130))
WORKERS = 7
DONE_NOTE = (
    "5·6단계 체크포인트 평가 뒤에 이어서 돌렸다. "
    "klppo_s4_s1 30만·25만 공식 평가는 이미 표에 있어 다시 달리지 않았다. "
    "ctrl_s4_s0 20만, klppo_s4_s1 20만을 평가한 뒤 "
    "네 체크포인트의 결정적 가속 RMS와 bs_02를 넣었다. "
    "klppo_s4_s1 10만·15만과 ctrl_s4_s0 5만·10만·15만은 이번 순서에 없다."
)


def alive(pid):
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(handle)
    return code.value == STILL_ACTIVE


def command_lines():
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='py.exe'\" "
        "| ForEach-Object { $_.CommandLine }"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", script],
            text=True, errors="replace", timeout=90)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"프로세스 목록 실패: {exc}", flush=True)
        return ""
    return out


def stage56_still_running(train_pids, watch_pids):
    if any(alive(pid) for pid in train_pids + watch_pids):
        return True
    text = command_lines()
    needles = ("overnight_watch_s56.py", "klppo1_s5_s0", "klppo1_s6_s0", "klppo1_s6_s1")
    return any(needle in text for needle in needles)


def wait_stage56(train_pids, watch_pids):
    while stage56_still_running(train_pids, watch_pids):
        left = [pid for pid in train_pids + watch_pids if alive(pid)]
        print(f"5·6단계 대기 {time.strftime('%H:%M:%S')} 살아 있는 PID {left}", flush=True)
        time.sleep(60)
    print("5·6단계 체크포인트 평가 프로세스 없음. 4단계 이어서 시작", flush=True)


def read_text(path):
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def extract_block(text, name):
    match = re.search(rf"<!-- {name}_START -->\n(.*?)<!-- {name}_END -->", text, re.S)
    return match.group(1) if match else ""


def write_block(name, body):
    text = replace_block(read_text(REPORT), name, body)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text)


def table_has(body, run, step):
    return re.search(rf"^\| {re.escape(run)} \| {int(step)} \|", body, re.M) is not None


def insert_before(body, marker, rows):
    fresh = [row for row in rows if row not in body]
    if not fresh:
        return body
    block = "\n".join(fresh)
    if marker in body:
        return body.replace(marker, block + "\n" + marker, 1)
    return body.rstrip() + "\n" + block + "\n"


def append_eval_row(line):
    body = extract_block(read_text(REPORT), "EVAL")
    if not body:
        raise SystemExit("4단계 결과 문서에 EVAL 블록이 없다.")
    body = insert_before(body, "\n중단 규칙:", [line])
    write_block("EVAL", body)
    refresh_plot()


def refresh_plot():
    body = extract_block(read_text(REPORT), "EVAL")
    by_run = {}
    for match in re.finditer(
            r"^\| (\S+) \| (\d+) \| ([0-9.]+)% \| ([0-9.]+)% \| ([0-9.]+)% \| ([0-9.eE+-]+) \|",
            body, re.M):
        run, step, official, perfect, collision, anchor = match.groups()
        by_run.setdefault(run, []).append((int(step), {
            "official": float(official) / 100.0,
            "perfect": float(perfect) / 100.0,
            "collision": float(collision) / 100.0,
            "anchor_rms": float(anchor),
        }))
    collected = [{"run": run, "points": points} for run, points in by_run.items()]
    figure = os.path.join(BASE, "results", "overnight_eval", "curves.png")
    try:
        plot(collected, figure)
    except Exception as exc:
        print(f"그림 갱신 실패: {exc}", flush=True)


def official_eval(run, step):
    body = extract_block(read_text(REPORT), "EVAL")
    if table_has(body, run, step):
        print(f"이미 있음 {run} {step}. 공식 평가 생략", flush=True)
        return
    from scripts.drive_parallel import evaluate
    path = ckpt(run, step)
    print(f"평가 {run} {step}", flush=True)
    started = time.perf_counter()
    try:
        _rows, stats = evaluate(
            path, SEEDS, shield="shadow", workers=WORKERS,
            deterministic=True, title=f"{run} step{step // 1000}k")
        official = stats["official_complete"]
        perfect = stats["perfect_complete"]
        collision = stats["collision"]
    except Exception as exc:
        print(f"평가 실패 {run} {step}: {exc}", flush=True)
        official = perfect = collision = None
    print(f"소요 {run} {step} {time.perf_counter() - started:.0f}초", flush=True)
    line = row_line(*record_of(run, step, official, perfect, collision))
    append_eval_row(line)
    print(f"보고서 행 추가 {run} {step}", flush=True)


def metric_lines(run, step, rows):
    sq = sum(row["sq_sum"] for row in rows)
    n_obs = sum(row["n_obs"] for row in rows)
    rms = (sq / n_obs) ** 0.5 if n_obs else float("nan")
    close = "예" if rms < 0.02 else "아니오"
    rms_line = f"| {run} | {step} | {rms:.4g} | {n_obs} | {close} |"
    fail_lines = []
    bs_lines = []
    for row in rows:
        if row["end_reason"] != "completed":
            kind = "새 실패 (학생은 완주)"
            if row["seed"] in FOCUS:
                kind += ", bs_02 시드"
            fail_lines.append(
                f"| {run} | {step} | {row['seed']} | {row['end_reason']} | "
                f"{row['nearest_stop_id'] or row['ego_edge']} | {kind} |")
        if run in KL_RUNS and row["seed"] in FOCUS:
            approach = row["approach"]
            bs_lines.append(
                f"| {run} | {step} | {row['seed']} | {fmt_gap(approach['min_gap'])} | "
                f"{fmt_speed(approach['min_speed'])} | "
                f"{fmt_dist(approach['decel_dist'], approach['seen'])} |")
    return rms_line, fail_lines, bs_lines


def append_metrics(run, step, rows):
    text = read_text(REPORT)
    rms_line, fail_lines, bs_lines = metric_lines(run, step, rows)
    rms_body = extract_block(text, "RMS")
    if not table_has(rms_body, run, step):
        write_block("RMS", rms_body.rstrip() + "\n" + rms_line + "\n")
        text = read_text(REPORT)
    fail_body = extract_block(text, "FAIL")
    fresh_fail = [line for line in fail_lines if line not in fail_body]
    if fresh_fail:
        write_block("FAIL", fail_body.rstrip() + "\n" + "\n".join(fresh_fail) + "\n")
        text = read_text(REPORT)
    bs_body = extract_block(text, "BS02")
    fresh_bs = [line for line in bs_lines if line not in bs_body]
    if fresh_bs:
        write_block("BS02", bs_body.rstrip() + "\n" + "\n".join(fresh_bs) + "\n")
    print(f"RMS·bs_02 행 추가 {run} {step}", flush=True)


def detail_drive(run, step):
    text = read_text(REPORT)
    if table_has(extract_block(text, "RMS"), run, step):
        print(f"RMS 이미 있음 {run} {step}. 추가 주행 생략", flush=True)
        return None
    path = ckpt(run, step)
    print(f"추가 주행 {run} {step}", flush=True)
    rows = drive_model(path)
    out_dir = os.path.join(BASE, "results", "klppo_extra")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"resume_{run}_{step}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    append_metrics(run, step, rows)
    return rows


def finish_notes():
    body = extract_block(read_text(REPORT), "EVAL")
    old = (
        "남은 순서는 ctrl_s4_s0 30만, 25만, klppo_s4_s1 30만, 25만, "
        "ctrl_s4_s0 20만, klppo_s4_s1 20만, 그다음 klppo_s4_s1 10만·15만, "
        "ctrl_s4_s0 5만·10만·15만이다."
    )
    if old in body:
        body = body.replace(old, DONE_NOTE, 1)
        write_block("EVAL", body)
    write_block("VERDICT", DONE_NOTE)
    write_block("SUMMARY", summary_line())


def summary_line():
    body = extract_block(read_text(REPORT), "EVAL")
    bc = 1.0
    violations = []
    for match in re.finditer(
            r"^\| (\S+) \| (\d+) \| ([0-9.]+)% \|", body, re.M):
        run, step, official = match.group(1), int(match.group(2)), float(match.group(3)) / 100.0
        if run == "s4_student":
            bc = official
            continue
        if bc - official >= 0.10 - 1e-9:
            violations.append((run, step, official, bc))

    def side(names):
        hit = [item for item in violations if item[0] in names]
        if not hit:
            return "무너지지 않음"
        bits = [f"{run} {step}스텝 공식 {rate:.1%} (BC {base:.1%})"
                for run, step, rate, base in hit]
        return "무너짐 — " + ", ".join(bits)

    return f"KL-PPO: {side(KL_RUNS)}. 대조군: {side(('ctrl_s4_s0',))}."


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pids", default="7652,25164,17936")
    parser.add_argument("--watch-pids", default="4100")
    args = parser.parse_args()
    train_pids = [int(part) for part in args.train_pids.split(",") if part.strip()]
    watch_pids = [int(part) for part in args.watch_pids.split(",") if part.strip()]
    print(f"학습 PID {train_pids}, 감시 PID {watch_pids}", flush=True)
    wait_stage56(train_pids, watch_pids)
    os.environ["CGC_STAGE"] = "4"
    os.environ["OBS_SCAN"] = "0"
    for run, step in ORDER:
        official_eval(run, step)
    for run, step in ORDER:
        detail_drive(run, step)
    finish_notes()
    print("4단계 이어서 끝", flush=True)
    # REPORTS는 다른 사본을 건드리지 않는다. 결과는 4단계 결과 문서만.
    del REPORTS, STUDENT, collected_from


if __name__ == "__main__":
    main()
