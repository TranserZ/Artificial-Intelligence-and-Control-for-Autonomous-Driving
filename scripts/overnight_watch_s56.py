"""5·6단계 KL-PPO 세 학습이 끝난 뒤 5만 간격 체크포인트와 출발 학생을 평가한다.

학습 프로세스와 따로 띄운다. 표준출력은 results/logs/overnight_watch_s56.out.
5단계는 OBS_SCAN=0, 6단계는 OBS_SCAN=1. 판은 900100–900129, 워커 7, shield shadow, 결정적.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import re
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = r"C:\Users\USER\Desktop\청계로드\docs\reports\overnight_2026-11-24_s56.md"
PY = "py"
SEEDS = "900100-900129"
WORKERS = 7
STILL_ACTIVE = 259

RUNS = (
    {
        "name": "klppo1_s5_s0",
        "stage": "5",
        "obs_scan": "0",
        "student_name": "s5_student",
        "student": r"C:\Users\USER\Desktop\청계로드-dongyeon-d54a761\models\dongyeon_bc\s5_student.pt",
    },
    {
        "name": "klppo1_s6_s0",
        "stage": "6",
        "obs_scan": "1",
        "student_name": "s6_student_obsfix",
        "student": r"C:\Users\USER\Desktop\청계로드-dongyeon-d54a761\models\dongyeon_bc\s6_student_obsfix.pt",
    },
    {
        "name": "klppo1_s6_s1",
        "stage": "6",
        "obs_scan": "1",
        "student_name": "s6_student_obsfix",
        "student": r"C:\Users\USER\Desktop\청계로드-dongyeon-d54a761\models\dongyeon_bc\s6_student_obsfix.pt",
    },
)


def alive(pid):
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(handle)
    return code.value == STILL_ACTIVE


def wait_all(pids):
    while True:
        left = [pid for pid in pids if alive(pid)]
        print(f"감시 {time.strftime('%H:%M:%S')} 남은 학습 {len(left)}개 {left}", flush=True)
        if not left:
            return
        time.sleep(60)


def read_text(path):
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def last_step(run_dir, log_text):
    csv_path = os.path.join(run_dir, "training_log.csv")
    best = 0
    if os.path.isfile(csv_path):
        with open(csv_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    best = max(best, int(float(row["steps"])))
                except (KeyError, TypeError, ValueError):
                    continue
    for match in re.finditer(r"steps=\s*(\d+)", log_text):
        best = max(best, int(match.group(1)))
    return best


def finished_ok(log_text):
    return "저장 완료:" in log_text


def error_tail(err_text):
    if "Traceback" not in err_text and "Error" not in err_text:
        return ""
    lines = [line for line in err_text.splitlines() if line.strip()]
    return "\n".join(lines[-25:])


def checkpoints(run_dir):
    found = []
    if not os.path.isdir(run_dir):
        return found
    for name in os.listdir(run_dir):
        match = re.fullmatch(r"step(\d+)k\.pt", name)
        if not match:
            continue
        step = int(match.group(1)) * 1000
        if step >= 50_000 and step % 50_000 == 0:
            found.append((step, os.path.join(run_dir, name)))
    return sorted(found)


def anchor_at(run_dir, step):
    path = os.path.join(run_dir, "training_log.csv")
    chosen = None
    if not os.path.isfile(path):
        return float("nan")
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                logged = int(float(row["steps"]))
                value = float(row["anchor_rms"])
            except (KeyError, TypeError, ValueError):
                continue
            if logged <= step:
                chosen = value
    return float("nan") if chosen is None else chosen


def replace_block(text, name, body):
    start = f"<!-- {name}_START -->"
    end = f"<!-- {name}_END -->"
    block = f"{start}\n{body.rstrip()}\n{end}"
    if start in text and end in text:
        pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
        return pattern.sub(lambda _match: block, text, count=1)
    return text.rstrip() + "\n\n" + block + "\n"


def evaluate_one(model, stage, obs_scan, title, json_out):
    env = os.environ.copy()
    env["CGC_STAGE"] = str(stage)
    env["OBS_SCAN"] = str(obs_scan)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        PY, "-3.13", "-u", os.path.join(BASE, "scripts", "drive_parallel.py"),
        "--model", model, "--seeds", SEEDS, "--shield", "shadow",
        "--workers", str(WORKERS), "--title", title, "--json-out", json_out,
    ]
    print("평가 시작", title, flush=True)
    proc = subprocess.run(cmd, cwd=BASE, env=env)
    if proc.returncode != 0 or not os.path.isfile(json_out):
        raise RuntimeError(f"{title} 평가 실패 code={proc.returncode}")
    with open(json_out, encoding="utf-8") as f:
        return json.load(f)


def fail_rows(payload):
    rows = []
    for row in payload.get("rows") or []:
        reason = row.get("end_reason")
        if reason in ("completed", "missed_stop"):
            continue
        pos = row.get("ego_lane_pos")
        place = f"{row.get('ego_edge')}" if pos is None else f"{row.get('ego_edge')} pos {pos}"
        rows.append((row.get("seed"), reason, place, row.get("place")))
    return rows


def pct(stats, key):
    return f"{stats.get(key, float('nan')):.1%}"


def status_block(pids):
    lines = ["| run | 상태 | 마지막 스텝 | 에러 | 로그 | TensorBoard |",
             "|---|---|---:|---|---|---|"]
    extras = []
    for run in RUNS:
        name = run["name"]
        run_dir = os.path.join(BASE, "results", name)
        log_text = read_text(os.path.join(BASE, "results", "logs", f"{name}.out"))
        err = error_tail(read_text(os.path.join(BASE, "results", "logs", f"{name}.err")))
        step = last_step(run_dir, log_text)
        done = "끝남" if finished_ok(log_text) else "끊김"
        lines.append(
            f"| {name} | {done} | {step} | {'없음' if not err else '있음'} | "
            f"`results/logs/{name}.out` | `results/{name}/tb` |")
        if err:
            extras.append(f"### {name} 에러\n\n```\n{err}\n```")
    lines.append("")
    lines.append(f"기다린 PID: {pids}")
    return "\n".join(lines + extras)


def one_line(student_rates, finals):
    bits = []
    for run in RUNS:
        name = run["name"]
        base_name = run["student_name"]
        base = student_rates.get(base_name)
        final = finals.get(name)
        if base is None or final is None:
            bits.append(f"{name}은 비교할 완주율이 없다")
            continue
        step, rate = final
        if rate > base + 1e-9:
            word = "높다"
        elif rate < base - 1e-9:
            word = "낮다"
        else:
            word = "같다"
        bits.append(
            f"{name} {step // 1000}천 공식 {rate:.1%}는 {base_name} {base:.1%}보다 {word}")
    return "출발 학생 대비: " + ". ".join(bits) + "."


def evaluate_all():
    out_dir = os.path.join(BASE, "results", "overnight_eval_s56")
    os.makedirs(out_dir, exist_ok=True)
    table = [
        "| run | 스텝 | 공식 완주 | 완벽 완주 | 충돌 | anchor_rms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    fails = ["| run | 스텝 | 시드 | 사유 | 위치 | 가까운 곳 |",
             "|---|---:|---:|---|---|---|"]
    student_rates = {}
    finals = {}
    seen_students = set()
    for run in RUNS:
        student_name = run["student_name"]
        if student_name not in seen_students:
            seen_students.add(student_name)
            payload = evaluate_one(
                run["student"], run["stage"], run["obs_scan"], student_name,
                os.path.join(out_dir, f"{student_name}.json"))
            stats = payload["stats"]
            student_rates[student_name] = float(stats["official_complete"])
            table.append(
                f"| {student_name} | 0 | {pct(stats, 'official_complete')} | "
                f"{pct(stats, 'perfect_complete')} | {pct(stats, 'collision')} | 0 |")
            for seed, reason, place, near in fail_rows(payload):
                fails.append(
                    f"| {student_name} | 0 | {seed} | {reason} | {place} | {near} |")
        series_final = None
        run_dir = os.path.join(BASE, "results", run["name"])
        for step, path in checkpoints(run_dir):
            payload = evaluate_one(
                path, run["stage"], run["obs_scan"], f"{run['name']} {step}",
                os.path.join(out_dir, f"{run['name']}_step{step // 1000}k.json"))
            stats = payload["stats"]
            rms = anchor_at(run_dir, step)
            rms_txt = "nan" if rms != rms else f"{rms:.4g}"
            table.append(
                f"| {run['name']} | {step} | {pct(stats, 'official_complete')} | "
                f"{pct(stats, 'perfect_complete')} | {pct(stats, 'collision')} | {rms_txt} |")
            series_final = (step, float(stats["official_complete"]))
            for seed, reason, place, near in fail_rows(payload):
                fails.append(
                    f"| {run['name']} | {step} | {seed} | {reason} | {place} | {near} |")
        if series_final:
            finals[run["name"]] = series_final
    body = "\n".join(table) + "\n\n실패 판 (공식 완주가 아닌 판). 위치는 엣지와 차로 위치, 가까운 곳은 정류장 또는 주정차 지점.\n\n"
    if len(fails) == 2:
        body += "실패 판 없음.\n"
    else:
        body += "\n".join(fails) + "\n"
    return body, one_line(student_rates, finals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pids", required=True)
    args = parser.parse_args()
    pids = [int(part) for part in args.pids.split(",") if part.strip()]
    print(f"학습 PID {pids} 가 끝날 때까지 대기", flush=True)
    wait_all(pids)
    print("학습 종료. 평가 시작", flush=True)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    text = read_text(REPORT)
    text = replace_block(text, "STATUS", status_block(pids))
    try:
        eval_body, summary = evaluate_all()
    except Exception as exc:
        eval_body = f"평가가 중간에 실패했다: {exc}"
        summary = f"출발 학생 대비 비교는 평가가 끝나기 전에 실패해서 확정하지 못했다. 원인: {exc}"
        print(eval_body, flush=True)
    text = replace_block(text, "EVAL", eval_body)
    text = replace_block(text, "SUMMARY", summary)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"보고서 갱신: {REPORT}", flush=True)
    print(summary, flush=True)
    run_resume_s4_eval()


def run_resume_s4_eval():
    """5·6단계 체크포인트 평가가 끝난 뒤 4단계 resume_klppo_eval을 이어서 돌린다."""
    log_out = os.path.join(BASE, "results", "logs", "resume_eval_after_s56.out")
    log_err = os.path.join(BASE, "results", "logs", "resume_eval_after_s56.err")
    os.makedirs(os.path.dirname(log_out), exist_ok=True)
    env = os.environ.copy()
    env["CGC_STAGE"] = "4"
    env["OBS_SCAN"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    script = os.path.join(BASE, "scripts", "resume_klppo_eval.py")
    print("5·6단계 평가 끝. 4단계 resume_klppo_eval 시작", flush=True)
    with open(log_out, "a", encoding="utf-8") as out_f, open(log_err, "a", encoding="utf-8") as err_f:
        proc = subprocess.run(
            [PY, "-3.13", "-u", script],
            cwd=BASE, env=env, stdout=out_f, stderr=err_f,
        )
    if proc.returncode != 0:
        print(f"resume_klppo_eval 실패 code={proc.returncode}. 로그: {log_err}", flush=True)
    else:
        print(f"resume_klppo_eval 끝. 로그: {log_out}", flush=True)


if __name__ == "__main__":
    main()
