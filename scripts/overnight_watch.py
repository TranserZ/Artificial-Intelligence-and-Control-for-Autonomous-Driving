"""세 본 학습이 끝난 뒤 5만 간격 체크포인트와 s4_student를 평가한다.

학습 프로세스와 따로 띄운다. 표준출력은 results/logs/overnight_watch.out.
최종 80판 비교는 하지 않는다.
"""
import argparse
import csv
import ctypes
import os
import re
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

RUNS = ("klppo_s4_s0", "klppo_s4_s1", "ctrl_s4_s0")
KL_RUNS = ("klppo_s4_s0", "klppo_s4_s1")
STUDENT = r"C:\Users\USER\Desktop\청계로드-bc-d2ecd99\models\bc_ppo\s4_student.pt"
REPORT = r"C:\Users\USER\Desktop\청계로드\docs\reports\overnight_2026-11-23.md"
SEEDS = list(range(900100, 900130))
WORKERS = 7
STILL_ACTIVE = 259


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
                best = max(best, int(float(row["steps"])))
    for match in re.finditer(r"steps=\s*(\d+)", log_text):
        best = max(best, int(match.group(1)))
    for match in re.finditer(r"\[step (\d+) 체크포인트", log_text):
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


def anchor_series(run_dir):
    path = os.path.join(run_dir, "training_log.csv")
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append((int(float(row["steps"])), float(row["anchor_rms"])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def anchor_at(series, step):
    chosen = None
    for logged, value in series:
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


def plot(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for name in ("Malgun Gothic", "맑은 고딕"):
        if any(name == font.name for font in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

    panels = (
        ("official", "공식 완주"),
        ("perfect", "완벽 완주"),
        ("collision", "충돌"),
        ("anchor_rms", "anchor_rms"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (key, title) in zip(axes.ravel(), panels):
        for row in rows:
            if row["run"] == "s4_student":
                continue
            xs = [item[0] / 1000 for item in row["points"]]
            ys = [item[1][key] for item in row["points"]]
            ax.plot(xs, ys, marker="o", label=row["run"])
        student = next((row for row in rows if row["run"] == "s4_student"), None)
        if student and key != "anchor_rms":
            ax.axhline(student["points"][0][1][key], color="black", linestyle="--",
                       label="s4_student")
        ax.set_title(title)
        ax.set_xlabel("스텝 (천)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def evaluate_all():
    from scripts.drive_parallel import evaluate

    out_dir = os.path.join(BASE, "results", "overnight_eval")
    os.makedirs(out_dir, exist_ok=True)
    collected = []
    print("s4_student 평가 시작", flush=True)
    _, student_stats = evaluate(
        STUDENT, SEEDS, shield="shadow", workers=WORKERS,
        deterministic=True, title="s4_student")
    collected.append({
        "run": "s4_student",
        "points": [(0, {
            "official": student_stats["official_complete"],
            "perfect": student_stats["perfect_complete"],
            "collision": student_stats["collision"],
            "anchor_rms": 0.0,
        })],
    })
    bc = student_stats["official_complete"]
    table_lines = [
        "| run | 스텝 | 공식 완주 | 완벽 완주 | 충돌 | anchor_rms | BC보다 10%p 이상 하락 |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| s4_student | 0 | {bc:.1%} | {student_stats['perfect_complete']:.1%} | "
        f"{student_stats['collision']:.1%} | 0 | 기준 |",
    ]
    violations = []
    for run in RUNS:
        run_dir = os.path.join(BASE, "results", run)
        series = anchor_series(run_dir)
        points = []
        for step, path in checkpoints(run_dir):
            print(f"평가 {run} {step}", flush=True)
            try:
                _, stats = evaluate(
                    path, SEEDS, shield="shadow", workers=WORKERS,
                    deterministic=True, title=f"{run} step{step // 1000}k")
            except Exception as exc:
                print(f"평가 실패 {run} {step}: {exc}", flush=True)
                table_lines.append(
                    f"| {run} | {step} | 실패 | 실패 | 실패 |  | {exc} |")
                continue
            rms = anchor_at(series, step)
            drop = bc - stats["official_complete"]
            violated = drop >= 0.10 - 1e-9
            if violated:
                violations.append((run, step, stats["official_complete"], bc))
            points.append((step, {
                "official": stats["official_complete"],
                "perfect": stats["perfect_complete"],
                "collision": stats["collision"],
                "anchor_rms": rms,
            }))
            table_lines.append(
                f"| {run} | {step} | {stats['official_complete']:.1%} | "
                f"{stats['perfect_complete']:.1%} | {stats['collision']:.1%} | "
                f"{rms:.4g} | {'위반' if violated else '아니오'} |")
        collected.append({"run": run, "points": points})
    figure = os.path.join(out_dir, "curves.png")
    try:
        plot(collected, figure)
        figure_line = f"그림: `{figure}`"
    except Exception as exc:
        figure = ""
        figure_line = f"그림을 만들지 못했다: {exc}"
    with open(os.path.join(out_dir, "table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines) + "\n")
    return "\n".join(table_lines), figure_line, violations, bc


def status_block():
    lines = ["| run | 끝났는지 | 마지막 스텝 | 에러 | 로그 | TensorBoard |",
             "|---|---|---:|---|---|---|"]
    summaries = []
    for run in RUNS:
        run_dir = os.path.join(BASE, "results", run)
        log_text = read_text(os.path.join(BASE, "results", "logs", f"{run}.out"))
        err = error_tail(read_text(os.path.join(BASE, "results", "logs", f"{run}.err")))
        step = last_step(run_dir, log_text)
        done = "끝남" if finished_ok(log_text) else "끊김"
        err_cell = "없음" if not err else "있음 (아래)"
        lines.append(
            f"| {run} | {done} | {step} | {err_cell} | "
            f"`results/logs/{run}.out` | `results/{run}/tb` |")
        if err:
            summaries.append(f"### {run} 에러\n\n```\n{err}\n```")
    return "\n".join(lines + [""] + summaries)


def one_line(violations):
    def side(names):
        hit = [item for item in violations if item[0] in names]
        if not hit:
            return "무너지지 않음"
        bits = [f"{run} {step}스텝 공식 {rate:.1%} (BC {bc:.1%})"
                for run, step, rate, bc in hit]
        return "무너짐 — " + ", ".join(bits)

    return f"KL-PPO: {side(KL_RUNS)}. 대조군: {side(('ctrl_s4_s0',))}."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pids", required=True, help="쉼표로 구분한 학습 PID")
    args = parser.parse_args()
    pids = [int(part) for part in args.pids.split(",") if part.strip()]
    print(f"학습 PID {pids} 가 끝날 때까지 대기", flush=True)
    wait_all(pids)
    print("학습 종료. 평가 시작", flush=True)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    text = read_text(REPORT)
    text = replace_block(text, "STATUS", status_block())
    try:
        table, figure_line, violations, _bc = evaluate_all()
        eval_body = table + "\n\n" + figure_line + "\n\n" + (
            "중단 규칙: 체크포인트 공식 완주가 s4_student보다 10%p 이상 낮으면 위반."
        )
        summary = one_line(violations)
    except Exception as exc:
        eval_body = f"평가가 중간에 실패했다: {exc}"
        summary = f"KL-PPO와 대조군의 무너짐 여부는 평가가 끝나기 전에 실패해서 확정하지 못했다. 원인: {exc}"
        print(eval_body, flush=True)
    text = replace_block(text, "EVAL", eval_body)
    text = replace_block(text, "SUMMARY", summary)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"보고서 갱신: {REPORT}", flush=True)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
