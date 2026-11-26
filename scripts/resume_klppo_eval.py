"""끝난 체크포인트는 그대로 두고, 남은 평가만 지정한 순서로 돌린다.

워커는 기존과 같이 7개다. torch.set_num_threads(1)은 run_seed 안에 이미 있다.
앞쪽 4개가 끝나면 결정적 가속 RMS와 bs_02 추가 주행을 하고, 그다음 나머지를 이어서 평가한다.
"""
import os
import sys
import time

os.environ["CGC_STAGE"] = "4"

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from scripts.klppo_result_extra import STUDENT, drive_model, publish
from scripts.overnight_watch import anchor_at, anchor_series, plot, replace_block

REPORTS = (
    r"C:\Users\USER\Desktop\청계로드\docs\reports\2026-11-23_kl_ppo_4단계_결과.md",
    r"C:\Users\USER\Desktop\청계로드\docs\reports\overnight_2026-11-23.md",
)
SEEDS = list(range(900100, 900130))
WORKERS = 7
BC = 1.0

# 이미 보고서에 있는 체크포인트. 다시 돌리지 않는다.
FINISHED = (
    ("s4_student", 0, 30 / 30, 30 / 30, 0 / 30),
    ("klppo_s4_s0", 50000, 30 / 30, 28 / 30, 0 / 30),
    ("klppo_s4_s0", 100000, 30 / 30, 30 / 30, 0 / 30),
    ("klppo_s4_s0", 150000, 28 / 30, 28 / 30, 1 / 30),
    ("klppo_s4_s0", 200000, 29 / 30, 29 / 30, 1 / 30),
    ("klppo_s4_s0", 250000, 28 / 30, 28 / 30, 2 / 30),
    ("klppo_s4_s0", 300000, 29 / 30, 25 / 30, 0 / 30),
    ("klppo_s4_s1", 50000, 28 / 30, 28 / 30, 2 / 30),
    ("ctrl_s4_s0", 300000, 24 / 30, 23 / 30, 6 / 30),
    ("ctrl_s4_s0", 250000, 24 / 30, 23 / 30, 6 / 30),
    ("klppo_s4_s1", 300000, 29 / 30, 29 / 30, 0 / 30),
    ("klppo_s4_s1", 250000, 26 / 30, 24 / 30, 4 / 30),
)

# 5·6단계 평가 뒤 이어서: 앞 4개 후 결정적 가속 RMS·bs_02, 그다음 나머지 체크포인트.
PRIORITY = (
    ("klppo_s4_s1", 300000),
    ("klppo_s4_s1", 250000),
    ("ctrl_s4_s0", 200000),
    ("klppo_s4_s1", 200000),
)
TAIL = (
    ("klppo_s4_s1", 100000),
    ("klppo_s4_s1", 150000),
    ("ctrl_s4_s0", 50000),
    ("ctrl_s4_s0", 100000),
    ("ctrl_s4_s0", 150000),
)


def read_text(path):
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def ckpt(run, step):
    return os.path.join(BASE, "results", run, f"step{step // 1000}k.pt")


def rms_of(run, step):
    if run == "s4_student":
        return 0.0
    return anchor_at(anchor_series(os.path.join(BASE, "results", run)), step)


def drop_cell(official):
    if official is None:
        return ""
    drop = BC - official
    return "위반" if drop >= 0.10 - 1e-9 else "아니오"


def row_line(run, step, official, perfect, collision, anchor):
    if official is None:
        return f"| {run} | {step} | 실패 | 실패 | 실패 |  |  |"
    anchor_cell = "0" if run == "s4_student" else f"{anchor:.4g}"
    flag = "기준" if run == "s4_student" else drop_cell(official)
    return (
        f"| {run} | {step} | {official:.1%} | {perfect:.1%} | "
        f"{collision:.1%} | {anchor_cell} | {flag} |"
    )


def collapsed_line(official):
    """BC 30/30의 90% 이하면 무너진 것이다."""
    limit = 0.9 * BC
    if official <= limit + 1e-12:
        return (
            f"대조군 300k는 무너졌다. 공식 완주 {official:.1%}는 "
            f"BC 30/30의 90%({limit:.0%}) 이하다."
        )
    return (
        f"대조군 300k는 무너지지 않았다. 공식 완주 {official:.1%}는 "
        f"BC 30/30의 90%({limit:.0%})보다 높다."
    )


def eval_body(rows, figure_line, collapse):
    lines = [
        "이미 끝난 행은 다시 돌리지 않는다.",
        "5·6단계 체크포인트 평가가 끝난 뒤 이어서: "
        "klppo_s4_s1 30만 → 25만 → ctrl_s4_s0 20만 → klppo_s4_s1 20만 → "
        "결정적 가속 RMS·bs_02 사례 분석, 그다음 klppo_s4_s1 10만·15만, "
        "ctrl_s4_s0 5만·10만·15만.",
        "하나 끝날 때마다 이 표에 행을 더한다.",
        "",
    ]
    if collapse:
        lines.append(collapse)
        lines.append("")
    lines.extend([
        "| run | 스텝 | 공식 완주 | 완벽 완주 | 충돌 | anchor_rms | BC보다 10%p 이상 하락 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    lines.extend(rows)
    lines.append("")
    lines.append("중단 규칙: 체크포인트 공식 완주가 s4_student보다 10%p 이상 낮으면 위반.")
    if figure_line:
        lines.append("")
        lines.append(figure_line)
    return "\n".join(lines)


def write_reports(rows, collapse, collected):
    figure = os.path.join(BASE, "results", "overnight_eval", "curves.png")
    os.makedirs(os.path.dirname(figure), exist_ok=True)
    figure_line = ""
    try:
        plot(collected, figure)
        figure_line = f"그림: `{figure}`"
    except Exception as exc:
        figure_line = f"그림을 만들지 못했다: {exc}"
    body = eval_body(rows, figure_line, collapse)
    for path in REPORTS:
        text = replace_block(read_text(path), "EVAL", body)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    with open(os.path.join(BASE, "results", "overnight_eval", "table.md"), "w", encoding="utf-8") as f:
        f.write(body + "\n")


def collected_from(records):
    by_run = {}
    for run, step, official, perfect, collision, anchor in records:
        if official is None:
            continue
        by_run.setdefault(run, []).append((step, {
            "official": official,
            "perfect": perfect,
            "collision": collision,
            "anchor_rms": anchor,
        }))
    return [{"run": run, "points": points} for run, points in by_run.items()]


def record_of(run, step, official, perfect, collision):
    anchor = rms_of(run, step) if official is not None else float("nan")
    return (run, step, official, perfect, collision, anchor)


def finished_pairs():
    return {(run, step) for run, step, *_ in FINISHED}


def collapse_from_records(records):
    for run, step, official, *_ in records:
        if run == "ctrl_s4_s0" and step == 300000 and official is not None:
            return collapsed_line(official)
    return ""


def run_extra(driven, targets, partial):
    for key, path in targets:
        if key in driven:
            continue
        label = key if isinstance(key, str) else f"{key[0]} {key[1]}"
        print(f"추가 주행 {label}", flush=True)
        try:
            driven[key] = drive_model(path)
        except Exception as exc:
            print(f"추가 주행 실패 {label}: {exc}", flush=True)
            continue
        if "s4_student" not in driven:
            continue
        try:
            publish(driven, partial=partial)
        except Exception as exc:
            print(f"추가 분석 기록 실패 {label}: {exc}", flush=True)


def evaluate_item(evaluate, run, step, records, rows, done):
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
    elapsed = time.perf_counter() - started
    print(f"소요 {run} {step} {elapsed:.0f}초", flush=True)
    rec = record_of(run, step, official, perfect, collision)
    records.append(rec)
    rows.append(row_line(*rec))
    if official is not None:
        done.append((run, step))
    return rec


def main():
    from scripts.drive_parallel import evaluate

    skip = finished_pairs()
    records = [record_of(*item) for item in FINISHED]
    rows = [row_line(*rec) for rec in records]
    done = [(run, step) for run, step, *_rest in FINISHED if run != "s4_student"]
    collapse = collapse_from_records(records)
    write_reports(rows, collapse, collected_from(records))
    pending = sum(1 for run, step in PRIORITY + TAIL if (run, step) not in skip)
    print(f"끝난 결과 {len(FINISHED)}행. 이어서 할 평가 {pending}개", flush=True)

    driven = {}
    for index, (run, step) in enumerate(PRIORITY):
        if (run, step) in skip:
            print(f"건너뜀 {run} {step} (이미 보고서에 있음)", flush=True)
        else:
            evaluate_item(evaluate, run, step, records, rows, done)
            write_reports(rows, collapse, collected_from(records))
            print(f"보고서 행 추가 {run} {step}", flush=True)
        if index == len(PRIORITY) - 1:
            print("우선 4개 처리 끝. 결정적 가속 RMS·bs_02 추가 분석", flush=True)
            targets = [("s4_student", STUDENT)]
            targets.extend((pair, ckpt(*pair)) for pair in done)
            run_extra(driven, targets, partial=True)

    for run, step in TAIL:
        if (run, step) in skip:
            print(f"건너뜀 {run} {step} (이미 보고서에 있음)", flush=True)
            continue
        evaluate_item(evaluate, run, step, records, rows, done)
        write_reports(rows, collapse, collected_from(records))
        print(f"보고서 행 추가 {run} {step}", flush=True)

    print("남은 체크포인트 끝. 추가 분석 마무리", flush=True)
    targets = [("s4_student", STUDENT)]
    targets.extend((pair, ckpt(*pair)) for pair in done)
    run_extra(driven, targets, partial=False)
    print("4단계 이어하기 끝", flush=True)


if __name__ == "__main__":
    main()
