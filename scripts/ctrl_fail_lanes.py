"""대조군 25만·30만에서 완주가 아닌 판의 차로와 차선변경을 학생과 비교한다.

시드 900100–900129, 결정적, shield shadow. 차로는 보도를 뺀 번호(0이 맨 오른쪽).
차선변경 횟수는 명령이 실제로 적용된 횟수다.
"""
import json
import os
import sys

os.environ["CGC_STAGE"] = "4"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from multiprocessing import get_context

from scripts.drive_parallel import (
    _ego_place, _make_env, end_reason, policy_action, prepare_sumo,
)

STUDENT = r"C:\Users\USER\Desktop\청계로드-bc-d2ecd99\models\bc_ppo\s4_student.pt"
CTRL = {
    300000: os.path.join(BASE, "results", "ctrl_s4_s0", "step300k.pt"),
    250000: os.path.join(BASE, "results", "ctrl_s4_s0", "step250k.pt"),
}
REPORT = r"C:\Users\USER\Desktop\청계로드\docs\reports\2026-11-23_kl_ppo_4단계_결과.md"
SEEDS = list(range(900100, 900130))
WORKERS = 7
REASONS = {
    "collision": "충돌",
    "red_light": "적색",
    "timeout": "시간초과",
    "lane_error": "차로 오류",
    "missed_stop": "정차 놓침",
    "vanished": "사라짐",
    "completed": "완주",
}


def lane_name(lane, edge):
    if lane is None:
        return "기록 없음"
    if edge and str(edge).startswith(":"):
        return f"교차로({lane})"
    side = {0: "오른쪽", 1: "왼쪽"}.get(int(lane))
    if side:
        return f"{side}({int(lane)})"
    return str(int(lane))


def snapshot(env):
    import traci
    if env.ego_id not in traci.vehicle.getIDList():
        return None
    edge = str(traci.vehicle.getRoadID(env.ego_id))
    try:
        lane = int(env._car_lane_index(env.ego_id))
    except Exception:
        lane = None
    try:
        odo = float(traci.vehicle.getDistance(env.ego_id))
    except Exception:
        odo = None
    return edge, lane, odo


def run_seed(sumocfg, seed, model):
    import numpy as np
    import torch
    from algorithms.kl_ppo import load_eval_policy
    from train import HPARAMS

    torch.set_num_threads(1)
    env = _make_env(sumocfg, "shadow")
    policy = load_eval_policy(
        model, env.observation_space.shape[0], env.action_space.shape[0], HPARAMS)
    obs, _ = env.reset(seed=int(seed))
    terminated = truncated = False
    info = {}
    lc = 0
    last = None
    trace = []
    next_stop = ""
    missed_ids = []
    try:
        while not (terminated or truncated):
            edge, stop = _ego_place(env)
            if stop:
                next_stop = stop
            snap = snapshot(env)
            action = policy_action(policy, obs)
            obs, _, terminated, truncated, info = env.step(action)
            if info.get("lane_change") and info.get("lane_change_applied"):
                lc += 1
            lane = info.get("lane_before")
            if snap is not None:
                snap_edge, snap_lane, odo = snap
                use_lane = snap_lane if lane is None else int(lane)
                last = (snap_edge or edge, use_lane, odo, lc)
                if odo is not None:
                    trace.append((float(odo), use_lane, snap_edge or edge, int(lc)))
        missed_ids = [str(st["id"]) for st, status in zip(
            getattr(env, "_stops", []), getattr(env, "_stop_status", []))
            if int(status) < 0]
    finally:
        env.close()
    reason = end_reason(info, terminated, truncated)
    stop_id = missed_ids[-1] if missed_ids else next_stop
    edge, lane, odo, _lc = last if last else ("", None, None, lc)
    return {
        "seed": int(seed),
        "end_reason": reason,
        "nearest_stop_id": stop_id,
        "edge": edge,
        "lane": lane,
        "odo": odo,
        "lane_changes": int(lc),
        "trace": trace,
    }


def drive(model, seeds):
    import shutil
    work, cfg = prepare_sumo()
    try:
        jobs = [(cfg, seed, model) for seed in seeds]
        if len(seeds) == 1:
            rows = [run_seed(*jobs[0])]
        else:
            with get_context("spawn").Pool(min(WORKERS, len(seeds))) as pool:
                rows = pool.starmap(run_seed, jobs)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return sorted(rows, key=lambda row: row["seed"])


def at_distance(trace, target):
    """target 주행거리에 가장 가까운 학생 기록. (차로, 그때까지 차선변경, 거리 차)."""
    if target is None or not trace:
        return None, None, None
    odo, lane, edge, lc = min(trace, key=lambda item: abs(item[0] - target))
    return lane, edge, lc, abs(odo - target)


def summarize(step, fails, students):
    by_seed = {row["seed"]: row for row in students}
    bits = []
    differ_lane = 0
    differ_lc = 0
    student_fail = 0
    lc_ctrl = []
    lc_student = []
    for row in fails:
        student = by_seed.get(row["seed"])
        if student is None:
            continue
        if student["end_reason"] != "completed":
            student_fail += 1
        lane_s, _edge_s, lc_s, gap = at_distance(student["trace"], row["odo"])
        lc_ctrl.append(row["lane_changes"])
        if lc_s is not None:
            lc_student.append(lc_s)
            if lc_s != row["lane_changes"]:
                differ_lc += 1
        if gap is not None and gap <= 30.0 and lane_s is not None and row["lane"] is not None and int(lane_s) != int(row["lane"]):
            differ_lane += 1
    n = len(fails)
    mean_c = sum(lc_ctrl) / len(lc_ctrl) if lc_ctrl else 0.0
    mean_s = sum(lc_student) / len(lc_student) if lc_student else 0.0
    bits.append(
        f"{step // 10000}만 실패 {n}판. 같은 시드에서 학생도 완주가 아닌 판은 {student_fail}개."
    )
    bits.append(
        f"실패한 거리까지 차선변경은 대조군 평균 {mean_c:.1f}회, 학생 평균 {mean_s:.1f}회. "
        f"횟수가 다른 판은 {differ_lc}/{n}."
    )
    bits.append(f"그 지점의 차로가 학생과 다른 판은 {differ_lane}/{n}.")
    return " ".join(bits)


def table(step, fails, students):
    by_seed = {row["seed"]: row for row in students}
    lines = [
        f"### ctrl_s4_s0 {step // 10000}만",
        "",
        "| 시드 | 종료 사유 | 가장 가까운 정류장 | 실패 직전 차로 | 차선변경 횟수 |",
        "|---:|---|---|---|---:|",
    ]
    cmp_lines = [
        "",
        "| 시드 | 대조군 차선변경 | 학생 차선변경 (같은 거리) | 대조군 차로 | 학생 그 지점 차로 | 학생 종료 |",
        "|---:|---:|---:|---|---|---|",
    ]
    for row in fails:
        lines.append(
            f"| {row['seed']} | {REASONS.get(row['end_reason'], row['end_reason'])} | "
            f"{row['nearest_stop_id'] or row['edge'] or ''} | "
            f"{lane_name(row['lane'], row['edge'])} | {row['lane_changes']} |"
        )
        student = by_seed.get(row["seed"])
        if student is None:
            cmp_lines.append(f"| {row['seed']} | {row['lane_changes']} |  | "
                             f"{lane_name(row['lane'], row['edge'])} |  |  |")
            continue
        lane_s, edge_s, lc_s, gap = at_distance(student["trace"], row["odo"])
        student_lane = lane_name(lane_s, edge_s)
        if gap is not None and gap > 30.0:
            student_lane = f"{student_lane} (거리 차 {gap:.0f} m)"
        lc_cell = "" if lc_s is None else str(lc_s)
        cmp_lines.append(
            f"| {row['seed']} | {row['lane_changes']} | {lc_cell} | "
            f"{lane_name(row['lane'], row['edge'])} | {student_lane} | "
            f"{REASONS.get(student['end_reason'], student['end_reason'])} |"
        )
    return "\n".join(lines + cmp_lines)


def write_report(body):
    with open(REPORT, encoding="utf-8") as f:
        text = f.read()
    start, end = "<!-- CTRL_LANE_START -->", "<!-- CTRL_LANE_END -->"
    block = f"{start}\n{body.rstrip()}\n{end}"
    if start in text and end in text:
        import re
        text = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S).sub(
            lambda _match: block, text, count=1)
    else:
        heading = "## 대조군 실패 판의 차로\n\n" + (
            "완주(`completed`)가 아닌 판이다. 차로는 보도를 뺀 번호이고 0이 맨 오른쪽이다. "
            "차선변경 횟수는 그 명령이 SUMO에 적용된 횟수다. "
            "학생 비교는 같은 시드를 결정적으로 다시 달려, 대조군이 멈춘 주행거리와 가장 가까운 지점이다.\n\n"
            + block + "\n"
        )
        needle = "## 한 줄"
        if needle in text:
            text = text.replace(needle, heading + "\n" + needle, 1)
        else:
            text = text.rstrip() + "\n\n" + heading
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text)


def slim(row):
    return {key: value for key, value in row.items() if key != "trace"}


def main():
    driven = {}
    for step, path in CTRL.items():
        print(f"주행 ctrl {step}", flush=True)
        driven[step] = drive(path, SEEDS)
        fails = [row for row in driven[step] if row["end_reason"] != "completed"]
        print(f"ctrl {step} 실패 {len(fails)}", flush=True)
    seeds = sorted({row["seed"] for rows in driven.values()
                    for row in rows if row["end_reason"] != "completed"})
    print(f"학생 주행 시드 {len(seeds)}개", flush=True)
    students = drive(STUDENT, seeds) if seeds else []
    parts = []
    for step in (300000, 250000):
        fails = [row for row in driven[step] if row["end_reason"] != "completed"]
        parts.append(summarize(step, fails, students))
        parts.append("")
        parts.append(table(step, fails, students))
        parts.append("")
    body = "\n".join(parts).rstrip()
    write_report(body)
    out = os.path.join(BASE, "results", "overnight_eval", "ctrl_fail_lanes.json")
    payload = {
        "ctrl": {str(step): [slim(row) for row in rows] for step, rows in driven.items()},
        "student": [slim(row) for row in students],
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(body, flush=True)
    print(f"기록 {REPORT}", flush=True)


if __name__ == "__main__":
    main()
