"""현재 30판 평가가 끝난 뒤, 결과 보고서에 추가 분석을 붙인다.

1. TensorBoard에서 2만 5천 스텝마다 anchor_rms, beta
2. 체크포인트별 실패 판의 시드·사유·정류장
3. 평가 주행 관측에서 s4_student와의 가속 평균 RMS
4. 900107·900109의 bs_02 접근 비교
5. 결론 한 줄과 다음 실험 하나

지금 돌아가는 overnight_watch와 동시에 주행하지 않는다.
"""
import ctypes
import json
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
OLD_REPORT = r"c:\Users\USER\Desktop\청계로드\docs\reports\overnight_2026-11-23.md"
REPORT = r"c:\Users\USER\Desktop\청계로드\docs\reports\2026-11-23_kl_ppo_4단계_결과.md"
SEEDS = list(range(900100, 900130))
FOCUS = (900107, 900109)
WORKERS = 7
TARGETS = list(range(25_000, 300_001, 25_000))
STILL_ACTIVE = 259
APPROACH_M = 150.0

NEXT_STEP = {
    "a": "다음 실험: 같은 설정으로 6단계 KL-PPO 시드 0을 30만 스텝 돌려, 기준 묶음이 전체 지도에서도 완주를 지키는지 본다.",
    "b": "다음 실험: 미뤄 둔 900130–900189 60판과 검증 900000–900019 20판으로, 둘의 공식 완주가 실제로 다른지 가른다.",
    "c": "다음 실험: beta 하한을 0.01 아래로 내리거나 목표 RMS를 0.1보다 키워 가속이 BC에서 떨어지게 한 뒤, 같은 30판으로 공식 완주와 평가 RMS를 다시 본다.",
    "kl_down": "다음 실험: 공식 완주가 처음 10%p 넘게 떨어진 체크포인트에서 멈추고, 그 시점의 beta·평가 RMS·실패 시드를 보고 묶음을 더 조인다.",
}


def alive(pid):
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(handle)
    return code.value == STILL_ACTIVE


def read_text(path):
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def replace_block(text, name, body):
    start, end = f"<!-- {name}_START -->", f"<!-- {name}_END -->"
    block = f"{start}\n{body.rstrip()}\n{end}"
    if start in text and end in text:
        pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
        return pattern.sub(lambda _match: block, text, count=1)
    return text.rstrip() + "\n\n" + block + "\n"


def extract_block(text, name):
    match = re.search(
        rf"<!-- {name}_START -->\n(.*?)<!-- {name}_END -->", text, re.S)
    return match.group(1).strip() if match else ""


def write_report(name, body):
    text = replace_block(read_text(REPORT), name, body)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text)


def scalars(tb_dir, tag):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return []
    return [(int(event.step), float(event.value)) for event in acc.Scalars(tag)]


def first_at_or_after(points, target):
    for step, value in points:
        if step >= target:
            return step, value
    return None, None


def tb_section():
    loaded = {}
    for run in RUNS:
        tb_dir = os.path.join(BASE, "results", run, "tb")
        loaded[run] = {
            "anchor_rms": scalars(tb_dir, "train/anchor_rms"),
            "beta": scalars(tb_dir, "train/beta"),
        }
    header = ("| 목표 스텝 | klppo_s4_s0 rms | beta | klppo_s4_s1 rms | beta | "
              "ctrl_s4_s0 rms | beta |")
    lines = [header, "|---:|---:|---:|---:|---:|---:|---:|"]
    for target in TARGETS:
        cells = [str(target)]
        for run in RUNS:
            rms_step, rms = first_at_or_after(loaded[run]["anchor_rms"], target)
            beta_step, beta = first_at_or_after(loaded[run]["beta"], target)
            if rms is None:
                cells.extend(["없음", "없음"])
            else:
                cells.append(f"{rms:.4g} @{rms_step}")
                cells.append("없음" if beta is None else f"{beta:.4g} @{beta_step}")
        lines.append("| " + " | ".join(cells) + " |")
    notes = []
    for run in KL_RUNS:
        by_step = {step: value for step, value in loaded[run]["anchor_rms"]}
        floored = [(step, by_step[step]) for step, beta in loaded[run]["beta"]
                   if abs(beta - 0.01) < 1e-6 and step in by_step]
        if not floored:
            notes.append(f"{run}: beta가 0.01에 붙은 기록이 없다.")
            continue
        values = [value for _, value in floored]
        notes.append(
            f"{run}: beta가 0.01인 기록 {len(floored)}개 "
            f"({floored[0][0]}~{floored[-1][0]}스텝). "
            f"그때 anchor_rms는 최소 {min(values):.4g}, "
            f"최대 {max(values):.4g}.")
    return "\n".join(lines) + "\n\n" + "\n".join(notes)


def checkpoints(run):
    run_dir = os.path.join(BASE, "results", run)
    found = []
    for name in os.listdir(run_dir):
        match = re.fullmatch(r"step(\d+)k\.pt", name)
        if not match:
            continue
        step = int(match.group(1)) * 1000
        if step >= 50_000 and step % 50_000 == 0:
            found.append((step, os.path.join(run_dir, name)))
    return sorted(found)


def side_clearance(ego_id):
    """옆 차로에서 길이 방향으로 겹친 차와의 가장 좁은 가로 여유(m). 없으면 None."""
    import traci
    if ego_id not in traci.vehicle.getIDList():
        return None
    edge = traci.vehicle.getRoadID(ego_id)
    if not edge or edge.startswith(":"):
        return None
    ego_lane = traci.vehicle.getLaneIndex(ego_id)
    ego_pos = traci.vehicle.getLanePosition(ego_id)
    ego_len = traci.vehicle.getLength(ego_id)
    ego_w = traci.vehicle.getWidth(ego_id)
    try:
        ego_lat = traci.vehicle.getLateralLanePosition(ego_id)
    except traci.TraCIException:
        ego_lat = 0.0
    best = None
    try:
        n_lanes = traci.edge.getLaneNumber(edge)
    except traci.TraCIException:
        return None
    for lane_index in range(n_lanes):
        if lane_index == ego_lane:
            continue
        lane_id = f"{edge}_{lane_index}"
        sign = 1.0 if lane_index > ego_lane else -1.0
        lo, hi = sorted((ego_lane, lane_index))
        try:
            center = 0.0
            for index in range(lo, hi):
                center += 0.5 * (traci.lane.getWidth(f"{edge}_{index}")
                                 + traci.lane.getWidth(f"{edge}_{index + 1}"))
        except traci.TraCIException:
            continue
        for vid in traci.lane.getLastStepVehicleIDs(lane_id):
            try:
                other_pos = traci.vehicle.getLanePosition(vid)
                other_len = traci.vehicle.getLength(vid)
                if abs(other_pos - ego_pos) > 0.5 * (ego_len + other_len):
                    continue
                other_w = traci.vehicle.getWidth(vid)
                other_lat = traci.vehicle.getLateralLanePosition(vid)
            except traci.TraCIException:
                continue
            gap = center - 0.5 * (ego_w + other_w) - sign * ego_lat + sign * other_lat
            if best is None or gap < best:
                best = gap
    return best


def run_detail(sumocfg, seed, model, student_path, shield):
    import numpy as np
    import torch
    from algorithms.kl_ppo import load_eval_policy
    from scripts.drive_parallel import _make_env, end_reason
    from train import HPARAMS

    torch.set_num_threads(1)
    env = _make_env(sumocfg, shield)
    policy = load_eval_policy(
        model, env.observation_space.shape[0], env.action_space.shape[0], HPARAMS)
    student = load_eval_policy(
        student_path, env.observation_space.shape[0], env.action_space.shape[0], HPARAMS)
    obs, _ = env.reset(seed=int(seed))
    terminated = truncated = False
    info = {}
    sq_sum = 0.0
    n_obs = 0
    last_edge = ""
    next_stop = ""
    missed_ids = []
    approach = {"seen": False, "min_gap": None, "min_speed": None, "decel_dist": None}
    try:
        import traci
        while not (terminated or truncated):
            edge, stop = "", ""
            try:
                if env.ego_id in traci.vehicle.getIDList():
                    edge = str(traci.vehicle.getRoadID(env.ego_id))
                current = env._current_stop()
                if current is not None:
                    stop = str(current["id"])
            except Exception:
                current = None
            if edge:
                last_edge = edge
            if stop:
                next_stop = stop
            action = np.asarray(policy.predict(obs, deterministic=True), dtype=np.float32)
            ref = np.asarray(student.predict(obs, deterministic=True), dtype=np.float32)
            diff = float(action[0] - ref[0])
            sq_sum += diff * diff
            n_obs += 1
            if int(seed) in FOCUS and stop == "bs_02":
                dist = None
                try:
                    nxt = env._next_stop()
                    if nxt is not None:
                        dist = float(nxt[0])
                except Exception:
                    dist = None
                if dist is not None and 0.0 <= dist <= APPROACH_M:
                    approach["seen"] = True
                    try:
                        speed = float(traci.vehicle.getSpeed(env.ego_id))
                    except Exception:
                        speed = None
                    if speed is not None and (approach["min_speed"] is None
                                               or speed < approach["min_speed"]):
                        approach["min_speed"] = speed
                    gap = side_clearance(env.ego_id)
                    if gap is not None and (approach["min_gap"] is None or gap < approach["min_gap"]):
                        approach["min_gap"] = gap
                    if float(action[0]) < 0.0 and approach["decel_dist"] is None:
                        approach["decel_dist"] = dist
            obs, _, terminated, truncated, info = env.step(action)
        missed_ids = [str(st["id"]) for st, status in zip(
            getattr(env, "_stops", []), getattr(env, "_stop_status", []))
            if int(status) < 0]
    finally:
        env.close()
    reason = end_reason(info, terminated, truncated)
    stop_id = missed_ids[-1] if missed_ids else next_stop
    return {
        "seed": int(seed),
        "end_reason": reason,
        "nearest_stop_id": stop_id,
        "ego_edge": last_edge,
        "sq_sum": sq_sum,
        "n_obs": n_obs,
        "approach": approach,
    }


def drive_model(model):
    import shutil
    from multiprocessing import get_context
    from scripts.drive_parallel import prepare_sumo

    work, cfg = prepare_sumo()
    try:
        jobs = [(cfg, seed, model, STUDENT, "shadow") for seed in SEEDS]
        with get_context("spawn").Pool(WORKERS) as pool:
            rows = pool.starmap(run_detail, jobs)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return rows


def fmt_gap(value):
    if value is None:
        return "옆 차 없음"
    return f"{value:.2f} m"


def fmt_speed(value):
    if value is None:
        return "기록 없음"
    return f"{value:.2f} m/s ({value * 3.6:.1f} km/h)"


def fmt_dist(value, seen):
    if not seen:
        return "bs_02 접근 없음"
    if value is None:
        return "감속 없음"
    return f"{value:.1f} m"


def fail_and_metrics(driven):
    student_rows = driven["s4_student"]
    student_bad = {row["seed"]: row["end_reason"] for row in student_rows
                   if row["end_reason"] != "completed"}
    fail_lines = [
        "| run | 스텝 | 시드 | 종료 사유 | 가장 가까운 정류장 | 구분 |",
        "|---|---:|---:|---|---|---|",
    ]
    any_fail = False
    rms_lines = [
        "| run | 스텝 | 평가 가속 RMS | 관측 수 | 0.02 미만 |",
        "|---|---:|---:|---:|---|",
    ]
    bs_lines = [
        "| run | 스텝 | 시드 | 최소 측면 간격 | 최소 속도 | 감속 시작 (남은 거리) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rms_by_model = {}
    official = {}
    for label, rows in driven.items():
        run, step = label if isinstance(label, tuple) else ("s4_student", 0)
        if label == "s4_student":
            run, step = "s4_student", 0
        sq = sum(row["sq_sum"] for row in rows)
        n_obs = sum(row["n_obs"] for row in rows)
        rms = (sq / n_obs) ** 0.5 if n_obs else float("nan")
        rms_by_model[(run, step)] = rms
        official[(run, step)] = sum(
            row["end_reason"] in ("completed", "missed_stop") for row in rows) / max(len(rows), 1)
        close = "예" if rms < 0.02 else "아니오"
        rms_lines.append(f"| {run} | {step} | {rms:.4g} | {n_obs} | {close} |")
        for row in rows:
            if row["end_reason"] == "completed":
                continue
            any_fail = True
            if run == "s4_student":
                kind = "학생 기준"
            elif row["seed"] in student_bad:
                kind = f"학생도 실패 ({student_bad[row['seed']]})"
            else:
                kind = "새 실패 (학생은 완주)"
            if row["seed"] in FOCUS:
                kind += ", bs_02 시드"
            fail_lines.append(
                f"| {run} | {step} | {row['seed']} | {row['end_reason']} | "
                f"{row['nearest_stop_id'] or row['ego_edge']} | {kind} |")
        if run in KL_RUNS or run == "s4_student":
            for row in rows:
                if row["seed"] not in FOCUS:
                    continue
                approach = row["approach"]
                bs_lines.append(
                    f"| {run} | {step} | {row['seed']} | {fmt_gap(approach['min_gap'])} | "
                    f"{fmt_speed(approach['min_speed'])} | "
                    f"{fmt_dist(approach['decel_dist'], approach['seen'])} |")
    if not any_fail:
        fail_lines.append("| 전체 |  |  |  |  | 완주가 아닌 판 없음 |")
    return ("\n".join(fail_lines), "\n".join(rms_lines), "\n".join(bs_lines),
            rms_by_model, official)


def verdict(rms_by_model, official):
    student = official[("s4_student", 0)]
    def held(run):
        rates = [rate for (name, _), rate in official.items() if name == run]
        return all(student - rate < 0.10 - 1e-9 for rate in rates)

    kl_held = all(held(run) for run in KL_RUNS)
    ctrl_held = held("ctrl_s4_s0")
    kl_rms = [value for (run, _), value in rms_by_model.items() if run in KL_RUNS]
    same = bool(kl_rms) and max(kl_rms) < 0.02
    if same:
        code = "c"
        line = (f"(c) KL-PPO가 BC와 사실상 같음. 평가 가속 RMS 최대 {max(kl_rms):.4g} "
                f"(기준 0.02 미만).")
    elif kl_held and not ctrl_held:
        code = "a"
        line = "(a) KL-PPO 유지 + 대조군 무너짐. KL-PPO 체크포인트는 BC보다 공식 완주 10%p 이상 낮지 않고, 대조군은 그렇다."
    elif kl_held and ctrl_held:
        code = "b"
        line = "(b) 둘 다 유지. KL-PPO와 대조군 모두 BC보다 공식 완주 10%p 이상 낮아진 체크포인트가 없다."
    else:
        code = "kl_down"
        line = "(셋 중 해당 없음) KL-PPO가 BC보다 공식 완주 10%p 이상 떨어진 체크포인트가 있다."
    bits = [f"s4_student 공식 완주 {student:.1%}"]
    for run in RUNS:
        rates = [(step, rate) for (name, step), rate in official.items() if name == run]
        worst = min(rates, key=lambda item: item[1])
        bits.append(f"{run} 최저 {worst[1]:.1%} @{worst[0]}")
    return line + " " + NEXT_STEP[code] + "\n\n" + ", ".join(bits) + "."


def merge_watcher():
    old = read_text(OLD_REPORT)
    if not old:
        return
    text = read_text(REPORT)
    for name in ("STATUS", "EVAL", "SUMMARY"):
        body = extract_block(old, name)
        if body:
            text = replace_block(text, name, body)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text)
    os.remove(OLD_REPORT)


def publish(driven, partial):
    """실패 판, 가속 RMS, bs_02를 보고서에 쓴다. partial이면 결론은 아직 확정이 아니다."""
    out_dir = os.path.join(BASE, "results", "klppo_extra")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "details.json"), "w", encoding="utf-8") as f:
        json.dump({str(key): value for key, value in driven.items()}, f, ensure_ascii=False)
    fail_body, rms_body, bs_body, rms_by_model, official = fail_and_metrics(driven)
    note = ""
    if partial:
        note = "공식 평가가 끝난 체크포인트만 넣었다. 나머지가 끝나면 이 표를 다시 쓴다.\n\n"
    write_report("FAIL", note + fail_body)
    write_report("RMS", note + rms_body)
    write_report("BS02", note + bs_body)
    have = {name for name, _step in official}
    if not partial and "s4_student" in have and all(run in have for run in RUNS):
        write_report("VERDICT", verdict(rms_by_model, official))
    elif partial:
        write_report("VERDICT", "앞쪽 평가의 추가 주행이 끝나는 대로 고른다. 아직 체크포인트가 다 모이지 않았다.")
    print("추가 분석 표 갱신", flush=True)


def wait_watch(pid):
    while alive(pid):
        print(f"감시 {pid} 대기 {time.strftime('%H:%M:%S')}", flush=True)
        time.sleep(60)
    print("감시 프로세스 종료", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch-pid", type=int, required=True)
    args = parser.parse_args()
    print("TensorBoard 읽는 중", flush=True)
    write_report("TB", tb_section())
    print("TB 표 기록", flush=True)
    wait_watch(args.watch_pid)
    merge_watcher()
    print("기존 평가 절을 새 보고서로 옮김", flush=True)
    driven = {"s4_student": None}
    print("주행 s4_student", flush=True)
    driven["s4_student"] = drive_model(STUDENT)
    for run in RUNS:
        for step, path in checkpoints(run):
            print(f"주행 {run} {step}", flush=True)
            driven[(run, step)] = drive_model(path)
    out_dir = os.path.join(BASE, "results", "klppo_extra")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "details.json"), "w", encoding="utf-8") as f:
        json.dump({str(key): value for key, value in driven.items()}, f, ensure_ascii=False)
    fail_body, rms_body, bs_body, rms_by_model, official = fail_and_metrics(driven)
    write_report("FAIL", fail_body)
    write_report("RMS", rms_body)
    write_report("BS02", bs_body)
    write_report("VERDICT", verdict(rms_by_model, official))
    print("추가 분석 기록 완료", flush=True)


if __name__ == "__main__":
    main()
