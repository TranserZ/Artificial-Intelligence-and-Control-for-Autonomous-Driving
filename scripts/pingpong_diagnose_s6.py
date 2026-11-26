"""6단계 정류장 핑퐁(5초 내 반대 차선변경) 진단 — 학습 없음."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from multiprocessing import get_context

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONGYEON = os.path.join(os.path.dirname(ROOT), "청계로드-dongyeon-d54a761")
sys.path.insert(0, ROOT)

from scripts.drive_parallel import end_reason, policy_action, prepare_sumo  # noqa: E402

SEEDS_30 = list(range(900100, 900130))
PING_WINDOW = 10  # 5s @ 0.5s
KL_200K = os.path.join(ROOT, "results", "klppo1_s6_s1", "step200k.pt")
STUDENT = os.path.join(DONGYEON, "models", "dongyeon_bc", "s6_student_obsfix.pt")
TEACHER_CKPT = os.path.join(DONGYEON, "models", "bc_ppo", "teachers", "s4_teacher.pt")
OUT = os.path.join(ROOT, "results", "pingpong_s6")


def _make_env(cfg=None, shield="shadow", teacher=False):
    from env.mdp_config import ACTION, OBSERVATION, SIMULATION
    from env.road_config import EGO, ROAD, TRAFFIC
    from env.sumo_env import SumoHighwayEnv
    from train import REWARD
    sim = {**SIMULATION, "shield": {**SIMULATION["shield"], "mode": shield}}
    if not teacher:
        return SumoHighwayEnv(
            cfg_path=cfg, road=ROAD, ego=EGO, mdp_sim=sim, mdp_obs=OBSERVATION,
            action=ACTION, reward=REWARD, traffic=TRAFFIC, gui=False,
        )
    sys.path.insert(0, os.path.join(DONGYEON, "models", "bc_ppo"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "teacher56", os.path.join(DONGYEON, "models", "dongyeon_bc", "teacher56.py"))
    mod = importlib.util.module_from_spec(spec)
    # This worktree already implements OBS_SCAN in SumoHighwayEnv.
    # Keep the teacher rules while avoiding a second scan wrapper around the same method.
    scan = os.environ.get("OBS_SCAN")
    try:
        os.environ["OBS_SCAN"] = "0"
        spec.loader.exec_module(mod)
    finally:
        if scan is None:
            os.environ.pop("OBS_SCAN", None)
        else:
            os.environ["OBS_SCAN"] = scan
    import bcppo  # noqa: E402
    env = bcppo.make_env(
        shield, lc_gate=bcppo.EXPERT_LC_GATE, teacher=True, env_cls=bcppo.TeacherEnv)
    env.eval_mode = True
    return env, bcppo


def _in_no_stop_zone(env, edge, pos):
    import traci
    if edge.startswith(":"):
        rid = edge[1 : edge.rfind("_")]
        if rid in getattr(env, "_no_stop_junctions", set()):
            return traci.vehicle.getSpeed(env.ego_id) < 0.5
    return any(z[0] == edge and z[1] <= pos <= z[2] for z in getattr(env, "_no_stop_zones", []))


def detect_pingpong(env, info_hist):
    """info_hist: list of per-step dicts with lane_applied, lane_cmd, step, stop_id, edge, pos."""
    events = []
    last_dir = 0
    last_step = -999
    for row in info_hist:
        cmd = int(row.get("lane_cmd", 0))
        applied = bool(row.get("lane_applied"))
        if cmd == 0 or not applied:
            continue
        step = int(row["step"])
        if last_dir != 0 and cmd == -last_dir and (step - last_step) <= PING_WINDOW:
            events.append({
                "step": step,
                "stop_id": row.get("stop_id", ""),
                "edge": row.get("edge", ""),
                "pos": row.get("pos"),
                "lane_idx": row.get("lane_idx"),
                "seq": (last_step, step),
            })
        last_dir = cmd
        last_step = step
    return events


def run_policy_episode(cfg, seed, model_path, policy_name, teacher=False):
    import traci
    from algorithms.kl_ppo import load_eval_policy
    from train import HPARAMS

    bcppo_mod = None
    if teacher:
        env, bcppo_mod = _make_env(cfg, "shadow", teacher=True)
        agent = bcppo_mod.make_agent(env, 0)
        bcppo_mod.load_weights(agent, TEACHER_CKPT)
        policy = agent
        act_fn = lambda o: np.asarray(policy.predict(o, deterministic=True), dtype=np.float32)
    else:
        env = _make_env(cfg, "shadow", teacher=False)
        policy = load_eval_policy(model_path, 67, 2, HPARAMS)
        act_fn = lambda o: policy_action(policy, o, 0.0, None)

    obs, _ = env.reset(seed=int(seed))
    hist = []
    step = 0
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        action = act_fn(obs)
        obs, _, terminated, truncated, info = env.step(action)
        step += 1
        edge = pos = lane_idx = ""
        st = env._current_stop()
        stop_id = st["id"] if st else ""
        if env.ego_id in traci.vehicle.getIDList():
            edge = traci.vehicle.getRoadID(env.ego_id)
            pos = float(traci.vehicle.getLanePosition(env.ego_id))
            lane_idx = int(traci.vehicle.getLaneIndex(env.ego_id))
        hist.append({
            "step": step,
            "stop_id": stop_id,
            "stop_i": env._stop_i,
            "edge": edge,
            "pos": pos,
            "lane_idx": lane_idx,
            "lane_cmd": int(info.get("lane_change", 0)),
            "lane_raw": float(info.get("lane_change_raw", 0)),
            "lane_applied": bool(info.get("lane_change_applied")),
            "bus_stop_obs": list(env._stop_obs()) if env.ego_id in traci.vehicle.getIDList() else [1, 0, 0],
            "in_stop_area": bool(st and env._in_stop_area(st)) if st else False,
            "dwell_steps": int(getattr(env, "_dwell_steps", 0)),
            "lc_forbidden": bool(info.get("lc_forbidden")),
            "no_stop_zone": _in_no_stop_zone(env, edge, pos) if edge else False,
            "shield_lc_gate": bool(info.get("shield_lc_gate")),
            "teacher_rule": bool(info.get("teacher_rule")) if teacher else False,
        })
    env.close()
    reason = end_reason(info, terminated, truncated)
    return reason, detect_pingpong(env, hist), hist


def _scan_job(job):
    seed, kind = job
    os.environ["CGC_STAGE"] = "6"
    os.environ["OBS_SCAN"] = "1"
    import torch
    torch.set_num_threads(1)
    import shutil
    work, cfg = prepare_sumo()
    try:
        if kind == "klppo":
            reason, pings, _ = run_policy_episode(cfg, seed, KL_200K, "klppo", teacher=False)
        elif kind == "student":
            reason, pings, _ = run_policy_episode(cfg, seed, STUDENT, "student", teacher=False)
        else:
            reason, pings, _ = run_policy_episode(cfg, seed, TEACHER_CKPT, "teacher56", teacher=True)
        return {"seed": seed, "policy": kind, "end_reason": reason, "pings": pings}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def step_trace(cfg, seed, model_path, teacher=False, lane_lock=False, teacher_accel_only=False):
    """한 판 매 스텝 기록 + lane lock 실험."""
    import traci
    from algorithms.kl_ppo import load_eval_policy
    from train import HPARAMS
    import bcppo as bcppo_mod

    if teacher or teacher_accel_only:
        env, bcppo_mod = _make_env(cfg, "on" if teacher_accel_only else "shadow", teacher=True)
        if teacher_accel_only:
            orig = bcppo_mod.TeacherEnv._stop_rule

            def _stop_accel_only(self, accel_raw, lane_raw):
                a, _, c = orig(self, accel_raw, 0.0)
                return a, 0.0, c

            bcppo_mod.TeacherEnv._stop_rule = _stop_accel_only
        agent = bcppo_mod.make_agent(env, 0)
        bcppo_mod.load_weights(agent, TEACHER_CKPT)
        act_fn = lambda o: np.asarray(agent.predict(o, deterministic=True), dtype=np.float32)
    else:
        env = _make_env(cfg, "shadow", teacher=False)
        policy = load_eval_policy(model_path, 67, 2, HPARAMS)
        act_fn = lambda o: policy_action(policy, o, 0.0, None)

    obs, _ = env.reset(seed=int(seed))
    rows = []
    step = 0
    terminated = truncated = False
    info = {}
    prev_stop_i = 0
    dwell_ok = False
    while not (terminated or truncated):
        action = act_fn(obs)
        if lane_lock:
            action = np.array([action[0], 0.0], dtype=np.float32)
        lane_probs = None
        if not teacher and not teacher_accel_only:
            import torch
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                dist = policy.policy.forward(obs_t)[0]
                if hasattr(dist, "distribution"):
                    lane_probs = dist.distribution.probs[0, 1].cpu().numpy().tolist()
        obs, _, terminated, truncated, info = env.step(action)
        step += 1
        sim_t = step * env.step_length
        edge = pos = lane_idx = ""
        n_lanes = None
        car_lo = car_hi = None
        st = env._current_stop()
        stop_id = st["id"] if st else ""
        if env.ego_id in traci.vehicle.getIDList():
            edge = traci.vehicle.getRoadID(env.ego_id)
            pos = float(traci.vehicle.getLanePosition(env.ego_id))
            lane_idx = int(traci.vehicle.getLaneIndex(env.ego_id))
            n_lanes = int(traci.edge.getLaneNumber(edge)) if not edge.startswith(":") else None
            lo, hi = env._car_lanes(edge)
            car_lo, car_hi = lo, hi
        bso = list(env._stop_obs()) if env.ego_id in traci.vehicle.getIDList() else [1, 0, 0]
        ns = env._next_stop()
        stop_changed = env._stop_i != prev_stop_i
        prev_stop_i = env._stop_i
        in_area = bool(st and env._in_stop_area(st)) if st else False
        v = float(info.get("speed") or 0)
        if in_area and v <= env.stop_speed:
            if env._dwell_steps * env.step_length >= env.stop_dwell_time - 1e-9:
                dwell_ok = True
        row = {
            "step": step, "sim_t": sim_t,
            "stop_id": stop_id, "stop_i": env._stop_i, "stop_changed": stop_changed,
            "stop_def": dict(st) if st else None,
            "next_stop_raw": list(ns) if ns else None,
            "edge": edge, "pos": pos, "lane_idx": lane_idx,
            "edge_lane_count": n_lanes, "car_lane_lo": car_lo, "car_lane_hi": car_hi,
            "bus_stop_obs": {"dist": bso[0], "lane_diff": bso[1], "dwell": bso[2]},
            "in_stop_area": in_area,
            "dwell_steps": env._dwell_steps,
            "speed": v,
            "lane_cmd": int(info.get("lane_change", 0)),
            "lane_raw": float(info.get("lane_change_raw", 0)),
            "lane_applied": bool(info.get("lane_change_applied")),
            "lane_before": info.get("lane_before"),
            "lane_after": info.get("lane_after"),
            "lc_forbidden": bool(info.get("lc_forbidden")),
            "no_stop_zone_slow": _in_no_stop_zone(env, edge, pos) if edge else False,
            "shield_lc_gate": bool(info.get("shield_lc_gate")),
            "shield_vehicle": bool(info.get("shield_vehicle")),
            "teacher_rule": bool(info.get("teacher_rule")),
            "lane_probs": lane_probs,
        }
        rows.append(row)
    env.close()
    return {
        "end_reason": end_reason(info, terminated, truncated),
        "dwell_success": dwell_ok,
        "stops_done": int(info.get("stops_done", 0) or 0),
        "rows": rows,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    import sys as _sys
    if "--no-teacher" in _sys.argv:
        jobs = [(s, "klppo") for s in SEEDS_30] + [(s, "student") for s in SEEDS_30]
        print(f"scan {len(jobs)} episodes (kl+student)...", flush=True)
    else:
        jobs = [(s, "klppo") for s in SEEDS_30] + [(s, "student") for s in SEEDS_30] + [(s, "teacher56") for s in SEEDS_30]
        print(f"scan {len(jobs)} episodes (+teacher56)...", flush=True)
    with get_context("spawn").Pool(7) as pool:
        scan = pool.map(_scan_job, jobs, chunksize=1)

    by_stop = defaultdict(lambda: Counter())
    for rec in scan:
        pol = rec["policy"]
        for ev in rec["pings"]:
            sid = ev.get("stop_id") or "unknown"
            by_stop[sid][pol] += 1

    # pick trace seed: most pings at bs_03 for student
    best_seed = 900114
    best_n = -1
    for rec in scan:
        if rec["policy"] != "student":
            continue
        n = sum(1 for p in rec["pings"] if p.get("stop_id") == "bs_03")
        if n > best_n:
            best_n = n
            best_seed = rec["seed"]
    if best_n <= 0:
        for rec in scan:
            if rec["policy"] == "student" and rec["pings"]:
                best_seed = rec["seed"]
                break

    os.environ["CGC_STAGE"] = "6"
    os.environ["OBS_SCAN"] = "1"
    import shutil
    work, cfg = prepare_sumo()
    try:
        trace = step_trace(cfg, best_seed, STUDENT, teacher=False, lane_lock=False)
        lock_policy = step_trace(cfg, best_seed, STUDENT, teacher=False, lane_lock=True)
        lock_teacher = step_trace(cfg, best_seed, TEACHER_CKPT, teacher_accel_only=True, lane_lock=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # find sim time for GUI around first ping at bs_03
    gui_hint = {"seed": best_seed, "sim_t": None, "step": None}
    for row in trace["rows"]:
        if row["stop_id"] == "bs_03" and row.get("lane_applied") and row["lane_cmd"] != 0:
            gui_hint["sim_t"] = row["sim_t"]
            gui_hint["step"] = row["step"]
            break

    payload = {
        "scan": scan,
        "by_stop": {k: dict(v) for k, v in by_stop.items()},
        "trace_seed": best_seed,
        "trace": trace,
        "lane_lock_policy": lock_policy,
        "lane_lock_teacher_accel": lock_teacher,
        "gui_hint": gui_hint,
    }
    with open(os.path.join(OUT, "diag.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("wrote", os.path.join(OUT, "diag.json"), "trace_seed", best_seed, flush=True)
    try:
        from scripts.build_pingpong_report import main as build_report
        build_report()
    except Exception as e:
        print("report skip:", e, flush=True)


if __name__ == "__main__":
    main()