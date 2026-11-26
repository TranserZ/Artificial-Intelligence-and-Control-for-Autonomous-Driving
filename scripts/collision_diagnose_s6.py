"""6단계 충돌 진단: 병렬 주행, 충돌 직전 2초(4스텝), teacher56 shield on 비교."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from multiprocessing import get_context

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.drive_parallel import (  # noqa: E402
    end_reason,
    parse_seeds,
    policy_action,
    prepare_sumo,
    summarize,
)
from utils.official_metrics import EpisodeMetrics, _place_label  # noqa: E402

PREFLIGHT = 4
NEIGHBOR_OFFSETS = (1, 0, -1)
KL_200K = os.path.join(ROOT, "results", "klppo1_s6_s1", "step200k.pt")
STUDENT = r"C:\Users\USER\Desktop\청계로드-dongyeon-d54a761\models\dongyeon_bc\s6_student_obsfix.pt"
OUT_DIR = os.path.join(ROOT, "results", "collision_diag_s6")
PARTIAL_DIR = os.path.join(OUT_DIR, "partials")
REPORT = os.path.join(
    r"C:\Users\USER\Desktop\청계로드\docs\reports",
    "2026-11-25_s6_충돌진단.md",
)
DONGYEON = r"C:\Users\USER\Desktop\청계로드-dongyeon-d54a761"
TEACHER56_PY = os.path.join(DONGYEON, "models", "dongyeon_bc", "teacher56.py")
TEACHER56_CKPT = os.path.join(DONGYEON, "models", "bc_ppo", "teachers", "s4_teacher.pt")


def partial_path(policy: str, seed: int) -> str:
    safe = policy.replace("/", "_")
    return os.path.join(PARTIAL_DIR, f"{safe}_{seed}.json")


def save_partial(policy: str, seed: int, row: dict, coll) -> None:
    os.makedirs(PARTIAL_DIR, exist_ok=True)
    with open(partial_path(policy, seed), "w", encoding="utf-8") as f:
        json.dump({"row": row, "collision": coll}, f, ensure_ascii=False)


def load_partial(policy: str, seed: int):
    path = partial_path(policy, seed)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_env(cfg, shield):
    from env.mdp_config import ACTION, OBSERVATION, SIMULATION
    from env.road_config import EGO, ROAD, TRAFFIC
    from env.sumo_env import SumoHighwayEnv
    from train import REWARD
    sim = {**SIMULATION, "shield": {**SIMULATION["shield"], "mode": shield}}
    return SumoHighwayEnv(
        cfg_path=cfg, road=ROAD, ego=EGO, mdp_sim=sim, mdp_obs=OBSERVATION,
        action=ACTION, reward=REWARD, traffic=TRAFFIC, gui=False)


def neighbor_gaps(env, offset):
    import traci
    if env.ego_id not in traci.vehicle.getIDList():
        return {
            "offset": offset,
            "obs_lead_gap": None,
            "obs_rear_gap": None,
            "scan_lead_gap": None,
            "scan_rear_gap": None,
            "ego_speed": 0.0,
        }
    obs_lead = env._neighbor_base(offset, leader=True)
    obs_rear = env._neighbor_base(offset, leader=False)
    scan_f, scan_r = env._scan_lane_neighbors(offset)
    ego_v = traci.vehicle.getSpeed(env.ego_id) if env.ego_id in traci.vehicle.getIDList() else 0.0
    return {
        "offset": offset,
        "obs_lead_gap": None if obs_lead is None else float(obs_lead[0]),
        "obs_rear_gap": None if obs_rear is None else float(obs_rear[0]),
        "scan_lead_gap": None if scan_f is None else float(scan_f[0]),
        "scan_rear_gap": None if scan_r is None else float(scan_r[0]),
        "ego_speed": float(ego_v),
    }


def collision_partner(env):
    import traci
    for c in traci.simulation.getCollisions():
        if env.ego_id not in (c.collider, c.victim):
            continue
        other = c.victim if c.collider == env.ego_id else c.collider
        speed, vtype = None, "unknown"
        try:
            if other in traci.vehicle.getIDList():
                speed = float(traci.vehicle.getSpeed(other))
                vtype = str(traci.vehicle.getTypeID(other))
            elif other in traci.person.getIDList():
                vtype = "pedestrian"
        except Exception:
            pass
        return {"id": other, "vtype": vtype, "speed": speed}
    return {"id": "", "vtype": "unknown", "speed": None}


def snapshot(env, info, action, policy_name):
    import traci
    if env.ego_id not in traci.vehicle.getIDList():
        return {
            "policy": policy_name,
            "speed": 0.0,
            "accel_model": float(action[0]),
            "accel_raw": float(action[0]),
            "lane_raw": float(action[1]),
            "lane_cmd": 0,
            "lane_applied": False,
            "lane_before": info.get("lane_before"),
            "lane_after": info.get("lane_after"),
            "edge": "",
            "lane_pos": None,
            "lane_idx": None,
            "place": "",
            "neighbors": [],
            "shield_vehicle": False,
            "shield_signal": False,
            "shield_ped": False,
            "shield_lc_gate": False,
            "shield_active": False,
            "gap_same": None,
            "in_stop": False,
            "stop_pen": 0.0,
        }
    edge, pos, lane_idx = "", None, None
    edge = str(traci.vehicle.getRoadID(env.ego_id))
    pos = float(traci.vehicle.getLanePosition(env.ego_id))
    lane_idx = int(traci.vehicle.getLaneIndex(env.ego_id))
    place = _place_label(env, info)
    return {
        "policy": policy_name,
        "speed": float(info.get("speed") or 0),
        "accel_model": float(info.get("accel_model", action[0])),
        "accel_raw": float(info.get("accel_raw", action[0])),
        "lane_raw": float(info.get("lane_change_raw", action[1])),
        "lane_cmd": int(info.get("lane_change", 0)),
        "lane_applied": bool(info.get("lane_change_applied")),
        "lane_before": info.get("lane_before"),
        "lane_after": info.get("lane_after"),
        "edge": edge,
        "lane_pos": pos,
        "lane_idx": lane_idx,
        "place": place,
        "neighbors": [neighbor_gaps(env, o) for o in NEIGHBOR_OFFSETS],
        "shield_vehicle": bool(info.get("shield_vehicle")),
        "shield_signal": bool(info.get("shield_signal")),
        "shield_ped": bool(info.get("shield_ped")),
        "shield_lc_gate": bool(info.get("shield_lc_gate")),
        "shield_active": bool(info.get("shield_active")),
        "gap_same": float(info["gap"]) if info.get("gap") is not None else None,
        "in_stop": bool(info.get("in_stop_area")),
        "stop_pen": float(info.get("stop_pen") or 0),
    }


def classify_collision(seed, steps, partner, tail, place, edge, pos):
    had_lc = any(s["lane_cmd"] != 0 or s["lane_applied"] for s in tail)
    partner_t = str(partner.get("vtype", "")).lower()
    if "moto" in partner_t or "bike" in partner_t or partner_t == "pedestrian":
        return "d_motorcycle_or_ped", had_lc
    missed_obs = False
    for s in tail[-PREFLIGHT:]:
        for n in s["neighbors"]:
            ol, sl = n["obs_lead_gap"], n["scan_lead_gap"]
            if ol is not None and sl is not None and sl < ol - 2.0 and sl < 15.0:
                missed_obs = True
    if had_lc and not missed_obs:
        return "a_lc_obs_visible", had_lc
    if had_lc and missed_obs:
        return "b_lc_obs_missed", had_lc
    if str(place).startswith("hs_") or (edge == "e908646367_6" and pos and 60 <= pos <= 85):
        return "c_rear_or_parked", had_lc
    if str(place).startswith("hs_") or "samil" in str(place) or edge == "e908646367_3":
        gap_small = any((s.get("gap_same") or 99) < 8 for s in tail)
        if gap_small or edge == "e908646367_3":
            return "c_rear_or_parked", had_lc
    return "e_other", had_lc


def run_episode(cfg, seed, model_path, policy_name, shield="shadow"):
    import traci
    from algorithms.kl_ppo import load_eval_policy
    from train import HPARAMS
    env = make_env(cfg, shield)
    policy = load_eval_policy(model_path, 67, 2, HPARAMS)
    obs, _ = env.reset(seed=int(seed))
    buf = deque(maxlen=PREFLIGHT + 1)
    met = EpisodeMetrics()
    terminated = truncated = False
    info = {}
    action = np.zeros(2, dtype=np.float32)
    yield_steps = shield_steps = 0
    coll_rec = None
    try:
        while not (terminated or truncated):
            odo_before = None
            if env.ego_id in traci.vehicle.getIDList():
                odo_before = float(traci.vehicle.getDistance(env.ego_id))
            action = policy_action(policy, obs, 0.0, None)
            obs, _, terminated, truncated, info = env.step(action)
            met.observe_step(env, info, odo_before)
            yield_steps += int(bool(info.get("yield_violation")))
            shield_steps += int(bool(info.get("shield_active")))
            buf.append(snapshot(env, info, action, policy_name))
            if info.get("collided"):
                partner = collision_partner(env)
                tail = list(buf)
                place = tail[-1]["place"] if tail else ""
                edge = tail[-1]["edge"] if tail else ""
                pos = tail[-1]["lane_pos"] if tail else None
                cls, had_lc = classify_collision(seed, met.step, partner, tail, place, edge, pos)
                coll_rec = {
                    "seed": seed,
                    "policy": policy_name,
                    "shield": shield,
                    "class": cls,
                    "had_lane_change": had_lc,
                    "partner": partner,
                    "place": place,
                    "edge": edge,
                    "lane_pos": pos,
                    "stops_done": int(info.get("stops_done", 0) or 0),
                    "preflight_steps": tail[-PREFLIGHT:] if len(tail) >= PREFLIGHT else tail,
                }
                break
    finally:
        reason = end_reason(info, terminated, truncated)
        met.finalize(env, info, reason, terminated, truncated)
        place_end = _place_label(env, info)
        env.close()
    row = met.row_dict(
        seed, reason,
        int(info.get("stops_done", 0) or 0),
        int(info.get("stops_total", 0) or 0),
        place_end,
        yield_steps=yield_steps,
        shield_steps=shield_steps,
        ego_edge=buf[-1]["edge"] if buf else "",
    )
    return row, coll_rec


def _stage1_job(args):
    seed, model_path, policy_name = args
    cached = load_partial(policy_name, seed)
    if cached is not None:
        return policy_name, cached["row"], cached["collision"]
    os.environ["CGC_STAGE"] = "6"
    os.environ["OBS_SCAN"] = "1"
    import torch
    torch.set_num_threads(1)
    work, cfg = prepare_sumo()
    try:
        row, coll = run_episode(cfg, seed, model_path, policy_name, "shadow")
        save_partial(policy_name, seed, row, coll)
        return policy_name, row, coll
    finally:
        shutil.rmtree(work, ignore_errors=True)


def teacher_end_from_json(metrics: dict) -> str:
    if metrics.get("collision_rate", 0) >= 0.5:
        return "collision"
    if metrics.get("wrong_lane_rate", 0) >= 0.5:
        return "lane_error"
    if metrics.get("red_violation_rate", 0) >= 0.5:
        return "red_light"
    if metrics.get("timeout_rate", 0) >= 0.5:
        return "timeout"
    if metrics.get("completion_rate", 0) >= 0.5 or metrics.get("success_rate", 0) >= 0.5:
        return "completed"
    return "other"


def _teacher56_job(seed: int) -> tuple[int, str]:
    cache = os.path.join(PARTIAL_DIR, f"teacher56_{seed}.json")
    if os.path.isfile(cache):
        with open(cache, encoding="utf-8") as f:
            return seed, json.load(f)["end_reason"]
    os.makedirs(PARTIAL_DIR, exist_ok=True)
    jpath = os.path.join(PARTIAL_DIR, f"_teacher56_eval_{seed}.json")
    env = {
        **os.environ,
        "CGC_STAGE": "6",
        "OBS_SCAN": "1",
        "TEACHER_AVOID": "1",
        "TEACHER_AVOID_GATE": "1",
        "BCPPO_YELLOW_DECEL": "2.4",
        "BCPPO_LC_GATE": '{"rear_tau":3.0,"approach_horizon":4.0,"min_rear_gap":8.0}',
    }
    cmd = [
        sys.executable,
        TEACHER56_PY,
        "eval",
        "--ckpt",
        TEACHER56_CKPT,
        "--shield",
        "on",
        "--expert-gate",
        "--episodes",
        "1",
        "--seed-base",
        str(seed),
        "--json",
        jpath,
    ]
    subprocess.run(cmd, cwd=DONGYEON, env=env, check=True, capture_output=True)
    with open(jpath, encoding="utf-8") as f:
        payload = json.load(f)
    reason = teacher_end_from_json(payload)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump({"end_reason": reason}, f)
    try:
        os.remove(jpath)
    except OSError:
        pass
    return seed, reason


def build_report(payload, seeds: list[int], workers: int):
    lines = [
        "# 6단계 충돌 진단 (2026-11-25)",
        "",
        f"개발 {len(seeds)}판 `900100–900159`, `OBS_SCAN=1`, 결정적, shield shadow, "
        f"워커 {workers}, 충돌 직전 **2초(4스텝)**. "
        f"모델: `klppo1_s6_s1 step200k`, 학생 `s6_student_obsfix`. "
        "2단계: 충돌 시드만 `teacher56`(s4_teacher + 회피 규칙, shield on).",
        "",
        f"## A~H 요약 ({len(seeds)}판)",
        "",
        "| 모델 | 공식 완주 | 완벽 완주 | 충돌 | wrong_lane | 무한대기+TO | 비정상정지 시간비(평균) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, st in payload["summary"].items():
        n = st["episodes"]
        lines.append(
            f"| {name} | {st['official_complete']*n:.0f}/{n} ({st['official_complete']:.1%}) | "
            f"{st['perfect_complete']*n:.0f}/{n} ({st['perfect_complete']:.1%}) | "
            f"{st['collision_count']}/{n} ({st['collision_rate']:.1%}) | "
            f"{st['wrong_lane_count']} | {st['inf_wait_episodes']} | "
            f"{st['green_stop_time_mean']:.1%} |"
        )
    lines.extend(["", "## 충돌 유형 비율", ""])
    for pol in ("klppo1_s6_s1_200k", "s6_student_obsfix"):
        colls = [c for c in payload["collisions"] if c["policy"] == pol]
        if not colls:
            lines.append(f"- **{pol}**: 충돌 0")
            continue
        from collections import Counter
        cnt = Counter(c["class"] for c in colls)
        lines.append(f"### {pol} ({len(colls)}판)")
        lines.append("| 유형 | 판수 | 비율 |")
        lines.append("|---|---:|---:|")
        labels = {
            "a_lc_obs_visible": "(a) 관측 보였는데 차선변경",
            "b_lc_obs_missed": "(b) 관측 놓쳤는데 차선변경",
            "c_rear_or_parked": "(c) 앞차·주정차 추돌",
            "d_motorcycle_or_ped": "(d) 오토바이·보행",
            "e_other": "(e) 기타",
        }
        for k, v in cnt.most_common():
            lines.append(f"| {labels.get(k,k)} | {v} | {v/len(colls):.0%} |")
        lines.append("")
    lines.extend([
        "## 충돌 위치", "",
        "| 시드 | 모델 | 유형 | place | edge | pos | 상대 | teacher56 on |",
        "|---:|---|---|---|---|---:|---|---|",
    ])
    teach = payload.get("teacher_on", {})
    for c in sorted(payload["collisions"], key=lambda x: (x["policy"], x["seed"])):
        t = teach.get(str(c["seed"]), teach.get(c["seed"], "?"))
        lines.append(
            f"| {c['seed']} | {c['policy']} | {c['class']} | {c['place']} | {c['edge']} | "
            f"{c.get('lane_pos', '')} | {c['partner'].get('vtype')} | {t} |"
        )
    lines.extend(["", "## 대표 사례 (충돌 직전 4스텝)", ""])
    examples = sorted(
        payload["collisions"],
        key=lambda c: (0 if c["policy"] == "klppo1_s6_s1_200k" else 1, c["seed"]),
    )[:3]
    for i, c in enumerate(examples, 1):
        lines.append(f"### 사례 {i}: 시드 {c['seed']} ({c['policy']}) — {c['class']}")
        lines.append(f"- place `{c['place']}`, partner `{c['partner']}`")
        for j, s in enumerate(c["preflight_steps"], 1):
            n0 = next((n for n in s["neighbors"] if n["offset"] == 0), {})
            lines.append(
                f"- t−{len(c['preflight_steps'])-j+1}: v={s['speed']:.2f} m/s, "
                f"accel={s['accel_model']:.2f}, lane_cmd={s['lane_cmd']}, "
                f"applied={s['lane_applied']}, edge={s['edge']} pos={s['lane_pos']}, "
                f"obs/scan lead0={n0.get('obs_lead_gap')} / {n0.get('scan_lead_gap')}, "
                f"shadow veh/sig/lc={s['shield_vehicle']}/{s['shield_signal']}/{s['shield_lc_gate']}"
            )
        lines.append("")
    lines.extend(["## C·D (무한대기·비정상 정지) 요약", ""])
    for pol, rows in payload["cd_events"].items():
        inf = [r for r in rows if r.get("inf_wait_count", 0) > 0 or r["end_reason"] == "timeout"]
        gs = [r for r in rows if r.get("green_stop_time_ratio", 0) > 0.05]
        lines.append(f"**{pol}**: 무한대기·timeout {len(inf)}판, 비정상정지 비율>5% {len(gs)}판.")
        jugyo = [r for r in inf if "jugyo" in str(r.get("place", "")) or str(r.get("ego_edge", "")).startswith("e218924602")]
        if jugyo:
            lines.append(f"- 주교(e218924602_*) 근처: {[r['seed'] for r in jugyo[:8]]}")
    lines.extend(_countermeasure_section(payload))
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote", REPORT, flush=True)


def _countermeasure_section(payload):
    colls = payload["collisions"]
    kl = [c for c in colls if c["policy"] == "klppo1_s6_s1_200k"]
    lc_frac = sum(1 for c in kl if c["class"].startswith("a_") or c["class"].startswith("b_")) / max(len(kl), 1)
    missed = sum(1 for c in kl if c["class"] == "b_lc_obs_missed") / max(len(kl), 1)
    parked = sum(1 for c in kl if c["class"] == "c_rear_or_parked") / max(len(kl), 1)
    return [
        "",
        "## 대책 적합성 (진단 근거)",
        "",
        "| 대책 | 적합 | 근거 |",
        "|---|:---:|---|",
        f"| ① 차선 학습+KL+τ | {'높음' if lc_frac >= 0.25 else '낮음'} | KL 200k 충돌 중 차선변경 연관 {lc_frac:.0%} |",
        f"| ② shadow 개입 시 감점 | 중간 | 직전 스텝 shield_vehicle/lc_gate 빈도는 사례 표 참고. 충돌 −20 대비 소감점(0.05~0.2/스텝) |",
        "| ③ replay 0.3 | 중간 | 900114·115·117 학생도 동일 장면 충돌·유전 실패 |",
        f"| ④ 관측 보강(도착예상시간) | {'높음' if missed >= 0.25 else '낮음'} | (b) 관측놓침+차선변경 {missed:.0%} |",
        f"| ⑤ 환경 조정(주교 주정차) | {'팀 결정' if parked >= 0.25 else '보류'} | (c) 주정차·samil/ipjeong {parked:.0%} |",
        "",
        "## 추천 조합 (확정 전안)",
        "",
        "① 차선 2차 + ② shadow-위반 소감점(0.1/스텝, 차선게이트·앞차 shield가 켜졌을 때) + ③ replay 0.3. "
        "④는 (b) 비율이 25% 넘으면 추가. ⑤는 ipjeong/samil 엣지 배치는 팀 결정.",
        "",
        "<!-- 설계 확정 후 시험 학습 진행 -->",
    ]


def estimate_minutes(n_seeds: int, workers: int, t_ep_min: float = 3.0, t_teacher_min: float = 4.0) -> float:
    n_jobs = n_seeds * 2
    stage1 = (n_jobs / workers) * t_ep_min
    n_coll = max(4, int(n_seeds * 0.2) * 2)
    stage2 = (n_coll / workers) * t_teacher_min
    return stage1 + stage2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="900100-900159")
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--force-30", action="store_true", help="1단계 30판(900100-900129)만")
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    if args.force_30:
        seeds = list(range(900100, 900130))
    workers = max(1, min(args.workers, 7))
    est = estimate_minutes(len(seeds), workers)
    if est > 125 and len(seeds) > 30:
        print(f"예상 {est:.0f}분 > 2시간 → 30판으로 축소", flush=True)
        seeds = list(range(900100, 900130))
        est = estimate_minutes(len(seeds), workers)
    print(f"진단: seeds={len(seeds)} workers={workers} 예상~{est:.0f}분", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.environ["CGC_STAGE"] = "6"
    os.environ["OBS_SCAN"] = "1"

    jobs = []
    for seed in seeds:
        jobs.append((seed, KL_200K, "klppo1_s6_s1_200k"))
        jobs.append((seed, STUDENT, "s6_student_obsfix"))

    print(f"=== 1단계 {len(jobs)} job (병렬 {workers}) ===", flush=True)
    ctx = get_context("spawn")
    all_rows = {"klppo1_s6_s1_200k": [], "s6_student_obsfix": []}
    collisions = []
    with ctx.Pool(workers) as pool:
        for policy_name, row, coll in pool.imap_unordered(_stage1_job, jobs, chunksize=1):
            all_rows[policy_name].append(row)
            if coll:
                collisions.append(coll)
            done = sum(len(v) for v in all_rows.values())
            if done % 14 == 0 or done == len(jobs):
                print(f"  1단계 {done}/{len(jobs)}", flush=True)

    for name in all_rows:
        all_rows[name].sort(key=lambda r: r["seed"])

    coll_seeds = sorted({c["seed"] for c in collisions})
    teacher_on = {}
    if coll_seeds:
        print(f"=== 2단계 teacher56 {len(coll_seeds)} 시드 ===", flush=True)
        with ctx.Pool(min(workers, len(coll_seeds))) as pool:
            for seed, reason in pool.imap_unordered(_teacher56_job, coll_seeds, chunksize=1):
                teacher_on[seed] = reason
                print(f"  teacher56 seed {seed} → {reason}", flush=True)

    payload = {
        "summary": {k: summarize(v) for k, v in all_rows.items()},
        "collisions": collisions,
        "teacher_on": teacher_on,
        "cd_events": all_rows,
        "meta": {"seeds": seeds, "workers": workers, "estimate_min": est},
    }
    with open(os.path.join(OUT_DIR, "diag.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    build_report(payload, seeds, workers)


if __name__ == "__main__":
    main()