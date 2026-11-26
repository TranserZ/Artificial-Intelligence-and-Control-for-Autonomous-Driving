"""PPO 또는 KL-PPO 체크포인트를 여러 판에 병렬로 달린다.

    $env:CGC_STAGE = "4"
    py -3.13 scripts/drive_parallel.py --model <ckpt> --seeds 900100-900129 --shield shadow
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from multiprocessing import get_context

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def parse_seeds(text: str) -> list:
    seeds = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def end_reason(info, terminated, truncated) -> str:
    if info.get("sumo_error"):
        return "sumo_error"
    if info.get("collided"):
        return "collision"
    if info.get("red_violation") and terminated:
        return "red_light"
    if info.get("wrong_lane"):
        return "lane_error"
    if info.get("arrived"):
        return "missed_stop" if int(info.get("missed_stops", 0) or 0) else "completed"
    if truncated:
        return "timeout"
    return "vanished"


def _make_env(sumocfg, shield):
    from env.mdp_config import ACTION, OBSERVATION, SIMULATION
    from env.road_config import EGO, ROAD, TRAFFIC
    from env.sumo_env import SumoHighwayEnv
    from train import REWARD
    sim = {**SIMULATION, "shield": {**SIMULATION["shield"], "mode": shield}}
    return SumoHighwayEnv(
        cfg_path=sumocfg, road=ROAD, ego=EGO, mdp_sim=sim, mdp_obs=OBSERVATION,
        action=ACTION, reward=REWARD, traffic=TRAFFIC, gui=False)


def _ego_place(env):
    """스텝 직전 위치. 충돌로 차가 사라지기 전의 엣지, 차로 위치, 다음 정류장."""
    import traci
    edge, stop, pos = "", "", None
    try:
        if env.ego_id in traci.vehicle.getIDList():
            edge = str(traci.vehicle.getRoadID(env.ego_id))
            if edge and not edge.startswith(":"):
                pos = float(traci.vehicle.getLanePosition(env.ego_id))
        current = env._current_stop()
        if current is not None:
            stop = str(current["id"])
    except Exception:
        pass
    return edge, stop, pos


def _closer_place(env, edge, pos, stop_id):
    """같은 엣지에서 더 가까운 정류장 또는 이번 판 주정차 지점. 없으면 다음 정류장."""
    stop = None
    for item in getattr(env, "_stops", []):
        if str(item.get("id")) == stop_id:
            stop = item
            break
    stop_dist = None
    if stop is not None and pos is not None and stop.get("edge") == edge:
        stop_dist = min(abs(pos - float(stop["start"])), abs(pos - float(stop["end"])))
    park_id, park_dist = "", None
    if pos is not None and edge and not edge.startswith(":"):
        for (hot_edge, _lane), positions in getattr(env, "_hotspot_taken", {}).items():
            if hot_edge != edge:
                continue
            for hot_pos in positions:
                dist = abs(float(hot_pos) - pos)
                if park_dist is None or dist < park_dist:
                    park_dist = dist
        if park_dist is not None:
            for hid, hot_edge, _count in getattr(env, "_hotspot_placed", []):
                if hot_edge == edge:
                    park_id = str(hid)
                    break
    if park_id and (stop_dist is None or park_dist < stop_dist):
        return park_id
    return stop_id


def policy_action(policy, obs, explore_std=0.0, rng=None):
    """평가와 같은 행동. 결정적 predict(가속 평균, 차선 argmax).

    explore_std > 0 이면 가속 평균에 N(0, σ)만 더하고 [-1, 1]로 자른다.
    """
    action = np.asarray(policy.predict(obs, deterministic=True), dtype=np.float32).copy()
    if explore_std and float(explore_std) > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        action[0] = np.float32(np.clip(action[0] + rng.normal(0.0, float(explore_std)), -1.0, 1.0))
    return action


def run_seed(sumocfg, seed, model, shield, deterministic=True, explore_std=0.0,
             lane_temp=0.0, stochastic_lane=False, trace_path=None, teacher=False, lane_hold=False):
    import torch
    import traci
    from algorithms.kl_ppo import KLPPO, load_eval_policy
    from train import HPARAMS
    from utils.official_metrics import EpisodeMetrics
    torch.set_num_threads(1)
    if teacher:
        from scripts.pingpong_diagnose_s6 import _make_env as make_teacher
        env, teacher_module = make_teacher(sumocfg, "on", teacher=True)
        policy = teacher_module.make_agent(env, 0)
        teacher_module.load_weights(policy, model)
        teacher_trace = {}
        original_rules = env.teacher_rules

        def record_teacher_rules(accel, lane):
            _, stop_lane, _ = env._stop_rule(accel, lane)
            accel, final_lane, used = original_rules(accel, lane)
            edge, stop_id, pos = _ego_place(env)
            stop = env._current_stop()
            hold = (lane_hold and stop_id == "bs_03" and edge == stop["edge"] and pos is not None
                    and stop["start"] - 0.5 <= pos <= stop["end"] + env.stop_overshoot
                    and traci.vehicle.getLaneIndex(env.ego_id) == stop["lane"])
            teacher_trace.update(teacher_stop_lane=stop_lane, teacher_final_lane=final_lane,
                                 lane_hold=hold)
            return accel, 0.0 if hold else final_lane, used

        env.teacher_rules = record_teacher_rules
    else:
        env = _make_env(sumocfg, shield)
        policy = load_eval_policy(
            model, env.observation_space.shape[0], env.action_space.shape[0], HPARAMS)
    if lane_temp > 0 and isinstance(policy, KLPPO):
        policy.policy.lane_temp = float(lane_temp)
    rng = np.random.default_rng(int(seed) + 17)
    obs, _ = env.reset(seed=int(seed))
    if trace_path:
        from traci import constants
        traci.vehicle.subscribeContext(env.ego_id, constants.CMD_GET_VEHICLE_VARIABLE,
                                       50.0, (constants.VAR_POSITION,))
    terminated = truncated = False
    info = {}
    met = EpisodeMetrics()
    speed_sum = speed_n = shield_steps = yield_steps = 0
    zone_steps = zone_over_steps = 0
    zone_max_excess = None
    last_edge = ""
    last_pos = None
    next_stop = ""
    missed_ids = []
    place = ""
    trace = []
    stop_obs_index = env.obs_names().index("obs.bus_stop.dist") if trace_path else None
    try:
        while not (terminated or truncated):
            edge, stop, pos = _ego_place(env)
            if edge:
                last_edge = edge
                last_pos = pos
            if stop:
                next_stop = stop
            odo_before = None
            if env.ego_id in traci.vehicle.getIDList():
                odo_before = float(traci.vehicle.getDistance(env.ego_id))
            if stochastic_lane and isinstance(policy, KLPPO):
                action = np.asarray(policy.predict(obs, deterministic=True), dtype=np.float32)
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    act_t = policy.policy.stochastic_lane_det_accel(obs_t, rng)
                    action[1] = float(act_t.squeeze(0).cpu().numpy()[1])
            elif deterministic:
                action = policy_action(policy, obs, explore_std, rng)
            else:
                action = np.asarray(policy.predict(obs, deterministic=False), dtype=np.float32)
            if trace_path:
                with torch.inference_mode():
                    network = policy.policy
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    logits = (network.lane_logits(obs_tensor) if isinstance(policy, KLPPO)
                              else network.lane_head(network.trunk(obs_tensor)))
                    lane_probs = logits.softmax(dim=-1).squeeze(0).cpu().numpy()
                ego = env.ego_id
                stop_definition = env._current_stop()
                alive = ego in traci.vehicle.getIDList()
                lane = traci.vehicle.getLaneIndex(ego) if alive else -1
                neighbors = 0
                if alive:
                    ego_x, ego_y = traci.vehicle.getPosition(ego)
                    for actor, values in (traci.vehicle.getContextSubscriptionResults(ego) or {}).items():
                        if actor != ego:
                            actor_x, actor_y = values[constants.VAR_POSITION]
                            neighbors += int((ego_x - actor_x)**2 + (ego_y - actor_y)**2 <= 50**2)
                trace.append({
                    "step": len(trace) + 1, "time": traci.simulation.getTime(),
                    "edge": edge, "pos": np.nan if pos is None else pos, "lane": lane,
                    "x": ego_x if alive else np.nan, "y": ego_y if alive else np.nan,
                    "lane_count": traci.edge.getLaneNumber(edge) if edge and not edge.startswith(":") else -1,
                    "stop": stop, "stop_definition": json.dumps(stop_definition),
                    "bus_stop": obs[stop_obs_index:stop_obs_index + 3].copy(),
                    "accel_cmd": float(action[0]), "lane_raw": float(action[1]), "lane_probs": lane_probs,
                    "teacher_stop_lane": np.nan, "teacher_final_lane": np.nan, "lane_hold": False,
                    "speed": traci.vehicle.getSpeed(ego) if alive else 0.0,
                    "neighbors_50m": neighbors, "active_vehicles": traci.vehicle.getIDCount() - int(alive),
                })
            try:
                obs, _, terminated, truncated, info = env.step(action)
            except (traci.TraCIException, traci.FatalTraCIError):
                info = {"sumo_error": True}
                terminated = True
            info["ego_edge"] = last_edge
            if info.get("sumo_error"):
                if trace_path:
                    trace[-1].update(lane_cmd=0, lane_after=-1, lane_delta=0,
                                     shield=False, shield_lane=False, lc_forbidden=False,
                                     accel_applied=np.nan, dwell_seconds=0.0, stop_success=False,
                                     stops_done=0)
                break
            if trace_path:
                if teacher:
                    trace[-1].update(teacher_trace)
                post_edge, _, _ = _ego_place(env)
                post_lane = traci.vehicle.getLaneIndex(ego) if post_edge else -1
                trace[-1].update(
                    lane_cmd=int(info.get("lane_change") or 0), lane_after=post_lane,
                    accel_applied=float(info.get("accel_raw", action[0])),
                    dwell_seconds=float(info.get("dwell_time", 0.0)),
                    stop_success=bool(info.get("stop_success")),
                    lane_delta=post_lane - lane if edge == post_edge and lane >= 0 and post_lane >= 0 else 0,
                    shield=bool(info.get("shield_active")), shield_lane=bool(info.get("shield_lc_gate")),
                    lc_forbidden=bool(info.get("lc_forbidden")), stops_done=int(info.get("stops_done") or 0),
                )
            met.observe_step(env, info, odo_before)
            if info.get("speed") is not None:
                speed_sum += float(info["speed"])
                speed_n += 1
            shield_steps += int(bool(info.get("shield_active")))
            yield_steps += int(bool(info.get("yield_violation")))
            if (info.get("in_school_zone") and info.get("speed") is not None
                    and info.get("zone_limit") is not None):
                zone_steps += 1
                excess = float(info["speed"]) - float(info["zone_limit"])
                if zone_max_excess is None or excess > zone_max_excess:
                    zone_max_excess = excess
                if excess > 0.0:
                    zone_over_steps += 1
        missed_ids = [str(st["id"]) for st, status in zip(
            getattr(env, "_stops", []), getattr(env, "_stop_status", []))
            if int(status) < 0]
        stop_id = missed_ids[-1] if missed_ids else next_stop
        place = _closer_place(env, last_edge, last_pos, stop_id)
    finally:
        try:
            env.close()
        except (traci.TraCIException, traci.FatalTraCIError):
            if not info.get("sumo_error"):
                raise
    reason = end_reason(info, terminated, truncated)
    stop_id = missed_ids[-1] if missed_ids else next_stop
    met.finalize(env, info, reason, terminated, truncated)
    row = met.row_dict(
        int(seed), reason,
        int(info.get("stops_done", 0) or 0),
        int(info.get("stops_total", 0) or 0),
        place or stop_id,
        steps=int(speed_n),
        avg_speed=(speed_sum / speed_n) if speed_n else 0.0,
        shield_steps=int(shield_steps),
        yield_steps=int(yield_steps),
        ego_edge=last_edge or str(info.get("ego_edge") or ""),
        ego_lane_pos=None if last_pos is None else round(float(last_pos), 1),
        missed_stop_ids=missed_ids,
        nearest_stop_id=stop_id,
        bs02_lc_fail=int(stop_id == "bs_02" and reason in ("collision", "timeout")),
        zone_steps=int(zone_steps),
        zone_over_steps=int(zone_over_steps),
        zone_max_excess=zone_max_excess,
    )
    if trace_path:
        from pathlib import Path
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = {key: np.asarray([record[key] for record in trace]) for key in trace[0]}
        columns["end_reason"] = np.asarray(reason)
        temp_path = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp_path, **columns)
        temp_path.replace(path)
    row["evaluation_environment"] = {
        key: os.environ.get(key) for key in sorted(
            {"CGC_STAGE", "CGC_THROUGH_VPH", "OBS_SCAN", "STOP_FIX", "PLACE_FIX",
             "DISABLE_SAMIL_EAST", "SUMO_HOME", "HOTSPOT_LABELS", "HOTSPOT_LABEL_WEIGHT", "KLPPO_REF_COMMIT"}
            | {k for k in os.environ if k.startswith(("CGC_", "SUMO_", "HOTSPOT_", "KLPPO_"))})
    }
    row["evaluation_settings"] = dict(shield=shield, deterministic=deterministic,
        explore_std=explore_std, lane_temp=lane_temp, stochastic_lane=stochastic_lane,
        teacher=teacher, lane_hold=lane_hold, model=str(model))
    return row


def _zone_max_overspeed(rows) -> float:
    """보호구역 안에서 제한을 넘긴 폭의 최댓값 (m/s). 한 번도 안 넘기면 0."""
    excesses = [r["zone_max_excess"] for r in rows if r.get("zone_max_excess") is not None]
    if not excesses:
        return float("nan")
    return max(0.0, max(excesses))


def _zone_overspeed_step_rate(rows) -> float:
    """보호구역 스텝 중 제한을 넘긴 비율."""
    steps = sum(int(r.get("zone_steps", 0)) for r in rows)
    if not steps:
        return float("nan")
    return sum(int(r.get("zone_over_steps", 0)) for r in rows) / steps


def summarize(rows) -> dict:
    from utils.official_metrics import summarize_official
    base = summarize_official(rows)
    n = max(len(rows), 1)
    reasons = Counter(r["end_reason"] for r in rows)
    steps = sum(r.get("steps", r.get("sim_steps", 0)) for r in rows)
    base.update({
        "collision": base["collision_rate"],
        "red_light": reasons.get("red_light", 0) / n,
        "timeout": reasons.get("timeout", 0) / n,
        "stop_success_mean": float(np.mean([r["stops_done"] for r in rows])) if rows else 0.0,
        "yield_steps": int(sum(r.get("yield_steps", 0) for r in rows)),
        "intervention": sum(r.get("shield_steps", 0) for r in rows) / max(steps, 1),
        "avg_speed": float(np.mean([r.get("avg_speed", 0) for r in rows])) if rows else 0.0,
        "bs02_lc_fail": int(sum(r.get("bs02_lc_fail", 0) for r in rows)),
        "zone_max_overspeed": _zone_max_overspeed(rows),
        "zone_overspeed_step_rate": _zone_overspeed_step_rate(rows),
    })
    return base


def format_table(title, stats) -> str:
    n = stats["episodes"]

    def pct(key):
        return f"{stats[key] * n:.0f}/{n} ({stats[key]:.1%})"

    lines = [
        f"### {title}",
        "",
        "| 지표 | 값 |",
        "|---|---:|",
        f"| 공식 완주 | {pct('official_complete')} |",
        f"| 완벽 완주 | {pct('perfect_complete')} |",
        f"| 정차 성공 | {stats['stop_success_mean']:.2f} |",
        f"| 충돌 | {pct('collision')} |",
        f"| 적색 | {pct('red_light')} |",
        f"| 시간초과 | {pct('timeout')} |",
        f"| 양보 위반 스텝 | {stats['yield_steps']} |",
        f"| 수동개입률 | {stats['intervention']:.1%} |",
        f"| 평균 속도 | {stats['avg_speed']:.2f} m/s |",
        f"| 보호구역 최고 초과 속도 | {stats['zone_max_overspeed'] * 3.6:.2f} km/h |",
        f"| 보호구역 과속 스텝 비율 | {stats['zone_overspeed_step_rate']:.1%} |",
        f"| bs_02 차선변경 실패 | {stats['bs02_lc_fail']} |",
        "",
    ]
    return "\n".join(lines)


def prepare_sumo():
    from env import road_builder
    from env.road_config import EGO, ROAD, TRAFFIC
    work = tempfile.mkdtemp(prefix="cheonggye_drive_")
    return work, road_builder.build(ROAD, EGO, TRAFFIC, work)


def evaluate(model, seeds, shield="shadow", workers=1, deterministic=True, title="",
             json_out="", lane_temp=0.0, stochastic_lane=False):
    work, cfg = prepare_sumo()
    try:
        jobs = [(cfg, seed, model, shield, deterministic, 0.0, lane_temp, stochastic_lane)
                for seed in seeds]
        if workers <= 1 or len(seeds) == 1:
            rows = [run_seed(*job) for job in jobs]
        else:
            with get_context("spawn").Pool(workers) as pool:
                rows = pool.starmap(run_seed, jobs)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    stats = summarize(rows)
    print(format_table(title or model, stats))
    if json_out:
        os.makedirs(os.path.dirname(os.path.abspath(json_out)), exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump({"title": title or model, "stats": stats, "rows": rows},
                      f, ensure_ascii=False, indent=2)
    return rows, stats


def main():
    parser = argparse.ArgumentParser(description="병렬 주행 평가")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--shield", choices=["off", "shadow", "on"], default="shadow")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--stochastic", action="store_true",
                        help="가속에 학습 σ 노이즈. KL-PPO 시험 주행은 이 옵션 없이 평균값")
    parser.add_argument("--title", default="")
    parser.add_argument("--json-out", default="", help="판별 결과 JSON")
    parser.add_argument("--lane-temp", type=float, default=0.0,
                        help=">0이면 KL-PPO 차선만 τ-Categorical 샘플(가속은 결정적)")
    parser.add_argument("--stochastic-lane", action="store_true",
                        help="--lane-temp 와 함께 차선만 확률 샘플")
    args = parser.parse_args()
    stoch = args.stochastic_lane or args.lane_temp > 0
    evaluate(args.model, parse_seeds(args.seeds), args.shield, args.workers,
             deterministic=not args.stochastic, title=args.title, json_out=args.json_out,
             lane_temp=args.lane_temp, stochastic_lane=stoch)


if __name__ == "__main__":
    main()
