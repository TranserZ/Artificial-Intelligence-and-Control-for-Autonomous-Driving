"""CGC_STAGE=1·6, 시드 3000–3029 IDM 진단. 학습 데이터로 저장하지 않는다.

  py -3.13 tools/diag/run_idm.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

SEEDS = list(range(3000, 3030))
STAGES = (1, 6)
OUT = os.path.join(ROOT, "results", "idm_diag")


def _collision_partner(env):
    import traci
    ego = env.ego_id
    try:
        ids = [vid for vid in traci.simulation.getCollidingVehiclesIDList() if vid != ego]
    except traci.TraCIException:
        ids = []
    out = []
    for vid in ids:
        row = {"id": vid}
        try:
            row["type"] = traci.vehicle.getTypeID(vid)
        except traci.TraCIException:
            row["type"] = None
        try:
            row["vclass"] = traci.vehicle.getVehicleClass(vid)
        except traci.TraCIException:
            row["vclass"] = None
        try:
            row["speed"] = float(traci.vehicle.getSpeed(vid))
        except traci.TraCIException:
            row["speed"] = None
        out.append(row)
    return out


def run_stage(stage: int):
    os.environ["CGC_STAGE"] = str(stage)
    # road_config는 import 시점에 단계를 읽으므로, 단계마다 프로세스를 나누거나 재import.
    for name in list(sys.modules):
        if name == "env.road_config" or name.startswith("env.road_config"):
            del sys.modules[name]
        if name == "env.curriculum" or name.startswith("env.curriculum"):
            del sys.modules[name]

    import traci  # noqa: WPS433
    from env.road_config import ROAD, EGO, TRAFFIC  # noqa: WPS433
    from env.mdp_config import SIMULATION, OBSERVATION, ACTION  # noqa: WPS433
    from env import road_builder  # noqa: WPS433
    from env.sumo_env import SumoHighwayEnv  # noqa: WPS433
    from train import REWARD  # noqa: WPS433
    from tools.diag.idm_follow import IdmKeepLaneStop as Policy  # noqa: WPS433

    globals()["traci"] = traci

    work = tempfile.TemporaryDirectory(prefix=f"idm_diag_s{stage}_")
    cfg = road_builder.build(ROAD, EGO, TRAFFIC, work.name)
    env = SumoHighwayEnv(
        cfg_path=cfg, road=ROAD, ego=EGO,
        mdp_sim={**SIMULATION, "shield": {**SIMULATION["shield"], "mode": "off"}},
        mdp_obs=OBSERVATION, action=ACTION, reward=REWARD, traffic=TRAFFIC, gui=False)
    policy = Policy(env)
    rows = []
    try:
        for seed in SEEDS:
            obs, _ = env.reset(seed=seed)
            done = False
            stops_ok = 0
            last_dist = 0.0
            last_speed = 0.0
            steps = 0
            ended = "timeout"
            partner = []
            while not done:
                action = policy.predict(obs)
                obs, _rew, term, trunc, info = env.step(action)
                steps += 1
                if info.get("speed") is not None:
                    last_speed = float(info["speed"])
                try:
                    last_dist = float(traci.vehicle.getDistance(env.ego_id))
                except Exception:
                    pass
                stops_ok += int(info.get("stop_success") or 0)
                if info.get("collided"):
                    ended = "collision"
                    partner = _collision_partner(env)
                elif info.get("red_violation") and term:
                    ended = "red"
                elif info.get("wrong_lane"):
                    ended = "wrong_lane"
                elif info.get("arrived"):
                    ended = "arrival"
                done = term or trunc
            missed = int(info.get("missed_stops") or 0)
            official = ended == "arrival" and missed == 0
            rows.append({
                "stage": stage,
                "seed": seed,
                "ended": ended,
                "official_complete": official,
                "collision": ended == "collision",
                "timeout": ended == "timeout",
                "arrived": ended == "arrival",
                "distance_m": round(last_dist, 2),
                "last_speed_mps": round(last_speed, 3),
                "steps": steps,
                "stop_success": stops_ok,
                "missed_stops": missed,
                "collision_partners": partner,
                "road_stage": ROAD.get("stage"),
                "elements": list(ROAD.get("elements") or []),
            })
            print(f"s{stage} seed={seed} {ended} dist={last_dist:.1f} "
                  f"stops={stops_ok} miss={missed} official={int(official)}",
                  flush=True)
    finally:
        env.close()
        work.cleanup()
    return rows


def summarize(rows):
    n = len(rows)
    c = Counter(r["ended"] for r in rows)
    coll = [r for r in rows if r["collision"]]
    timeouts = [r for r in rows if r["timeout"]]
    partners = Counter()
    for r in coll:
        if not r["collision_partners"]:
            partners["(unknown)"] += 1
        for p in r["collision_partners"]:
            partners[p.get("type") or p.get("id") or "?"] += 1
    return {
        "n": n,
        "official_complete": sum(r["official_complete"] for r in rows) / n,
        "collision": c["collision"] / n,
        "timeout": c["timeout"] / n,
        "arrival": c["arrival"] / n,
        "ended": dict(c),
        "mean_stop_success": sum(r["stop_success"] for r in rows) / n,
        "mean_missed_stops": sum(r["missed_stops"] for r in rows) / n,
        "collision_distance_m": [r["distance_m"] for r in coll],
        "collision_partners": dict(partners),
        "timeout_distance_m": [r["distance_m"] for r in timeouts],
        "timeout_last_speed_mps": [r["last_speed_mps"] for r in timeouts],
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    all_rows = []
    summaries = {}
    for stage in STAGES:
        rows = run_stage(stage)
        all_rows.extend(rows)
        summaries[str(stage)] = summarize(rows)
        path = os.path.join(OUT, f"stage{stage}_episodes.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print("wrote", path)
    payload = {
        "policy": "IDM keep-lane + 10s bus stop (not training data)",
        "seeds": SEEDS,
        "held_out_not_used": True,
        "summary": summaries,
    }
    summary_path = os.path.join(OUT, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print("summary", summary_path)


if __name__ == "__main__":
    main()
