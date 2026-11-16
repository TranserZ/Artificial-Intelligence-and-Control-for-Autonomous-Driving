"""학습된(민기) PPO 모델로 청계천로 편도 환경을 N 에피소드 평가하며
스텝별 궤적과 지점 통과 여부를 data_huiyoung/trajectories/ 에 저장한다.

사용법:
    py -3.13 scripts/collect_trajectories.py \
        --model results/run_20261108_114607/best_model.pt --episodes 100

저장 구조:
    data_huiyoung/trajectories/baseline_run_<timestamp>/
        episode_001.csv ... episode_100.csv   (스텝별: t,ego_x,ego_y,speed,accel_raw,
                                                lane_change,lane_change_applied,reward)
        summary.csv                           (에피소드별 종료 사유·종료 엣지·
                                                스텝 수·지점 통과 여부)
"""
import argparse
import csv
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env.road_config import ROAD, EGO, TRAFFIC  # noqa: E402
from env.mdp_config import SIMULATION, OBSERVATION, ACTION  # noqa: E402
from env import road_builder  # noqa: E402
from env.sumo_env import SumoHighwayEnv  # noqa: E402
import traci  # noqa: E402  (SUMO_HOME은 env.sumo_env import 시 sys.path에 추가됨)

from test import load_agent  # noqa: E402  (PPO/BC 체크포인트 자동 판별 로더)
from train import REWARD  # noqa: E402  (학습 때와 같은 보상 조건으로 평가)

from checkpoints import CheckpointTracker  # noqa: E402


def collect(model_path, episodes, out_dir, no_filter=False):
    sumocfg = road_builder.build(ROAD, EGO, TRAFFIC, os.path.join(ROOT, "env", "sumo"))
    env = SumoHighwayEnv(
        cfg_path=sumocfg, road=ROAD, ego=EGO,
        mdp_sim={**SIMULATION, "ped_filter": not no_filter}, mdp_obs=OBSERVATION,
        action=ACTION, reward=REWARD, traffic=TRAFFIC, gui=False)
    agent = load_agent(model_path, env.observation_space.shape[0],
                       env.action_space.shape[0], "auto")

    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []
    try:
        for ep in range(1, episodes + 1):
            obs, _ = env.reset(seed=ep)
            tracker = CheckpointTracker()
            rows = []
            last_edge, last_pos, last_lane = None, None, None
            reason = "timeout"
            ep_red, ep_yield = False, False
            t = 0
            while True:
                action = agent.predict(obs, deterministic=True)
                if env.ego_id in traci.vehicle.getIDList():
                    last_edge = traci.vehicle.getRoadID(env.ego_id)
                    last_pos = float(traci.vehicle.getLanePosition(env.ego_id))
                    last_lane = int(traci.vehicle.getLaneIndex(env.ego_id))
                    x, y = traci.vehicle.getPosition(env.ego_id)
                else:
                    x = y = None
                obs, reward, terminated, truncated, info = env.step(action)
                t += 1
                if info.get("speed") is not None:
                    dist = traci.vehicle.getDistance(env.ego_id) \
                        if env.ego_id in traci.vehicle.getIDList() else None
                    if dist is not None:
                        tracker.update(dist, float(info["speed"]))
                rows.append({
                    "t": t, "ego_x": x, "ego_y": y,
                    "speed": info.get("speed"),
                    "accel_raw": info.get("accel_raw"),
                    "lane_change": info.get("lane_change"),
                    "lane_change_applied": int(bool(info.get("lane_change_applied"))),
                    "reward": reward,
                })
                ep_red = ep_red or bool(info.get("red_violation"))
                ep_yield = ep_yield or bool(info.get("yield_violation"))
                if info.get("collided"):
                    reason = "collision"
                elif info.get("red_violation") and terminated:
                    reason = "red_violation"
                elif info.get("wrong_lane"):
                    reason = "wrong_lane"
                elif info.get("arrived"):
                    reason = "arrival"
                if terminated or truncated:
                    if not (info.get("collided") or info.get("arrived")
                           or info.get("wrong_lane")
                           or (info.get("red_violation") and terminated)):
                        reason = "timeout" if truncated else "unknown"
                    break

            ep_path = os.path.join(out_dir, f"episode_{ep:03d}.csv")
            with open(ep_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

            row = {"episode": ep, "steps": t, "reason": reason,
                  "arrived": int(reason == "arrival"), "collided": int(reason == "collision"),
                  "timeout": int(reason == "timeout"),
                  "red_violation_any": int(ep_red), "yield_violation_any": int(ep_yield),
                  "end_edge": last_edge, "end_lane_pos": last_pos, "end_lane": last_lane}
            row.update(tracker.as_row())
            summary_rows.append(row)
            print(f"[ep{ep:03d}] steps={t} 종료={reason} 종료엣지={last_edge} "
                 f"통과=({''.join(c for c in 'ABCDEFGHI' if tracker.passed.get(c))})")
    finally:
        env.close()

    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n저장 완료: {out_dir}  (에피소드 {len(summary_rows)}개, summary.csv 포함)")
    return summary_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(
        ROOT, "results", "run_20261108_114607", "best_model.pt"))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default=None,
                    help="생략 시 data_huiyoung/trajectories/baseline_run_<timestamp>/")
    ap.add_argument("--no-filter", action="store_true")
    a = ap.parse_args()

    out_dir = a.out or os.path.join(
        ROOT, "data_huiyoung", "trajectories",
        f"baseline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    collect(a.model, a.episodes, out_dir, a.no_filter)
