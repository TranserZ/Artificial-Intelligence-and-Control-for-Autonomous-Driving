"""규칙 기반 전문가 운전자 — HRL 하위 정책(crosswalk/schoolzone) BC 학습용 시범 데이터.

규칙 (단순화, 차선변경은 하지 않음):
    - 신호 적/황색이 가까우면: 정지
    - 횡단보도에 보행자가 있어(다가오는 중) 대기해야 하면: 정지
    - 어린이보호구역이고 제한속도(8.33 m/s) 이상이면: 감속
    - 그 외 앞차 간격 기준: >20m 가속 / <10m 감속 / 그 사이 유지
    - 앞차가 없으면: 가속

학습된 정책이 아니라 traci로 환경 내부 상태(선행차 간격, 신호, 보행자,
어린이보호구역 제한속도)를 직접 조회하는 스크립트 컨트롤러다. BC 전문가
시범 데이터를 만드는 용도이며, 그 자체를 평가 대상 정책으로도 쓸 수 있도록
evaluate_policy()와 같은 predict(obs, deterministic) 인터페이스를 맞춘다.

사용법 (100 에피소드 수집):
    py -3.13 scripts/rule_based_expert.py --episodes 100
"""
import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env.road_config import ROAD, EGO, TRAFFIC  # noqa: E402
from env.mdp_config import SIMULATION, OBSERVATION, ACTION  # noqa: E402
from env import road_builder  # noqa: E402
from env.sumo_env import SumoHighwayEnv  # noqa: E402
import traci  # noqa: E402

from train import REWARD  # noqa: E402


class RuleBasedExpert:
    """단순 규칙 기반 종방향 컨트롤러. 차선변경은 항상 0(유지)."""

    def __init__(self, env: SumoHighwayEnv):
        self.env = env

    def predict(self, obs, deterministic: bool = True) -> np.ndarray:
        env = self.env
        ego = env.ego_id
        if ego not in traci.vehicle.getIDList():
            return np.array([0.0, 0.0], dtype=np.float32)

        v = traci.vehicle.getSpeed(ego)
        edge = traci.vehicle.getRoadID(ego)

        # 1) 신호 정지: 적/황색이 가깝고 지금 속도로 접근 중이면 정지
        sig = env._next_signal()
        if sig is not None:
            _, dist, state, _ = sig
            level = env._SIGNAL_STOP_LEVEL.get(state, 1.0)
            if level > 0.0 and dist <= max(v * 2.0, 10.0):
                return np.array([-1.0, 0.0], dtype=np.float32)

        # 2) 횡단보도 보행자 대기: 다가오는 보행자가 있으면 정지
        if env._waiting_for_ped():
            return np.array([-1.0, 0.0], dtype=np.float32)

        # 3) 어린이보호구역: 제한속도 이상이면 감속
        school_limit = env._school_limits.get(edge)
        if school_limit is not None and v >= school_limit:
            return np.array([-0.5, 0.0], dtype=np.float32)

        # 4) 앞차 간격 규칙
        lead = traci.vehicle.getLeader(ego, env.visibility)
        if lead is None:
            accel = 1.0
        else:
            gap = lead[1]
            if gap > 20.0:
                accel = 1.0
            elif gap < 10.0:
                accel = -1.0
            else:
                accel = 0.0

        # 어린이보호구역 안에서 가속할 때는 제한속도를 넘지 않게 완만히
        if school_limit is not None and v < school_limit:
            accel = min(accel, 0.3)
        return np.array([accel, 0.0], dtype=np.float32)


def build_env(no_filter=False):
    sumocfg = road_builder.build(ROAD, EGO, TRAFFIC, os.path.join(ROOT, "env", "sumo"))
    return SumoHighwayEnv(
        cfg_path=sumocfg, road=ROAD, ego=EGO,
        mdp_sim={**SIMULATION, "ped_filter": not no_filter}, mdp_obs=OBSERVATION,
        action=ACTION, reward=REWARD, traffic=TRAFFIC, gui=False)


def collect(episodes, out_dir, no_filter=False):
    """BC 데이터 형식: 에피소드별 CSV(observation 40컬럼, action_accel,
    action_lane_change, reward)."""
    env = build_env(no_filter)
    expert = RuleBasedExpert(env)
    obs_dim = env.observation_space.shape[0]
    os.makedirs(out_dir, exist_ok=True)
    fieldnames = [f"obs_{i}" for i in range(obs_dim)] + [
        "action_accel", "action_lane_change", "reward"]

    n_collide = n_arrive = n_timeout = 0
    try:
        for ep in range(1, episodes + 1):
            obs, _ = env.reset(seed=ep)
            rows = []
            terminated = truncated = False
            while not (terminated or truncated):
                action = expert.predict(obs, deterministic=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                row = {f"obs_{i}": float(obs[i]) for i in range(obs_dim)}
                row["action_accel"] = float(action[0])
                row["action_lane_change"] = float(action[1])
                row["reward"] = float(reward)
                rows.append(row)
                obs = next_obs
            n_collide += int(bool(info.get("collided")))
            n_arrive += int(bool(info.get("arrived")))
            n_timeout += int(truncated and not info.get("collided") and not info.get("arrived"))

            path = os.path.join(out_dir, f"episode_{ep:03d}.csv")
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)
            print(f"[ep{ep:03d}] steps={len(rows)} "
                 f"종료={'collision' if info.get('collided') else 'arrival' if info.get('arrived') else 'timeout/기타'}")
    finally:
        env.close()

    print(f"\n규칙 기반 전문가 {episodes} 에피소드 수집 완료: {out_dir}")
    print(f"  충돌률={n_collide/episodes:.0%} 완주율={n_arrive/episodes:.0%} "
         f"시간초과={n_timeout/episodes:.0%}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default=None,
                    help="생략 시 data_huiyoung/bc_expert_data/rule_based_<timestamp>/")
    ap.add_argument("--no-filter", action="store_true")
    a = ap.parse_args()
    out_dir = a.out or os.path.join(
        ROOT, "data_huiyoung", "bc_expert_data",
        f"rule_based_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    collect(a.episodes, out_dir, a.no_filter)
