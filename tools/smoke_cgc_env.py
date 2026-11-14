"""청계천로 학습 환경 연결 점검 (PyTorch 없이).

train.py와 같은 설정으로 SumoHighwayEnv를 만들고, 모델 대신 단순 규칙
(앞이 비면 가속, 아니면 감속, 차선 유지)으로 몇 에피소드를 돌려
관측 차원, 경로상 횡단보도·신호 인식, 종료 사유를 확인한다.

사용: python tools/smoke_cgc_env.py [--episodes 3] [--max-steps 1500]
"""
import argparse
import ast
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env.road_config import ROAD, EGO, TRAFFIC  # noqa: E402
from env.mdp_config import SIMULATION, OBSERVATION, ACTION  # noqa: E402
from env.mdp_config import REWARD as REWARD_DEFAULTS  # noqa: E402
from env import road_builder  # noqa: E402
from env.sumo_env import SumoHighwayEnv  # noqa: E402


def reward_overrides():
    """train.py의 REWARD_OVERRIDES를 torch import 없이 읽는다."""
    tree = ast.parse(open(os.path.join(ROOT, "train.py"), encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "REWARD_OVERRIDES" for t in node.targets):
            v = node.value
            if isinstance(v, ast.Call) and getattr(v.func, "id", None) == "dict":
                return {kw.arg: ast.literal_eval(kw.value) for kw in v.keywords}
            return ast.literal_eval(v)
    return {}


def rule_accel(env):
    """점검용 규칙: 앞차·빨간불이 가까우면 감속, 아니면 완만히 가속 (학습 정책 아님)."""
    import traci
    ego = env.ego_id
    if ego not in traci.vehicle.getIDList():
        return 0.0
    v = traci.vehicle.getSpeed(ego)
    safe = 2.0 * v + 6.0
    lead = traci.vehicle.getLeader(ego, 80)
    if lead and lead[1] < safe:
        return -1.0
    for _, _, dist, state in traci.vehicle.getNextTLS(ego)[:1]:
        if state in "rRyY" and dist < v * v / 4.0 + 8.0:
            return -1.0
    return 0.3 if v < 8.0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=1500)
    a = ap.parse_args()

    cfg = road_builder.build(ROAD, EGO, TRAFFIC, os.path.join(ROOT, "env", "sumo"))
    reward = {**REWARD_DEFAULTS, **reward_overrides()}
    env = SumoHighwayEnv(cfg_path=cfg, road=ROAD, ego=EGO, mdp_sim=SIMULATION,
                         mdp_obs=OBSERVATION, action=ACTION, reward=reward,
                         traffic=TRAFFIC, gui=False)
    print(f"sumocfg {os.path.basename(cfg)} | 관측 {env.observation_space.shape[0]}차원 "
          f"| 행동 {env.action_space.shape[0]}차원")
    try:
        for ep in range(a.episodes):
            obs, _ = env.reset(seed=ep)
            print(f"[ep{ep}] 경로상 횡단보도 {len(env._route_crossings)}개, "
                  f"ego 경로 {env.ego_route}")
            total, info, bad = 0.0, {}, 0
            for t in range(1, a.max_steps + 1):
                if not np.all(np.isfinite(obs)):
                    bad += 1
                act = np.array([rule_accel(env), 0.0], dtype=np.float32)
                obs, r, term, trunc, info = env.step(act)
                total += r
                if term or trunc:
                    break
            keys = [k for k in ("collided", "arrived", "red_violation", "ped_collision",
                                "yield_violation", "missed_stops") if k in info]
            print(f"  steps={t} return={total:.1f} 종료={'term' if term else 'trunc' if trunc else 'max'} "
                  + " ".join(f"{k}={info[k]}" for k in keys)
                  + (f" | NaN 관측 {bad}" if bad else ""))
    finally:
        env.close()


if __name__ == "__main__":
    main()
