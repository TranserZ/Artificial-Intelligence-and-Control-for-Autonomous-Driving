"""여러 정책(민기 PPO, 규칙 기반 전문가, 추후 HRL 하위 정책)을 같은 편도
환경에서 N 에피소드씩 평가하고 비교 표(Markdown)를 만든다.

사용법:
    py -3.13 scripts/evaluate_all.py --episodes 100
    py -3.13 scripts/evaluate_all.py --episodes 20 --models minki_ppo rule_based
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.road_config import ROAD, EGO, TRAFFIC  # noqa: E402
from env.mdp_config import SIMULATION, OBSERVATION, ACTION  # noqa: E402
from env import road_builder  # noqa: E402
from env.sumo_env import SumoHighwayEnv  # noqa: E402
import traci  # noqa: E402

from train import REWARD  # noqa: E402
from test import load_agent  # noqa: E402

from checkpoints import CheckpointTracker, ALL_CODES  # noqa: E402
from rule_based_expert import RuleBasedExpert  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 비교할 모델 목록 — name -> (kind, path).  kind: "ppo"/"bc" 체크포인트 또는 "rule"
MODEL_REGISTRY = {
    "minki_ppo": ("ppo", os.path.join(ROOT, "results", "run_20261108_114607", "best_model.pt")),
    "rule_based": ("rule", None),
}


def build_env():
    sumocfg = road_builder.build(ROAD, EGO, TRAFFIC, os.path.join(ROOT, "env", "sumo"))
    return SumoHighwayEnv(
        cfg_path=sumocfg, road=ROAD, ego=EGO,
        mdp_sim=SIMULATION, mdp_obs=OBSERVATION,
        action=ACTION, reward=REWARD, traffic=TRAFFIC, gui=False)


def run_episodes(policy, env, n_episodes):
    n_collide = n_arrive = n_timeout = n_filter_steps = n_steps = 0
    speeds, lane_changes = [], []
    pass_counts = {c: 0 for c in ALL_CODES}
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=ep)
        tracker = CheckpointTracker()
        terminated = truncated = False
        ep_lc = 0
        while not (terminated or truncated):
            action = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            n_steps += 1
            if info.get("speed") is not None:
                speeds.append(float(info["speed"]))
            if info.get("lane_change", 0) != 0 and info.get("lane_change_applied"):
                ep_lc += 1
            n_filter_steps += int(bool(info.get("filter_active")))
            if env.ego_id in traci.vehicle.getIDList():
                dist = traci.vehicle.getDistance(env.ego_id)
                tracker.update(dist, info.get("speed") or 0.0)
        lane_changes.append(ep_lc)
        if info.get("collided"):
            n_collide += 1
        elif info.get("arrived"):
            n_arrive += 1
        else:
            n_timeout += 1
        for c in ALL_CODES:
            pass_counts[c] += tracker.passed.get(c, False)

    return {
        "n_episodes": n_episodes,
        "success_rate": n_arrive / n_episodes,
        "collision_rate": n_collide / n_episodes,
        "timeout_rate": n_timeout / n_episodes,
        "mean_speed": sum(speeds) / len(speeds) if speeds else 0.0,
        "filter_rate": n_filter_steps / max(n_steps, 1),
        "lane_changes": sum(lane_changes) / n_episodes,
        "checkpoint_pass_rate": {c: pass_counts[c] / n_episodes for c in ALL_CODES},
    }


def make_policy(kind, path, env):
    if kind == "rule":
        return RuleBasedExpert(env)
    return load_agent(path, env.observation_space.shape[0], env.action_space.shape[0], "auto")


def to_markdown(results: dict) -> str:
    lines = [
        "# 청계천로 편도 — 모델 비교 평가",
        "",
        f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| 모델 | 에피소드 | 완주율 | 충돌률 | 시간초과율 | 평균속도(m/s) | 수동개입률 | 차선변경/에피소드 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['n_episodes']} | {r['success_rate']:.0%} | "
            f"{r['collision_rate']:.0%} | {r['timeout_rate']:.0%} | "
            f"{r['mean_speed']:.2f} | {r['filter_rate']:.1%} | {r['lane_changes']:.1f} |")
    lines += ["", "## 지점별 통과율", "", "| 모델 | " + " | ".join(ALL_CODES) + " |",
             "|---|" + "---|" * len(ALL_CODES)]
    for name, r in results.items():
        cp = r["checkpoint_pass_rate"]
        lines.append(f"| {name} | " + " | ".join(f"{cp[c]:.0%}" for c in ALL_CODES) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--models", nargs="*", default=list(MODEL_REGISTRY.keys()),
                    help="MODEL_REGISTRY 키 중 평가할 것만 지정 (기본: 전부)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    env = build_env()
    results = {}
    try:
        for name in a.models:
            kind, path = MODEL_REGISTRY[name]
            print(f"=== {name} ({kind}) {a.episodes} 에피소드 평가 중 ===")
            policy = make_policy(kind, path, env)
            results[name] = run_episodes(policy, env, a.episodes)
            print(f"  완주율={results[name]['success_rate']:.0%} "
                 f"충돌률={results[name]['collision_rate']:.0%}")
    finally:
        env.close()

    out = a.out or os.path.join(
        ROOT, "data_huiyoung", "eval_logs",
        f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    md = to_markdown(results)
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n저장: {out}")
    print(md)
