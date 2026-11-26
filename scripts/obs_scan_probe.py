"""6단계 한 판에서, 버스가 서 있는 동안 옆 차로를 놓친 차를 센다.

동연님 obsfix.py 주석의 probe와 같다. 속도가 정차 기준 이하인 스텝에서
±1 차선 스캔의 가장 가까운 앞/뒤를 관측(_neighbor)과 비교한다.
스캔 gap이 음수면 0으로 두고, visibility 안인데 관측이 없거나 더 멀면 놓친 차다.
OBS_SCAN=1이면 놓친 차가 0이어야 한다.

    $env:CGC_STAGE = "6"
    $env:OBS_SCAN = "1"
    py -3.13 scripts/obs_scan_probe.py --model <s6_student_obsfix.pt> --seed 900100
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.drive_parallel import policy_action, prepare_sumo, _make_env  # noqa: E402


def probe_episode(model, seed, shield):
    import torch
    from algorithms.kl_ppo import load_eval_policy
    from train import HPARAMS

    torch.set_num_threads(1)
    _work, cfg = prepare_sumo()
    env = _make_env(cfg, shield)
    policy = load_eval_policy(
        model, env.observation_space.shape[0], env.action_space.shape[0], HPARAMS)
    obs_dim = int(env.observation_space.shape[0])
    stopped = scan_cars = missed = base_missed = 0
    try:
        obs, _ = env.reset(seed=int(seed))
        terminated = truncated = False
        while not (terminated or truncated):
            if env.ego_id in __import__("traci").vehicle.getIDList():
                speed = float(__import__("traci").vehicle.getSpeed(env.ego_id))
                if speed <= float(env.stop_speed):
                    stopped += 1
                    for offset in (-1, 1):
                        front, rear = env._scan_lane_neighbors(offset)
                        for leader, scanned in ((True, front), (False, rear)):
                            if scanned is None:
                                continue
                            gap = max(float(scanned[0]), 0.0)
                            if gap > env.visibility:
                                continue
                            scan_cars += 1
                            seen = env._neighbor(offset, leader=leader)
                            if seen is None or float(seen[0]) > gap + 1e-4:
                                missed += 1
                            base = env._neighbor_base(offset, leader=leader)
                            if base is None or float(base[0]) > gap + 1e-4:
                                base_missed += 1
            action = policy_action(policy, obs)
            obs, _, terminated, truncated, _info = env.step(action)
    finally:
        env.close()
    return {
        "obs_dim": obs_dim,
        "stopped": stopped,
        "scan_cars": scan_cars,
        "missed": missed,
        "base_missed": base_missed,
        "seed": int(seed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=900100)
    parser.add_argument("--shield", default="shadow")
    args = parser.parse_args()
    if os.environ.get("OBS_SCAN", "0") != "1":
        raise SystemExit("probe는 OBS_SCAN=1 에서 돌린다.")
    if os.environ.get("CGC_STAGE") != "6":
        raise SystemExit("probe는 CGC_STAGE=6 에서 돌린다.")
    result = probe_episode(args.model, args.seed, args.shield)
    print(
        f"obs_dim={result['obs_dim']} seed={result['seed']} "
        f"stopped={result['stopped']} scan_cars={result['scan_cars']} "
        f"missed={result['missed']} base_missed={result['base_missed']}",
        flush=True,
    )
    if result["obs_dim"] != 67 or result["stopped"] <= 0 or result["missed"] != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
