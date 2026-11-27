"""단일 시드 GUI 주행 — 충돌 장면 확인용.

    $env:CGC_STAGE=6; $env:OBS_SCAN=1
    py -3.13 evaluation/gui_drive_seed.py --seed 900114
    py -3.13 evaluation/gui_drive_seed.py --seed 900114 --model student
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.drive_parallel import end_reason, policy_action, prepare_sumo  # noqa: E402

KL_200K = os.path.join(ROOT, "results", "klppo1_s6_s1", "step200k.pt")
STUDENT = os.path.join(os.path.dirname(ROOT), "청계로드-dongyeon-d54a761", "models", "dongyeon_bc", "s6_student_obsfix.pt")


def main():
    parser = argparse.ArgumentParser(description="SUMO-GUI 단일 판 재생")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", choices=["klppo", "student"], default="klppo")
    parser.add_argument("--shield", choices=["shadow", "on", "off"], default="shadow")
    parser.add_argument("--delay", type=float, default=150.0, help="sumo-gui delay ms (클수록 느림)")
    parser.add_argument("--pause", action="store_true", help="충돌/종료 후 GUI 일시정지 + Enter 대기")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--screenshots", type=Path)
    parser.add_argument("--from-time", type=float, default=0.0)
    parser.add_argument("--to-time", type=float, default=float("inf"))
    parser.add_argument("--stage", type=int, choices=(5, 6), default=6)
    args = parser.parse_args()
    if args.screenshots:
        args.screenshots.mkdir(parents=True, exist_ok=True)

    os.environ["CGC_STAGE"] = str(args.stage)
    os.environ["OBS_SCAN"] = "1"

    from environment.mdp_config import ACTION, OBSERVATION, SIMULATION
    from environment.road_config import EGO, ROAD, TRAFFIC
    from environment.sumo_env import SumoHighwayEnv
    from training.train import HPARAMS, REWARD
    from algorithms.kl_ppo import load_eval_policy
    import traci

    model_path = str(args.checkpoint) if args.checkpoint else (STUDENT if args.model == "student" else KL_200K)
    sim = {
        **SIMULATION,
        "shield": {**SIMULATION["shield"], "mode": args.shield},
        "gui_delay": float(args.delay),
    }

    work, cfg = prepare_sumo()
    env = None
    try:
        env = SumoHighwayEnv(
            cfg_path=cfg, road=ROAD, ego=EGO, mdp_sim=sim, mdp_obs=OBSERVATION,
            action=ACTION, reward=REWARD, traffic=TRAFFIC, gui=True,
        )
        policy = load_eval_policy(model_path, 67, 2, HPARAMS)
        obs, _ = env.reset(seed=int(args.seed))
        views = traci.gui.getIDList()
        view = views[0]
        traci.gui.trackVehicle(view, env.ego_id)
        traci.gui.setZoom(view, 1100)
        terminated = truncated = False
        info = {}
        step = 0
        title = f"seed={args.seed} model={args.model} shield={args.shield}"
        print(title, flush=True)
        while not (terminated or truncated):
            now = traci.simulation.getTime()
            if args.screenshots and now > args.to_time:
                break
            if args.screenshots and args.from_time <= now <= args.to_time:
                traci.gui.screenshot(view, str((args.screenshots / f"{now:.1f}s.png").resolve()))
            action = policy_action(policy, obs, 0.0, None)
            obs, _, terminated, truncated, info = env.step(action)
            step += 1
            if info.get("collided"):
                break
        reason = end_reason(info, terminated, truncated) if terminated or truncated else "capture_done"
        edge, pos = "", ""
        if env.ego_id in traci.vehicle.getIDList():
            edge = traci.vehicle.getRoadID(env.ego_id)
            pos = traci.vehicle.getLanePosition(env.ego_id)
        print(
            f"== {title} 종료: {reason} step={step} "
            f"stops={info.get('stops_done')}/{info.get('stops_total')} "
            f"edge={edge} pos={pos}",
            flush=True,
        )
        if args.pause:
            try:
                traci.gui.setStopOff()
            except Exception:
                pass
            try:
                traci.simulation.pause()
            except Exception:
                pass
            print("sumo-gui에서 장면 확인 후 이 창에서 Enter → 창 닫기", flush=True)
            input()
    finally:
        if env is not None:
            env.close()
        import shutil
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()