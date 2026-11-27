"""최종 모델(KL-PPO 체크포인트) 주행을 sumo-gui 화면으로 찍는다. 평가(scripts/drive_parallel.py)와 같은 환경·정책.

    CGC_STAGE=5 OBS_SCAN=0 python render_kl.py --repo ~/work/cgc-klrender \
        --ckpt results/klppo1_s5_s0/step250k.pt --seed 900377 --out out/k_900377 [--nogui] [--max-steps N] [--zoom Z]
"""
import argparse, json, os, sys, tempfile
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True); p.add_argument("--ckpt", required=True)
p.add_argument("--seed", type=int, required=True); p.add_argument("--out", required=True)
p.add_argument("--nogui", action="store_true"); p.add_argument("--max-steps", type=int, default=0)
p.add_argument("--zoom", type=float, default=0.0)
p.add_argument("--width", type=int, default=1280); p.add_argument("--height", type=int, default=720)
p.add_argument("--from-step", type=int, default=0, help="이 스텝부터 화면을 저장")
args = p.parse_args()

ROOT = os.path.abspath(os.path.expanduser(args.repo)); OUT = os.path.abspath(args.out)
os.makedirs(OUT, exist_ok=True); os.chdir(ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torch
torch.set_num_threads(1)
from env.mdp_config import ACTION, OBSERVATION, SIMULATION
from env.road_config import EGO, ROAD, TRAFFIC
from env import road_builder
from env.sumo_env import SumoHighwayEnv
from train import REWARD, HPARAMS
from algorithms.kl_ppo import load_eval_policy

work = tempfile.TemporaryDirectory(prefix="cgc_render_")
cfg = road_builder.build(ROAD, EGO, TRAFFIC, work.name)
sim = {**SIMULATION, "shield": {**SIMULATION["shield"], "mode": "shadow"}, "gui_delay": 0.0}
if args.zoom:
    sim["gui_zoom"] = args.zoom
env = SumoHighwayEnv(cfg_path=cfg, road=ROAD, ego=EGO, mdp_sim=sim, mdp_obs=OBSERVATION,
                     action=ACTION, reward=REWARD, traffic=TRAFFIC, gui=not args.nogui)
policy = load_eval_policy(args.ckpt, env.observation_space.shape[0], env.action_space.shape[0], HPARAMS)
import traci

KEEP = ("stops_done", "missed_stops", "collided", "arrived", "red_violation", "wrong_lane", "shield_active")
log = []; end = "?"
try:
    obs, _ = env.reset(seed=args.seed)
    t = 0
    while True:
        act = np.asarray(policy.predict(obs, deterministic=True), dtype=np.float32)
        if not args.nogui and t >= args.from_step:
            if args.zoom:
                try:
                    traci.gui.trackVehicle("View #0", env.ego_id); traci.gui.setZoom("View #0", args.zoom)
                except Exception:
                    pass
            traci.gui.screenshot("View #0", os.path.join(OUT, f"f_{t + 1:05d}.png"), args.width, args.height)
        obs, r, term, trunc, info = env.step(act)
        t += 1
        row = {"t": t, "a": round(float(act[0]), 3), "lc": int(round(float(act[1])))}
        try:
            ego = env.ego_id
            x, y = traci.vehicle.getPosition(ego)
            row.update(x=round(x, 1), y=round(y, 1), v=round(traci.vehicle.getSpeed(ego), 2),
                       lane=traci.vehicle.getLaneIndex(ego), edge=traci.vehicle.getRoadID(ego),
                       odo=round(traci.vehicle.getDistance(ego), 1))
        except Exception:
            pass
        for k in KEEP:
            if k in info:
                v = info[k]
                row[k] = (round(float(v), 2) if isinstance(v, (float, np.floating)) else
                          int(v) if isinstance(v, (bool, np.bool_, int, np.integer)) else v)
        log.append(row)
        if term or trunc or (args.max_steps and t >= args.max_steps):
            end = ("충돌" if info.get("collided") else "도착" if info.get("arrived") else
                   "적색 위반" if info.get("red_violation") else "차로 오류" if info.get("wrong_lane")
                   else "시간 초과" if trunc else "중단")
            break
    print(f"판 {args.seed}: {end}, {t * 0.5:.0f}초, 정차 성공 {info.get('stops_done')} / 놓침 {info.get('missed_stops')}", flush=True)
finally:
    json.dump({"seed": args.seed, "end": end, "step_length": 0.5, "ckpt": args.ckpt, "steps": log},
              open(os.path.join(OUT, "log.json"), "w"), ensure_ascii=False)
    try: env.close()
    except Exception: pass
    work.cleanup()
