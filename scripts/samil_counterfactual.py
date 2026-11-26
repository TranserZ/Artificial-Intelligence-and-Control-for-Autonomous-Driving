"""Paired GUI interventions on the Samil parking site. One process per episode."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pingpong_diagnose_s6 import STUDENT, TEACHER_CKPT, _make_env
from scripts.drive_parallel import prepare_sumo, policy_action, end_reason

OUT = ROOT / "results/samil_counterfactual"
EDGE = "e908646367_3"


def episode(model, seed, off):
    import torch
    os.environ.update(CGC_STAGE="6", OBS_SCAN="1", STOP_FIX="0", DISABLE_SAMIL_EAST=str(off))
    torch.set_num_threads(1)
    import traci
    from train import HPARAMS
    from algorithms.kl_ppo import load_eval_policy
    directory = OUT / f"{model}_{seed}_{'off' if off else 'on'}"
    directory.mkdir(parents=True, exist_ok=True)
    cfgdir, cfg = prepare_sumo()
    env = None
    try:
        if model == "teacher":
            env, module = _make_env(cfg, "on", teacher=True)
            agent = module.make_agent(env, 0)
            module.load_weights(agent, TEACHER_CKPT)
            rules = env.teacher_rules
            commands = {}
            def logged_rules(acc, lane):
                _, stop_cmd, _ = env._stop_rule(acc, lane)
                result = rules(acc, lane)
                commands.update(stop_cmd=float(stop_cmd), final_cmd=float(result[1]))
                return result
            env.teacher_rules = logged_rules
        else:
            env = _make_env(cfg, "shadow")
            checkpoint = STUDENT if model == "student" else ROOT / "results/klppo3_s6_recovery/step8k.pt"
            agent = load_eval_policy(str(checkpoint), 67, 2, HPARAMS)
            commands = {}
        env.gui = env.gui_autostart = env.eval_mode = True
        env.gui_delay = 0
        env.gui_track_ego = False
        obs, _ = env.reset(seed=seed)
        view = traci.gui.getIDList()[0]
        x, y = traci.simulation.convert2D(EDGE, 49.0, laneIndex=1)
        traci.gui.setOffset(view, x, y)
        traci.gui.setZoom(view, 2400)
        rows = []
        parked = {}
        stop_ok = False
        pings = 0
        last_change = -100
        previous_direction = 0
        longest = still = 0
        while True:
            t = traci.simulation.getTime()
            ego = env.ego_id
            edge = traci.vehicle.getRoadID(ego)
            pos = traci.vehicle.getLanePosition(ego)
            lane = traci.vehicle.getLaneIndex(ego)
            speed = traci.vehicle.getSpeed(ego)
            xy = traci.vehicle.getPosition(ego)
            st = env._current_stop()
            stop = st["id"] if st else ""
            area = edge == EDGE and 24 <= pos <= 77
            ids = traci.vehicle.getIDList()
            for vid in ids:
                if vid.startswith("hs_samil_east_") and vid not in parked:
                    parked[vid] = [traci.vehicle.getTypeID(vid), traci.vehicle.getLaneID(vid),
                                   traci.vehicle.getLanePosition(vid), list(traci.vehicle.getPosition(vid))]
            if area:
                traci.gui.screenshot(view, str(directory / f"{t:07.1f}.png"))
            still = still+1 if edge == EDGE and 52 <= pos <= 70 and speed < .1 else 0
            longest = max(longest, still)
            action = policy_action(agent, obs, 0, None)
            bso = env._stop_obs()
            obs, _, term, trunc, info = env.step(action)
            if stop == "bs_03" and info.get("stop_success"):
                stop_ok = True
            active = ego in traci.vehicle.getIDList()
            delta = traci.vehicle.getLaneIndex(ego)-lane if active and traci.vehicle.getRoadID(ego)==edge else 0
            if delta:
                if (stop=="bs_03" and edge==EDGE and 25 <= pos <= 50 and
                        delta*previous_direction < 0 and len(rows)-last_change<=10):
                    pings += 1
                last_change, previous_direction = len(rows), delta
            rows.append(dict(time=t,edge=edge,pos=pos,lane=lane,x=xy[0],y=xy[1],speed=speed,
                             stop=stop,lane_delta=delta,accel=float(action[0]),lane_raw=float(action[1]),
                             bus_stop=bso,gate=bool(info.get("shield_lc_gate")),**commands))
            if term or trunc:
                break
        np.savez_compressed(directory/"trace.npz", **{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
        summary=dict(model=model,seed=seed,off=off,parked=parked,bs03_ping=pings,
                     bs03_dwell=stop_ok,samil_stop_sec=longest*.5,reason=end_reason(info,term,trunc),
                     frames=len(list(directory.glob("*.png"))))
        (directory/"episode.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf8")
    finally:
        if env is not None:
            env.close()
        shutil.rmtree(cfgdir, ignore_errors=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--case", nargs=3)
    args=parser.parse_args()
    if args.case:
        model,seed,off=args.case
        episode(model,int(seed),int(off))
    else:
        for model,seed in (("student",900103),("recovery",900100),("teacher",900103),("teacher",900100)):
            for off in (0,1):
                directory=OUT/f"{model}_{seed}_{'off' if off else 'on'}"
                if not (directory/"episode.json").exists():
                    subprocess.run([sys.executable,__file__,"--case",model,str(seed),str(off)],check=True)
                record=json.loads((directory/"episode.json").read_text(encoding="utf8"))
                print(record,flush=True)

