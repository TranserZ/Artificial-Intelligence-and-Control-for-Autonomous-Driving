"""env-v1 테스트 평가 (torch 없이). 사용: eval_v1.py <model.pt> <episodes> <seed0> <out.jsonl>"""
import sys, os, collections, json
import numpy as np
ROOT = os.path.expanduser("~/work/cheonggyecheon"); sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptload
from smoke_cgc_env import ROAD, EGO, TRAFFIC, SIMULATION, OBSERVATION, ACTION, REWARD_DEFAULTS, road_builder, SumoHighwayEnv, reward_overrides
class Pol:
    def __init__(s, p):
        d = ptload.load(p); s.W = [(d[f"trunk.{i}.weight"], d[f"trunk.{i}.bias"]) for i in (0, 2)]
        s.mu = (d["mu_head.weight"], d["mu_head.bias"]); s.ln = (d["lane_head.weight"], d["lane_head.bias"])
        s.dim = s.W[0][0].shape[1]
    def __call__(s, o):
        z = np.asarray(o, np.float32)
        for W, b in s.W: z = np.maximum(W @ z + b, 0)
        return np.array([np.tanh(s.mu[0] @ z + s.mu[1])[0], (1.0, 0.0, -1.0)[int(np.argmax(s.ln[0] @ z + s.ln[1]))]], np.float32)
path, n, seed0, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
pol = Pol(path)
cfg = os.environ["CGC_CFG"]
env = SumoHighwayEnv(cfg_path=cfg, road=ROAD, ego=EGO, mdp_sim=SIMULATION, mdp_obs=OBSERVATION, action=ACTION,
                     reward={**REWARD_DEFAULTS, **reward_overrides()}, traffic=TRAFFIC, gui=False)
assert env.obs_dim == pol.dim, (env.obs_dim, pol.dim)
env.shield_mode = "shadow"; env.eval_mode = True
K = ("shield_vehicle", "shield_signal", "shield_ped", "shield_lc_gate", "shield_active", "lc_forbidden")
E = ("red_violation", "yield_violation", "ped_collision")
done = {json.loads(l)["seed"] for l in open(out)} if os.path.exists(out) else set()
with open(out, "a") as f:
    for ep in range(n):
        if seed0 + ep in done: continue
        obs, _ = env.reset(seed=seed0 + ep); c = collections.Counter(); spd = []; ret = 0.0; lcs = 0
        for t in range(1, 100000):
            obs, r, te, tr, info = env.step(pol(obs)); ret += r
            for k in K + E: c[k] += bool(info.get(k))
            spd.append(info.get("speed") or 0.0); lcs += bool(info.get("lane_changed") or info.get("lane_change_applied"))
            if te or tr: break
        end = ("collided" if info.get("collided") else "arrived" if info.get("arrived") else
               "wrong_lane" if info.get("wrong_lane") else "timeout" if tr else "other")
        rec = dict(seed=seed0 + ep, end=end, steps=t, ret=ret, mean_speed=float(np.mean(spd)),
                   stops_done=info.get("stops_done"), missed=info.get("missed_stops"), stops_total=info.get("stops_total"),
                   pos_err=info.get("stop_pos_errors"), complete=bool(end == "arrived" and info.get("missed_stops", 1) == 0),
                   **{k: c[k] for k in K + E})
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(ep, end, t, rec["stops_done"], rec["missed"], f"{c['shield_active']/t:.1%}", flush=True)
env.close()
