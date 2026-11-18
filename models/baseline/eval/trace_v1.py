"""한 판 추적: trace_v1.py <model.pt> <seed>  (학습과 같은 보상으로)"""
import sys, os
ROOT = os.path.expanduser("~/work/cheonggyecheon"); sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, traci, ptload
from smoke_cgc_env import ROAD, EGO, TRAFFIC, SIMULATION, OBSERVATION, ACTION, SumoHighwayEnv
import ast
from smoke_cgc_env import REWARD_DEFAULTS
_t = ast.parse(open(os.path.join(ROOT, "train.py")).read())
_ov = next(n.value for n in _t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "REWARD_OVERRIDES")
REWARD = {**REWARD_DEFAULTS, **{k.arg: ast.literal_eval(k.value) for k in _ov.keywords}}
class Pol:
    def __init__(s, p):
        d = ptload.load(p); s.W = [(d[f"trunk.{i}.weight"], d[f"trunk.{i}.bias"]) for i in (0, 2)]
        s.mu = (d["mu_head.weight"], d["mu_head.bias"]); s.ln = (d["lane_head.weight"], d["lane_head.bias"])
    def __call__(s, o):
        z = np.asarray(o, np.float32)
        for W, b in s.W: z = np.maximum(W @ z + b, 0)
        return np.array([np.tanh(s.mu[0] @ z + s.mu[1])[0], (1.0, 0.0, -1.0)[int(np.argmax(s.ln[0] @ z + s.ln[1]))]], np.float32)
pol = Pol(sys.argv[1]); seed = int(sys.argv[2])
env = SumoHighwayEnv(cfg_path=os.environ["CGC_CFG"], road=ROAD, ego=EGO, mdp_sim=SIMULATION, mdp_obs=OBSERVATION, action=ACTION,
                     reward=REWARD, traffic=TRAFFIC, gui=False)
env.shield_mode = "shadow"; env.eval_mode = True
obs, _ = env.reset(seed=seed); ret = 0; acc = {}
for t in range(1, 100000):
    a = pol(obs); obs, r, te, tr, info = env.step(a); ret += r
    if t % 100 == 0 or te or tr or info.get("stop_success") or info.get("stop_missed"):
        ego = env.ego_id
        pos = (traci.vehicle.getRoadID(ego), round(traci.vehicle.getLanePosition(ego)), traci.vehicle.getLaneIndex(ego)) if ego in traci.vehicle.getIDList() else None
        print(t, pos, "v=%.1f" % (info.get("speed") or 0), "a=%.2f lc=%+.0f" % tuple(a), "r=%.3f ret=%.1f" % (r, ret),
              "stopArea=%s dwell=%.1f done=%s miss=%s lcf=%s" % (info.get("in_stop_area"), info.get("dwell_time") or 0, info.get("stops_done"), info.get("missed_stops"), info.get("lc_forbidden")), flush=True)
    if te or tr: break
env.close()
