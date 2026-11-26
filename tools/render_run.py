"""학생 모델 주행을 sumo-gui 화면으로 찍는다 (torch 없이 numpy로 계산).

    CGC_STAGE=6 OBS_SCAN=1 python render_run.py --repo ~/work/cgc-s56 \
        --ckpt models/dongyeon_bc/s6_student_v5.pt --seed 1000 --out out/s6_1000 [--nogui] [--max-steps N]

프레임: <out>/f_00001.png ...  로그: <out>/log.json (스텝별 위치·속도·사건)
"""
import argparse
import ast
import io
import json
import os
import pickle
import sys
import tempfile
import zipfile

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True)
p.add_argument("--ckpt", required=True)
p.add_argument("--seed", type=int, default=1000)
p.add_argument("--out", required=True)
p.add_argument("--nogui", action="store_true")
p.add_argument("--max-steps", type=int, default=0)
p.add_argument("--zoom", type=float, default=0.0)
p.add_argument("--width", type=int, default=1280)
p.add_argument("--height", type=int, default=720)
p.add_argument("--every", type=int, default=1, help="몇 스텝마다 한 장")
p.add_argument("--near", action="store_true", help="주변 오토바이·합류차 수 기록")
args = p.parse_args()

ROOT = os.path.abspath(os.path.expanduser(args.repo))
OUT = os.path.abspath(args.out)
os.makedirs(OUT, exist_ok=True)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


# ── .pt 를 torch 없이 읽기 ────────────────────────────────────
def load_state_dict(path):
    zf = zipfile.ZipFile(path)
    pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
    base = pkl[: -len("data.pkl")]

    class Storage:
        def __init__(self, dtype):
            self.dtype = dtype

    def rebuild(storage, offset, size, stride, *_):
        arr = storage
        n = int(np.prod(size)) if len(size) else 1
        return arr[offset: offset + n].reshape(size).copy()

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "torch._utils" and name == "_rebuild_tensor_v2":
                return rebuild
            if mod == "torch" and name.endswith("Storage"):
                return Storage({"FloatStorage": np.float32, "DoubleStorage": np.float64,
                                "LongStorage": np.int64}[name])
            if mod == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            raise pickle.UnpicklingError(f"{mod}.{name}")

        def persistent_load(self, pid):
            _, storage, key, _, _ = pid
            return np.frombuffer(zf.read(f"{base}data/{key}"), dtype=storage.dtype)

    return U(io.BytesIO(zf.read(pkl))).load()


class NumpyHybrid:
    """HybridActorCritic(67→256→256 ReLU, 가감속 μ + 차선 3-way) 결정적 행동."""
    LANE_CMDS = (1.0, 0.0, -1.0)

    def __init__(self, path):
        sd = load_state_dict(path)
        self.trunk = []
        i = 0
        while f"trunk.{i}.weight" in sd or f"trunk.{i + 1}.weight" in sd:
            if f"trunk.{i}.weight" in sd:
                self.trunk.append((sd[f"trunk.{i}.weight"], sd[f"trunk.{i}.bias"]))
            i += 1
        self.mu = (sd["mu_head.weight"], sd["mu_head.bias"])
        self.lane = (sd["lane_head.weight"], sd["lane_head.bias"])
        self.obs_dim = self.trunk[0][0].shape[1]

    def predict(self, obs):
        z = np.asarray(obs, dtype=np.float32)
        for w, b in self.trunk:
            z = np.maximum(w @ z + b, 0.0)
        mu = np.tanh(self.mu[0] @ z + self.mu[1])
        logits = self.lane[0] @ z + self.lane[1]
        return np.array([float(mu[0]), self.LANE_CMDS[int(np.argmax(logits))]], dtype=np.float32)


from env.road_config import ROAD, EGO, TRAFFIC                 # noqa: E402
from env.mdp_config import SIMULATION, OBSERVATION, ACTION     # noqa: E402
from env.mdp_config import REWARD as REWARD_DEFAULTS           # noqa: E402
from env import road_builder                                   # noqa: E402
from env.sumo_env import SumoHighwayEnv                        # noqa: E402

if os.environ.get("OBS_SCAN", "0") == "1":
    sys.path.insert(0, os.path.join(ROOT, "models", "dongyeon_bc"))
    import obsfix
    obsfix.apply()


def train_reward():
    tree = ast.parse(open(os.path.join(ROOT, "train.py"), encoding="utf-8").read())
    node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "REWARD_OVERRIDES")
    return {**REWARD_DEFAULTS, **{k.arg: ast.literal_eval(k.value) for k in node.keywords}}


policy = NumpyHybrid(os.path.join(ROOT, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt)
work = tempfile.TemporaryDirectory(prefix="cgc_render_")
cfg = road_builder.build(ROAD, EGO, TRAFFIC, work.name)
sim = {**SIMULATION, "shield": {**SIMULATION["shield"], "mode": "shadow"}, "gui_delay": 0.0}
if args.zoom:
    sim["gui_zoom"] = args.zoom
env = SumoHighwayEnv(cfg_path=cfg, road=ROAD, ego=EGO, mdp_sim=sim, mdp_obs=OBSERVATION,
                     action=ACTION, reward=train_reward(), traffic=TRAFFIC, gui=not args.nogui)
env.eval_mode, env.shield_mode = True, "shadow"
assert policy.obs_dim == env.obs_dim, (policy.obs_dim, env.obs_dim)

import traci                                                   # noqa: E402

KEEP = ("stops_done", "missed_stops", "collided", "arrived", "red_violation", "wrong_lane",
        "shield_active", "in_stop_area", "dwell_time")
log = []
end = "?"
try:
    obs, _ = env.reset(seed=args.seed)
    t = 0
    while True:
        act = policy.predict(obs)
        if not args.nogui and t % args.every == 0:
            if args.zoom:
                try:
                    traci.gui.trackVehicle("View #0", env.ego_id)
                    traci.gui.setZoom("View #0", args.zoom)
                except Exception:
                    pass
            traci.gui.screenshot("View #0", os.path.join(OUT, f"f_{t // args.every + 1:05d}.png"),
                                 args.width, args.height)
        obs, r, term, trunc, info = env.step(act)
        t += 1
        row = {"t": t, "a": round(float(act[0]), 3), "lc": int(act[1])}
        try:
            ego = env.ego_id
            x, y = traci.vehicle.getPosition(ego)
            row.update(x=round(x, 1), y=round(y, 1), v=round(traci.vehicle.getSpeed(ego), 2),
                       lane=traci.vehicle.getLaneIndex(ego), edge=traci.vehicle.getRoadID(ego),
                       odo=round(traci.vehicle.getDistance(ego), 1))
            if args.near:
                mo = cu = 0
                for vid in traci.vehicle.getIDList():
                    if vid == ego:
                        continue
                    ty = traci.vehicle.getTypeID(vid)
                    if ty not in ("motorcycle", "cutin_car"):
                        continue
                    px, py = traci.vehicle.getPosition(vid)
                    if (px - x) ** 2 + (py - y) ** 2 < 900 and traci.vehicle.getSpeed(vid) > 1.0:
                        if ty == "motorcycle":
                            mo += 1
                        else:
                            cu += 1
                row.update(mo=mo, cu=cu)
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
    print(f"판 {args.seed}: {end}, {t * env.step_length:.0f}초, 정차 성공 {info.get('stops_done')}"
          f" / 놓침 {info.get('missed_stops')}", flush=True)
finally:
    json.dump({"seed": args.seed, "stage": ROAD.get("stage"), "end": end, "step_length": 0.5,
               "ckpt": args.ckpt, "steps": log}, open(os.path.join(OUT, "log.json"), "w"),
              ensure_ascii=False)
    try:
        env.close()
    except Exception:
        pass
    work.cleanup()
