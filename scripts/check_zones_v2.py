"""정차금지지대(청계5가 교차로)·청계2가 안전지대 확인 + GUI 스크린샷.

- 운전: keep-lane(목표 속도) + 상황별 강제 조작. 감점은 같은 판을 감점 켬/끔으로 돌려
  같은 행동을 재생한 보상 차이(끔 − 켬)로 잰다 (scripts/check_special_zones.py와 같은 방식).
- 청계5가 교차로는 r0가 지나지 않아 회차 경로 r0_loop로 확인한다.
결과: data_huiyoung/report/zone_checks_v2.json, 스크린샷 data_huiyoung/report/zone_screens/v2_*.png
실행: py -X utf8 scripts/check_zones_v2.py [--no-gui]
"""
import json, math, os, re, sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "tools")); sys.path.insert(0, os.path.join(BASE, "scripts"))
os.chdir(BASE)
import numpy as np  # noqa: E402
import traci, sumolib  # noqa: E402
import zone_markings as zm  # noqa: E402
import validate_geometry as vg  # noqa: E402
from env.road_config import ROAD, EGO, TRAFFIC  # noqa: E402
from env.mdp_config import SIMULATION, OBSERVATION, ACTION  # noqa: E402
from env.sumo_env import SumoHighwayEnv  # noqa: E402
from train import REWARD  # noqa: E402

CFG = os.path.join(BASE, "env", "sumo", "cgc.sumocfg")
NETF = os.path.join(BASE, "env", "sumo", "cgc.net.xml")
OUT = os.path.join(BASE, "data_huiyoung", "report", "zone_checks_v2.json")
SHOTS = os.path.join(BASE, "data_huiyoung", "report", "zone_screens")
J5 = "cluster_n5768277866_n871729539_n9040757788"   # 청계5가 사거리
EGO_ID = "ego"
net = sumolib.net.readNet(NETF, withInternal=True)


def make_env(gui=False, **rw):
    env = SumoHighwayEnv(cfg_path=CFG, road=ROAD, ego=EGO, mdp_sim=SIMULATION, mdp_obs=OBSERVATION,
                         action=ACTION, reward={**REWARD, **rw}, traffic=TRAFFIC, gui=gui,
                         gui_autostart=True)
    env._hs_orig = list(env._hotspots)
    return env


def accel_to(env, vt):
    v = traci.vehicle.getSpeed(EGO_ID)
    need = (vt - v) / env.step_length
    return float(np.clip(need / (env.max_accel if need >= 0 else env.max_decel), -1, 1))


def run(env, seed, ctl=None, actions=None, route="r0_loop", hs_prob=0.0, max_steps=2400, on_step=None):
    env.ego_routes = {route: 1.0}
    env._hotspots = [(h, e, a, b, v, hs_prob) for h, e, a, b, v, _ in env._hs_orig]
    obs, _ = env.reset(seed=seed)
    recs, acts, mem = [], [], {}
    for t in range(max_steps):
        if EGO_ID not in traci.vehicle.getIDList():
            break
        if actions is not None and t >= len(actions):
            break             # 재생: A가 멈춘 지점까지만
        st = dict(rid=traci.vehicle.getRoadID(EGO_ID), pos=traci.vehicle.getLanePosition(EGO_ID),
                  odo=traci.vehicle.getDistance(EGO_ID), v=traci.vehicle.getSpeed(EGO_ID))
        a = actions[t] if actions is not None else ctl(env, st, t, mem)
        acts.append(a)
        obs, r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
        alive = EGO_ID in traci.vehicle.getIDList()
        rec = dict(t=t, r=float(r), odo=traci.vehicle.getDistance(EGO_ID) if alive else None,
                   rid=traci.vehicle.getRoadID(EGO_ID) if alive else None,
                   v=traci.vehicle.getSpeed(EGO_ID) if alive else None,
                   sig=info.get("signal_state"), collided=bool(info.get("collided")))
        recs.append(rec)
        if on_step:
            on_step(t)
        if term or trunc or mem.get("stop_run"):
            break
    env.close()
    return recs, acts


def paired(seed, ctl, **kw):
    """감점 켬(A)으로 주행 → 같은 행동을 감점 끔(B)으로 재생 → 스텝별 감점 = B.r − A.r."""
    A = make_env()
    recA, acts = run(A, seed, ctl, **kw)
    B = make_env(no_stop_zone_penalty=0.0)
    recB, _ = run(B, seed, actions=acts, **kw)
    out = []
    for a, b in zip(recA, recB):
        if a["odo"] is not None and b["odo"] is not None and abs(a["odo"] - b["odo"]) > 0.05:
            raise RuntimeError(f"재생 어긋남 t={a['t']}")
        out.append(dict(a, pen=round(b["r"] - a["r"], 4)))
    return out


def in_j5(rid):
    return bool(rid) and rid.startswith(":") and rid[1:rid.rfind("_")] == J5


res = {}
# 청계5가가 r0_loop에서 몇 m인지 (교차로 진입·진출)
rou = open(os.path.join(BASE, "env", "sumo", "cgc.rou.xml"), encoding="utf-8").read()
loop_ids = re.search(r'<route id="r0_loop" edges="([^"]*)"', rou).group(1).split()
items, _ = vg.build_route(net, vg.load_connections(NETF), loop_ids)
jit = next(it for it in items if it["kind"] == "internal" and it["id"] == J5)
res["j5"] = dict(id=J5, coord=[round(c, 1) for c in net.getNode(J5).getCoord()],
                 lonlat=[round(v, 6) for v in vg.unproject(*net.getNode(J5).getCoord())],
                 r0_loop_entry=round(jit["start"], 1), r0_loop_exit=round(jit["start"] + jit["length"], 1),
                 r0_end=2099.4, internal_len=round(jit["length"], 1))
print("청계5가:", res["j5"])

# ───── C-1. 청계5가 정차금지 ─────
def ctl_inside(env, st, t, mem):
    """교차로 안에 들어가면 바로 제동해 10초 정지, 그다음 재출발."""
    if mem.get("done"):
        if not in_j5(st["rid"]) and mem.get("left_j"):
            mem["stop_run"] = True
        mem["left_j"] = mem.get("left_j") or not in_j5(st["rid"])
        return [accel_to(env, 8.0), 0.0]
    if in_j5(st["rid"]):
        mem["in"] = True
    if mem.get("in"):
        if st["v"] < 0.1:
            mem["held"] = mem.get("held", 0) + 1
            if mem["held"] >= 20:
                mem["done"] = True
        return [-1.0, 0.0]
    return [accel_to(env, 8.0), 0.0]


def ctl_before(env, st, t, mem):
    """교차로 직전 엣지(e1455526426_1) 끝 정지선 앞에서 10초 정지 (적색 대기와 같은 자리)."""
    if mem.get("done"):
        if in_j5(st["rid"]):
            mem["stop_run"] = True        # 교차로에 들어가면 끝 (안에서 서는 경우와 섞이지 않게)
        return [accel_to(env, 8.0), 0.0]
    if st["rid"] == "e1455526426_1" or st["rid"].startswith(":n4202872186"):
        L = traci.lane.getLength(traci.vehicle.getLaneID(EGO_ID))
        left = (L - st["pos"]) if st["rid"] == "e1455526426_1" else 14.0 + L - st["pos"]
        if st["v"] < 0.1 and left < 4.0:
            mem["held"] = mem.get("held", 0) + 1
            if mem["held"] >= 20:
                mem["done"] = True
            return [-1.0, 0.0]
        return [accel_to(env, min(8.0, math.sqrt(2 * 2.0 * max(left - 1.5, 0.0)))), 0.0]
    return [accel_to(env, 8.0), 0.0]


ns = {}
for name, ctl in (("교차로 안 10초 정지", ctl_inside), ("정지선 앞 10초 정지", ctl_before)):
    for seed in range(40, 52):
        d = paired(seed, ctl)
        inj = [x for x in d if in_j5(x["rid"])]
        stopped_in = [x for x in inj if x["v"] is not None and x["v"] < 0.5]
        if name.startswith("교차로 안") and len(stopped_in) < 20:
            continue      # 적색에 막혀 교차로에 못 들어간 판 → 다음 seed
        before = [x for x in d if x["rid"] == "e1455526426_1" and x["v"] is not None and x["v"] < 0.1]
        if name.startswith("정지선") and len(before) < 20:
            continue
        pen = [x for x in d if x["pen"] > 1e-6]
        ns[name] = dict(seed=seed, steps_in_junction=len(inj), stopped_in_junction=len(stopped_in),
                        stopped_before_line=len(before),
                        penalized=len(pen), penalized_in_junction=sum(1 for x in pen if in_j5(x["rid"])),
                        pen_values=sorted({x["pen"] for x in pen}),
                        penalized_lanes=sorted({x["rid"] for x in pen}),
                        signal_while_waiting=sorted({x["sig"] for x in before if x["sig"]}),
                        collided=any(x["collided"] for x in d))
        print("C-1", name, ns[name])
        break
res["j5_no_stop"] = ns

# ───── C-2. 청계2가 안전지대: 차량 위치가 사다리꼴 안에 들어가는 스텝 수 ─────
row = next(r for r in zm._read_csv("safety_zones.csv"))
fr = zm.EdgeFrame(net.getEdge(row["edge_id"]))
a, b, w0, w1 = (float(row[k]) for k in ("start_pos", "end_pos", "width_start", "width_end"))
ss = [a + (b - a) * i / 20 for i in range(21)]
poly = [fr.xy(s, fr.width) for s in ss] + [fr.xy(s, fr.width - (w0 + (w1 - w0) * (s - a) / (b - a))) for s in reversed(ss)]


def inside(x, y):
    c = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


cnt = dict(ego_steps=0, bg_vehicle_steps=0, bg_vehicles=set(), lanes=set(), steps=0)


def count(_t):
    cnt["steps"] += 1
    for vid in traci.vehicle.getIDList():
        x, y = traci.vehicle.getPosition(vid)          # 앞범퍼 중심
        ang = math.radians(90 - traci.vehicle.getAngle(vid))
        L = traci.vehicle.getLength(vid)
        cx, cy = x - math.cos(ang) * L / 2, y - math.sin(ang) * L / 2   # 차체 중심
        if inside(cx, cy) or inside(x, y):
            if vid == EGO_ID:
                cnt["ego_steps"] += 1
            else:
                cnt["bg_vehicle_steps"] += 1
                cnt["bg_vehicles"].add(vid)
            cnt["lanes"].add(traci.vehicle.getLaneID(vid))


def ctl_keep(env, st, t, mem):
    if st["odo"] > 1000:
        mem["stop_run"] = True
    return [accel_to(env, 8.0), 0.0]


for seed in (60, 61, 62):
    E = make_env()
    run(E, seed, ctl_keep, route="r0", on_step=count)
res["safety_zone"] = dict(edge=row["edge_id"], range=f"{a:g}~{b:g} m", left=True, w=f"{w0:g}→{w1:g}",
                          episodes=3, sim_steps=cnt["steps"], ego_steps_inside=cnt["ego_steps"],
                          bg_vehicle_steps_inside=cnt["bg_vehicle_steps"], bg_vehicles=len(cnt["bg_vehicles"]),
                          lanes_inside=sorted(cnt["lanes"]),
                          lane_layout=[(ln.getIndex(), "보도" if not ln.allows("passenger") else "주행", ln.getWidth())
                                       for ln in net.getEdge(row["edge_id"]).getLanes()])
print("C-2", res["safety_zone"])

# ───── C-3. 스크린샷 ─────
if "--no-gui" not in sys.argv:
    os.makedirs(SHOTS, exist_ok=True)
    G = make_env(gui=True)
    G.ego_routes = {"r0": 1.0}
    G.reset(seed=11)
    for _ in range(60):
        G.step(np.array([0.0, 0.0], dtype=np.float32))
    V = "View #0"
    try:
        traci.gui.trackVehicle(V, "")
    except traci.TraCIException:
        pass
    j5xy = net.getNode(J5).getCoord()
    spots = [("v2_1_cheonggye5_nostop", j5xy, 5000),
             ("v2_2_cheonggye3_none", zm.EdgeFrame(net.getEdge("e908646367_6")).xy(25.0, 3.0), 9000),
             ("v2_3_sewoon_none", zm.EdgeFrame(net.getEdge("e218924602_3")).xy(15.0, 3.0), 9000),
             ("v2_4_baeogae_none", zm.EdgeFrame(net.getEdge("e218924602_6")).xy(105.0, 3.0), 9000),
             ("v2_5_cheonggye2_safety", fr.xy(5.0, fr.width - 1.0), 12000)]
    for name, (x, y), zoom in spots:
        traci.gui.setOffset(V, x, y)
        traci.gui.setZoom(V, zoom)
        traci.simulationStep(); traci.simulationStep()
        traci.gui.screenshot(V, os.path.join(SHOTS, name + ".png"), 1100, 700)
        traci.simulationStep(); traci.simulationStep()
    G.close()
    res["screenshots"] = [s[0] + ".png" for s in spots]
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print("saved", OUT)
