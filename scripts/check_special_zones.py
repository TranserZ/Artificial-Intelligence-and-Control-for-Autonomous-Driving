"""특수구역 정상작동 확인: 상황을 일부러 만들어 감점·관측·주정차를 확인한다.

- 운전: keep-lane(차선 유지) + 목표 속도 추종. 시나리오마다 필요한 조작만 강제로 넣는다.
- 감점 측정: 같은 판(seed)을 감점 켬(A) / 그 감점만 0(B)으로 두 번 돌리고, A의 행동을 B에 그대로
  재생한다. 스텝별 보상 차 A−B = 그 감점의 실제 값 (다른 감점과 섞이지 않음).
결과: 같은 폴더 zone_checks.json
"""
import json, os, sys
BASE = r"C:\Users\USER\Desktop\청계로드"
sys.path.insert(0, BASE); os.chdir(BASE)
import numpy as np  # noqa: E402
import traci  # noqa: E402
from env.road_config import ROAD, EGO, TRAFFIC  # noqa: E402
from env.mdp_config import SIMULATION, OBSERVATION, ACTION  # noqa: E402
from env.sumo_env import SumoHighwayEnv  # noqa: E402
from train import REWARD  # noqa: E402

CFG = os.path.join(BASE, "env", "sumo", "cgc.sumocfg")
OUT = os.path.join(BASE, "data_huiyoung", "report", "special_zone_checks.json")
EGO_ID = "ego"


def make_env(**reward_over):
    return SumoHighwayEnv(cfg_path=CFG, road=ROAD, ego=EGO, mdp_sim=SIMULATION, mdp_obs=OBSERVATION,
                          action=ACTION, reward={**REWARD, **reward_over}, traffic=TRAFFIC, gui=False)


def accel_to(env, v_target):
    v = traci.vehicle.getSpeed(EGO_ID)
    need = (v_target - v) / env.step_length
    return float(np.clip(need / (env.max_accel if need >= 0 else env.max_decel), -1, 1))


def state(env, obs):
    if EGO_ID not in traci.vehicle.getIDList():
        return None
    ld = traci.vehicle.getLeader(EGO_ID, 50.0)
    return dict(odo=traci.vehicle.getDistance(EGO_ID), edge=traci.vehicle.getRoadID(EGO_ID),
                pos=traci.vehicle.getLanePosition(EGO_ID), lane=traci.vehicle.getLaneIndex(EGO_ID),
                v=traci.vehicle.getSpeed(EGO_ID), o9=float(obs[9]) * 50.0,
                o40=float(obs[40]), o41=float(obs[41]),
                lead=(ld[0] if ld else None), lead_gap=(float(ld[1]) if ld else None))


def drive(env, seed, controller, hotspot_prob=None, drop_stops=False, max_steps=2400, actions=None):
    """1판 주행. controller(env, st, t, mem) -> action. actions를 주면 그대로 재생."""
    if hotspot_prob is not None:
        env._hotspots = [(h, e, a, b, v, hotspot_prob) for h, e, a, b, v, _ in env._hotspots_orig]
    obs, _ = env.reset(seed=seed)
    if drop_stops:   # 보호구역 속도 시험: 정류장 정차 없이 통과 (정류장 규칙이 속도를 잡지 않게)
        while traci.vehicle.getStops(EGO_ID, 1):
            traci.vehicle.replaceStop(EGO_ID, 0, "")
    recs, mem, acts = [], {}, []
    for t in range(max_steps):
        st = state(env, obs)
        if st is None:
            break
        a = actions[t] if actions is not None else controller(env, st, t, mem)
        acts.append(a)
        obs, r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
        st2 = state(env, obs) or {}
        recs.append(dict(t=t, r=float(r), a=list(map(float, a)), **{k: st2.get(k) for k in
                    ("odo", "edge", "pos", "lane", "v", "o9", "o40", "o41", "lead", "lead_gap")},
                    collided=bool(info.get("collided")), arrived=bool(info.get("arrived")),
                    red=bool(info.get("red_violation"))))
        if term or trunc:
            break
    env.close()   # TraCI 연결은 하나라 판마다 닫는다 (다른 env 인스턴스가 다음 판을 연다)
    return recs, acts


def diff(recA, recB):
    out = []
    for a, b in zip(recA, recB):
        if abs((a["odo"] or 0) - (b["odo"] or 0)) > 0.05:
            raise RuntimeError(f"재생이 어긋남 t={a['t']}: {a['odo']} vs {b['odo']}")
        out.append(dict(a, pen=round(b["r"] - a["r"], 4)))
    return out


results = {}

# ───────────────────── 1. 불법주정차 ─────────────────────
envA = make_env()
envA._hotspots_orig = list(envA._hotspots)
HS = {h: (e, a, b) for h, e, a, b, _, _ in envA._hotspots_orig}


def detour_ctl(env, st, t, mem):
    """keep-lane(목표 8 m/s) + 정차 차량이 20 m 안에서 막으면 왼쪽으로 차선변경 명령."""
    lane = 0.0
    if st["lead"] and st["lead_gap"] < 20.0 and traci.vehicle.isStopped(st["lead"]):
        lane = 1.0
    return [accel_to(env, 8.0), lane]


def check_hotspot(h, seed):
    """seed 판에서 지점 h: 생성 위치·차로, 버스가 지날 때까지 정차 유지, 관측 obs[9]."""
    e, a, b = HS[h]
    envA._hotspots = list(envA._hotspots_orig)
    obs, _ = envA.reset(seed=seed)
    init = {v: (traci.vehicle.getLaneIndex(v), round(traci.vehicle.getLanePosition(v), 2),
                traci.vehicle.getTypeID(v)) for v in traci.vehicle.getIDList() if v.startswith(f"hs_{h}_")}
    right = envA._car_lanes(e)[0]
    seen = dict(first_gap=None, o9_first_in50=None, min_gap=999.0, min_o9=999.0)
    passed, end, on_e = None, None, False
    for t in range(2400):
        st = state(envA, obs)
        if st is None:
            break
        if st["lead"] and st["lead"].startswith(f"hs_{h}_"):
            if seen["first_gap"] is None:
                seen["first_gap"] = round(st["lead_gap"], 1)
            if st["lead_gap"] < 50:
                if seen["o9_first_in50"] is None:
                    seen["o9_first_in50"] = (round(st["lead_gap"], 1), round(st["o9"], 1))
                seen["min_gap"] = min(seen["min_gap"], round(st["lead_gap"], 1))
                seen["min_o9"] = min(seen["min_o9"], round(st["o9"], 1))
        if st["edge"] == e:
            on_e = True
        if passed is None and locals().get("on_e") and st["edge"] != e:
            ids = traci.vehicle.getIDList()
            still = [v for v in init if v in ids and traci.vehicle.isStopped(v)
                     and abs(traci.vehicle.getLanePosition(v) - init[v][1]) < 0.05
                     and traci.vehicle.getLaneIndex(v) == init[v][0]]
            passed = dict(t=round(traci.simulation.getTime(), 1), still_stopped=len(still), ego_lane=st["lane"])
            break
        obs, r, term, trunc, info = envA.step(np.asarray(detour_ctl(envA, st, t, {}), dtype=np.float32))
        if term or trunc:
            end = dict(collided=bool(info.get("collided")), arrived=bool(info.get("arrived")))
            break
    envA.close()
    return dict(seed=seed, edge=e, zone=f"{a:g}~{b:g}", right_lane=right,
                vehicles=[(i[2], i[1], i[0]) for i in init.values()],
                all_rightmost=bool(init) and all(i[0] == right for i in init.values()),
                passed=passed, obs=seen, end=end)


scan, hs_res = [], {}
for sd in range(0, 12):
    envA._hotspots = list(envA._hotspots_orig)
    envA.reset(seed=sd)
    got = [h for h, _, n in envA._hotspot_placed if n > 0]
    envA.close()
    scan.append((sd, got))
for h in ["jugyo"] if "--jugyo-only" in sys.argv else HS:
    for sd, got in scan:
        if h in got:
            r = check_hotspot(h, sd)
            if r["passed"] is not None:     # 버스가 그 지점을 실제로 지난 판
                hs_res[h] = r
                break
            hs_res.setdefault(h + "_실패", []).append(r)
results["hotspots"] = dict(scan=scan, res=hs_res)
print("시드 탐색:", scan)
for k, v in hs_res.items():
    print("1.", k, json.dumps(v, ensure_ascii=False))

# ───────────────────── 2. 정차금지지대 ─────────────────────
envB = make_env(no_stop_zone_penalty=0.0)
envB._hotspots_orig = list(envB._hotspots)


def stop_ctl(stop_odo, hold_s=10.0):
    """주행거리 stop_odo에 서도록 (감속 a=2.5 m/s² 곡선) 제동하고 hold_s초 정지 후 다시 출발."""
    def ctl(env, st, t, mem):
        if mem.get("done"):
            return [accel_to(env, 8.0), 0.0]
        dist = stop_odo - st["odo"]
        if st["v"] < 0.1 and dist < 3.0:
            mem["held"] = mem.get("held", 0) + 1
            if mem["held"] >= int(hold_s / env.step_length):
                mem["done"] = True
            return [-1.0, 0.0]
        v_ref = float(np.sqrt(2.0 * 2.0 * max(dist, 0.0)))
        return [accel_to(env, min(8.0, v_ref)), 0.0]
    return ctl


ZONES = {"청계3가 접근": ("e908646367_6", 10.0, 40.0)}   # 배오개다리는 실제 정차금지지대 없음 (11/17 확인)
ns = {}
for name, zname, stop_odo in (("청계3가 접근: 구역 안 (1186 m)", "청계3가 접근", 1186.0),
                              ("청계3가 접근: 구역 바로 앞 (1165 m)", "청계3가 접근", 1165.0)):
    edge, z0, z1 = ZONES[zname]
    recA, acts = drive(envA, 7, stop_ctl(stop_odo), hotspot_prob=0.0)
    recB, _ = drive(envB, 7, None, hotspot_prob=0.0, actions=acts)
    d = diff(recA, recB)
    hold = [x for x in d if x["v"] is not None and x["v"] < 0.1 and abs((x["odo"] or 0) - stop_odo) < 4.0]
    on_e = [x for x in d if x["edge"] == edge]
    pen_e = [x for x in on_e if x["pen"] > 1e-6]
    stop_in_zone = [x for x in on_e if x["v"] < 0.5 and z0 <= x["pos"] <= z1]
    ns[name] = dict(
        forced_stop_at=(round(min(x["pos"] for x in hold if x["edge"] == edge), 1) if any(x["edge"] == edge for x in hold) else None),
        forced_stop_edge=(hold[0]["edge"] if hold else None),
        forced_hold_steps=len(hold),
        forced_hold_penalized=sum(1 for x in hold if x["pen"] > 1e-6),
        pen_values=sorted({x["pen"] for x in pen_e}),
        stopped_in_zone_steps=len(stop_in_zone),
        penalized_steps_on_edge=len(pen_e),
        penalized_not_in_zone=sum(1 for x in pen_e if not (z0 <= x["pos"] <= z1)),
        other_stops_in_zone=sorted({round(x["pos"], 1) for x in stop_in_zone if x not in hold})[:6],
    )
    print("2.", name, ns[name])
results["no_stop"] = ns

# ───────────────────── 3. 어린이보호구역 ─────────────────────
envC = make_env(zone_overspeed_penalty=0.0, zone_approach_penalty=0.0)
envC._hotspots_orig = list(envC._hotspots)
Z0, Z1 = envA._zones[0][0] if envA._zones else 1307.0, envA._zones[0][1] if envA._zones else 1558.0
sz = {}
def entry_speed(seed, vt):
    def ctl(env, st, t, mem):
        return [accel_to(env, vt if st["odo"] > 900 else 8.0), 0.0]
    recs, _ = drive(envA, seed, ctl, hotspot_prob=0.0, drop_stops=True, max_steps=1200)
    ins = [x for x in recs if x["odo"] and Z0 <= x["odo"] <= Z1]
    return min((x["v"] for x in ins), default=0.0)


best = max(range(21, 33), key=lambda sd: entry_speed(sd, 40 / 3.6))
print("3. 보호구역 시험 seed", best)
for label, v_kmh in (("40 km/h", 40.0), ("28 km/h", 28.0)):
    vt = v_kmh / 3.6

    def speed_ctl(env, st, t, mem, vt=vt):
        return [accel_to(env, vt if st["odo"] > 900 else 8.0), 0.0]

    recA, acts = drive(envA, best, speed_ctl, hotspot_prob=0.0, drop_stops=True)
    recC, _ = drive(envC, best, None, hotspot_prob=0.0, drop_stops=True, actions=acts)
    d = diff(recA, recC)
    inside = [x for x in d if x["odo"] is not None and Z0 <= x["odo"] <= Z1]
    over = [x for x in inside if x["v"] > 8.33 + 0.3]
    over_nopen = [x for x in over if x["pen"] <= 1e-6]
    under_pen = [x for x in inside if x["v"] <= 8.33 + 0.3 and x["pen"] > 1e-6]
    before = [x for x in d if x["odo"] is not None and Z0 - 120 <= x["odo"] < Z0]
    obs_first = next((x for x in d if x["o40"] is not None and x["o40"] < 1.0), None)
    gaps, prev = [], None   # 감점 없는 과속 구간 (연속 스텝 묶음)
    for x in over_nopen:
        if prev is not None and x["t"] == prev["t"] + 1:
            gaps[-1][1] = x["odo"]
        else:
            gaps.append([x["odo"], x["odo"], x["edge"]])
        prev = x
    sz[label] = dict(
        zone=(round(Z0, 1), round(Z1, 1)),
        inside_steps=len(inside), inside_speed=(round(min(x["v"] for x in inside) * 3.6, 1),
                                                round(max(x["v"] for x in inside) * 3.6, 1)) if inside else None,
        inside_internal_steps=sum(1 for x in inside if x["edge"].startswith(":")),
        over_steps=len(over), over_penalized=len(over) - len(over_nopen),
        over_internal_penalized=sum(1 for x in over if x["edge"].startswith(":") and x["pen"] > 1e-6),
        pen_values=sorted({x["pen"] for x in inside if x["pen"] > 1e-6})[:5],
        pen_max=max((x["pen"] for x in inside), default=0.0),
        gaps_no_penalty=[(round(a, 1), round(b, 1), e) for a, b, e in gaps],
        under_but_penalized=len(under_pen),
        approach_pen_steps=sum(1 for x in before if x["pen"] > 1e-6),
        approach_pen_max=max((x["pen"] for x in before), default=0.0),
        obs_first=(round(obs_first["odo"], 1), round(obs_first["o40"], 3), round(obs_first["o41"], 3)) if obs_first else None,
        obs_at_entry=next(((round(x["odo"], 1), round(x["o40"], 3), round(x["o41"], 3)) for x in d if x["odo"] and x["odo"] >= Z0), None),
        obs_after=next(((round(x["odo"], 1), round(x["o40"], 3), round(x["o41"], 3)) for x in d if x["odo"] and x["odo"] > Z1 + 5), None),
        end=dict(collided=d[-1]["collided"], arrived=d[-1]["arrived"], red=any(x["red"] for x in d)),
        slowest_inside=(lambda m: (round(m["odo"], 1), m["edge"], round(m["v"] * 3.6, 1), m["lead"], m["lead_gap"] and round(m["lead_gap"], 1)))(min(inside, key=lambda x: x["v"])) if inside else None,
    )
    print("3.", label, sz[label])
results["school"] = sz
for e in (envA, envB, envC):
    e.close()
json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print("saved", OUT)
