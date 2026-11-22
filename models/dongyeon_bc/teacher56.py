"""5·6단계 선생님 보강 (강동연): 민기 님 bc_ppo 선생님(PPO + shield on + 정류장·신호 규칙)에
"서 있는 주정차 차를 옆 차로로 비키는" 차선 규칙을 더한다. bc_ppo 파일은 고치지 않고 클래스 메서드만 덮어쓴다.

    TEACHER_AVOID=1 CGC_STAGE=6 python models/dongyeon_bc/teacher56.py collect --ckpt ... --data ...
    (bcppo.py의 모든 모드를 그대로 받는다: collect / dagger / bc / eval)
    CGC_STAGE=6 python models/dongyeon_bc/teacher56.py diag --ckpt ... --seeds 600000-600003

왜: 6단계 선생님 100판 중 99판 시간초과 (정류장 2~3곳 뒤에서 멈춤, shield 개입 92%).
선생님 규칙에 주정차 회피가 없고, PPO 선생님은 3단계까지 학습돼 서 있는 차 뒤에서 차선을 바꾸지 않는다.
회피 규칙은 강동연 규칙 선생 v4(rule_expert.py: 6단계 완주 73%, 5단계 100%)의 것을 옮겼다.
  - 같은 차로 경로 앞 AVOID_LOOKAHEAD m 안에 불법주정차(hs_)가 있거나, 30 m 안 앞차가 서 있는 주정차 차면
  - 옆 차로가 비었을 때(앞 8 m, 뒤 8 m + 3 s × 다가오는 속도) 그쪽으로 차선 명령
  - 정류장 규칙이 차로를 정하는 중(다음 정류장 150 m 안, 정류장 차로가 아님)이면 끼어들지 않는다
차선 명령은 그 뒤 선생님 게이트(shield)가 한 번 더 검사한다.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models", "bc_ppo"))

import numpy as np  # noqa: E402
import bcppo  # noqa: E402

PARKED_PREFIX = ("hs_", "park", "moto_park", "truck")
AVOID_LOOKAHEAD = 80.0
LANE_GAP = 8.0
OVERTAKE_RANGE = 30.0


def _side_clear(env, ego, direction):
    import traci
    edge = traci.vehicle.getRoadID(ego)
    if edge.startswith(":"):
        return False
    li = traci.vehicle.getLaneIndex(ego) + direction
    if not env._is_car_lane(edge, li):
        return False
    v = traci.vehicle.getSpeed(ego)
    pos = traci.vehicle.getLanePosition(ego)
    length = traci.vehicle.getLength(ego)
    if pos - length < 0.0:
        return False
    for vid in traci.lane.getLastStepVehicleIDs(f"{edge}_{li}"):
        p = traci.vehicle.getLanePosition(vid)
        if p >= pos:
            if p - traci.vehicle.getLength(vid) - pos < LANE_GAP:
                return False
        elif (pos - length) - p < LANE_GAP + 3.0 * max(0.0, traci.vehicle.getSpeed(vid) - v):
            return False
    return True


def _parked_ahead(env, ego):
    import traci
    route = traci.vehicle.getRoute(ego)
    ahead = set(route[traci.vehicle.getRouteIndex(ego):])
    mine = env._car_lane_index(ego)
    for vid in traci.vehicle.getIDList():
        if not vid.startswith("hs_"):
            continue
        road = traci.vehicle.getRoadID(vid)
        if road not in ahead or road.startswith(":"):
            continue
        try:
            d = traci.vehicle.getDrivingDistance(ego, road, traci.vehicle.getLanePosition(vid))
        except traci.TraCIException:
            continue
        if 0.0 < d < AVOID_LOOKAHEAD and env._car_lane_index(vid) == mine:
            return True
    return False


def _stopped_parked_leader(ego):
    import traci
    lead = traci.vehicle.getLeader(ego, 80.0)
    return (lead is not None and lead[0] and lead[1] < OVERTAKE_RANGE
            and lead[0].startswith(PARKED_PREFIX) and traci.vehicle.getSpeed(lead[0]) < 0.1)


def avoid_rule(env, accel_raw, lane_raw):
    import traci
    ego = env.ego_id
    if ego not in traci.vehicle.getIDList():
        return accel_raw, lane_raw, False
    if traci.vehicle.getRoadID(ego).startswith(":") or env._quantize_lane_change(lane_raw) != 0:
        return accel_raw, lane_raw, False
    ns = env._next_stop()
    if ns is not None and 0 <= ns[0] <= bcppo.STOP_RULE_RANGE and abs(env._stop_obs()[1]) > 1e-6:
        return accel_raw, lane_raw, False          # 정류장 규칙이 차로를 정하는 중
    if not (_parked_ahead(env, ego) or _stopped_parked_leader(ego)):
        return accel_raw, lane_raw, False
    for dr in (1, -1):
        if _side_clear(env, ego, dr):
            return accel_raw, float(dr), True
    return accel_raw, lane_raw, False


_orig_rules = bcppo.TeacherEnv.teacher_rules


def teacher_rules(self, accel_raw, lane_raw):
    accel_raw, lane_raw, used = _orig_rules(self, accel_raw, lane_raw)
    accel_raw, lane_raw, used_avoid = avoid_rule(self, accel_raw, lane_raw)
    return accel_raw, lane_raw, used or used_avoid


_orig_gate = bcppo.ExpertGateEnv._lane_change_is_safe


def _avoid_situation(env):
    """주정차 차 뒤에 막혀 천천히 가거나 서 있는가."""
    import traci
    ego = env.ego_id
    if ego not in traci.vehicle.getIDList() or traci.vehicle.getSpeed(ego) >= 3.0:
        return False
    return _stopped_parked_leader(ego) or _parked_ahead(env, ego)


def lane_gate(self, direction):
    """선생님 차선변경 게이트: 주정차 차 뒤에 막혔을 때만 규칙 선생 v4 기준(앞 8 m, 뒤 8 m + 3 s × 다가오는 속도)으로 본다.

    1차 보강(회피 차선 명령만)은 6단계 완주 1% → 15%. 남은 85%는 hs_samil_east·hs_jugyo 뒤에서 서 있었다:
    시범용 넉넉한 게이트(뒤차 3 s·접근 4 s)는 서 있는 버스 기준 옆 차로에 약 32 m 빈틈이 필요하고,
    교차로 앞뒤 20 m 금지 때문에 교차로 바로 뒤 주정차(주교, 엣지 7 m)에서는 아예 못 바꾼다.
    """
    if isinstance(self, bcppo.TeacherEnv) and _avoid_situation(self):
        return _side_clear(self, self.ego_id, direction)
    return _orig_gate(self, direction)


if os.environ.get("TEACHER_AVOID", "1") == "1":
    bcppo.TeacherEnv.teacher_rules = teacher_rules
if os.environ.get("TEACHER_AVOID_GATE", "1") == "1":
    bcppo.ExpertGateEnv._lane_change_is_safe = lane_gate


def diag(argv):
    """선생님이 오래 서 있는 곳(엣지·위치·앞차)을 판마다 찍는다."""
    import argparse
    import traci
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seeds", required=True)
    a = p.parse_args(argv)
    lo, hi = (int(x) for x in a.seeds.split("-"))
    env = bcppo.make_env("on", lc_gate=bcppo.EXPERT_LC_GATE, teacher=True)
    env.eval_mode = True
    agent = bcppo.make_agent(env, 0)
    bcppo.load_weights(agent, a.ckpt)
    for seed in range(lo, hi + 1):
        obs, _ = env.reset(seed=seed)
        still, reported, t = 0, set(), 0
        while True:
            act = agent.predict(obs, deterministic=True)
            obs, r, te, tr, info = env.step(act)
            t += 1
            ego = env.ego_id
            if ego in traci.vehicle.getIDList():
                v = traci.vehicle.getSpeed(ego)
                still = still + 1 if v < 0.1 else 0
                edge = traci.vehicle.getRoadID(ego)
                if still == 120 and (edge, info.get("stops_done")) not in reported:   # 60초 정지
                    reported.add((edge, info.get("stops_done")))
                    lead = traci.vehicle.getLeader(ego, 80.0)
                    ld = (lead[0], round(lead[1], 1), round(traci.vehicle.getSpeed(lead[0]), 1)) if lead and lead[0] else None
                    print(f"  seed {seed} t {t} 60초 정지: {edge} lane {traci.vehicle.getLaneIndex(ego)} "
                          f"pos {traci.vehicle.getLanePosition(ego):.0f} 정차 {info.get('stops_done')} 앞차 {ld} "
                          f"신호 {env._next_signal()}", flush=True)
            if te or tr:
                print(f"== seed {seed}: {bcppo.end_reason(info, te, tr)} {t} step 정차 {info.get('stops_done')}", flush=True)
                break
    env.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "diag":
        diag(sys.argv[2:])
    else:
        bcppo.main()
