"""6단계 실패 진단 (강동연): 학생 혼자(shield shadow) 운전하며 실패 순간을 자세히 찍는다.

    OBS_SCAN=1 CGC_STAGE=6 python models/dongyeon_bc/diag8.py CKPT SEED_BASE N

신호 위반: 넘기 직전 신호 id·색(r/u = 적색, s = 일시정지 미준수)·정지선 거리·속도·남은 시간·가속 명령 이력
충돌: 상대 차(종류·속도)·충돌 유형·직전 차선 명령·내 차선 위치
줄 머리 "RED"/"COL"/"END"로 grep 해서 모은다.
"""
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import teacher56  # noqa: E402,F401  (OBS_SCAN=1이면 관측 보강)
bcppo = teacher56.bcppo
import traci  # noqa: E402

ckpt, s0, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
env = bcppo.make_env("shadow")
env.eval_mode = True
agent = bcppo.make_agent(env, 0)
bcppo.load_weights(agent, ckpt)
for seed in range(s0, s0 + n):
    obs, _ = env.reset(seed=seed)
    hist = collections.deque(maxlen=16)
    t = 0
    while True:
        act = agent.predict(obs, deterministic=True)
        ego = env.ego_id
        sig = None
        if ego in traci.vehicle.getIDList():
            sig = env._next_signal()
            ld = traci.vehicle.getLeader(ego, 60.0)
            hist.append((t, round(traci.vehicle.getSpeed(ego), 1), round(float(act[0]), 2),
                         env._quantize_lane_change(float(act[1])) if len(act) > 1 else 0,
                         traci.vehicle.getRoadID(ego), traci.vehicle.getLaneIndex(ego),
                         round(traci.vehicle.getLanePosition(ego)),
                         (sig[0][-10:], round(sig[1], 1), sig[2], round(sig[3] or -1, 1)) if sig else None,
                         (ld[0][:14], round(ld[1], 1), round(traci.vehicle.getSpeed(ld[0]), 1)) if ld and ld[0] else None))
        obs, r, te, tr, info = env.step(act)
        t += 1
        cs = [(c.collider, c.victim, c.type, c.colliderType, c.victimType, round(c.colliderSpeed, 1), round(c.victimSpeed, 1),
               c.lane) for c in traci.simulation.getCollisions()]
        if te or tr:
            why = bcppo.end_reason(info, te, tr)
            print(f"END seed {seed}: {why} t {t} 정차 {info.get('stops_done')}", flush=True)
            if why == "red_light" and hist:
                h = hist[-1]
                print(f"RED seed {seed} edge {h[4]} lane {h[5]} pos {h[6]} v {h[1]} sig {h[7]} "
                      f"accel {[x[2] for x in list(hist)[-8:]]} v_hist {[x[1] for x in list(hist)[-8:]]}", flush=True)
            if cs:
                print(f"COL seed {seed} {cs} lc_hist {[x[3] for x in list(hist)[-10:]]} last {hist[-1] if hist else None}", flush=True)
            if why != "completed":
                for h in hist:
                    print("    ", h, flush=True)
            break
env.close()
