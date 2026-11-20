"""GRU-PPO 주행 화면 (torch 없이). sumo-gui + 정류장 대기 현황 창(민기님 tools/live_board.py).

    python models/dongyeon/play_gru.py models/dongyeon/checkpoints/ts0/final.pt --seed 1000
    python models/dongyeon/play_gru.py <model.pt> --seed 1000 --episodes 3 --track

- 모델은 numpy로 계산한다 (np_policy.py, PyTorch GRU와 같은 식).
- 평가와 같은 조건: shield 감시 모드(개입하지 않고 기록만), eval_mode, 결정적 행동.
- 대기 현황 창의 버스 상태는 env-v1 정류장 판정(환경이 직접 함)에서 읽는다.
- 공용 파일은 고치지 않는다.
"""
import argparse
import ast
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.road_config import ROAD, EGO, TRAFFIC                 # noqa: E402
from env.mdp_config import SIMULATION, OBSERVATION, ACTION     # noqa: E402
from env.mdp_config import REWARD as REWARD_DEFAULTS           # noqa: E402
from env import road_builder                                   # noqa: E402
from env.sumo_env import SumoHighwayEnv                        # noqa: E402
from np_policy import NumpyGRUPolicy                           # noqa: E402


def train_reward():
    """train.py의 REWARD_OVERRIDES를 torch import 없이 읽는다 (학습과 같은 보상으로 표시)."""
    tree = ast.parse(open(os.path.join(ROOT, "train.py"), encoding="utf-8").read())
    node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "REWARD_OVERRIDES")
    return {**REWARD_DEFAULTS, **{k.arg: ast.literal_eval(k.value) for k in node.keywords}}


def main():
    p = argparse.ArgumentParser(description="GRU-PPO 주행 화면")
    p.add_argument("model")
    p.add_argument("--seed", type=int, default=1000, help="첫 판 시드 (테스트 판 1000~1099)")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--track", action="store_true", help="화면이 버스를 따라감")
    p.add_argument("--no-board", action="store_true", help="대기 현황 창 끄기")
    args = p.parse_args()

    policy = NumpyGRUPolicy(args.model)
    work = tempfile.TemporaryDirectory(prefix="cheonggye_play_")
    cfg = road_builder.build(ROAD, EGO, TRAFFIC, work.name)
    sim = {**SIMULATION, "shield": {**SIMULATION["shield"], "mode": "shadow"}}
    env = SumoHighwayEnv(cfg_path=cfg, road=ROAD, ego=EGO, mdp_sim=sim, mdp_obs=OBSERVATION,
                         action=ACTION, reward=train_reward(), traffic=TRAFFIC, gui=True)
    env.eval_mode, env.shield_mode = True, "shadow"
    if policy.obs_dim != env.obs_dim:
        raise SystemExit(f"모델 입력 {policy.obs_dim} ≠ 환경 관측 {env.obs_dim}")

    import traci
    board = names = order = None
    meta_path = os.path.join(work.name, "cgc.line_meta.json")
    if not args.no_board and os.path.exists(meta_path):
        from live_board import LiveBoard, build_map_data
        meta = json.load(open(meta_path, encoding="utf-8"))
        order = list(max((ln["stops"] for ln in meta["lines"].values()), key=len, default=[]))
        order += [s for s in meta["stops"] if s not in order]
        names = {s: meta["stops"][s].get("name", s) for s in order}
        colors = {s: meta["stops"][s].get("color") or "#2b6570" for s in order}
        numbers = {s: meta["stops"][s].get("number") or i + 1 for i, s in enumerate(order)}
        board = LiveBoard(order, names, colors,
                          build_map_data(os.path.join(work.name, "cgc.net.xml"), meta), numbers)
        print(f"대기 현황 창: {board.start()}")

    def bus_row(info):
        ego = env.ego_id
        if ego not in traci.vehicle.getIDList():
            return []
        cur = env._current_stop()
        speed = traci.vehicle.getSpeed(ego)
        nm = (names or {}).get(cur["id"], cur["id"]) if cur else None
        done, missed = info.get("stops_done", 0), info.get("missed_stops", 0)
        if cur and info.get("in_stop_area") and speed <= 0.1:
            state = f"정차 중 · {nm} ({info.get('dwell_time', 0):.0f}/10초)"
        elif speed < 0.5:
            state = "멈춤"
        else:
            state = "주행 중"
        state += f" · 정차 성공 {done} / 놓침 {missed}"
        if info.get("shield_active"):
            state += " · 개입 상황"
        x, y = traci.vehicle.getPosition(ego)
        cap = traci.vehicletype.getPersonCapacity(traci.vehicle.getTypeID(ego))
        return [{"id": ego, "onboard": len(env._onboard), "capacity": cap, "state": state,
                 "next_stop": nm or "종점", "speed_kmh": speed * 3.6,
                 "at_stop": cur["id"] if cur and info.get("in_stop_area") and speed <= 0.1 else None,
                 "x": round(x, 1), "y": round(y, 1)}]

    try:
        for ep in range(args.episodes):
            seed = args.seed + ep
            obs, _ = env.reset(seed=seed)
            policy.reset()
            if args.track:
                try:
                    traci.gui.trackVehicle("View #0", env.ego_id)
                    traci.gui.setZoom("View #0", 800)
                except traci.TraCIException:
                    pass
            info, t, ret = {}, 0, 0.0
            while True:
                obs, r, term, trunc, info = env.step(policy.predict(obs))
                t += 1
                ret += r
                if board is not None and t % 2 == 0:
                    waits = {}
                    for sid in order:
                        try:
                            waits[sid] = traci.busstop.getPersonCount(sid)
                        except traci.TraCIException:
                            pass
                    board.update(traci.simulation.getTime(), waits, bus_row(info))
                if term or trunc:
                    break
            end = ("충돌" if info.get("collided") else "도착" if info.get("arrived") else
                   "적색 위반" if info.get("red_violation") else "차로 오류" if info.get("wrong_lane")
                   else "시간 초과" if trunc else "기타")
            print(f"판 {seed}: {end}, {t * env.step_length:.0f}초, 정차 성공 {info.get('stops_done')}"
                  f" / 놓침 {info.get('missed_stops')}, 보상 {ret:.1f}", flush=True)
    except (KeyboardInterrupt, traci.FatalTraCIError):
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass
        if board is not None:
            board.stop()
        work.cleanup()


if __name__ == "__main__":
    main()
