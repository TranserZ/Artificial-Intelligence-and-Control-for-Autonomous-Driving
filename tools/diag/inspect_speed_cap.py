"""v3 ego 속도 명령 상한: 도로 30·보호구역 20을 넘을 수 있는지.

  py -3.13 tools/diag/inspect_speed_cap.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def load_env(stage: int):
    os.environ["CGC_STAGE"] = str(stage)
    for name in list(sys.modules):
        if name.startswith("env.road_config") or name == "env.road_config":
            del sys.modules[name]
        if name.startswith("env.curriculum") or name == "env.curriculum":
            del sys.modules[name]
    from env.road_config import ROAD, EGO, TRAFFIC  # noqa: WPS433
    from env.mdp_config import SIMULATION, OBSERVATION, ACTION  # noqa: WPS433
    from env import road_builder  # noqa: WPS433
    from env.sumo_env import SumoHighwayEnv  # noqa: WPS433
    from train import REWARD  # noqa: WPS433

    work = tempfile.TemporaryDirectory(prefix=f"speedcap_s{stage}_")
    cfg = road_builder.build(ROAD, EGO, TRAFFIC, work.name)
    env = SumoHighwayEnv(
        cfg_path=cfg, road=ROAD, ego=EGO,
        mdp_sim={**SIMULATION, "shield": {**SIMULATION["shield"], "mode": "off"}},
        mdp_obs=OBSERVATION, action=ACTION, reward=REWARD, traffic=TRAFFIC, gui=False)
    return env, work, ROAD, EGO


def measure(stage: int):
    import numpy as np
    import traci

    env, work, ROAD, EGO = load_env(stage)
    try:
        env.reset(seed=3000)
        peak = 0.0
        in_zone_peak = 0.0
        in_zone_steps = 0
        for _ in range(80):
            _obs, _r, term, trunc, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
            if info.get("speed") is not None:
                peak = max(peak, float(info["speed"]))
            if info.get("in_school_zone"):
                in_zone_steps += 1
                in_zone_peak = max(in_zone_peak, float(info["speed"] or 0.0))
            if term or trunc:
                break
        csv_path = os.path.join(ROOT, "data", "road_data", "school_zones.csv")
        return {
            "stage": stage,
            "elements": list(ROAD.get("elements") or []),
            "road_speed_limit_mps": ROAD.get("speed_limit"),
            "vehicle_max_speed": ROAD.get("vehicle_max_speed"),
            "ego_max_speed": EGO.get("max_speed"),
            "env.vmax_obs_norm": env.vmax,
            "env.ego_vmax_setSpeed_cap": env.ego_vmax,
            "setSpeed_clip": "np.clip(v+dv, 0, ego_vmax)",
            "speedMode": 0,
            "school_zones_csv": csv_path,
            "peak_speed_80steps_accel1": peak,
            "in_school_zone_steps": in_zone_steps,
            "in_school_zone_peak": in_zone_peak,
            "can_exceed_road_30kmh": env.ego_vmax > 8.33 + 1e-6,
            "can_exceed_school_20kmh_by_command_cap": env.ego_vmax > 5.56 + 1e-6,
        }
    finally:
        env.close()
        work.cleanup()


def main():
    rows = [measure(1), measure(4)]
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    dest = os.path.join(ROOT, "results", "idm_diag", "speed_cap.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
