"""IDM 차간거리 유지 + 정류장 10초 정차. 차선변경 없음.

진단 전용. 학습 데이터로 쓰지 않는다. 실행 중 shield는 끈다.
선행차·정류장 위치는 TraCI(특권)로 읽는다. 배포 정책이 아니다.

IDM: Treiber, Hennecke, Helbing, "Congested traffic states in empirical
observations and microscopic simulations", Phys. Rev. E 62, 2000.
계수: v0=ego_vmax, a=EGO accel, b=EGO decel, T=1.5 s, s0=2.0 m, delta=4.
"""
from __future__ import annotations

import math

import numpy as np
import traci


class IdmKeepLaneStop:
    def __init__(self, env, time_headway=1.5, min_gap=2.0, delta=4.0):
        self.env = env
        self.T = float(time_headway)
        self.s0 = float(min_gap)
        self.delta = float(delta)

    def predict(self, obs, deterministic=True):
        env = self.env
        ego = env.ego_id
        if ego not in traci.vehicle.getIDList():
            return np.array([0.0, 0.0], dtype=np.float32)
        v = float(traci.vehicle.getSpeed(ego))
        a_max = float(env.max_accel)
        b = float(env.max_decel)
        v0 = max(float(env.ego_vmax), 0.1)
        dt = float(env.step_length)

        gap, v_lead = self._leader_gap_speed(v)
        s_star = self.s0 + v * self.T + v * (v - v_lead) / (2.0 * math.sqrt(max(a_max * b, 1e-6)))
        acc = a_max * (1.0 - (v / v0) ** self.delta - (s_star / max(gap, 0.1)) ** 2)

        st = env._current_stop()
        if st is not None and env._in_stop_area(st):
            dwell = env._dwell_steps * dt
            if dwell < env.stop_dwell_time - 1e-9:
                # 구간 안에서는 0.1 m/s 이하가 될 때까지 풀 브레이크. 안 그러면
                # IDM 잔여속도가 stop_speed를 넘겨 10초가 안 쌓인다.
                acc = -b

        if acc >= 0.0:
            raw = acc / max(a_max, 1e-6)
        else:
            raw = acc / max(b, 1e-6)
        return np.array([float(np.clip(raw, -1.0, 1.0)), 0.0], dtype=np.float32)

    def _leader_gap_speed(self, v):
        env = self.env
        ego = env.ego_id
        gap, v_lead = 1e6, v
        try:
            lead = traci.vehicle.getLeader(ego, 150.0)
        except traci.TraCIException:
            lead = None
        if lead is not None:
            gap, v_lead = float(lead[1]), float(traci.vehicle.getSpeed(lead[0]))

        ns = env._next_stop()
        st = env._current_stop()
        # 정류장 차로가 아니면 가상 선행으로 세우지 않는다. 세우면 옆 차로에서
        # 영원히 서고 놓침도 성공도 안 나온다 (차선변경 없는 진단의 정보 손실).
        if ns is not None and st is not None and abs(env._stop_obs()[1]) < 0.26:
            d_end = float(ns[0])
            # 정류장 끝 8 m 앞(구간 안)을 정지 목표로 둔다. 입구에 s0=2 m
            # 여유를 두면 start 앞에서 평형이 되어 10초가 안 쌓인다.
            target = d_end - 8.0
            if 0.2 < target < 80.0 and target < gap:
                gap, v_lead = target, 0.0
        return gap, v_lead
