"""보행자 안전 필터 — PPO의 가감속 명령을 적용 직전에 검사해 필요하면 제동으로 바꾼다.

횡단보도와 무관하게 주변 보행자를 본다.
1) 내 경로를 따라 앞쪽 길을 점들로 펼친다.
2) 보행자가 지금 방향·속도로 곧 내 길 위로 오는지 본다.
   (횡단보도에 들어오려는 사람, 보행 녹색일 때 입구 주변에서 움직이는 사람도 포함)
3) 오면 정지 지점을 잡는다. 교차로 전이면 정지선, 횡단보도 안이면 가장자리 앞,
   그 밖에는 보행자 BUFFER m 앞.
4) 그 지점에 멈출 수 있는 속도보다 PPO 명령이 빠를 때만 제동으로 덮어쓴다.

신호와 앞차는 보지 않는다 (PPO가 학습). 브레이크만 밟을 수 있다.
켜고 끄기: mdp_config.SIMULATION["ped_filter"], test.py --no-filter
"""
import numpy as np
import traci

HORIZON, DT_PRED = 3.0, 0.5
HORIZON_ENTRY = 6.0   # 정지선 앞: 교차로를 빠져나갈 시간까지 본다
HALF_WIDTH = 1.5      # 차 폭 절반 0.9 + 여유 0.6. 차로 중심에서 보도까지 약 1.6 m라 보도는 제외
BUFFER = 2.0          # 보행자 앞 정지 여유 (m)
LOOKAHEAD = 45.0      # 앞쪽 길을 볼 거리 (m)
NEAR = 2.5            # "바로 앞" 범위 (m)
ENTRY_RADIUS = 6.0    # 횡단보도 입구 주변 반경 (m). 1.3 m/s면 4.6초 거리
HOLD = 2.0            # 보행자가 없어져도 더 기다리는 시간 (s). 잠깐 난 틈에 출발하면 다음 사람과 겹친다
CREEP = 2.0           # 횡단보도에 걸쳤을 때 빠져나가는 속도 (m/s)
BRAKE = 3.0           # 평소 감속 (뒤차 추돌 방지). 더 필요하면 최대 감속까지 clip으로 허용
REACT = 0.5           # 반응 시간 (s) = 1스텝


class PedSafetyFilter:
    def __init__(self, env):
        self.env = env
        self.conflict_now = False       # 이번 스텝에 멈춰야 할 보행자가 있는가
        self._past_line_blocked = False
        self._hold_until = 0.0
        self._chains = {}

    def _chain(self, lane):
        """lane에서 시작해 경로를 따라가는 차선 id 목록."""
        if lane in self._chains:
            return self._chains[lane]
        chain, total, cur = [lane], traci.lane.getLength(lane), lane
        while total < LOOKAHEAD + 60:
            if cur.startswith(":"):
                links = traci.lane.getLinks(cur)
                if not links:
                    break
                cur = links[0][0]
            else:
                nxt = self.env._lane_route_link(cur)
                if nxt is None:
                    break
                cur = nxt[4] or nxt[0]
            if cur in chain:
                break
            chain.append(cur)
            total += traci.lane.getLength(cur)
        self._chains[lane] = chain
        return chain

    def _path(self, ego):
        """앞범퍼부터 앞쪽 길의 점 (x, y, 거리), 정지선까지 거리 (교차로 안이면 None)."""
        lane = traci.vehicle.getLaneID(ego)
        off = -traci.vehicle.getLanePosition(ego)
        pts, stopline = [], None
        for l in self._chain(lane):
            shape = np.array(traci.lane.getShape(l), dtype=float)
            seg = np.hypot(*np.diff(shape, axis=0).T)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            length = traci.lane.getLength(l)
            scale = length / max(cum[-1], 1e-6)
            s = np.arange(0.0, cum[-1], 1.0)
            pts.append(np.c_[np.interp(s, cum, shape[:, 0]),
                             np.interp(s, cum, shape[:, 1]), off + s * scale])
            off += length
            if stopline is None and not l.startswith(":"):
                nxt = self.env._lane_route_link(l)
                # 정지선 = 신호 교차로나 횡단보도가 있는 노드 앞. 엣지를 나누기만 한 지점
                # (좌회전 포켓 시작, 어린이 보호구역 경계)도 내부 차로가 있지만 정지선이 아니다.
                if nxt is not None and nxt[4].startswith(":") and self.env.is_stop_junction(nxt[4]):
                    stopline = off
            if off > LOOKAHEAD:
                break
        p = np.vstack(pts)
        return p[(p[:, 2] >= 0.0) & (p[:, 2] <= LOOKAHEAD)], stopline

    def _stop_distance(self, ego):
        """멈춰야 할 지점까지 남은 거리 (멈출 이유가 없으면 None)."""
        self._past_line_blocked = False
        path, stopline = self._path(ego)
        if len(path) == 0:
            return None
        before_line = stopline is not None
        times = np.arange(0.0, (HORIZON_ENTRY if before_line else HORIZON) + 1e-6, DT_PRED)
        ex, ey = traci.vehicle.getPosition(ego)
        persons = traci.person.getIDList()
        best, near_now = None, False
        for pid in persons:
            x, y = traci.person.getPosition(pid)
            if np.hypot(x - ex, y - ey) > LOOKAHEAD + 5:
                continue
            a = np.radians(traci.person.getAngle(pid))   # SUMO 각도: 북=0, 시계방향
            v = traci.person.getSpeed(pid)
            fut = np.c_[x + np.sin(a) * v * times, y + np.cos(a) * v * times]
            d = np.hypot(fut[:, None, 0] - path[:, 0], fut[:, None, 1] - path[:, 1])
            hit = d <= HALF_WIDTH
            if hit.any():
                s = float(path[hit.any(axis=0), 2].min())
                best = s if best is None else min(best, s)
                if (d[0, path[:, 2] <= NEAR] <= HALF_WIDTH).any():
                    near_now = True

        # 횡단보도로 들어오려는 보행자 (모퉁이를 도는 중이면 지금 방향으로는 예측이 안 된다)
        odo = traci.vehicle.getDistance(ego)
        margin = float(self.env.rw.get("yield_margin", 3.0)) + 1.0
        for c in self.env._route_crossings:
            ahead = c["s"] - odo
            if not 0 <= ahead <= LOOKAHEAD:
                continue
            gap, toward = self.env._crossing_ped_state(c)
            if gap is not None and gap <= margin and toward:
                best = ahead if best is None else min(best, ahead)
            # 출발 판단: 보행 녹색이면 횡단보도 입구 주변에서 움직이는 사람도 기다린다
            if before_line and c["link"] is not None:
                tl, idx = c["link"]
                if traci.trafficlight.getRedYellowGreenState(tl)[idx] in "Gg":
                    k = int(np.argmin(np.abs(path[:, 2] - ahead)))
                    for pid in persons:
                        x, y = traci.person.getPosition(pid)
                        if (np.hypot(x - path[k, 0], y - path[k, 1]) <= ENTRY_RADIUS
                                and traci.person.getSpeed(pid) >= 0.3):
                            best = ahead if best is None else min(best, ahead)
                            break

        now = traci.simulation.getTime()
        if now < self._hold_until - HOLD - 0.01:
            self._hold_until = 0.0            # 새 에피소드 (시간이 0부터 다시 시작)
        if before_line and best is not None and stopline <= best:
            self._hold_until = now + HOLD
        if before_line and best is None and now < self._hold_until and stopline <= 10.0:
            return max(stopline - 0.5, 0.0)
        if best is None:
            return None
        if before_line and stopline <= best:
            return max(stopline - 0.5, 0.0)   # 교차로 전: 정지선에서 멈춘다
        # 충돌 지점이 횡단보도 안이면 가장자리 앞에서 멈춘다
        for c in self.env._route_crossings:
            ahead = c["s"] - odo
            if 0 <= ahead <= best <= ahead + c["width"]:
                return max(ahead - 1.0, 0.0)
        if not before_line and best < BUFFER and not near_now:
            self._past_line_blocked = True    # 이미 걸침 + 바로 앞은 비었음
        return max(best - BUFFER, 0.0)

    def apply(self, accel_raw, ego):
        """(적용할 가감속 raw, 덮어썼는가)를 반환한다."""
        env = self.env
        d = self._stop_distance(ego)
        self.conflict_now = d is not None
        if d is None:
            return accel_raw, False
        if self._past_line_blocked:
            target = CREEP
        elif d < 0.3:
            target = 0.0
        else:
            target = float(-BRAKE * REACT + np.sqrt((BRAKE * REACT) ** 2 + 2 * BRAKE * d))
        v = traci.vehicle.getSpeed(ego)
        cmd = accel_raw * (env.max_accel if accel_raw >= 0 else env.max_decel)
        if v + cmd * env.step_length <= target:
            return accel_raw, False
        dv = (target - v) / env.step_length
        raw = dv / env.max_accel if dv >= 0 else dv / env.max_decel
        return float(np.clip(raw, -1.0, 1.0)), True
