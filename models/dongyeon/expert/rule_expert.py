"""모방학습 선생: 규칙 운전자 (강동연). 환경 내부 상태(traci)를 직접 보는 "특권 정보" 선생이다.

학생(GRU 정책)은 관측 67차원만 보고, 선생은 SUMO 내부 값을 본다. Learning by Cheating(Chen et al.,
CoRL 2019)처럼 특권 선생 → 관측만 보는 학생으로 옮기는 구조다. 시험(평가)에서는 쓰지 않는다
(env-v1 원칙: 환경 안 규칙 없음. 선생은 학습 데이터를 만들 때만 쓴다).

규칙 (우선순위 낮은 것 → 높은 것, 목표 속도의 최솟값을 고른다):
  1) 기본: 최고속도(EGO max_speed), 어린이보호구역이면 그 제한속도
  2) 정류장 150 m 안: 정류장 차로로 (옆 차로 8 m 안에 차가 없을 때만), 정류장 끝 4 m 앞에 서도록 감속,
     정차 중이면 브레이크 (환경 dwell이 끝날 때까지)
  3) 신호 적·황: 정지선 3 m 앞에 서도록 감속 (황색은 2.5 m/s²로 못 서면 통과)
  4) 횡단 보행자 대기: 정지
  5) 앞차가 서 있고(불법주정차 등) 30 m 안이면 옆 차로로 추월 (적색 대기 줄이면 기다림)
  6) 앞차 안전거리: v² ≤ v_lead² + 2·1.2·(gap − 6)

행동 형식은 환경과 같다: [가감속 −1~1, 차선 −1/0/+1] (차선 +1 = 왼쪽, SUMO 차로 번호가 커지는 쪽).
"""
import numpy as np
import traci


class RuleExpert:
    """predict(obs, deterministic) 인터페이스 (공통 평가 스크립트 policy factory와 같음)."""

    version = "rule_expert_v4"   # v4: 옆길 합류 차 양보, 차선변경 간격 직접 측정, 불법주정차 미리 비키기

    def __init__(self, env, stop_lookahead=150.0, lane_gap=8.0, follow_decel=1.2, stop_decel=1.0,
                 signal_decel=1.2, yellow_decel=2.5, overtake_range=30.0):
        self.env = env
        self.stop_lookahead = stop_lookahead
        self.lane_gap = lane_gap
        self.follow_decel = follow_decel
        self.stop_decel = stop_decel
        self.signal_decel = signal_decel
        self.yellow_decel = yellow_decel
        self.overtake_range = overtake_range
        self.side_prefixes = ("side_", "through_merge_")
        self.parked_lookahead = 80.0
        self.yield_range, self.yield_lateral, self.yield_gap, self.yield_min_speed = 30.0, 10.0, 6.0, 1.0
        self.last_reason = ""

    # 공통 평가 스크립트가 판 시작을 알려 주지 않아도 되도록 상태를 두지 않는다.
    def reset(self):
        self.last_reason = ""

    def _side_clear(self, ego, direction, parked_overtake=False):
        """direction 쪽 옆 차로로 지금 옮겨도 되는가.

        기본: traci couldChangeLane(SUMO 판단) + 옆 차로 앞뒤 lane_gap m 안에 차 없음.
        parked_overtake: 서 있는 불법주정차 뒤에서 천천히 추월할 때. couldChangeLane은 ego 차선변경
          모드(0)·경로 차로 선호 때문에 빈 차로에도 False를 내는 경우가 있어(6단계 주교 앞에서 확인)
          대신 차로 존재 + 옆 차로 뒤 차가 오는 속도까지 본 더 큰 간격으로 직접 판단한다.
        """
        edge = traci.vehicle.getRoadID(ego)
        if edge.startswith(":"):
            return False
        if not self.env._is_car_lane(edge, traci.vehicle.getLaneIndex(ego) + direction):
            return False
        if not parked_overtake and not traci.vehicle.couldChangeLane(ego, direction):
            return False
        # 목표 차로의 차들을 직접 읽어 앞뒤 간격을 잰다 (getNeighbors는 겹친 차를 놓치는 경우가 있었다)
        target_lane = f"{edge}_{traci.vehicle.getLaneIndex(ego) + direction}"
        v = traci.vehicle.getSpeed(ego)
        pos = traci.vehicle.getLanePosition(ego)
        length = traci.vehicle.getLength(ego)
        for vid in traci.lane.getLastStepVehicleIDs(target_lane):
            p = traci.vehicle.getLanePosition(vid)
            if p >= pos:     # 앞 차: 그 차 꽁무니 − 내 앞머리
                if p - traci.vehicle.getLength(vid) - pos < self.lane_gap:
                    return False
            else:            # 뒤 차: 내 꽁무니 − 그 차 앞머리, 다가오는 속도만큼 여유
                need = self.lane_gap + (3.0 if parked_overtake else 1.5) * max(0.0, traci.vehicle.getSpeed(vid) - v)
                if (pos - length) - p < need:
                    return False
        # 버스 꽁무니가 아직 앞 엣지(교차로)에 걸쳐 있으면 그쪽 뒤 차를 못 보므로 바꾸지 않음
        if pos - length < 0.0:
            return False
        return True

    def _parked_ahead(self, ego):
        """경로 앞 lookahead m 안에, 버스와 같은 차로 번호(오른쪽부터)로 서 있는 불법주정차가 있는가.

        6단계 주교(e218924602_4, 34 m)는 교차로 바로 뒤 엣지 오른쪽 차로를 막는다. 교차로 안에서는
        차선을 못 바꾸므로 교차로에 들어가기 전에 미리 비켜야 한다 (선생의 특권 정보: 앞 엣지의 주정차 위치).
        """
        route = traci.vehicle.getRoute(ego)
        idx = traci.vehicle.getRouteIndex(ego)
        ahead_edges = set(route[idx:])
        my_lane = self.env._car_lane_index(ego)
        for vid in traci.vehicle.getIDList():
            if not vid.startswith("hs_"):
                continue
            road = traci.vehicle.getRoadID(vid)
            if road not in ahead_edges or road.startswith(":"):
                continue
            try:
                d = traci.vehicle.getDrivingDistance(ego, road, traci.vehicle.getLanePosition(vid))
            except traci.TraCIException:
                continue
            if 0.0 < d < self.parked_lookahead and self.env._car_lane_index(vid) == my_lane:
                return True
        return False

    def _yield_side(self, ego):
        """옆길에서 합류 중인 차(경로 밖·교차로 안)가 버스 진행 방향 앞에 있으면 그 앞에 서는 속도."""
        route = set(traci.vehicle.getRoute(ego))
        ex, ey = traci.vehicle.getPosition(ego)
        ang = np.radians(90.0 - traci.vehicle.getAngle(ego))   # SUMO 각도: 북 0°, 시계 방향
        hx, hy = np.cos(ang), np.sin(ang)
        best = None
        for vid in traci.vehicle.getIDList():
            if not vid.startswith(self.side_prefixes):
                continue
            road = traci.vehicle.getRoadID(vid)
            if road in route:            # 이미 버스 경로에 올라온 차는 앞차 규칙이 맡는다
                continue
            # 옆길에서 서서 기다리는 차는 무시: 교차로 안에 들어왔거나 움직이며 다가오는 차만
            if not road.startswith(":") and traci.vehicle.getSpeed(vid) < self.yield_min_speed:
                continue
            x, y = traci.vehicle.getPosition(vid)
            dx, dy = x - ex, y - ey
            ahead = dx * hx + dy * hy           # 진행 방향 거리
            lateral = abs(-dx * hy + dy * hx)   # 옆 거리
            if 0.0 < ahead < self.yield_range and lateral < self.yield_lateral:
                v = float(np.sqrt(max(0.0, 2 * self.follow_decel * max(0.0, ahead - self.yield_gap))))
                best = v if best is None else min(best, v)
        return best

    def act(self):
        env, ego = self.env, self.env.ego_id
        if ego not in traci.vehicle.getIDList():
            self.last_reason = "no_ego"
            return np.array([0.0, 0.0], dtype=np.float32)
        v = traci.vehicle.getSpeed(ego)
        edge = traci.vehicle.getRoadID(ego)
        vmax = float(getattr(env, "ego_vmax", env.vmax))
        lim = env._school_limits.get(edge)
        target = min(vmax, lim) if lim else vmax
        reason = "cruise" if not lim else "school_zone"
        lane = 0.0

        # 2) 정류장
        ns = env._next_stop()
        so = env._stop_obs()
        if ns is not None and 0 <= ns[0] <= self.stop_lookahead:
            if abs(so[1]) > 1e-6:
                dr = 1 if so[1] > 0 else -1
                if self._side_clear(ego, dr):
                    lane = float(dr)
            if so[2] > 0 or (env._in_stop_area(env._current_stop()) and v < 0.5):
                self.last_reason = "dwell"
                return np.array([-1.0, 0.0], dtype=np.float32)
            stop_v = float(np.sqrt(max(0.0, 2 * self.stop_decel * (ns[0] - 4.0))))
            if stop_v < target:
                target, reason = stop_v, "stop_approach"

        # 3) 신호
        sig = env._next_signal()
        if sig is not None:
            _, dist, state, _ = sig
            if env._SIGNAL_STOP_LEVEL.get(state, 1.0) > 0.0 and dist > -1:
                can_pass_yellow = state == "y" and v * v / (2 * self.yellow_decel) > dist
                if not can_pass_yellow:
                    sv = float(np.sqrt(max(0.0, 2 * self.signal_decel * (dist - 3.0))))
                    if sv < target:
                        target, reason = sv, "signal"

        # 4) 보행자
        if env._waiting_for_ped():
            target, reason = 0.0, "pedestrian"

        # 5-0) 앞 엣지 불법주정차를 교차로 전에 미리 비키기 (같은 차로면 옆 차로로)
        if lane == 0.0 and not edge.startswith(":") and self._parked_ahead(ego):
            for dr in (1, -1):
                if self._side_clear(ego, dr, parked_overtake=True):
                    lane, reason = float(dr), "avoid_parked"
                    break

        # 5) 서 있는 앞차 추월
        lead = traci.vehicle.getLeader(ego, 80.0)
        if (lead is not None and lane == 0.0 and lead[1] < self.overtake_range
                and traci.vehicle.getSpeed(lead[0]) < 0.1):
            red_ahead = (sig is not None and env._SIGNAL_STOP_LEVEL.get(sig[2], 1.0) > 0.0
                         and sig[1] < lead[1] + 30.0)
            parked = lead[0].startswith(("hs_", "park", "moto_park", "truck"))
            if parked or not red_ahead:
                for dr in (1, -1):
                    if self._side_clear(ego, dr, parked_overtake=parked and v < 3.0):
                        lane, reason = float(dr), "overtake"
                        break

        # 5-1) 옆길 합류 차 양보: 아직 버스 경로에 올라오지 않은(옆길·교차로 안) 끼어드는 차가
        #      버스 앞쪽 yield_range m 안에 있으면 그 차 앞 yield_gap m에 서도록 감속 (3단계 이상 충돌 원인)
        yv = self._yield_side(ego)
        if yv is not None and yv < target:
            target, reason = yv, "yield_side"

        # 6) 앞차 안전거리
        if lead is not None:
            lv = traci.vehicle.getSpeed(lead[0])
            safe = float(np.sqrt(max(0.0, lv * lv + 2 * self.follow_decel * max(0.0, lead[1] - 6.0))))
            if safe < target:
                target, reason = safe, "follow"

        acc = (target - v) / 0.5
        acc = acc / env.max_accel if acc >= 0 else acc / env.max_decel
        self.last_reason = reason
        return np.array([float(np.clip(acc, -1, 1)), lane], dtype=np.float32)

    def predict(self, obs, deterministic=True):
        return self.act()


def make_policy(checkpoint_path, env, config_path=None):
    """공통 평가 스크립트 policy factory. checkpoint_path는 선생 설명 파일(내용은 쓰지 않음)."""
    return RuleExpert(env)
