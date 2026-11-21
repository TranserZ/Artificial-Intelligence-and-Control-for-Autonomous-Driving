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

    version = "rule_expert_v3"

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
        self.last_reason = ""

    # 공통 평가 스크립트가 판 시작을 알려 주지 않아도 되도록 상태를 두지 않는다.
    def reset(self):
        self.last_reason = ""

    def _side_clear(self, ego, direction):
        """direction 쪽 옆 차로가 앞뒤 lane_gap m 안에 비어 있고 SUMO가 차로 변경 가능하다고 하는가."""
        if not traci.vehicle.couldChangeLane(ego, direction):
            return False
        # getNeighbors mode: bit0 = 왼쪽(1)/오른쪽(0), bit1 = 앞(1)/뒤(0)
        side = 1 if direction > 0 else 0
        nb = (traci.vehicle.getNeighbors(ego, side)              # 그 쪽 뒤
              + traci.vehicle.getNeighbors(ego, 0b10 | side))    # 그 쪽 앞
        return all(abs(g) > self.lane_gap for _, g in nb)

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

        # 5) 서 있는 앞차 추월
        lead = traci.vehicle.getLeader(ego, 80.0)
        if (lead is not None and lane == 0.0 and lead[1] < self.overtake_range
                and traci.vehicle.getSpeed(lead[0]) < 0.1):
            red_ahead = (sig is not None and env._SIGNAL_STOP_LEVEL.get(sig[2], 1.0) > 0.0
                         and sig[1] < lead[1] + 30.0)
            parked = lead[0].startswith(("hs_", "park", "moto_park", "truck"))
            if parked or not red_ahead:
                for dr in (1, -1):
                    if self._side_clear(ego, dr):
                        lane, reason = float(dr), "overtake"
                        break

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
