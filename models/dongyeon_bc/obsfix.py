"""옆 차로 관측 보강 (강동연, 실험용 — env 파일은 고치지 않고 메서드만 덮어쓴다).

왜: 6단계 학생 충돌 26/30이 "주정차 차 뒤에서 차선변경 → 옆 차로 차와 측면 충돌".
옆 차로(±1) 관측은 traci getNeighbors만 쓰고 gap < 0(나란히 겹친 차)은 버린다.
버스가 서 있을 때 차선 직접 스캔과 비교하면 앞차의 약 79%, 뒷차의 약 41%를 관측이 놓쳤다
(probe: 6단계 3판 정지 735 step). 차선변경 게이트(_lane_change_is_safe)는 이미 스캔을 함께 보므로
선생님은 보고 학생은 못 보는 차가 생긴다 → 흉내만으로는 배울 수 없다.

보강: ±1 차로는 getNeighbors와 차선 스캔 중 가까운 쪽, 겹친 차(gap < 0)는 gap 0으로 보인다.
    OBS_SCAN=1 이면 teacher56.py를 import할 때 적용된다.
"""
from env.sumo_env import SumoHighwayEnv

_orig_neighbor = SumoHighwayEnv._neighbor


def _neighbor(self, offset, leader):
    if abs(offset) != 1:
        return _orig_neighbor(self, offset, leader)
    n = _orig_neighbor(self, offset, leader)
    front, rear = self._scan_lane_neighbors(offset)
    s = front if leader else rear
    if s is not None:
        gap = max(float(s[0]), 0.0)
        if gap <= self.visibility and (n is None or gap < n[0]):
            n = (gap, float(s[1]))
    return n


def apply():
    SumoHighwayEnv._neighbor = _neighbor
