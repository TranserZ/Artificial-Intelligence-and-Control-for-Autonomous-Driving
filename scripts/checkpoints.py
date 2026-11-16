"""청계천로 편도(동행) 경로의 분석용 지점 정의 — 트래젝토리 수집/평가 스크립트 공용.

경로는 청계광장(0 m)에서 광장시장(2146 m)까지. 각 지점은 route 누적거리
(traci.vehicle.getDistance 기준, m)로 정의한다. F(어린이보호구역)만 구간이고
나머지는 통과 판정용 지점(그 거리 이상 도달했는가)이다.
"""

# (코드, 이름, 누적거리 m) — 통과 판정: 에피소드 중 odometer가 이 값 이상 도달했는가.
POINT_CHECKPOINTS = [
    ("A", "청계광장 출발", 25.0),
    ("B", "무교로 통과", 161.0),
    ("C", "광교 통과", 463.0),
    ("D", "삼일대로 통과", 911.0),
    ("E", "청계3가 통과", 1333.0),
    ("G", "세운상가 통과", 1610.0),
    ("H", "배오개다리 통과", 1824.0),
    ("I", "광장시장 완주", 2146.0),
]

# 구간 판정(F): 어린이보호구역. 통과 여부 + 구간 내 평균속도(제한 8.33 m/s 대비)를 본다.
SCHOOL_ZONE_CHECKPOINT = ("F", "어린이보호구역 통과", 1333.0, 1610.0)
SCHOOL_ZONE_SPEED_LIMIT = 8.33


class CheckpointTracker:
    """한 에피소드 동안 매 스텝 odometer(dist)를 넣어 지점 통과/구간 평균속도를 집계."""

    def __init__(self):
        self.passed = {code: False for code, _, _ in POINT_CHECKPOINTS}
        self.passed["F"] = False
        self._f_speeds = []

    def update(self, dist: float, speed: float):
        for code, _, pos in POINT_CHECKPOINTS:
            if dist >= pos:
                self.passed[code] = True
        lo, hi = SCHOOL_ZONE_CHECKPOINT[2], SCHOOL_ZONE_CHECKPOINT[3]
        if lo <= dist <= hi:
            self._f_speeds.append(speed)
        if dist >= hi:
            self.passed["F"] = True

    def school_zone_mean_speed(self):
        if not self._f_speeds:
            return float("nan")
        return sum(self._f_speeds) / len(self._f_speeds)

    def as_row(self) -> dict:
        row = {f"pass_{code}": int(self.passed[code]) for code, _, _ in POINT_CHECKPOINTS}
        row["pass_F"] = int(self.passed["F"])
        row["school_zone_mean_speed"] = self.school_zone_mean_speed()
        return row


ALL_CODES = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
NAMES = {code: name for code, name, *_ in POINT_CHECKPOINTS}
NAMES["F"] = SCHOOL_ZONE_CHECKPOINT[1]
