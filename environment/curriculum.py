"""청계천 단계별 맵 (커리큘럼): 단계 번호 → 켜지는 도로 요소.

뒤 단계는 앞 단계 요소를 모두 포함한다. 관측(67차원)은 모든 단계에서 같다 —
꺼진 요소는 관측에서 "없음" 값으로 나온다. 그래서 1단계에서 학습한 모델을
그대로 2단계로 이어서 학습할 수 있다.

    1 bus_stop         버스정류장 (정차 판정, 승객: 탑승 → 2정거장 뒤 하차)
    2 signal           신호등 (실측 주기) + 신호 횡단보도 + 횡단 보행자
    3 cutin            끼어들기 (측면 도로 합류 cutin_car)
    4 school_zone      어린이보호구역 (제한속도 20 km/h)
    5 motorcycle       오토바이 (차선변경이 일반 차량의 약 1.5배) + 노상 주차 오토바이
    6 illegal_parking  불법주정차 (차로를 막는 다발지점 4곳)

일반 차량(본선·반대 방향·프리필)은 1단계부터 항상 있다.
신호 없는 횡단보도와 자전거는 모든 단계에서 없다.

단계 정하기 (뒤가 앞을 덮어쓴다):
    env/road_config.py CHEONGGYECHEON["stage"]  →  환경변수 CGC_STAGE
"""
import os

STAGE_ORDER = ("bus_stop", "signal", "cutin", "school_zone", "motorcycle", "illegal_parking")
MAX_STAGE = len(STAGE_ORDER)


def resolve_stage(default=MAX_STAGE) -> int:
    """환경변수 CGC_STAGE가 있으면 그 값, 없으면 default. 1~MAX_STAGE로 검사한다."""
    raw = os.environ.get("CGC_STAGE")
    stage = int(raw) if raw not in (None, "") else int(default)
    if not 1 <= stage <= MAX_STAGE:
        raise ValueError(f"단계는 1~{MAX_STAGE}: {stage}")
    return stage


def elements(stage: int) -> tuple:
    """그 단계에서 켜지는 요소 이름들 (STAGE_ORDER 앞에서부터 stage개)."""
    return STAGE_ORDER[:int(stage)]


def road_summary(road: dict) -> dict:
    """결과 파일(config.json, 평가 json)에 남길 도로 단계 정보.

    CGC_STAGE 환경변수로 바꾼 값도 그대로 남는다 (road_config.py 스냅샷에는 기본값만 있음).
    """
    return {
        "layout": road.get("layout"),
        "stage": road.get("stage"),
        "elements": list(road.get("elements") or []),
        "through_vph": road.get("through_vph"),
        "vehicle_max_speed": road.get("vehicle_max_speed"),
    }


def has(road_or_elements, name: str) -> bool:
    """ROAD 설정(dict, "elements" 키) 또는 요소 목록에 name이 켜져 있는가.

    "elements"가 없는 설정(청계천 외 맵, 예전 설정)은 모든 요소가 켜진 것으로 본다.
    """
    if isinstance(road_or_elements, dict):
        els = road_or_elements.get("elements")
        if els is None:
            return True
    else:
        els = road_or_elements
    return name in els
