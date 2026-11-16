"""청계천로 SUMO 시나리오 생성: 경로·교통 흐름·GUI 설정·sumocfg.

전제: tools/osm_to_sumo.py 로 만든 nod/edg + netconvert 로 만든 cgc.net.xml.

버스 노선·정류장은 config/lines.yaml 에서 읽는다 (없으면 내장 기본값 = 청계A01).
  - 노선(lines)은 정류장 id 순서만 적는다. 경로는 정류장 순서대로 자동 계산된다.
  - 노선별 정차 순서는 cgc.line_meta.json 으로 함께 저장한다 (sumo_env가 읽음).

실제 상황 → 시뮬레이션 대응 (근거는 docs/cheonggyecheon_research.md)
  자율주행 셔틀(청계A01)         → 빨간 ego_type, 청계광장→광장시장 동행 경로, 시속 20km급
  끼어드는 차 (접촉 사고 원인)    → ① 측면 도로에서 본선으로 합류하는 차량
                                  ② 본선 2개 차로 사이를 오가는 차량(차선변경 허용)
  불법주정차·정차 차량            → 본선 우측 차로(0번)에 잠깐 서는 차량 → 뒤차가 옆 차로로 끼어듦
  오토바이                        → 폭이 좁고 공격적인 별도 차종
  일반 차량 controller 4종 혼합   → Krauss/IDM/EIDM/ACC (수업 기본 설정과 동일 비율)
"""
import argparse
import csv
import heapq
import json
import math
import os
import sys
import random

SUMO_HOME = os.environ.get("SUMO_HOME", os.path.expanduser("~/src/sumo-1.27.1-mesa"))
sys.path.append(os.path.join(SUMO_HOME, "tools"))
import sumolib  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NO_STOP_CSV = os.path.join(ROOT, "data", "road_data", "no_stop_zones.csv")

START_EDGE = "e73541403_1"      # 청계광장 (동행 첫 구간)
END_EDGE = "e1212424783_4"      # 청계5가·광장시장 부근 (동행 마지막 구간)

# 로드뷰에서 오토바이가 서 있던 구간. 광장시장 쪽이 가장 밀집.
MOTO_PARK_EDGES = [
    "e73541403_2", "e73541403_6",
    "e218924602_6", "e1212424783_1", "e1212424783_2",
    "e1212424783_3", "e1212424783_4",
]

# 가게 앞 배달 트럭이 하역 중인 구간과 위치(엣지 길이 기준 비율). 1~2대씩.
TRUCK_PARK_SPOTS = {
    "e218924602_5": (0.35, 0.65),
    "e1212424783_2": (0.5,),
    "e1212424783_3": (0.5,),
}


def passenger_lane_indexes(edge):
    """승용차가 달릴 수 있는 차로. 보도가 0번이면 오른쪽 승용차는 1번."""
    return [ln.getIndex() for ln in edge.getLanes() if ln.allows("passenger")]


def load_no_stop_zones(path=NO_STOP_CSV):
    """정차금지지대 (edge_id, start_pos, end_pos). 파일이 없으면 빈 목록."""
    if not os.path.exists(path):
        return []
    zones = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            zones.append((row["edge_id"].strip(), float(row["start_pos"]), float(row["end_pos"])))
    return zones


def in_no_stop(zones, edge_id, pos):
    return any(eid == edge_id and start <= pos <= end for eid, start, end in zones)


def in_excluded_zone(no_stop_zones, bus_zones, edge_id, pos, margin=2.0):
    """정차금지지대이거나, 정류장 정차 구간(margin 포함) 안이면 True."""
    if in_no_stop(no_stop_zones, edge_id, pos):
        return True
    return any(start - margin <= pos <= end + margin
               for start, end in bus_zones.get(edge_id, []))


def fit_bus_stop(length, start, end):
    """정류장 endPos가 엣지보다 길면 끝에 맞춘다.

    15 m 창이 엣지 안에 안 들어가면 끝에서 15 m(엣지가 더 짧으면 처음부터)로 당긴다.
    """
    if length <= 0.2:
        return 0.0, round(max(length - 0.01, 0.1), 2)
    end = min(float(end), length - 0.1)
    start = min(float(start), end - 0.1)
    if end - start < 8.0:
        start = max(0.0, end - 15.0)
    return round(start, 1), round(end, 1)


def place_bus_stop(net, chains, eid, start, end, min_room=20.0):
    """정류장 좌표가 엣지보다 길면 경로상 다음 엣지로 넘긴다 (fit_bus_stop 전에).

    교차로가 합쳐지며 엣지가 짧아지면(예: 관수교 e218924602_1 14.8 m) 원래 좌표(39~54 m)가
    엣지 밖으로 나간다. 그때 12 m 버스가 서기 어려운 엣지 끝에 욱여넣지 않고, 같은 노선의
    다음 엣지로 남은 거리만큼 옮긴다. chains: 노선 엣지 목록들 (동행, 서행).
    """
    for chain in chains:
        ids = [e.getID() for e in chain]
        if eid not in ids:
            continue
        k = ids.index(eid)
        while k + 1 < len(chain) and start > chain[k].getLength() - min_room:
            start -= chain[k].getLength()
            end -= chain[k].getLength()
            k += 1
        start = max(start, 0.0)
        return chain[k].getID(), start, max(end, start + 15.0)
    return eid, start, end


def bus_stop_lane_index(edge):
    """보도가 lane 0이면 정류장은 승용차 차로 1. 아니면 0."""
    lane0 = edge.getLane(0)
    if lane0.allows("pedestrian") and not lane0.allows("passenger") and edge.getLaneNumber() > 1:
        return 1
    return 0


def _pos_dir(shape, s):
    """차로 중심선(shape)을 따라 s(m) 지점의 (x, y, 진행 단위벡터)."""
    acc = 0.0
    for (x1, y1), (x2, y2) in zip(shape, shape[1:]):
        seg = math.hypot(x2 - x1, y2 - y1)
        if seg > 0 and acc + seg >= s:
            t = (s - acc) / seg
            return x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, (x2 - x1) / seg, (y2 - y1) / seg
        acc += seg
    (x1, y1), (x2, y2) = shape[-2], shape[-1]
    seg = math.hypot(x2 - x1, y2 - y1) or 1.0
    return x2, y2, (x2 - x1) / seg, (y2 - y1) / seg


def _rect(cx, cy, dx, dy, nx, ny, half_l, half_w):
    """중심 (cx,cy), 도로방향 (dx,dy), 보도방향 (nx,ny)인 직사각형 shape 문자열."""
    pts = [(cx - dx * half_l - nx * half_w, cy - dy * half_l - ny * half_w),
           (cx + dx * half_l - nx * half_w, cy + dy * half_l - ny * half_w),
           (cx + dx * half_l + nx * half_w, cy + dy * half_l + ny * half_w),
           (cx - dx * half_l + nx * half_w, cy - dy * half_l + ny * half_w)]
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)


# 정류장 시안 (사용자 제작 지시서, 2026-11) — 색과 치수
SHELTER = {
    "roof": "184,190,194",       # 지붕 상판 무광 밝은 회색 #B8BEC2
    "teal": "43,101,112",        # 도로 쪽 띠·표지판 청록 #2B6570
    "curb": "142,144,144",       # 연석 중간 회색 #8E9090
    "tactile": "210,174,88",     # 황색 포장 띠 #D2AE58
    "pole": "52,58,62",          # 기둥 차콜 #343A3E
    "shadow": "0,0,0,45",        # 지붕 그림자 (옅게)
    "curb_w": 0.15,              # 연석 폭 (SUMO는 연석을 안 그려서 직접 그린다)
    "tactile_from": 0.25,        # 연석 보도 쪽 모서리 → 황색 띠 시작
    "tactile_w": 0.30,           # 황색 띠 폭
    "roof_from": 0.80,           # 연석 보도 쪽 모서리 → 지붕 도로 쪽 끝
    # 시안은 4.80 × 1.55 m. 대기 승객 수에 비해 작아 보여서 길이 ×3, 폭 ×1.5
    # (사용자 요청 2026-11). 길이 14.40 m는 정차 구간(15 m)에 꼭 맞는다.
    "roof_len": 14.40,           # 지붕 길이 (연석과 평행)
    "roof_depth": 2.30,          # 지붕 폭
    "teal_edge": 0.12,           # 탑뷰에서 보이는 청록 테두리 폭
    "sign_gap": 0.55,            # 쉘터 오른쪽 끝(도로에서 봤을 때) → 표지판
    "sign_in": 0.40,             # 연석 보도 쪽 모서리 → 표지판 기둥 중심
    "sign_plate": (0.40, 0.06),  # 표지판 판 (폭, 두께) — 면이 도로를 향함
    "icon": 0.70,                # 지도 기호 (청록 바탕 흰 버스) 한 변
}


# 정류장별 구분 색 — sumo-gui 정류장 테두리, 대기 현황 표의 이름, 지도 점에 같은 색.
# 잔디(초록)·보도(회색) 위에서도, 흰 바탕 표의 글자로도 잘 보이는 색으로 골랐다.
# 정류장 순서(lines.yaml stops 순서)대로 돌아가며 쓰고, stops.<id>.color로 바꿀 수 있다.
STOP_PALETTE = ["#D7263D", "#F46036", "#B8860B", "#8C5A2B", "#C2187A", "#7B3FBF",
                "#283593", "#1B65C9", "#00838F", "#E05297", "#2E7D32", "#546E7A"]
STOP_FRAME = {
    "line_w": 0.40,              # 테두리 선 폭 (m)
    "fill_alpha": 50,            # 안쪽 옅은 색면 투명도 (0~255)
    "pad_s": (1.0, 0.8),         # 표지판 앞 / 지붕 끝 뒤로 더 두는 여유 (m)
    "pad_road": 0.35,            # 차도 쪽으로 넘어가는 폭 (m) — 버스가 서는 곳까지
    "pad_back": 0.50,            # 지붕 뒤로 더 두는 여유 (m)
}


def _hex_rgb(h):
    h = h.lstrip("#")
    return ",".join(str(int(h[i:i + 2], 16)) for i in (0, 2, 4))


def _bus_icon(sid, cx, cy, size, layer):
    """청록 바탕 정사각형 + 흰 버스 정면 픽토그램 (화면 기준 똑바로 선 지도 기호)."""
    S = SHELTER
    h = size / 2

    def box(name, x0, y0, x1, y1, color, lay):
        # 아이콘 좌표 (-1~1, 위가 +y) → 세계 좌표
        pts = [(cx + x0 * h, cy + y0 * h), (cx + x1 * h, cy + y0 * h),
               (cx + x1 * h, cy + y1 * h), (cx + x0 * h, cy + y1 * h)]
        shape = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        return (f'    <poly id="bsi_{sid}_{name}" color="{color}" fill="1" '
                f'layer="{lay}" shape="{shape}"/>')

    w = "255,255,255"
    return [
        box("bg", -1.00, -1.00, 1.00, 1.00, S["teal"], layer),
        box("body", -0.56, -0.52, 0.56, 0.72, w, layer + 0.1),       # 버스 몸통
        box("roof", -0.44, 0.72, 0.44, 0.80, w, layer + 0.1),        # 지붕 둥근 느낌
        box("glass", -0.44, 0.08, 0.44, 0.58, S["teal"], layer + 0.2),  # 앞유리
        box("lampL", -0.42, -0.30, -0.24, -0.14, S["teal"], layer + 0.2),
        box("lampR", 0.24, -0.30, 0.42, -0.14, S["teal"], layer + 0.2),
        box("wheelL", -0.50, -0.78, -0.26, -0.52, w, layer + 0.1),   # 바퀴
        box("wheelR", 0.26, -0.78, 0.50, -0.52, w, layer + 0.1),
    ]


def stop_shelter_polys(edge, lane_idx, sid, start_pos, end_pos, name, lines,
                       color="#2B6570", number=None):
    """서울 도심 가로변 버스정류장 (사용자 시안·제작 지시서)을 탑뷰로 그린다.

    차도 → 보도 방향으로: 아스팔트 → 회색 연석 → (0.25 m) → 황색 포장 띠 0.30 m
    → 보도 여백 → 청록 테두리가 있는 밝은 회색 지붕 4.80 × 1.55 m.
    도로에서 쉘터를 봤을 때 오른쪽 끝(= 버스가 들어오는 쪽, 상류)에서 0.55 m
    바깥에 독립형 표지판: 얇은 청록 판 + 청록 바탕 흰 버스 지도 기호.
    유리·벤치·기둥은 지붕 아래라 탑뷰에서 보이지 않으므로 그리지 않는다.
    정류장 이름은 지붕 위 라벨(POI type)로 — sumo-gui 글꼴에 한글이 없어 로마자.
    정류장 구역 테두리: 정류장마다 다른 색(color)의 사각 테두리 + 옅은 색면으로
    멀리서도 정류장을 찾고 서로 구분할 수 있게 한다 (대기 현황 표·지도와 같은 색).
    전부 그림일 뿐 시뮬레이션 동작에는 영향이 없다.

    층(layer): 차로 면이 layer 20 미만 폴리곤을 덮으므로 21 이상에 그린다.
    (노면 지그재그는 없음 — 실제 청계천로 정류장 위성사진에 없다)
    """
    S = SHELTER
    shape = edge.getLane(lane_idx).getShape()
    curb_in = edge.getLane(lane_idx).getWidth() / 2 + S["curb_w"]   # 차로 중심 → 연석 보도 쪽 모서리

    def strip(pid, color, s0, s1, off0, off1, layer):
        """차로를 따라 s0~s1, 차로 중심에서 보도 쪽으로 off0~off1 m인 띠.
        차로가 휘어도 따라가도록 1 m 간격으로 점을 찍는다."""
        n = max(1, int(math.ceil(abs(s1 - s0))))
        near, far = [], []
        for i in range(n + 1):
            px, py, dx, dy = _pos_dir(shape, max(s0 + (s1 - s0) * i / n, 0.05))
            nx, ny = dy, -dx                                # 진행방향 오른쪽 = 보도 쪽
            near.append((px + nx * off0, py + ny * off0))
            far.append((px + nx * off1, py + ny * off1))
        pts = near + far[::-1]
        return (f'    <poly id="{pid}" color="{color}" fill="1" layer="{layer}" '
                f'shape="{" ".join(f"{x:.2f},{y:.2f}" for x, y in pts)}"/>')

    # 쉘터 위치: 버스 앞머리가 서는 end_pos 바로 뒤에 지붕 (버스 출입문 옆)
    roof_s1 = end_pos - 0.2
    roof_s0 = roof_s1 - S["roof_len"]
    r0 = curb_in + S["roof_from"]
    r1 = r0 + S["roof_depth"]
    t0 = curb_in + S["tactile_from"]
    sign_s = roof_s0 - S["sign_gap"]                       # 상류 쪽 = 도로에서 봤을 때 오른쪽
    P = [
        # 연석: 정차 구간 + 표지판까지
        strip(f"bsc_{sid}", S["curb"], min(start_pos, sign_s - 0.3), end_pos,
              curb_in - S["curb_w"], curb_in, 21),
        # 황색 포장 띠: 쉘터 길이에 맞춰 끝낸다
        strip(f"bst_{sid}", S["tactile"], roof_s0, roof_s1, t0, t0 + S["tactile_w"], 21),
        # 지붕 그림자 (옅고 짧게, 뒤쪽으로 살짝)
        strip(f"bsh_{sid}", S["shadow"], roof_s0 + 0.15, roof_s1 + 0.15, r0 + 0.2, r1 + 0.2, 21.5),
        # 지붕 상판
        strip(f"bsr_{sid}", S["roof"], roof_s0, roof_s1, r0, r1, 22),
        # 도로 쪽 청록 테두리 (상판 가장자리 0.08 m)
        strip(f"bse_{sid}", S["teal"], roof_s0, roof_s1, r0, r0 + S["teal_edge"], 22.1),
    ]
    # 독립형 표지판: 기둥 + 도로를 향한 얇은 청록 판 + 지도 기호
    px, py, dx, dy = _pos_dir(shape, max(sign_s, 0.05))
    nx, ny = dy, -dx
    cx, cy = px + nx * (curb_in + S["sign_in"]), py + ny * (curb_in + S["sign_in"])
    pw, pt = S["sign_plate"]
    P.append(f'    <poly id="bss_{sid}" color="{S["teal"]}" fill="1" layer="23" '
             f'shape="{_rect(cx, cy, dx, dy, nx, ny, pw / 2, pt / 2)}"/>')
    P.append(f'    <poly id="bsp_{sid}" color="{S["pole"]}" fill="1" layer="23.1" '
             f'shape="{_rect(cx, cy, dx, dy, nx, ny, 0.035, 0.035)}"/>')
    P += _bus_icon(sid, cx, cy, S["icon"], 24)
    # 정류장 구역 테두리 (정류장별 색): 옅은 색면 + 사각 테두리.
    # 표지판부터 지붕 끝까지, 차도 가장자리(버스가 서는 곳)부터 지붕 뒤까지.
    F = STOP_FRAME
    rgb = _hex_rgb(color)
    fs0, fs1 = sign_s - F["pad_s"][0], roof_s1 + F["pad_s"][1]
    fo0, fo1 = curb_in - S["curb_w"] - F["pad_road"], r1 + F["pad_back"]
    lw = F["line_w"]
    P += [
        strip(f"bsf_{sid}_fill", f"{rgb},{F['fill_alpha']}", fs0, fs1, fo0, fo1, 20.5),
        strip(f"bsf_{sid}_n", f"{rgb},230", fs0, fs1, fo0, fo0 + lw, 24.5),       # 차도 쪽
        strip(f"bsf_{sid}_f", f"{rgb},230", fs0, fs1, fo1 - lw, fo1, 24.5),       # 뒤쪽
        strip(f"bsf_{sid}_a", f"{rgb},230", fs0, fs0 + lw, fo0, fo1, 24.5),       # 앞
        strip(f"bsf_{sid}_b", f"{rgb},230", fs1 - lw, fs1, fo0, fo1, 24.5),       # 끝
    ]
    # 정류장 이름: 지붕 한가운데 라벨 (POI 점 자체는 투명, type 글자만 보인다)
    mx, my, mdx, mdy = _pos_dir(shape, (roof_s0 + roof_s1) / 2)
    off = (r0 + r1) / 2
    short = " · ".join(sorted({ln.split("_")[0] for ln in lines})) or "-"
    num = f"{number}. " if number is not None else ""
    P.append(f'    <poi id="bsg_{sid}" x="{mx + mdy * off:.2f}" y="{my - mdx * off:.2f}" '
             f'color="0,0,0,0" layer="25" width="0.1" height="0.1" '
             f'type="{num}{name} | {short}"/>')
    return P


# ──────────────────────────────────────────────────────────────
# 버스 노선 설정 (config/lines.yaml). 파일이 없으면 아래 기본값(청계A01)을 쓴다.
#   stops: 정류장 id → {edge, start, end, name}  (실측 좌표, 자동 보정됨)
#   lines: 노선 이름 → {route_id, stops(정차 순서), start_edge, end_edge}
# ──────────────────────────────────────────────────────────────
LINES_YAML = os.path.join(ROOT, "config", "lines.yaml")

DEFAULT_LINE_CONFIG = {
    "passenger_rate": 0.01,   # 정류장별 승객 도착률 (1/s). 약 100초에 1명
    "stops": {
        "bs_01": {"edge": "e73541403_2", "start": 22, "end": 37, "name": "청계광장", "name_en": "Cheonggye Plaza"},
        "bs_02": {"edge": "e73541403_8", "start": 45, "end": 60, "name": "청계1가.광교", "name_en": "Gwanggyo"},
        "bs_03": {"edge": "e908646367_3", "start": 30, "end": 45, "name": "청계2가.삼일교(동)", "name_en": "Samilgyo (E)"},
        "bs_04": {"edge": "e218924602_1", "start": 39, "end": 54, "name": "청계3가.관수교(동)", "name_en": "Gwansugyo (E)"},
        "bs_05": {"edge": "e218924602_6", "start": 68, "end": 83, "name": "청계4가.배오개다리", "name_en": "Baeogae Bridge"},
        "bs_06": {"edge": "e1212424785_1", "start": 184, "end": 199, "name": "청계5가.광장시장", "name_en": "Gwangjang Market"},
        "bs_07": {"edge": "e732213463_1", "start": 113, "end": 128, "name": "청계4가.세운교", "name_en": "Sewoongyo"},
        "bs_08": {"edge": "e732213463_4", "start": 43, "end": 58, "name": "청계3가.관수교(서)", "name_en": "Gwansugyo (W)"},
        "bs_09": {"edge": "e1212424791_3", "start": 38, "end": 53, "name": "청계2가.수표교", "name_en": "Supyogyo"},
        "bs_10": {"edge": "e908694578_1", "start": 43, "end": 58, "name": "청계2가.삼일교(서)", "name_en": "Samilgyo (W)"},
        "bs_11": {"edge": "e908694578_3", "start": 66, "end": 81, "name": "청계1가.장통교", "name_en": "Jangtonggyo"},
    },
    "lines": {
        "A01_east": {"route_id": "r0",
                     "stops": ["bs_01", "bs_02", "bs_03", "bs_04", "bs_05"],
                     "start_edge": "e73541403_1", "end_edge": "e1212424783_4"},
        "A01_loop": {"route_id": "r0_loop",
                     "stops": ["bs_01", "bs_02", "bs_03", "bs_04", "bs_05", "bs_06",
                               "bs_07", "bs_08", "bs_09", "bs_10", "bs_11"],
                     "start_edge": "e73541403_1", "end_edge": "e183272596_1"},
    },
}


def load_line_config(path=None):
    """config/lines.yaml을 읽는다. 파일이 없으면 내장 기본값(청계A01)."""
    path = path or LINES_YAML
    if not os.path.exists(path):
        print(f"노선 설정 없음({path}) → 내장 기본값(청계A01) 사용")
        return DEFAULT_LINE_CONFIG
    try:
        import yaml
    except ImportError:
        raise SystemExit(f"{path}를 읽으려면 pyyaml이 필요합니다: pip install pyyaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("stops", "lines"):
        if not isinstance(cfg.get(key), dict) or not cfg[key]:
            raise SystemExit(f"{path}: '{key}' 항목이 비어 있거나 형식이 틀립니다")
    for lname, lc in cfg["lines"].items():
        if "route_id" not in lc or not lc.get("stops"):
            raise SystemExit(f"{path}: 노선 {lname}에 route_id/stops가 필요합니다")
        unknown = [s for s in lc["stops"] if s not in cfg["stops"]]
        if unknown:
            raise SystemExit(f"{path}: 노선 {lname}의 정류장 {unknown}이 stops에 없습니다")
    return cfg


def shortest(net, a, b):
    path, _ = net.getShortestPath(net.getEdge(a), net.getEdge(b))
    return path


def line_route(net, stop_edges, start_edge=None, end_edge=None,
               main_name="청계천로", off_main_factor=25.0):
    """정류장을 순서대로 지나는 노선 경로(엣지 목록)를 자동 계산한다.

    구간(정류장→다음 정류장)마다 최단경로를 찾되, 본선(main_name이 이름에 든
    엣지) 밖은 길이를 off_main_factor배로 쳐서 종로 쪽 지름길로 빠지지 않게
    한다. 회차 연결로처럼 본선 밖을 꼭 지나야 하는 구간에서만 밖으로 나간다.
    """
    def cost(e):
        return e.getLength() * (1.0 if main_name in (e.getName() or "") else off_main_factor)

    def seg(a, b):
        if a is b:
            return [a]
        dist, prev = {a: 0.0}, {}
        pq = [(0.0, a.getID(), a)]
        while pq:
            d, _, e = heapq.heappop(pq)
            if e is b:
                break
            if d > dist.get(e, float("inf")):
                continue
            for nxt in e.getOutgoing():
                if not (nxt.allows("bus") or nxt.allows("passenger")):
                    continue
                nd = d + cost(nxt)
                if nd < dist.get(nxt, float("inf")) - 1e-9:
                    dist[nxt] = nd
                    prev[nxt] = e
                    heapq.heappush(pq, (nd, nxt.getID(), nxt))
        if b not in prev:
            raise RuntimeError(f"경로 없음: {a.getID()} → {b.getID()}")
        path, e = [b], b
        while e is not a:
            e = prev[e]
            path.append(e)
        return path[::-1]

    def get(eid):
        if not eid:
            return None
        try:
            return net.getEdge(eid)
        except Exception:
            print(f"노선 기준 엣지 {eid}가 net.xml에 없어 정류장 엣지로 대신합니다")
            return None

    pts = [p for p in (get(start_edge),) if p is not None]
    pts += [net.getEdge(eid) for eid in stop_edges]
    endp = get(end_edge)
    if endp is not None:
        pts.append(endp)
    out = []
    for a, b in zip(pts, pts[1:]):
        part = seg(a, b)
        out.extend(part if not out else part[1:])
    return out or pts


BAD_SIDE = {"e506246961_1b", "e329995230_2a", "e329995230_2b", "e228795468_3a"}  # 합류부에서 계속 정체(순간이동) 확인


EAST_WAYS = ["73541403", "1536659790", "908646367", "218924602", "1212424783"]  # 청계광장→광장시장
WEST_WAYS = ["1212424785", "732213463", "1212424791", "908694578", "350127794"]  # 광장시장→청계광장


def chain_route(net, way_ids):
    """OSM way 순서대로 엣지를 이어 붙인다. 끊기면 그 구간만 최단경로로 잇는다.
    (전체를 최단경로로 구하면 종로 쪽 지름길로 우회하는 문제가 있었다.)"""
    import re
    out = []
    for w in way_ids:
        es = sorted((e for e in net.getEdges() if e.getID().startswith(f"e{w}_")),
                    key=lambda e: int(re.search(r"_(\d+)", e.getID()).group(1)))
        for e in es:
            if not out:
                out.append(e)
            elif e in out[-1].getOutgoing():
                out.append(e)
            else:
                mid, _ = net.getShortestPath(out[-1], e)
                if not mid:
                    raise RuntimeError(f"경로가 끊김: {out[-1].getID()} → {e.getID()}")
                out.extend(mid[1:])
    return out


def side_sources(net, route_edges, min_len=25.0, limit=8):
    """본선 경로로 들어오는 측면 도로 중 충분히 긴 진입 엣지를 찾는다."""
    on = {e.getID() for e in route_edges}
    cand = []
    for e in route_edges:
        for inc in e.getFromNode().getIncoming():
            if inc.getID() in on or inc.getID() in BAD_SIDE or not inc.allows("passenger"):
                continue
            if inc.getLength() >= min_len and inc.getFromNode().getType() != "dead_end_x":
                cand.append(inc)
    seen, out = set(), []
    for e in cand:
        if e.getID() not in seen:
            seen.add(e.getID())
            out.append(e)
    return out[:limit]


# 보행자가 보행구역에서 갇히는(jammed) 것이 검증에서 확인된 교차로 — 이곳을 지나는 보행 경로는 만들지 않는다
BAD_PED_JUNCTIONS = {"cluster_n5863541381_n871729555"}


def across_dest(pnet, from_eid, candidates, rng, max_dx=400.0):
    """from_eid에서 걸어서 갈 수 있는 반대편(길 건너) 엣지 하나. 없으면 None.

    하차 승객의 목적지를 길 건너편으로 주면 횡단보도를 건너는 장면이 생긴다.
    """
    try:
        a = pnet.getEdge(from_eid)
    except Exception:
        return None
    ax = a.getFromNode().getCoord()[0]
    cand = [b for b in candidates
            if b.allows("pedestrian") and b.getLength() > 15
            and abs(b.getFromNode().getCoord()[0] - ax) <= max_dx]
    rng.shuffle(cand)
    for b in cand[:30]:
        path, _ = pnet.getShortestPath(a, b, vClass="pedestrian")
        if path and not any(e.getToNode().getID() in BAD_PED_JUNCTIONS
                            or e.getFromNode().getID() in BAD_PED_JUNCTIONS for e in path):
            return b.getID()
    return None


def ped_pairs(net, east, west, rng, n):
    """본선 남측(동행)·북측(서행) 보도 사이를 건너는 보행자 (출발, 도착) 엣지 쌍.
    실제로 걸어서 갈 수 있는 쌍만 고른다 (횡단보도 경로가 있는 것)."""
    def usable(e):
        return e.allows("pedestrian") and e.getLength() > 15
    E = [e for e in east if usable(e)]
    W = [e for e in west if usable(e)]
    pairs, tries = [], 0
    while len(pairs) < n and tries < 4000 and E and W:
        tries += 1
        a, b = rng.choice(E), rng.choice(W)
        ax = a.getFromNode().getCoord()[0]
        bx = b.getFromNode().getCoord()[0]
        if not 80 <= abs(ax - bx) <= 500:
            continue
        if rng.random() < 0.5:
            a, b = b, a
        path, _ = net.getShortestPath(a, b, vClass="pedestrian")
        if path and any(e.getToNode().getID() in BAD_PED_JUNCTIONS
                        or e.getFromNode().getID() in BAD_PED_JUNCTIONS for e in path):
            continue
        if path and (a.getID(), b.getID()) not in pairs:
            pairs.append((a.getID(), b.getID()))
    return pairs


def build(net_path, out_dir, args):
    net = sumolib.net.readNet(net_path)
    # 보도 차로가 있으면 차량 차로는 1번부터 (netconvert --sidewalks.guess)
    # 최단경로는 600m 지점에서 본선 대신 짧은 연결로로 빠지므로 OSM way 순서대로 잇는다
    # (출발 구간이 신호교차로에 합쳐져 사라질 수 있어 START_EDGE 대신 경로 첫 엣지를 쓴다)
    route = chain_route(net, EAST_WAYS)
    lane0 = route[0].getLane(0)
    first_car_lane = 1 if lane0.allows("pedestrian") and not lane0.allows("passenger") else 0
    route = route[:[e.getID() for e in route].index(END_EDGE) + 1]
    west = chain_route(net, WEST_WAYS)
    srcs = side_sources(net, route)
    # 세종대로 합류(발표 ①: 접촉사고 원인 재현)는 일반 측면 합류와 분리해
    # 빈도를 직접 지정한다 (중복 투입 방지를 위해 일반 합류 목록에서는 뺀다).
    sejong_srcs = [e for e in srcs if "세종대로" in (e.getName() or "")]
    srcs = [e for e in srcs if e not in sejong_srcs]
    # 정류장 (config/lines.yaml): 실제 정차 구간(엣지 길이로 잘린 뒤 좌표) —
    # 불법주정차가 버스 정차 자리를 막지 않도록 이 구간(±margin)은 모든 주정차에서 뺀다.
    line_cfg = load_line_config(getattr(args, "lines", None))
    bus_stop_zones = {}
    placed_stops = []        # (sid, eid, start, end, name) — add.xml에 쓰는 순서
    placed = {}              # sid → (eid, start, end, name)
    for sid, st in line_cfg["stops"].items():
        try:
            eid, start, end = place_bus_stop(net, (route, west), st["edge"],
                                             float(st["start"]), float(st["end"]))
            s, e2 = fit_bus_stop(net.getEdge(eid).getLength(), start, end)
        except Exception:
            print("정류장 배치 실패 (엣지가 net.xml에 없음?):", sid, st.get("edge"))
            continue
        placed_stops.append((sid, eid, s, e2, st["name"]))
        placed[sid] = (eid, s, e2, st["name"])
        bus_stop_zones.setdefault(eid, []).append((s, e2))
    # 노선 경로: 정류장 순서대로 자동 계산 (lines.yaml의 stops 순서만 바꾸면 노선이 바뀐다)
    line_routes = {}         # 노선 이름 → (route_id, 엣지 목록, 정차할 정류장 순서)
    for lname, lc in line_cfg["lines"].items():
        sids = [s for s in lc["stops"] if s in placed]
        edges = line_route(net, [placed[s][0] for s in sids],
                           lc.get("start_edge"), lc.get("end_edge"))
        route_id = str(lc["route_id"])
        if route_id in ("main_east", "main_west"):
            raise SystemExit(f"노선 {lname}: route_id '{route_id}'는 배경 교통용 이름이라 쓸 수 없습니다")
        line_routes[lname] = (route_id, edges, sids)
    if args.line not in line_routes:
        raise SystemExit(f"--line {args.line}: lines.yaml에 없는 노선 (있는 것: {list(line_routes)})")
    ids = " ".join(e.getID() for e in route)
    rid = line_routes[args.line][0]   # 데모 ego가 달릴 route id (기본 A01_east → r0)
    stop_edges = {eid for _, eid, _, _, _ in placed_stops}   # 불법주정차 제외 엣지
    rng = random.Random(7)

    ctrl = [  # (id, model, prob, accel, decel, tau, minGap, maxSpeed, sigma, color)
        ("krauss", "Krauss", 0.35, 2.0, 4.5, 1.0, 2.5, 8.3, 0.3, "1,1,0"),
        ("idm", "IDM", 0.30, 1.8, 3.5, 1.2, 2.0, 8.3, None, "0,1,1"),
        ("eidm", "EIDM", 0.20, 2.0, 4.0, 1.1, 2.0, 8.3, None, "1,0.6,0"),
        ("acc", "ACC", 0.15, 2.0, 4.0, 1.5, 2.0, 8.3, None, "1,0,1"),
    ]
    L = ["<routes>"]
    # ego 차종: 학습 환경 모드에서는 road_config.EGO 값(가속·감속·최고속도)을 그대로 쓴다
    #   (학습된 모델이 그 물리 한계로 학습됐으므로). 데모 모드는 셔틀 실측 속도(약 20km/h).
    e_acc = args.ego_accel if args.ego_accel is not None else 1.5
    e_dec = args.ego_decel if args.ego_decel is not None else 4.0
    e_max = args.ego_max_speed if args.ego_max_speed is not None else args.shuttle_kmh / 3.6
    # 자율주행 셔틀(청계A01) 실제 차체 — 일반 승용차 대신 버스 규격(길이/폭/vClass).
    # boardingDuration: 승객 1명 타는 데 2.5초 [가정] — 한 명씩 타는 모습이 보이도록
    # SUMO 기본값(0.5초)보다 길게. 승하차가 끝날 때까지 정차(15초)가 자동 연장된다.
    if args.roii:
        # ROii(오토노머스에이투지) 사양: 4.9×2.095 m, 2890 kg [명세 §1],
        # 승객 정원 8명(시험운전자 제외 — 서울시 2025-09 보도) 입석 금지,
        # 최고 40 km/h·실주행 약 30 km/h (speedFactor 0.75). emergencyDecel [가정].
        # 색: 서울 시내버스(지선) 초록 — 실차는 흰색+랩핑이지만 화면 식별이 어렵다.
        L.append(f'    <vType id="ego_type" accel="{e_acc}" decel="{e_dec}" maxSpeed="{e_max:.2f}" '
                 f'length="4.9" width="2.095" mass="2890" personCapacity="8" '
                 f'speedFactor="0.75" emergencyDecel="5.0" '
                 f'color="0.36,0.69,0.15" vClass="bus" guiShape="bus" boardingDuration="2.5"/>')
    else:
        L.append(f'    <vType id="ego_type" accel="{e_acc}" decel="{e_dec}" maxSpeed="{e_max:.2f}" '
                 f'length="12.0" width="2.5" color="0.36,0.69,0.15" vClass="bus" guiShape="bus" '
                 f'boardingDuration="2.5"/>')
    L.append('    <vTypeDistribution id="carDist">')
    for cid, model, p, acc, dec, tau, gap, vmax, sigma, color in ctrl:
        sg = f' sigma="{sigma}"' if sigma is not None else ""
        L.append(f'        <vType id="car_{cid}" probability="{p}" carFollowModel="{model}" accel="{acc}" '
                 f'decel="{dec}" tau="{tau}" minGap="{gap}" maxSpeed="{vmax}" speedDev="0.1" '
                 f'color="{color}" vClass="passenger" guiShape="passenger/sedan"{sg} '
                 f'lcStrategic="1" lcCooperative="0.5" lcAssertive="1.2" lcImpatience="0.2"/>')
    L.append("    </vTypeDistribution>")
    # 끼어드는 차: 차선변경에 적극적이고 간격이 좁은 차량 (택시·배달차 가정)
    L.append('    <vType id="cutin_car" accel="2.6" decel="4.5" tau="1.0" minGap="1.5" maxSpeed="9.7" '
             'speedDev="0.15" lcAssertive="3" lcImpatience="1" lcSpeedGain="3" lcStrategic="1" '
             'jmIgnoreFoeProb="0.35" jmTimegapMinor="0.3" jmIgnoreFoeSpeed="3" color="0.9,0.9,0.9" vClass="taxi" guiShape="passenger/sedan"/>')
    L.append('    <vType id="motorcycle" vClass="motorcycle" guiShape="motorcycle" length="2.2" width="0.8" '
             'accel="3.5" decel="5.0" maxSpeed="11" minGap="1.0" lcAssertive="2" lcImpatience="0.5" lcCooperative="0.5" '
             'jmIgnoreFoeProb="0.2" color="0.2,0.2,0.2"/>')
    # OSM 조사: highway=cycleway 13개는 자동차 도로가 아니라 변환에서 빠진다.
    # 청계천로 자동차 way 14개에는 cycleway:left=track(천 쪽 분리 자전거도로)이 있으나
    # 그 태그는 전용 차로로 만들지 않는다. 대신 본선에 저속 자전거 흐름을 둔다.
    # maxSpeed 5 m/s, speedFactor 0.8 → 대략 3–5 m/s.
    L.append('    <vType id="bicycle" vClass="bicycle" guiShape="bicycle" length="1.8" width="0.65" '
             'accel="1.2" decel="3.0" maxSpeed="5.0" speedFactor="0.8" speedDev="0.15" minGap="0.5" '
             'color="0,0.55,0.25"/>')
    L.append('    <vType id="parked_car" accel="2.0" decel="4.5" maxSpeed="8.3" color="0.5,0.5,0.5" '
             'vClass="passenger" guiShape="passenger/van"/>')
    # 광장시장에 세워 둔 오토바이. 차로를 막지 않는 정식 주차(parking=true).
    L.append('    <vType id="parked_moto" vClass="motorcycle" length="2.2" width="0.8" '
             'accel="2.0" decel="4.5" maxSpeed="8.3" color="0.15,0.15,0.15" guiShape="motorcycle"/>')
    # 가게 앞 하역 중인 배달 트럭. 차로를 막는 정식 주차(parking=true), 하역 시간 300초.
    L.append('    <vType id="parked_truck" vClass="delivery" length="6.0" width="2.3" '
             'accel="2.0" decel="4.0" maxSpeed="8.3" color="0.3,0.3,0.4" guiShape="delivery"/>')
    # 배경 교통용 본선 route (노선과 분리 — 노선을 바꿔도 배경 교통은 그대로)
    L.append(f'    <route id="main_east" edges="{ids}"/>')
    if west:
        L.append(f'    <route id="main_west" edges="{" ".join(e.getID() for e in west)}"/>')
    # 버스 노선 route (lines.yaml): 정류장 순서로 자동 계산한 경로
    for lname, (route_id, edges, sids) in line_routes.items():
        L.append(f'    <route id="{route_id}" edges="{" ".join(e.getID() for e in edges)}"/>')
    # 본선 통과 교통 (동행/서행)
    L.append(f'    <flow id="through_e" type="carDist" route="main_east" begin="0" end="100000" '
             f'vehsPerHour="{args.through_vph}" departLane="best" departSpeed="max"/>')
    if west:
        L.append(f'    <flow id="through_w" type="carDist" route="main_west" begin="0" end="100000" '
                 f'vehsPerHour="{args.through_vph}" departLane="best" departSpeed="max"/>')
    # 측면 도로 합류(끼어들기 원인 ①). 목적지는 본선 끝 엣지.
    for e in srcs:
        L.append(f'    <flow id="side_{e.getID()}" type="cutin_car" from="{e.getID()}" to="{END_EDGE}" '
                 f'begin="0" end="100000" vehsPerHour="{args.side_vph}" departLane="best" '
                 f'departSpeed="max"/>')
    # 세종대로 합류 (발표 ①: 접촉사고 원인 재현용). 가볍게 60초에 1대.
    for i, e in enumerate(sejong_srcs):
        L.append(f'    <flow id="through_merge_{i}" type="cutin_car" from="{e.getID()}" to="{END_EDGE}" '
                 f'begin="0" end="100000" period="60" departLane="best" departSpeed="max"/>')
    # 오토바이: 본선 전체를 달리며 차선 사이를 파고든다
    L.append(f'    <flow id="moto" type="motorcycle" route="main_east" begin="0" end="100000" '
             f'vehsPerHour="{args.moto_vph}" departLane="best" departSpeed="max"/>')
    # 동행·서행 전 구간. 20초에 한 대면 4 m/s 기준으로 약 80 m 간격.
    L.append(f'    <flow id="bike_e" type="bicycle" route="main_east" begin="0" end="100000" '
             f'period="20" departLane="best" departSpeed="max"/>')
    if west:
        L.append('    <flow id="bike_w" type="bicycle" route="main_west" begin="0" end="100000" '
                 'period="20" departLane="best" departSpeed="max"/>')
    # 광장시장 구간: 15 m 간격으로 오토바이를 세워 둔다. 정차금지지대 위치는 뺀다.
    no_stop = load_no_stop_zones()
    moto_n = 0
    for eid in MOTO_PARK_EDGES:
        try:
            e = net.getEdge(eid)
        except Exception:
            print("오토바이 주정차 엣지 없음:", eid)
            continue
        lanes = passenger_lane_indexes(e)
        if not lanes:
            continue
        lane = lanes[0]
        pos = 15.0
        while pos < e.getLength() - 1.0:
            if not in_excluded_zone(no_stop, bus_stop_zones, eid, pos):
                depart = max(pos - 1.0, 0.1)
                L.append(
                    f'    <flow id="parked_moto_{moto_n}" type="parked_moto" from="{eid}" to="{eid}" '
                    f'begin="0" end="300" period="301" departLane="{lane}" departPos="{depart:.1f}" '
                    f'departSpeed="0">'
                    f'<stop lane="{eid}_{lane}" endPos="{pos:.1f}" duration="300" parking="true"/>'
                    f'</flow>')
                moto_n += 1
            pos += 15.0
    # 가게 앞 하역 중인 배달 트럭 (TRUCK_PARK_SPOTS). 차로를 막는 정식 주차.
    truck_n = 0
    for eid, fracs in TRUCK_PARK_SPOTS.items():
        try:
            e = net.getEdge(eid)
        except Exception:
            print("배달 트럭 주정차 엣지 없음:", eid)
            continue
        lanes = passenger_lane_indexes(e)
        if not lanes:
            continue
        lane = lanes[0]
        for frac in fracs:
            pos = e.getLength() * frac
            if in_excluded_zone(no_stop, bus_stop_zones, eid, pos):
                continue
            depart = max(pos - 1.0, 0.1)
            L.append(
                f'    <flow id="parked_truck_{truck_n}" type="parked_truck" from="{eid}" to="{eid}" '
                f'begin="0" end="300" period="301" departLane="{lane}" departPos="{depart:.1f}" '
                f'departSpeed="0">'
                f'<stop lane="{eid}_{lane}" endPos="{pos:.1f}" duration="300" parking="true"/>'
                f'</flow>')
            truck_n += 1
    # 승객을 태울 노선: ego가 달리는 route id (학습은 EGO["routes"], 데모는 --line)
    pax_rate = (args.passenger_rate if args.passenger_rate is not None
                else float(line_cfg.get("passenger_rate", 0.01)))
    pax_route_ids = ([s for s in args.pax_route_ids.split(",") if s]
                     if args.pax_route_ids else [rid])
    pax_lines = [(ln, lr) for ln, lr in line_routes.items() if lr[0] in pax_route_ids]
    # 보행 경로(횡단보도·보행구역)를 따라가려면 internal/pedestrian 연결을 함께 읽어야 한다
    pnet = east_main = west_main = None
    if args.ped_pph > 0 or (pax_rate > 0 and pax_lines):
        pnet = sumolib.net.readNet(net_path, withInternal=True, withPedestrianConnections=True)
        main_edges = [e for e in pnet.getEdges() if "청계천로" in (e.getName() or "")]
        def east(e):
            return e.getToNode().getCoord()[0] > e.getFromNode().getCoord()[0]
        east_main = [e for e in main_edges if east(e)]
        west_main = [e for e in main_edges if not east(e)]
    # 보행자: 청계천 남·북 보도를 오가며 신호 횡단보도를 건넌다 (팀 천호사거리와 같은 personFlow 방식)
    if args.ped_pph > 0:
        pairs = ped_pairs(pnet, east_main, west_main, rng, args.ped_pairs)
        L.append('    <vType id="ped" vClass="pedestrian" speed="1.3" color="0,0.4,1"/>')
        per = args.ped_pph / max(1, len(pairs))
        for i, (a, b) in enumerate(pairs):
            L.append(f'    <personFlow id="ped_{i}" type="ped" begin="0" end="100000" '
                     f'period="{3600.0 / per:.1f}" departPos="random">\n'
                     f'        <walk from="{a}" to="{b}" arrivalPos="random"/>\n    </personFlow>')
        print(f"보행자 경로 {len(pairs)}쌍, 시간당 {args.ped_pph}명")
    # 승객 (계획서 §2.3): 정류장 보도에서 걸어와 기다리고, ego 버스를 타고
    # (lines=노선 이름 — sumo_env가 reset에서 ego에 setLine), 두 정거장 뒤(또는 종점)에서
    # 내려 걸어간다. 목적지는 가능하면 길 건너편 → 횡단보도 건너는 장면이 생긴다.
    # 도착은 포아송(period=exp): passenger_rate 0.01 ≈ 100초에 1명씩 정류장에 모인다.
    if pax_rate > 0 and pax_lines:
        L.append('    <vType id="pax" vClass="pedestrian" speed="1.3" color="1,0.55,0"/>')
        n_pax_flows = 0
        for lname, (route_id2, edges2, sids2) in pax_lines:
            edge_dir = {e.getID() for e in east_main}
            for i, sid in enumerate(sids2[:-1]):
                dest_sid = sids2[min(i + 2, len(sids2) - 1)]
                src_eid = placed[sid][0]
                dest_eid = placed[dest_sid][0]
                # 하차 정류장이 동행이면 길 건너 목적지는 서행 보도에서 찾는다 (반대도 같음)
                opposite = west_main if dest_eid in edge_dir else east_main
                walk_to = across_dest(pnet, dest_eid, opposite, rng) or dest_eid
                L.append(
                    f'    <personFlow id="pax_{lname}_{sid}" type="pax" begin="0" end="100000" '
                    f'period="exp({pax_rate})" departPos="random">\n'
                    f'        <walk from="{src_eid}" busStop="{sid}"/>\n'
                    f'        <ride busStop="{dest_sid}" lines="{lname}"/>\n'
                    f'        <walk to="{walk_to}" arrivalPos="random"/>\n    </personFlow>')
                n_pax_flows += 1
        print(f"승객 흐름 {n_pax_flows}개 (노선 {[ln for ln, _ in pax_lines]}, "
              f"도착률 {pax_rate}/s/정류장)")
    # 데모용 ego(셔틀): 학습 환경에서는 sumo_env가 TraCI로 직접 투입하므로 끌 수 있다.
    # line 속성 + 정류장 stop이 있어야 기다리는 승객(lines=노선 이름)을 태운다.
    if args.demo_ego:
        demo_stops = "".join(f'\n        <stop busStop="{sid}" duration="15"/>'
                             for sid in line_routes[args.line][2])
        L.append(f'    <flow id="ego_demo" type="ego_type" route="{rid}" begin="{args.ego_begin}" '
                 f'end="100000" period="{args.ego_period}" departLane="{first_car_lane}" '
                 f'departSpeed="0" line="{args.line}">{demo_stops}\n    </flow>')
    # 불법주정차 (1-5): 동행·서행 전 구간을 30 m 간격으로 촘촘히 채운다
    # (기존 엣지당 1~2곳 방식 대비 밀도 약 2배). 제외: 정류장 엣지,
    # 정차금지지대, 승용차 1차로 엣지(이미 좁아 중복 안 함).
    PARK_SPACING = 30.0
    park_spots = []
    for edges, route_name in ((route, "main_east"), (west, "main_west")):
        for e in edges:
            eid = e.getID()
            if eid in stop_edges:    # 정류장 엣지는 뺀다 (버스가 댈 자리)
                continue
            lanes = passenger_lane_indexes(e)
            if len(lanes) < 2:
                continue
            pos = PARK_SPACING / 2.0
            while pos < e.getLength() - 1.0:
                if not in_excluded_zone(no_stop, bus_stop_zones, eid, pos):
                    park_spots.append((e, lanes[0], route_name, pos))
                pos += PARK_SPACING
    for i, (e, lane, route_name, pos) in enumerate(park_spots):
        L.append(f'    <flow id="illegal_park_{i}" type="parked_car" route="{route_name}" '
                 f'begin="{args.ego_begin + 20 + (i % 30) * 4}" end="100000" period="{args.park_period}" '
                 f'departLane="{lane}" departSpeed="max">'
                 f'<stop lane="{e.getID()}_{lane}" endPos="{pos:.1f}" duration="{rng.randint(25, 50)}" '
                 f'parking="false"/></flow>')
    L.append("</routes>")
    with open(os.path.join(out_dir, "cgc.rou.xml"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")

    # 정류장 11개. 보도가 lane 0이면 승용차 차로(_1)에 둔다.
    # color: 서울 시내버스 표지판 느낌의 반투명 파란색 (기본 초록 상자 대신).
    stop_lines = {}
    for lname, lc in line_cfg["lines"].items():
        for sid in lc["stops"]:
            stop_lines.setdefault(sid, []).append(lname)
    # 정류장 번호(1부터)와 구분 색: lines.yaml stops 순서대로 팔레트를 돌려 쓴다
    stop_number, stop_color = {}, {}
    for i, sid in enumerate(line_cfg["stops"]):
        stop_number[sid] = i + 1
        stop_color[sid] = line_cfg["stops"][sid].get("color") or STOP_PALETTE[i % len(STOP_PALETTE)]
    add = ["<additional>"]
    written_stops = 0
    for sid, eid, start_pos, end_pos, name in placed_stops:
        edge = net.getEdge(eid)
        lane = bus_stop_lane_index(edge)
        safe_name = name.replace("&", "&amp;").replace('"', "")
        # color 알파 0 = busStop 자체 그림(파란 면 + 독일식 'H' 표지)을 숨긴다.
        # 정류장의 겉모습은 아래 한국식 그림(쉘터·표지판)이 담당.
        # lines도 일부러 안 넣는다 (넣으면 GUI가 노선 글자를 도로 위에 그린다).
        add.append(
            f'    <busStop id="{sid}" lane="{eid}_{lane}" startPos="{start_pos}" '
            f'endPos="{end_pos}" name="{safe_name}" color="0,0,0,0"/>')
        # 보도 위에 한국식 정류장 쉘터(지붕·기둥·벤치·뒷벽)와 표지판을 그린다.
        # 표지판 글자는 로마자(name_en) — sumo-gui 내장 글꼴에 한글이 없어
        # 한글은 □□로 깨진다. (busStop의 name(한글)은 클릭 정보창에는 나온다)
        label = line_cfg["stops"].get(sid, {}).get("name_en") or sid
        add += stop_shelter_polys(edge, lane, sid, start_pos, end_pos,
                                  label, stop_lines.get(sid, []),
                                  color=stop_color.get(sid, "#2B6570"),
                                  number=stop_number.get(sid))
        written_stops += 1
    add.append("</additional>")
    with open(os.path.join(out_dir, "cgc.add.xml"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(add) + "\n")

    # 노선 메타: route id → 정차할 정류장 순서. sumo_env가 reset에서 읽어 setBusStop 한다.
    meta = {
        "lines": {route_id: {"line": lname, "stops": sids,
                             "length_m": round(sum(e.getLength() for e in edges), 1)}
                  for lname, (route_id, edges, sids) in line_routes.items()},
        # color·number: 대기 현황 창(tools/live_board.py)이 표·지도에 같은 색을 쓴다
        "stops": {sid: {"edge": eid, "start": s, "end": e2, "name": name,
                        "name_en": line_cfg["stops"].get(sid, {}).get("name_en", sid),
                        "number": stop_number.get(sid), "color": stop_color.get(sid)}
                  for sid, eid, s, e2, name in placed_stops},
    }
    with open(os.path.join(out_dir, "cgc.line_meta.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    # GUI: 전체 도로가 한 화면에 보이도록 (x 0~2275, y -280~210)
    # persons: 기본값은 사람을 작은 삼각형(화살표)으로 그려서 뭔지 알아보기 어렵다.
    #   personQuality=2 → 사람 모양, person_minSize=2 → 멀리서도 점 이상 크기 보장.
    # pois: 지붕 한가운데 투명 POI의 type에 넣은 "정류장 이름 | 노선"을
    #   밝은 회색 지붕 위에 진한 청록 글씨로 띄운다 (poiName은 id가 떠서 안 쓴다).
    #   constantSize=0 + size=1.3 → 글자 높이가 실세계 1.3 m로 고정:
    #   줌 인하면 읽히고, 줌 아웃하면 다른 것들과 같이 작아진다.
    #   (constantSize=1은 화면 픽셀 고정이라 줌 아웃 시 세상보다 커 보였음)
    with open(os.path.join(out_dir, "cgc.gui.xml"), "w", encoding="utf-8", newline="\n") as f:
        f.write('<viewsettings>\n'
                '    <scheme name="real world">\n'
                '        <persons personMode="0" personQuality="2" person_minSize="2.0" '
                'person_exaggeration="1.5" person_constantSize="0"/>\n'
                '        <pois poi_minSize="0" poiType_show="1" poiType_size="1.3" '
                'poiType_color="28,72,80" poiType_bgColor="0,0,0,0" '
                'poiType_constantSize="0"/>\n'
                # busStop의 SUMO 기본 그림(초록 면 + 독일식 H 표지)을 사실상 숨긴다
                # (크기 배율 0.01). 정류장 겉모습은 쉘터·표지판 폴리곤이 담당.
                '        <additionals addMode="0" add_minSize="0" add_exaggeration="0.01" '
                'addName_show="0" addFullName_show="0"/>\n'
                '    </scheme>\n'
                f'    <viewport zoom="{args.zoom}" x="{args.view_x}" y="{args.view_y}" angle="0"/>\n'
                '    <delay value="100"/>\n</viewsettings>\n')
    with open(os.path.join(out_dir, "cgc.sumocfg"), "w", encoding="utf-8", newline="\n") as f:
        f.write('<configuration>\n    <input>\n        <net-file value="cgc.net.xml"/>\n'
                '        <route-files value="cgc.rou.xml"/>\n'
                '        <additional-files value="cgc.add.xml"/>\n    </input>\n'
                '    <gui_only>\n        <gui-settings-file value="cgc.gui.xml"/>\n    </gui_only>\n'
                '</configuration>\n')
    lines_txt = ", ".join(f"{ln} ({route_id}: 정류장 {len(sids)}곳, {len(edges)}엣지)"
                          for ln, (route_id, edges, sids) in line_routes.items())
    print(f"본선 {len(route)}개 엣지 / 노선: {lines_txt}")
    print(f"측면 합류 {len(srcs)}곳(+세종대로 {len(sejong_srcs)}곳), "
          f"불법주정차 {len(park_spots)}곳, 오토바이 주차 {moto_n}대, 트럭 주정차 {truck_n}대, "
          f"정류장 {written_stops}개")
    print("측면 합류 엣지:", [e.getID() for e in srcs])
    print("세종대로 합류 엣지:", [e.getID() for e in sejong_srcs])


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="env/cheonggyecheon/cgc.net.xml")
    ap.add_argument("--out", default="env/cheonggyecheon")
    ap.add_argument("--lines", default=None,
                    help=f"노선 설정 YAML (기본 {LINES_YAML}, 없으면 내장 청계A01)")
    ap.add_argument("--line", default="A01_east",
                    help="데모 ego가 달릴 노선 이름 (lines.yaml의 lines 키)")
    ap.add_argument("--route-id", default="r0",
                    help="(호환용, 무시됨 — route id는 lines.yaml의 route_id로 정한다)")
    ap.add_argument("--loop-route-id", default="r0_loop",
                    help="(호환용, 무시됨 — 왕복 노선은 lines.yaml의 A01_loop)")
    ap.add_argument("--shuttle-kmh", type=float, default=20.0)
    ap.add_argument("--roii", action="store_true",
                    help="ego를 실제 청계A01 셔틀(ROii) 제원으로: 4.9 m, 정원 8명, 하늘색")
    ap.add_argument("--ego-accel", type=float, default=None)
    ap.add_argument("--ego-decel", type=float, default=None)
    ap.add_argument("--ego-max-speed", type=float, default=None, help="m/s")
    ap.add_argument("--passenger-rate", type=float, default=None,
                    help="정류장별 승객 도착률 (1/s). 기본: lines.yaml의 passenger_rate. 0이면 없음")
    ap.add_argument("--pax-route-ids", default=None,
                    help="승객을 만들 노선의 route id들 (쉼표 구분). 기본: --line 노선만")
    ap.add_argument("--through-vph", type=int, default=500)
    ap.add_argument("--side-vph", type=int, default=40)
    ap.add_argument("--moto-vph", type=int, default=120)
    ap.add_argument("--ped-pph", type=int, default=400)
    ap.add_argument("--ped-pairs", type=int, default=24)
    ap.add_argument("--park-period", type=int, default=90)
    ap.add_argument("--demo-ego", dest="demo_ego", action="store_true", default=True)
    ap.add_argument("--no-demo-ego", dest="demo_ego", action="store_false",
                    help="학습 환경에서는 sumo_env가 ego를 직접 투입하므로 데모 셔틀을 끈다")
    ap.add_argument("--ego-begin", type=int, default=60)  # 흐름은 출발시각 순으로 적어야 함
    ap.add_argument("--ego-period", type=int, default=300)
    ap.add_argument("--zoom", type=int, default=250)
    ap.add_argument("--view-x", type=float, default=1100)
    ap.add_argument("--view-y", type=float, default=-30)
    return ap.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    build(a.net, a.out, a)
