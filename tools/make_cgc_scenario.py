"""청계천로 SUMO 시나리오 생성: 경로·교통 흐름·GUI 설정·sumocfg.

전제: tools/osm_to_sumo.py 로 만든 nod/edg + netconvert 로 만든 cgc.net.xml.

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

# 정류장 엣지는 불법주정차에서 뺀다 (버스가 댈 자리).
EXCLUDE_EDGES_FOR_PARKING = [
    "e73541403_2", "e73541403_8", "e908646367_3", "e218924602_1",
    "e218924602_6", "e1212424785_1", "e732213463_1", "e732213463_4",
    "e1212424791_3", "e908694578_1", "e908694578_3",
]
# 광장시장 구간은 엣지당 2곳 (길이의 30%, 70%).
DENSE_PARK_EDGES = {
    "e218924602_4", "e218924602_5", "e218924602_6",
    "e1212424783_1", "e1212424783_2", "e1212424783_3", "e1212424783_4",
    "e1212424785_1", "e1212424785_2",
}
# 로드뷰에서 오토바이가 서 있던 광장시장 구간.
MOTO_PARK_EDGES = [
    "e218924602_6", "e1212424783_1", "e1212424783_2",
    "e1212424783_3", "e1212424783_4",
]


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


def bus_stop_lane_index(edge):
    """보도가 lane 0이면 정류장은 승용차 차로 1. 아니면 0."""
    lane0 = edge.getLane(0)
    if lane0.allows("pedestrian") and not lane0.allows("passenger") and edge.getLaneNumber() > 1:
        return 1
    return 0


# 분석 좌표. endPos가 엣지보다 길면 fit_bus_stop이 맞춘다.
BUS_STOPS = [
    ("bs_01", "e73541403_2", 22, 37, "청계광장"),
    ("bs_02", "e73541403_8", 45, 60, "청계1가.광교"),
    ("bs_03", "e908646367_3", 30, 45, "청계2가.삼일교(동)"),
    ("bs_04", "e218924602_1", 39, 54, "청계3가.관수교(동)"),
    ("bs_05", "e218924602_6", 68, 83, "청계4가.배오개다리"),
    ("bs_06", "e1212424785_1", 184, 199, "청계5가.광장시장"),
    ("bs_07", "e732213463_1", 113, 128, "청계4가.세운교"),
    ("bs_08", "e732213463_4", 43, 58, "청계3가.관수교(서)"),
    ("bs_09", "e1212424791_3", 38, 53, "청계2가.수표교"),
    ("bs_10", "e908694578_1", 43, 58, "청계2가.삼일교(서)"),
    ("bs_11", "e908694578_3", 66, 81, "청계1가.장통교"),
]


def shortest(net, a, b):
    path, _ = net.getShortestPath(net.getEdge(a), net.getEdge(b))
    return path


BAD_SIDE = {"e506246961_1b", "e329995230_2a", "e329995230_2b", "e228795468_3a"}  # 합류부에서 계속 정체(순간이동) 확인


EAST_WAYS = ["73541403", "1536659790", "908646367", "218924602", "1212424783"]  # 청계광장→광장시장
WEST_WAYS = ["1212424785", "732213463", "1212424791", "908694578", "350127794"]  # 광장시장→청계광장

# 청계광장 회차(e183272596_1) + 청계5가 좌회전 두 번으로 도는 왕복.
# OSM 중심선 기준 63개. netconvert가 교차로에 삼킨 id는 resolve_loop_edges가 뺀다.
LOOP_EDGE_IDS = (
    "e73541403_1 e73541403_2 e73541403_3 e73541403_4 e73541403_5 e73541403_6 "
    "e73541403_7 e73541403_8 "
    "e1536659790_1 e1536659790_2 e1536659790_3 e1536659790_4 e1536659790_5 e1536659790_6 "
    "e908646367_1 e908646367_2 e908646367_3 e908646367_4 e908646367_5 e908646367_6 "
    "e218924602_1 e218924602_2 e218924602_3 e218924602_4 e218924602_5 e218924602_6 "
    "e1212424783_1 e1212424783_2 e1212424783_3 e1212424783_4 "
    "e1455526426_1 e482167259_1a e608946831_1a e608946832_1a "
    "e1212424785_1 e1212424785_2 "
    "e732213463_1 e732213463_2 e732213463_3 e732213463_4 e732213463_5 "
    "e1212424791_1 e1212424791_2 e1212424791_3 e1212424791_4 e1212424791_5 "
    "e1212424791_6 e1212424791_7 e1212424791_8 "
    "e908694578_1 e908694578_2 e908694578_3 e908694578_4 e908694578_5 e908694578_6 "
    "e908694578_7 "
    "e350127794_1 e350127794_2 e350127794_3 e350127794_4 e350127794_5 e350127794_6 "
    "e183272596_1"
).split()


def resolve_loop_edges(net, edge_ids):
    """왕복 edge 목록을 현재 net.xml에 맞게 고친다.

    청계5가 짧은 연결(e482167259_1a, e608946832_1a)은 선행 확인에서 id가 살아 있었다.
    그보다 앞선 본선에서 netconvert가 없앤 e73541403_7, e1212424791_8처럼
    목록에만 있고 네트워크에 없는 id는 빼고, 앞뒤가 직접 이어지는지 본다.
    """
    present, missing = [], []
    for eid in edge_ids:
        try:
            present.append(net.getEdge(eid))
        except Exception:
            missing.append(eid)
    if missing:
        print("왕복 route에서 제외 (net.xml에 없음):", " ".join(missing))
    out = []
    for edge in present:
        if out and edge not in out[-1].getOutgoing():
            raise RuntimeError(
                f"왕복 경로가 끊김: {out[-1].getID()} → {edge.getID()} "
                f"(제외된 엣지: {' '.join(missing) or '없음'})")
        out.append(edge)
    return out


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
    first_car_lane = 1 if net.getEdge(START_EDGE).getLane(0).allows("pedestrian") and \
        not net.getEdge(START_EDGE).getLane(0).allows("passenger") else 0
    # 최단경로는 600m 지점에서 본선 대신 짧은 연결로로 빠지므로 OSM way 순서대로 잇는다
    route = chain_route(net, EAST_WAYS)
    route = route[:[e.getID() for e in route].index(END_EDGE) + 1]
    west = chain_route(net, WEST_WAYS)
    srcs = side_sources(net, route)
    ids = " ".join(e.getID() for e in route)
    rid = args.route_id   # 학습 환경(sumo_env)은 ego/배경 경로를 "r0"으로 찾는다
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
    L.append(f'    <vType id="ego_type" accel="{e_acc}" decel="{e_dec}" maxSpeed="{e_max:.2f}" '
             f'length="5.0" width="2.0" color="1,0,0" vClass="passenger" guiShape="bus"/>')
    L.append('    <vTypeDistribution id="carDist">')
    for cid, model, p, acc, dec, tau, gap, vmax, sigma, color in ctrl:
        sg = f' sigma="{sigma}"' if sigma is not None else ""
        L.append(f'        <vType id="car_{cid}" probability="{p}" carFollowModel="{model}" accel="{acc}" '
                 f'decel="{dec}" tau="{tau}" minGap="{gap}" maxSpeed="{vmax}" speedDev="0.1" '
                 f'color="{color}" vClass="passenger" guiShape="passenger/sedan"{sg} '
                 f'lcStrategic="1" lcCooperative="0.5" lcAssertive="1.2" lcImpatience="0.2"/>')
    L.append("    </vTypeDistribution>")
    # 끼어드는 차: 차선변경에 적극적이고 간격이 좁은 차량 (택시·배달차 가정)
    L.append('    <vType id="cutin_car" accel="2.6" decel="4.5" tau="0.8" minGap="1.2" maxSpeed="9.7" '
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
    L.append(f'    <route id="{rid}" edges="{ids}"/>')
    loop = resolve_loop_edges(net, LOOP_EDGE_IDS)
    loop_ids = " ".join(e.getID() for e in loop)
    L.append(f'    <route id="{args.loop_route_id}" edges="{loop_ids}"/>')
    if west:
        L.append(f'    <route id="main_west" edges="{" ".join(e.getID() for e in west)}"/>')
    # 본선 통과 교통 (동행/서행)
    L.append(f'    <flow id="through_e" type="carDist" route="{rid}" begin="0" end="100000" '
             f'vehsPerHour="{args.through_vph}" departLane="best" departSpeed="max"/>')
    if west:
        L.append(f'    <flow id="through_w" type="carDist" route="main_west" begin="0" end="100000" '
                 f'vehsPerHour="{args.through_vph}" departLane="best" departSpeed="max"/>')
    # 측면 도로 합류(끼어들기 원인 ①). 목적지는 본선 끝 엣지.
    for e in srcs:
        L.append(f'    <flow id="side_{e.getID()}" type="cutin_car" from="{e.getID()}" to="{END_EDGE}" '
                 f'begin="0" end="100000" vehsPerHour="{args.side_vph}" departLane="best" '
                 f'departSpeed="max"/>')
    # 오토바이: 본선 전체를 달리며 차선 사이를 파고든다
    L.append(f'    <flow id="moto" type="motorcycle" route="{rid}" begin="0" end="100000" '
             f'vehsPerHour="{args.moto_vph}" departLane="best" departSpeed="max"/>')
    # 동행·서행 전 구간. 45초에 한 대면 4 m/s 기준으로 약 180 m 간격.
    L.append(f'    <flow id="bike_e" type="bicycle" route="{rid}" begin="0" end="100000" '
             f'period="45" departLane="best" departSpeed="max"/>')
    if west:
        L.append('    <flow id="bike_w" type="bicycle" route="main_west" begin="0" end="100000" '
                 'period="45" departLane="best" departSpeed="max"/>')
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
            if not in_no_stop(no_stop, eid, pos):
                depart = max(pos - 1.0, 0.1)
                L.append(
                    f'    <flow id="parked_moto_{moto_n}" type="parked_moto" from="{eid}" to="{eid}" '
                    f'begin="0" end="300" period="301" departLane="{lane}" departPos="{depart:.1f}" '
                    f'departSpeed="0">'
                    f'<stop lane="{eid}_{lane}" endPos="{pos:.1f}" duration="300" parking="true"/>'
                    f'</flow>')
                moto_n += 1
            pos += 15.0
    # 보행자: 청계천 남·북 보도를 오가며 신호 횡단보도를 건넌다 (팀 천호사거리와 같은 personFlow 방식)
    if args.ped_pph > 0:
        # 보행 경로(횡단보도·보행구역)를 따라가려면 internal/pedestrian 연결을 함께 읽어야 한다
        pnet = sumolib.net.readNet(net_path, withInternal=True, withPedestrianConnections=True)
        main_edges = [e for e in pnet.getEdges() if "청계천로" in (e.getName() or "")]
        def east(e):
            return e.getToNode().getCoord()[0] > e.getFromNode().getCoord()[0]
        pairs = ped_pairs(pnet, [e for e in main_edges if east(e)], [e for e in main_edges if not east(e)],
                          rng, args.ped_pairs)
        L.append('    <vType id="ped" vClass="pedestrian" speed="1.3" color="0,0.4,1"/>')
        per = args.ped_pph / max(1, len(pairs))
        for i, (a, b) in enumerate(pairs):
            L.append(f'    <personFlow id="ped_{i}" type="ped" begin="0" end="100000" '
                     f'period="{3600.0 / per:.1f}" departPos="random">\n'
                     f'        <walk from="{a}" to="{b}" arrivalPos="random"/>\n    </personFlow>')
        print(f"보행자 경로 {len(pairs)}쌍, 시간당 {args.ped_pph}명")
    # 데모용 ego(셔틀): 학습 환경에서는 sumo_env가 TraCI로 직접 투입하므로 끌 수 있다.
    if args.demo_ego:
        L.append(f'    <flow id="ego_demo" type="ego_type" route="{rid}" begin="{args.ego_begin}" '
                 f'end="100000" period="{args.ego_period}" departLane="{first_car_lane}" departSpeed="0"/>')
    # 불법주정차: 동행·서행 중 길이 30 m 초과, 승용차 차로 2개 이상인 엣지.
    # 정류장 엣지와 정차금지지대 위치는 제외. 광장시장 구간은 30%·70% 두 곳.
    park_spots = []
    for edges, route_name in ((route, rid), (west, "main_west")):
        for e in edges:
            if e.getID() in EXCLUDE_EDGES_FOR_PARKING or e.getLength() <= 30:
                continue
            lanes = passenger_lane_indexes(e)
            if len(lanes) < 2:
                continue
            fracs = (0.3, 0.7) if e.getID() in DENSE_PARK_EDGES else (0.6,)
            for frac in fracs:
                pos = e.getLength() * frac
                if in_no_stop(no_stop, e.getID(), pos):
                    continue
                park_spots.append((e, lanes[0], route_name, pos))
    for i, (e, lane, route_name, pos) in enumerate(park_spots):
        L.append(f'    <flow id="illegal_park_{i}" type="parked_car" route="{route_name}" '
                 f'begin="{args.ego_begin + 40 + 40 * i}" end="100000" period="{args.park_period}" '
                 f'departLane="{lane}" departSpeed="max">'
                 f'<stop lane="{e.getID()}_{lane}" endPos="{pos:.1f}" duration="{rng.randint(25, 50)}" '
                 f'parking="false"/></flow>')
    L.append("</routes>")
    with open(os.path.join(out_dir, "cgc.rou.xml"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")

    # 정류장 11개. 보도가 lane 0이면 승용차 차로(_1)에 둔다.
    add = ["<additional>"]
    written_stops = 0
    for sid, eid, start, end, name in BUS_STOPS:
        try:
            edge = net.getEdge(eid)
        except Exception:
            print("정류장 엣지 없음:", sid, eid)
            continue
        lane = bus_stop_lane_index(edge)
        start_pos, end_pos = fit_bus_stop(edge.getLength(), start, end)
        safe_name = name.replace("&", "&amp;").replace('"', "")
        add.append(
            f'    <busStop id="{sid}" lane="{eid}_{lane}" startPos="{start_pos}" '
            f'endPos="{end_pos}" name="{safe_name}"/>')
        written_stops += 1
    add.append("</additional>")
    with open(os.path.join(out_dir, "cgc.add.xml"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(add) + "\n")

    # GUI: 전체 도로가 한 화면에 보이도록 (x 0~2275, y -280~210)
    with open(os.path.join(out_dir, "cgc.gui.xml"), "w", encoding="utf-8", newline="\n") as f:
        f.write('<viewsettings>\n    <scheme name="real world"/>\n'
                f'    <viewport zoom="{args.zoom}" x="{args.view_x}" y="{args.view_y}" angle="0"/>\n'
                '    <delay value="100"/>\n</viewsettings>\n')
    with open(os.path.join(out_dir, "cgc.sumocfg"), "w", encoding="utf-8", newline="\n") as f:
        f.write('<configuration>\n    <input>\n        <net-file value="cgc.net.xml"/>\n'
                '        <route-files value="cgc.rou.xml"/>\n'
                '        <additional-files value="cgc.add.xml"/>\n    </input>\n'
                '    <gui_only>\n        <gui-settings-file value="cgc.gui.xml"/>\n    </gui_only>\n'
                '</configuration>\n')
    print(f"경로 {len(route)}개 엣지, 왕복 {args.loop_route_id} {len(loop)}개 엣지, "
          f"측면 합류 {len(srcs)}곳, 불법주정차 {len(park_spots)}곳, 오토바이 주차 {moto_n}대, "
          f"정류장 {written_stops}개")
    print("측면 합류 엣지:", [e.getID() for e in srcs])


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="env/cheonggyecheon/cgc.net.xml")
    ap.add_argument("--out", default="env/cheonggyecheon")
    ap.add_argument("--route-id", default="main_east")
    ap.add_argument("--loop-route-id", default="r0_loop",
                    help="청계광장·청계5가에서 되돌아오는 왕복 route id")
    ap.add_argument("--shuttle-kmh", type=float, default=20.0)
    ap.add_argument("--ego-accel", type=float, default=None)
    ap.add_argument("--ego-decel", type=float, default=None)
    ap.add_argument("--ego-max-speed", type=float, default=None, help="m/s")
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
