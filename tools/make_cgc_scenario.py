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
import math
import os
import sys
import random

SUMO_HOME = os.environ.get("SUMO_HOME", os.path.expanduser("~/src/sumo-1.27.1-mesa"))
sys.path.append(os.path.join(SUMO_HOME, "tools"))
import sumolib  # noqa: E402

START_EDGE = "e73541403_1"      # 청계광장 (동행 첫 구간)
END_EDGE = "e1212424783_4"      # 청계5가·광장시장 부근 (동행 마지막 구간)


def shortest(net, a, b):
    path, _ = net.getShortestPath(net.getEdge(a), net.getEdge(b))
    return path


BAD_SIDE = {"e506246961_1b", "e329995230_2a", "e329995230_2b", "e228795468_3a"}  # 합류부에서 계속 정체(순간이동) 확인


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
        if path and (a.getID(), b.getID()) not in pairs:
            pairs.append((a.getID(), b.getID()))
    return pairs


def build(net_path, out_dir, args):
    net = sumolib.net.readNet(net_path)
    # 보도 차로가 있으면 차량 차로는 1번부터 (netconvert --sidewalks.guess)
    first_car_lane = 1 if net.getEdge(START_EDGE).getLane(0).allows("pedestrian") and \
        not net.getEdge(START_EDGE).getLane(0).allows("passenger") else 0
    route = shortest(net, START_EDGE, END_EDGE)
    west = chain_route(net, WEST_WAYS)
    srcs = side_sources(net, route)
    ids = " ".join(e.getID() for e in route)
    rng = random.Random(7)

    ctrl = [  # (id, model, prob, accel, decel, tau, minGap, maxSpeed, sigma, color)
        ("krauss", "Krauss", 0.35, 2.0, 4.5, 1.0, 2.5, 8.3, 0.3, "1,1,0"),
        ("idm", "IDM", 0.30, 1.8, 3.5, 1.2, 2.0, 8.3, None, "0,1,1"),
        ("eidm", "EIDM", 0.20, 2.0, 4.0, 1.1, 2.0, 8.3, None, "1,0.6,0"),
        ("acc", "ACC", 0.15, 2.0, 4.0, 1.5, 2.0, 8.3, None, "1,0,1"),
    ]
    L = ["<routes>"]
    L.append(f'    <vType id="ego_type" accel="1.5" decel="4.0" maxSpeed="{args.shuttle_kmh/3.6:.2f}" '
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
    L.append('    <vType id="parked_car" accel="2.0" decel="4.5" maxSpeed="8.3" color="0.5,0.5,0.5" '
             'vClass="passenger" guiShape="passenger/van"/>')
    L.append(f'    <route id="main_east" edges="{ids}"/>')
    if west:
        L.append(f'    <route id="main_west" edges="{" ".join(e.getID() for e in west)}"/>')
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
    # 오토바이: 본선 전체를 달리며 차선 사이를 파고든다
    L.append(f'    <flow id="moto" type="motorcycle" route="main_east" begin="0" end="100000" '
             f'vehsPerHour="{args.moto_vph}" departLane="best" departSpeed="max"/>')
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
        L.append(f'    <flow id="ego_demo" type="ego_type" route="main_east" begin="{args.ego_begin}" '
                 f'end="100000" period="{args.ego_period}" departLane="{first_car_lane}" departSpeed="0"/>')
    # 불법주정차: 본선 중간 엣지의 우측 차로(0번)에 잠깐 서는 차 (뒤차가 왼쪽으로 끼어들게 함)
    stop_edges = [e for e in route if e.getLength() > 60 and e.getLaneNumber() >= 2 + first_car_lane][:4]
    for i, e in enumerate(stop_edges):
        pos = e.getLength() * 0.6
        L.append(f'    <flow id="illegal_park_{i}" type="parked_car" route="main_east" begin="{args.ego_begin + 40 + 40 * i}" '
                 f'end="100000" period="{args.park_period}" departLane="{first_car_lane}" departSpeed="max">'
                 f'<stop lane="{e.getID()}_{first_car_lane}" endPos="{pos:.1f}" duration="{rng.randint(25, 50)}" parking="false"/>'
                 f'</flow>')
    L.append("</routes>")
    with open(os.path.join(out_dir, "cgc.rou.xml"), "w") as f:
        f.write("\n".join(L) + "\n")

    # GUI: 전체 도로가 한 화면에 보이도록 (x 0~2275, y -280~210)
    with open(os.path.join(out_dir, "cgc.gui.xml"), "w") as f:
        f.write('<viewsettings>\n    <scheme name="real world"/>\n'
                f'    <viewport zoom="{args.zoom}" x="{args.view_x}" y="{args.view_y}" angle="0"/>\n'
                '    <delay value="100"/>\n</viewsettings>\n')
    with open(os.path.join(out_dir, "cgc.sumocfg"), "w") as f:
        f.write('<configuration>\n    <input>\n        <net-file value="cgc.net.xml"/>\n'
                '        <route-files value="cgc.rou.xml"/>\n    </input>\n'
                '    <gui_only>\n        <gui-settings-file value="cgc.gui.xml"/>\n    </gui_only>\n'
                '</configuration>\n')
    print(f"경로 {len(route)}개 엣지, 측면 합류 {len(srcs)}곳, 불법주정차 {len(stop_edges)}곳")
    print("측면 합류 엣지:", [e.getID() for e in srcs])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="env/cheonggyecheon/cgc.net.xml")
    ap.add_argument("--out", default="env/cheonggyecheon")
    ap.add_argument("--shuttle-kmh", type=float, default=20.0)
    ap.add_argument("--through-vph", type=int, default=500)
    ap.add_argument("--side-vph", type=int, default=40)
    ap.add_argument("--moto-vph", type=int, default=120)
    ap.add_argument("--ped-pph", type=int, default=400)
    ap.add_argument("--ped-pairs", type=int, default=24)
    ap.add_argument("--park-period", type=int, default=90)
    ap.add_argument("--demo-ego", action="store_true", default=True)
    ap.add_argument("--ego-begin", type=int, default=60)  # 흐름은 출발시각 순으로 적어야 함
    ap.add_argument("--ego-period", type=int, default=300)
    ap.add_argument("--zoom", type=int, default=250)
    ap.add_argument("--view-x", type=float, default=1100)
    ap.add_argument("--view-y", type=float, default=-30)
    a = ap.parse_args()
    build(a.net, a.out, a)
