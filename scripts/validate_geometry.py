"""도로 형상(geometry) 재현도 검증 — 시뮬레이션 도로가 실제 청계천로와 얼마나 같은가.

코드·환경은 건드리지 않고, 생성된 SUMO 파일(env/sumo/cgc.*)을 읽어 실제 기준값과 비교한다.

기준값(실제)
  - 차로 수        : data/road_data/cgc_lanes.csv (로드뷰 실측)
  - 정류장 좌표    : 청계A01_정류소_좌표*.csv (경도/위도 + 이전정류소_직선거리_m)
  - 어린이보호구역 : 로드뷰 노선 거리 1333 m(기점) ~ 1610 m(종점)
  - 정차금지지대   : data/road_data/no_stop_zones.csv (엣지 내 위치만 있음, 노선 거리 기준값 없음)
  - 노선 지점      : 무교로 161, 광교 463, 삼일대로 911, 청계3가 1333, 세운상가 1610,
                     배오개다리 1824, 광장시장 2146 (m, 로드뷰)

시뮬레이션
  - env/sumo/cgc.net.xml, cgc.add.xml(busStop), cgc.rou.xml의 동행 ego 경로 r0 (= A01_east)
  - 이 net.xml은 PROJ 없이 tools/osm_to_sumo.project()의 등장방형 투영으로 만들어져
    projParameter="!"이다. 그래서 sumolib convertXY2LonLat 대신 project()의 역변환을 쓰고,
    OSM 노드 ID가 남아 있는 교차로로 역변환이 맞는지 먼저 검증한다.
  - 도로망 정리로 사라진 엣지 ID는 88bd499 시점 net.xml(git)의 같은 엣지 중앙점을
    현재 경로의 가장 가까운 엣지/교차로 내부 차로에 공간 매칭한다 (두 net은 같은 투영).

실행:  py -X utf8 scripts/validate_geometry.py [--rebuild]
출력:  data_huiyoung/report/geometry_validation.md / .csv
"""
import argparse
import csv
import glob
import math
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "tools"))
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import sumolib  # noqa: E402
from sumolib.geomhelper import (distancePointToPolygon,  # noqa: E402
                                positionAtShapeOffset, polyLength)
import osm_to_sumo  # noqa: E402  (ORIGIN, project — net 좌표를 만든 그 투영)

SUMO_DIR = os.path.join(BASE, "env", "sumo")
NET = os.path.join(SUMO_DIR, "cgc.net.xml")
ADD = os.path.join(SUMO_DIR, "cgc.add.xml")
ROU = os.path.join(SUMO_DIR, "cgc.rou.xml")
OSM = os.path.join(BASE, "data", "osm", "cheonggyecheon.osm")
LANES_CSV = os.path.join(BASE, "data", "road_data", "cgc_lanes.csv")
NO_STOP_CSV = os.path.join(BASE, "data", "road_data", "no_stop_zones.csv")
SCHOOL_CSV = os.path.join(BASE, "data", "road_data", "school_zones.csv")
OLD_NET_REV = "88bd499"                      # 도로망 정리(bus_line) 이전 net
OLD_NET_PATH = "env/cheonggyecheon/cgc.net.xml"
OUT_DIR = os.path.join(BASE, "data_huiyoung", "report")

ROUTE_ID = "r0"
ROUTE_REAL_LENGTH = 2146.0
SCHOOL_REAL = (1333.0, 1610.0)
POS_TOL = 15.0                               # 위치 오차 판정 기준 (m)
MATCH_TOL = 5.0                              # 옛 엣지 → 현재 경로 공간 매칭 허용 횡거리 (m)

# 노선 지점: (이름, 로드뷰 거리 m, SUMO에서 찾는 방법, 방법 설명)
#   cross     : 경로 교차로 중 그 이름의 도로가 만나는 곳 (교차로 중심 = 진입 + 내부 길이/2)
#   old_edge  : cgc_lanes.csv에서 그 지점으로 적힌 옛 엣지가 매칭된 위치
#   edge_start/edge_end : 엣지 경계 (cgc_lanes.csv 사유 열의 기점/종점 표기)
#   route_end : 경로 끝
LANDMARKS = [
    ("무교로", 161.0, ("cross", "무교로"), "교차 도로명 '무교로' 교차로 중심"),
    ("광교", 463.0, ("old_edge", "e73541403_6"),
     "cgc_lanes.csv '광교 사거리' 엣지(e73541403_6)의 매칭 위치"),
    ("삼일대로", 911.0, ("cross", "삼일대로"), "교차 도로명 '삼일대로' 교차로 중심"),
    ("청계3가", 1333.0, ("edge_start", "e218924602_1"),
     "e218924602_1 시작 (cgc_lanes.csv '어린이보호구역 기점')"),
    ("세운상가", 1610.0, ("edge_end", "e218924602_3"),
     "e218924602_3 끝 (cgc_lanes.csv '어린이보호구역 종점')"),
    ("배오개다리", 1824.0, ("cross", "배오개다리"), "교차 도로명 '배오개다리' 교차로 중심"),
    ("광장시장", 2146.0, ("route_end", None), "경로 r0 끝 (cgc_lanes.csv '청계5가 직전')"),
]


# ──────────────────────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────────────────────
def f1(v):
    """숫자를 소수점 1자리 문자열로 (None이면 '-')."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.1f}"


def unproject(x, y):
    """osm_to_sumo.project()의 역변환: net x,y(m) → (lon, lat)."""
    k = math.pi / 180.0
    o_lon, o_lat = osm_to_sumo.ORIGIN
    lon = o_lon + x / (k * 6378137.0 * math.cos(o_lat * k))
    lat = o_lat + y / (k * 6378137.0)
    return lon, lat


def haversine(lon1, lat1, lon2, lat2):
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def norm_name(s):
    """정류장 이름 비교용: 괄호 표기·공백 제거."""
    return re.sub(r"\(.*?\)|\s", "", s or "")


def passenger_lanes(edge):
    return sum(1 for ln in edge.getLanes() if ln.allows("passenger"))


def judge_pos(err):
    if err is None:
        return "측정 불가"
    return "양호" if abs(err) <= POS_TOL else "미흡"


# ──────────────────────────────────────────────────────────────
# 경로: 엣지별 경로상 시작 위치 (교차로 내부 차로 길이 포함 = 차량 주행거리계와 같은 기준)
# ──────────────────────────────────────────────────────────────
def load_connections(net_path):
    """(from 엣지, fromLane, to 엣지) → via 내부 차로 id. 내부 차로끼리의 연결도 포함."""
    conns = {}
    for _, el in ET.iterparse(net_path):
        if el.tag == "connection" and el.get("via"):
            conns[(el.get("from"), el.get("fromLane"), el.get("to"))] = el.get("via")
        el.clear()
    return conns


def via_chain(net, conns, e_from, e_to):
    """e_from → e_to 사이의 교차로 내부 차로 목록 (승용차 차로 중 맨 오른쪽 기준)."""
    lanes = [ln for ln in e_from.getLanes() if ln.allows("passenger")]
    for ln in lanes:
        via = conns.get((e_from.getID(), str(ln.getIndex()), e_to.getID()))
        if not via:
            continue
        chain = []
        while via:
            chain.append(net.getLane(via))
            edge_id, idx = via.rsplit("_", 1)
            via = conns.get((edge_id, idx, e_to.getID()))
        return chain
    return []


def build_route(net, conns, edge_ids):
    """경로 요소 목록: dict(kind='edge'|'internal', id, start, length, shape, ...)."""
    items, s = [], 0.0
    for i, eid in enumerate(edge_ids):
        e = net.getEdge(eid)
        lane = next(ln for ln in e.getLanes() if ln.allows("passenger"))
        items.append(dict(kind="edge", id=eid, start=s, length=e.getLength(),
                          shape=lane.getShape(), edge=e))
        s += e.getLength()
        if i + 1 < len(edge_ids):
            chain = via_chain(net, conns, e, net.getEdge(edge_ids[i + 1]))
            node = e.getToNode()
            j_len = sum(ln.getLength() for ln in chain)
            items.append(dict(kind="internal", id=node.getID(), start=s, length=j_len,
                              shape=[p for ln in chain for p in ln.getShape()], node=node,
                              entry_edge=eid, exit_edge=edge_ids[i + 1]))
            s += j_len
    return items, s


def route_pos_of_point(items, x, y):
    """점 (x, y)를 경로에 투영: (경로상 위치 m, 횡거리 m, 매칭된 요소)."""
    best = None
    for it in items:
        shp = it["shape"]
        if len(shp) < 2:
            continue
        d = distancePointToPolygon((x, y), shp)
        if best is None or d < best[0]:
            best = (d, it)
    d, it = best
    # 요소 안에서의 위치 = 가장 가까운 선분까지의 누적 길이 (모양 길이 → 요소 길이로 비례)
    shp = it["shape"]
    acc, best_off, best_d = 0.0, 0.0, None
    for a, b in zip(shp[:-1], shp[1:]):
        seg = math.dist(a, b)
        if seg > 0:
            t = max(0.0, min(1.0, ((x - a[0]) * (b[0] - a[0]) + (y - a[1]) * (b[1] - a[1])) / seg ** 2))
            px, py = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            dd = math.hypot(x - px, y - py)
            if best_d is None or dd < best_d:
                best_d, best_off = dd, acc + t * seg
        acc += seg
    scale = it["length"] / acc if acc > 0 else 1.0
    return it["start"] + best_off * scale, d, it


def cross_names(node, route_ids):
    names = set()
    for e in node.getIncoming() + node.getOutgoing():
        if e.getID() in route_ids or e.getID().startswith(":") or not e.getName():
            continue
        names.add(e.getName())
    return sorted(names)


# ──────────────────────────────────────────────────────────────
# 기준값 로딩
# ──────────────────────────────────────────────────────────────
def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_stop_csv():
    hits = sorted(glob.glob(os.path.join(BASE, "청계A01_정류소_좌표*.csv")))
    return hits[0] if hits else None


def osm_way_tags(way_ids):
    tags = {}
    for _, el in ET.iterparse(OSM):
        if el.tag == "way" and el.get("id") in way_ids:
            tags[el.get("id")] = {t.get("k"): t.get("v") for t in el.findall("tag")}
        if el.tag in ("node", "way", "relation"):
            el.clear()
    return tags


def osm_nodes(node_ids):
    out = {}
    for _, el in ET.iterparse(OSM):
        if el.tag == "node" and el.get("id") in node_ids:
            out[el.get("id")] = (float(el.get("lon")), float(el.get("lat")))
        if el.tag in ("node", "way", "relation"):
            el.clear()
    return out


def osm_dir_lanes(tags):
    """OSM 태그에서 진행 방향 차로 수. 없으면 None."""
    if tags is None:
        return None
    if tags.get("lanes:forward"):
        return int(tags["lanes:forward"])
    if tags.get("lanes"):
        n = int(tags["lanes"])
        return n if tags.get("oneway") in ("yes", "1", "true") else max(1, n // 2)
    return None


def load_old_net(scratch):
    path = os.path.join(scratch, f"cgc_net_{OLD_NET_REV}.xml")
    if not os.path.exists(path):
        data = subprocess.run(["git", "-C", BASE, "show", f"{OLD_NET_REV}:{OLD_NET_PATH}"],
                              capture_output=True, check=True).stdout
        with open(path, "wb") as f:
            f.write(data)
    return sumolib.net.readNet(path)


def traci_route_length(net_path, edge_ids, last_len):
    """SUMO가 직접 계산한 경로 주행거리 (교차 검증용). 실패하면 None.

    처음~끝을 한 번에 물으면 SUMO가 최단시간 경로(다른 엣지)를 고르므로,
    이웃한 엣지 쌍마다 getDistanceRoad를 물어 더한다.
    """
    try:
        import traci
        traci.start([sumolib.checkBinary("sumo"), "-n", net_path, "--no-warnings", "true",
                     "--no-step-log", "true", "--end", "1"])
        try:
            total = sum(traci.simulation.getDistanceRoad(a, 0.0, b, 0.0, isDriving=True)
                        for a, b in zip(edge_ids[:-1], edge_ids[1:]))
            return float(total + last_len)
        finally:
            traci.close()
    except Exception as exc:  # noqa: BLE001
        print("traci 교차 검증 실패:", exc)
        return None


# ──────────────────────────────────────────────────────────────
# 측정
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rebuild", action="store_true",
                    help="측정 전에 road_builder로 env/sumo/cgc.*를 다시 생성 (생성 파일은 git 무시 대상)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    if args.rebuild:
        from env.road_config import ROAD, EGO, TRAFFIC
        from env import road_builder
        road_builder.build(ROAD, EGO, TRAFFIC, SUMO_DIR)

    os.makedirs(args.out_dir, exist_ok=True)
    scratch = os.path.join(tempfile.gettempdir(), "cgc_validate_geometry")   # 옛 net 사본
    os.makedirs(scratch, exist_ok=True)

    head = subprocess.run(["git", "-C", BASE, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    net_mtime = datetime.fromtimestamp(os.path.getmtime(NET)).strftime("%Y-%m-%d %H:%M")

    net = sumolib.net.readNet(NET, withInternal=True)
    conns = load_connections(NET)
    rou = open(ROU, encoding="utf-8").read()
    m = re.search(rf'<route id="{ROUTE_ID}" edges="([^"]*)"', rou)
    route_ids = m.group(1).split()
    route_set = set(route_ids)
    items, route_len = build_route(net, conns, route_ids)
    edge_items = {it["id"]: it for it in items if it["kind"] == "edge"}
    old_net = load_old_net(scratch)

    rows = []          # CSV 상세표: 항목, 대상, 실제, 시뮬레이션, 오차, 단위, 판정, 비고
    unmeasured = []    # (항목, 이유)

    def add(cat, target, real, sim, err, unit, verdict, note=""):
        rows.append(dict(항목=cat, 대상=target,
                         실제=real if isinstance(real, str) else f1(real),
                         시뮬레이션=sim if isinstance(sim, str) else f1(sim),
                         오차=err if isinstance(err, str) else f1(err),
                         단위=unit, 판정=verdict, 비고=note))

    # ---- 0. 투영 검증: OSM 노드 ID가 남은 경로 교차로 ----
    node_ids = {}
    for it in items:
        for nd in (it["edge"].getFromNode(), it["edge"].getToNode()) if it["kind"] == "edge" else ():
            mm = re.fullmatch(r"n(\d+)", nd.getID())
            if mm:
                node_ids[mm.group(1)] = nd
    osm_xy = osm_nodes(set(node_ids))
    proj_res = []
    for oid, nd in node_ids.items():
        if oid in osm_xy:
            px, py = osm_to_sumo.project(*osm_xy[oid])
            nx, ny = nd.getCoord()
            proj_res.append(math.hypot(px - nx, py - ny))
            # 역변환 왕복 확인
            lon, lat = unproject(nx, ny)
            assert abs(haversine(lon, lat, *osm_xy[oid]) - proj_res[-1]) < 0.5
    proj_res.sort()
    proj_note = (f"OSM 노드 {len(proj_res)}개: 중앙값 {f1(proj_res[len(proj_res) // 2])} m, "
                 f"최대 {f1(proj_res[-1])} m" if proj_res else "검증 노드 없음")

    # ---- 1. 차로 수 ----
    lane_rows = read_csv(LANES_CSV)
    way_tags = osm_way_tags({r["edge_id"].split("_")[0][1:] for r in lane_rows})
    lane_detail = []
    for r in lane_rows:
        eid, real = r["edge_id"].strip(), int(r["lanes"])
        osm_v = osm_dir_lanes(way_tags.get(eid.split("_")[0][1:]))
        old_v = passenger_lanes(old_net.getEdge(eid)) if old_net.hasEdge(eid) else None
        status, cur_v, match_note = "", None, ""
        if net.hasEdge(eid):
            cur_v = passenger_lanes(net.getEdge(eid))
            status = "ID 유지" + ("" if eid in route_set else " (경로 밖)")
        elif old_net.hasEdge(eid):
            oe = old_net.getEdge(eid)
            shp = oe.getShape()
            mx, my = positionAtShapeOffset(shp, polyLength(shp) / 2.0)
            pos, dist, it = route_pos_of_point(items, mx, my)
            if dist > MATCH_TOL:
                status = f"경로 밖 (가장 가까운 경로 요소까지 {f1(dist)} m)"
            elif it["kind"] == "edge":
                cur_v = passenger_lanes(it["edge"])
                status = f"ID 변경 → {it['id']}"
            else:
                status = f"교차로 내부로 병합 ({it['entry_edge']}→{it['exit_edge']})"
            match_note = (f"옛 길이 {f1(oe.getLength())} m, 중앙점이 경로 {f1(pos)} m에 매칭 "
                          f"(횡거리 {f1(dist)} m)")
        else:
            status = "옛·현재 net 모두 없음"
        lane_detail.append(dict(edge=eid, real=real, osm=osm_v, old=old_v, cur=cur_v,
                                status=status, note=match_note or r.get("reason", "")))
        verdict = "-" if cur_v is None else ("일치" if cur_v == real else "불일치")
        add("1.차로 수", eid, f"{real}", "-" if cur_v is None else f"{cur_v}",
            "-" if cur_v is None else f"{cur_v - real:+d}", "차로", verdict,
            f"{status}; OSM {osm_v if osm_v is not None else '태그 없음'}; "
            f"{OLD_NET_REV} net {old_v if old_v is not None else '-'}; {match_note}".strip("; "))
    n_all = len(lane_detail)
    osm_ok = sum(1 for d in lane_detail if d["osm"] == d["real"])
    old_cmp = [d for d in lane_detail if d["old"] is not None]
    old_ok = sum(1 for d in old_cmp if d["old"] == d["real"])
    cur_cmp = [d for d in lane_detail if d["cur"] is not None]
    cur_ok = sum(1 for d in cur_cmp if d["cur"] == d["real"])
    kept = [d for d in lane_detail if d["status"].startswith("ID 유지")]
    kept_ok = sum(1 for d in kept if d["cur"] == d["real"])
    merged = [d for d in lane_detail if d["status"].startswith("교차로 내부")]
    # 현재 경로 엣지 중 CSV에 없는 것
    not_in_csv = [e for e in route_ids if e not in {d["edge"] for d in lane_detail}]

    # ---- 2. 정류장 ----
    stop_csv = find_stop_csv()
    stop_detail, stop_pairs = [], []
    add_root = ET.parse(ADD).getroot()
    bus_stops = {b.get("id"): b for b in add_root.iter("busStop")}
    old_add = subprocess.run(["git", "-C", BASE, "show", f"{OLD_NET_REV}:env/cheonggyecheon/cgc.add.xml"],
                             capture_output=True).stdout
    old_stops = ({b.get("id"): b for b in ET.fromstring(old_add).iter("busStop")}
                 if old_add else {})
    route_stops = [b for b in bus_stops.values() if b.get("lane").rsplit("_", 1)[0] in route_set]
    route_stops.sort(key=lambda b: edge_items[b.get("lane").rsplit("_", 1)[0]]["start"]
                     + float(b.get("startPos")))
    if stop_csv:
        real_stops = [r for r in read_csv(stop_csv) if r["방향"].strip() == "갈 때"]
        prev = None
        for i, rs in enumerate(real_stops):
            rl, rt = float(rs["경도_X"]), float(rs["위도_Y"])
            bs = route_stops[i] if i < len(route_stops) else None
            if bs is None or norm_name(bs.get("name")) != norm_name(rs["정류소명"]):
                unmeasured.append((f"정류장 {rs['정류소명']}", "이름이 같은 SUMO 정류장을 찾지 못함"))
                continue
            lane = net.getLane(bs.get("lane"))
            mid = (float(bs.get("startPos")) + float(bs.get("endPos"))) / 2.0
            shp = lane.getShape()
            sx, sy = positionAtShapeOffset(shp, mid * polyLength(shp) / lane.getLength())
            slon, slat = unproject(sx, sy)
            err = haversine(rl, rt, slon, slat)
            rpos = edge_items[lane.getEdge().getID()]["start"] + mid
            # 실측 좌표를 현재 경로에 투영한 위치 (정류장이 경로를 따라 얼마나 어긋났는지)
            real_pos, real_lat, _ = route_pos_of_point(items, *osm_to_sumo.project(rl, rt))
            # 같은 정류장의 88bd499 net 위치 오차 (도로망 정리 전후 비교)
            old_err = None
            ob = old_stops.get(bs.get("id"))
            if ob is not None and old_net.hasEdge(ob.get("lane").rsplit("_", 1)[0]):
                olane = old_net.getLane(ob.get("lane"))
                omid = (float(ob.get("startPos")) + float(ob.get("endPos"))) / 2.0
                oshp = olane.getShape()
                ox, oy = positionAtShapeOffset(oshp, omid * polyLength(oshp) / olane.getLength())
                old_err = haversine(rl, rt, *unproject(ox, oy))
            stop_detail.append(dict(name=rs["정류소명"], id=bs.get("id"), err=err, route_pos=rpos,
                                    real=(rl, rt), sim=(slon, slat), real_pos=real_pos,
                                    real_lat=real_lat, old_err=old_err,
                                    lane=bs.get("lane"), old_lane=ob.get("lane") if ob is not None else None))
            add("2.정류장 위치", f"{bs.get('id')} {rs['정류소명']}",
                f"({rl:.6f}, {rt:.6f})", f"({slon:.6f}, {slat:.6f})", err, "m", judge_pos(err),
                f"busStop 중심 = {bs.get('lane')} {f1(mid)} m, 경로상 {f1(rpos)} m; "
                f"실측 좌표의 경로 투영 {f1(real_pos)} m (경로 방향 차이 {f1(rpos - real_pos)} m); "
                f"{OLD_NET_REV} net 오차 {f1(old_err)} m")
            if prev is not None and rs["이전정류소_직선거리_m"].strip():
                real_d = float(rs["이전정류소_직선거리_m"])
                sim_d = haversine(*prev["sim"], slon, slat)
                coord_d = haversine(*prev["real"], rl, rt)
                stop_pairs.append(dict(pair=f"{prev['name']} → {rs['정류소명']}", real=real_d,
                                       sim=sim_d, coord=coord_d, err=sim_d - real_d,
                                       pct=(sim_d - real_d) / real_d * 100.0))
                add("2.정류장 간 직선거리", f"{prev['name']} → {rs['정류소명']}", real_d, sim_d,
                    sim_d - real_d, "m", judge_pos(sim_d - real_d),
                    f"오차율 {f1((sim_d - real_d) / real_d * 100.0)}%, 실제 좌표로 계산한 직선거리 {f1(coord_d)} m")
            prev = stop_detail[-1]
    else:
        unmeasured.append(("정류장 위치", "청계A01_정류소_좌표 CSV를 찾지 못함"))

    # ---- 3. 노선 길이 · 지점 ----
    traci_len = traci_route_length(NET, route_ids, net.getEdge(route_ids[-1]).getLength())
    add("3.노선 길이", f"경로 {ROUTE_ID} 총 길이", ROUTE_REAL_LENGTH, route_len,
        route_len - ROUTE_REAL_LENGTH, "m", judge_pos(route_len - ROUTE_REAL_LENGTH),
        f"오차율 {f1((route_len - ROUTE_REAL_LENGTH) / ROUTE_REAL_LENGTH * 100.0)}%; "
        f"엣지 {f1(sum(it['length'] for it in items if it['kind'] == 'edge'))} m + 교차로 내부 "
        f"{f1(sum(it['length'] for it in items if it['kind'] == 'internal'))} m; "
        f"traci 주행거리 {f1(traci_len)} m")
    # 기점: 옛 출발 엣지 e73541403_1 (cgc_lanes.csv '청계광장 출발')
    origin_note = ""
    if not net.hasEdge("e73541403_1") or "e73541403_1" not in route_set:
        if old_net.hasEdge("e73541403_1"):
            oe = old_net.getEdge("e73541403_1")
            sx, sy = oe.getShape()[0]
            r0x, r0y = items[0]["shape"][0]
            origin_note = (f"cgc_lanes.csv의 출발 엣지 e73541403_1(옛 길이 {f1(oe.getLength())} m)이 "
                           f"현재 경로에 없음. 옛 출발점 ↔ 현재 경로 시작점 직선거리 "
                           f"{f1(math.hypot(sx - r0x, sy - r0y))} m")
    lm_detail = []
    for name, real, (how, key), desc in LANDMARKS:
        sim, extra = None, ""
        if how == "cross":
            hits = [it for it in items if it["kind"] == "internal"
                    and key in cross_names(it["node"], route_set)]
            if hits:
                it = hits[0]
                sim = it["start"] + it["length"] / 2.0
                extra = (f"교차로 {it['id'][:40]}: 진입 {f1(it['start'])} m, "
                         f"진출 {f1(it['start'] + it['length'])} m")
        elif how == "old_edge":
            if old_net.hasEdge(key):
                shp = old_net.getEdge(key).getShape()
                mx, my = positionAtShapeOffset(shp, polyLength(shp) / 2.0)
                sim, d, it = route_pos_of_point(items, mx, my)
                cn = cross_names(it["node"], route_set) if it["kind"] == "internal" else []
                extra = (f"매칭 {it['kind']} {it['id'][:40]} (횡거리 {f1(d)} m), "
                         f"교차 도로 {', '.join(cn) or '-'}")
        elif how == "edge_start" and key in edge_items:
            sim = edge_items[key]["start"]
        elif how == "edge_end" and key in edge_items:
            sim = edge_items[key]["start"] + edge_items[key]["length"]
        elif how == "route_end":
            sim = route_len
        if sim is None:
            unmeasured.append((f"지점 {name}", f"SUMO에서 기준 지점을 찾지 못함 ({desc})"))
        err = None if sim is None else sim - real
        lm_detail.append(dict(name=name, real=real, sim=sim, err=err, desc=desc, extra=extra))
        add("3.노선 지점", name, real, sim, err, "m", judge_pos(err), f"{desc}. {extra}".strip())

    # ---- 4. 특수 구간 ----
    school_rows = read_csv(SCHOOL_CSV)
    sewoon = [r["edge_id"] for r in school_rows if "세운상가" in r["zone_name"]]
    other_zone = [r["edge_id"] for r in school_rows if "세운상가" not in r["zone_name"]]
    zone_detail = {}
    on_route = [e for e in sewoon if e in edge_items]
    if on_route:
        z0 = edge_items[on_route[0]]["start"]
        z1 = edge_items[on_route[-1]]["start"] + edge_items[on_route[-1]]["length"]
        enforced = sum(edge_items[e]["length"] for e in on_route)
        zone_detail = dict(start=z0, end=z1, enforced=enforced, edges=on_route,
                           missing=[e for e in sewoon if e not in edge_items])
        add("4.어린이보호구역", "시작 (세운상가)", SCHOOL_REAL[0], z0, z0 - SCHOOL_REAL[0], "m",
            judge_pos(z0 - SCHOOL_REAL[0]), f"school_zones.csv 첫 엣지 {on_route[0]} 시작")
        add("4.어린이보호구역", "끝 (세운상가)", SCHOOL_REAL[1], z1, z1 - SCHOOL_REAL[1], "m",
            judge_pos(z1 - SCHOOL_REAL[1]), f"school_zones.csv 마지막 엣지 {on_route[-1]} 끝")
        add("4.어린이보호구역", "구간 길이 (세운상가)", SCHOOL_REAL[1] - SCHOOL_REAL[0], z1 - z0,
            (z1 - z0) - (SCHOOL_REAL[1] - SCHOOL_REAL[0]), "m",
            judge_pos((z1 - z0) - (SCHOOL_REAL[1] - SCHOOL_REAL[0])),
            f"이 중 감점이 실제로 적용되는 일반 엣지 길이 {f1(enforced)} m "
            f"(교차로 내부 차로에서는 감점 안 함, sumo_env.step)")
    other_on = [e for e in other_zone if e in edge_items]
    if other_on:
        o0 = edge_items[other_on[0]]["start"]
        o1 = edge_items[other_on[-1]]["start"] + edge_items[other_on[-1]]["length"]
        add("4.어린이보호구역", "배오개 보호구역 (school_zones.csv)", "기준값 없음", f"{f1(o0)}~{f1(o1)}",
            "-", "m", "측정 불가",
            f"엣지 {', '.join(other_on)}; 없는 엣지 {', '.join(e for e in other_zone if e not in edge_items) or '-'}")
    # 정류장 그림(id가 bs로 시작: 쉘터·표지판·핀 등)이 아닌 폴리곤 = 노면 표시 후보
    polys = list(add_root.iter("poly"))
    n_zone_poly = sum(1 for p in polys if not (p.get("id") or "").startswith("bs"))
    nsz_detail = []
    for r in read_csv(NO_STOP_CSV):
        if not (r.get("edge_id") or "").strip():
            jid = (r.get("junction_id") or "").strip()
            nsz_detail.append(dict(edge=f"교차로 {jid[:40]}", a=0.0, b=0.0, s0=None, s1=None, len=None,
                                   note="교차로 내부 전체 (r0 경로 밖일 수 있음)", reason=r["reason"]))
            add("4.정차금지지대", f"교차로 {jid}", "기준값 없음", "-", "-", "m", "측정 불가",
                f"{r['reason']}; 교차로 단위")
            continue
        eid, a, b = r["edge_id"], float(r["start_pos"]), float(r["end_pos"])
        if eid not in edge_items:
            nsz_detail.append(dict(edge=eid, a=a, b=b, s0=None, s1=None, len=None,
                                   note="현재 경로에 엣지 없음", reason=r["reason"]))
            add("4.정차금지지대", f"{eid} {a:.0f}~{b:.0f} m", "기준값 없음", "-", "-", "m", "측정 불가",
                f"{r['reason']}; 현재 경로에 엣지 없음")
            continue
        it = edge_items[eid]
        elen = it["length"]
        note = ""
        if old_net.hasEdge(eid):
            oe = old_net.getEdge(eid)
            shift = math.dist(oe.getShape()[0], it["edge"].getShape()[0])
            if abs(oe.getLength() - elen) > 1.0 or shift > 1.0:
                note = (f"{OLD_NET_REV} net에서 이 엣지 길이 {f1(oe.getLength())} m → 현재 {f1(elen)} m, "
                        f"시작점 이동 {f1(shift)} m. ")
        if a >= elen:
            note += f"엣지 길이 {f1(elen)} m보다 시작 위치가 뒤 → 구간이 엣지 밖이라 적용되지 않음"
        elif b > elen:
            note += f"엣지 길이 {f1(elen)} m를 넘음 → {f1(elen - a)} m만 적용"
        s0, s1 = it["start"] + a, it["start"] + min(b, elen)
        nsz_detail.append(dict(edge=eid, a=a, b=b, s0=s0, s1=s1, len=elen, note=note,
                               reason=r["reason"]))
        add("4.정차금지지대", f"{eid} {a:.0f}~{b:.0f} m", "기준값 없음",
            f"{f1(s0)}~{f1(s1)}" if a < elen else "-", "-", "m",
            "적용 안 됨" if a >= elen else "측정 불가(기준값 없음)",
            f"{r['reason']}; 엣지 길이 {f1(elen)} m. {note}".strip())

    # ---- 5. 개수 ----
    tls = set()
    for i in range(len(route_ids) - 1):
        e, nxt = net.getEdge(route_ids[i]), net.getEdge(route_ids[i + 1])
        for c in e.getConnections(nxt):
            if c.getTLSID():
                tls.add(c.getTLSID())
    crossings = set()
    route_nodes = {it["id"] for it in items if it["kind"] == "internal"} | {
        net.getEdge(route_ids[0]).getFromNode().getID(), net.getEdge(route_ids[-1]).getToNode().getID()}
    for _, el in ET.iterparse(NET):
        if el.tag == "edge" and el.get("function") == "crossing":
            ce = set((el.get("crossingEdges") or "").split())
            if ce & route_set:
                crossings.add(el.get("id"))
        el.clear()
    counts = [("경로상 신호 교차로", len(tls), f"신호 {len(tls)}개 (TLS id 기준)"),
              ("경로상 정류장", len(route_stops), ", ".join(b.get("id") for b in route_stops)),
              ("경로상 횡단보도", len(crossings), "crossingEdges가 경로 엣지를 가로지르는 횡단보도")]
    for name, n, note in counts:
        add("5.개수", name, "", f"{n}", "", "개", "-", note)

    # ──────────────────────────────────────────────────────────────
    # 출력
    # ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "geometry_validation.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    L = []
    L.append("# 도로 형상 재현도 검증 (청계천로 동행 A01_east)\n")
    L.append(f"- 측정 대상: `env/sumo/cgc.net.xml` (생성 {net_mtime}, 코드 HEAD `{head}`), "
             f"경로 `{ROUTE_ID}` 엣지 {len(route_ids)}개")
    L.append(f"- 판정 기준: 차로 수 100% 일치 / 위치 오차 {POS_TOL:.0f} m 이내 → 양호")
    L.append(f"- 좌표 변환: net.xml에 투영 정보가 없어(`projParameter=\"!\"`) "
             f"`tools/osm_to_sumo.project()`(원점 {osm_to_sumo.ORIGIN})의 역변환 사용. "
             f"투영 검증 — {proj_note}")
    L.append("- 생성 스크립트: `scripts/validate_geometry.py`, 원본 표: `geometry_validation.csv`\n")

    # 요약표
    L.append("## 요약\n")
    L.append("| 항목 | 실제 | 시뮬레이션 | 오차 | 판정 |")
    L.append("|---|---|---|---|---|")
    pct = lambda a, b: f1(a / b * 100.0) if b else "-"  # noqa: E731
    L.append(f"| 차로 수 일치율 (OSM 원본, {n_all}구간) | 로드뷰 {n_all}구간 | {osm_ok}/{n_all} 일치 | "
             f"{pct(osm_ok, n_all)}% | {'양호' if osm_ok == n_all else '미흡'} |")
    L.append(f"| 차로 수 일치율 (참고: {OLD_NET_REV} 커밋 net) | 로드뷰 {len(old_cmp)}구간 | "
             f"{old_ok}/{len(old_cmp)} 일치 | {pct(old_ok, len(old_cmp))}% | "
             f"{'양호' if old_ok == len(old_cmp) else '미흡'} |")
    L.append(f"| 차로 수 일치율 (현재 net, 비교 가능 구간) | 로드뷰 {len(cur_cmp)}구간 | "
             f"{cur_ok}/{len(cur_cmp)} 일치 | {pct(cur_ok, len(cur_cmp))}% | "
             f"{'양호' if cur_ok == len(cur_cmp) else '미흡'} |")
    if stop_detail:
        errs = [d["err"] for d in stop_detail]
        L.append(f"| 정류장 위치 오차 ({len(errs)}곳) | 실측 좌표 | 평균 {f1(sum(errs) / len(errs))} m, "
                 f"최대 {f1(max(errs))} m | 최대 {f1(max(errs))} m | {judge_pos(max(errs))} |")
    if stop_pairs:
        worst = max(stop_pairs, key=lambda p: abs(p["err"]))
        L.append(f"| 정류장 간 직선거리 ({len(stop_pairs)}구간) | CSV 값 | "
                 f"평균 오차율 {f1(sum(abs(p['pct']) for p in stop_pairs) / len(stop_pairs))}% | "
                 f"최대 {f1(worst['err'])} m ({f1(worst['pct'])}%) | {judge_pos(worst['err'])} |")
    L.append(f"| 노선 총 길이 | {f1(ROUTE_REAL_LENGTH)} m | {f1(route_len)} m | "
             f"{f1(route_len - ROUTE_REAL_LENGTH)} m ({f1((route_len - ROUTE_REAL_LENGTH) / ROUTE_REAL_LENGTH * 100)}%) | "
             f"{judge_pos(route_len - ROUTE_REAL_LENGTH)} |")
    lm_ok = [d for d in lm_detail if d["err"] is not None]
    if lm_ok:
        worst = max(lm_ok, key=lambda d: abs(d["err"]))
        n_good = sum(1 for d in lm_ok if abs(d["err"]) <= POS_TOL)
        L.append(f"| 노선 지점 위치 ({len(lm_ok)}곳) | 로드뷰 거리 | 양호 {n_good}/{len(lm_ok)} | "
                 f"최대 {f1(worst['err'])} m ({worst['name']}) | {judge_pos(worst['err'])} |")
    if zone_detail:
        e0 = zone_detail["start"] - SCHOOL_REAL[0]
        e1 = zone_detail["end"] - SCHOOL_REAL[1]
        L.append(f"| 어린이보호구역 시작 / 끝 | {f1(SCHOOL_REAL[0])} / {f1(SCHOOL_REAL[1])} m | "
                 f"{f1(zone_detail['start'])} / {f1(zone_detail['end'])} m | {f1(e0)} / {f1(e1)} m | "
                 f"{'양호' if max(abs(e0), abs(e1)) <= POS_TOL else '미흡'} |")
    n_dead = sum(1 for d in nsz_detail if d["s0"] is not None and d["a"] >= d["len"])
    L.append(f"| 정차금지지대 ({len(nsz_detail)}곳) | 기준값 없음 | 경로 위 {sum(1 for d in nsz_detail if d['s0'] is not None)}곳, "
             f"엣지 밖 {n_dead}곳 | - | 측정 불가 |")
    for name, n, _ in counts:
        L.append(f"| {name} 수 |  | {n} |  | - |")

    # 상세 1
    L.append("\n## 1. 차로 수\n")
    L.append(f"보정 전(OSM 원본) **{pct(osm_ok, n_all)}%** → 보정 후(현재 net) "
             f"**{pct(cur_ok, len(cur_cmp))}%** (비교 가능한 {len(cur_cmp)}구간 기준). "
             f"ID가 그대로 남은 구간만 보면 {kept_ok}/{len(kept)} ({pct(kept_ok, len(kept))}%).\n")
    L.append(f"참고: 저장소에 커밋된 {OLD_NET_REV} 시점 net.xml(`{OLD_NET_PATH}`)은 "
             f"{pct(old_ok, len(old_cmp))}%로 OSM 원본과 같다. 차로 수 보정(cgc_lanes.csv)이 "
             "반영되기 전에 저장된 파일로 보인다. 현재 net은 road_builder가 매번 보정을 적용해 새로 만든다.\n")
    L.append("차로 수 = 승용차가 다닐 수 있는 차로 수 (net.xml의 lane 0 보도는 제외). "
             "OSM 값은 way의 `lanes` 태그 (편도 `oneway=yes`라 그대로 진행 방향 차로 수).\n")
    L.append(f"| 엣지 (cgc_lanes.csv) | 로드뷰 | OSM 원본 | {OLD_NET_REV} net | 현재 net | 판정 | 현재 net에서 | 비고 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for d in lane_detail:
        v = "-" if d["cur"] is None else ("일치" if d["cur"] == d["real"] else "불일치")
        L.append(f"| {d['edge']} | {d['real']} | {d['osm'] if d['osm'] is not None else '태그 없음'} | "
                 f"{d['old'] if d['old'] is not None else '-'} | {d['cur'] if d['cur'] is not None else '-'} | "
                 f"{v} | {d['status']} | {d['note']} |")
    if merged:
        L.append(f"\n도로망 정리(`8fd2371`, `1352c0a` 교차로 병합)로 {len(merged)}개 엣지가 교차로 내부 차로로 "
                 "흡수되어 엣지 단위 차로 수를 비교할 수 없다 (분모에서 제외).")
    if not_in_csv:
        L.append(f"\n현재 경로 엣지 중 cgc_lanes.csv에 없는 엣지: {', '.join(not_in_csv)} (기준값 없음)")

    # 상세 2
    L.append("\n## 2. 정류장 위치 (동행 '갈 때' 5곳)\n")
    L.append("| 정류장 | SUMO id (차로) | 실제 (경도, 위도) | SUMO (경도, 위도) | 오차 (m) | 판정 | "
             "SUMO 경로상 (m) | 실측 좌표의 경로 투영 (m) | 경로 방향 차이 (m) | "
             f"{OLD_NET_REV} net 오차 (m) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for d in stop_detail:
        L.append(f"| {d['name']} | {d['id']} (`{d['lane']}`) | {d['real'][0]:.6f}, {d['real'][1]:.6f} | "
                 f"{d['sim'][0]:.6f}, {d['sim'][1]:.6f} | {f1(d['err'])} | {judge_pos(d['err'])} | "
                 f"{f1(d['route_pos'])} | {f1(d['real_pos'])} | {f1(d['route_pos'] - d['real_pos'])} | "
                 f"{f1(d['old_err'])} |")
    if stop_detail:
        errs = [d["err"] for d in stop_detail]
        L.append(f"\n평균 {f1(sum(errs) / len(errs))} m, 최대 {f1(max(errs))} m. "
                 "SUMO 좌표 = busStop 시작·끝 중간 지점의 차로 중심선. "
                 "'실측 좌표의 경로 투영' = 실제 정류장 좌표에서 현재 경로에 내린 수선의 발 위치.")
        moved = [d for d in stop_detail if d["old_err"] is not None
                 and d["err"] > POS_TOL and d["old_err"] <= POS_TOL]
        for d in moved:
            L.append(f"\n- **{d['name']} ({d['id']})**: {OLD_NET_REV} net에서는 `{d['old_lane']}`에 있어 오차 "
                     f"{f1(d['old_err'])} m였는데, 현재는 `{d['lane']}`에 놓여 {f1(d['err'])} m 어긋난다. "
                     f"경로 방향으로 {f1(d['route_pos'] - d['real_pos'])} m 뒤쪽이다. 도로망 정리 후 정류장 "
                     "엣지 배치(make_cgc_scenario)가 바뀐 것으로 보인다.")
        L.append("")
    L.append("| 구간 | CSV 직선거리 (m) | SUMO 직선거리 (m) | 오차 (m) | 오차율 (%) | 실제 좌표로 계산 (m) | 판정 |")
    L.append("|---|---|---|---|---|---|---|")
    for p in stop_pairs:
        L.append(f"| {p['pair']} | {f1(p['real'])} | {f1(p['sim'])} | {f1(p['err'])} | {f1(p['pct'])} | "
                 f"{f1(p['coord'])} | {judge_pos(p['err'])} |")

    # 상세 3
    L.append("\n## 3. 노선 길이 · 지점\n")
    L.append(f"- 경로 `{ROUTE_ID}` 총 길이 **{f1(route_len)} m** (일반 엣지 "
             f"{f1(sum(it['length'] for it in items if it['kind'] == 'edge'))} m + 교차로 내부 차로 "
             f"{f1(sum(it['length'] for it in items if it['kind'] == 'internal'))} m) vs 실제 {f1(ROUTE_REAL_LENGTH)} m → "
             f"오차 {f1(route_len - ROUTE_REAL_LENGTH)} m ({f1((route_len - ROUTE_REAL_LENGTH) / ROUTE_REAL_LENGTH * 100)}%)")
    L.append(f"- 교차 검증: SUMO(traci `getDistanceRoad`)가 계산한 같은 구간 주행거리 {f1(traci_len)} m")
    L.append("- 경로상 위치 = 차량 주행거리계와 같은 기준 (교차로 내부 차로 길이 포함)")
    if origin_note:
        L.append(f"- 기점 주의: {origin_note}. 실제 거리의 0 m(로드뷰 기점)와 SUMO 경로의 0 m가 "
                 "같은 지점인지는 저장소 자료로 확인할 수 없어 보정하지 않았다.")
    lm_errs = [d["err"] for d in lm_detail if d["err"] is not None]
    bias = sum(lm_errs) / len(lm_errs) if lm_errs else 0.0
    L.append("\n| 지점 | 실제 (m) | SUMO (m) | 오차 (m) | 판정 | 참고: 평균 오프셋 제거 후 (m) | SUMO 기준 지점 | 비고 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for d in lm_detail:
        resid = None if d["err"] is None else d["err"] - bias
        L.append(f"| {d['name']} | {f1(d['real'])} | {f1(d['sim'])} | {f1(d['err'])} | {judge_pos(d['err'])} | "
                 f"{f1(resid)} | {d['desc']} | {d['extra']} |")
    if lm_errs:
        L.append(f"\n오차가 모두 같은 쪽(SUMO가 짧음)이고 평균 {f1(bias)} m다. 기점 차이 같은 공통 오프셋이 있다고 "
                 "가정하면 지점 사이 간격의 오차는 '평균 오프셋 제거 후' 열 정도다. "
                 "이 열은 참고용이고 판정에는 쓰지 않았다 (기점 위치 기준값이 없어 오프셋을 확정할 수 없음).")
    L.append("\n경로 요소별 위치 (참고):\n")
    L.append("| 요소 | id | 경로상 시작 (m) | 길이 (m) | 승용차 차로 | 교차 도로 |")
    L.append("|---|---|---|---|---|---|")
    for it in items:
        if it["kind"] == "edge":
            L.append(f"| 엣지 | {it['id']} | {f1(it['start'])} | {f1(it['length'])} | "
                     f"{passenger_lanes(it['edge'])} | |")
        else:
            L.append(f"| 교차로 | {it['id'][:48]} | {f1(it['start'])} | {f1(it['length'])} | | "
                     f"{', '.join(cross_names(it['node'], route_set))} |")

    # 상세 4
    L.append("\n## 4. 특수 구간\n")
    L.append("### 어린이보호구역\n")
    if zone_detail:
        L.append("| 구분 | 실제 (m) | SUMO (m) | 오차 (m) | 판정 |")
        L.append("|---|---|---|---|---|")
        L.append(f"| 시작 | {f1(SCHOOL_REAL[0])} | {f1(zone_detail['start'])} | "
                 f"{f1(zone_detail['start'] - SCHOOL_REAL[0])} | {judge_pos(zone_detail['start'] - SCHOOL_REAL[0])} |")
        L.append(f"| 끝 | {f1(SCHOOL_REAL[1])} | {f1(zone_detail['end'])} | "
                 f"{f1(zone_detail['end'] - SCHOOL_REAL[1])} | {judge_pos(zone_detail['end'] - SCHOOL_REAL[1])} |")
        L.append(f"| 길이 | {f1(SCHOOL_REAL[1] - SCHOOL_REAL[0])} | {f1(zone_detail['end'] - zone_detail['start'])} | "
                 f"{f1((zone_detail['end'] - zone_detail['start']) - (SCHOOL_REAL[1] - SCHOOL_REAL[0]))} | - |")
        L.append(f"\n- SUMO 보호구역 = `school_zones.csv`의 '세운상가' 엣지 {', '.join(zone_detail['edges'])} "
                 "(감점 `school_zone_speeding_penalty`가 이 엣지 기준으로 적용).")
        L.append(f"- 이 구간 {f1(zone_detail['end'] - zone_detail['start'])} m 중 일반 엣지는 "
                 f"{f1(zone_detail['enforced'])} m뿐이고 나머지는 교차로 내부 차로다. "
                 "sumo_env는 교차로 내부(`:`로 시작하는 엣지)에서 보호구역 감점을 하지 않으므로 "
                 "실제로 속도 감점이 걸리는 길이는 이만큼이다.")
    if other_on:
        L.append(f"- `school_zones.csv`에는 '배오개 보호구역' 엣지({', '.join(other_zone)})도 있으나 "
                 "실제 기준값이 주어지지 않아 비교하지 않았다 (기준값 없음).")
    L.append(f"- 관측 [40–41](보호구역 거리·제한속도)은 add.xml의 적색 포장·노란 이중선 폴리곤에서 "
             f"구간을 읽는데, 청계천로 add.xml의 폴리곤 {len(polys)}개는 모두 정류장 그림(id `bs*`)이고 "
             f"그 밖의 폴리곤은 {n_zone_poly}개다. "
             "그래서 청계천로에서는 관측이 항상 '보호구역 없음'(1, 1)이다 "
             "(`data/merge_check.npz` 1622스텝에서 obs[40] < 1인 스텝 0개로 확인).")
    L.append("\n### 정차금지지대\n")
    L.append("| 엣지 | 엣지 내 구간 (m) | 현재 엣지 길이 (m) | SUMO 경로상 위치 (m) | 실제 노선 거리 | 비고 |")
    L.append("|---|---|---|---|---|---|")
    for d in nsz_detail:
        pos = (f"{f1(d['s0'])} ~ {f1(d['s1'])}" if d["s0"] is not None and d["a"] < (d["len"] or 0) else "-")
        L.append(f"| {d['edge']} | {d['a']:.0f} ~ {d['b']:.0f} | {f1(d['len'])} | {pos} | 기준값 없음 | "
                 f"{d['reason']}. {d['note']} |")

    # 상세 5
    L.append("\n## 5. 개수\n")
    L.append("| 항목 | 실제 | 시뮬레이션 | 비고 |")
    L.append("|---|---|---|---|")
    for name, n, note in counts:
        L.append(f"| {name} |  | {n} | {note} |")

    # 측정 못 한 항목
    L.append("\n## 측정하지 못한 항목과 이유\n")
    base_unmeasured = [
        ("정차금지지대 노선상 위치 오차", "no_stop_zones.csv에 엣지 내 위치만 있고 로드뷰 노선 거리 기준값이 없음"),
        ("신호 교차로·정류장·횡단보도 실제 개수", "저장소에 실제 개수 자료 없음 (요청대로 실제값 칸은 비움)"),
        ("배오개 보호구역 위치", "실제 기준값이 주어지지 않음"),
        ("로드뷰 기점과 SUMO 경로 기점의 일치 여부",
         "로드뷰 0 m 지점의 좌표가 저장소에 없어 기점 차이를 보정하지 않음"),
    ]
    if merged:
        base_unmeasured.append((f"교차로로 병합된 {len(merged)}개 구간의 차로 수",
                                "현재 net에서 일반 엣지가 아니라 교차로 내부 차로라 엣지 단위 비교 불가"))
    for name, why in base_unmeasured + unmeasured:
        L.append(f"- **{name}**: {why}")

    md_path = os.path.join(args.out_dir, "geometry_validation.md")
    with open(md_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print(f"저장: {md_path}\n저장: {csv_path}")


if __name__ == "__main__":
    main()
