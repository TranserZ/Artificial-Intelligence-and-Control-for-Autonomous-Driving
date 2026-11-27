"""OSM(청계천로 일대) → SUMO 노드/엣지 XML 변환기.

이 PC의 SUMO 빌드에는 PROJ가 없어 `netconvert --osm-files`를 쓸 수 없다.
그래서 OSM의 위경도를 직접 미터 좌표로 바꾸고, SUMO plain XML(nod/edg)로
써서 netconvert가 읽게 한다.

데이터 출처: © OpenStreetMap contributors (ODbL), data/osm/cheonggyecheon.osm
좌표계: 기준점(청계광장 서쪽 끝) 기준 동쪽=+x, 북쪽=+y (m). 지역이 2.5 km라
등장방형(equirectangular) 근사로 충분하다 (오차 < 0.1%).

OSM에 없는 값은 아래 DEFAULT_*에서 채우며, 어떤 값이 가정인지 NOTES에 기록한다.
"""
import argparse
import csv
import math
import os
import xml.etree.ElementTree as ET
from collections import Counter

ORIGIN = (126.9772, 37.5693)            # (lon, lat) 청계광장 서쪽 끝 부근
BBOX = (126.9760, 37.5668, 127.0030, 37.5712)  # west, south, east, north
# 다른 지역(예: 연산교차로)은 환경변수로 덮어쓴다: OSM_ORIGIN="lon,lat" OSM_BBOX="w,s,e,n"
if os.environ.get("OSM_ORIGIN"):
    ORIGIN = tuple(float(v) for v in os.environ["OSM_ORIGIN"].split(","))
    BBOX = tuple(float(v) for v in os.environ["OSM_BBOX"].split(","))

# 자동차가 다닐 수 있는 도로 종류 (보행자·자전거 전용 제외)
CAR_HIGHWAYS = {"primary", "secondary", "tertiary", "unclassified", "residential",
                "service", "living_street", "primary_link", "secondary_link",
                "tertiary_link"}
# OSM에 maxspeed가 없을 때 쓰는 값 (km/h). 도시부 일반 도로 기준의 가정.
DEFAULT_KMH = {"primary": 50, "secondary": 50, "tertiary": 40, "unclassified": 30,
               "residential": 30, "service": 20, "living_street": 20}
CHEONGGYECHEON_KMH = 30   # 청계천로: OSM이 일부 구간에 maxspeed=30으로 태그함 → 전 구간에 적용(가정)


CORRIDOR_NAME = "청계천로"

# ── 배경 도로 정리 (2026-11 도로망 점검) ─────────────────────────────────────
# 시뮬레이션이 실제로 쓰는 건 청계천로와 거기 맞닿은 옆길뿐이고, 나머지는 화면용
# 배경이다. OSM을 그대로 옮기면 배경이 지저분해서 아래 규칙으로 정리한다.
# 청계천로와 거기 맞닿은 도로(측면 합류 후보)는 어떤 규칙도 적용하지 않는다.
MINOR_HIGHWAYS = {"residential", "service", "living_street", "unclassified"}
BIG_HIGHWAYS = {"primary", "secondary", "tertiary",
                "primary_link", "secondary_link", "tertiary_link"}
# 주차장 통로·건물 진입로: 섬이나 막다른 토막이 되어 끊긴 것처럼 보인다 → 뺀다
DROP_SERVICE = {"parking_aisle", "driveway", "drive-through"}
# OSM에서 몇 m 차이로 안 이어진 골목 끝 → 이 거리 안의 교차점에 붙인다
SNAP_DIST = 12.0
# 지도 경계에서 이 거리 안의 막다른 끝은 '잘린 도로'라 정리하지 않는다
BORDER_MARGIN = 25.0
# 청계천로 노드에서 이 거리 안에 닿는 도로는 정리하지 않는다 (시뮬레이션 보호)
PROTECT_RADIUS = 40.0
# 청계천로에서 교차점 이 단계 안의 교차점에 닿는 도로도 정리하지 않는다
PROTECT_HOPS = 3
# 차로 수 태그가 없는 큰 도로: 같은 이름의 이웃 구간 값을 이어받고, 없으면 이 값
# (방향당 차로 수). 예전엔 무조건 1이라 세종대로 8차로 중간이 1차로가 됐다.
DEFAULT_LANES_PER_DIR = {"primary": 2, "secondary": 2, "tertiary": 1,
                         "primary_link": 1, "secondary_link": 1, "tertiary_link": 1}
# 골목 차로 폭 (m). OSM에 폭 정보가 전혀 없어 모두 SUMO 기본 3.2 m였다 (실제보다 넓음)
MINOR_LANE_WIDTH = {"residential": 2.75, "unclassified": 2.75,
                    "service": 2.5, "living_street": 2.5}


def project(lon, lat):
    k = math.pi / 180.0
    x = (lon - ORIGIN[0]) * k * 6378137.0 * math.cos(ORIGIN[1] * k)
    y = (lat - ORIGIN[1]) * k * 6378137.0
    return x, y


def inside(lon, lat):
    return BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]


def load_school_speeds():
    """어린이보호구역 edge id → 제한속도 m/s. 파일이 없으면 빈 dict."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "road_data", "school_zones.csv")
    speeds = {}
    if not os.path.exists(path):
        return speeds
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            speeds[row["edge_id"].strip()] = float(row["speed_limit_ms"])
    return speeds


def load_lane_overrides():
    """로드뷰로 확인한 실제 차로 수. edge id → 차로 수. 파일이 없으면 빈 dict.

    OSM의 lanes 태그는 누락·오기가 많아(특히 청계천로 편도 구간), 현장 확인한
    값으로 덮어쓴다. data/road_data/cgc_lanes.csv.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "road_data", "cgc_lanes.csv")
    lanes = {}
    if not os.path.exists(path):
        return lanes
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lanes[row["edge_id"].strip()] = int(row["lanes"])
    return lanes


def kmh_of(tags):
    raw = tags.get("maxspeed")
    if raw:
        try:
            return float(raw.split()[0])
        except ValueError:
            pass
    if "청계천로" in (tags.get("name") or ""):
        return CHEONGGYECHEON_KMH
    return DEFAULT_KMH.get(tags.get("highway"), 30)


def convert(osm_path, out_dir, signals_on=True, school_on=True):
    """OSM → cgc.nod.xml / cgc.edg.xml.

    signals_on=False: 신호 교차로를 모두 신호 없는 교차로(priority)로 만든다 (단계별 맵 1단계).
    school_on=False: 어린이보호구역 엣지에 보호구역 제한속도를 넣지 않는다 (보호구역 단계 전).
    """
    root = ET.parse(osm_path).getroot()
    nodes, node_tags = {}, {}
    for n in root.findall("node"):
        nid = n.get("id")
        nodes[nid] = (float(n.get("lon")), float(n.get("lat")))
        node_tags[nid] = {t.get("k"): t.get("v") for t in n.findall("tag")}

    ways = []
    for w in root.findall("way"):
        tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
        if tags.get("highway") not in CAR_HIGHWAYS:
            continue
        if tags.get("area") == "yes" or tags.get("access") in ("no", "private"):
            continue
        if tags.get("tunnel") == "building_passage":
            continue
        refs = [nd.get("ref") for nd in w.findall("nd") if nd.get("ref") in nodes]
        # bbox 밖 구간은 잘라낸다 (연속된 안쪽 구간만 남김)
        run = []
        for r in refs:
            if inside(*nodes[r]):
                run.append(r)
            else:
                if len(run) >= 2:
                    ways.append((w.get("id"), tags, run))
                run = []
        if len(run) >= 2:
            ways.append((w.get("id"), tags, run))

    signals = {nid for nid, t in node_tags.items()
               if t.get("highway") == "traffic_signals"}
    corridor_nodes = _protected_nodes(ways, nodes, signals)
    n_snap = _snap_near_misses(ways, nodes, signals, corridor_nodes)

    # 교차점: 둘 이상의 way가 공유하는 노드, 끝점, 신호등 노드
    use = Counter()
    for _, _, refs in ways:
        for r in set(refs):
            use[r] += 1
    lanes_fill = _fill_big_road_lanes(ways)

    def is_junction(r, ends):
        return use[r] > 1 or r in ends or r in signals

    edges, used_nodes, seq = [], set(), Counter()
    for wi, (wid, tags, refs) in enumerate(ways):
        ends = {refs[0], refs[-1]}
        # junction 노드에서 way를 자른다
        cut, cur = [], [refs[0]]
        for r in refs[1:]:
            cur.append(r)
            if is_junction(r, ends):
                cut.append(cur)
                cur = [r]
        if len(cur) >= 2:
            cut.append(cur)
        oneway = tags.get("oneway")
        lanes = tags.get("lanes")
        speed = kmh_of(tags) / 3.6
        for seg in cut:
            if seg[0] == seg[-1]:
                continue
            pts = [project(*nodes[r]) for r in seg]
            fwd = oneway in ("yes", "true", "1")
            bwd = oneway == "-1"
            # 차로 수 태그 없는 큰 도로: 이웃 구간에서 이어받은 값 (청계천로와 맞닿은
            # 구간은 그대로 둔다 — 측면 합류 등 시뮬레이션이 쓰는 도로를 바꾸지 않게)
            fill = None
            if not lanes and wi in lanes_fill and not (
                    seg[0] in corridor_nodes or seg[-1] in corridor_nodes):
                fill = lanes_fill[wi]
            if fwd or bwd:
                n_lane = int(lanes) if lanes else (fill or 1)
                s = seg if fwd else list(reversed(seg))
                p = pts if fwd else list(reversed(pts))
                seq[wid] += 1
                edges.append((f"{wid}_{seq[wid]}", s[0], s[-1], n_lane, speed, p,
                              tags))
                used_nodes.update([s[0], s[-1]])
            else:
                per = max(1, int(lanes) // 2) if lanes else (fill or 1)
                seq[wid] += 1
                edges.append((f"{wid}_{seq[wid]}a", seg[0], seg[-1], per, speed, pts, tags))
                edges.append((f"{wid}_{seq[wid]}b", seg[-1], seg[0], per, speed,
                              list(reversed(pts)), tags))
                used_nodes.update([seg[0], seg[-1]])

    # 배경 정리: 주차장 통로·진입로, 막다른 골목 토막을 뺀다. 구간 자르기(엣지 id)는
    # 위에서 원래 데이터 기준으로 이미 끝났으므로 남은 엣지의 id는 그대로다.
    n_before = len(edges)
    edges = _prune_background(edges, nodes, corridor_nodes)
    used_nodes = {r for _, a, b, _, _, _, _ in edges for r in (a, b)}
    n_pruned = n_before - len(edges)

    os.makedirs(out_dir, exist_ok=True)
    # 중간 지점(연결 도로 2개 이하)에 있는 신호가 14 m/s 넘는 빠른 도로 위라면
    # 보도·횡단보도가 생성되지 않아 "항상 초록" 신호가 되므로 일반 노드로 둔다.
    incident = {}
    for _, a, b, _, speed, _, _ in edges:
        for r in (a, b):
            incident.setdefault(r, []).append(speed)

    # OSM은 신호를 교차로 중심이 아니라 진입로 정지선 노드에 찍는 경우가 많다.
    # OSM_STOPLINE_PROMOTE=1이면 정지선 신호(연결 도로 2개 이하)에서 60 m 안의 첫 교차점을
    # 신호교차로로 승격하고, 정지선 신호 자체는 일반 노드로 돌린다.
    promote, stopline = set(), set()
    if os.environ.get("OSM_STOPLINE_PROMOTE"):
        adj = {}
        for _, a, b, _, _, _, _ in edges:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        for r in used_nodes:
            if r in signals and len(adj.get(r, ())) <= 2:
                seen, frontier = {r}, [(r, 0.0)]
                while frontier:
                    cur, d = frontier.pop(0)
                    if cur != r and len(adj.get(cur, ())) >= 3:
                        promote.add(cur)
                        stopline.add(r)
                        break
                    for nb in adj.get(cur, ()):
                        if nb in seen:
                            continue
                        seen.add(nb)
                        x0, y0 = project(*nodes[cur]); x1, y1 = project(*nodes[nb])
                        nd = d + math.hypot(x1 - x0, y1 - y0)
                        if nd <= 60.0:
                            frontier.append((nb, nd))

    # OSM에 신호 태그가 빠진 곳 보강: 청계천로가 간선(primary/secondary/tertiary)과
    # 만나는 교차점은 실제로 신호교차로이므로 신호로 본다 (OSM_INFER_SIGNALS=0이면 끔).
    inferred = set()
    if os.environ.get("OSM_INFER_SIGNALS", "1") != "0":
        main_cls = {"primary", "secondary", "tertiary"}
        on_corridor, crossing = set(), set()
        for _, tags, refs in ways:
            name = tags.get("name") or ""
            if CORRIDOR_NAME in name:
                on_corridor.update(refs)
            elif tags.get("highway", "").replace("_link", "") in main_cls:
                crossing.update(refs)
        inferred = {r for r in used_nodes
                    if r in on_corridor and r in crossing and r not in stopline}

    def useful_signal(r):
        if r in inferred:
            return True
        if r in promote:
            return True
        if r in stopline:
            return False
        spd = incident.get(r, [])
        return r in signals and not (len(spd) <= 2 and max(spd, default=0) > 14.0)

    n_drop = sum(1 for r in used_nodes if r in signals and not useful_signal(r))
    with open(os.path.join(out_dir, "cgc.nod.xml"), "w", encoding="utf-8", newline="\n") as f:
        f.write("<nodes>\n")
        for r in sorted(used_nodes):
            x, y = project(*nodes[r])
            kind = "traffic_light" if signals_on and useful_signal(r) else "priority"
            f.write(f'    <node id="n{r}" x="{x:.2f}" y="{y:.2f}" type="{kind}"/>\n')
        f.write("</nodes>\n")
    school_speeds = load_school_speeds() if school_on else {}
    lane_overrides = load_lane_overrides()
    n_school = 0
    n_lanes_fixed = 0
    with open(os.path.join(out_dir, "cgc.edg.xml"), "w", encoding="utf-8", newline="\n") as f:
        f.write("<edges>\n")
        for eid, a, b, n_lane, speed, pts, tags in edges:
            # 모양은 양 끝(노드 위치)까지 넣는다. 중간 점만 주면 netconvert가 그 점들만 도로로 써서
            # 엣지가 마지막 중간 점에서 끝나고 노드까지 남은 거리가 전부 교차로 내부가 된다
            # (2026-11-21 발견: 경로 2.1 km 중 761 m가 교차로 내부, 보호구역 70 m → 250 m).
            # OSM_SHAPE_ENDS=0이면 예전 방식(중간 점만).
            sp = pts if os.environ.get("OSM_SHAPE_ENDS", "1") != "0" else pts[1:-1]
            shape = " ".join(f"{x:.2f},{y:.2f}" for x, y in sp)
            attr = f' shape="{shape}"' if shape else ""
            name = (tags.get("name") or "").replace('"', "").replace("&", "")
            name_attr = f' name="{name}"' if name else ""
            full_id = f"e{eid}"
            if full_id in school_speeds:
                speed = school_speeds[full_id]
                n_school += 1
            if full_id in lane_overrides:
                n_lane = lane_overrides[full_id]
                n_lanes_fixed += 1
            # 골목 차로 폭 (청계천로·맞닿은 도로 제외 — 시뮬레이션이 쓰는 도로는 그대로)
            width_attr = ""
            lw = MINOR_LANE_WIDTH.get(tags.get("highway"))
            if lw and a not in corridor_nodes and b not in corridor_nodes \
                    and CORRIDOR_NAME not in (tags.get("name") or ""):
                width_attr = f' width="{lw}"'
            f.write(f'    <edge id="{full_id}" from="n{a}" to="n{b}" numLanes="{n_lane}" '
                    f'speed="{speed:.2f}" priority="{ {"primary":4,"secondary":3,"tertiary":2}.get(tags.get("highway"),1) }" '
                    f'{name_attr}{width_attr}{attr}/>\n')
        f.write("</edges>\n")

    n_tl = sum(1 for r in used_nodes if signals_on and useful_signal(r))
    print(f"노드 {len(used_nodes)}개 (신호등 {n_tl}, 추정 신호 {len(inferred)}, 빠른 도로 중간 신호 {n_drop}개 제외), "
          f"엣지 {len(edges)}개, 어린이보호구역 속도 {n_school}개, 차로 수 보정 {n_lanes_fixed}개 → {out_dir}")
    print(f"배경 정리: 끊긴 끝 잇기 {n_snap}곳, 막다른 토막·주차장 통로 {n_pruned}개 엣지 제거, "
          f"차로 수 채운 큰 도로 {len(lanes_fill)}개 way")
    return edges


def _protected_nodes(ways, nodes, signals):
    """정리 규칙을 적용하지 않을 노드 = 시뮬레이션이 쓰는 도로 주변.

    (1) 청계천로 노드에서 PROTECT_RADIUS 안의 노드 — netconvert가 근처 교차점을
        하나로 합치므로, 합쳐진 교차로의 다른 노드에 닿은 도로를 바꿔도 교차로
        모양이 바뀌어 본선 구간 길이·연결이 달라진다.
    (2) 청계천로에서 교차점 PROTECT_HOPS 단계 안의 교차점 — 측면 합류 도로(본선에서
        한 단계)의 먼 쪽 교차로와 그 이웃 교차로 모양까지 그대로 두기 위해.
        (실측: 이웃 교차로가 바뀌면 그 사이 짧은 연결의 교차로 합치기가 달라져
         측면 합류 도로 길이가 바뀌었다)"""
    corridor = {r for _, tags, refs in ways
                if CORRIDOR_NAME in (tags.get("name") or "") for r in refs}
    use = Counter()
    for _, _, refs in ways:
        for r in set(refs):
            use[r] += 1
    # 교차점끼리의 이웃 관계 (way를 따라 다음 교차점)
    adj = {}
    for _, _, refs in ways:
        js = [r for i, r in enumerate(refs)
              if use[r] > 1 or r in signals or i in (0, len(refs) - 1)]
        for p, q in zip(js, js[1:]):
            adj.setdefault(p, set()).add(q)
            adj.setdefault(q, set()).add(p)
    prot = set(corridor)
    frontier = set(corridor)
    for _ in range(PROTECT_HOPS):
        frontier = {q for p in frontier for q in adj.get(p, ())} - prot
        prot |= frontier
    cxy = [project(*nodes[r]) for r in corridor]
    for _, _, refs in ways:
        for r in refs:
            if r in prot:
                continue
            x, y = project(*nodes[r])
            if any(abs(x - cx) <= PROTECT_RADIUS and abs(y - cy) <= PROTECT_RADIUS
                   and math.hypot(x - cx, y - cy) <= PROTECT_RADIUS for cx, cy in cxy):
                prot.add(r)
    return prot


def _inside_margin(lon, lat):
    """지도 경계에서 BORDER_MARGIN 이상 안쪽인가 (경계에서 잘린 도로 끝은 정리 대상 아님)."""
    x0, y0 = project(BBOX[0], BBOX[1])
    x1, y1 = project(BBOX[2], BBOX[3])
    x, y = project(lon, lat)
    m = BORDER_MARGIN
    return x0 + m < x < x1 - m and y0 + m < y < y1 - m


def _snap_near_misses(ways, nodes, signals, corridor_nodes):
    """OSM에서 몇 m 차이로 안 이어진 골목 끝을 가까운 교차점에 붙인다 (ways를 고친다).

    대상: 골목(MINOR_HIGHWAYS) way의 끝 노드 중 다른 way와 공유하지 않는(막다른) 것.
    붙일 곳: SNAP_DIST 안의 '이미 교차점이거나 다른 way의 끝인' 노드 — 이미 잘리는
    지점이라 다른 도로의 구간 나누기(엣지 id)가 바뀌지 않는다. 청계천로 노드는 제외.
    반환: 붙인 끝의 수."""
    use = Counter()
    for _, _, refs in ways:
        for r in set(refs):
            use[r] += 1
    anchors = {r for r, n in use.items() if n > 1} | signals
    for _, _, refs in ways:
        anchors.update((refs[0], refs[-1]))
    anchors -= corridor_nodes
    anchor_xy = [(r, project(*nodes[r])) for r in anchors if r in nodes]
    n = 0
    for i, (wid, tags, refs) in enumerate(ways):
        if tags.get("highway") not in MINOR_HIGHWAYS:
            continue
        for k in (0, -1):
            end = refs[k]
            if use[end] > 1 or end in signals or not _inside_margin(*nodes[end]):
                continue
            ex, ey = project(*nodes[end])
            best, bd = None, SNAP_DIST
            own = set(refs)
            for r, (x, y) in anchor_xy:
                if r in own:
                    continue
                d = math.hypot(x - ex, y - ey)
                if d < bd:
                    best, bd = r, d
            if best is not None:
                refs[k] = best
                use[best] += 1
                n += 1
    return n


def _fill_big_road_lanes(ways):
    """차로 수(lanes) 태그가 없는 큰 도로 way → 방향당 차로 수 {way index: n}.
    같은 이름의 이웃 way(끝점 공유)에서 값을 이어받고(여러 번 반복해 퍼뜨림),
    끝까지 못 받으면 DEFAULT_LANES_PER_DIR. 청계천로는 cgc_lanes.csv가 맡으므로 제외."""
    def per_dir(tags):
        n = int(tags["lanes"])
        return n if tags.get("oneway") in ("yes", "true", "1", "-1") else max(1, n // 2)

    by_node = {}
    for i, (_, tags, refs) in enumerate(ways):
        for r in (refs[0], refs[-1]):
            by_node.setdefault(r, []).append(i)
    known = {}
    for i, (_, tags, _) in enumerate(ways):
        if tags.get("highway") in BIG_HIGHWAYS and tags.get("lanes"):
            try:
                known[i] = per_dir(tags)
            except ValueError:
                pass
    todo = [i for i, (_, tags, _) in enumerate(ways)
            if tags.get("highway") in BIG_HIGHWAYS and not tags.get("lanes")
            and CORRIDOR_NAME not in (tags.get("name") or "")]
    fill = {}
    for _ in range(20):                      # 이웃에서 이웃으로 퍼뜨린다
        changed = False
        for i in todo:
            if i in fill:
                continue
            _, tags, refs = ways[i]
            name = tags.get("name")
            if not name:
                continue
            vals = [known.get(j, fill.get(j)) for r in (refs[0], refs[-1])
                    for j in by_node.get(r, []) if j != i and ways[j][1].get("name") == name]
            vals = [v for v in vals if v]
            if vals:
                fill[i] = max(vals)
                changed = True
        if not changed:
            break
    for i in todo:
        fill.setdefault(i, DEFAULT_LANES_PER_DIR.get(ways[i][1].get("highway"), 1))
    return fill


def _prune_background(edges, nodes, corridor_nodes):
    """배경 정리: (1) 주차장 통로·건물 진입로 (2) 막다른 골목 토막을 반복해서 뺀다.

    청계천로 노드에 닿는 구간은 남긴다 (측면 합류 후보가 바뀌지 않게).
    양방향 도로의 a/b 엣지는 한 구간(같은 id에서 a/b를 뗀 것)으로 같이 다룬다.
    지도 경계 근처의 끝은 '잘린 도로'이므로 막다른 길로 보지 않는다."""
    def key(eid):
        return eid[:-1] if eid[-1] in "ab" else eid

    def protected(a, b, tags):
        return (a in corridor_nodes or b in corridor_nodes
                or CORRIDOR_NAME in (tags.get("name") or ""))

    segs = {}
    for e in edges:
        segs.setdefault(key(e[0]), []).append(e)
    drop = {k for k, es in segs.items()
            if es[0][6].get("service") in DROP_SERVICE
            and not protected(es[0][1], es[0][2], es[0][6])}
    while True:
        deg = Counter()
        for k, es in segs.items():
            if k in drop:
                continue
            for r in {es[0][1], es[0][2]}:
                deg[r] += 1
        new = set()
        for k, es in segs.items():
            if k in drop:
                continue
            a, b, tags = es[0][1], es[0][2], es[0][6]
            if tags.get("highway") not in MINOR_HIGHWAYS or protected(a, b, tags):
                continue
            for r in (a, b):
                if deg[r] == 1 and _inside_margin(*nodes[r]):
                    new.add(k)
                    break
        if not new:
            break
        drop |= new
    return [e for e in edges if key(e[0]) not in drop]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm", default="data/osm/cheonggyecheon.osm")
    ap.add_argument("--out", default="environment/cheonggyecheon")
    a = ap.parse_args()
    convert(a.osm, a.out)


# 청계5가 회차의 짧은 연결. 명시적 <join>에서는 빼 두지만,
# netconvert --junctions.join 이 주변 교차로를 합치며 id가 사라질 수 있다.
# 그때는 make_cgc_scenario.resolve_loop_edges 가 그 id를 빼고 이웃 엣지로 잇는다.
KEEP_SHORT_EDGES = {"e482167259_1a", "e608946832_1a"}


def _cluster_members(nid):
    """교차로 id → 원래 OSM 노드 id 목록. netconvert가 합친 교차로는
    'cluster_nA_nB' 꼴이라 멤버를 꺼낼 수 있다. '#N more'로 잘린 이름은 복원
    불가능하므로 None (join 대상에서 제외)."""
    if not nid.startswith("cluster_"):
        return [nid]
    if "#" in nid:
        return None
    parts = nid[len("cluster_"):].split("_")
    return parts if parts and all(p.startswith("n") for p in parts) else None


def tiny_edge_joins(net_path, nod_path, min_len=2.0, min_len_main=8.0):
    """netconvert 결과에서 교차로 사이에 남은 '토막 엣지'를 찾아, 양 끝 교차로를
    하나로 합치라는 <join>을 nod 파일에 쓴다. 반환값: nod 파일이 바뀌었는가.

    두 가지를 토막으로 본다.
    - min_len(2 m) 미만: 어떤 엣지든. (예: 청계3가 부근 0.2 m — 오른쪽 차로 끊김)
    - min_len_main(8 m) 미만이면서 차로 3개 이상(본선 방향): 차 한 대(5 m + 간격)가
      들어가지 못하는 길이. 신호 대기 선두차가 꼬리를 앞 교차로 곡선 위에 걸친 채
      대각선으로 서서 두 차로를 다 막는 현상의 원인. (2차로짜리 중앙 횡단 연결로는
      일부러 제외 — 그 교차로들을 합치면 동·서 방향 도로가 한 덩어리가 된다.)

    청계5가 회차 엣지(KEEP_SHORT_EDGES)와 막다른 끝(dead_end)은 제외한다.
    이미 nod에 있는 <join>과 노드가 겹치면 한 그룹으로 합쳐 다시 쓴다
    (netconvert는 겹치는 join 그룹을 받지 않는다)."""
    net = ET.parse(net_path).getroot()
    jtype = {j.get("id"): j.get("type") for j in net.findall("junction")}
    groups = []
    for e in net.findall("edge"):
        if e.get("function") or e.get("id") in KEEP_SHORT_EDGES:
            continue
        lanes = e.findall("lane")
        if not lanes:
            continue
        length = max(float(l.get("length")) for l in lanes)
        if not (length < min_len or (length < min_len_main and len(lanes) >= 3)):
            continue
        a, b = e.get("from"), e.get("to")
        if not a or not b or "dead_end" in (jtype.get(a), jtype.get(b)):
            continue
        ma, mb = _cluster_members(a), _cluster_members(b)
        if ma is None or mb is None:
            continue
        groups.append(set(ma) | set(mb))
    # 기존 <join> 그룹을 읽어 함께 병합한다
    text = open(nod_path, encoding="utf-8").read()
    old_lines = [ln for ln in text.splitlines() if "<join nodes=" in ln]
    old_groups = [set(ln.split('nodes="')[1].split('"')[0].split()) for ln in old_lines]
    merged = []
    for g in groups + old_groups:
        g = set(g)
        while True:          # 겹치는 그룹을 모두 흡수할 때까지
            hit = next((m for m in merged if m & g), None)
            if hit is None:
                break
            merged.remove(hit)
            g |= hit
        merged.append(g)
    new_set = {frozenset(g) for g in merged}
    if new_set == {frozenset(g) for g in old_groups}:
        return 0             # 바뀐 것이 없다 (재실행 불필요)
    body = "".join(f'    <join nodes="{" ".join(sorted(g))}"/>\n'
                   for g in sorted(merged, key=lambda s: sorted(s)[0]))
    text = "\n".join(ln for ln in text.splitlines() if "<join nodes=" not in ln)
    text = text.replace("</nodes>", body + "</nodes>")
    open(nod_path, "w", encoding="utf-8", newline="\n").write(text)
    return len(merged)


def unsignalized_crossing_discards(net_path, con_path):
    """신호 없는 교차로의 횡단보도를 지우는 연결 파일(con.xml)을 쓴다. 반환: 지울 개수.

    netconvert --crossings.guess는 신호 없는 교차로에도 횡단보도를 만든다. 단계별 맵에서는
    신호 없는 횡단보도를 모두 없애기로 했다 (보행자는 신호 횡단보도로만 건넌다).
    netconvert --sumo-net-file net --connection-files con 으로 적용한다.
    """
    root = ET.parse(net_path).getroot()
    jtype = {j.get("id"): j.get("type") for j in root.iter("junction")}
    lines = []
    for e in root.iter("edge"):
        if e.get("function") != "crossing":
            continue
        node = e.get("id")[1:].rsplit("_c", 1)[0]
        if jtype.get(node) != "traffic_light":
            lines.append(f'    <crossing node="{node}" edges="{e.get("crossingEdges")}" discard="true"/>')
    with open(con_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("<connections>\n" + "\n".join(lines) + "\n</connections>\n")
    return len(lines)


def _passenger_lane_indexes(edge_el):
    """보도를 뺀 승용차 차로 번호. 선행 확인에서 보도는 0, 오른쪽 승용차는 1이었다."""
    out = []
    for lane in edge_el.findall("lane"):
        disallow = lane.get("disallow") or ""
        allow = lane.get("allow")
        if "passenger" in disallow.split():
            continue
        if allow in (None, "") or "passenger" in allow.split():
            out.append(int(lane.get("index")))
    return out


def _write_connections(con_path, lines, append):
    if append and os.path.exists(con_path):
        text = open(con_path, encoding="utf-8").read().replace("</connections>", "").rstrip() + "\n"
    else:
        text = "<connections>\n"
    fresh = []
    for line in lines:
        if line.strip() and line.strip() in text:
            continue
        fresh.append(line)
    body = "\n".join(fresh)
    if body:
        text += body + "\n"
    text += "</connections>\n"
    with open(con_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return len(fresh)


def route_straight_connections(net_path, rou_path, route_id, con_path, append=False):
    """버스 경로(route_id) 위에서 직진으로 이어지지 않는 차로에 직진 연결을 추가하는
    connection 파일을 만든다. OSM에 차로별 진행 방향(turn:lanes)이 없어 netconvert가
    옆길이 있는 교차로에서 오른쪽 차로를 우회전 전용으로 추측한 곳을 보정한다.
    (가정: 청계천로 동행 2차로는 교차로를 직진으로 통과) 추가한 연결 수를 반환."""
    import re
    net = ET.parse(net_path).getroot()
    rou = open(rou_path, encoding="utf-8").read()
    m = re.search(rf'<route id="{route_id}" edges="([^"]+)"', rou)
    if not m:
        return 0
    route = m.group(1).split()
    lanes, conns = {}, set()
    for e in net.findall("edge"):
        if e.get("function"):
            continue
        lanes[e.get("id")] = _passenger_lane_indexes(e)
    for c in net.findall("connection"):
        conns.add((c.get("from"), c.get("to"), int(c.get("fromLane"))))
    pairs_all = {}
    for c in net.findall("connection"):
        if not c.get("from", "").startswith(":"):
            pairs_all.setdefault((c.get("from"), c.get("to")), []).append(
                (int(c.get("fromLane")), int(c.get("toLane"))))
    out = []
    for a, b in zip(route, route[1:]):
        la, lb = lanes.get(a, []), lanes.get(b, [])
        if not lb or all((a, b, fl) in conns for fl in la):
            continue
        # 차로마다 같은 순번의 차로로 곧게 잇는다 (오른쪽→오른쪽, 왼쪽→왼쪽)
        want = {fl: lb[min(i, len(lb) - 1)] for i, fl in enumerate(la)}
        for fl, tl in want.items():
            if (a, b, fl) not in conns:
                out.append(f'    <connection from="{a}" to="{b}" fromLane="{fl}" toLane="{tl}"/>')
        # 다른 차로에서 같은 목표 차로로 비스듬히 들어가는 연결은 지운다
        # (교차로 안에서 두 흐름이 한 차로로 합쳐지며 충돌하던 문제)
        for fl, tl in pairs_all.get((a, b), []):
            if want.get(fl) != tl and tl in want.values():
                out.append(f'    <delete from="{a}" to="{b}" fromLane="{fl}" toLane="{tl}"/>')
    _write_connections(con_path, out, append)
    return len(out)


# 오른쪽 승용차 차로(선행 확인: lane 1)에 좌회전을 명시할 회차 쌍.
# 짧은 엣지가 남아 있으면 그 쌍을, 교차로에 흡수되면 살아 있는 이웃 쌍을 쓴다.
# 청계광장 회차도 같은 방식으로 lane 1이 끊겨 있어 함께 잇는다.
LOOP_LEFT_PAIRS = [
    ("e1455526426_1", "e482167259_1a"),
    ("e1455526426_1", "e608946831_1a"),
    ("e608946832_1a", "e1212424785_1"),
    ("e608946831_1a", "e1212424785_1"),
    ("e350127794_6", "e183272596_1"),
    ("e183272596_1", "e73541403_1"),
]


def loop_left_connections(net_path, con_path, append=True):
    """r0_loop 회차의 좌회전을 오른쪽 승용차 차로에 붙인다.

    선행 확인: 청계5가 첫 좌회전(e1455526426_1)은 lane 2→다음 lane 1만 있고,
    둘째 좌회전은 서행 lane 2(왼쪽)로만 들어간다. 버스는 lane 1에서 출발하므로
    그 차로의 좌회전이 없으면 wrong_lane으로 멈춘다.
    이미 그 fromLane에서 다른 toLane으로 이어져 있어도, 오른쪽 승용차 차로(가장 작은
    승용차 lane)로의 연결이 없으면 하나 더 넣는다. 추가한 연결 수를 반환한다.
    """
    net = ET.parse(net_path).getroot()
    edges = {}
    for e in net.findall("edge"):
        if not e.get("function"):
            edges[e.get("id")] = e
    linked = set()
    full = set()
    for c in net.findall("connection"):
        linked.add((c.get("from"), c.get("to")))
        full.add((c.get("from"), c.get("to"), int(c.get("fromLane")), int(c.get("toLane"))))
    # 직진 보정이 같은 파일을 먼저 썼으면 그 연결은 다시 넣지 않는다
    if append and os.path.exists(con_path):
        import re
        for m in re.finditer(
                r'from="([^"]+)" to="([^"]+)" fromLane="(\d+)" toLane="(\d+)"',
                open(con_path, encoding="utf-8").read()):
            full.add((m.group(1), m.group(2), int(m.group(3)), int(m.group(4))))
    out = []
    for a, b in LOOP_LEFT_PAIRS:
        if (a, b) not in linked or a not in edges or b not in edges:
            continue
        la, lb = _passenger_lane_indexes(edges[a]), _passenger_lane_indexes(edges[b])
        if not la or not lb:
            continue
        fl, tl = la[0], lb[0]
        if (a, b, fl, tl) in full:
            continue
        out.append(f'    <connection from="{a}" to="{b}" fromLane="{fl}" toLane="{tl}"/>')
        full.add((a, b, fl, tl))
    _write_connections(con_path, out, append)
    return len(out)
