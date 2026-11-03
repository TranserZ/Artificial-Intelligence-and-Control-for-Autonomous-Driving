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


def project(lon, lat):
    k = math.pi / 180.0
    x = (lon - ORIGIN[0]) * k * 6378137.0 * math.cos(ORIGIN[1] * k)
    y = (lat - ORIGIN[1]) * k * 6378137.0
    return x, y


def inside(lon, lat):
    return BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]


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


def convert(osm_path, out_dir):
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

    # 교차점: 둘 이상의 way가 공유하는 노드, 끝점, 신호등 노드
    use = Counter()
    for _, _, refs in ways:
        for r in set(refs):
            use[r] += 1
    signals = {nid for nid, t in node_tags.items()
               if t.get("highway") == "traffic_signals"}

    def is_junction(r, ends):
        return use[r] > 1 or r in ends or r in signals

    edges, used_nodes, seq = [], set(), Counter()
    for wid, tags, refs in ways:
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
            if fwd or bwd:
                n_lane = int(lanes) if lanes else 1
                s = seg if fwd else list(reversed(seg))
                p = pts if fwd else list(reversed(pts))
                seq[wid] += 1
                edges.append((f"{wid}_{seq[wid]}", s[0], s[-1], n_lane, speed, p,
                              tags))
                used_nodes.update([s[0], s[-1]])
            else:
                per = max(1, int(lanes) // 2) if lanes else 1
                seq[wid] += 1
                edges.append((f"{wid}_{seq[wid]}a", seg[0], seg[-1], per, speed, pts, tags))
                edges.append((f"{wid}_{seq[wid]}b", seg[-1], seg[0], per, speed,
                              list(reversed(pts)), tags))
                used_nodes.update([seg[0], seg[-1]])

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

    def useful_signal(r):
        if r in promote:
            return True
        if r in stopline:
            return False
        spd = incident.get(r, [])
        return r in signals and not (len(spd) <= 2 and max(spd, default=0) > 14.0)

    n_drop = sum(1 for r in used_nodes if r in signals and not useful_signal(r))
    with open(os.path.join(out_dir, "cgc.nod.xml"), "w") as f:
        f.write("<nodes>\n")
        for r in sorted(used_nodes):
            x, y = project(*nodes[r])
            kind = "traffic_light" if useful_signal(r) else "priority"
            f.write(f'    <node id="n{r}" x="{x:.2f}" y="{y:.2f}" type="{kind}"/>\n')
        f.write("</nodes>\n")
    with open(os.path.join(out_dir, "cgc.edg.xml"), "w") as f:
        f.write("<edges>\n")
        for eid, a, b, n_lane, speed, pts, tags in edges:
            shape = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts[1:-1])
            attr = f' shape="{shape}"' if shape else ""
            name = (tags.get("name") or "").replace('"', "").replace("&", "")
            name_attr = f' name="{name}"' if name else ""
            f.write(f'    <edge id="e{eid}" from="n{a}" to="n{b}" numLanes="{n_lane}" '
                    f'speed="{speed:.2f}" priority="{ {"primary":4,"secondary":3,"tertiary":2}.get(tags.get("highway"),1) }" '
                    f'{name_attr}{attr}/>\n')
        f.write("</edges>\n")

    n_tl = sum(1 for r in used_nodes if useful_signal(r))
    print(f"노드 {len(used_nodes)}개 (신호등 {n_tl}, 빠른 도로 중간 신호 {n_drop}개 제외), 엣지 {len(edges)}개 → {out_dir}")
    return edges


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm", default="data/osm/cheonggyecheon.osm")
    ap.add_argument("--out", default="env/cheonggyecheon")
    a = ap.parse_args()
    convert(a.osm, a.out)
