"""차로 수 로드뷰 조사용 체크리스트 만들기 → data/road_data/cgc_lanes_checklist.csv

배경 (2026-11 도로망 점검): 차가 실제로 다니는 도로 중 동행 본선(버스 학습 노선)은
data/road_data/cgc_lanes.csv에 로드뷰로 확인한 차로 수가 있다. 하지만 서행 본선·회차
연결로·측면 합류로는 OSM 값 또는 기본값(1차로)이라 검증되지 않았다. OSM은 동행 본선
21구간을 전부 '2차로'로 적었는데 로드뷰와 11구간(52%)이 달랐다 → 서행도 의심스럽다.

이 스크립트는 '시뮬레이션이 쓰는데 로드뷰 확인값이 없는' 구간을 모아 조사표를 만든다.

조사표 열
    edge_id, lanes, reason   ← cgc_lanes.csv와 같은 형식. 조사자가 lanes·reason을 채운다
    현재_차로, 출처            ← 지금 시뮬레이션 값과 그 출처 (OSM / 기본값)
    구분, 도로명, 구간(앞→뒤 교차 도로), 길이_m, 진행방향
    중간_위도, 중간_경도, 로드뷰, 구글_스트리트뷰  ← 구간 가운데 지점 링크
    시작/끝 위경도

조사 방법: 로드뷰 링크를 열고 '진행방향' 쪽을 보며, 이 방향으로 차가 다닐 수 있는
차로 수(버스전용·주정차 칸 포함 여부는 reason에 적기)를 lanes에 쓴다.
다 채운 뒤: lanes가 채워진 줄의 앞 세 열을 data/road_data/cgc_lanes.csv 끝에 붙인다.
→ 다음 빌드(train.py / view_road.py)부터 반영된다 (tools/osm_to_sumo.py가 읽음).

사용: python tools/make_lane_checklist.py   (env/sumo 빌드가 있어야 한다)
"""
import csv
import math
import os
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import osm_to_sumo  # noqa: E402  (ORIGIN: 좌표 원점)

os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
import sumolib  # noqa: E402

NET = os.path.join(ROOT, "env", "sumo", "cgc.net.xml")
ROU = os.path.join(ROOT, "env", "sumo", "cgc.rou.xml")
OSM = os.path.join(ROOT, "data", "osm", "cheonggyecheon.osm")
LANES_CSV = os.path.join(ROOT, "data", "road_data", "cgc_lanes.csv")
OUT = os.path.join(ROOT, "data", "road_data", "cgc_lanes_checklist.csv")


def to_lonlat(x, y):
    """osm_to_sumo.project의 역변환 (등장방형 근사, 원점 ORIGIN)."""
    k = math.pi / 180.0
    lon0, lat0 = osm_to_sumo.ORIGIN
    return (x / (k * 6378137.0 * math.cos(lat0 * k)) + lon0, y / (k * 6378137.0) + lat0)


def compass(dx, dy):
    """진행 벡터 → 방위 (북=0°, 시계방향)와 한국어 8방위."""
    deg = (math.degrees(math.atan2(dx, dy)) + 360) % 360
    names = ["북", "북동", "동", "남동", "남", "남서", "서", "북서"]
    return deg, names[int((deg + 22.5) // 45) % 8]


def main():
    net = sumolib.net.readNet(NET)
    rou = ET.parse(ROU).getroot()
    osm = {w.get("id"): {t.get("k"): t.get("v") for t in w.findall("tag")}
           for w in ET.parse(OSM).getroot().iter("way")}
    with open(LANES_CSV, encoding="utf-8-sig") as f:
        verified = {r["edge_id"].strip() for r in csv.DictReader(f)}

    routes = {r.get("id"): r.get("edges").split() for r in rou.iter("route") if r.get("id")}
    # 구분: 서행 본선(main_west 순서), 회차 연결로(왕복 노선에만 있는 짧은 연결), 측면 합류로
    order, kind = [], {}
    for e in routes.get("main_west", []):
        order.append(e)
        kind[e] = "서행 본선"
    for e in routes.get("r0_loop", []):
        if e not in kind and e not in routes.get("r0", []):
            order.append(e)
            kind[e] = "회차 연결로" if "청계천로" not in (net.getEdge(e).getName() or "") \
                else "서행 본선"
    for fl in rou.iter("flow"):
        fid, src = fl.get("id"), fl.get("from")
        if src and (fid.startswith("side_") or fid.startswith("through_merge")) and src not in kind:
            order.append(src)
            kind[src] = "측면 합류로"

    rows = []
    for eid in order:
        if eid in verified or not net.hasEdge(eid):
            continue
        e = net.getEdge(eid)
        lanes = [l for l in e.getLanes() if l.allows("passenger")]
        shape = lanes[-1].getShape() if lanes else e.getShape()
        tags = osm.get(eid[1:].split("_")[0], {})
        oneway = tags.get("oneway") in ("yes", "true", "1", "-1")
        if tags.get("lanes"):
            src = f"OSM (lanes={tags['lanes']}{'' if oneway else ', 양방향 합계'})"
        else:
            src = "기본값 (OSM에 차로 수 없음)"
        # 구간 이름: 앞·뒤 교차로에서 만나는 다른 도로 이름
        def cross_names(node):
            names = {x.getName() for x in list(node.getIncoming()) + list(node.getOutgoing())
                     if x.getName() and x.getName() != e.getName()}
            return "/".join(sorted(names)) or "이름 없는 교차점"
        (x0, y0), (x1, y1) = shape[0], shape[-1]
        mx, my = sumolib.geomhelper.positionAtShapeOffset(shape, e.getLength() / 2)
        deg, d8 = compass(x1 - x0, y1 - y0)
        lon0, lat0 = to_lonlat(x0, y0)
        lon1, lat1 = to_lonlat(x1, y1)
        mlon, mlat = to_lonlat(mx, my)
        rows.append({
            "edge_id": eid, "lanes": "", "reason": "",
            "현재_차로": len(lanes), "출처": src,
            "구분": kind[eid], "도로명": e.getName() or "(이름 없음)",
            "구간": f"{cross_names(e.getFromNode())} → {cross_names(e.getToNode())}",
            "길이_m": round(e.getLength(), 1),
            "진행방향": f"{d8} ({deg:.0f}°)",
            "중간_위도": f"{mlat:.6f}", "중간_경도": f"{mlon:.6f}",
            "로드뷰": f"https://map.kakao.com/link/roadview/{mlat:.6f},{mlon:.6f}",
            "구글_스트리트뷰": ("https://www.google.com/maps/@?api=1&map_action=pano"
                           f"&viewpoint={mlat:.6f},{mlon:.6f}&heading={deg:.0f}"),
            "시작_위경도": f"{lat0:.6f},{lon0:.6f}", "끝_위경도": f"{lat1:.6f},{lon1:.6f}",
        })

    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:   # BOM: 엑셀에서 한글 안 깨짐
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    c = Counter(r["구분"] for r in rows)
    print(f"조사 대상 {len(rows)}구간 ({', '.join(f'{k} {v}' for k, v in c.items())}) → {OUT}")


if __name__ == "__main__":
    main()
