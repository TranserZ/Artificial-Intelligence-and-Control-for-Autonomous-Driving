"""국가 표준노드링크(MOCT_LINK)의 차로 수(LANES)를 청계천 도로 구간에 짝지어 가져온다.

데이터: data/road_data/MOCT_LINK.shp/.dbf (국가교통정보센터 표준노드링크, 사용자 제공)
  - 좌표계: ITRF2000 중부원점 TM (원점 127°E 38°N, 가산 200000/600000, 축척 1.0)
  - LANES = 그 링크 진행 방향의 차로 수 (양방향 도로는 방향별 링크가 따로 있다)

방법: 우리 엣지(SUMO 차로 중심선)를 위경도 → TM으로 바꿔, 진행 방향이 같고(30° 이내)
      가장 가까운 링크를 찾는다. 엣지 위 여러 점의 평균 거리로 고르고, 거리가 멀면
      (MAX_DIST) 짝 없음으로 둔다.

사용:
    python tools/nodelink_lanes.py            # 시뮬레이션이 쓰는 엣지 비교표 출력
    python tools/nodelink_lanes.py --csv OUT  # 엣지별 결과를 CSV로 저장
라이브러리: numpy, sumolib만 (pyproj·pyshp 없이 SHP/DBF를 직접 읽는다).
"""
import argparse
import csv
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import osm_to_sumo  # noqa: E402  (ORIGIN, BBOX)

os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
import sumolib  # noqa: E402

DATA = os.path.join(ROOT, "data", "road_data")
MAX_DIST = 15.0       # m — 이보다 멀면 같은 도로로 보지 않는다
MAX_ANGLE = 30.0      # ° — 진행 방향 차이 허용

# ── 좌표 변환 ───────────────────────────────────────────────────────────────
A, F = 6378137.0, 1 / 298.257222101           # GRS80
E2 = F * (2 - F)
LAT0, LON0, K0, FE, FN = 38.0, 127.0, 1.0, 200000.0, 600000.0


def _meridian(phi):
    e4, e6 = E2 * E2, E2 ** 3
    return A * ((1 - E2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
                - (3 * E2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * math.sin(2 * phi)
                + (15 * e4 / 256 + 45 * e6 / 1024) * math.sin(4 * phi)
                - (35 * e6 / 3072) * math.sin(6 * phi))


M0 = _meridian(math.radians(LAT0))


def lonlat_to_tm(lon, lat):
    """위경도 → 중부원점 TM (Snyder 식, 서울 범위에서 cm 수준)."""
    phi, lam = math.radians(lat), math.radians(lon - LON0)
    ep2 = E2 / (1 - E2)
    n = A / math.sqrt(1 - E2 * math.sin(phi) ** 2)
    t, c = math.tan(phi) ** 2, ep2 * math.cos(phi) ** 2
    a = lam * math.cos(phi)
    m = _meridian(phi)
    x = FE + K0 * n * (a + (1 - t + c) * a ** 3 / 6 + (5 - 18 * t + t * t + 72 * c - 58 * ep2) * a ** 5 / 120)
    y = FN + K0 * (m - M0 + n * math.tan(phi) * (a * a / 2 + (5 - t + 9 * c + 4 * c * c) * a ** 4 / 24
                                                 + (61 - 58 * t + t * t + 600 * c - 330 * ep2) * a ** 6 / 720))
    return x, y


def local_to_lonlat(x, y):
    """SUMO 좌표 → 위경도 (osm_to_sumo.project의 역변환)."""
    k = math.pi / 180.0
    lon0, lat0 = osm_to_sumo.ORIGIN
    return x / (k * 6378137.0 * math.cos(lat0 * k)) + lon0, y / (k * 6378137.0) + lat0


# ── SHP/DBF 직접 읽기 (지역 범위 안의 링크만) ─────────────────────────────────
def read_links(bbox_tm, margin=150.0):
    x0, y0, x1, y1 = bbox_tm[0] - margin, bbox_tm[1] - margin, bbox_tm[2] + margin, bbox_tm[3] + margin
    shx = open(os.path.join(DATA, "MOCT_LINK.shx"), "rb").read()
    n = (len(shx) - 100) // 8
    offs = np.frombuffer(shx[100:], dtype=">i4").reshape(n, 2)[:, 0].astype(np.int64) * 2
    links = {}
    with open(os.path.join(DATA, "MOCT_LINK.shp"), "rb") as f:
        for i, off in enumerate(offs):
            f.seek(off + 8)
            head = f.read(44)
            stype = struct.unpack("<i", head[:4])[0]
            if stype != 3:
                continue
            bx0, by0, bx1, by1 = struct.unpack("<4d", head[4:36])
            if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
                continue
            nparts, npts = struct.unpack("<2i", head[36:44])
            f.read(4 * nparts)
            pts = np.frombuffer(f.read(16 * npts), dtype="<f8").reshape(npts, 2)
            links[i] = pts
    # DBF: 해당 레코드만
    with open(os.path.join(DATA, "MOCT_LINK.dbf"), "rb") as f:
        head = f.read(32)
        nrec, hlen, rlen = struct.unpack("<IHH", head[4:12])
        f.seek(32)
        raw = f.read(hlen - 32)
        fields, pos = [], 1
        for j in range(0, len(raw), 32):
            if raw[j] == 0x0D:
                break
            name = raw[j:j + 11].split(b"\0")[0].decode("cp949", "replace")
            ln = raw[j + 16]
            fields.append((name, pos, ln))
            pos += ln
        attrs = {}
        for i in links:
            f.seek(hlen + i * rlen)
            rec = f.read(rlen)
            attrs[i] = {name: rec[p:p + ln].decode("cp949", "replace").strip() for name, p, ln in fields}
    return links, attrs


def _seg_dist_dir(p, pts):
    """점 p에서 폴리라인 pts까지 최단거리와 그 구간의 방향(라디안)."""
    a, b = pts[:-1], pts[1:]
    ab = b - a
    L2 = (ab ** 2).sum(1)
    L2[L2 == 0] = 1e-9
    t = np.clip(((p - a) * ab).sum(1) / L2, 0, 1)
    proj = a + ab * t[:, None]
    d = np.hypot(*(proj - p).T)
    k = int(np.argmin(d))
    return d[k], math.atan2(ab[k][1], ab[k][0])


def match_edges(net, edge_ids, links, attrs):
    """엣지 id → 짝지은 링크 정보 (또는 None)."""
    out = {}
    for eid in edge_ids:
        e = net.getEdge(eid)
        lanes = [l for l in e.getLanes() if l.allows("passenger")]
        if not lanes:
            continue
        # 차로들의 가운데 선 (오른쪽·왼쪽 차로 중심의 평균) 위 점들
        shp = np.array(lanes[len(lanes) // 2].getShape())
        if len(lanes) % 2 == 0 and len(lanes) >= 2:
            shp = (np.array(lanes[len(lanes) // 2 - 1].getShape()) + shp) / 2 \
                if len(lanes[len(lanes) // 2 - 1].getShape()) == len(shp) else shp
        L = e.getLength()
        samples = []
        for frac in (0.2, 0.5, 0.8) if L > 6 else (0.5,):
            x, y = sumolib.geomhelper.positionAtShapeOffset(shp.tolist(), L * frac)
            samples.append(np.array(lonlat_to_tm(*local_to_lonlat(x, y))))
        # 엣지 진행 방향 (TM 좌표)
        (sx, sy), (ex, ey) = shp[0], shp[-1]
        a0 = np.array(lonlat_to_tm(*local_to_lonlat(sx, sy)))
        a1 = np.array(lonlat_to_tm(*local_to_lonlat(ex, ey)))
        heading = math.atan2(*(a1 - a0)[::-1])
        best = None
        for i, pts in links.items():
            if np.min(np.hypot(*(pts - samples[len(samples) // 2]).T)) > 400:
                continue
            ds, ok = [], True
            for p in samples:
                d, ang = _seg_dist_dir(p, pts)
                diff = abs((math.degrees(ang - heading) + 180) % 360 - 180)
                if diff > MAX_ANGLE:
                    ok = False
                    break
                ds.append(d)
            if not ok:
                continue
            md = float(np.mean(ds))
            if md <= MAX_DIST and (best is None or md < best[0]):
                best = (md, i)
        if best:
            at = attrs[best[1]]
            out[eid] = {"dist": round(best[0], 1), "link_id": at.get("LINK_ID"),
                        "lanes": int(float(at.get("LANES") or 0)), "road_name": at.get("ROAD_NAME"),
                        "rank": at.get("ROAD_RANK"), "update": at.get("UPDATEDATE")}
        else:
            out[eid] = None
    return out


def used_edges(rou_path):
    rou = ET.parse(rou_path).getroot()
    routes = {r.get("id"): r.get("edges").split() for r in rou.iter("route") if r.get("id")}
    kind = {}
    for e in routes.get("r0", []):
        kind[e] = "동행 본선"
    for e in routes.get("main_west", []) + routes.get("r0_loop", []):
        kind.setdefault(e, "서행 본선")
    for fl in rou.iter("flow"):
        if fl.get("from") and fl.get("id").startswith(("side_", "through_merge")):
            kind.setdefault(fl.get("from"), "측면 합류로")
    return kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=os.path.join(ROOT, "env", "sumo", "cgc.net.xml"))
    ap.add_argument("--rou", default=os.path.join(ROOT, "env", "sumo", "cgc.rou.xml"))
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    net = sumolib.net.readNet(args.net)
    b = osm_to_sumo.BBOX
    c0, c1 = lonlat_to_tm(b[0], b[1]), lonlat_to_tm(b[2], b[3])
    links, attrs = read_links((c0[0], c0[1], c1[0], c1[1]))
    kind = used_edges(args.rou)
    m = match_edges(net, list(kind), links, attrs)

    with open(os.path.join(DATA, "cgc_lanes.csv"), encoding="utf-8-sig") as f:
        roadview = {r["edge_id"].strip(): int(r["lanes"]) for r in csv.DictReader(f)}
    rows = []
    for eid, k in kind.items():
        e = net.getEdge(eid)
        sim = sum(l.allows("passenger") for l in e.getLanes())
        r = m.get(eid)
        rows.append({"구분": k, "edge_id": eid, "도로명": e.getName() or "", "길이_m": round(e.getLength(), 1),
                     "시뮬_차로": sim, "로드뷰_차로": roadview.get(eid, ""),
                     "노드링크_차로": r["lanes"] if r else "", "노드링크_거리_m": r["dist"] if r else "",
                     "노드링크_ID": r["link_id"] if r else "", "노드링크_도로명": r["road_name"] if r else ""})
    print(f"지역 링크 {len(links)}개 읽음")
    print(f"{'구분':8} {'엣지':16} {'길이':>6} {'시뮬':>3} {'로드뷰':>4} {'노드링크':>5} {'거리':>5}  링크 도로명")
    for r in rows:
        print(f"{r['구분']:8} {r['edge_id']:16} {r['길이_m']:6.1f} {r['시뮬_차로']:3} "
              f"{str(r['로드뷰_차로']):>4} {str(r['노드링크_차로']):>5} {str(r['노드링크_거리_m']):>5}  {r['노드링크_도로명']}")
    both = [r for r in rows if r["로드뷰_차로"] != "" and r["노드링크_차로"] != ""]
    agree = [r for r in both if r["로드뷰_차로"] == r["노드링크_차로"]]
    print(f"\n로드뷰 vs 노드링크: {len(agree)}/{len(both)} 일치 "
          f"(길이 기준 {sum(r['길이_m'] for r in agree):.0f}/{sum(r['길이_m'] for r in both):.0f} m)")
    nomatch = [r["edge_id"] for r in rows if r["노드링크_차로"] == ""]
    print("짝 없음:", nomatch)
    if args.csv:
        with open(args.csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("저장:", args.csv)


if __name__ == "__main__":
    main()
