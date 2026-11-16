"""청계천로 노면 표시: 어린이보호구역 + 정차금지지대 → cgc.add.xml

근거 데이터
  data/road_data/school_zones.csv   (edge_id, speed_limit_ms, zone_name)
  data/road_data/no_stop_zones.csv  (edge_id, start_pos, end_pos, reason)

그리는 것
  어린이보호구역 (zone_name 첫 단어로 묶음: "세운상가", "배오개" …)
    - 적색 미끄럼방지포장: 구역 첫 엣지 앞 10 m, 전 차로    type="school_zone" + param speed_limit
    - 노란 이중 가장자리선: 구역 엣지마다 오른쪽 끝        type="school_zone_line" + param zone_id
    - 노면 (30)·"어린이 보호구역" 글자, 시작·해제 표지판   type="school_zone_mark" (보기용)
  정차금지지대
    - 노란 테두리 + 45° 빗금, 전 차로                      type="no_stop_zone"

env/sumo_env.py 의 _setup_school_zones 는 위 type 으로 보호구역을 읽는다
(적색 포장 = 시작·제한속도, 같은 zone_id 노란선 = 범위). 그래서 이 표시가 있어야
관측 [40, 41] (보호구역까지 거리·제한속도)와 구역 접근·과속 감점이 동작한다.

단독 실행 (cgc.add.xml 의 표시 부분만 다시 쓴다. 나머지 시나리오는 그대로):
    python tools/zone_markings.py
make_cgc_scenario.py 도 cgc.add.xml 을 만들 때 이 모듈을 부른다.
"""
import argparse
import csv
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROAD_DATA = os.path.join(ROOT, "data", "road_data")
BEGIN = "    <!-- zone-markings:begin (tools/zone_markings.py가 생성) -->"
END = "    <!-- zone-markings:end -->"

PAVE = "150,35,35"        # 적색 포장 (천호사거리 road_builder SCHOOL.pave_color와 같은 색)
YELLOW = "255,200,0"      # 노란 표시 (road_builder SCHOOL.line_color와 같은 색)
ROAD_LAYER = 20.0
LINE_LAYER = 300.1        # SUMO 차선 표시보다 위 (road_builder와 같은 층)
ASSET_REL = "../assets"   # env/cheonggyecheon/cgc.add.xml 기준 그림 경로
ASSET = {                 # env/school_assets.py FILES/SIZE와 같은 값 (그 모듈은 PIL이 필요해 복사)
    "speed": ("school_speed30.png", 2.0, 2.8),
    "text": ("school_text.png", 2.3, 5.0),
    "sign_start": ("school_sign_start.png", 4.0, 2.6),
    "sign_end": ("school_sign_end.png", 4.0, 2.0),
}
PAVE_LEN = 10.0           # 시점부 적색 포장 길이 (m)
HATCH_STEP = 2.0          # 정차금지 빗금 간격 (m)
HATCH_W = 0.35            # 빗금 굵기 (m)
BORDER_W = 0.15           # 테두리 굵기 (m)


# ──────────────────────────────────────────────────────────────
# 차로 좌표계: s = 엣지 시작부터 거리, t = 맨 오른쪽 차도 차로의 오른쪽 끝에서 왼쪽으로 거리
# ──────────────────────────────────────────────────────────────
def _car_lanes(edge):
    return [ln for ln in edge.getLanes() if ln.allows("passenger") or ln.allows("bus")]


class EdgeFrame:
    def __init__(self, edge):
        lanes = _car_lanes(edge)
        self.edge = edge
        self.base = lanes[0]                                   # 맨 오른쪽 차도 차로
        self.shape = [tuple(p[:2]) for p in self.base.getShape()]
        self.length = self.base.getLength()
        geo = sum(math.dist(a, b) for a, b in zip(self.shape, self.shape[1:]))
        self.scale = geo / self.length if self.length > 0 else 1.0   # 차로 길이 ↔ 도형 길이
        self.half0 = self.base.getWidth() / 2.0
        self.width = sum(ln.getWidth() for ln in lanes)        # 차도 전체 폭
        self.centers = []                                       # 차로 중심 t
        t = 0.0
        for ln in lanes:
            self.centers.append(t + ln.getWidth() / 2.0)
            t += ln.getWidth()

    def _at(self, s):
        """s 지점의 (x, y, 진행 방향 단위벡터)."""
        s = min(max(s, 0.0), self.length) * self.scale
        acc = 0.0
        segs = list(zip(self.shape, self.shape[1:]))
        for (x0, y0), (x1, y1) in segs:
            seg = math.dist((x0, y0), (x1, y1))
            if seg < 1e-9:
                continue
            if acc + seg >= s or (x1, y1) == segs[-1][1]:
                k = min(max((s - acc) / seg, 0.0), 1.0)
                return x0 + (x1 - x0) * k, y0 + (y1 - y0) * k, ((x1 - x0) / seg, (y1 - y0) / seg)
            acc += seg
        (x0, y0), (x1, y1) = segs[-1]
        seg = math.dist((x0, y0), (x1, y1)) or 1.0
        return x1, y1, ((x1 - x0) / seg, (y1 - y0) / seg)

    def xy(self, s, t):
        x, y, (ux, uy) = self._at(s)
        off = t - self.half0                                    # 기준 차로 중심에서 왼쪽으로
        return x - uy * off, y + ux * off

    def heading(self, s):
        """그림 위쪽이 진행 방향을 향하게 하는 POI angle.

        sumo-gui(1.28 확인)는 POI 그림을 angle만큼 반시계로 돌린다. 그림 위쪽(북, 수학각 90°)을
        진행 방향(수학각 θ)으로 보내려면 θ - 90."""
        _, _, (ux, uy) = self._at(s)
        return (math.degrees(math.atan2(uy, ux)) - 90.0) % 360.0

    def band(self, s0, s1, t0, t1, step=2.0):
        """s0~s1, t0~t1 사각형 (곡선을 따라 step m마다 꼭짓점)."""
        n = max(1, int(math.ceil((s1 - s0) / step)))
        ss = [s0 + (s1 - s0) * i / n for i in range(n + 1)]
        return [self.xy(s, t0) for s in ss] + [self.xy(s, t1) for s in reversed(ss)]


# ──────────────────────────────────────────────────────────────
def _read_csv(name):
    path = os.path.join(ROAD_DATA, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


class _Writer:
    def __init__(self):
        self.lines, self.n = [], 0

    def poly(self, ptype, color, layer, pts, params=None):
        self.n += 1
        shape = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        head = (f'    <poly id="{ptype}_{self.n}" type="{ptype}" color="{color}" fill="1" '
                f'layer="{layer}" shape="{shape}"')
        if params:
            body = "".join(f'        <param key="{k}" value="{v}"/>\n' for k, v in params.items())
            self.lines.append(f"{head}>\n{body}    </poly>")
        else:
            self.lines.append(head + "/>")

    def image(self, key, x, y, angle, layer):
        self.n += 1
        f, w, h = ASSET[key]
        self.lines.append(f'    <poi id="school_zone_mark_{self.n}" type="school_zone_mark" '
                          f'x="{x:.2f}" y="{y:.2f}" width="{w}" height="{h}" angle="{angle:.1f}" '
                          f'layer="{layer}" imgFile="{ASSET_REL}/{f}"/>')


def marking_lines(net, log=print):
    """cgc.add.xml에 넣을 줄 목록 (BEGIN/END 표시 포함)."""
    w = _Writer()

    # ---- 어린이보호구역 ----
    zones = {}                                   # 이름 → [(엣지, 제한속도)] (CSV 순서 유지)
    for r in _read_csv("school_zones.csv"):
        eid = r["edge_id"].strip()
        if not net.hasEdge(eid):
            log(f"[zone_markings] 보호구역 엣지가 도로망에 없음 (건너뜀): {eid}")
            continue
        name = (r.get("zone_name") or "zone").split()[0]
        zones.setdefault(name, []).append((net.getEdge(eid), float(r["speed_limit_ms"])))
    for zi, (name, items) in enumerate(zones.items()):
        zid = f"z{zi + 1}"
        first, limit = items[0]
        fr = EdgeFrame(first)
        # 1) 시점부 적색 포장 (첫 엣지가 짧으면 그 길이까지)
        s1 = min(PAVE_LEN, fr.length)
        w.poly("school_zone", PAVE, ROAD_LAYER, fr.band(0.0, s1, 0.0, fr.width),
               {"speed_limit": f"{limit:g}", "zone_id": zid, "zone_name": name})
        # 2) 노면 그림: 포장 위 "어린이 보호구역", 그 뒤 (30). 차로마다
        for key, s in (("text", min(4.0, fr.length / 2)), ("speed", min(PAVE_LEN + 4.0, fr.length - 1.5))):
            for t in fr.centers:
                x, y = fr.xy(s, t)
                w.image(key, x, y, fr.heading(s), ROAD_LAYER + 0.3)
        # 3) 노란 이중 가장자리선: 구역 엣지마다
        for edge, _ in items:
            e = EdgeFrame(edge)
            for t0 in (0.10, 0.40):
                w.poly("school_zone_line", YELLOW, LINE_LAYER, e.band(0.0, e.length, t0, t0 + 0.12),
                       {"zone_id": zid})
        # 4) 표지판 (오른쪽 보도 위, 위에서 본 아이콘이라 글자는 북쪽이 위)
        x, y = fr.xy(1.0, -2.0)
        w.image("sign_start", x, y, 0.0, 6.0)
        last = EdgeFrame(items[-1][0])
        x, y = last.xy(last.length - 1.0, -2.0)
        w.image("sign_end", x, y, 0.0, 6.0)
        log(f"[zone_markings] 보호구역 {name} ({zid}): 엣지 {len(items)}개, "
            f"{limit * 3.6:.0f} km/h")

    # ---- 정차금지지대 ----
    for r in _read_csv("no_stop_zones.csv"):
        eid = r["edge_id"].strip()
        if not net.hasEdge(eid):
            log(f"[zone_markings] 정차금지 엣지가 도로망에 없음 (건너뜀): {eid}")
            continue
        e = EdgeFrame(net.getEdge(eid))
        a, b = float(r["start_pos"]), min(float(r["end_pos"]), e.length)
        if a >= b:
            log(f"[zone_markings] 정차금지 구간이 엣지 밖 (건너뜀): {eid} "
                f"{r['start_pos']}~{r['end_pos']} m, 엣지 길이 {e.length:.1f} m — {r.get('reason', '')}")
            continue
        W = e.width
        bw = BORDER_W
        for s0, s1, t0, t1 in ((a, b, 0.0, bw), (a, b, W - bw, W),      # 오른쪽·왼쪽 테두리
                               (a, a + bw, 0.0, W), (b - bw, b, 0.0, W)):  # 앞·뒤 테두리
            w.poly("no_stop_zone", YELLOW, LINE_LAYER, e.band(s0, s1, t0, t1))
        # 45° 빗금: 직선 s = c + t 를 사각형 [a,b]×[0,W] 안으로 자른다
        c = a - W + HATCH_STEP
        while c < b:
            lo, hi = max(0.0, a - c), min(W, b - c)
            if hi - lo > 0.3:
                h = HATCH_W / 2
                pts = [e.xy(c + lo - h, lo), e.xy(c + hi - h, hi), e.xy(c + hi + h, hi), e.xy(c + lo + h, lo)]
                w.poly("no_stop_zone", YELLOW, LINE_LAYER - 0.05, pts)
            c += HATCH_STEP
        log(f"[zone_markings] 정차금지 {eid} {a:g}~{b:g} m — {r.get('reason', '')}")

    return [BEGIN] + w.lines + [END]


def patch_add_file(add_path, net, log=print):
    """cgc.add.xml 에서 BEGIN~END 부분만 새로 쓴다 (없으면 </additional> 앞에 넣는다)."""
    with open(add_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    if BEGIN in lines and END in lines:
        i, j = lines.index(BEGIN), lines.index(END)
        lines = lines[:i] + lines[j + 1:]
    k = max(i for i, ln in enumerate(lines) if ln.strip() == "</additional>")
    lines = lines[:k] + marking_lines(net, log) + lines[k:]
    with open(add_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def _import_sumolib():
    try:
        import sumolib  # noqa: F401
    except ImportError:
        home = os.environ.get("SUMO_HOME", os.path.expanduser("~/src/sumo-1.27.1-mesa"))
        sys.path.append(os.path.join(home, "tools"))
    import sumolib
    return sumolib


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "env", "cheonggyecheon"))
    args = ap.parse_args()
    sumolib = _import_sumolib()
    net = sumolib.net.readNet(os.path.join(args.out_dir, "cgc.net.xml"))
    patch_add_file(os.path.join(args.out_dir, "cgc.add.xml"), net)
    print("완료:", os.path.join(args.out_dir, "cgc.add.xml"))
