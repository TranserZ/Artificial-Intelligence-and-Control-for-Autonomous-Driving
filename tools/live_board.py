"""정류장 대기 현황 창 — view_road.py --live 가 띄우는 두 번째 창.

sumo-gui 옆에 브라우저 새 창으로 띄운다. 파이썬 표준 라이브러리만 쓴다
(이 PC와 팀원 환경에 tkinter·Qt 같은 데스크톱 GUI 라이브러리가 없어서).

화면 구성
    · 지도: 청계천 일대 도로 + 정류장 위치(정류장별 색 원, 번호) + 운행 중인 버스
    · 표: 정류장별 대기 인원 (이름은 정류장 색, 막대는 '지금 가장 많은 정류장 = 100%')
    · 표: 운행 중인 버스 (탑승/정원, 상태, 다음 정류장, 속도)
    정류장 색은 시나리오 생성기가 cgc.line_meta.json에 적어 둔 값 —
    sumo-gui의 정류장 테두리 색과 같다.

구조
    시뮬레이션 루프 ──update(상태)──▶ LiveBoard (메모리)
    브라우저 ◀── /map (처음 한 번: 도로·정류장 좌표), /state (1초마다: 인원·버스) ──
    로컬 웹 서버 (127.0.0.1 — 이 PC에서만 접속)

사용 (view_road.py 안에서):
    board = LiveBoard(stop_order, stop_names, stop_colors, map_data)
    board.start()                      # 서버 시작 + 브라우저 창 열기
    board.update(sim_time, waits, buses)
"""
import json
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>정류장 대기 현황</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1d2330; --muted:#6b7385; --line:#e3e6ec;
          --teal:#2b6570; --bus:#5cb026; --red:#d64545; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font: 15px/1.45 "Noto Sans KR","Malgun Gothic","Apple SD Gothic Neo",sans-serif; }
  header { padding:12px 20px; background:var(--teal); color:#fff;
           display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }
  header h1 { margin:0; font-size:18px; }
  header .t { font-variant-numeric: tabular-nums; opacity:.92; }
  main { padding:16px 20px; display:grid; gap:16px; max-width:1180px;
         grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
  .wide { grid-column: 1 / -1; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  .card h2 { margin:0; padding:10px 14px; font-size:14px; color:var(--muted);
             border-bottom:1px solid var(--line); font-weight:600;
             display:flex; justify-content:space-between; gap:12px; }
  .card h2 small { font-weight:400; }
  table { width:100%; border-collapse:collapse; }
  th, td { padding:7px 14px; text-align:left; border-bottom:1px solid var(--line); }
  th { font-size:12px; color:var(--muted); font-weight:600; background:#fafbfc; }
  tr:last-child td { border-bottom:0; }
  td.n { width:52px; font-variant-numeric: tabular-nums; font-weight:700; font-size:17px; text-align:right; }
  td.bar { width:38%; }
  .stopname { font-weight:700; display:flex; align-items:center; gap:8px; }
  .badge { flex:none; width:22px; height:22px; border-radius:50%; color:#fff;
           font-size:12px; font-weight:700; display:inline-flex; align-items:center; justify-content:center; }
  .track { height:12px; background:#eef0f4; border-radius:6px; overflow:hidden; }
  .fill { height:100%; border-radius:6px; transition: width .4s; }
  .zero { color:#b4bac6; font-weight:400; }
  .tag { display:inline-block; padding:1px 8px; border-radius:10px;
         font-size:12px; font-weight:600; background:var(--bus); color:#fff; }
  .sum { padding:9px 14px; color:var(--muted); font-size:13px; }
  .empty { padding:14px; color:var(--muted); }
  .stale { padding:8px 20px; background:#fff4d6; color:#7a5a00; display:none; }
  #map { display:block; width:100%; height:auto; background:#eef1ec; }
  #map .road { fill:none; stroke:#c9ccd1; stroke-linecap:round; stroke-linejoin:round; }
  #map .main { fill:none; stroke:#8a9099; stroke-linecap:round; stroke-linejoin:round; }
  #map .stoplbl { font-weight:700; paint-order: stroke; stroke:#fff; stroke-width:4px; }
  .legend { display:flex; gap:14px; flex-wrap:wrap; padding:8px 14px; color:var(--muted); font-size:13px;
            border-top:1px solid var(--line); }
  .legend i { display:inline-block; width:12px; height:12px; border-radius:50%; vertical-align:-1px; margin-right:4px; }
</style></head><body>
<header><h1>🚏 청계천 정류장 대기 현황</h1><span class="t" id="t">연결 중…</span></header>
<div class="stale" id="stale">시뮬레이션과 연결이 끊겼습니다 (sumo-gui가 닫혔을 수 있음).</div>
<main>
  <section class="card wide"><h2>청계천 일대 지도 <small>원 = 정류장 (숫자 = 번호, 옆 = 대기 인원) · 초록 사각형 = 운행 중인 버스</small></h2>
    <svg id="map" xmlns="http://www.w3.org/2000/svg"></svg>
    <div class="legend" id="legend"></div>
  </section>
  <section class="card"><h2>정류장별 대기 인원 <small id="maxnote"></small></h2>
    <table><thead><tr><th>정류장</th><th style="text-align:right">대기</th><th></th></tr></thead>
    <tbody id="stops"></tbody></table>
    <div class="sum" id="sum"></div>
  </section>
  <section class="card"><h2>운행 중인 버스</h2>
    <table><thead><tr><th>버스</th><th>탑승 / 정원</th><th>상태</th><th>다음 정류장</th><th>속도</th></tr></thead>
    <tbody id="buses"></tbody></table>
  </section>
</main>
<script>
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const NS = "http://www.w3.org/2000/svg";
let MAP = null, toSvg = null, unit = 1, fails = 0;
const stopEls = {};   // sid → {count:<text>}
let busLayer = null;

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}

async function loadMap() {
  MAP = await (await fetch("/map", {cache: "no-store"})).json();
  const svg = document.getElementById("map");
  const [x0, y0, x1, y1] = MAP.bbox;            // 지도에 보일 범위 (SUMO 좌표, m)
  const W = x1 - x0, H = y1 - y0;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  toSvg = (x, y) => [x - x0, y1 - y];           // SUMO는 위가 +y, SVG는 아래가 +y
  unit = W / 1100;                               // 화면 약 1100px 기준 1px의 크기 (m)
  const pts = shape => shape.map(p => toSvg(p[0], p[1]).map(v => v.toFixed(1)).join(",")).join(" ");
  const roads = el("g", {}, svg);
  // 도로는 실제 폭(m)으로 그리되 너무 가늘면 최소 굵기를 보장한다
  for (const r of MAP.roads.filter(r => !r.main))
    el("polyline", {points: pts(r.shape), class: "road",
                    "stroke-width": Math.max(r.w || 6, 2.2 * unit)}, roads);
  for (const j of (MAP.junctions || []))         // 교차로 면 (도로가 끊겨 보이지 않게)
    el("polygon", {points: pts(j), fill: "#c9ccd1"}, roads);
  for (const r of MAP.roads.filter(r => r.main)) // 청계천로는 진하게, 맨 위에
    el("polyline", {points: pts(r.shape), class: "main",
                    "stroke-width": Math.max(r.w || 8, 3.5 * unit)}, roads);
  busLayer = el("g", {}, svg);                   // 버스는 정류장 아래에 그린다
  const stops = el("g", {}, svg);
  const legend = [];
  for (const s of MAP.stops) {
    const [cx, cy] = toSvg(s.x, s.y);
    const g = el("g", {}, stops);
    el("circle", {cx, cy, r: 11 * unit, fill: s.color, stroke: "#fff", "stroke-width": 2.5 * unit}, g);
    const num = el("text", {x: cx, y: cy + 4.2 * unit, "text-anchor": "middle", fill: "#fff",
                            "font-size": 12 * unit, "font-weight": 700}, g);
    num.textContent = s.number;
    // 대기 인원: 정류장이 도로 위쪽이면 위에, 아래쪽이면 아래에 (서로 겹치지 않게)
    const below = s.side === "below";
    const cnt = el("text", {x: cx, y: cy + (below ? 30 : -18) * unit, "text-anchor": "middle",
                            fill: s.color, "font-size": 14 * unit, class: "stoplbl"}, g);
    el("title", {}, g).textContent = `${s.number}. ${s.name}`;
    stopEls[s.id] = {count: cnt};
    legend.push(`<span><i style="background:${s.color}"></i>${s.number}. ${esc(s.name)}</span>`);
  }
  document.getElementById("legend").innerHTML = legend.join("");
}

function drawBuses(buses) {
  if (!busLayer) return;
  busLayer.innerHTML = "";
  for (const b of buses) {
    if (b.x == null) continue;
    const [cx, cy] = toSvg(b.x, b.y);
    const s = 9 * unit;
    el("rect", {x: cx - s, y: cy - s, width: 2 * s, height: 2 * s, rx: 3 * unit,
                fill: "var(--bus)", stroke: "#fff", "stroke-width": 2 * unit}, busLayer);
    const t = el("text", {x: cx, y: cy + 3.8 * unit, "text-anchor": "middle", fill: "#fff",
                          "font-size": 10 * unit, "font-weight": 700}, busLayer);
    t.textContent = b.onboard;
  }
}

async function tick() {
  try {
    const s = await (await fetch("/state", {cache: "no-store"})).json(); fails = 0;
    document.getElementById("stale").style.display = "none";
    const m = Math.floor(s.time / 60), sec = Math.floor(s.time % 60);
    document.getElementById("t").textContent = `시뮬레이션 ${m}분 ${String(sec).padStart(2,"0")}초`;
    // 막대 100% = 지금 가장 많이 기다리는 정류장 (모두 0명이면 막대 없음)
    const max = Math.max(0, ...s.stops.map(x => x.waiting));
    document.getElementById("maxnote").textContent = max ? `막대 끝 = 최대 ${max}명` : "";
    document.getElementById("stops").innerHTML = s.stops.map(x => {
      const pct = max ? 100 * x.waiting / max : 0;
      const tag = x.bus_here ? ` <span class="tag">버스 정차 중</span>` : "";
      return `<tr><td><span class="stopname" style="color:${x.color}">
          <span class="badge" style="background:${x.color}">${x.number}</span>${esc(x.name)}${tag}</span></td>
        <td class="n ${x.waiting ? "" : "zero"}">${x.waiting}</td>
        <td class="bar"><div class="track"><div class="fill" style="width:${pct}%;background:${x.color}"></div></div></td></tr>`;
    }).join("");
    const total = s.stops.reduce((a, x) => a + x.waiting, 0);
    document.getElementById("sum").textContent = `전체 대기 ${total}명`;
    for (const x of s.stops)
      if (stopEls[x.id]) stopEls[x.id].count.textContent = x.waiting ? `${x.waiting}명` : "";
    drawBuses(s.buses);
    document.getElementById("buses").innerHTML = s.buses.length ? s.buses.map(b =>
      `<tr><td>${esc(b.id)}</td><td><b>${b.onboard}</b> / ${b.capacity}${b.onboard >= b.capacity ? ' <span class="tag" style="background:var(--red)">만차</span>' : ""}</td>
       <td>${esc(b.state)}</td><td>${esc(b.next_stop)}</td><td>${b.speed_kmh.toFixed(0)} km/h</td></tr>`).join("")
      : `<tr><td colspan="5" class="empty">아직 운행 중인 버스가 없습니다 (첫 버스는 60초에 출발)</td></tr>`;
  } catch (e) {
    if (++fails >= 3) document.getElementById("stale").style.display = "block";
  }
}
loadMap().catch(() => {}).finally(() => { tick(); setInterval(tick, 1000); });
</script></body></html>
"""


class LiveBoard:
    def __init__(self, stop_order, stop_names, stop_colors=None, map_data=None,
                 stop_numbers=None):
        """stop_order: 표에 나올 정류장 id 순서, stop_names: id → 표시 이름(한글),
        stop_colors: id → "#RRGGBB", stop_numbers: id → 번호,
        map_data: {"bbox": [x0,y0,x1,y1], "roads": [{shape, main}], "stops": [{id,x,y,side}]}
        (map_data의 stops에는 이름·색·번호를 여기서 채워 넣는다)."""
        self.stop_order = list(stop_order)
        self.stop_names = dict(stop_names)
        self.stop_colors = dict(stop_colors or {})
        self.stop_numbers = dict(stop_numbers or {s: i + 1 for i, s in enumerate(self.stop_order)})
        self._map = map_data or {"bbox": [0, 0, 1, 1], "roads": [], "stops": []}
        for s in self._map.get("stops", []):
            s.setdefault("name", self.stop_names.get(s["id"], s["id"]))
            s.setdefault("color", self._color(s["id"]))
            s.setdefault("number", self.stop_numbers.get(s["id"], ""))
        self._map_json = json.dumps(self._map, ensure_ascii=False).encode("utf-8")
        self._lock = threading.Lock()
        self._state = self._make_state(0.0, {}, [])
        self._server = None
        self.url = None

    def _color(self, sid):
        return self.stop_colors.get(sid, "#2b6570")

    def _make_state(self, sim_time, waits, buses):
        here = {b.get("at_stop") for b in buses if b.get("at_stop")}
        return {"time": float(sim_time), "buses": buses,
                "stops": [{"id": s, "name": self.stop_names.get(s, s), "color": self._color(s),
                           "number": self.stop_numbers.get(s, ""),
                           "waiting": int(waits.get(s, 0)), "bus_here": s in here}
                          for s in self.stop_order]}

    def start(self, open_browser=True):
        board = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith("/state"):
                    with board._lock:
                        body = json.dumps(board._state, ensure_ascii=False).encode("utf-8")
                    ctype = "application/json; charset=utf-8"
                elif self.path.startswith("/map"):
                    body, ctype = board._map_json, "application/json; charset=utf-8"
                elif self.path in ("/", "/index.html"):
                    body, ctype = _PAGE.encode("utf-8"), "text/html; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):   # 요청마다 터미널에 찍히는 로그 끄기
                pass

        # 빈 포트를 자동으로 고른다. 127.0.0.1 = 이 PC에서만 접속 가능
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{port}/"
        if open_browser:
            webbrowser.open_new(self.url)      # 브라우저 새 창
        return self.url

    def update(self, sim_time, waits, buses):
        """waits: {stop_id: 대기 인원}, buses: [{id, onboard, capacity, state, next_stop,
        speed_kmh, at_stop, x, y}] (at_stop: 지금 정차 중인 정류장 id 또는 None)."""
        state = self._make_state(sim_time, waits, buses)
        with self._lock:
            self._state = state

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server = None


def build_map_data(net_path, meta, margin=120.0):
    """sumolib으로 도로망을 읽어 지도용 데이터를 만든다.
    - roads: 승용차가 다닐 수 있는 일반 엣지의 모양 (청계천로는 main=True로 굵게)
    - stops: 정류장 중심 좌표 (정차 차로의 start~end 가운데)
    - bbox: 정류장들을 다 담는 범위 + margin (m)"""
    import sumolib
    from sumolib.geomhelper import positionAtShapeOffset
    net = sumolib.net.readNet(net_path)
    stops = []
    for sid, st in meta["stops"].items():
        try:
            edge = net.getEdge(st["edge"])
        except KeyError:
            continue
        # 정류장은 보도 바로 옆 차로 (보도가 lane 0이면 lane 1)
        lane = edge.getLane(1 if edge.getLaneNumber() > 1 and
                            not edge.getLane(0).allows("passenger") else 0)
        x, y = positionAtShapeOffset(lane.getShape(), (st["start"] + st["end"]) / 2)
        stops.append({"id": sid, "x": round(x, 1), "y": round(y, 1)})
    if not stops:
        return None
    xs = [s["x"] for s in stops]
    ys = [s["y"] for s in stops]
    bbox = [min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin]
    # 정류장 대기 인원 글자 위치: 두 일방통행 도로 중 아래쪽(남쪽) 정류장은 원 아래에
    mid_y = (min(ys) + max(ys)) / 2
    for s in stops:
        s["side"] = "below" if s["y"] < mid_y else "above"
    roads = []
    for e in net.getEdges(withInternal=False):
        if not e.allows("passenger"):
            continue
        shape = e.getShape()
        if all(not (bbox[0] - 50 <= x <= bbox[2] + 50 and bbox[1] - 50 <= y <= bbox[3] + 50)
               for x, y in shape):
            continue                                  # 지도 범위 밖 도로는 뺀다
        roads.append({"shape": [[round(x, 1), round(y, 1)] for x, y in shape],
                      "w": round(sum(l.getWidth() for l in e.getLanes()), 1),   # 실제 도로 폭 (m)
                      "main": "청계천로" in (e.getName() or "")})
    # 교차로 면: 엣지만 그리면 교차로마다 도로가 끊겨 점선처럼 보여서 같이 칠한다
    junctions = []
    for n in net.getNodes():
        shape = n.getShape()
        if len(shape) < 3 or n.getType() == "dead_end":
            continue
        x, y = n.getCoord()[:2]
        if not (bbox[0] - 50 <= x <= bbox[2] + 50 and bbox[1] - 50 <= y <= bbox[3] + 50):
            continue
        junctions.append([[round(px, 1), round(py, 1)] for px, py, *_ in shape])
    return {"bbox": [round(v, 1) for v in bbox], "roads": roads,
            "junctions": junctions, "stops": stops}
