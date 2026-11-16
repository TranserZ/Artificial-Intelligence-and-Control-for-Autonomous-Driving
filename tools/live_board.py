"""정류장 대기 인원 실시간 표 — view_road.py --live 가 띄우는 두 번째 창.

sumo-gui 옆에 브라우저 새 창으로 표를 띄운다. 파이썬 표준 라이브러리만 쓴다
(이 PC와 팀원 환경에 tkinter·Qt 같은 데스크톱 GUI 라이브러리가 없어서).

구조
    시뮬레이션 루프 ──update(상태)──▶ LiveBoard (메모리)
    브라우저 표 ◀──1초마다 /state 요청 (JSON)── 로컬 웹 서버 (127.0.0.1, 이 PC에서만 접속)

사용 (view_road.py 안에서):
    board = LiveBoard(stop_order, stop_names)
    board.start()                      # 서버 시작 + 브라우저 창 열기
    board.update(sim_time, waits, buses)
"""
import json
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>정류장 대기 현황</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1d2330; --muted:#6b7385; --line:#e3e6ec;
          --blue:#1e61bf; --green:#5cb026; --amber:#e8a317; --red:#d64545; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font: 15px/1.45 "Noto Sans KR","Malgun Gothic","Apple SD Gothic Neo",sans-serif; }
  header { padding:14px 20px; background:var(--blue); color:#fff;
           display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }
  header h1 { margin:0; font-size:18px; }
  header .t { font-variant-numeric: tabular-nums; opacity:.9; }
  main { padding:16px 20px; display:grid; gap:16px; max-width:900px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  .card h2 { margin:0; padding:10px 14px; font-size:14px; color:var(--muted);
             border-bottom:1px solid var(--line); font-weight:600; }
  table { width:100%; border-collapse:collapse; }
  th, td { padding:8px 14px; text-align:left; border-bottom:1px solid var(--line); }
  th { font-size:12px; color:var(--muted); font-weight:600; background:#fafbfc; }
  tr:last-child td { border-bottom:0; }
  td.n { width:56px; font-variant-numeric: tabular-nums; font-weight:700; font-size:17px; text-align:right; }
  td.seq { width:44px; color:var(--muted); font-variant-numeric: tabular-nums; }
  td.bar { width:40%; }
  .track { height:10px; background:#eef0f4; border-radius:5px; overflow:hidden; }
  .fill { height:100%; background:var(--blue); border-radius:5px; transition: width .4s; }
  .fill.mid { background:var(--amber); } .fill.hi { background:var(--red); }
  .zero { color:#b4bac6; font-weight:400; }
  .tag { display:inline-block; margin-left:6px; padding:1px 8px; border-radius:10px;
         font-size:12px; background:var(--green); color:#fff; }
  .sum { padding:10px 14px; color:var(--muted); font-size:13px; }
  .empty { padding:14px; color:var(--muted); }
  .stale { padding:8px 20px; background:#fff4d6; color:#7a5a00; display:none; }
</style></head><body>
<header><h1>🚏 청계천 정류장 대기 현황</h1><span class="t" id="t">연결 중…</span></header>
<div class="stale" id="stale">시뮬레이션과 연결이 끊겼습니다 (sumo-gui가 닫혔을 수 있음).</div>
<main>
  <section class="card"><h2>정류장별 대기 인원</h2>
    <table><thead><tr><th>#</th><th>정류장</th><th style="text-align:right">대기</th><th></th></tr></thead>
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
let fails = 0;
async function tick() {
  try {
    const r = await fetch("/state", {cache: "no-store"});
    const s = await r.json(); fails = 0;
    document.getElementById("stale").style.display = "none";
    const m = Math.floor(s.time / 60), sec = Math.floor(s.time % 60);
    document.getElementById("t").textContent = `시뮬레이션 ${m}분 ${String(sec).padStart(2,"0")}초`;
    const max = Math.max(5, ...s.stops.map(x => x.waiting));
    document.getElementById("stops").innerHTML = s.stops.map((x, i) => {
      const pct = 100 * x.waiting / max;
      const cls = x.waiting >= 10 ? "hi" : x.waiting >= 5 ? "mid" : "";
      const tag = x.bus_here ? `<span class="tag">버스 정차 중</span>` : "";
      return `<tr><td class="seq">${i+1}</td><td>${esc(x.name)}${tag}</td>
        <td class="n ${x.waiting ? "" : "zero"}">${x.waiting}</td>
        <td class="bar"><div class="track"><div class="fill ${cls}" style="width:${pct}%"></div></div></td></tr>`;
    }).join("");
    const total = s.stops.reduce((a, x) => a + x.waiting, 0);
    document.getElementById("sum").textContent = `전체 대기 ${total}명 · 막대 색: 파랑 0–4명, 주황 5–9명, 빨강 10명 이상`;
    document.getElementById("buses").innerHTML = s.buses.length ? s.buses.map(b =>
      `<tr><td>${esc(b.id)}</td><td><b>${b.onboard}</b> / ${b.capacity}${b.onboard >= b.capacity ? ' <span class="tag" style="background:var(--red)">만차</span>' : ""}</td>
       <td>${esc(b.state)}</td><td>${esc(b.next_stop)}</td><td>${b.speed_kmh.toFixed(0)} km/h</td></tr>`).join("")
      : `<tr><td colspan="5" class="empty">아직 운행 중인 버스가 없습니다 (첫 버스는 60초에 출발)</td></tr>`;
  } catch (e) {
    if (++fails >= 3) document.getElementById("stale").style.display = "block";
  }
}
tick(); setInterval(tick, 1000);
</script></body></html>
"""


class LiveBoard:
    def __init__(self, stop_order, stop_names):
        """stop_order: 표에 나올 정류장 id 순서, stop_names: id → 표시 이름(한글)."""
        self.stop_order = list(stop_order)
        self.stop_names = dict(stop_names)
        self._lock = threading.Lock()
        self._state = {"time": 0.0, "buses": [],
                       "stops": [{"id": s, "name": self.stop_names.get(s, s),
                                  "waiting": 0, "bus_here": False} for s in self.stop_order]}
        self._server = None
        self.url = None

    def start(self, open_browser=True):
        board = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith("/state"):
                    with board._lock:
                        body = json.dumps(board._state, ensure_ascii=False).encode("utf-8")
                    ctype = "application/json; charset=utf-8"
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
        speed_kmh, at_stop}] (at_stop: 지금 정차 중인 정류장 id 또는 None)."""
        here = {b.get("at_stop") for b in buses if b.get("at_stop")}
        with self._lock:
            self._state = {
                "time": float(sim_time),
                "stops": [{"id": s, "name": self.stop_names.get(s, s),
                           "waiting": int(waits.get(s, 0)), "bus_here": s in here}
                          for s in self.stop_order],
                "buses": buses,
            }

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server = None
