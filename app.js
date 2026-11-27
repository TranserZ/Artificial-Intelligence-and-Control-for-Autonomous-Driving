/* 청계로드 프로젝트 페이지 */

// 제출 링크. 비어 있으면 "준비 중"으로 보인다. 링크가 생기면 여기만 채우면 된다.
//   slides:  발표 PDF. site 폴더에 slides.pdf 로 넣고 "slides.pdf" 라고 적으면 로그인 없이 열린다.
//   dataset: Hugging Face 데이터셋 주소 (https://huggingface.co/datasets/...)
const LINKS = {
  slides: "",
  dataset: "https://huggingface.co/datasets/CheonggyeRoad/Dataset",
};

for (const a of document.querySelectorAll("a[data-link]")) {
  const key = a.dataset.link;
  if (!(key in LINKS)) continue;
  if (LINKS[key]) {
    a.href = LINKS[key];
    const sub = a.querySelector("span");
    if (sub) sub.textContent = sub.textContent.replace(" · 준비 중", "");
  } else {
    a.removeAttribute("href");
    a.setAttribute("aria-disabled", "true");
  }
}

// ── 노선 (출발점에서 잰 거리, m) ───────────────────
const LEN = 2137;
const STOPS = [
  [73, "청계광장"],
  [528, "청계1가·광교"],
  [985, "청계2가·삼일교"],
  [1377, "청계3가·관수교"],
  [1760, "청계4가·배오개다리"],
];
const SIGNALS = [153, 436, 859, 923, 1067, 1320, 1803];
const ZONE = [1335, 1597];

const NS = "http://www.w3.org/2000/svg";
const X0 = 40, X1 = 1080;
const xOf = (m) => X0 + (Math.min(Math.max(m, 0), LEN) / LEN) * (X1 - X0);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function el(parent, tag, attrs, text) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (text != null) n.textContent = text;
  parent.appendChild(n);
  return n;
}

function drawLine(svg, y) {
  const blue = css("--blue"), soft = css("--blue-soft"), ink = css("--ink"), zone = css("--zone");
  el(svg, "line", { x1: X0 - 22, x2: X1 + 14, y1: y, y2: y, stroke: soft, "stroke-width": 8, "stroke-linecap": "round" });
  el(svg, "path", { d: `M${X1 + 8} ${y - 9} l11 9 l-11 9`, fill: "none", stroke: soft, "stroke-width": 4, "stroke-linecap": "round", "stroke-linejoin": "round" });
  const done = el(svg, "line", { x1: X0 - 22, x2: X0 - 22, y1: y, y2: y, stroke: blue, "stroke-width": 8, "stroke-linecap": "round" });
  el(svg, "line", { x1: xOf(ZONE[0]), x2: xOf(ZONE[1]), y1: y - 11, y2: y - 11, stroke: zone, "stroke-width": 4 });
  el(svg, "text", { x: (xOf(ZONE[0]) + xOf(ZONE[1])) / 2, y: y - 19, "text-anchor": "middle", class: "t-zone" }, "어린이보호구역 20");
  for (const m of SIGNALS) {
    el(svg, "line", { x1: xOf(m), x2: xOf(m), y1: y - 4, y2: y + 18, stroke: ink, "stroke-width": 1.5 });
    el(svg, "circle", { cx: xOf(m), cy: y + 21, r: 4, fill: ink });
  }
  const rings = STOPS.map(([m, name]) => {
    const c = el(svg, "circle", { cx: xOf(m), cy: y, r: 10, fill: "#fff", stroke: blue, "stroke-width": 5 });
    el(svg, "text", { x: xOf(m), y: y + 50, "text-anchor": "middle", class: "t-name" }, name);
    el(svg, "text", { x: xOf(m), y: y + 67, "text-anchor": "middle", class: "t-km" }, m.toLocaleString("ko-KR") + " m");
    return c;
  });
  return { done, rings };
}

// ── 주행 기록과 영상 맞추기 ────────────────────────
const RUN = window.RUN;
const line = document.getElementById("line");
if (line && RUN) {
  const Y = 58;
  const { done, rings } = drawLine(line, Y);
  const bus = el(line, "g", { class: "bus" });
  el(bus, "rect", { x: -17, y: -10, width: 34, height: 20, rx: 5, fill: css("--here"), stroke: css("--ink"), "stroke-width": 2 });
  el(bus, "rect", { x: 5, y: -6, width: 7, height: 12, rx: 1.5, fill: css("--ink") });

  const video = document.getElementById("runVideo");
  const btn = document.getElementById("playBtn");
  const out = {
    t: document.getElementById("rT"), v: document.getElementById("rV"),
    d: document.getElementById("rD"), s: document.getElementById("rS"),
    next: document.getElementById("nextStop"),
  };
  const scroller = document.getElementById("lineScroll");
  const N = RUN.odo.length;
  let lastIdx = -1;

  function show(idx) {
    idx = Math.min(Math.max(idx, 0), N - 1);
    if (idx === lastIdx) return;
    lastIdx = idx;
    const m = RUN.odo[idx], x = xOf(m);
    bus.setAttribute("transform", `translate(${x.toFixed(1)} ${Y})`);
    done.setAttribute("x2", x.toFixed(1));
    const stopped = RUN.stops[idx];
    rings.forEach((c, i) => c.setAttribute("fill", i < stopped ? css("--blue") : "#fff"));
    const sec = Math.round(idx * 0.5);
    out.t.textContent = Math.floor(sec / 60) + ":" + String(sec % 60).padStart(2, "0");
    out.v.textContent = RUN.v[idx];
    out.d.textContent = m.toLocaleString("ko-KR");
    out.s.textContent = stopped;
    out.next.textContent = stopped < STOPS.length ? STOPS[stopped][1] : "종점";
    if (scroller.scrollWidth > scroller.clientWidth + 4) {
      const px = (x / 1120) * scroller.scrollWidth;
      scroller.scrollLeft = px - scroller.clientWidth / 2;
    }
  }
  show(0);

  function tick() {
    show(Math.floor(video.currentTime * RUN.fps));
    if (!video.paused) requestAnimationFrame(tick);
  }
  function setBtn() {
    const playing = !video.paused;
    btn.textContent = playing ? "멈춤" : "재생";
    btn.setAttribute("aria-pressed", String(playing));
  }
  video.addEventListener("play", () => { setBtn(); requestAnimationFrame(tick); });
  video.addEventListener("pause", setBtn);
  video.addEventListener("seeked", () => show(Math.floor(video.currentTime * RUN.fps)));
  btn.addEventListener("click", () => { video.paused ? video.play() : video.pause(); });
  video.addEventListener("click", () => { video.paused ? video.play() : video.pause(); });

  function seekFromEvent(ev) {
    const r = line.getBoundingClientRect();
    const vx = ((ev.clientX - r.left) / r.width) * 1120;
    const m = ((vx - X0) / (X1 - X0)) * LEN;
    let i = RUN.odo.findIndex((o) => o >= m);
    if (i < 0) i = N - 1;
    video.currentTime = i / RUN.fps;
    show(i);
  }
  line.addEventListener("click", seekFromEvent);

  const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!calm && "IntersectionObserver" in window) {
    let started = false;
    new IntersectionObserver((entries, io) => {
      if (entries[0].isIntersecting && !started) {
        started = true;
        video.play().catch(() => {});
        io.disconnect();
      }
    }, { threshold: 0.5 }).observe(video);
  }
}

// ── 실제 선형 ──────────────────────────────────────
const plan = document.getElementById("plan");
if (plan && RUN) {
  const pts = RUN.route;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), maxY = Math.max(...ys);
  const minY = Math.min(...ys);
  const k = (X1 - X0) / (maxX - minX);
  const ky = Math.min(k * 3, 62 / (maxY - minY));   // 세로는 그림 높이에 맞춘다
  const px = (p) => [X0 + (p[0] - minX) * k, 44 + (maxY - p[1]) * ky];
  el(plan, "polyline", {
    points: pts.map((p) => px(p).map((v) => v.toFixed(1)).join(",")).join(" "),
    fill: "none", stroke: css("--blue"), "stroke-width": 5, "stroke-linecap": "round", "stroke-linejoin": "round",
  });
  // 정류장: 주행 기록에서 그 거리에 처음 닿은 지점
  let acc = 0; const cum = [0];
  for (let i = 1; i < pts.length; i++) { acc += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); cum.push(acc); }
  for (const [m, name] of STOPS) {
    const target = (m / LEN) * acc;
    const i = Math.max(1, cum.findIndex((c) => c >= target));
    const [x, y] = px(pts[i]);
    el(plan, "circle", { cx: x, cy: y, r: 6, fill: "#fff", stroke: css("--blue"), "stroke-width": 3.5 });
    el(plan, "text", { x, y: y - 14, "text-anchor": "middle" }, name);
  }
  el(plan, "line", { x1: X0, x2: X0 + 200 * k, y1: 134, y2: 134, stroke: css("--ink-2"), "stroke-width": 1.5 });
  el(plan, "text", { x: X0 + 200 * k + 8, y: 138 }, "200 m");
}

// ── 실패 위치 ──────────────────────────────────────
const failLine = document.getElementById("failLine");
if (failLine && window.FAILS) {
  const Y = 118;
  drawLine(failLine, Y);
  const fill = { red: css("--red"), col: css("--ink"), to: "#fff" };
  const stack = {};
  for (const [m, kind] of window.FAILS) {
    const bin = Math.round(m / 40);
    stack[bin] = (stack[bin] || 0) + 1;
    el(failLine, "circle", {
      cx: xOf(bin * 40), cy: Y - 30 - (stack[bin] - 1) * 13, r: 5,
      fill: fill[kind], stroke: kind === "to" ? css("--ink-2") : "none", "stroke-width": 2,
    });
  }
}

// ── 구간 표시, 반복 영상 ───────────────────────────
if ("IntersectionObserver" in window) {
  const links = new Map();
  for (const a of document.querySelectorAll(".bar-nav a")) links.set(a.getAttribute("href").slice(1), a);
  const seen = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      for (const a of links.values()) a.removeAttribute("aria-current");
      const a = links.get(e.target.id);
      if (a) a.setAttribute("aria-current", "true");
    }
  }, { rootMargin: "-30% 0px -65% 0px" });
  for (const id of links.keys()) { const s = document.getElementById(id); if (s) seen.observe(s); }

  const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const loops = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting && !calm) e.target.play().catch(() => {});
      else e.target.pause();
    }
  }, { threshold: 0.35 });
  for (const v of document.querySelectorAll("video.loop")) {
    if (calm) v.controls = true;
    loops.observe(v);
  }
}

// ── 세로 노선: 읽은 위치까지 진하게, 지난 구간은 고리를 채운다 ──
const route = document.getElementById("route");
if (route) {
  const secs = [...route.querySelectorAll(".sec")];
  let queued = false;
  function mark() {
    queued = false;
    const r = route.getBoundingClientRect();
    const at = innerHeight * 0.35;
    const p = Math.min(Math.max(at - r.top, 0), r.height);
    route.style.setProperty("--p", p.toFixed(0) + "px");
    for (const s of secs) s.classList.toggle("passed", s.getBoundingClientRect().top <= at);
  }
  route.classList.add("live");
  mark();
  const ask = () => { if (!queued) { queued = true; requestAnimationFrame(mark); } };
  addEventListener("scroll", ask, { passive: true });
  addEventListener("resize", ask);
}
