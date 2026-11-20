"""버스 경로 위 신호의 시간을 실측값으로 맞춘다 (data/road_data/signal_timing.csv).

netconvert가 자동으로 만든 신호 계획(주기 90초)의 단계 순서와 신호 모양(어느 방향이
녹색인지)은 그대로 두고, 단계 시간만 늘이거나 줄여서 버스 방향이

    녹색 ego_green 초 → 황색 ego_yellow 초 → 적색 (cycle - 녹색 - 황색) 초

가 되게 한다. 적색 동안의 다른 방향 단계(교차 도로 녹색, 보행 신호)는 원래 비율대로
늘어난다. 학습에 필요한 특징은 "버스 방향 신호가 언제 얼마나 녹색인가"이므로
이것만 실측과 맞추고, 교차 방향의 세부 현시는 맞추지 않는다 (화면에만 보임).

CSV 열: cross_street(교차 도로 이름, SUMO 교차로를 찾는 열쇠), name, cycle, ego_green,
        ego_yellow, source, note
"""
import csv
import os
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMING_CSV = os.path.join(ROOT, "data", "road_data", "signal_timing.csv")
MIN_PHASE = 1.0   # 늘이거나 줄여도 단계 하나는 이 시간(s)보다 짧아지지 않게


def load_timing(path=TIMING_CSV):
    """교차 도로 이름 → {cycle, green, yellow, name, source}. 파일이 없으면 빈 dict."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if not (r.get("cycle") or "").strip():
                continue
            out[r["cross_street"].strip()] = {
                "cycle": float(r["cycle"]), "green": float(r["ego_green"]),
                "yellow": float(r["ego_yellow"]), "name": r.get("name", ""),
                "source": r.get("source", "")}
    return out


def _scale(durs, target):
    """durs 합이 target이 되게 비율대로 늘이거나 줄인다 (단계마다 MIN_PHASE 이상)."""
    total = sum(durs)
    if total <= 0:
        return durs
    new = [max(MIN_PHASE, d * target / total) for d in durs]
    # MIN_PHASE로 올린 만큼 나머지에서 뺀다
    extra = sum(new) - target
    big = [i for i, d in enumerate(new) if d > MIN_PHASE]
    for i in big:
        new[i] -= extra * new[i] / sum(new[j] for j in big)
    return new


def apply(net_path, rou_path, route_id="r0", path=TIMING_CSV, log=print):
    """net.xml의 경로 위 신호 시간을 CSV 값으로 바꿔 쓴다. 반환: 바꾼 신호 수."""
    timing = load_timing(path)
    if not timing:
        log("[신호] signal_timing.csv 없음 — netconvert 기본 신호 그대로")
        return 0
    tree = ET.parse(net_path)
    root = tree.getroot()
    route = next((r.get("edges").split() for r in ET.parse(rou_path).getroot().iter("route")
                  if r.get("id") == route_id), None)
    if route is None:
        log(f"[신호] route {route_id} 없음")
        return 0
    names = {e.get("id"): e.get("name") or "" for e in root.iter("edge")}
    ends = {e.get("id"): e.get("to") for e in root.iter("edge") if e.get("function") is None}
    edges_at = {}
    for e in root.iter("edge"):
        if e.get("function") is None:
            for n in (e.get("from"), e.get("to")):
                edges_at.setdefault(n, set()).add(e.get("id"))
    logics = {t.get("id"): t for t in root.iter("tlLogic")}
    # 경로 위 신호: (tl id, 버스 링크 번호, 교차 도로 이름들)
    done = 0
    seen = set()
    for a, b in zip(route[:-1], route[1:]):
        con = next((c for c in root.iter("connection")
                    if c.get("from") == a and c.get("to") == b and c.get("tl")), None)
        if con is None or con.get("tl") in seen:
            continue
        tl, link = con.get("tl"), int(con.get("linkIndex"))
        seen.add(tl)
        node = ends.get(a)
        cross = {names.get(e, "") for e in edges_at.get(node, ())} - {""}
        key = next((k for k in timing if any(k in c for c in cross)), None)
        if key is None:
            log(f"[신호] {tl}: 교차 도로 {sorted(cross)} — CSV에 없어 기본 신호 유지")
            continue
        t = timing[key]
        logic = logics[tl]
        phases = list(logic.iter("phase"))
        kinds = []
        for p in phases:
            st = p.get("state")[link]
            kinds.append("G" if st in "Gg" else "y" if st == "y" else "r")
        target = {"G": t["green"], "y": t["yellow"], "r": t["cycle"] - t["green"] - t["yellow"]}
        if target["r"] <= 0 or not all(k in kinds for k in "Gyr"):
            log(f"[신호] {tl} ({t['name']}): 버스 방향 녹·황·적이 모두 있지 않아 건너뜀 {kinds}")
            continue
        for kind in "Gyr":
            idx = [i for i, k in enumerate(kinds) if k == kind]
            new = _scale([float(phases[i].get("duration")) for i in idx], target[kind])
            for i, d in zip(idx, new):
                phases[i].set("duration", f"{d:.2f}")
                for attr in ("minDur", "maxDur"):      # 고정 시간 신호로
                    if attr in phases[i].attrib:
                        del phases[i].attrib[attr]
        logic.set("type", "static")
        done += 1
        log(f"[신호] {t['name']} ({key}): 주기 {t['cycle']:g} s, 버스 방향 녹 {t['green']:g} "
            f"/ 황 {t['yellow']:g} / 적 {target['r']:g} s — {t['source']}")
    tree.write(net_path, encoding="UTF-8", xml_declaration=True)
    return done
