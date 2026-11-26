"""Rebuild physical event locations exclusively from saved traces and episode JSON."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/overnight_s6_final"
OUT=ROOT/"results/samil_counterfactual"
REPORT=ROOT/"docs/reports/2026-11-26_s6_위치재집계.md"


import sys
sys.path.insert(0, str(ROOT))
from utils.official_metrics import nearest_interval as nearest


def network_geometry():
    lanes = {}
    geometry = json.loads((OUT/"route_lane_geometry.json").read_text(encoding="utf8"))
    for lane_id, (shape, length) in geometry["lanes"].items():
        points = np.asarray([[float(v) for v in pair.split(",")[:2]]
                             for pair in shape.split()])
        if len(points) > 1:
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            lanes[lane_id] = (arc, points, length)
    return lanes


def analyze(folder, writer, hotspots, counts, unresolved, lanes):
    for f in sorted(folder.glob("*.json")):
        trace=folder/"traces"/(f.stem+".npz")
        if not trace.exists():
            continue
        row=json.loads(f.read_text(encoding="utf8"))
        with np.load(trace) as raw:
            a={k:raw[k] for k in ("edge","lane","pos","speed","time","lane_delta","stop_definition","x","y") if k in raw}
        stops=[]
        definitions=a["stop_definition"]
        for i in np.r_[0,np.flatnonzero(definitions[:-1]!=definitions[1:])+1]:
            st=json.loads(str(definitions[i]))
            if st: stops.append((st["id"],st["edge"],st["start"],st["end"]))
        def event(kind,i,duration,method):
            edge=str(a["edge"][i]); pos=float(a["pos"][i])
            sn,sd=nearest(stops,edge,pos); hn,hd=nearest(hotspots,edge,pos)
            options=sorted((distance,n) for n,distance in ((sn,sd),(hn,hd)) if distance is not None)
            label="기타"
            if options and options[0][0]<=10:
                label="/".join(n for distance,n in options if distance==options[0][0])
            counts[(folder.name,kind,label)]+=1
            x = y = ""
            source = ""
            if "x" in a and np.isfinite(a["x"][i]):
                x, y, source = a["x"][i], a["y"][i], "트레이스"
            elif np.isfinite(pos):
                geometry = lanes.get(f"{edge}_{a['lane'][i]}")
                if geometry:
                    arc, points, length = geometry
                    offset = pos * arc[-1] / length
                    x, y = (round(float(np.interp(offset, arc, points[:, k])), 2) for k in (0, 1))
                    source = "차로 중심선 추정"
            writer.writerow([folder.name,f.stem,kind,a["time"][i],edge,pos,a["lane"][i],
                             x,y,source,
                             sn,sd,hn,hd,label,duration,method])
        changes=np.flatnonzero(a["lane_delta"])
        for prev,i in zip(changes,changes[1:]):
            if i-prev<=10 and a["lane_delta"][prev]*a["lane_delta"][i]<0:
                event("핑퐁",i,"","반전 명령 직전")
        ids=np.flatnonzero(a["speed"]<.1)
        segments=[v for v in np.split(ids,np.where(np.diff(ids)>1)[0]+1) if len(v)>=120]
        cursor=-1
        for h in row.get("inf_wait_runs",[]):
            # Old archives store duration but omit timestamps/red-signal masks.
            # Match only a unique physically consistent stationary segment.
            n=h["steps"]; edge=h.get("edge","")
            candidates=[v for v in segments if v[-1]>cursor and len(v)>=n-1 and
                        (not edge or str(a["edge"][v[0]])==edge)]
            exact=[v for v in candidates if abs(len(v)-n)<=1]
            match=exact if exact else candidates
            if len(match)!=1 or (not exact and
                    (np.ptp(a["pos"][match[0]])>1 or len(set(a["edge"][match[0]]))!=1)):
                unresolved[folder.name]+=1
                continue
            v=match[0]; cursor=int(v[-1])
            event("장기대기",v[0],h["sec"],"길이 일치" if exact else "정지 위치 일치·시작시각 근사")


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    with (ROOT/"data/road_data/illegal_parking_hotspots.csv").open(encoding="utf-8-sig") as f:
        hotspots=[(r["hotspot_id"],r["edge_id"],float(r["start_pos"]),float(r["end_pos"])) for r in csv.DictReader(f)]
    counts=Counter(); unresolved=Counter()
    lanes=network_geometry()
    with (OUT/"location_events.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        w.writerow("모델 시드 사건 SUMO초 edge pos_m 차로 x y 좌표출처 승강장 승강장거리_m 주정차구간 주정차구간거리_m 분류 지속초 위치복원".split())
        for folder in sorted(BASE.iterdir()):
            if (folder/"traces").is_dir(): analyze(folder,w,hotspots,counts,unresolved,lanes)
    lines=["# 저장 트레이스 기반 위치 재집계","","동일 edge의 종방향 구간까지 거리를 계산했다. 구간 안은 0 m, 경계 밖 10 m 이내는 가장 가까운 시설, 나머지는 기타다. 위치 분류는 해당 차로 점유나 이륜차의 실제 배치를 뜻하지 않는다. 차로와 두 거리는 CSV에 함께 기록했다.",
           "", "핑퐁은 5초 안 실제 반대 차로 이동이다. 장기 대기는 기존 적색 제외 공식 기록의 길이와 정지 트레이스를 맞췄다. 여러 후보가 있으면 미확정으로 남겼다. 과거 트레이스에 x,y가 없으면 대조 주행에 쓰인 6단계 네트워크 차로 중심선에서 추정해 구분 표시했다. 거리 분류에는 원본 edge·pos를 사용했다.",
           "", "[전체 사건 좌표 CSV](../../results/samil_counterfactual/location_events.csv)", "",
           "| 모델 | 사건 | 실제 위치 분류 | 발생 횟수 |","|---|---|---|---:|"]
    for (group,kind,label),n in sorted(counts.items()):
        lines.append(f"| {group} | {kind} | {label} | {n} |")
    for group,n in sorted(unresolved.items()): lines.append(f"| {group} | 장기대기 | 위치 미확정 | {n} |")
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf8")


if __name__=="__main__": main()

