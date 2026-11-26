"""Render the paired experiment report and unaltered source frame links."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
from samil_counterfactual import OUT,ROOT,EDGE

REPORT=ROOT/"docs/reports/2026-11-26_s6_samil_대조.md"


def render():
    lines=["# 삼일대로 동쪽 주정차 대조 실험","",
           "정류장 bs_03은 차로 1의 30–45 m, samil_east는 같은 차로의 52–70 m다. 동일 모델·시드에서 DISABLE_SAMIL_EAST=1만 바꿨다. 이륜차 종류·수 추첨과 주변 차량 초기 배치용 자리 예약은 두 조건에서 동일하게 처리했다. 실제 배치 수는 SUMO에 존재한 차량 ID와 좌표로 확인했다.",
           "", "핑퐁은 bs_03을 향하는 동안 같은 edge의 25–50 m에서 관측된 5초 이내 실제 반대 차로 이동 횟수다. 정차 성공은 환경의 stop_success로 확인했다. 주정차 구간 최장 정지는 52–70 m 안에서 속도 0.1 m/s 미만인 연속 시간이다. 이 지표는 통제 비교용으로 신호색을 걸러내지 않는다.",
           "", "[기존 판 위치 재집계](2026-11-26_s6_위치재집계.md)", "",
           "| 정책 | 시드 | 이륜차 조건 | 실배치 | bs_03 핑퐁 | 10초 정차 | samil 최장 정지 | 종료 |",
           "|---|---:|---|---:|---:|---|---:|---|"]
    scenes=[]
    for model,seed in (("student",900103),("recovery",900100),("teacher",900103),("teacher",900100)):
        for condition in ("on","off"):
            folder=OUT/f"{model}_{seed}_{condition}"
            if not (folder/"episode.json").exists(): continue
            rec=json.loads((folder/"episode.json").read_text(encoding="utf8"))
            lines.append(f"| {model} | {seed} | {condition} | {len(rec['parked'])} | {rec['bs03_ping']} | "
                         f"{'성공' if rec['bs03_dwell'] else '실패'} | {rec['samil_stop_sec']:.1f}초 | {rec['reason']} |")
            with np.load(folder/"trace.npz") as z: a=dict(z)
            area=(a["edge"]==EDGE)&(a["pos"]>=24)&(a["pos"]<=77)
            visible=np.flatnonzero(area)
            if not len(visible):
                scenes.append(f"- {folder.name} 해당 구간에 도달하지 않았다.")
                continue
            dwell=area&(a["pos"]>=30)&(a["pos"]<=45)&(a["lane"]==1)&(a["speed"]<0.1)
            park=area&(a["pos"]>=52)&(a["pos"]<=70)&(a["speed"]<0.1)
            event=np.flatnonzero(dwell if seed==900103 else park)
            j=event[0] if len(event) else visible[len(visible)//2]
            departures=np.flatnonzero(area&(np.arange(len(area))>j)&(a["lane"]!=a["lane"][j]))
            k=departures[0] if len(departures) else (event[-1] if len(event) else visible[-1])
            indices=[visible[0],j,min(k,visible[-1]),visible[-1]]
            times=[float(a["time"][i]) for i in indices]
            files=[folder/f"{t:07.1f}.png" for t in times]
            if not all(f.exists() for f in files):
                raise RuntimeError(f"Missing frames: {files}")
            montage=Image.new("RGB",(948,4*225),"white")
            draw=ImageDraw.Draw(montage)
            for n,(f,t,i) in enumerate(zip(files,times,indices)):
                with Image.open(f) as img:
                    montage.paste(img.convert("RGB").crop((0,350,948,550)),(0,n*225+24))
                draw.text((8,n*225+4),f"{model}  seed={seed}  {condition}   SUMO {t:.1f}s   pos={a['pos'][i]:.1f}m   lane={a['lane'][i]}   speed={a['speed'][i]:.2f}m/s",fill="black")
            montage.save(folder/"scene.jpg",quality=92)
            prefix=f"../../results/samil_counterfactual/{folder.name}"
            scenes.extend([f"### {model} {seed} · {condition}", "",
                           f"실제 주정차 앞범퍼 위치(m): {', '.join(format(v[2],'.1f') for v in rec['parked'].values()) or '없음'}.",
                           "", f"![시간순 장면]({prefix}/scene.jpg)", "",
                           " · ".join(f"[{t:.1f}초 원본]({prefix}/{f.name})" for t,f in zip(times,files)), ""])
            if model=="teacher":
                gate=area&(a["pos"]>=25)&(a["pos"]<=50)&(a["stop"]=="bs_03")
                override=gate&(a["stop_cmd"]==0)&(a["final_cmd"]!=0)
                scenes.extend([f"정류장 규칙의 유지가 후속 규칙에서 변경으로 바뀐 스텝 {np.count_nonzero(override)}회. "
                               f"그중 실제 차로가 바뀐 스텝 {np.count_nonzero(override&(a['lane_delta']!=0))}회.", ""])
    lines.extend(["","## 화면 및 규칙 로그","",*scenes])
    review=OUT/"review.txt"
    if review.exists(): lines.extend([review.read_text(encoding="utf8")])
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf8")


if __name__=="__main__": render()

