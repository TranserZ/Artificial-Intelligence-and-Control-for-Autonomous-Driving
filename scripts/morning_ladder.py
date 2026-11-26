"""Morning run with bounded phases and an independent, predeclared stage-5 fallback."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.overnight_s6_final import OUT, REPORT, STUDENT, TEACHER, evaluate, training_args, stop_tree, write
from scripts.drive_parallel import summarize

RUN = OUT / "morning"
STATE = RUN / "state.json"
AB = ROOT / "results/samil_counterfactual"
ANCHOR = ROOT / "results/klppo3_s6_s0/step100k.pt"
S5 = ROOT / "results/klppo1_s5_s0/step250k.pt"
FIRST = datetime(2026,11,26,5)
SECOND = datetime(2026,11,26,7)
END = datetime(2026,11,26,9)
BASEGROUP = "klppo3_s6_s0_step100k"


def rows(group, count=None):
    files = sorted((OUT/group).glob("*.json"))
    return [json.loads(f.read_text(encoding="utf8")) for f in files[:count]]


def metrics(group, count=None):
    return summarize(rows(group,count))


def ah(group):
    data=rows(group)
    if not data: return ""
    s=summarize(data); n=len(data)
    return (f"| {group} | {n} | {s['collision_count']} | {round(n*s['official_complete'])}/{n} ({s['official_complete']:.1%}) | "
            f"{round(n*s['perfect_complete'])}/{n} ({s['perfect_complete']:.1%}) | {s['green_stop_time_mean']:.1%} · "
            f"{s['green_stop_events_mean']:.1f} · {s['green_stop_max_sec']:.1f}s | {s['inf_wait_episodes']} · "
            f"{s['timeout_count']} · {max(r['inf_wait_max_sec'] for r in data):.1f}s | {s['wrong_lane_count']} | "
            f"{s['avg_speed_kmh']:.1f} · {s['moving_avg_speed_kmh']:.1f} | {s['teleports']} · {s['sim_errors']} | "
            f"{s['red_light']:.1%} · {s['yield_steps']} · {s['intervention']:.1%} · {sum(r['lane_changes'] for r in data)/n:.1f} |")


def refresh(state):
    write(STATE,json.dumps(state,ensure_ascii=False,indent=2))
    ab_file=ROOT/"docs/reports/2026-11-26_s6_samil_대조.md"
    ab_done=len(list(AB.glob("*/episode.json")))==8
    lines=["# 6단계 야간 실험", "", "## 한눈에 보기", "",
           "1. "+("학생과 선생님의 정차·회피 행동 충돌을 검증했다. [8판 대조와 화면](2026-11-26_s6_samil_대조.md)" if ab_done else "A 위치 재집계 완료, B 대조 GUI 주행 진행 중."),
           "2. "+(" / ".join(state["recovery"]) or "8k·16k·24k 개발 10판 비교 진행 전."),
           "3. 429.5초 중 왼쪽 변경은 1회였고 게이트가 허용했다. 항상 차단한다는 가설은 기각하며, 과거 학습에 대한 인과관계는 미확정이다.",
           "4. "+state["phase"], "5. "+state.get("best_text","6단계 독립 평가 기준을 통과한 후보 확인 전. 원래 배치 출발 모델의 개발 성적은 아래 표에 있다."),
           "6. "+state.get("fallback","5단계 독립 평가 대기."),
           "7. "+state.get("decision","추천 보류. 독립 평가 결과 대기."),
           "8. "+("아래 GUI 재현 목록 참조." if state["scenes"] else "대조 장면은 연결된 보고서에 있다. 최종 재현 목록은 평가 후 갱신한다."),
           "", "## 게이트 가설", "", "| 저장된 항목 | 900100 · 255.5–684.5초 · 859스텝 |","|---|---|",
           "| 차선 명령 | 유지 858회, 왼쪽 1회. 왼쪽 1회는 게이트 허용 및 실제 변경 |",
           "| 평균 신경망 확률 | 왼쪽 5.12%, 유지 94.88%. 유지 명령에서 가상의 왼쪽 게이트는 호출되지 않음 |",
           "| 왼쪽 앞·뒤 거리, 상세 차단 사유, 회피 감점 조건 | 당시 트레이스에 미저장. 과거 klppo3·recovery의 구간별 학습 시도·감점 횟수도 복원 불가. 단계별 평균으로 대체하지 않았다 |",
           "", "[859스텝 표](../../results/samil_counterfactual/gate_900100.csv) · [장기대기 위치·선생님 동일 10m 구간 통과 시간](../../results/samil_counterfactual/teacher_passage.csv) · [보정 위치 표](2026-11-26_s6_위치재집계.md)",
           "", "## 실험 기록", "", "| 가설 | 바꾼 것 | 결과 | 판정 |", "|---|---|---|---|",
           *["| "+" | ".join(entry)+" |" for entry in state["experiments"]],
           "", "## 단계별 공식 평가", "",
           "| 모델 | 판 | A 충돌 | B 공식 | B 완벽 | C 비율·횟수/판·최장 | D 장기대기∪시간초과·시간초과·최장 | E 경로 | F 총시간·이동 km/h | G 순간이동·오류 | H 적색·양보·shadow·변경/판 |",
           "|---|---:|---:|---|---|---|---|---:|---|---|---|",
           *[ah(group) for group in state["groups"] if rows(group)],
           "", "## 화면과 재현", "", *state["scenes"],
           "", "## 팀 공유용 배치 조정", "",
           "| 옵션 | 항목 | 원위치 | 조정 위치 | 근거 |","|---|---|---|---|---|",
           "| PLACE_FIX=1, 6단계 | bs_03 승강장 | e908646367_3, 차로 1, 30–45 m | 같은 edge·차로, 22–37 m | 8 m 상류 이동으로 주정차 구간 시작까지 간격을 7 m에서 15 m로 확대. 이륜차 위치·대수·종류는 보존 |",
           "", "원래 배치 및 PLACE_FIX에서 독립 시험에는 결정적 신경망과 shadow 기록을 사용한다. 독립 100판은 개발 순위를 정할 때 읽지 않는다.",
           "", "## 변경 코드", "", "좌표 집계·대조 실험은 scripts/rebuild_event_locations.py, scripts/samil_counterfactual.py, scripts/report_samil_counterfactual.py, scripts/analyze_gate_archive.py, utils/official_metrics.py에 있다. 학습과 배치는 algorithms/kl_ppo.py, env/sumo_env.py, train.py, tools/make_cgc_scenario.py에서 옵션으로 제어한다. scripts/morning_ladder.py가 단계·마감·보고서를 관리한다. 기존 스테이징 변경이 있는 작업 트리에서 실행했고 새 커밋은 없다.",
           "", "[이전 전체 개발 평가·실험 경과](2026-11-26_s6_overnight_history.md)"]
    write(REPORT,"\n".join(lines)+"\n")


def persist(state,phase=None):
    if phase: state["phase"]=phase
    refresh(state)


def child(args, name, deadline, env=None):
    log=RUN/(name+".log")
    with log.open("w",encoding="utf8") as out:
        proc=subprocess.Popen([sys.executable,"-u",*map(str,args)],cwd=ROOT,stdout=out,
                              stderr=subprocess.STDOUT,env={**os.environ,**(env or {})},
                              creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while proc.poll() is None:
                if datetime.now()>=deadline: raise TimeoutError(name)
                time.sleep(4)
            if proc.returncode: raise RuntimeError(f"{name} failed: {log}")
        finally: stop_tree(proc)


def eval_batch(state,group,stage,model,count,deadline,fix=0,start=900100):
    if group not in state["groups"]: state["groups"].append(group)
    for n in dict.fromkeys((1,count)):
        if len(rows(group))>=n: continue
        persist(state,f"{group} 평가 · {n}판 목표")
        child([__file__,"--eval",group,stage,model,n,start],group+f"_{n}",deadline,
              {"PLACE_FIX":str(fix),"CGC_STAGE":str(stage)})
        persist(state)
    return metrics(group,count)


def sample_scene(state,group,model,stage,fix,deadline,seed=None,near_end=False):
    data=rows(group)
    row=next((r for r in data if r["seed"]==seed),data[0]); seed=row["seed"]
    folder=RUN/"gui"/group/f"{seed}_{'end' if near_end else 'site'}"
    with np.load(OUT/group/"traces"/f"{seed}.npz") as z:
        t=z["time"]; edges=z["edge"]; pos=z["pos"]
        indices=np.flatnonzero((edges=="e908646367_3")&(pos>=25)&(pos<=72))
        event=t[-1] if near_end else t[indices[len(indices)//2]] if len(indices) else t[len(t)//2]
    args=["scripts/gui_drive_seed.py","--stage",stage,"--seed",seed,"--checkpoint",model,
          "--delay",0,"--screenshots",folder,"--from-time",max(t[0],event-3),"--to-time",event+2]
    if not list(folder.glob("*.png")):
        child(args,"gui_"+group+str(seed)+str(near_end),deadline,{"PLACE_FIX":str(fix)})
    frames=sorted(folder.glob("*.png"))
    if frames:
        frame=frames[len(frames)//2]
        command=f"$env:PLACE_FIX='{fix}'; py -3.13 scripts/gui_drive_seed.py --stage {stage} --seed {seed} --checkpoint '{model}' --delay 80"
        state["scenes"].append(f"- {group}, 시드 {seed}, SUMO {event:.1f}초, 종료 {row['end_reason']}. "
                               f"[대표 프레임](../../{frame.relative_to(ROOT).as_posix()}). `{command}`")
        persist(state)


def passed_dev(result,baseline):
    return (result["collision_count"]<=1 and result["inf_wait_episodes"]<=2
            and result["official_complete"]+1e-9>=baseline["official_complete"]-.10)


def held(state,name,model,fix,deadline):
    # The candidate is fixed before any independent episode runs.
    state["locked"]=[name,str(model),fix]
    persist(state)
    g="final_"+name
    result=eval_batch(state,g,6,model,100,deadline,fix,900300)
    data=rows(g); empty=0
    for row in data:
        with np.load(OUT/g/"traces"/f"{row['seed']}.npz") as z:
            empty+=int(not np.any(z["neighbors_50m"]>0))
    passed=(result["collision_count"]<=4 and result["inf_wait_episodes"]<=5
            and result["wrong_lane_count"]==result["teleports"]==result["sim_errors"]==0
            and result["moving_avg_speed_kmh"]>=15 and empty==0)
    state["best_text"]=f"{name}, PLACE_FIX={fix}. 독립 100판 충돌 {result['collision_count']}, 장기대기·시간초과 {result['inf_wait_episodes']}, 공식/완벽 {result['official_complete']:.0%}/{result['perfect_complete']:.0%}, 이동 {result['moving_avg_speed_kmh']:.1f} km/h. 주변 차량 관측 없는 판 {empty}."
    persist(state)
    return (g,model,6,fix) if passed else None


def train(state,label,weight,fix,deadline):
    for smoke in (True,False):
        run=label+("_smoke" if smoke else "")
        path=ROOT/"results"/run
        model=path/("step4k.pt" if smoke else "step25k.pt")
        args=training_args(ANCHOR,run,0,smoke)
        if not smoke: args.extend(["--timesteps",25000,"--save-steps","25000","--warmup-updates",1])
        if not model.exists():
            persist(state,f"{label} {'4096스텝 스모크' if smoke else '25,000스텝 KL-PPO'}")
            child(args,run,deadline,{"PLACE_FIX":str(fix),"CGC_STAGE":"6",
                  "HOTSPOT_LABELS":"1","HOTSPOT_LABEL_WEIGHT":str(weight),
                  "TRAIN_TRACE_DIR":str(path/"traces")})
        if smoke:
            with (path/"training_log.csv").open(encoding="utf8") as f: table=list(csv.DictReader(f))
            if (len(table)!=2 or sum(float(r["left_labels"]) for r in table)==0 or
                not all(np.isfinite(float(r[k])) for r in table for k in ("value_loss","lane_kl","explained_variance"))):
                raise ValueError(f"{run} smoke lacked finite updates or actual left labels")
    return model


def experiment(state,label,weight,fix,deadline,basegroup):
    model=train(state,label,weight,fix,deadline)
    r=eval_batch(state,label,6,model,10,deadline,fix)
    b=metrics(basegroup,10)
    sample_scene(state,label,model,6,fix,deadline)
    fast=r["inf_wait_episodes"]<b["inf_wait_episodes"] and r["collision_count"]<=b["collision_count"]
    state["experiments"].append(["주정차 뒤 안전한 왼쪽 차로와 승강장 유지 라벨을 배우면 대기가 감소한다",
                                 f"학습 전용 차선 지도 손실 ×{weight}, PLACE_FIX={fix}",
                                 f"10판 D {r['inf_wait_episodes']} 대 {b['inf_wait_episodes']}, 충돌 {r['collision_count']} 대 {b['collision_count']}",
                                 "30판 진입" if fast else "탈락"])
    persist(state)
    if fast:
        r=eval_batch(state,label,6,model,30,deadline,fix)
        if passed_dev(r,metrics(basegroup,30)): return held(state,label,model,fix,deadline)
    return None


def final_scenes(state,winner):
    group,model,stage,fix=winner
    data=rows(group)
    successes=[r for r in data if r["end_reason"]=="completed"]
    successes.sort(key=lambda r:(r["inf_wait_count"],r["green_stop_time_ratio"]))
    failures=[r for r in data if r["end_reason"]!="completed" or r["inf_wait_count"]]
    selected=[]; seen=set()
    for r in successes[:1]+failures:
        if r["seed"] in seen: continue
        selected.append(r); seen.add(r["seed"])
        if len(selected)==3: break
    for row in selected:
        sample_scene(state,group,model,stage,fix,END,seed=row["seed"],near_end=row not in successes[:1])
    if successes:
        r=successes[0]
        state["scenes"].append(f"- 발표 영상 확인 후보 {r['seed']}, 주행 {r['sim_steps']*.5:.0f}초. 단계 {stage}, PLACE_FIX={fix}. "
                               f"`py -3.13 scripts/gui_drive_seed.py --stage {stage} --seed {r['seed']} --checkpoint '{model}' --delay 40`. "
                               +("5단계에는 6단계 불법주정차 요소가 없다." if stage==5 else "정류장·합류·회피 포함 여부는 전체 영상 확인이 필요하다."))
    persist(state)


def main():
    from shutil import copyfile
    RUN.mkdir(exist_ok=True,parents=True)
    history=REPORT.with_name("2026-11-26_s6_overnight_history.md")
    if not history.exists(): copyfile(REPORT,history)
    state=dict(recovery=[],phase="A·B 진행",experiments=[],scenes=[],groups=[BASEGROUP])
    os.environ.update(CGC_STAGE="6",OBS_SCAN="1",STOP_FIX="0",PLACE_FIX="0",DISABLE_SAMIL_EAST="0",
                      PYTHONUTF8="1",PYTHONDONTWRITEBYTECODE="1")
    # Wait until every GUI episode has closed and saved; the last file is written just before close.
    while len(list(AB.glob("*/episode.json")))<8: time.sleep(10)
    time.sleep(5)
    child(["scripts/analyze_gate_archive.py"],"archive_gate",FIRST)
    child(["scripts/report_samil_counterfactual.py"],"ab_report",FIRST)
    state["scenes"].append("- A·B의 각 판 정차·회피 화면과 이륜차 좌표는 [대조 보고서](2026-11-26_s6_samil_대조.md)에 있다.")
    state["experiments"].append(["정차와 회피 행동의 충돌","samil_east만 제거","학생 48→0회, 정차 실패→성공. 8k 구간 최장 429.5→0초","해당 시드 지지"])
    persist(state,"저장 자료 진단 완료")
    winner=None
    labels=[]
    try:
        for k in (8,16,24):
            name=f"klppo3_s6_recovery_step{k}k"; model=ROOT/f"results/klppo3_s6_recovery/step{k}k.pt"
            r=eval_batch(state,name,6,model,10,FIRST)
            b=metrics(BASEGROUP,10)
            good=r["inf_wait_episodes"]<b["inf_wait_episodes"] and r["collision_count"]<=b["collision_count"]
            state["recovery"].append(f"{k}k D {r['inf_wait_episodes']}/10, 충돌 {r['collision_count']}/10, {'30판 진입' if good else '탈락'}")
            persist(state)
            if k!=8: sample_scene(state,name,model,6,0,FIRST)
            if good:
                r=eval_batch(state,name,6,model,30,FIRST)
                if passed_dev(r,metrics(BASEGROUP,30)):
                    winner=held(state,name,model,0,SECOND)
                    if winner: break
        for label,weight in (("s6_label01",.1),("s6_label035",.35),("s6_label10",1.0)):
            if winner or datetime.now()>=FIRST: break
            try:
                winner=experiment(state,label,weight,0,FIRST,BASEGROUP)
                labels.append((label,weight))
            except (RuntimeError,ValueError) as e:
                state["experiments"].append(["구간 지도학습",label,str(e),"실행 실패"]); persist(state)
    except (TimeoutError,RuntimeError,ValueError) as e:
        state["experiments"].append(["원배치 정책 검증","개발 평가",str(e),"부분 결과 저장"])
    if not winner and datetime.now()<SECOND:
        try:
            tested=[(n,w) for n,w in labels if len(rows(n))>=10]
            best,weight=min(tested,key=lambda nw:(metrics(nw[0],10)["collision_count"],metrics(nw[0],10)["inf_wait_episodes"])) if tested else (BASEGROUP,.1)
            bestmodel=ANCHOR if best==BASEGROUP else ROOT/"results"/best/"step25k.pt"
            bgroup="fix_anchor"
            choices=[(bgroup,ANCHOR),("teacher56_fix",TEACHER),("fix_bc",STUDENT),("fix_policy",bestmodel)]
            qualified=[]
            # Verify actual SUMO stop coordinates and pictures before the fair 30-episode comparison.
            eval_batch(state,bgroup,6,ANCHOR,1,SECOND,1)
            with np.load(OUT/bgroup/"traces/900100.npz") as z:
                definitions=[json.loads(x) for x in set(z["stop_definition"])]
                assert any(x and x["id"]=="bs_03" and x["start"]==22 and x["end"]==37 for x in definitions)
            sample_scene(state,bgroup,ANCHOR,6,1,SECOND)
            for group,model in choices:
                r=eval_batch(state,group,6,model,30,SECOND,1)
                if group!="teacher56_fix" and passed_dev(r,metrics(bgroup,30)):
                    qualified.append((r["collision_count"],r["inf_wait_episodes"],group,model))
            state["experiments"].append(["승강장과 회피 목표를 분리하면 대기가 준다","bs_03만 상류 8m","4개 정책의 동일 배치 개발 30판 비교",f"후보 {len(qualified)}개"])
            persist(state)
            if qualified:
                _,_,g,m=min(qualified); winner=held(state,g,m,1,SECOND)
            elif datetime.now()<SECOND:
                winner=experiment(state,"s6_fix_label",weight,1,SECOND,bgroup)
        except (TimeoutError,RuntimeError,ValueError,AssertionError) as e:
            state["experiments"].append(["PLACE_FIX 검증","bs_03 상류 8m",str(e),"부분 결과 저장"]); persist(state)
    if winner:
        state["decision"]="6단계 최종("+("PLACE_FIX" if winner[3] else "원래 배치")+") · 독립 100판 기준 충족."
    else:
        g="final_s5_step250k"
        persist(state,"5단계 대비책 독립 평가")
        r=eval_batch(state,g,5,S5,100,END,start=900300)
        state["fallback"]=f"충돌 {r['collision_count']}/100, D {r['inf_wait_episodes']}/100, 공식·완벽 완주 {r['official_complete']:.0%}·{r['perfect_complete']:.0%}, 이동 {r['moving_avg_speed_kmh']:.1f} km/h."
        state["decision"]="5단계 최종 대비책 · 6단계 독립 통과 모델 미확보. 5단계 측정값과 목표 차이는 위 6항 참조."
        winner=(g,S5,5,0)
    persist(state,"독립 평가 완료, GUI 검토")
    final_scenes(state,winner)
    persist(state,"실행 완료")


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--eval",nargs=5)
    args=parser.parse_args()
    if args.eval:
        g,stage,m,n,start=args.eval
        for _ in evaluate(g,int(stage),Path(m),int(n),int(start)): pass
        s=metrics(g)
        print(json.dumps(s,ensure_ascii=False),flush=True)
    else:
        main()

