from pathlib import Path
import csv
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/samil_counterfactual"
BASE=ROOT/"results/overnight_s6_final"


def diagnose():
    values=[]
    with np.load(BASE/"klppo3_s6_recovery_step8k/traces/900100.npz") as trace:
        a=dict(trace)
    ixs=np.flatnonzero((a["edge"]=="e908646367_3")&(a["time"]>=255.5)&(a["time"]<685.0))
    assert len(ixs)==859 and np.all(a["speed"][ixs]<0.1)
    with (OUT/"gate_900100.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        w.writerow("step time edge pos lane left_prob keep_prob right_prob command actual_delta left_gate gate_reason left_front_m left_rear_m recovery_reward_condition".split())
        for i in ixs:
            cmd=int(a["lane_cmd"][i])
            verdict=("차단" if a["shield_lane"][i] else "허용") if cmd==1 else "미검사"
            w.writerow([a["step"][i],a["time"][i],a["edge"][i],a["pos"][i],a["lane"][i],
                        *a["lane_probs"][i],cmd,a["lane_delta"][i],verdict,"미저장","미저장","미저장","미저장"])
    windows=defaultdict(list)
    for path in (BASE/"teacher56/traces").glob("*.npz"):
        with np.load(path) as trace:
            edges=trace["edge"]; positions=trace["pos"]; stamps=trace["time"]
            for edge in set(edges):
                ids=np.flatnonzero((edges==edge)&np.isfinite(positions))
                if not len(ids): continue
                for band in np.unique(positions[ids]//10):
                    select=ids[positions[ids]//10==band]
                    for run in np.split(select,np.where(np.diff(select)!=1)[0]+1):
                        end=run[-1]+1
                        if (end<len(edges) and edges[end]==edge and positions[end]>=(band+1)*10
                                and run[0]>0 and edges[run[0]-1]==edge and positions[run[0]-1]<band*10):
                            windows[(edge,int(band))].append((path.stem,float(stamps[end]-stamps[run[0]])))
    groups=defaultdict(list)
    with (OUT/"location_events.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["사건"]!="장기대기": continue
            pos=float(r["pos_m"])
            if np.isfinite(pos):
                groups[(r["모델"],r["분류"],r["edge"],int(pos//10))].append(float(r["지속초"]))
    with (OUT/"teacher_passage.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        w.writerow(["모델","구간","edge","10m구간_시작","대기사건","최장정지_s","선생님_통과판수","선생님_통과시간_중앙값_s","선생님_시드별_통과초"])
        for (group,label,edge,band),seconds in sorted(groups.items()):
            teacher=windows[(edge,band)]
            w.writerow([group,label,edge,band*10,len(seconds),max(seconds),len(teacher),
                        float(np.median([v for _,v in teacher])) if teacher else "",
                        " ".join(f"{s}={v:g}" for s,v in sorted(teacher))])


if __name__=="__main__": diagnose()

