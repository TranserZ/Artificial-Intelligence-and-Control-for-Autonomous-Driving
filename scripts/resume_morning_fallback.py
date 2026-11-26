import os,json
from datetime import datetime,timedelta
from scripts import morning_ladder as m
os.environ.update(OBS_SCAN="0",CGC_STAGE="5",PLACE_FIX="0",STOP_FIX="0",DISABLE_SAMIL_EAST="0",PYTHONUTF8="1")
s=json.loads(m.STATE.read_text(encoding="utf8"))
s["best_text"]="원래 배치 s6_label035 개발 30판 충돌 2, 장기대기 19, 시간초과 0, 공식·완벽 완주 22, 이동 18.9 km/h. 후보 기준 미달로 6단계 독립 시험 미실시."
s["experiments"].append(["승강장과 회피 구간 분리","bs_03 상류 8m","출발 모델 장기대기·시간초과 24/30, 선생님 29/30. 03:25 보고서 교체 오류로 학생과 새 정책 비교 중단","배치 개선 미입증"])
s["phase"]="03:25 보고서 저장 오류로 중단. 마감 후 5단계 독립 100판 복구 실행."
m.END=datetime.now()+timedelta(hours=8)
original=json.loads((m.ROOT/"results/overnight_eval_s56/klppo1_s5_s0_step250k.json").read_text(encoding="utf8"))["rows"]
for expected in original[:5]:
    seed=expected["seed"]
    path=m.OUT/"repro_s5_scan0"/f"{seed}.json"
    actual=json.loads(path.read_text(encoding="utf8"))
    if actual["end_reason"] != expected["end_reason"] or actual["evaluation_environment"]["OBS_SCAN"] != "0":
        raise ValueError(f"5단계 개발 시드 재현 불일치: {seed}")
m.persist(s)
g="final_s5_step250k_scan0"
r=m.eval_batch(s,g,5,m.S5,100,m.END,start=900300)
assert {row["seed"] for row in m.rows(g)}==set(range(900300,900400))
s["fallback"]=f"충돌 {r['collision_count']}/100, 장기대기·시간초과 {r['inf_wait_episodes']}/100, 공식·완벽 완주 {r['official_complete']:.0%}·{r['perfect_complete']:.0%}, 이동 {r['moving_avg_speed_kmh']:.1f} km/h."
s["decision"]="5단계 대비책. 6단계 개발 기준 통과 모델 없음. 5단계 적합성은 위 실측값 참조."
m.persist(s,"5단계 100판 완료, GUI 확인 중")
m.final_scenes(s,(g,m.S5,5,0))
m.persist(s,"5단계 독립 평가 및 GUI 재현 완료")
