"""기존 overnight_eval_s56 JSON만으로 1차 6단계 공식 지표 표 (C·D 등 일부 없음)."""
from __future__ import annotations

import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "results", "overnight_eval_s56")
OUT = os.path.join(
    r"C:\Users\USER\Desktop\청계로드\docs\reports",
    "2026-11-25_s6_1차_공식지표_분석.md",
)

BASELINE_OFF = 0.567  # s6_student_obsfix 30판


def load_stats(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["stats"], d.get("rows", [])


def row_line(name, step, stats, note=""):
    n = stats["episodes"]
    off = stats["official_complete"]
    perf = stats["perfect_complete"]
    coll = stats["collision"]
    wrong = stats.get("reasons", {}).get("lane_error", 0) / max(n, 1)
    excl = ""
    if off < BASELINE_OFF - 0.10:
        excl = f"후보 제외(공식 {off:.1%} < 기준 {BASELINE_OFF:.1%}-10%p)"
    return (
        f"| {name} | {step} | {off*n:.0f}/{n} ({off:.1%}) | {perf*n:.0f}/{n} ({perf:.1%}) | "
        f"{coll*n:.0f}/{n} ({coll:.1%}) | {wrong*n:.0f}/{n} | 없음 | 없음 | 없음 | "
        f"{stats.get('avg_speed', 0)*3.6:.1f} km/h | {stats.get('yield_steps', '-')} | "
        f"{stats.get('intervention', 0):.1%} | {excl or note} |"
    )


def collision_heuristic(rows):
    hits = []
    for r in rows:
        if r.get("end_reason") != "collision":
            continue
        cls = "e_other"
        if r.get("bs02_lc_fail"):
            cls = "a_lane_change_side"
        elif str(r.get("place", "")).startswith("hs_"):
            cls = "c_illegal_park"
        elif r.get("nearest_stop_id") == "bs_02":
            cls = "a_lane_change_side"
        hits.append((r["seed"], cls, r.get("place"), r.get("ego_edge")))
    return hits


def main():
    student_path = os.path.join(EVAL, "s6_student_obsfix.json")
    st, st_rows = load_stats(student_path)
    lines = [
        "# 6단계 1차 KL-PPO 공식 지표 분석 (개발 30판, JSON 재분석)",
        "",
        f"데이터: `{EVAL}`. SUMO 재실행 없음. **C 비정상 정지·D 무한 대기·F 움직임 평균·G 안정성·충돌 상세 분류**는 "
        "판별 기록이 없어 `없음`이다. `utils/official_metrics.py` + `drive_parallel.py`로 이후 평가에서 계산한다.",
        "",
        f"**기준선** s6_student_obsfix: 공식 {st['official_complete']:.1%} "
        f"({st['official_complete']*30:.0f}/30), 완벽 {st['perfect_complete']:.1%}, "
        f"충돌 {st['collision']:.1%} ({st['collision']*30:.0f}/30).",
        "",
        "선택 순서(충돌↓ → 무한대기+timeout+wrong_lane → 비정상정지 → 움직임속도 → 공식완주, "
        "공식이 기준보다 10%p 낮으면 후보 제외):",
        "",
        "| run | step | 공식 완주 | 완벽 완주 | A 충돌 | E wrong_lane | C 비정상정지 | D 무한대기 | G teleport | F 평균(km/h) | H 양보 | H 개입률 | 비고 |",
        "|---|---:|---|---|---|---|---|---|---|---|---|---|---|",
        row_line("s6_student_obsfix", 0, st, "기준"),
    ]
    ranked = []
    for prefix in ("klppo1_s6_s0", "klppo1_s6_s1"):
        for f in sorted(glob.glob(os.path.join(EVAL, prefix + "_step*.json"))):
            step = int(f.split("step")[1].split("k")[0]) * 1000
            stats, _ = load_stats(f)
            ranked.append((stats["collision"], stats["official_complete"], step, prefix, stats, f))
    ranked.sort(key=lambda x: (x[0], -x[1]))
    for _, _, step, prefix, stats, _ in ranked:
        note = ""
        if prefix == "klppo1_s6_s1" and step == 200000:
            note = "τ·2차 출발점 기본 후보"
        lines.append(row_line(prefix, step, stats, note))

    best = min(
        (x for x in ranked if x[1] >= BASELINE_OFF - 0.10),
        key=lambda x: (x[0], -x[2]),
        default=None,
    )
    lines.extend(["", "## 충돌 판 (휴리스틱, 재분석 필요)", ""])
    all_coll = []
    for prefix in ("klppo1_s6_s0", "klppo1_s6_s1"):
        for f in glob.glob(os.path.join(EVAL, prefix + "_step*.json")):
            _, rows = load_stats(f)
            all_coll.extend([(os.path.basename(f), *h) for h in collision_heuristic(rows)])
    _, rows_st = load_stats(student_path)
    all_coll.extend([("s6_student_obsfix.json", *h) for h in collision_heuristic(rows_st)])
    lc = sum(1 for c in all_coll if c[2] == "a_lane_change_side")
    tot = len(all_coll)
    lines.append(f"JSON에 저장된 충돌 판 {tot}건 중 bs_02·bs02_lc_fail·hs_* 휴리스틱으로 차선변경·주정차 추정 {lc}건 "
                 f"({100*lc/max(tot,1):.0f}%). **정확 분류는 A~H 재평가 후 갱신.**")
    lines.append("")
    lines.append("| json | 시드 | 추정 유형 | place | edge |")
    lines.append("|---|---:|---|---|---|")
    for j, seed, cls, place, edge in sorted(all_coll)[:40]:
        lines.append(f"| {j} | {seed} | {cls} | {place} | {edge} |")
    if tot > 40:
        lines.append(f"| … | … | ({tot-40}건 생략) | | |")

    lines.extend([
        "",
        "## 결론 (1차, 개발 30판)",
        "",
    ])
    if best:
        _, off, step, prefix, stats, _ = best
        lines.append(
            f"- 통과 목표(충돌 &lt;5%)는 **모든 체크포인트에서 미달**. 최소 충돌(공식완주 10%p 규칙 통과)은 "
            f"**{prefix} {step//1000}k** — 충돌 {stats['collision']:.1%} ({stats['collision']*30:.0f}/30), "
            f"공식 {stats['official_complete']:.1%}, 완벽 {stats['perfect_complete']:.1%}."
        )
    else:
        lines.append("- 공식 완주 10%p 규칙을 통과하는 체크포인트가 없음.")
    lines.append(
        f"- 차선변경 추정 비율이 높으면 **2차 차선 학습** 근거. 수요일까지 6단계 2차(`klppo2_s6`) 시도, "
        f"출발점 기본값 `klppo1_s6_s1 step200k`."
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()