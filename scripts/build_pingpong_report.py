"""results/pingpong_s6/diag.json → docs/reports/2026-11-25_s6_핑퐁_원인.md"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAG = os.path.join(ROOT, "results", "pingpong_s6", "diag.json")
REPORT = os.path.join(ROOT, "..", "청계로드", "docs", "reports", "2026-11-25_s6_핑퐁_원인.md")
if not os.path.isdir(os.path.dirname(REPORT)):
    REPORT = os.path.join(r"C:\Users\USER\Desktop\청계로드", "docs", "reports", "2026-11-25_s6_핑퐁_원인.md")


def _ping_segment(rows, stop_id="bs_03", win=30):
    """핑퐁 구간: stop_id 근처에서 lane_applied 교번."""
    seg = [r for r in rows if r.get("stop_id") == stop_id or (
        r.get("edge") == "e908646367_3" and r.get("pos") and 5 <= r["pos"] <= 80)]
    if not seg:
        seg = rows
    return seg[-win:]


def main():
    with open(DIAG, encoding="utf-8") as f:
        p = json.load(f)
    scan = p["scan"]
    policies = sorted({r["policy"] for r in scan})
    # stop × policy counts
    table = defaultdict(lambda: Counter())
    ep_with_ping = Counter()
    for r in scan:
        pol = r["policy"]
        ep_with_ping[pol] += 1 if r["pings"] else 0
        for ev in r["pings"]:
            sid = ev.get("stop_id") or "unknown"
            table[sid][pol] += 1

    lines = [
        "# 6단계 정류장 핑퐁 원인 (2026-11-25)",
        "",
        "개발 **30판** `900100–900129`, 핑퐁 = 적용된 차선변경 후 **5초(10스텝) 이내** 반대 변경. "
        "학습 없음, `scripts/pingpong_diagnose_s6.py` 산출.",
        "",
        "## 1. 정류장별 핑퐁 집계",
        "",
        "| 정류장 | " + " | ".join(policies) + " | 합계 |",
        "|---|" + "|".join(["---:"] * (len(policies) + 1)) + "|",
    ]
    all_stops = sorted(table.keys(), key=lambda s: -sum(table[s].values()))
    for sid in all_stops:
        cells = [str(table[sid].get(pol, 0)) for p in policies]
        tot = sum(table[sid].values())
        lines.append(f"| {sid} | " + " | ".join(cells) + f" | **{tot}** |")
    lines.extend([
        "",
        "**핑퐁 있는 판 수** (30판 중): " + ", ".join(
            f"{pol} {ep_with_ping[pol]}/30" for pol in policies),
        "",
        f"**대표 시드**: `{p['trace_seed']}` (학생, bs_03 핑퐁 많은 판)",
        "",
        "## 2. 차선 고정 실험 (같은 시드)",
        "",
        "| 조건 | 종료 | bs_03 10초 정차 성공 | stops_done |",
        "|---|---|---:|---:|",
    ])
    tr = p["trace"]
    lk = p["lane_lock_policy"]
    lt = p.get("lane_lock_teacher_accel", {})
    lines.append(
        f"| 학생 shadow (기본) | {tr['end_reason']} | {'예' if tr.get('dwell_success') else '아니오'} | {tr.get('stops_done')} |"
    )
    lines.append(
        f"| 학생 shadow + **차선 0 고정** | {lk['end_reason']} | {'예' if lk.get('dwell_success') else '아니오'} | {lk.get('stops_done')} |"
    )
    if lt:
        lines.append(
            f"| teacher56 가속만 + **차선 0 고정** | {lt.get('end_reason','?')} | "
            f"{'예' if lt.get('dwell_success') else '아니오'} | {lt.get('stops_done','?')} |"
        )
    lines.extend([
        "",
        "**해석**",
        "- 차선 고정 시 10초 정차에 성공 → 정류장 geometry·정차 판정은 가능, **차선 목표(관측 `lane_diff`·정책/선생님 정류장 규칙)가 뒤집히며 핑퐁**.",
        "- 차선 고정해도 정차 실패 → **정류장 정의·`in_stop_area`·승강장 옆 차로** 쪽 환경 버그 가능.",
        "",
        "## 3. 가설 판정",
        "",
        "| 가설 | 판정 | 근거 |",
        "|---|:---:|---|",
    ])
    # heuristic from trace rows at bs_03
    rows = p["trace"]["rows"]
    diffs = []
    flips = 0
    prev = None
    for r in rows:
        if r.get("stop_id") != "bs_03":
            continue
        d = r["bus_stop_obs"]["lane_diff"]
        diffs.append(d)
        if prev is not None and prev * d < -0.01:
            flips += 1
        prev = d
    lines.append(
        f"| (a) `lane_diff` 부호 뒤집힘 | {'**유력**' if flips >= 2 else '약함'} | "
        f"bs_03 구간 부호 변화 {flips}회, 샘플 {diffs[:8]}… |"
    )
    lines.append(
        "| (b) edge마다 차로 수·기준 바뀜 | 검토 | `_stop_obs`가 `want`는 정류장 edge, `have`는 **현재 edge**의 `_car_lanes` 기준. "
        "승강장 옆·정류장 차로가 다르면 상대 인덱스 불일치 가능 |"
    )
    lines.append(
        "| (c) 정차금지·정류장 규칙 충돌 | 부분 | `e908646367_3` 0–10m 안전지대 직후 bs_03. "
        "no_stop 감점과 정차 규칙이 같은 속도 구간 겹칠 수 있음 (선생님 `_stop_rule` 별도) |"
    )
    lines.append(
        "| (d) 다음 정류장 전환 | | trace에서 `stop_changed`·`stop_i` 열 참고 (아래 발췌) |"
    )
    lines.append(
        "| (e) 기타 | | 측면 합류·shield 없는 shadow에서 학생/KL의 연속 LC 명령 |"
    )
    lines.extend([
        "",
        "## 4. 수정안 (코드 미반영)",
        "",
        "### 환경·관측",
        "- `_stop_obs` [1]: **want/have를 동일 edge 기준**으로 계산하거나, 정류장 edge와 현재 edge가 다를 때 "
        "lane_diff를 **0 또는 ‘아직 정류장 edge 아님’** 으로 클리핑.",
        "- bs_03: SUMO `busstop` lane vs **승강장 옆 주행 차로**가 1칸 어긋나면 `in_stop_area`가 정차를 인정 못 할 수 있음 — "
        "`_in_stop_area`와 stop lane 정의 재확인.",
        "- 영향: **6단계 전 모델·선생·KL 재평가** (관측 3칸 변경). 5단계는 bus_stop 관측 동일 구조면 회귀 확인.",
        "",
        "### 선생님 규칙 (`bcppo.TeacherEnv._stop_rule`)",
        "- `diff = _stop_obs()[1]`로 `want = sign(diff)` — 관측이 뒤집히면 **선생님도 핑퐁**. "
        "관측 수정 후 선생 시범/DAgger **재수집 여부** 검토.",
        "- 임시: 정류장 edge 안에서는 `want=0` 히스테리시스(한 번 정류장 차로 도달 시 유지).",
        "",
        "## 5. GUI 재현",
        "",
        f"- **시드**: `{p['trace_seed']}` (학생: `s6_student_obsfix`)",
    ])
    gh = p.get("gui_hint", {})
    if gh.get("sim_t"):
        lines.append(f"- **핑퐁 시작 근처 시뮬 시각**: 약 **{gh['sim_t']:.0f}s** (step {gh['step']})")
    lines.append("- **엣지**: `e908646367_3`, **정류장**: `bs_03`, 안전지대 **0–10m** 직후")
    lines.append("- **장면**: 승강장 **옆 차로**에서 `lane_diff`·모델 차선 출력이 ±로 바뀌며 0.5s마다 LC")
    lines.append("")
    lines.append("```powershell")
    lines.append("$env:CGC_STAGE=6; $env:OBS_SCAN=1")
    lines.append("Set-Location C:\\Users\\USER\\Desktop\\청계로드-kl-ppo")
    lines.append(f"py -3.13 scripts/gui_drive_seed.py --seed {p['trace_seed']} --model student --delay 80 --pause")
    lines.append("```")
    lines.extend([
        "",
        "## 6. bs_03 근처 스텝 발췌 (핑퐁 구간)",
        "",
        "| step | t | lane | lane_idx | pos | dist | lane_diff | dwell | in_area | lc_forbid | no_stop |",
        "|---:|---:|---|---:|---:|---:|---:|---:|:---:|---|---|",
    ])
    for r in _ping_segment(rows):
        if r.get("lane_applied") or abs(r["bus_stop_obs"]["lane_diff"]) > 0.05:
            b = r["bus_stop_obs"]
            lines.append(
                f"| {r['step']} | {r['sim_t']:.1f} | {r['lane_cmd']}/{r['lane_applied']} | "
                f"{r.get('lane_idx')} | {r.get('pos', 0):.1f} | {b['dist']:.3f} | {b['lane_diff']:.3f} | "
                f"{b['dwell']:.2f} | {r['in_stop_area']} | {r['lc_forbidden']} | {r['no_stop_zone_slow']} |"
            )
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote", REPORT)


if __name__ == "__main__":
    main()