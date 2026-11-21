"""v3(3600 step, 놓침 −20) + 1단계 학습 가중치로 Knox 세 궤적의 gamma 0.99 할인 합.

학습기를 돌리지 않는다. 가정은 표 아래에 적는다.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from train import HPARAMS, REWARD  # noqa: E402


GAMMA = float(HPARAMS["gamma"])
MAX_STEPS = 3600
DT = 0.5
ROUTE_M = 2146.0
# 경로상 정류장 대략 위치 (geometry_validation.md, m)
STOP_M = (29.5, 497.4, 943.2, 1314.5, 1716.0)
# 1단계 SUMO 데모 버스 466 s (docs/env_stage_maps.md) + 정차 5×10 s 를 여유로 포함
COMPLETE_S = 466.0 + 5 * 10.0
V_CRUISE = 8.33


def disc(t, r):
    return (GAMMA ** t) * r


def geom(c, t0, n):
    """상수 c를 t0부터 n스텝."""
    if n <= 0:
        return 0.0
    return c * (GAMMA ** t0) * (1.0 - GAMMA ** n) / (1.0 - GAMMA)


def progress_along(times_dist):
    """times_dist: (t, 누적 거리). 스텝마다 Δs/L * progress_weight."""
    pw = float(REWARD["progress_weight"])
    g = 0.0
    prev_s, prev_t = 0.0, 0
    for t, s in times_dist:
        ds = max(s - prev_s, 0.0)
        if t > prev_t:
            # 구간에 균등 지급
            per = pw * ds / ROUTE_M / (t - prev_t)
            g += geom(per, prev_t, t - prev_t)
        prev_s, prev_t = s, t
    return g


def crash():
    return disc(0, -float(REWARD["collision_penalty"]))


def idle_empty():
    # 앞이 먼 빈 도로에서 계속 정지: green_stop base+extra = 0.15, 정류장 구간 아님
    # 놓침은 도착할 때만 남으니 시간초과에서는 0 (sumo_env._update_stops)
    c = float(REWARD["green_stop_penalty"]) + float(REWARD["green_stop_extra"])
    g = geom(-c, 0, MAX_STEPS)
    g += disc(MAX_STEPS - 1, -float(REWARD["timeout_penalty"]))
    return g, c


def complete():
    t_end = int(round(COMPLETE_S / DT))
    t_end = min(t_end, MAX_STEPS - 1)
    g = 0.0
    # 진행: 정류장 사이는 순항, 정류장에서 10 s 정지
    t = 0
    s = 0.0
    pts = [(0, 0.0)]
    bonus = float(REWARD["stop_success_bonus"])
    for sm in STOP_M:
        travel = max(sm - s, 0.0) / V_CRUISE
        t += int(round(travel / DT))
        s = sm
        pts.append((t, s))
        g += disc(t, bonus)  # 정차 성공 시점 ≈ 구간 진입 직후. 정차 10 s 끝에 지급
        t += int(round(10.0 / DT))
        pts.append((t, s))
    travel = max(ROUTE_M - s, 0.0) / V_CRUISE
    t += int(round(travel / DT))
    t = min(t, t_end)
    pts.append((t, ROUTE_M))
    g += progress_along(pts)
    g += disc(t, float(REWARD["arrival_bonus"]))
    return g, t, pts


def main():
    g_crash = crash()
    g_idle, idle_c = idle_empty()
    g_succ, t_succ, pts = complete()
    order = g_crash < g_idle < g_succ
    # 움직이는 정책 충돌 확률 p, 즉시 충돌 vs 완주. 정차보다 나으려면
    # (1-p) G_succ + p G_crash > G_idle
    denom = g_succ - g_crash
    p_star = (g_succ - g_idle) / denom if denom else float("nan")
    out = {
        "gamma": GAMMA,
        "max_steps": MAX_STEPS,
        "reward_used": {
            "speed_weight": REWARD["speed_weight"],
            "green_stop_penalty": REWARD["green_stop_penalty"],
            "green_stop_extra": REWARD["green_stop_extra"],
            "progress_weight": REWARD["progress_weight"],
            "collision_penalty": REWARD["collision_penalty"],
            "arrival_bonus": REWARD["arrival_bonus"],
            "timeout_penalty": REWARD["timeout_penalty"],
            "stop_success_bonus": REWARD["stop_success_bonus"],
            "missed_stop_penalty": REWARD["missed_stop_penalty"],
        },
        "G": {
            "crash_t0": round(g_crash, 4),
            "idle_empty_road": round(g_idle, 4),
            "complete": round(g_succ, 4),
        },
        "idle_step_penalty": idle_c,
        "complete_end_step": t_succ,
        "complete_stop_times_steps": [p[0] for p in pts],
        "crash_lt_idle_lt_complete": order,
        "p_crash_below_which_moving_beats_idle": round(p_star, 4),
        "one_term_if_fail": None if order else "progress_weight",
        "gamma_T_timeout": GAMMA ** (MAX_STEPS - 1),
        "gamma_T_complete": GAMMA ** t_succ,
    }
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    dest = os.path.join(ROOT, "results", "idm_diag", "discounted_returns.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")
    print("crash < idle < complete:", order)
    if not order:
        print("바꿀 항 하나: progress_weight 만 키운다 (B 보고서가 남긴 다음 칸).")


if __name__ == "__main__":
    main()
