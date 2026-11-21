"""env-v2 공통 평가 진입점.

사용법은 `py -3.13 scripts/evaluate_all.py --help` 또는
`rules/team/experiment_protocol.md` 2-6을 따른다. 실제 구현은 common_eval.py에 있다.
아래의 구버전 함수들은 이전 결과를 읽는 코드와의 import 호환용으로 남긴다.
"""
if __name__ == "__main__":
    # env-v2 공통 진입점. 먼저 실행해 --help/병합은 SUMO 의존성 없이도 쓸 수 있다.
    from common_eval import main
    main()
    raise SystemExit

import argparse
import math
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.road_config import ROAD, EGO, TRAFFIC  # noqa: E402
from env.mdp_config import SIMULATION, OBSERVATION, ACTION  # noqa: E402
from env import road_builder  # noqa: E402
from env.sumo_env import SumoHighwayEnv  # noqa: E402
import traci  # noqa: E402
import traci.constants as tc  # noqa: E402

from train import REWARD  # noqa: E402
from test import load_agent  # noqa: E402

from checkpoints import CheckpointTracker, ALL_CODES  # noqa: E402
from rule_based_expert import RuleBasedExpert  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 비교할 모델 목록 — name -> (kind, path).  kind: "ppo"/"bc" 체크포인트 또는 "rule"
MODEL_REGISTRY = {
    "minki_ppo": ("ppo", os.path.join(ROOT, "results", "run_20261108_114607", "best_model.pt")),
    "rule_based": ("rule", None),
}


def build_env():
    sumocfg = road_builder.build(ROAD, EGO, TRAFFIC, os.path.join(ROOT, "env", "sumo"))
    return SumoHighwayEnv(
        cfg_path=sumocfg, road=ROAD, ego=EGO,
        mdp_sim=SIMULATION, mdp_obs=OBSERVATION,
        action=ACTION, reward=REWARD, traffic=TRAFFIC, gui=False)


# 계획서 §6 기준값
PED_RADIUS = 40.0    # 보행자 관측 반경 m (context subscription)
NEAR_DIST = 3.0      # 근접 통과: 보행자와의 거리 < 3 m
NEAR_SPEED = 1.0     #            이고 차량 속도 > 1 m/s
HARD_BRAKE = 3.0     # 급제동: 감속 > 3 m/s²


class SafetyMetrics:
    """계획서 §6 보행자 상호작용·승차감 지표 (평가 전용, 환경·보상 영향 없음).

    보행자는 TraCI context subscription(ego 주변 40 m의 person)으로 받아서
    스텝당 TraCI 호출을 2번(목록, ego 위치)으로 유지한다.
    """

    def __init__(self, env):
        self.env = env
        self.dt = float(env.step_length)
        self.near_events = 0            # 근접 통과 횟수 (보행자별 진입 1회)
        self.hard_brakes = 0            # 급제동 횟수 (연속 스텝은 1회)
        self.jerk_sq, self.jerk_n = 0.0, 0
        self.ep_min_dists = []          # 에피소드별 보행자 최소거리 m
        self.total_km = 0.0
        self._reset_ep()

    def _reset_ep(self):
        self._near_ids = set()
        self._prev_v = self._prev_a = None
        self._braking = False
        self._min_dist = float("inf")
        self._dist_m = 0.0

    def start_episode(self):
        self._reset_ep()
        try:
            traci.vehicle.subscribeContext(
                self.env.ego_id, tc.CMD_GET_PERSON_VARIABLE, PED_RADIUS,
                [tc.VAR_POSITION])
        except traci.TraCIException:
            pass    # 반경 안에 보행자가 아직 없어도 구독은 유지된다

    def step(self, info):
        ego = self.env.ego_id
        v = info.get("speed")
        if v is None:
            return
        # 가속도(속도 차분) → 급제동·jerk. 정차 중(dwelling)은 빼지 않는다:
        # 어차피 속도 0이라 가속도 0으로 들어간다.
        if self._prev_v is not None:
            a = (v - self._prev_v) / self.dt
            if self._prev_a is not None:
                j = (a - self._prev_a) / self.dt
                self.jerk_sq += j * j
                self.jerk_n += 1
            braking = a <= -HARD_BRAKE
            if braking and not self._braking:
                self.hard_brakes += 1
            self._braking = braking
            self._prev_a = a
        self._prev_v = v
        if ego not in traci.vehicle.getIDList():
            return
        try:
            self._dist_m = max(self._dist_m, traci.vehicle.getDistance(ego))
            peds = traci.vehicle.getContextSubscriptionResults(ego) or {}
            ex, ey = traci.vehicle.getPosition(ego)
        except traci.TraCIException:
            return
        near_now = set()
        for pid, vals in peds.items():
            px, py = vals.get(tc.VAR_POSITION, (None, None))
            if px is None:
                continue
            d = math.hypot(px - ex, py - ey)
            self._min_dist = min(self._min_dist, d)
            if d < NEAR_DIST:
                near_now.add(pid)
        if v > NEAR_SPEED:   # 빠른 채로 보행자 옆을 지나갈 때만 "근접 통과"
            self.near_events += len(near_now - self._near_ids)
        self._near_ids = near_now

    def end_episode(self):
        if self._min_dist < float("inf"):
            self.ep_min_dists.append(self._min_dist)
        self.total_km += self._dist_m / 1000.0

    def summary(self, n_episodes):
        km = max(self.total_km, 1e-9)
        return {
            "near_pass_per_ep": self.near_events / n_episodes,
            "ped_min_dist_mean": (sum(self.ep_min_dists) / len(self.ep_min_dists))
                                 if self.ep_min_dists else None,
            "ped_min_dist_min": min(self.ep_min_dists) if self.ep_min_dists else None,
            "hard_brakes_per_km": self.hard_brakes / km,
            "jerk_rms": math.sqrt(self.jerk_sq / self.jerk_n) if self.jerk_n else 0.0,
        }


def run_episodes(policy, env, n_episodes):
    n_collide = n_arrive = n_timeout = n_filter_steps = n_steps = 0
    n_red = n_yield = 0                   # 신호 위반·양보 위반 (스텝 이벤트 합)
    n_stops_total = n_stops_served = 0    # 정류장 정차율 (실제 정차 횟수 / 노선 정류장 수)
    arrive_times = []                     # 완주 에피소드의 주행 시간 (s)
    speeds, lane_changes = [], []
    pass_counts = {c: 0 for c in ALL_CODES}
    metrics = SafetyMetrics(env)
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=ep)
        metrics.start_episode()
        tracker = CheckpointTracker()
        terminated = truncated = False
        ep_lc = ep_steps = ep_dwells = 0
        was_dwelling = False
        while not (terminated or truncated):
            action = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            n_steps += 1
            ep_steps += 1
            dwelling = bool(info.get("dwelling"))
            ep_dwells += int(dwelling and not was_dwelling)   # 정차 시작 1회 = 정류장 1곳
            was_dwelling = dwelling
            if info.get("speed") is not None:
                speeds.append(float(info["speed"]))
            if info.get("lane_change", 0) != 0 and info.get("lane_change_applied"):
                ep_lc += 1
            n_filter_steps += int(bool(info.get("filter_active")))
            n_red += int(bool(info.get("red_violation")))
            n_yield += int(bool(info.get("yield_violation")))
            metrics.step(info)
            if env.ego_id in traci.vehicle.getIDList():
                dist = traci.vehicle.getDistance(env.ego_id)
                tracker.update(dist, info.get("speed") or 0.0)
        metrics.end_episode()
        lane_changes.append(ep_lc)
        # 정류장 정차율: 노선 정류장 수 대비 실제 정차 횟수
        # (건너뛰었거나 도달 못 한 정류장은 정차로 안 센다)
        line_stops = getattr(env, "_line_stops", {}).get(info.get("route"))
        n_stops_total += len(line_stops) if line_stops else 5   # 구버전 fallback: 동행 5곳
        n_stops_served += ep_dwells
        if info.get("collided"):
            n_collide += 1
        elif info.get("arrived"):
            n_arrive += 1
            arrive_times.append(ep_steps * metrics.dt)
        else:
            n_timeout += 1
        for c in ALL_CODES:
            pass_counts[c] += tracker.passed.get(c, False)

    out = {
        "n_episodes": n_episodes,
        "success_rate": n_arrive / n_episodes,
        "collision_rate": n_collide / n_episodes,
        "timeout_rate": n_timeout / n_episodes,
        "mean_speed": sum(speeds) / len(speeds) if speeds else 0.0,
        "filter_rate": n_filter_steps / max(n_steps, 1),
        "lane_changes": sum(lane_changes) / n_episodes,
        "red_per_ep": n_red / n_episodes,
        "yield_per_ep": n_yield / n_episodes,
        "stop_served_rate": n_stops_served / max(n_stops_total, 1),
        "arrive_time_mean": (sum(arrive_times) / len(arrive_times)) if arrive_times else None,
        "checkpoint_pass_rate": {c: pass_counts[c] / n_episodes for c in ALL_CODES},
    }
    out.update(metrics.summary(n_episodes))
    return out


def make_policy(kind, path, env):
    if kind == "rule":
        return RuleBasedExpert(env)
    return load_agent(path, env.observation_space.shape[0], env.action_space.shape[0], "auto")


def to_markdown(results: dict) -> str:
    lines = [
        "# 청계천로 편도 — 모델 비교 평가",
        "",
        f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| 모델 | 에피소드 | 완주율 | 충돌률 | 시간초과율 | 평균속도(m/s) | 수동개입률 | 차선변경/에피소드 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['n_episodes']} | {r['success_rate']:.0%} | "
            f"{r['collision_rate']:.0%} | {r['timeout_rate']:.0%} | "
            f"{r['mean_speed']:.2f} | {r['filter_rate']:.1%} | {r['lane_changes']:.1f} |")
    def _f(v, fmt):
        return format(v, fmt) if v is not None else "—"
    lines += [
        "", "## 보행자 상호작용·승차감·정류장 (계획서 §6)", "",
        "근접 통과 = 보행자 3 m 안을 1 m/s 넘는 속도로 지나간 횟수. "
        "급제동 = 감속 3 m/s² 초과. 최소거리 = 에피소드별 보행자 최소거리의 평균(최솟값).",
        "",
        "| 모델 | 근접통과/에피 | 보행자 최소거리(m) | 급제동/km | jerk RMS(m/s³) "
        "| 신호위반/에피 | 양보위반/에피 | 정류장 정차율 | 완주시간 평균(s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        md_mean, md_min = r.get("ped_min_dist_mean"), r.get("ped_min_dist_min")
        dist_txt = f"{md_mean:.1f} ({md_min:.1f})" if md_mean is not None else "—"
        lines.append(
            f"| {name} | {r['near_pass_per_ep']:.2f} | {dist_txt} | "
            f"{r['hard_brakes_per_km']:.2f} | {r['jerk_rms']:.2f} | "
            f"{r['red_per_ep']:.2f} | {r['yield_per_ep']:.2f} | "
            f"{r['stop_served_rate']:.0%} | {_f(r.get('arrive_time_mean'), '.0f')} |")
    lines += ["", "## 지점별 통과율", "", "| 모델 | " + " | ".join(ALL_CODES) + " |",
             "|---|" + "---|" * len(ALL_CODES)]
    for name, r in results.items():
        cp = r["checkpoint_pass_rate"]
        lines.append(f"| {name} | " + " | ".join(f"{cp[c]:.0%}" for c in ALL_CODES) + " |")
    return "\n".join(lines) + "\n"
