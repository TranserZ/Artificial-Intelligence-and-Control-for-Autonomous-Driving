"""교수님 공지 A~H 평가 지표 (판별 기록 + 집계)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


STEP_SEC = 0.5
INF_WAIT_SEC = 60.0
INF_WAIT_STEPS = int(INF_WAIT_SEC / STEP_SEC)
GREEN_STOP_SPEED = 1.5
ZERO_SPEED = 0.1


@dataclass
class EpisodeMetrics:
    step: int = 0
    sim_steps: int = 0
    distance_m: float = 0.0
    move_time_s: float = 0.0
    speed_time_integral: float = 0.0
    green_stop_steps: int = 0
    green_stop_events: int = 0
    _in_green_stop: bool = False
    green_stop_run: int = 0
    green_stop_max_steps: int = 0
    stall_steps: int = 0
    _stall_run: int = 0
    stall_max_steps: int = 0
    inf_wait_runs: list = field(default_factory=list)
    _stall_location: dict = field(default_factory=dict)
    _stall_place: str = ""
    _stall_edge: str = ""
    lane_changes: int = 0
    teleports: int = 0
    sim_errors: int = 0
    last_lane_before: Any = None
    lc_steps_before_collision: int = 99999
    _steps_since_lc: int = 99999
    collision_class: str = ""
    collision_partner: str = ""
    collision_place: str = ""

    def observe_step(self, env, info: dict, odo_before: float | None):
        self.step += 1
        self.sim_steps += 1
        alive = info.get("speed") is not None
        speed = float(info["speed"]) if alive else 0.0
        if alive and odo_before is not None:
            odo = None
            try:
                import traci
                if env.ego_id in traci.vehicle.getIDList():
                    odo = float(traci.vehicle.getDistance(env.ego_id))
                    self.distance_m += max(0.0, odo - float(odo_before))
            except Exception:
                pass
        if speed > ZERO_SPEED:
            self.move_time_s += STEP_SEC
            self.speed_time_integral += speed * STEP_SEC

        import traci
        present = env.ego_id in traci.vehicle.getIDList()
        signal = env._next_signal() if present else None
        red = signal is not None and signal[2] in ("r", "R")
        red_or_yellow = signal is not None and signal[2] in ("r", "R", "y", "Y")
        red_wait = red and 0 <= signal[1] <= 15.0
        dwelling = bool(info.get("dwelling") or info.get("stop_success"))
        nearby_signal = red_or_yellow and 0 <= signal[1] <= 15.0
        leader = traci.vehicle.getLeader(env.ego_id, 15.0) if present and speed < GREEN_STOP_SPEED else None
        nearby_leader = bool(leader and leader[0] and leader[1] <= 15.0)
        pedestrian = False
        if present and speed < GREEN_STOP_SPEED:
            offset = traci.vehicle.getDistance(env.ego_id)
            crossings = {crossing["id"] for crossing in env._route_crossings
                         if 0 <= crossing["s"] - offset <= 15.0}
            if crossings:
                cache = getattr(env, "_ped_cache", {})
                if cache.get("t") == traci.simulation.getTime():
                    pedestrian = not crossings.isdisjoint(cache["roads"].values())
                else:
                    pedestrian = any(traci.person.getRoadID(person) in crossings
                                     for person in traci.person.getIDList())
        if present and speed < GREEN_STOP_SPEED and not any((nearby_signal, nearby_leader, pedestrian, dwelling)):
            self.green_stop_steps += 1
            self.green_stop_run += 1
            self.green_stop_max_steps = max(self.green_stop_max_steps, self.green_stop_run)
            if not self._in_green_stop:
                self.green_stop_events += 1
                self._in_green_stop = True
        else:
            self._in_green_stop = False
            self.green_stop_run = 0

        self.teleports += int(env.ego_id in traci.simulation.getStartingTeleportIDList())
        if present and speed < ZERO_SPEED and not red_wait:
            self._stall_run += 1
            self.stall_steps += 1
            if self._stall_run == 1:
                self._stall_edge = traci.vehicle.getRoadID(env.ego_id)
                pos = float(traci.vehicle.getLanePosition(env.ego_id))
                x, y = traci.vehicle.getPosition(env.ego_id)
                platforms = [(st["id"], st["edge"], st["start"], st["end"]) for st in env._stops]
                parking = [(name, road, lo, hi) for name, road, lo, hi, *_ in env._hotspots]
                platform, stop_distance = nearest_interval(platforms, self._stall_edge, pos)
                hotspot, hotspot_distance = nearest_interval(parking, self._stall_edge, pos)
                self._stall_location = dict(pos=pos, x=x, y=y, start_time=traci.simulation.getTime(),
                                            platform=platform, platform_distance_m=stop_distance,
                                            hotspot=hotspot, hotspot_distance_m=hotspot_distance)
                self._stall_place = f"{self._stall_edge} @ {pos:.1f} m"
        else:
            if self._stall_run >= INF_WAIT_STEPS:
                self.inf_wait_runs.append({
                    "steps": self._stall_run,
                    "sec": self._stall_run * STEP_SEC,
                    "place": self._stall_place, **self._stall_location,
                    "edge": self._stall_edge,
                })
            if self._stall_run > self.stall_max_steps:
                self.stall_max_steps = self._stall_run
            self._stall_run = 0

        if info.get("lane_change_applied") and int(info.get("lane_change") or 0) != 0:
            self.lane_changes += 1
            self._steps_since_lc = 0
        else:
            self._steps_since_lc += 1

        if info.get("collided"):
            self.lc_steps_before_collision = min(
                self.lc_steps_before_collision, self._steps_since_lc)

    def finalize(self, env, info: dict, end_reason: str, terminated: bool, truncated: bool):
        if self._stall_run >= INF_WAIT_STEPS:
            self.inf_wait_runs.append({
                "steps": self._stall_run,
                "sec": self._stall_run * STEP_SEC,
                "place": self._stall_place, **self._stall_location,
                "edge": self._stall_edge,
            })
        if self._stall_run > self.stall_max_steps:
            self.stall_max_steps = self._stall_run
        self.sim_errors = int(end_reason in ("vanished", "sumo_error"))
        if info.get("collided"):
            self.collision_place = _place_label(env, info)
            self.collision_class, self.collision_partner = _classify_collision(
                env, info, self.lc_steps_before_collision)

    def row_dict(self, seed: int, end_reason: str, stops_done: int, stops_total: int,
                 place: str, **extra) -> dict:
        n = max(self.sim_steps, 1)
        avg_kmh = (self.distance_m / (n * STEP_SEC)) * 3.6
        moving_kmh = (self.speed_time_integral / max(self.move_time_s, 1e-6)) * 3.6
        max_stall_sec = self.stall_max_steps * STEP_SEC
        max_inf = max((r["sec"] for r in self.inf_wait_runs), default=0.0)
        return {
            "seed": seed,
            "end_reason": end_reason,
            "stops_done": stops_done,
            "stops_total": stops_total,
            "place": place,
            "sim_steps": self.sim_steps,
            "distance_m": round(self.distance_m, 1),
            "green_stop_time_ratio": self.green_stop_steps / n,
            "green_stop_event_count": self.green_stop_events,
            "green_stop_max_sec": self.green_stop_max_steps * STEP_SEC,
            "inf_wait_count": len(self.inf_wait_runs),
            "inf_wait_max_sec": max_inf,
            "inf_wait_runs": self.inf_wait_runs,
            "stall_max_sec": max_stall_sec,
            "lane_changes": self.lane_changes,
            "avg_speed_kmh": avg_kmh,
            "moving_avg_speed_kmh": moving_kmh,
            "slow_moving": moving_kmh < 15.0,
            "teleports": self.teleports,
            "sim_errors": self.sim_errors,
            "collision_class": self.collision_class,
            "collision_partner": self.collision_partner,
            "collision_place": self.collision_place,
            **extra,
        }


def nearest_interval(zones, edge, pos):
    distances = [(max(float(start) - pos, 0.0, pos - float(end)), name)
                 for name, road, start, end in zones if road == edge]
    if not distances or not math.isfinite(pos):
        return "", None
    distance, name = min(distances)
    return name, round(float(distance), 2)


def _place_label(env, info: dict) -> str:
    edge = ""
    try:
        import traci
        if env.ego_id in traci.vehicle.getIDList():
            edge = str(traci.vehicle.getRoadID(env.ego_id))
            pos = float(traci.vehicle.getLanePosition(env.ego_id))
        else:
            pos = None
        stop = ""
        current = env._current_stop()
        if current is not None:
            stop = str(current["id"])
        if pos is not None and edge and not edge.startswith(":"):
            for (hot_edge, _lane), positions in getattr(env, "_hotspot_taken", {}).items():
                if hot_edge != edge:
                    continue
                for hot_pos in positions:
                    if abs(float(hot_pos) - pos) < 15.0:
                        for hid, hot_e, _ in getattr(env, "_hotspot_placed", []):
                            if hot_e == edge:
                                return str(hid)
        return stop
    except Exception:
        return str(info.get("nearest_stop_id") or "")


def _classify_collision(env, info: dict, steps_since_lc: int) -> tuple[str, str]:
    partner = "unknown"
    try:
        import traci
        for c in traci.simulation.getCollisions():
            if env.ego_id not in (c.collider, c.victim):
                continue
            other = c.victim if c.collider == env.ego_id else c.collider
            if other in traci.vehicle.getIDList():
                vtype = traci.vehicle.getTypeID(other)
                partner = vtype
            elif other in traci.person.getIDList():
                partner = "pedestrian"
            break
    except Exception:
        pass
    gap = info.get("gap")
    place = _place_label(env, info)
    if partner == "pedestrian" or info.get("ped_collision"):
        return "d_motorcycle_or_ped", "pedestrian"
    if str(place).startswith("hs_"):
        return "c_illegal_park", partner
    if steps_since_lc <= 15 or int(info.get("lane_change") or 0) != 0:
        return "a_lane_change_side", partner
    if gap is not None and float(gap) < 8.0:
        return "b_rear_end", partner
    if "motorcycle" in str(partner).lower() or "bike" in str(partner).lower():
        return "d_motorcycle_or_ped", partner
    return "e_other", partner


def summarize_official(rows: list[dict], n: int | None = None) -> dict:
    n = n or max(len(rows), 1)
    reasons = {}
    for r in rows:
        reasons[r["end_reason"]] = reasons.get(r["end_reason"], 0) + 1
    official = reasons.get("completed", 0) + reasons.get("missed_stop", 0)
    collisions = reasons.get("collision", 0)
    wrong = reasons.get("lane_error", 0)
    timeouts = reasons.get("timeout", 0)
    inf_eps = sum(1 for r in rows if r.get("inf_wait_count", 0) > 0 or r["end_reason"] == "timeout")
    trouble = sum(1 for r in rows if r.get("inf_wait_count", 0) > 0 or r["end_reason"] in ("timeout", "lane_error"))
    return {
        "episodes": len(rows),
        "official_complete": official / n,
        "perfect_complete": reasons.get("completed", 0) / n,
        "collision_count": collisions,
        "collision_rate": collisions / n,
        "wrong_lane_count": wrong,
        "timeout_count": timeouts,
        "inf_wait_episodes": inf_eps,
        "wait_timeout_wrong_lane_episodes": trouble,
        "green_stop_max_sec": max((r.get("green_stop_max_sec", 0) for r in rows), default=0),
        "green_stop_time_mean": float(sum(r.get("green_stop_time_ratio", 0) for r in rows) / n),
        "green_stop_events_mean": float(sum(r.get("green_stop_event_count", 0) for r in rows) / n),
        "moving_avg_speed_kmh": float(sum(r.get("moving_avg_speed_kmh", 0) for r in rows) / n),
        "avg_speed_kmh": float(sum(r.get("avg_speed_kmh", 0) for r in rows) / n),
        "teleports": int(sum(r.get("teleports", 0) for r in rows)),
        "sim_errors": int(sum(r.get("sim_errors", 0) for r in rows)),
        "red_light": reasons.get("red_light", 0) / n,
        "yield_steps": int(sum(r.get("yield_steps", 0) for r in rows)),
        "intervention": float(sum(r.get("shield_steps", 0) for r in rows))
        / max(sum(r.get("sim_steps", 0) for r in rows), 1),
        "reasons": reasons,
    }


def checkpoint_sort_key(stats: dict, baseline_official: float) -> tuple:
    """선택 순서 + 공식 완주 10%p 규칙(낮으면 맨 뒤)."""
    if stats["official_complete"] < baseline_official - 0.10:
        return (1, 999, 999, 999, 999, -stats["official_complete"])
    return (
        0,
        stats["collision_count"],
        stats["wait_timeout_wrong_lane_episodes"],
        stats["green_stop_time_mean"],
        -stats["moving_avg_speed_kmh"],
        -stats["official_complete"],
    )