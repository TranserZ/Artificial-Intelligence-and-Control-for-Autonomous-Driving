"""env-v2 공통 평가기 구현.

체크포인트를 지정된 판 시드에서 평가해 판별 CSV/메타/trace를 만들거나,
병렬 실행으로 생긴 조각을 검증·병합해 학습 시드 요약과 그래프를 만든다.
각 평가 프로세스는 별도 임시 SUMO 디렉터리를 사용한다.
"""
import argparse
import csv
import glob
import gzip
import importlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT)

from checkpoints import ALL_CODES, NAMES, CheckpointTracker  # noqa: E402

EPISODE_COLUMNS = [
    "seed", "steps", "distance_km", "avg_speed_kmh", "end_reason",
    "missed_stops", "stop_count", "shield_step_ratio", "events_total",
    "events_vehicle", "events_signal", "events_ped_filter", "events_lc_gate",
    "collision_with", "invalid_lc", "ped_yield_violations",
    "school_zone_speeding_ratio", "lane_changes", "max_decel", "mean_abs_jerk",
    "wall_time_s", *[f"pass_{code}" for code in ALL_CODES],
]
TRACE_COLUMNS = [
    "seed", "distance_m", "speed_kmh", "shield_active", "shield_vehicle",
    "shield_signal", "shield_ped_filter", "shield_lc_gate", "collision",
]
EVENT_KEYS = {
    "vehicle": "shield_vehicle",
    "signal": "shield_signal",
    "ped_filter": "shield_ped",
    "lc_gate": "shield_lc_gate",
}


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_ok(*args):
    try:
        subprocess.check_call(
            ["git", *args], cwd=ROOT, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def runtime_versions():
    packages = {}
    for name in (
            "gymnasium", "matplotlib", "numpy",
            "tensorboard", "torch", "traci"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        sumo = subprocess.check_output(
            ["sumo", "--version"], text=True,
            stderr=subprocess.STDOUT).splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        sumo = None
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "sumo": sumo,
        "packages": packages,
    }


def parse_range(text):
    match = re.fullmatch(r"\s*(\d+)(?:-(\d+))?\s*", text or "")
    if not match:
        raise argparse.ArgumentTypeError(
            "시드는 시작-끝 형식이어야 합니다 (예: 1000-1099)")
    start, end = int(match.group(1)), int(match.group(2) or match.group(1))
    if start > end:
        raise argparse.ArgumentTypeError("시드 시작값이 끝값보다 클 수 없습니다")
    return start, end


def safe_name(text):
    value = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text)).strip("._")
    if not value:
        raise ValueError("파일명으로 쓸 수 있는 모델 이름이 필요합니다")
    return value


def atomic_write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    os.replace(temporary, path)


def atomic_write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    opener = gzip.open if path.endswith(".gz") else open
    with opener(
            temporary, "wt", newline="",
            encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(
            path, "rt", newline="",
            encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def load_factory(spec):
    if ":" not in spec:
        raise ValueError("--policy-factory는 module:function 형식이어야 합니다")
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"{spec}은 호출 가능한 함수가 아닙니다")
    return function


def make_policy(args, env):
    """기본 PPO/BC 또는 모델 브랜치의 factory로 정책을 만든다.

    factory:
      make_policy(checkpoint_path, env, config_path=None) -> predict를 가진 객체
    """
    if args.policy_factory:
        return load_factory(args.policy_factory)(
            args.ckpt, env, args.model_config)
    from test import load_agent
    return load_agent(
        args.ckpt, env.observation_space.shape[0],
        env.action_space.shape[0], args.algorithm)


class EventCounter:
    """두 스텝 연속 비활성 뒤 재개입할 때만 새 이벤트로 센다."""

    def __init__(self):
        self.count = 0
        self.in_event = False
        self.inactive_steps = 0

    def update(self, active):
        if active:
            if not self.in_event:
                self.count += 1
                self.in_event = True
            self.inactive_steps = 0
        elif self.in_event:
            self.inactive_steps += 1
            if self.inactive_steps >= 2:
                self.in_event = False


def get_odometer(env, fallback):
    import traci
    try:
        if env.ego_id in traci.vehicle.getIDList():
            return max(
                fallback, float(traci.vehicle.getDistance(env.ego_id)))
    except traci.TraCIException:
        pass
    return fallback


def get_end_reason(info, terminated, truncated):
    if info.get("collided"):
        return "collision"
    if info.get("red_violation") and terminated:
        return "red_light"
    if info.get("wrong_lane"):
        return "lane_error"
    if info.get("arrived"):
        return (
            "missed_stop"
            if int(info.get("missed_stops", 0) or 0)
            else "completed")
    if truncated:
        return "timeout"
    return "vanished"


def run_episode(policy, env, seed):
    started = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    tracker = CheckpointTracker()
    counters = {
        "total": EventCounter(),
        **{key: EventCounter() for key in EVENT_KEYS},
    }
    trace, speeds, accelerations, jerks = [], [], [], []
    steps = shield_steps = invalid_lc = yield_violations = lane_changes = 0
    zone_steps = zone_speeding_steps = 0
    distance_m = 0.0
    previous_speed = previous_accel = None
    info, terminated, truncated = {}, False, False

    while not (terminated or truncated):
        action = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        active = bool(info.get("shield_active"))
        shield_steps += int(active)
        counters["total"].update(active)
        for key, info_key in EVENT_KEYS.items():
            counters[key].update(bool(info.get(info_key)))

        command = int(info.get("lane_change", 0) or 0)
        applied = bool(info.get("lane_change_applied"))
        invalid_lc += int(
            command != 0 and not applied
            and not info.get("shield_lc_gate"))
        lane_changes += int(command != 0 and applied)
        yield_violations += int(bool(info.get("yield_violation")))

        speed = info.get("speed")
        if speed is not None:
            speed = float(speed)
            speeds.append(speed)
            if previous_speed is not None:
                accel = (
                    speed - previous_speed) / float(env.step_length)
                accelerations.append(accel)
                if previous_accel is not None:
                    jerks.append(
                        (accel - previous_accel) / float(env.step_length))
                previous_accel = accel
            previous_speed = speed

        if info.get("in_school_zone"):
            zone_steps += 1
            limit = info.get("zone_limit")
            zone_speeding_steps += int(
                speed is not None and limit is not None
                and speed > float(limit) + 0.3)

        distance_m = get_odometer(env, distance_m)
        tracker.update(distance_m, speed or 0.0)
        trace.append({
            "seed": seed,
            "distance_m": round(distance_m, 3),
            "speed_kmh": round((speed or 0.0) * 3.6, 4),
            "shield_active": int(active),
            "shield_vehicle": int(bool(info.get("shield_vehicle"))),
            "shield_signal": int(bool(info.get("shield_signal"))),
            "shield_ped_filter": int(bool(info.get("shield_ped"))),
            "shield_lc_gate": int(bool(info.get("shield_lc_gate"))),
            "collision": int(bool(info.get("collided"))),
        })

    row = {
        "seed": seed,
        "steps": steps,
        "distance_km": round(distance_m / 1000.0, 6),
        "avg_speed_kmh": (
            round(statistics.fmean(speeds) * 3.6, 4)
            if speeds else 0.0),
        "end_reason": get_end_reason(info, terminated, truncated),
        "missed_stops": int(info.get("missed_stops", 0) or 0),
        "stop_count": int(info.get("stops_done", 0) or 0),
        "shield_step_ratio": round(
            shield_steps / max(steps, 1), 8),
        "events_total": counters["total"].count,
        "events_vehicle": counters["vehicle"].count,
        "events_signal": counters["signal"].count,
        "events_ped_filter": counters["ped_filter"].count,
        "events_lc_gate": counters["lc_gate"].count,
        "collision_with": (
            "pedestrian" if info.get("ped_collision")
            else "vehicle" if info.get("collided") else "none"),
        "invalid_lc": invalid_lc,
        "ped_yield_violations": yield_violations,
        "school_zone_speeding_ratio": round(
            zone_speeding_steps / max(zone_steps, 1), 8),
        "lane_changes": lane_changes,
        "max_decel": round(
            max([0.0, *(-value for value in accelerations)]), 4),
        "mean_abs_jerk": (
            round(statistics.fmean(map(abs, jerks)), 4)
            if jerks else 0.0),
        "wall_time_s": round(time.perf_counter() - started, 3),
        **tracker.as_row(),
    }
    return row, trace


def environment_status(tag):
    tag_commit = _git("rev-parse", "--verify", f"{tag}^{{commit}}")
    return {
        "target_tag": tag,
        "tag_commit": tag_commit,
        "current_commit": _git("rev-parse", "HEAD"),
        "tag_is_ancestor": bool(
            tag_commit and _git_ok(
                "merge-base", "--is-ancestor", tag, "HEAD")),
        "tracked_files_dirty": bool(
            _git("status", "--porcelain", "--untracked-files=no")),
    }


def evaluate(args):
    from env import road_builder
    from env.mdp_config import ACTION, OBSERVATION, SIMULATION
    from env.road_config import EGO, ROAD, TRAFFIC
    from env.sumo_env import SumoHighwayEnv
    from train import REWARD

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(
            f"체크포인트가 없습니다: {args.ckpt}")
    status = environment_status(args.env_tag)
    if (
            not args.allow_unverified_env
            and not status["tag_is_ancestor"]):
        raise SystemExit(
            f"{args.env_tag} 태그가 없거나 현재 커밋의 조상이 아닙니다. "
            "공식 평가는 확정 태그에서만 실행하세요. 개발 스모크만 "
            "--allow-unverified-env를 사용할 수 있습니다.")

    start_seed, end_seed = args.seeds
    model_name = safe_name(args.model_name)
    step_name = safe_name(args.ckpt_step)
    stem = (
        f"ts{args.train_seed}_step{step_name}_{args.shield}_"
        f"{start_seed}-{end_seed}")
    out_dir = os.path.abspath(args.out)
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    trace_path = os.path.join(
        out_dir, f"{stem}.trace.csv.gz")
    meta_path = os.path.join(out_dir, f"{stem}.json")
    output_paths = (csv_path, trace_path, meta_path)
    if (
            not args.overwrite
            and any(os.path.exists(path) for path in output_paths)):
        raise FileExistsError(
            f"기존 평가 파일이 있습니다: {stem} "
            "(--overwrite로 교체)")

    sim = {
        **SIMULATION,
        "shield": {
            **SIMULATION.get("shield", {}),
            "mode": args.shield,
        },
    }
    rows, traces = [], []
    prefix = f"cheonggye_eval_{os.getpid()}_"
    with tempfile.TemporaryDirectory(prefix=prefix) as sumo_dir:
        cfg_path = road_builder.build(
            ROAD, EGO, TRAFFIC, sumo_dir)
        env = SumoHighwayEnv(
            cfg_path=cfg_path, road=ROAD, ego=EGO,
            mdp_sim=sim, mdp_obs=OBSERVATION, action=ACTION,
            reward=REWARD, traffic=TRAFFIC, gui=False)
        env.eval_mode = True
        env.shield_mode = args.shield
        try:
            policy = make_policy(args, env)
            for seed in range(start_seed, end_seed + 1):
                print(
                    f"[{model_name} ts{args.train_seed} "
                    f"{args.ckpt_step}] seed {seed}",
                    flush=True)
                row, episode_trace = run_episode(
                    policy, env, seed)
                rows.append(row)
                traces.extend(episode_trace)
        finally:
            env.close()

    atomic_write_csv(
        csv_path, EPISODE_COLUMNS, rows)
    atomic_write_csv(
        trace_path, TRACE_COLUMNS, traces)
    meta = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(
            timespec="seconds"),
        "model_name": model_name,
        "checkpoint": os.path.abspath(args.ckpt),
        "checkpoint_step": args.ckpt_step,
        "train_seed": args.train_seed,
        "scenario_seed_range": [start_seed, end_seed],
        "shield": args.shield,
        "algorithm": args.algorithm,
        "policy_factory": args.policy_factory,
        "model_config": args.model_config,
        "runtime": runtime_versions(),
        "environment": status,
        "deterministic": True,
        "eval_mode": True,
        "episode_csv": os.path.basename(csv_path),
        "trace_csv": os.path.basename(trace_path),
        "seed_validation": {
            "expected_count": end_seed - start_seed + 1,
            "actual_count": len(rows),
            "missing": [],
            "duplicates": [],
            "out_of_range": [],
            "passed": (
                len(rows) == end_seed - start_seed + 1),
        },
    }
    atomic_write_json(meta_path, meta)
    print(f"평가 저장: {csv_path}")


def numeric(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def summarize_rows(rows):
    count = len(rows)
    total_steps = sum(
        int(numeric(row, "steps")) for row in rows)
    total_distance = sum(
        numeric(row, "distance_km") for row in rows)
    reasons = defaultdict(int)
    for row in rows:
        reasons[row["end_reason"]] += 1
    return {
        "episodes": count,
        "completion_rate": (
            reasons["completed"] / max(count, 1)),
        "shield_step_ratio": (
            sum(
                numeric(row, "shield_step_ratio")
                * int(numeric(row, "steps"))
                for row in rows)
            / max(total_steps, 1)),
        "interventions_per_km": (
            sum(
                int(numeric(row, "events_total"))
                for row in rows)
            / max(total_distance, 1e-12)),
        "avg_speed_kmh": (
            statistics.fmean(
                numeric(row, "avg_speed_kmh")
                for row in rows)
            if rows else 0.0),
        "missed_stops": sum(
            int(numeric(row, "missed_stops"))
            for row in rows),
        "collision_vehicle": sum(
            row.get("collision_with") == "vehicle"
            for row in rows),
        "collision_pedestrian": sum(
            row.get("collision_with") == "pedestrian"
            for row in rows),
        "end_reasons": dict(sorted(reasons.items())),
        "pass_rates": {
            code: (
                sum(
                    int(numeric(row, f"pass_{code}"))
                    for row in rows)
                / max(count, 1))
            for code in ALL_CODES
        },
    }


def load_piece(path):
    meta_path = os.path.splitext(path)[0] + ".json"
    trace_path = os.path.splitext(path)[0] + ".trace.csv.gz"
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"메타 파일이 없습니다: {meta_path}")
    if not os.path.isfile(trace_path):
        raise FileNotFoundError(
            f"trace 파일이 없습니다: {trace_path}")
    with open(meta_path, encoding="utf-8") as file:
        meta = json.load(file)
    return meta, read_csv(path), read_csv(trace_path)


def validate_group(pieces, expected):
    rows, traces, source_files = [], [], []
    for path, _, piece_rows, piece_traces in pieces:
        source_files.append(os.path.basename(path))
        rows.extend(piece_rows)
        traces.extend(piece_traces)
    seen = defaultdict(int)
    for row in rows:
        seen[int(row["seed"])] += 1
    expected_set = set(expected)
    actual_set = set(seen)
    missing = sorted(expected_set - actual_set)
    duplicates = sorted(
        seed for seed, count in seen.items()
        if count > 1)
    outside = sorted(actual_set - expected_set)
    passed = (
        not (missing or duplicates or outside)
        and len(rows) == len(expected))
    result = {
        "expected_count": len(expected),
        "actual_count": len(rows),
        "missing": missing,
        "duplicates": duplicates,
        "out_of_range": outside,
        "passed": passed,
    }
    if not passed:
        raise ValueError(
            f"시드 병합 검사 실패: {result}")
    rows.sort(key=lambda row: int(row["seed"]))
    traces.sort(
        key=lambda row: (
            int(row["seed"]), float(row["distance_m"])))
    return rows, traces, source_files, result


def mean_sd(values):
    values = list(values)
    return {
        "mean": (
            statistics.fmean(values) if values else 0.0),
        "sd": (
            statistics.stdev(values)
            if len(values) > 1 else 0.0),
    }


def save_plots(out_dir, label, all_rows, traces):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if os.name == "nt":
            plt.rcParams["font.family"] = "Malgun Gothic"
            plt.rcParams["axes.unicode_minus"] = False
    except ImportError as exc:
        raise RuntimeError(
            "평가 그래프 생성에 matplotlib가 필요합니다") from exc

    if traces:
        distances = np.asarray([
            float(row["distance_m"]) for row in traces])
        speeds = np.asarray([
            float(row["speed_kmh"]) for row in traces])
        last_edge = max(
            2200.0, float(distances.max()) + 50.0)
        edges = np.arange(0.0, last_edge, 50.0)
        indices = np.digitize(distances, edges)
        x_values, y_values = [], []
        for index in range(1, len(edges)):
            values = speeds[indices == index]
            if values.size:
                x_values.append(
                    (edges[index - 1] + edges[index]) / 2.0)
                y_values.append(float(values.mean()))
        figure, axis = plt.subplots(figsize=(11, 4))
        axis.plot(x_values, y_values, linewidth=1.8)
        axis.axvspan(
            1333, 1610, color="gold", alpha=0.18,
            label="school zone")
        axis.set(
            xlabel="Route distance (m)",
            ylabel="Mean speed (km/h)",
            title=f"{label} - speed by distance")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            os.path.join(
                out_dir, f"{label}_speed_by_distance.png"),
            dpi=160)
        plt.close(figure)

    rates = [
        sum(
            int(numeric(row, f"pass_{code}"))
            for row in all_rows)
        / max(len(all_rows), 1)
        for code in ALL_CODES
    ]
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar(ALL_CODES, rates)
    axis.set_ylim(0, 1.05)
    axis.set(
        xlabel="Checkpoint", ylabel="Pass rate",
        title=f"{label} - checkpoint pass rate")
    for index, (code, rate) in enumerate(
            zip(ALL_CODES, rates)):
        axis.text(
            index, rate + 0.02, f"{rate:.0%}",
            ha="center", fontsize=8)
        axis.text(
            index, -0.14, NAMES[code],
            ha="center", va="top", fontsize=7,
            rotation=20)
    figure.tight_layout()
    figure.savefig(
        os.path.join(
            out_dir, f"{label}_checkpoint_pass_rate.png"),
        dpi=160)
    plt.close(figure)

    location_types = [
        ("collision", "collision", "x"),
        ("shield_vehicle", "vehicle", "o"),
        ("shield_signal", "signal", "s"),
        ("shield_ped_filter", "pedestrian", "^"),
        ("shield_lc_gate", "lane gate", "D"),
    ]
    figure, axis = plt.subplots(figsize=(11, 4))
    for level, (key, name, marker) in enumerate(
            location_types):
        locations = [
            float(row["distance_m"])
            for row in traces
            if int(float(row[key])) != 0
        ]
        if locations:
            axis.scatter(
                locations, [level] * len(locations),
                s=10, marker=marker, alpha=0.35,
                label=name)
    axis.set_yticks(
        range(len(location_types)),
        [item[1] for item in location_types])
    axis.set(
        xlabel="Route distance (m)",
        title=f"{label} - collision/intervention locations")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        os.path.join(
            out_dir, f"{label}_event_locations.png"),
        dpi=160)
    plt.close(figure)


def summary_markdown(summary):
    lines = [
        f"# {summary['model_name']} 공통 평가 요약",
        "",
        f"- 체크포인트: `{summary['checkpoint_step']}`",
        f"- shield: `{summary['shield']}`",
        (
            f"- 판 시드: "
            f"{summary['scenario_seed_range'][0]}~"
            f"{summary['scenario_seed_range'][1]}"),
        (
            "- 학습 시드: "
            + ", ".join(map(str, summary["train_seeds"]))),
        "",
        (
            "| 학습 시드 | 완주율 | 수동개입률 | "
            "km당 개입 | 평균 속도(km/h) |"),
        "|---|---:|---:|---:|---:|",
    ]
    items = sorted(
        summary["per_train_seed"].items(),
        key=lambda pair: int(pair[0]))
    for seed, item in items:
        lines.append(
            f"| {seed} | {item['completion_rate']:.2%} | "
            f"{item['shield_step_ratio']:.2%} | "
            f"{item['interventions_per_km']:.3f} | "
            f"{item['avg_speed_kmh']:.2f} |")
    aggregate = summary["three_seed_summary"]
    lines += [
        "",
        (
            "- 완주율 평균 ± 표준편차: "
            f"{aggregate['completion_rate']['mean']:.2%} ± "
            f"{aggregate['completion_rate']['sd']:.2%}"),
        (
            "- 수동개입률 평균 ± 표준편차: "
            f"{aggregate['shield_step_ratio']['mean']:.2%} ± "
            f"{aggregate['shield_step_ratio']['sd']:.2%}"),
        (
            "- 목표 판정: "
            + (
                "통과" if aggregate["target_passed"]
                else "미통과")
            + " (완주율 평균 ≥95%, 수동개입률 평균 ≤16%)"),
        "",
    ]
    return "\n".join(lines)


def merge_results(args):
    pattern = os.path.abspath(args.merge_glob)
    paths = sorted(
        path for path in glob.glob(pattern)
        if not path.endswith(".trace.csv")
        and "_merged_" not in os.path.basename(path))
    if not paths:
        raise FileNotFoundError(
            f"병합할 CSV가 없습니다: {args.merge_glob}")
    expected_start, expected_end = args.expected_seeds
    expected = range(expected_start, expected_end + 1)
    groups = defaultdict(list)
    for path in paths:
        meta, rows, traces = load_piece(path)
        verified = bool(
            meta.get("environment", {}).get("tag_is_ancestor"))
        if not args.allow_unverified_env and not verified:
            raise ValueError(
                f"환경 태그가 검증되지 않은 조각은 공식 병합할 수 없습니다: {path}")
        key = (
            meta["model_name"],
            int(meta["train_seed"]),
            str(meta["checkpoint_step"]),
            meta["shield"],
        )
        groups[key].append(
            (path, meta, rows, traces))

    identities = {
        (key[0], key[2], key[3])
        for key in groups
    }
    if len(identities) != 1:
        raise ValueError(
            "한 번에 한 모델/체크포인트/shield만 "
            f"병합할 수 있습니다: {identities}")
    model_name, checkpoint_step, shield = next(
        iter(identities))
    found_train_seeds = {key[1] for key in groups}
    if (
            not args.allow_partial_train_seeds
            and found_train_seeds != {0, 1, 2}):
        raise ValueError(
            "공식 요약에는 학습 시드 0, 1, 2가 모두 필요합니다. "
            f"현재: {sorted(found_train_seeds)}")
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    all_rows, all_traces = [], []
    per_seed, validation = {}, {}
    source_files = []

    sorted_groups = sorted(
        groups.items(), key=lambda item: item[0][1])
    for key, pieces in sorted_groups:
        train_seed = key[1]
        rows, traces, sources, check = validate_group(
            pieces, expected)
        stem = (
            f"ts{train_seed}_step"
            f"{safe_name(checkpoint_step)}_{shield}_"
            f"merged_{expected_start}-{expected_end}")
        atomic_write_csv(
            os.path.join(out_dir, f"{stem}.csv"),
            EPISODE_COLUMNS, rows)
        atomic_write_csv(
            os.path.join(out_dir, f"{stem}.trace.csv.gz"),
            TRACE_COLUMNS, traces)
        per_seed[str(train_seed)] = summarize_rows(rows)
        validation[str(train_seed)] = check
        source_files.extend(sources)
        all_rows.extend(rows)
        all_traces.extend(traces)

    train_seeds = sorted(map(int, per_seed))
    completion = [
        per_seed[str(seed)]["completion_rate"]
        for seed in train_seeds
    ]
    intervention = [
        per_seed[str(seed)]["shield_step_ratio"]
        for seed in train_seeds
    ]
    aggregate = {
        "completion_rate": mean_sd(completion),
        "shield_step_ratio": mean_sd(intervention),
        "target_passed": (
            statistics.fmean(completion) >= 0.95 - 1e-12
            and statistics.fmean(intervention)
            <= 0.16 + 1e-12),
    }
    label = (
        f"step{safe_name(checkpoint_step)}_{shield}_"
        f"{expected_start}-{expected_end}")
    summary = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(
            timespec="seconds"),
        "model_name": model_name,
        "checkpoint_step": checkpoint_step,
        "shield": shield,
        "scenario_seed_range": [
            expected_start, expected_end],
        "train_seeds": train_seeds,
        "per_train_seed": per_seed,
        "three_seed_summary": aggregate,
        "seed_validation": validation,
        "source_files": sorted(set(source_files)),
    }
    summary_path = os.path.join(
        out_dir, f"summary_{label}.json")
    atomic_write_json(summary_path, summary)
    with open(
            os.path.join(
                out_dir, f"summary_{label}.md"),
            "w", encoding="utf-8") as file:
        file.write(summary_markdown(summary))
    save_plots(
        out_dir, label, all_rows, all_traces)
    print(f"병합·요약 완료: {summary_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="env-v2 공통 모델 평가/병합")
    mode = parser.add_mutually_exclusive_group(
        required=True)
    mode.add_argument(
        "--ckpt", help="평가할 체크포인트")
    mode.add_argument(
        "--merge-glob",
        help="병합할 판별 CSV glob (따옴표로 감쌀 것)")
    parser.add_argument("--model-name")
    parser.add_argument(
        "--train-seed", type=int, choices=[0, 1, 2])
    parser.add_argument("--ckpt-step")
    parser.add_argument("--seeds", type=parse_range)
    parser.add_argument(
        "--shield", choices=["shadow", "on", "off"],
        default="shadow")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--algorithm", choices=["auto", "ppo", "bc"],
        default="auto")
    parser.add_argument(
        "--policy-factory",
        help=(
            "다른 알고리즘용 module:function 정책 로더 "
            "(predict 메서드 반환)"))
    parser.add_argument(
        "--model-config",
        help="policy factory에 넘길 모델 설정 파일")
    parser.add_argument(
        "--env-tag", default="env-v2")
    parser.add_argument(
        "--allow-unverified-env", action="store_true",
        help=(
            "개발 스모크 전용: 환경 태그 검사를 "
            "통과하지 않아도 실행"))
    parser.add_argument(
        "--overwrite", action="store_true")
    parser.add_argument(
        "--allow-partial-train-seeds", action="store_true",
        help="개발 병합 전용: 학습 시드 0, 1, 2 일부만 허용")
    parser.add_argument(
        "--expected-seeds", type=parse_range)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.ckpt:
        required = {
            "--model-name": args.model_name,
            "--train-seed": args.train_seed,
            "--ckpt-step": args.ckpt_step,
            "--seeds": args.seeds,
        }
        missing = [
            name for name, value in required.items()
            if value is None
        ]
        if missing:
            parser.error(
                "평가 모드 필수 옵션: "
                + ", ".join(missing))
        evaluate(args)
    else:
        if args.expected_seeds is None:
            parser.error(
                "병합 모드에는 --expected-seeds가 필요합니다")
        merge_results(args)


if __name__ == "__main__":
    main()
