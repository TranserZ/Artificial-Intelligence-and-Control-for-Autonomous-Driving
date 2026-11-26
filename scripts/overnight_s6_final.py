"""Run resumable stage-6 evaluations and aggregate saved episode files."""
import argparse
import csv
import subprocess
import time
from collections import Counter
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import signal
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.drive_parallel import prepare_sumo, run_seed, summarize

OUT = ROOT / "results/overnight_s6_final"
REPORT = ROOT / "docs/reports/2026-11-26_s6_overnight.md"
STUDENT = ROOT.parent / "청계로드-dongyeon-d54a761/models/dongyeon_bc/s6_student_obsfix.pt"
TEACHER = STUDENT.parents[1] / "bc_ppo/teachers/s4_teacher.pt"
JOBS = [
    ("stage6_student", 6, STUDENT, 30),
    ("stage6_kl200k", 6, ROOT / "results/klppo1_s6_s1/step200k.pt", 30),
    ("stage5_kl250k", 5, ROOT / "results/klppo1_s5_s0/step250k.pt", 10),
]


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(.2 * (attempt + 1))
    path.write_text(content, encoding="utf-8")


def drive(job):
    cfg, seed, model, group = job
    result = group / f"{seed}.json"
    trace = group / "traces" / f"{seed}.npz"
    if not (result.exists() and trace.exists()):
        row = run_seed(cfg, seed, str(model), "shadow", trace_path=trace,
                       teacher=group.name.startswith("teacher"),
                       lane_hold=group.name == "teacher_bs03_hold")
        write(result, json.dumps(row, ensure_ascii=False))


def report(output, jobs, destination):
    lines = [
        "# 6단계 야간 실험", "", "## 한눈에 보기", "",
        "1. 기존 진단은 판별 저장 이전에 bcppo import 오류로 종료됐다. 핑퐁 원인과 수정 여부는 검증 중이다.",
        "2. 독립 100판 평가 결과 대기 중.",
        "3. 최종 환경 채택 판정 보류.",
        "4. 학생 900103 (376.0–376.5초, bs_03)과 900100 (424.0–424.5초, 주정차 구간)을 화면에서 확인했다. 최종 후보 장면은 독립 평가 뒤 선별한다.",
        "5. 비정상 정지, 15 km/h 미만의 평균 이동 속도, 반경 50 m 내 차량 빈도를 확인한다. 공용 환경과 팀원 파일은 유지했다.",
        "", "## 0단계 정리", "",
        "기존 진단 폴더에 판별 결과가 없었다. 부모가 종료된 SUMO 두 개를 정리했고 TensorBoard는 유지했다.",
        "", "## 개발 평가", "",
    ]
    for name, _, _, count in jobs:
        group = output / name
        files = sorted(group.glob("*.json"))
        lines.extend([f"### {name} ({len(files)}/{count}판)", ""])
        if not files:
            continue
        rows = [json.loads(file.read_text(encoding="utf-8")) for file in files]
        stats = summarize(rows)
        lines.extend(["| 지표 | 값 |", "|---|---:|"])
        for key, value in stats.items():
            if key in ("collision_rate", "official_complete", "perfect_complete"):
                value = f"{round(value * len(rows))}/{len(rows)} ({value:.1%})"
            elif isinstance(value, float):
                value = f"{value:.5g}"
            lines.append(f"| {key} | {value} |")
        lines.append(f"| 평균 이동 속도 15 km/h 미만 | {sum(row.get('slow_moving', False) for row in rows)}/{len(rows)} |")
        reversals = Counter()
        displays = {}
        traffic_steps = total_steps = empty_episodes = 0
        for file in files:
            with np.load(group / "traces" / f"{file.stem}.npz") as trace:
                traffic = np.count_nonzero(trace["neighbors_50m"])
                traffic_steps += traffic
                total_steps += len(trace["step"])
                empty_episodes += int(traffic == 0)
                columns = dict(trace)
                changes = np.flatnonzero(columns["lane_delta"])
                for before, after in zip(changes, changes[1:]):
                    if after - before <= 10 and columns["lane_delta"][before] * columns["lane_delta"][after] < 0:
                        key = (columns["edge"][after], int(columns["pos"][after] // 10) * 10 if np.isfinite(columns["pos"][after]) else -1)
                        reversals[key] += 1
                        displays.setdefault(key, (file.stem, columns["time"][after]))
        lines.extend([
            f"| 5초 이내 실제 반대 방향 차선 변경 | {reversals.total()} |",
            f"| 주변 차량이 있었던 스텝 | {traffic_steps}/{total_steps} ({traffic_steps / total_steps:.1%}) |",
            f"| 주변 차량이 한 번도 없었던 판 | {empty_episodes}/{len(files)} |", "",
            "| edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |",
            "|---|---:|---:|---:|---:|",
        ])
        for (edge, band), events in reversals.most_common(15):
            seed, time = displays[(edge, band)]
            lines.append(f"| {edge} | {band} | {events} | {seed} | {time:.1f} |")
        lines.append("")
    heldout = [(name, output / name) for name, _, _, count in jobs if name.startswith("heldout_") and count == 100]
    verdicts = []
    for name, group in heldout:
        files = list(group.glob("*.json"))
        if len(files) != 100 or "klppo3_" not in name:
            continue
        result = summarize([json.loads(path.read_text(encoding="utf-8")) for path in files])
        verdicts.append((name, result))
    if verdicts:
        name, result = verdicts[0]
        passed = result["collision_count"] <= 4 and not any(result[k] for k in ("wrong_lane_count", "teleports", "sim_errors"))
        lines[5] = (f"2. {name}, 충돌 {result['collision_count']}/100, 공식 완주 {result['official_complete']:.1%}, "
                    f"완벽 완주 {result['perfect_complete']:.1%}, 무한대기·시간초과 {result['inf_wait_episodes']}/100.")
        lines[6] = ("3. 독립 100판 공식 안전 기준 통과. 정지·속도·현장 장면을 함께 검토한다."
                    if passed else "3. 독립 100판 공식 안전 기준 미달.")
    notes = OUT / "progress.md"
    if notes.exists():
        lines.extend([notes.read_text(encoding="utf-8"), ""])
    lines.extend([
        "## 오케스트레이터 계획", "",
        "- 정리와 1판 스모크 약 5분.",
        "- 6단계 학생·KL 60판, 5단계 10판, 선생님 및 통제 실험과 GUI 확인 최대 2시간.",
        "- 확인된 환경 원인이 있을 때 STOP_FIX 수정 및 재검증. 원인 진단과 합산 최대 4시간.",
        "- 기존 체크포인트별 개발 30판 평가. 소요 시간은 첫 평가 처리량으로 산정.",
        "- 두 시드 병렬 학습. 1차 실측 10.68스텝/초 기준 25만 스텝은 시드당 약 6시간 30분이며, 동시 실행 시 달라진다. 학습 종료 또는 건강검진 중단 후 평가로 전환한다.",
        "- 독립 100판 평가, GUI 확인 및 보고 최대 2시간.", "",
    ])
    write(destination, "\n".join(lines))


def evaluate(name, stage, model, count, start=900100, workers=4, output=OUT):
    group = output / name
    seeds = [seed for seed in range(start, start + count)
             if not ((group / f"{seed}.json").exists() and (group / "traces" / f"{seed}.npz").exists())]
    if not seeds:
        return
    if not model.is_file():
        raise FileNotFoundError(model)
    os.environ["CGC_STAGE"] = str(stage)
    work, cfg = prepare_sumo()
    try:
        tasks = [(cfg, seed, model, group) for seed in seeds]
        with get_context("spawn").Pool(min(workers, len(tasks))) as pool:
            for _ in pool.imap_unordered(drive, tasks):
                yield None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def stats(name, count):
    files = [OUT / name / f"{seed}.json" for seed in range(900100, 900100 + count)]
    rows = [json.loads(file.read_text(encoding="utf-8")) for file in files]
    return summarize(rows)


def pings(name, count):
    total = 0
    for seed in range(900100, 900100 + count):
        with np.load(OUT / name / "traces" / f"{seed}.npz") as trace:
            delta = trace["lane_delta"]
        changes = np.flatnonzero(delta)
        total += sum(after-before <= 10 and delta[after]*delta[before] < 0
                     for before, after in zip(changes, changes[1:]))
    return total


def run_command(args, name, background=False):
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f"{name}.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "-u", *map(str, args)], cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                                start_new_session=os.name != "nt")
        if background:
            return proc
        if proc.wait():
            raise RuntimeError(f"See {logs / (name + '.log')}")


def stop_tree(proc):
    if proc.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)


def training_args(model, name, seed, smoke=False):
    args = ["train.py", "--ref", model, "--run-name", name, "--seed", seed,
            "--explore-std", 0.079, "--unfreeze-lane", "--lane-temp", 1.0,
            "--stage6-lane-rewards", "--replay-prob", 0.3,
            "--timesteps", 4096 if smoke else 250000]
    if model != STUDENT:
        args.extend(["--init", model])
    if smoke:
        args.extend(["--warmup-updates", 1, "--save-steps", "2048,4096"])
    return args


def note(text):
    with (OUT / "progress.md").open("a", encoding="utf-8") as stream:
        stream.write(text + "\n\n")


def pipeline():
    from utils.official_metrics import checkpoint_sort_key
    import math

    jobs = list(JOBS)

    def update():
        report(OUT, jobs, REPORT)

    def batch(name, stage, model, count, start=900100, workers=4):
        if not any(row[0] == name for row in jobs):
            jobs.append((name, stage, model, count))
        for _ in evaluate(name, stage, model, count, start, workers):
            update()

    os.environ.update(CGC_STAGE="6", OBS_SCAN="1", STOP_FIX="0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        for seed, start in ((900103, 372), (900100, 422)):
            folder = OUT / "gui" / str(seed)
            if len(list(folder.glob("*.png"))) >= 15:
                continue
            run_command(["scripts/gui_drive_seed.py", "--seed", seed, "--model", "student",
                         "--delay", 0, "--screenshots", folder,
                         "--from-time", start, "--to-time", start + 7], f"gui_{seed}")
            note(f"- 학생 수동 검토용 GUI 시드 {seed}, {start}–{start + 7}초. `{folder}`")
            update()
    except RuntimeError as exc:
        note(f"- GUI 캡처 실패. {exc}")
        update()

    smoke_jobs = [("stage6_student", 6, STUDENT, 1), ("teacher56", 6, TEACHER, 1),
                  ("teacher_bs03_hold", 6, TEACHER, 1)]
    for name, stage, model, count in smoke_jobs:
        smoke_output = OUT / "validated_smoke"
        start = 900103 if name == "teacher_bs03_hold" else 900100
        for _ in evaluate(name, stage, model, count, start=start, output=smoke_output):
            report(smoke_output, smoke_jobs, smoke_output / "report.md")
    batch("teacher56", 6, TEACHER, 10)
    control_group = OUT / "teacher_bs03_hold"
    if not control_group.exists():
        shutil.copytree(OUT / "validated_smoke/teacher_bs03_hold", control_group)
    batch("teacher_bs03_hold", 6, TEACHER, 1, start=900103)
    with np.load(OUT / "teacher56/traces/900103.npz") as original, np.load(
            OUT / "teacher_bs03_hold/traces/900103.npz") as control:
        def platform_dwell(trace):
            stop = trace["stop"] == "bs_03"
            if np.any(trace["stop_success"][stop]):
                return 10.0
            return float(np.max(trace["dwell_seconds"][stop], initial=0))
        applied = int(np.count_nonzero(control["lane_hold"]))
        note(f"- 선생님 시드 900103의 bs_03 최대 정차는 원본 {platform_dwell(original):.1f}초, "
             f"차로 유지 대조군 {platform_dwell(control):.1f}초. 유지 명령 적용 {applied}스텝.")
        update()
    for name, stage, model, count in JOBS:
        batch(name, stage, model, count)
    os.environ["CGC_STAGE"] = "6"
    candidates = [("stage6_student", STUDENT), ("stage6_kl200k", JOBS[1][2])]
    for branch in ("klppo1_s6_s0", "klppo1_s6_s1"):
        for model in sorted((ROOT / "results" / branch).glob("step*k.pt")):
            if model != JOBS[1][2]:
                candidates.append((f"{branch}_{model.stem}", model))
    student_rate = stats("stage6_student", 30)["official_complete"]
    for name, model in candidates:
        batch(name, 6, model, 30)
    ranked = [(name, model, stats(name, 30)) for name, model in candidates
              if len(list((OUT / name).glob("*.json"))) == 30]
    ranked.sort(key=lambda item: checkpoint_sort_key(item[2], student_rate))
    name, model, baseline = ranked[0]
    note(f"## 출발 모델\n\n- {name}, 충돌 {baseline['collision_count']}/30, 공식 완주 {baseline['official_complete']:.1%}.")
    update()
    run_command(training_args(model, "klppo3_smoke", 0, smoke=True), "training_smoke")
    with (ROOT / "results/klppo3_smoke/training_log.csv").open(encoding="utf-8") as stream:
        smoke = list(csv.DictReader(stream))
    metrics = ("lane_reversal_penalty", "unsafe_lane_change_penalty", "lane_kl")
    if len(smoke) != 2 or any(not all(math.isfinite(float(row[key])) for row in smoke)
                              or not any(float(row[key]) > 0 for row in smoke) for key in metrics):
        raise RuntimeError(f"Smoke did not exercise every requested metric: {smoke}")
    note("- 2048스텝 × 2 스모크에서 두 감점과 비대칭 벌점에 유한한 양수 값이 기록됐다.")
    update()
    os.environ["CGC_STAGE"] = "6"
    base10 = stats(name, 10)
    base_pings = pings(name, 10)

    def train_pair(retry=False):
        label = "_retry" if retry else ""
        processes = {}
        stopped = set()
        checked = set()
        try:
            for seed in (0, 1):
                run = f"klppo3_s6_s{seed}{label}"
                args = training_args(model, run, seed)
                if retry:
                    args.extend(["--unsafe-lane-penalty", 1.5])
                processes[run] = run_command(args, f"training_s{seed}{label}", True)
            while any(proc.poll() is None for proc in processes.values()):
                for run, proc in processes.items():
                    if run in stopped:
                        continue
                    if proc.poll() not in (None, 0):
                        stopped.add(run)
                        note(f"- {run} 학습 종료 코드 {proc.returncode}. 학습 로그 참조.")
                        update()
                        continue
                    for step in (50, 100):
                        ckpt = ROOT / "results" / run / f"step{step}k.pt"
                        group = f"health_{run}_{step}k"
                        if group in checked or not ckpt.is_file():
                            continue
                        checked.add(group)
                        batch(group, 6, ckpt, 10, workers=2)
                        crashes = stats(group, 10)["collision_count"]
                        reversals = pings(group, 10)
                        bad = crashes >= base10["collision_count"] + 3 or reversals > base_pings
                        note(f"- {group} 충돌 {crashes}/10 (출발 {base10['collision_count']}/10), "
                             f"핑퐁 {reversals} (출발 {base_pings}). {'중단' if bad else '통과'}.")
                        update()
                        if bad:
                            stopped.add(run)
                            stop_tree(proc)
                            break
                if any(proc.poll() is None for proc in processes.values()):
                    time.sleep(20)
        finally:
            for proc in processes.values():
                stop_tree(proc)
        return list(processes), stopped

    runs, stopped = train_pair()
    if len(stopped) == 2:
        note("- 두 시드가 중단됐다. 차선 변경 중 위험 판정 감점을 1.0에서 1.5로 올리는 재시도를 한 번 진행한다.")
        update()
        retry_runs, _ = train_pair(retry=True)
        runs.extend(retry_runs)
    finals = []
    for run in runs:
        for step in (50, 100, 150, 200, 250):
            checkpoint = ROOT / "results" / run / f"step{step}k.pt"
            if checkpoint.is_file():
                group = f"{run}_{checkpoint.stem}"
                batch(group, 6, checkpoint, 30)
                finals.append((group, checkpoint, stats(group, 30)))
    finals.sort(key=lambda item: checkpoint_sort_key(item[2], baseline["official_complete"]))
    finals = [item for item in finals if not checkpoint_sort_key(item[2], baseline["official_complete"])[0]]
    if not finals or finals[0][2]["collision_count"] > 2:
        note("## 최종 판정\n\n개발 30판에서 독립 평가 진입 기준을 충족한 모델이 없다.")
        update()
        return
    for group, checkpoint, _ in [(name, model, baseline), *finals[:2]]:
        held = f"heldout_{group}"
        batch(held, 6, checkpoint, 100, start=900300)
        rows = [json.loads(file.read_text(encoding="utf-8")) for file in (OUT / held).glob("*.json")]
        result = summarize(rows)
        traffic_steps = total_steps = empty_runs = 0
        for row in rows:
            with np.load(OUT / held / "traces" / f"{row['seed']}.npz") as trace:
                traffic = trace["neighbors_50m"]
                occupied = int(np.count_nonzero(traffic))
                traffic_steps += occupied
                total_steps += len(traffic)
                empty_runs += int(occupied == 0)
        traffic_rate = traffic_steps / max(total_steps, 1)
        if group == name:
            baseline_held = result
            baseline_traffic = traffic_rate
        passed = result["collision_count"] <= 4 and not any(result[k] for k in ("wrong_lane_count", "teleports", "sim_errors"))
        note(f"- {held} 충돌 {result['collision_count']}/100, 공식 완주 {result['official_complete']:.1%}, "
             f"완벽 완주 {result['perfect_complete']:.1%}, 무한대기·시간초과 {result['inf_wait_episodes']}/100, "
             f"공식 기준 {'통과' if passed else '미달'}.")
        normal = (result["moving_avg_speed_kmh"] >= max(15.0, baseline_held["moving_avg_speed_kmh"] - 1.0)
                  and result["inf_wait_episodes"] <= baseline_held["inf_wait_episodes"]
                  and result["green_stop_time_mean"] <= baseline_held["green_stop_time_mean"]
                  and result["official_complete"] >= baseline_held["official_complete"] - 0.10
                  and empty_runs == 0 and traffic_rate >= baseline_traffic - 0.10)
        note(f"- {held} 정상 주행 {'충족' if normal else '미달'}. "
             f"이동 중 평균 {result['moving_avg_speed_kmh']:.1f} km/h, "
             f"비정상 정지 {result['green_stop_time_mean']:.1%}, "
             f"반경 50 m 차량 관측 {traffic_rate:.1%}, 차량이 한 번도 없었던 판 {empty_runs}/100. "
             f"채택 기준 {'충족' if passed and normal and group != name else '미충족'}.")
        update()
    group, checkpoint, _ = finals[0]
    held_group = OUT / f"heldout_{group}"
    rows = sorted((json.loads(file.read_text(encoding="utf-8")) for file in held_group.glob("*.json")),
                  key=lambda row: row["seed"])
    failures = [row for row in rows if row["end_reason"] != "completed"][:3]
    successes = [row for row in rows if row["end_reason"] == "completed" and not row["inf_wait_count"]]
    for row in [*failures, *successes[:1]]:
        seed = row["seed"]
        with np.load(held_group / "traces" / f"{seed}.npz") as trace:
            event = float(trace["time"][-1])
            if row["end_reason"] == "completed":
                successful_dwell = np.flatnonzero(trace["stops_done"] >= 3)
                if len(successful_dwell):
                    event = float(trace["time"][successful_dwell[0]])
        folder = OUT / "gui" / group / str(seed)
        try:
            run_command(["scripts/gui_drive_seed.py", "--seed", seed, "--checkpoint", checkpoint,
                         "--delay", 0, "--screenshots", folder,
                         "--from-time", event - (12 if row["end_reason"] == "completed" else 3),
                         "--to-time", event + 1], f"gui_final_{seed}")
        except RuntimeError as exc:
            note(f"- 최종 후보 GUI 실패, 시드 {seed}. {exc}")
            continue
        note(f"- 최종 후보 GUI 시드 {seed}, {row['end_reason']}, SUMO {event:.1f}초. "
             f"`py -3.13 scripts/gui_drive_seed.py --seed {seed} --checkpoint {checkpoint} --delay 80`. "
             f"프레임 `{folder}`.")
        update()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    args = parser.parse_args()
    if args.pipeline:
        try:
            pipeline()
        except Exception as exc:
            note(f"## 실행 중단\n\n{type(exc).__name__}: {exc}")
            with REPORT.open("a", encoding="utf-8") as stream:
                stream.write(f"\n## 실행 중단\n\n{type(exc).__name__}: {exc}\n")
            raise
        return
    output = OUT / "smoke" if args.smoke else OUT
    destination = output / "report.md" if args.smoke else REPORT
    jobs = [(name, stage, model, 1) for name, stage, model, _ in JOBS[:1]] if args.smoke else JOBS
    report(output, jobs, destination)
    if args.report_only:
        return
    os.environ.update(OBS_SCAN="1", STOP_FIX="0")
    for name, stage, model, count in jobs:
        if not model.is_file():
            raise FileNotFoundError(model)
        os.environ["CGC_STAGE"] = str(stage)
        work, cfg = prepare_sumo()
        try:
            tasks = [(cfg, seed, model, output / name) for seed in range(900100, 900100 + count)]
            with get_context("spawn").Pool(min(4, count)) as pool:
                for _ in pool.imap_unordered(drive, tasks):
                    report(output, jobs, destination)
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()

