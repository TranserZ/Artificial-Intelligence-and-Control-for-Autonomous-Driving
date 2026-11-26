"""Run a short hotspot-focused KL-PPO trial after the existing SUMO pipeline."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.overnight_s6_final import OUT, ROOT, STUDENT, evaluate, note, run_command, training_args
from scripts.drive_parallel import summarize

ANCHOR = ROOT / "results/klppo3_s6_s0/step100k.pt"
NAME = "klppo3_s6_recovery"
STATUS = OUT / "recovery_status.md"


def record(message):
    with STATUS.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n\n")
    note(message)


def counts(name, start, count):
    rows = [json.loads((OUT / name / f"{seed}.json").read_text(encoding="utf-8"))
            for seed in range(start, start + count)]
    metrics = summarize(rows)
    if len(rows) != count:
        raise RuntimeError(f"Incomplete group {name}")
    return metrics


def score(stats):
    return (stats["inf_wait_episodes"], stats["timeout_count"],
            stats["collision_count"], stats["green_stop_time_mean"],
            -stats["moving_avg_speed_kmh"])


def show(name, stats):
    record(f"- {name}, 충돌 {stats['collision_count']}/{stats['episodes']}, "
           f"장기대기·시간초과 {stats['inf_wait_episodes']}/{stats['episodes']}, "
           f"이동 중 {stats['moving_avg_speed_kmh']:.1f} km/h, "
           f"비정상 정지 {stats['green_stop_time_mean']:.1%}.")


def normal(stats):
    return not any(stats[key] for key in ("inf_wait_episodes", "timeout_count",
                                          "wrong_lane_count", "sim_errors", "teleports")) and stats["moving_avg_speed_kmh"] >= 15.0


def run(after_pid):
    STATUS.write_text("# 6단계 주정차 구간 집중 재학습\n\n", encoding="utf-8")
    record("기존 SUMO 평가가 끝난 뒤 4,096스텝 스모크, 24,576스텝 KL-PPO를 차례로 실행한다. "
           "개발판의 완주율은 탈락 조건에서 제외한다.")
    while after_pid and subprocess.run(
            ["tasklist", "/FI", f"PID eq {after_pid}", "/FO", "CSV", "/NH"],
            capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW).stdout.find(
                f'","{after_pid}","'.encode()) >= 0:
        time.sleep(15)
    os.environ.update(CGC_STAGE="6", OBS_SCAN="1", STOP_FIX="0")
    smoke_name = NAME + "_smoke"
    args = training_args(ANCHOR, smoke_name, 0, smoke=True)
    args.extend(["--hotspot-recovery", "--replay-prob", 0.65])
    run_command(args, "recovery_smoke")
    with (ROOT / "results" / smoke_name / "training_log.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keys = ("hotspot_stall_penalty", "unsafe_accel_penalty")
    if len(rows) != 2 or not all(math.isfinite(float(row[key])) for row in rows for key in keys):
        raise RuntimeError("Recovery training smoke produced invalid statistics")
    record(f"- 스모크 통과. 보상 항 평균 최댓값 {dict((key, max(float(row[key]) for row in rows)) for key in keys)}.")
    args = training_args(ANCHOR, NAME, 0)
    args.extend(["--hotspot-recovery", "--replay-prob", 0.65, "--timesteps", 24576,
                 "--warmup-updates", 1, "--save-steps", "8000,16000,24000"])
    run_command(args, "recovery_train")
    results = []
    for step in (8, 16, 24):
        checkpoint = ROOT / "results" / NAME / f"step{step}k.pt"
        name = NAME + f"_step{step}k"
        for _ in evaluate(name, 6, checkpoint, 10):
            pass
        result = counts(name, 900100, 10)
        show(name, result)
        results.append((score(result), name, checkpoint, result))
    eligible = [item for item in results if normal(item[3]) and item[3]["collision_count"] <= 1]
    if not eligible:
        record("- 개발 10판에서 안전·정상 주행 조건을 둘 다 충족한 체크포인트가 없다.")
        return
    _, name, checkpoint, _ = min(eligible)
    for _ in evaluate(name, 6, checkpoint, 30):
        pass
    result = counts(name, 900100, 30)
    show(name, result)
    if not normal(result) or result["collision_count"] > 2:
        record("- 개발 30판 기준 미달. 독립 100판 시드는 사용하지 않았다.")
        return
    heldout = "heldout_" + name
    for _ in evaluate(heldout, 6, checkpoint, 100, start=900300):
        pass
    result = counts(heldout, 900300, 100)
    show(heldout, result)
    occupied = []
    import numpy as np
    for seed in range(900300, 900400):
        with np.load(OUT / heldout / "traces" / f"{seed}.npz") as trace:
            occupied.append(float(np.mean(trace["neighbors_50m"] > 0)))
    passed = normal(result) and result["collision_count"] <= 4 and min(occupied) > 0
    record(f"- 독립 100판 {'통과' if passed else '미달'}. "
           f"주변 차량 관측 스텝 비율 평균 {sum(occupied) / 100:.1%}, "
           f"주변 차량이 전혀 없던 판 {occupied.count(0)}/100.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-pid", type=int, default=0)
    try:
        run(parser.parse_args().after_pid)
    except Exception as error:
        record(f"- 재학습 실행 오류. {type(error).__name__} {error}")
        raise

