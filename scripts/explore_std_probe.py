"""가속 평균에 N(0, σ)만 얹는다. 로더와 행동은 scripts/drive_parallel.py 와 같다.

이 트리는 env/road_config.py 가 CGC_STAGE 를 읽지 않으면 단계가 4가 아니므로 바로 멈춘다.
4단계 비교는 단계를 읽는 작업 트리에서 돌린다.

    $env:CGC_STAGE = "4"
    py -3.13 scripts/explore_std_probe.py --model <s4_student.pt> --seeds 900100-900129 --stds 0.079,0.10,0.15 --shield shadow
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def parse_seeds(text: str):
    from scripts.drive_parallel import parse_seeds as _parse
    return _parse(text)


def require_stage(model: str) -> None:
    stage_env = os.environ.get("CGC_STAGE")
    print(f"CGC_STAGE={stage_env!r}", flush=True)
    print(f"모델={os.path.abspath(model)}", flush=True)
    from env.road_config import ROAD
    stage = ROAD.get("stage")
    elements = list(ROAD.get("elements") or [])
    print(f"도로 단계={stage} 요소={elements}", flush=True)
    if stage != 4:
        raise SystemExit(
            f"도로 단계가 4가 아닙니다 (CGC_STAGE={stage_env!r}, road.stage={stage}). 멈춥니다.")


def run_rows(cfg, seeds, model, shield, explore_std, workers):
    from scripts.drive_parallel import run_seed
    from multiprocessing import get_context
    jobs = [(cfg, seed, model, shield, True, float(explore_std)) for seed in seeds]
    if workers <= 1 or len(seeds) == 1:
        rows = [run_seed(*job) for job in jobs]
    else:
        with get_context("spawn").Pool(workers) as pool:
            rows = pool.starmap(run_seed, jobs)
    reasons = Counter(r["end_reason"] for r in rows)
    official = sum(1 for r in rows if r["end_reason"] in ("completed", "missed_stop"))
    return rows, reasons, official / max(len(rows), 1)


def main():
    parser = argparse.ArgumentParser(description="가속 σ 노이즈 폭 시험")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", default="900100-900129")
    parser.add_argument("--stds", default="0.079,0.10,0.15")
    parser.add_argument("--shield", choices=["off", "shadow", "on"], default="shadow")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    require_stage(args.model)
    seeds = parse_seeds(args.seeds)
    stds = [float(s) for s in args.stds.split(",") if s.strip()]
    from scripts.drive_parallel import prepare_sumo
    work, cfg = prepare_sumo()
    try:
        print(f"결정적 주행 {len(seeds)}판, 워커 {args.workers}", flush=True)
        _, det_reasons, det_rate = run_rows(cfg, seeds, args.model, args.shield, 0.0, args.workers)
        print(f"결정적 공식 완주 {det_rate:.1%}  {dict(det_reasons)}", flush=True)
        if det_rate < 0.90:
            print("기준 주행 이상 — 환경 또는 모델 확인 필요")
            return
        chosen = None
        print()
        print("| σ | 공식 완주 | 결정적과의 차이 | 종료 |")
        print("|---:|---:|---:|---|")
        print(f"| 결정적 | {det_rate:.1%} | 0 | {dict(det_reasons)} |")
        for std in stds:
            print(f"σ={std} 주행", flush=True)
            _, reasons, rate = run_rows(cfg, seeds, args.model, args.shield, std, args.workers)
            gap = rate - det_rate
            print(f"| {std:.3f} | {rate:.1%} | {gap:+.1%} | {dict(reasons)} |")
            if gap >= -0.05 - 1e-9:
                chosen = std
        print()
        if chosen is None:
            print("5%p 안에 남는 σ가 없습니다. 가장 작은 값보다 더 좁혀야 합니다.")
        else:
            print(f"고른 σ = {chosen:.3f} (5%p 이내인 것 중 가장 큼). "
                  f"log_std로 두면 {math.log(chosen):.3f}.")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
