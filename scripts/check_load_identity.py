"""업데이트 0회 KL-PPO와 s4_student를 같은 로더로 10판 비교한다.

둘 다 scripts/drive_parallel.py 의 load_eval_policy + policy.predict(결정적) 로 달린다.
도로 단계가 4가 아니면 바로 멈춘다.

    $env:CGC_STAGE = "4"
    py -3.13 scripts/check_load_identity.py --student <s4_student.pt> --seeds 900100-900109 --shield shadow
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def require_stage(student: str) -> None:
    stage_env = os.environ.get("CGC_STAGE")
    print(f"CGC_STAGE={stage_env!r}", flush=True)
    print(f"학생={os.path.abspath(student)}", flush=True)
    from env.road_config import ROAD
    stage = ROAD.get("stage")
    print(f"도로 단계={stage} 요소={list(ROAD.get('elements') or [])}", flush=True)
    if stage != 4:
        raise SystemExit(
            f"도로 단계가 4가 아닙니다 (CGC_STAGE={stage_env!r}, road.stage={stage}). 멈춥니다.")


def zero_update_checkpoint(student_path: str, dest: str, explore_std: float) -> None:
    """가중치를 한 번도 업데이트하지 않은 KL-PPO 체크포인트."""
    import torch
    from algorithms.kl_ppo import KLPPO, student_from_state_dict
    raw = torch.load(student_path, map_location="cpu", weights_only=False)
    student = student_from_state_dict(raw, "relu")
    agent = KLPPO(student, explore_std=explore_std, device="cpu", seed=0)
    agent.save(dest)


def main():
    parser = argparse.ArgumentParser(description="로드 직후 KL-PPO와 학생의 동일성")
    parser.add_argument("--student", required=True)
    parser.add_argument("--seeds", default="900100-900109")
    parser.add_argument("--shield", choices=["off", "shadow", "on"], default="shadow")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--explore-std", type=float, default=0.1,
                        help="체크포인트에 기록만 한다. 이 검사의 주행은 결정적이다")
    args = parser.parse_args()
    require_stage(args.student)
    from scripts.drive_parallel import parse_seeds, prepare_sumo, run_seed
    from multiprocessing import get_context
    import shutil
    seeds = parse_seeds(args.seeds)
    work, cfg = prepare_sumo()
    fd, ckpt = tempfile.mkstemp(prefix="kl_zero_", suffix=".pt")
    os.close(fd)
    try:
        zero_update_checkpoint(args.student, ckpt, args.explore_std)
        print(f"0회 업데이트 체크포인트={ckpt}", flush=True)
        print("로더=scripts.drive_parallel.load_eval_policy  행동=policy.predict(결정적)", flush=True)

        def drive(model):
            jobs = [(cfg, seed, model, args.shield, True, 0.0) for seed in seeds]
            if args.workers <= 1 or len(seeds) == 1:
                return [run_seed(*job) for job in jobs]
            with get_context("spawn").Pool(args.workers) as pool:
                return pool.starmap(run_seed, jobs)

        student_rows = drive(args.student)
        zero_rows = drive(ckpt)
        by_student = {r["seed"]: r for r in student_rows}
        bad = []
        for r in zero_rows:
            other = by_student[r["seed"]]
            same = (r["end_reason"] == other["end_reason"] and int(r["steps"]) == int(other["steps"]))
            mark = "일치" if same else "다름"
            print(f"seed={r['seed']} student {other['end_reason']} {other['steps']}  "
                  f"zero {r['end_reason']} {r['steps']}  {mark}", flush=True)
            if not same:
                bad.append(r["seed"])
        if bad:
            raise SystemExit(f"불일치 시드 {bad}")
        print(f"{len(seeds)}판 종료 사유·길이 일치.")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if os.path.isfile(ckpt):
            os.remove(ckpt)


if __name__ == "__main__":
    main()
