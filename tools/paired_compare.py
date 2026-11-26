"""같은 시드에서 기준 학생과 KL-PPO를 나란히 달린다.

    $env:CGC_STAGE = "4"
    py -3.13 tools/paired_compare.py --a <s4_student.pt> --b <kl.pt> --seeds 900130-900189 --shield shadow
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.drive_parallel import evaluate, format_table, parse_seeds  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="두 정책을 같은 판에서 비교")
    parser.add_argument("--a", required=True, help="기준 학생")
    parser.add_argument("--b", required=True, help="KL-PPO")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--shield", choices=["off", "shadow", "on"], default="shadow")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    rows_a, stats_a = evaluate(args.a, seeds, args.shield, args.workers, title="A 기준 학생")
    rows_b, stats_b = evaluate(args.b, seeds, args.shield, args.workers, title="B KL-PPO")
    by_a = {r["seed"]: r for r in rows_a}
    fixed, broken = [], []
    for r in rows_b:
        a = by_a[r["seed"]]
        a_ok = a["end_reason"] in ("completed", "missed_stop")
        b_ok = r["end_reason"] in ("completed", "missed_stop")
        if (not a_ok) and b_ok:
            fixed.append(r["seed"])
        if a_ok and (not b_ok):
            broken.append(r["seed"])
    print(format_table("A", stats_a))
    print(format_table("B", stats_b))
    print(f"BC 실패 → PPO 성공: {len(fixed)} {fixed}")
    print(f"BC 성공 → PPO 실패: {len(broken)} {broken}")
    print("실패 위치 (B에서 공식 완주가 아닌 판):")
    for r in rows_b:
        if r["end_reason"] not in ("completed", "missed_stop"):
            a = by_a[r["seed"]]
            print(f"  {r['seed']}  B {r['end_reason']} @{r['ego_edge']} {r['nearest_stop_id']}"
                  f"  | A {a['end_reason']} @{a['ego_edge']} {a['nearest_stop_id']}")


if __name__ == "__main__":
    main()
