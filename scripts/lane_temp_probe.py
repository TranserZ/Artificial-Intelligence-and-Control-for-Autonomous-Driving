"""τ 시험: 결정적 vs 차선 τ 샘플, 개발 30판."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.drive_parallel import evaluate, parse_seeds

INIT = os.path.join(ROOT, "results", "klppo1_s6_s1", "step200k.pt")
SEEDS = "900100-900129"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=INIT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    os.environ["CGC_STAGE"] = "6"
    os.environ["OBS_SCAN"] = "1"
    seeds = parse_seeds(SEEDS)
    det_rows, det = evaluate(args.model, seeds, workers=args.workers, title="det argmax",
                             json_out=os.path.join(ROOT, "results", "lane_temp_probe", "det.json"))
    det_coll = int(det["collision_count"])
    best_tau = 1.0
    rows_out = []
    for tau in (0.3, 0.5, 1.0):
        tau_rows, st = evaluate(
            args.model, seeds, workers=args.workers,
            title=f"lane τ={tau}", lane_temp=tau, stochastic_lane=True,
            json_out=os.path.join(ROOT, "results", "lane_temp_probe", f"tau_{tau}.json"),
        )
        extra = int(st["collision_count"]) - det_coll
        ok = extra <= 2
        rows_out.append((tau, st, extra, ok))
        if ok:
            best_tau = max(best_tau, tau)
    print(f"det 충돌 {det_coll}/30, 선택 τ={best_tau}")
    for tau, st, extra, ok in rows_out:
        n = st["episodes"]
        print(
            f"τ={tau}: 충돌 {st['collision_count']}/{n} (det 대비 +{extra}), "
            f"무한대기+timeout {st['inf_wait_episodes']}, 공식 {st['official_complete']:.1%}, "
            f"완벽 {st['perfect_complete']:.1%}, "
            f"차선변경/판 {sum(r.get('lane_changes', 0) for r in tau_rows)/max(n,1):.2f}"
        )
    with open(os.path.join(ROOT, "results", "lane_temp_probe", "chosen_tau.txt"), "w") as f:
        f.write(str(best_tau))


if __name__ == "__main__":
    main()