"""collect_trajectories.py가 만든 summary.csv를 지점별 통과율 표로 집계한다.

사용법:
    py -3.13 scripts/analyze_checkpoints.py \
        data_huiyoung/trajectories/baseline_run_<timestamp>/summary.csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checkpoints import ALL_CODES, NAMES  # noqa: E402


def analyze(summary_path, out_path):
    with open(summary_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n == 0:
        raise ValueError(f"{summary_path}에 에피소드가 없습니다.")

    out_rows = []
    for code in ALL_CODES:
        n_pass = sum(int(r.get(f"pass_{code}", 0)) for r in rows)
        out_rows.append({
            "code": code, "name": NAMES[code],
            "n_episodes": n, "n_pass": n_pass,
            "pass_rate": round(n_pass / n, 4),
        })
    speeds = [float(r["school_zone_mean_speed"]) for r in rows
             if r.get("school_zone_mean_speed") not in (None, "", "nan")]
    school_zone_mean_speed = sum(speeds) / len(speeds) if speeds else float("nan")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["code", "name", "n_episodes", "n_pass", "pass_rate"])
        w.writeheader()
        w.writerows(out_rows)
        w.writerow({"code": "F_mean_speed", "name": "어린이보호구역 평균속도(m/s)",
                   "n_episodes": "", "n_pass": "", "pass_rate": round(school_zone_mean_speed, 3)
                   if school_zone_mean_speed == school_zone_mean_speed else "nan"})

    print(f"지점별 통과율 ({n} 에피소드):")
    for r in out_rows:
        print(f"  {r['code']} {r['name']:20s} {r['n_pass']:3d}/{n} = {r['pass_rate']:.0%}")
    print(f"  F 어린이보호구역 평균속도: {school_zone_mean_speed:.2f} m/s "
         f"(제한 8.33 m/s)")
    print(f"\n저장: {out_path}")
    return out_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("summary_csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data_huiyoung", "eval_logs", "checkpoint_analysis.csv")
    analyze(a.summary_csv, out)
