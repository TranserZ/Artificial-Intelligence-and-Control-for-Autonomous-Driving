"""scripts/drive_parallel.py 평가에 "신호 대기를 뺀 장기대기"를 덧붙여 잰다 (저장소 파일은 고치지 않는다).

    CGC_STAGE=5 OBS_SCAN=0 STOP_FIX=0 PLACE_FIX=0 CGC_THROUGH_VPH=400 \
      python eval_wait.py --repo ~/work/cgc-klppo --model results/klppo1_s5_s0/step250k.pt \
      --seeds 900300-900399 --workers 7 --json-out out.json

공식 장기대기(D): 60초 이상 연속 정지. 정지선 15 m 이내 적색 대기만 뺀다 (utils/official_metrics.py).
덧붙인 값: 60초 이상 연속 정지에서 아래 "신호 대기" 스텝을 뺀다.
  - 다음 신호가 적색/황색이고 정지선까지 15 m 이내, 또는
  - 다음 신호가 적색/황색이고 15 m 안에 앞차가 있음 (신호 대기열)
"""
import argparse
import json
import os
import shutil
import sys
from multiprocessing import get_context

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True)
p.add_argument("--model", required=True)
p.add_argument("--seeds", required=True)
p.add_argument("--workers", type=int, default=7)
p.add_argument("--json-out", required=True)
ARGS = p.parse_args()
REPO = os.path.abspath(os.path.expanduser(ARGS.repo))
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))


def _patch():
    import traci
    from utils import official_metrics as om
    if getattr(om.EpisodeMetrics, "_wait2_patched", False):
        return
    orig_obs, orig_fin, orig_row = (om.EpisodeMetrics.observe_step, om.EpisodeMetrics.finalize,
                                    om.EpisodeMetrics.row_dict)

    def state(self):
        return self.__dict__.setdefault("_w2", dict(run=0, runs=[], place="", t0=0.0, sig=0, stop=0))

    def close(self):
        st = state(self)
        if st["run"] >= om.INF_WAIT_STEPS:
            st["runs"].append(dict(sec=st["run"] * om.STEP_SEC, place=st["place"], start_time=st["t0"]))
        st["run"] = 0

    def observe(self, env, info, odo_before):
        st = state(self)
        try:
            present = env.ego_id in traci.vehicle.getIDList()
            speed = float(info["speed"]) if info.get("speed") is not None else 0.0
            if present and speed < om.ZERO_SPEED:
                st["stop"] += 1
                sig = env._next_signal()
                ry = sig is not None and sig[2] in ("r", "R", "y", "Y")
                near = ry and 0 <= sig[1] <= 15.0
                leader = traci.vehicle.getLeader(env.ego_id, 15.0)
                queued = ry and bool(leader and leader[0] and leader[1] <= 15.0)
                if near or queued:
                    st["sig"] += 1
                    close(self)
                else:
                    if st["run"] == 0:
                        st["place"] = "%s @ %.1f m" % (traci.vehicle.getRoadID(env.ego_id),
                                                       traci.vehicle.getLanePosition(env.ego_id))
                        st["t0"] = traci.simulation.getTime()
                    st["run"] += 1
            else:
                close(self)
        except traci.TraCIException:
            pass
        return orig_obs(self, env, info, odo_before)

    def finalize(self, *a, **kw):
        close(self)
        return orig_fin(self, *a, **kw)

    def row_dict(self, *a, **kw):
        row = orig_row(self, *a, **kw)
        st = state(self)
        row.update(nosig_wait_count=len(st["runs"]), nosig_wait_runs=st["runs"],
                   signal_wait_sec=st["sig"] * om.STEP_SEC, stopped_sec=st["stop"] * om.STEP_SEC)
        return row

    om.EpisodeMetrics.observe_step = observe
    om.EpisodeMetrics.finalize = finalize
    om.EpisodeMetrics.row_dict = row_dict
    om.EpisodeMetrics._wait2_patched = True


def run(job):
    _patch()
    import drive_parallel as dp
    return dp.run_seed(*job)


if __name__ == "__main__":
    import drive_parallel as dp
    seeds = dp.parse_seeds(ARGS.seeds)
    work, cfg = dp.prepare_sumo()
    try:
        jobs = [(cfg, s, ARGS.model, "shadow", True, 0.0, 0.0, False) for s in seeds]
        if ARGS.workers <= 1:
            rows = [run(j) for j in jobs]
        else:
            with get_context("spawn").Pool(ARGS.workers) as pool:
                rows = pool.map(run, jobs, chunksize=1)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    stats = dp.summarize(rows)
    n = len(rows)
    d_off = sum(1 for r in rows if r["inf_wait_count"] > 0)
    d_nosig = sum(1 for r in rows if r["nosig_wait_count"] > 0)
    dist = sum(r["distance_m"] for r in rows)
    secs = sum(r["sim_steps"] for r in rows) * 0.5
    extra = dict(episodes=n, official_inf_wait_episodes=d_off, nosig_wait_episodes=d_nosig,
                 timeout_episodes=sum(1 for r in rows if r["end_reason"] == "timeout"),
                 nosig_wait_or_timeout=sum(1 for r in rows if r["nosig_wait_count"] > 0 or r["end_reason"] == "timeout"),
                 pooled_avg_speed_kmh=dist / max(secs, 1e-9) * 3.6,
                 mean_moving_speed_kmh=sum(r["moving_avg_speed_kmh"] for r in rows) / n,
                 mean_avg_speed_kmh=sum(r["avg_speed_kmh"] for r in rows) / n,
                 signal_wait_share_of_stopped=sum(r["signal_wait_sec"] for r in rows) / max(sum(r["stopped_sec"] for r in rows), 1e-9))
    print(dp.format_table(ARGS.model, stats))
    print("EXTRA", json.dumps(extra, ensure_ascii=False))
    os.makedirs(os.path.dirname(os.path.abspath(ARGS.json_out)), exist_ok=True)
    json.dump(dict(model=ARGS.model, stats=stats, extra=extra, rows=rows),
              open(ARGS.json_out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
