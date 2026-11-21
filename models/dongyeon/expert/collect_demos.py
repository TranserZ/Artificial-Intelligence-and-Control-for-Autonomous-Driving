"""모방학습 시범 데이터 수집 (강동연): 규칙 선생(rule_expert.py)이 운전하며 (관측, 선생 행동)을 기록한다.

    CGC_STAGE=1 python models/dongyeon/expert/collect_demos.py --episodes 200 --out data/demos/stage1

설계 (자료조사 근거는 models/dongyeon/expert/README.md):
  - 판 번호: 학습용 범위 20000 + 단계×10000 + i. 테스트(1000~2999)·검증(900000~) 판과 겹치지 않는다
    (env/sumo_env.py HELD_OUT_SEEDS).
  - 관측은 행동을 고르기 "직전" 관측 (학생이 실제로 보게 될 것). 행동 라벨은 선생의 깨끗한 행동.
  - --noise σ > 0이면 실제로 실행하는 가감속에만 잡음을 더한다(라벨은 그대로). 선생이 평소 안 가는
    상태(조금 빠르거나 느린 상태)와 거기서 회복하는 행동이 데이터에 들어간다 (분포 이동 완화,
    DAgger 전 단계). 기본 0.
  - 실패한 판도 버리지 않고 기록하고, 판 표(episodes.csv)에 종료 사유를 남긴다. 학습 때 거른다.
  - 학생 GRU는 입력에 "직전 행동"을 받는다. 시범 데이터의 직전 행동은 선생 행동이라 학생이
    직전 행동을 베끼는 copycat 문제(Wen et al., NeurIPS 2020)가 생길 수 있어 prev_action을 따로 저장한다
    (학습 때 끄거나 잡음을 섞을 수 있게).

출력 (out 폴더):
  demos.npz      obs (N,67) float32, act (N,2) float32 [가감속, 차선 −1/0/+1], exec_act (N,2) 실제 실행,
                 prev_act (N,2), lane_cls (N,) int8 (0 오른쪽 1 유지 2 왼쪽 = 학생 차선 head 순서),
                 reward (N,), ep (N,) 판 번호(0부터), t (N,) 판 안 스텝, reason (N,) int8 (REASONS 순서)
  episodes.csv   판마다 seed, 단계, 길이, 종료 사유, 정차·놓침, 보상 합, 공식 완주·완벽 여부
  meta.json      단계·도로 요약·선생 버전·git 커밋·잡음·판 번호 범위
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from env.road_config import ROAD, EGO, TRAFFIC  # noqa: E402
from env.mdp_config import SIMULATION, OBSERVATION, ACTION, REWARD as REWARD_DEFAULTS  # noqa: E402
from env.curriculum import road_summary  # noqa: E402
from env import road_builder  # noqa: E402
from env.sumo_env import HELD_OUT_SEEDS, SumoHighwayEnv  # noqa: E402
from models.dongyeon.expert.rule_expert import RuleExpert  # noqa: E402
import traci  # noqa: E402

REASONS = ("cruise", "school_zone", "stop_approach", "dwell", "signal", "pedestrian", "overtake", "follow", "no_ego", "yield_side", "avoid_parked")


def train_reward():
    """train.py의 REWARD_OVERRIDES를 torch 없이 읽는다 (보상 합 기록용, 학습과 같은 보상)."""
    import ast
    tree = ast.parse(open(os.path.join(ROOT, "train.py"), encoding="utf-8").read())
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "REWARD_OVERRIDES":
            return {**REWARD_DEFAULTS, **{k.arg: ast.literal_eval(k.value) for k in n.value.keywords}}
    return dict(REWARD_DEFAULTS)


def end_reason(info, truncated):
    """scripts/common_eval.py get_end_reason과 같은 순서."""
    if info.get("collided"):
        return "collision"
    if info.get("red_violation"):
        return "red_light"
    if info.get("wrong_lane"):
        return "lane_error"
    if info.get("arrived"):
        return "missed_stop" if int(info.get("missed_stops") or 0) > 0 else "completed"
    return "timeout" if truncated else "other"


def held_out(seed):
    return any(a <= seed <= b for a, b in HELD_OUT_SEEDS)


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def main():
    p = argparse.ArgumentParser(description="규칙 선생 시범 데이터 수집")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed-base", type=int, default=None, help="기본 20000 + 단계×10000")
    p.add_argument("--noise", type=float, default=0.0, help="실행 가감속 잡음 σ (라벨은 깨끗한 행동)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    stage = ROAD.get("stage")
    base = args.seed_base if args.seed_base is not None else 20000 + int(stage or 0) * 10000
    seeds = list(range(base, base + args.episodes))
    bad = [s for s in seeds if held_out(s)]
    if bad:
        raise SystemExit(f"테스트·검증 판과 겹칩니다: {bad[:5]}")

    os.makedirs(args.out, exist_ok=True)
    work = tempfile.TemporaryDirectory(prefix="demos_")
    cfg = road_builder.build(ROAD, EGO, TRAFFIC, work.name)
    sim = {**SIMULATION, "shield": {**SIMULATION["shield"], "mode": "shadow"}}
    env = SumoHighwayEnv(cfg_path=cfg, road=ROAD, ego=EGO, mdp_sim=sim, mdp_obs=OBSERVATION, action=ACTION,
                         reward=train_reward(), traffic=TRAFFIC, gui=False)
    env.eval_mode, env.shield_mode = True, "shadow"   # 실패 판 다시 열기 끔, shield는 기록만
    expert = RuleExpert(env)
    rng = np.random.default_rng(base)

    cols = {k: [] for k in ("obs", "act", "exec_act", "prev_act", "lane_cls", "reward", "ep", "t", "reason")}
    episodes = []
    started = time.time()
    for i, seed in enumerate(seeds):
        obs, _ = env.reset(seed=seed)
        prev = np.zeros(2, dtype=np.float32)
        t, ret, shield_steps = 0, 0.0, 0
        while True:
            a = expert.predict(obs)
            ex = a.copy()
            if args.noise > 0:
                ex[0] = float(np.clip(ex[0] + rng.normal(0.0, args.noise), -1.0, 1.0))
            cols["obs"].append(np.asarray(obs, dtype=np.float32))
            cols["act"].append(a)
            cols["exec_act"].append(ex)
            cols["prev_act"].append(prev)
            cols["lane_cls"].append(int(round(a[1])) + 1)
            cols["ep"].append(i)
            cols["t"].append(t)
            cols["reason"].append(REASONS.index(expert.last_reason) if expert.last_reason in REASONS else -1)
            obs, r, term, trunc, info = env.step(ex)
            cols["reward"].append(float(r))
            shield_steps += int(bool(info.get("shield_active")))
            prev = ex
            t += 1
            ret += float(r)
            if term or trunc:
                break
        er = end_reason(info, trunc)
        episodes.append(dict(ep=i, seed=seed, stage=stage, steps=t, end_reason=er,
                             stops_done=info.get("stops_done"), missed_stops=info.get("missed_stops"),
                             ret=round(ret, 2), official=int(er in ("completed", "missed_stop")),
                             perfect=int(er == "completed"), shield_steps=shield_steps))
        print(f"[{i + 1}/{len(seeds)}] seed {seed} {er} {t} step 정차 {info.get('stops_done')} "
              f"놓침 {info.get('missed_stops')} ({(time.time() - started) / 60:.1f}분)", flush=True)
    env.close()

    np.savez_compressed(
        os.path.join(args.out, "demos.npz"),
        obs=np.stack(cols["obs"]), act=np.stack(cols["act"]), exec_act=np.stack(cols["exec_act"]),
        prev_act=np.stack(cols["prev_act"]), lane_cls=np.array(cols["lane_cls"], dtype=np.int8),
        reward=np.array(cols["reward"], dtype=np.float32), ep=np.array(cols["ep"], dtype=np.int32),
        t=np.array(cols["t"], dtype=np.int32), reason=np.array(cols["reason"], dtype=np.int8))
    with open(os.path.join(args.out, "episodes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(episodes[0]))
        w.writeheader()
        w.writerows(episodes)
    n = len(episodes)
    meta = dict(teacher=RuleExpert.version, stage=stage, road=road_summary(ROAD), noise=args.noise,
                seeds=[seeds[0], seeds[-1]], episodes=n, steps=len(cols["t"]),
                obs_dim=int(np.asarray(cols["obs"][0]).shape[0]), reasons=list(REASONS),
                official_rate=sum(e["official"] for e in episodes) / n,
                perfect_rate=sum(e["perfect"] for e in episodes) / n,
                git_commit=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
                wall_minutes=round((time.time() - started) / 60, 1))
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: meta[k] for k in ("stage", "episodes", "steps", "official_rate", "perfect_rate")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
