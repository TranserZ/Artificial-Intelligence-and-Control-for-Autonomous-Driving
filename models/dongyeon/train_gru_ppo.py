"""GRU-PPO 학습 (강동연). 공용 train.py는 고치지 않고, 환경·보상·shield·검증 판 설정은 그대로 가져온다.

실행 (저장소 루트에서):
    python models/dongyeon/train_gru_ppo.py --seed 0 --total-steps 500000 \
        --out-dir models/dongyeon/checkpoints/ts0 --max-hours 11

베이스라인과 같은 것 (train.py에서 import): 보상 가중치 REWARD, shield off, 실패 판 다시 열기(0.3),
학습 판 시드(제외 범위), 검증 판 900000~900019(20판), 30만/50만 step 체크포인트,
PPO 하이퍼파라미터(lr 1e-4, n_steps 2048, epochs 10, γ 0.99, λ 0.95, clip 0.2, 차선 엔트로피 0.01,
초기 σ 0.61, 차선 유지 편향 0.5, gradient clip 0.5).

바꾼 것 (실험 축 = 네트워크에 기억 추가): config.json의 "model" 항목과 models/dongyeon/README.md 참고.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import train as base                                    # noqa: E402  공용 설정 (실행되지 않음, 정의만)
from env.road_config import ROAD, EGO, TRAFFIC          # noqa: E402
from env.curriculum import road_summary                 # noqa: E402
from env.mdp_config import OBSERVATION, ACTION          # noqa: E402
from env import road_builder                            # noqa: E402
from env.sumo_env import HELD_OUT_SEEDS, SumoHighwayEnv  # noqa: E402
from utils.logger import RunLogger                      # noqa: E402
from models.dongyeon.gru_ppo import GRUPPO, load_policy # noqa: E402
import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402

SIMULATION, REWARD = base.SIMULATION, base.REWARD       # train.py가 shield off·replay 0.3을 이미 넣어 둠

GRU_HPARAMS = dict(
    # ---- 베이스라인과 같은 값 ----
    lr=base.HPARAMS["lr"], n_steps=base.HPARAMS["n_steps"], n_epochs=base.HPARAMS["n_epochs"],
    gamma=base.HPARAMS["gamma"], gae_lambda=base.HPARAMS["gae_lambda"],
    clip_eps=base.HPARAMS["clip_eps"], value_coef=base.HPARAMS["value_coef"],
    entropy_coef=base.HPARAMS["entropy_coef"], lane_entropy_coef=base.HPARAMS["lane_entropy_coef"],
    max_grad_norm=base.HPARAMS["max_grad_norm"],
    normalize_advantage=base.HPARAMS["normalize_advantage"],
    init_log_std=base.HPARAMS["init_log_std"], lane_keep_bias=base.HPARAMS["lane_keep_bias"],
    hidden=base.HPARAMS["hidden_sizes"][0],     # 첫 층 256 (베이스라인 MLP 첫 층과 같음)
    # ---- 기억 관련 (새 값) ----
    gru_hidden=256,       # 둘째 층을 MLP 256 → GRU 256으로
    seq_len=64,           # truncated BPTT 길이 64 스텝 = 32초 (gradient만 끊김, 실행 중 기억은 판 전체)
    seqs_per_batch=2,     # 미니배치 = 64스텝 조각 2개 = 128스텝 → 업데이트당 16미니배치 × 10에폭
    prev_action=True,     # 입력에 직전 행동 2개 (Ni et al. 2022)
    seed=0, device="auto",
)
EVAL_INTERVAL = 50        # 50 업데이트(약 10만 step)마다 검증 20판. 베이스라인 25는 학습 시간의 절반을 평가에 씀
TRAIN_PURPOSE = "GRU-PPO (강동연): 베이스라인 PPO와 같은 환경·보상·하이퍼파라미터, 네트워크만 GRU로 기억 추가"


def best_score(m):
    """학습 중 best (검증 20판, 감시 모드). 공식 판정 기준에 맞춤.

    1) 자동 판정 통과(공식 완주율 ≥95%, 충돌률 <5%)한 것 중 수동개입률 낮은 것
    2) 아직 없으면 공식 완주율 → 충돌률 낮음 → 주행 거리(평균 속도 × 평균 길이) 순.
       공용 train.py는 완주 0%일 때 개입률이 가장 낮은 것(= 서 있는 모델)을 고르는 허점이 있다.
    """
    passed = m.success_rate >= 0.95 - 1e-9 and m.collision_rate < 0.05
    if passed:
        return (1, -m.intervention_rate, 0.0, 0.0)
    return (0, m.success_rate, -m.collision_rate, m.mean_speed * m.ep_length)


def write_config(run_dir, agent, *, seed, requested, status, tb, resume, started, init=None):
    cfg = {
        "schema_version": 2,
        "model": {
            "name": "dongyeon_gru_ppo",
            "axis": "알고리즘/네트워크 (MLP → GRU 기억)",
            "class": "models.dongyeon.gru_ppo.GRUPPO",
            "arch": agent.policy.arch,
            "hparams": GRU_HPARAMS,
            "eval_interval_updates": EVAL_INTERVAL,
            "best_rule": "자동 판정 통과 중 개입률 최저, 없으면 공식 완주율→충돌률→주행거리",
            "policy_factory": "models.dongyeon.eval_adapter:make_policy",
        },
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status,
        "wall_hours": round((time.time() - started) / 3600, 3),
        "runtime": base.runtime_versions(),
        "git": {
            "commit": base._git("rev-parse", "HEAD"),
            "exact_tag": base._git("describe", "--tags", "--exact-match"),
            "target_environment": base.TARGET_ENV,
            "target_is_ancestor": base._git_ok("merge-base", "--is-ancestor", base.TARGET_ENV, "HEAD"),
            "dirty": bool(base._git("status", "--porcelain", "--untracked-files=no")),
        },
        "training": {
            "seed": seed, "requested_steps": requested, "actual_steps": agent.num_timesteps,
            "resume_source": resume, "init_source": init, "save_steps": list(base.SAVE_STEPS),
            "validation_seeds": [base.EVAL_SEED_BASE, base.EVAL_SEED_BASE + base.EVAL_EPISODES - 1],
        },
        # 단계별 도로 (CGC_STAGE 반영). scripts/common_eval.py가 이 값으로 학습·평가 단계 일치를 검사한다.
        "road": road_summary(ROAD),
        "environment": {
            "held_out_seed_ranges": [list(r) for r in HELD_OUT_SEEDS],
            "simulation": SIMULATION, "observation": OBSERVATION, "action": ACTION, "reward": REWARD,
            "ego_sumo": {"speed_mode": 0, "lane_change_mode": 0, "time_to_teleport": -1,
                         "collision_action": "remove", "collision_check_junctions": True},
        },
        "artifacts": {"tensorboard_enabled": tb, "tensorboard_dir": "tb", "curves_dir": "curves",
                      "final_checkpoint": "final.pt", "final_resume_state": "final.resume.pt"},
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")


def main():
    p = argparse.ArgumentParser(description="GRU-PPO 학습")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=500_000, help="최종 누적 목표 step")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resume", default=None, help="*.resume.pt (optimizer·누적 step까지 복원)")
    p.add_argument("--init", default=None,
                   help="앞 단계 모델 .pt (가중치만 가져오고 optimizer·step은 새로 시작, 팀 단계별 학습 방식)")
    p.add_argument("--save-steps", default=None, help="기본 300000,500000")
    p.add_argument("--init-log-std", type=float, default=None,
                   help="--init 뒤 가감속 σ의 log를 이 값으로 (BC 모델은 σ를 학습하지 않으므로 탐험 크기를 정한다)")
    p.add_argument("--max-hours", type=float, default=None,
                   help="이 시간이 지나면 업데이트 경계에서 멈추고 model.resume.pt를 남긴다 (Kaggle 12시간 제한 대비)")
    p.add_argument("--eval-interval", type=int, default=None)
    args = p.parse_args()
    started = time.time()

    save_steps = (tuple(int(s) for s in args.save_steps.split(",") if s.strip())
                  if args.save_steps else base.SAVE_STEPS)
    base.SAVE_STEPS = save_steps
    eval_interval = EVAL_INTERVAL if args.eval_interval is None else args.eval_interval
    GRU_HPARAMS["seed"] = args.seed
    if args.init:   # BC 모델(직전 행동 입력 없음 등)에서 시작하면 그 구성을 따른다
        from models.dongyeon.gru_ppo import _torch_load
        arch = _torch_load(args.init, "cpu")["arch"]
        GRU_HPARAMS["prev_action"] = bool(arch.get("prev_action", True))

    work = tempfile.TemporaryDirectory(prefix=f"cheonggye_gru_{os.getpid()}_")
    cfg_path = road_builder.build(ROAD, EGO, TRAFFIC, work.name)
    run_dir = os.path.abspath(args.out_dir)
    os.makedirs(run_dir, exist_ok=True)
    logger = RunLogger(run_dir)
    env = SumoHighwayEnv(cfg_path=cfg_path, road=ROAD, ego=EGO, mdp_sim=SIMULATION,
                         mdp_obs=OBSERVATION, action=ACTION, reward=REWARD, traffic=TRAFFIC, gui=False)
    agent = GRUPPO(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0],
                   **GRU_HPARAMS)
    if args.resume and args.init:
        raise SystemExit("--resume과 --init은 함께 쓸 수 없습니다.")
    if args.init:
        agent.policy.load_state_dict(load_policy(args.init, device=str(agent.device)).state_dict())
        print(f"앞 단계 모델에서 시작: {args.init}", flush=True)
        if args.init_log_std is not None:
            with torch.no_grad():
                agent.policy.log_std.fill_(args.init_log_std)
            print(f"초기 σ = {np.exp(args.init_log_std):.2f}", flush=True)
    if args.resume:
        agent.load_training_state(args.resume)
        if agent.num_timesteps >= args.total_steps:
            raise SystemExit(f"이미 {agent.num_timesteps} step입니다. --total-steps를 더 크게 주세요.")
        print(f"이어 학습: {args.resume} ({agent.num_timesteps} → {args.total_steps})", flush=True)
    print(f"결과 폴더: {run_dir}\nobs_dim={agent.policy.obs_dim} arch={agent.policy.arch} "
          f"device={agent.device}", flush=True)

    paths = {k: os.path.join(run_dir, f"{k}.pt") for k in ("model", "final", "best_model")}
    best = {"score": None}
    tb = logger.writer is not None
    write_config(run_dir, agent, seed=args.seed, requested=args.total_steps, status="running",
                 tb=tb, resume=args.resume, started=started, init=args.init)

    def checkpoint():
        agent.save(paths["model"])
        agent.save_training_state(os.path.join(run_dir, "model.resume.pt"))
        logger.save_csv()
        write_config(run_dir, agent, seed=args.seed, requested=args.total_steps, status="running",
                     tb=tb, resume=args.resume, started=started, init=args.init)
        print(f"    [checkpoint @{agent.num_timesteps}]", flush=True)

    def on_eval(m):
        score = best_score(m)
        if best["score"] is None or score > best["score"]:
            best["score"] = score
            agent.save(paths["best_model"])
            print(f"    [best @{agent.num_timesteps}] 공식완주={m.success_rate:.0%} "
                  f"충돌={m.collision_rate:.0%} 개입={m.intervention_rate:.1%}", flush=True)

    def save_step(step):
        name = f"step{step // 1_000_000}m" if step % 1_000_000 == 0 else f"step{step // 1000}k"
        agent.save(os.path.join(run_dir, name + ".pt"))
        agent.save_training_state(os.path.join(run_dir, name + ".resume.pt"))
        print(f"    [step {step} 체크포인트 @{agent.num_timesteps}] {name}.pt", flush=True)

    def should_stop():
        return args.max_hours is not None and time.time() - started > args.max_hours * 3600

    status = "interrupted_or_failed"
    try:
        status = agent.learn(env, total_timesteps=args.total_steps, logger=logger, log_interval=1,
                             eval_interval=eval_interval, eval_episodes=base.EVAL_EPISODES,
                             checkpoint_interval=base.CHECKPOINT_INTERVAL, checkpoint_fn=checkpoint,
                             eval_fn=on_eval, eval_seed_base=base.EVAL_SEED_BASE,
                             save_steps=save_steps, save_fn=save_step, should_stop=should_stop)
    finally:
        env.close()
        work.cleanup()
        logger.close()
        agent.save(paths["model"])
        agent.save_training_state(os.path.join(run_dir, "model.resume.pt"))
        if status == "completed":
            agent.save(paths["final"])
            agent.save_training_state(os.path.join(run_dir, "final.resume.pt"))
        write_config(run_dir, agent, seed=args.seed, requested=args.total_steps, status=status,
                     tb=tb, resume=args.resume, started=started, init=args.init)
        print(f"\n끝: {status} @{agent.num_timesteps} step, {(time.time() - started) / 3600:.2f}시간",
              flush=True)


if __name__ == "__main__":
    main()
