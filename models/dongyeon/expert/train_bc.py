"""모방학습 3단계: GRU 학생 행동 복제(BC) + critic 초기화 (강동연).

    python models/dongyeon/expert/train_bc.py --data <demos 폴더> [<폴더> ...] --out <결과 폴더> \
        [--init <앞 BC/PPO .pt>] --epochs 30 --seed 0

결과 폴더의 bc.pt는 GRU-PPO 모델 파일과 같은 형식(gru_ppo_v1)이다. 그래서
  - 공통 평가: --ckpt <폴더>/bc.pt --policy-factory models.dongyeon.eval_adapter:make_policy
  - PPO 이어 학습: train_gru_ppo.py --init <폴더>/bc.pt
  - DAgger 수집: collect_demos.py --student <폴더>/bc.pt
에 그대로 넣는다. config.json(도로 단계 포함)도 같이 쓴다 (공통 평가 단계 검사용).

손실 (자료조사 근거는 README.md)
  - 가감속: 선생 깨끗한 가감속과 mu(tanh)의 MSE. 분산(log_std)은 학습하지 않는다 —
    NLL로 배우면 σ가 0 가까이 줄어 PPO 이어 학습 때 탐험이 사라진다.
  - 차선: 3분류 교차 엔트로피 + 클래스 가중치. 차선 변경 라벨이 0.1~0.8%뿐이라 그대로 두면 "유지"만 배운다.
    가중치 = min(cap, sqrt(유지 수 / 변경 수)) (역빈도의 제곱근, 상한).
  - critic: 판별 할인 수익(γ=0.99, 학습 보상)으로 가치 회귀. PPO 첫 업데이트에서 엉망인 critic이
    BC 정책을 망치지 않게 한다. actor·critic GRU가 따로라 서로 gradient가 섞이지 않고,
    gradient clip도 따로 한다.
  - 직전 행동 입력: 기본은 데이터의 "직전에 실제 실행한 행동"(DAgger 데이터에서는 학생 자신의 행동).
    --prev zero는 copycat(Wen et al. 2020) 비교용.

학습은 판 전체를 한 번에 GRU에 넣는다 (판 길이 700~2000 step, 은닉 0에서 시작 = 실행 때와 같음).
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from models.dongyeon import train_gru_ppo as tg  # noqa: E402  (GRU_HPARAMS·write_config 재사용)
from models.dongyeon.gru_ppo import FORMAT, GRUActorCritic, load_policy  # noqa: E402


def load_episodes(dirs, gamma):
    """폴더들의 demos.npz → 판 목록. 판마다 obs·prev·accel·lane(학생 head 순서)·ret."""
    eps = []
    for d in dirs:
        z = np.load(os.path.join(d, "demos.npz"))
        ep = z["ep"]
        cuts = np.flatnonzero(np.diff(ep)) + 1
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(ep)]):
            r = z["reward"][a:b].astype(np.float64)
            ret = np.zeros_like(r)
            g = 0.0
            for t in range(len(r) - 1, -1, -1):
                g = r[t] + gamma * g
                ret[t] = g
            eps.append(dict(
                src=d, obs=z["obs"][a:b].astype(np.float32), prev=z["prev_act"][a:b].astype(np.float32),
                acc=np.clip(z["act"][a:b, 0], -1, 1).astype(np.float32),
                # 학생 차선 head: 0=왼쪽(+1) 1=유지 2=오른쪽(−1) (gru_ppo.LANE_CMDS)
                lane=(1 - np.round(z["act"][a:b, 1])).astype(np.int64),
                ret=ret.astype(np.float32)))
    return eps


def batches(eps, size, rng, shuffle=True):
    order = np.arange(len(eps))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), size):
        yield [eps[j] for j in order[i:i + size]]


def pad(group, device, prev_mode):
    T, B = max(len(e["acc"]) for e in group), len(group)
    D = group[0]["obs"].shape[1]
    obs = np.zeros((T, B, D), np.float32)
    prev = np.zeros((T, B, 2), np.float32)
    acc = np.zeros((T, B), np.float32)
    lane = np.ones((T, B), np.int64)
    ret = np.zeros((T, B), np.float32)
    mask = np.zeros((T, B), np.float32)
    for j, e in enumerate(group):
        n = len(e["acc"])
        obs[:n, j], acc[:n, j], lane[:n, j], ret[:n, j], mask[:n, j] = e["obs"], e["acc"], e["lane"], e["ret"], 1
        if prev_mode == "data":
            prev[:n, j] = e["prev"]
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    return t(obs), t(prev), t(acc), t(lane), t(ret), t(mask)


def forward(policy, obs, prev):
    """판 전체 (T,B,·) → mu (T,B), logits (T,B,3), value (T,B). 은닉 0에서 시작."""
    x = policy.inputs(obs, prev)
    za, _ = policy.actor.gru(torch.relu(policy.actor.pre(x)))
    zc, _ = policy.critic.gru(torch.relu(policy.critic.pre(x)))
    mu, logits, value = policy._heads(za, zc)
    return mu.squeeze(-1), logits, value


def run_epoch(policy, eps, opt, args, cw, device, rng, train):
    policy.train(train)
    s = dict(n=0.0, acc_mse=0.0, lane_ce=0.0, v_mse=0.0, n_change=0.0, hit=0.0, n_keep=0.0, false_change=0.0)
    for group in batches(eps, args.batch_eps, rng, shuffle=train):
        obs, prev, acc, lane, ret, mask = pad(group, device, args.prev)
        with torch.set_grad_enabled(train):
            mu, logits, value = forward(policy, obs, prev)
            m = mask.sum()
            acc_mse = (((mu - acc) ** 2) * mask).sum() / m
            ce = F.cross_entropy(logits.reshape(-1, 3), lane.reshape(-1), weight=cw, reduction="none")
            w = cw[lane.reshape(-1)] * mask.reshape(-1)
            lane_ce = (ce * mask.reshape(-1)).sum() / w.sum()
            v_mse = (((value - ret) ** 2) * mask).sum() / m
            loss = args.acc_coef * acc_mse + args.lane_coef * lane_ce + args.value_coef * v_mse
        if train:
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor_params(policy), args.max_grad_norm)
            nn.utils.clip_grad_norm_(critic_params(policy), args.max_grad_norm)
            opt.step()
        with torch.no_grad():
            pred = logits.argmax(-1)
            chg, keep = (lane != 1) & (mask > 0), (lane == 1) & (mask > 0)
            s["n"] += float(m)
            s["acc_mse"] += float(acc_mse) * float(m)
            s["lane_ce"] += float(lane_ce) * float(m)
            s["v_mse"] += float(v_mse) * float(m)
            s["n_change"] += float(chg.sum())
            s["hit"] += float(((pred == lane) & chg).sum())
            s["n_keep"] += float(keep.sum())
            s["false_change"] += float(((pred != 1) & keep).sum())
    n = max(s["n"], 1.0)
    return dict(acc_mse=s["acc_mse"] / n, lane_ce=s["lane_ce"] / n, v_mse=s["v_mse"] / n,
                change_recall=s["hit"] / max(s["n_change"], 1.0),
                false_change_rate=s["false_change"] / max(s["n_keep"], 1.0))


def actor_params(p):
    return list(p.actor.parameters()) + list(p.mu_head.parameters()) + list(p.lane_head.parameters())


def critic_params(p):
    return list(p.critic.parameters()) + list(p.v_head.parameters())


def score(v):
    """검증 판 기준 best: 가감속 MSE + 차선 손실 (가치 손실은 정책 품질과 무관하니 뺀다)."""
    return v["acc_mse"] + v["lane_ce"]


def main():
    p = argparse.ArgumentParser(description="GRU 학생 BC (규칙 선생 모방)")
    p.add_argument("--data", nargs="+", required=True, help="demos.npz가 든 폴더들 (선생 시범 + DAgger 수집)")
    p.add_argument("--out", required=True)
    p.add_argument("--init", default=None, help="이어서 학습할 .pt (DAgger 다음 회차, 앞 단계 모델)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-eps", type=int, default=16, help="미니배치 = 판 몇 개")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--acc-coef", type=float, default=1.0)
    p.add_argument("--lane-coef", type=float, default=1.0)
    p.add_argument("--value-coef", type=float, default=0.01, help="critic 손실 (actor와 파라미터가 따로라 크기만 조절)")
    p.add_argument("--lane-weight-cap", type=float, default=30.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--prev", choices=("data", "zero"), default="data")
    p.add_argument("--log-std", type=float, default=None, help="저장할 σ의 log (기본: 베이스라인 초기값 그대로)")
    args = p.parse_args()
    started = time.time()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gamma = tg.GRU_HPARAMS["gamma"]
    eps = load_episodes(args.data, gamma)
    idx = rng.permutation(len(eps))
    n_val = max(1, int(round(len(eps) * args.val_frac)))
    val = [eps[i] for i in idx[:n_val]]
    train = [eps[i] for i in idx[n_val:]]
    lanes = np.concatenate([e["lane"] for e in train])
    counts = np.bincount(lanes, minlength=3).astype(np.float64)
    w = np.minimum(args.lane_weight_cap, np.sqrt(counts[1] / np.maximum(counts, 1.0)))
    w[1] = 1.0
    cw = torch.as_tensor(w, dtype=torch.float32, device=device)
    print(f"판 {len(eps)} (학습 {len(train)} / 검증 {len(val)}), step {len(lanes)}, "
          f"차선 라벨 왼/유지/오른 = {counts.astype(int).tolist()}, 가중치 {np.round(w, 2).tolist()}, "
          f"device {device}", flush=True)

    h = tg.GRU_HPARAMS
    policy = GRUActorCritic(train[0]["obs"].shape[1], h["hidden"], h["gru_hidden"], h["init_log_std"],
                            h["lane_keep_bias"], h["prev_action"]).to(device)
    if args.init:
        policy.load_state_dict(load_policy(args.init, device=str(device)).state_dict())
        print(f"이어서: {args.init}", flush=True)
    params = actor_params(policy) + critic_params(policy)   # log_std 제외
    opt = torch.optim.Adam(params, lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    best, hist = None, []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(policy, train, opt, args, cw, device, rng, True)
        va = run_epoch(policy, val, opt, args, cw, device, rng, False)
        hist.append(dict(epoch=epoch, train=tr, val=va))
        tag = ""
        if best is None or score(va) < best:
            best = score(va)
            torch.save({"format": FORMAT, "arch": policy.arch, "state_dict": policy.state_dict()},
                       os.path.join(args.out, "bc_best_val.pt"))
            tag = " *"
        print(f"[{epoch}/{args.epochs}] 학습 acc {tr['acc_mse']:.4f} lane {tr['lane_ce']:.4f} | "
              f"검증 acc {va['acc_mse']:.4f} lane {va['lane_ce']:.4f} v {va['v_mse']:.2f} "
              f"변경 재현 {va['change_recall']:.0%} 오변경 {va['false_change_rate']:.3%} "
              f"({(time.time() - started) / 60:.1f}분){tag}", flush=True)

    # 마지막 에폭을 bc.pt로 (검증 손실 best는 bc_best_val.pt — 폐루프 성능은 공통 평가로 따로 본다)
    if args.log_std is not None:
        with torch.no_grad():
            policy.log_std.fill_(args.log_std)
    torch.save({"format": FORMAT, "arch": policy.arch, "state_dict": policy.state_dict()},
               os.path.join(args.out, "bc.pt"))
    agent = SimpleNamespace(policy=policy, num_timesteps=0)
    tg.write_config(args.out, agent, seed=args.seed, requested=0, status="bc", tb=False, resume=None,
                    started=started, init=args.init)
    cfg_path = os.path.join(args.out, "config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    cfg["model"]["name"] = "dongyeon_gru_bc"
    cfg["bc"] = dict(vars(args), lane_class_counts=counts.astype(int).tolist(), lane_weights=w.tolist(),
                     episodes=len(eps), steps=int(sum(len(e["acc"]) for e in eps)),
                     data_meta=[json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
                                if os.path.isfile(os.path.join(d, "meta.json")) else None for d in args.data],
                     history=hist)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
    print(f"저장: {args.out}/bc.pt ({(time.time() - started) / 60:.1f}분)", flush=True)


if __name__ == "__main__":
    main()
