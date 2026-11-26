"""KL-PPO 단위 검사. SUMO를 띄우지 않는다.

    py -3.13 scripts/test_kl_ppo.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from algorithms.kl_ppo import (  # noqa: E402
    KLPPO, adjust_beta, explained_variance, load_eval_policy, student_from_state_dict)
from env.sumo_env import SumoHighwayEnv  # noqa: E402
from train import HPARAMS, REWARD  # noqa: E402

STUDENT = r"C:\Users\USER\Desktop\청계로드-bc-d2ecd99\models\bc_ppo\s4_student.pt"
ROWS = []


def check(name, ok, detail=""):
    ROWS.append((name, bool(ok), detail))
    print(("통과" if ok else "실패"), name, detail)


def load_student():
    sd = torch.load(STUDENT, map_location="cpu", weights_only=False)
    return student_from_state_dict(sd, "relu"), sd


def fill(agent, obs, actions, logp, values, rewards):
    agent.buffer.clear()
    agent._lane_log_probs = []
    for i in range(len(obs)):
        agent.buffer.add(obs[i], actions[i], float(logp[i]), float(rewards[i]),
                         float(values[i]), False)
        agent._lane_log_probs.append(0.0)


def main():
    assert os.path.isfile(STUDENT), STUDENT
    student, _ = load_student()
    obs_dim = int(student.trunk[0].in_features)
    agent = KLPPO(student, explore_std=0.10, n_epochs=2, minibatch_size=32,
                  n_steps=32, device="cpu", seed=0)

    same_trunk = all(torch.equal(a, b) for a, b in zip(
        agent.critic.trunk.parameters(), student.trunk.parameters()))
    same_v = all(torch.equal(a, b) for a, b in zip(
        agent.critic.v_head.parameters(), student.v_head.parameters()))
    check("critic 초기값 = trunk+v_head", same_trunk and same_v)

    check("log_std 미학습", (not agent.policy.log_std.requires_grad)
          and abs(agent.policy.current_std - 0.10) < 1e-6)

    rng = np.random.default_rng(0)
    obs = rng.normal(size=(40, obs_dim)).astype(np.float32)
    match = 0
    accel_only = True
    for row in obs:
        action, log_p, _, _ = agent.act(row)
        with torch.no_grad():
            det = student.det_action(torch.as_tensor(row).unsqueeze(0))
            mu = agent.policy.accel_mu(torch.as_tensor(row).unsqueeze(0))
            std = torch.exp(agent.policy.log_std)
            expect = torch.distributions.Normal(mu, std).log_prob(
                torch.as_tensor(action[:1])).sum().item()
        if abs(float(action[1]) - float(det[0, 1])) < 1e-6:
            match += 1
        if abs(log_p - expect) > 1e-4:
            accel_only = False
    check("학습 롤아웃 차선 = s4_student argmax", match == len(obs),
          f"{match}/{len(obs)}")
    check("log_prob는 가속만", accel_only)

    obs_t = torch.as_tensor(obs[:32])
    with torch.no_grad():
        rms, kl = agent.policy.anchor_stats(obs_t, 0.079)
    check("로드 직후 RMS=0", float(rms) < 1e-5 and float(kl) < 1e-5,
          f"rms={float(rms):.3e}")

    actions = np.stack([agent.predict(row) for row in obs[:32]]).astype(np.float32)
    fill(agent, obs[:32], actions, np.zeros(32), np.zeros(32), rng.normal(size=32))
    before = {k: v.detach().clone() for k, v in agent.policy.state_dict().items()}
    warm = agent.update_critic_only(rng.normal(size=32).astype(np.float32))
    same = all(torch.equal(v, before[k]) for k, v in agent.policy.state_dict().items())
    critic_moved = not all(torch.equal(a, b) for a, b in zip(
        agent.critic.parameters(), student.v_head.parameters())) or True
    # v_head alone can stay close; trunk of critic should be allowed to move.
    moved = any(not torch.equal(a, b) for a, b in zip(
        agent.critic.parameters(),
        list(student.trunk.parameters()) + list(student.v_head.parameters())))
    check("warmup 중 정책 불변", same)
    check("warmup이 critic을 갱신", moved, f"ev {warm['ev_before']:.3f} → {warm['explained_variance']:.3f}")

    lane_before = agent.policy.lane_head.weight.detach().clone()
    advantages = rng.normal(size=32).astype(np.float32)
    returns = rng.normal(size=32).astype(np.float32)
    fill(agent, obs[:32], actions, np.zeros(32), np.zeros(32), np.zeros(32))
    agent.update(advantages, returns)
    check("가속 업데이트 뒤에도 차선 머리 고정",
          torch.equal(agent.policy.lane_head.weight, lane_before))
    match2 = 0
    for row in obs:
        action, _, _, _ = agent.act(row)
        with torch.no_grad():
            det = student.det_action(torch.as_tensor(row).unsqueeze(0))
        if abs(float(action[1]) - float(det[0, 1])) < 1e-6:
            match2 += 1
    check("업데이트 뒤 롤아웃 차선도 100% 일치", match2 == len(obs), f"{match2}/{len(obs)}")

    check("β 1.5배 초과면 2배", adjust_beta(0.1, 0.16, 0.1) == 0.2)
    check("β 1/1.5 미만이면 절반이고 0이 아님",
          adjust_beta(0.1, 0.05, 0.1) == 0.05 and adjust_beta(1e-8, 0.0, 0.1) > 0)
    check("β가 beta-min 아래로 안 내려감",
          adjust_beta(0.012, 0.05, 0.1) == 0.01
          and adjust_beta(0.01, 0.05, 0.1) == 0.01
          and adjust_beta(0.03, 0.05, 0.1, beta_min=0.02) == 0.02
          and adjust_beta(0.0, 0.05, 0.1) == 0.0)
    check("β 목표 안이면 유지", adjust_beta(0.1, 0.10, 0.1) == 0.1)
    off = KLPPO(student, 0.10, no_anchor=True, n_epochs=1, minibatch_size=32, device="cpu")
    check("--no-anchor 는 β=0", off.beta == 0.0)

    agent.target_kl = 0.02
    agent.n_epochs = 4
    fill(agent, obs[:32], actions, np.full(32, 5.0), np.zeros(32), np.zeros(32))
    stopped = agent.update(advantages, returns)
    check("가속 KL early stop", stopped["epochs"] == 1, f"epochs={stopped['epochs']}")

    check("explained_variance 정의", abs(explained_variance([1, 2, 3], [1, 2, 3]) - 1) < 1e-6)

    check("보상 블록 speed_weight 0", REWARD["speed_weight"] == 0.0
          and REWARD["missed_stop_penalty"] == 20.0)
    check("옵션 꺼짐: 기본 lr 그대로", HPARAMS["lr"] == 1e-4)
    check("예약 시드는 HELD_OUT과 별도",
          (not SumoHighwayEnv.is_held_out_seed(400050))
          and SumoHighwayEnv.is_bc_reserved_seed(400050)
          and SumoHighwayEnv.is_bc_reserved_seed(470010)
          and not SumoHighwayEnv.is_bc_reserved_seed(42))
    seq = [400050, 470010, 1500, 42]

    def fake_randint(lo, hi):
        return seq.pop(0)

    old = np.random.randint
    np.random.randint = fake_randint
    try:
        drawn = SumoHighwayEnv.sample_train_seed()
    finally:
        np.random.randint = old
    check("학습 판 샘플이 예약·홀드아웃을 건너뜀", drawn == 42, f"got {drawn}")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "kl.pt")
        fresh = KLPPO(student, explore_std=0.10, n_epochs=1, minibatch_size=32, device="cpu")
        fresh.save(path)
        loaded = load_eval_policy(path, obs_dim, 2, HPARAMS)
        from algorithms.ppo import PPO
        student_agent = PPO(obs_dim=obs_dim, act_dim=2, **{**HPARAMS, "device": "cpu", "seed": 0})
        student_agent.load(STUDENT)
        same_act = 0
        for row in obs:
            pred = np.asarray(loaded.predict(row, deterministic=True), dtype=np.float32)
            base = np.asarray(student_agent.predict(row, deterministic=True), dtype=np.float32)
            if pred.shape == (2,) and np.allclose(pred, base, atol=1e-5):
                same_act += 1
        check("0회 체크포인트를 같은 로더로 읽으면 s4_student 결정적 행동과 일치",
              same_act == len(obs), f"{same_act}/{len(obs)}")

    uf = KLPPO(student, 0.10, unfreeze_lane=True, lane_temp=0.5, lane_beta=1.0,
               lane_beta_min=0.1, n_epochs=1, minibatch_size=32, device="cpu")
    obs_t = torch.as_tensor(obs[:8])
    idx_cold = torch.argmax(uf.policy.lane_logits(obs_t), dim=-1)
    idx_tau = torch.argmax(uf.policy._tempered_logits(obs_t), dim=-1)
    check("τ로 argmax 불변", torch.equal(idx_cold, idx_tau))
    with torch.no_grad():
        lk0 = float(uf.policy.lane_kl_to_ref(obs_t).item())
    check("로드 직후 차선 KL≈0", lk0 < 1e-5, f"kl={lk0:.3e}")
    check("β_lane 하한",
          adjust_beta(0.15, 0.002, 0.01, beta_min=0.1) == 0.1)
    frozen = KLPPO(student, 0.10, unfreeze_lane=False, n_epochs=1, device="cpu")
    a1, lp1, llp1, _ = frozen.act(obs[0])
    a2, lp2, llp2, _ = uf.act(obs[0])
    check("옵션 끔=차선 argmax", abs(float(a1[1]) - float(
        student.det_action(torch.as_tensor(obs[0]).unsqueeze(0))[0, 1])) < 1e-6)
    check("unfreeze=차선 로그p 포함", abs(lp2 - lp1) > 1e-9 or abs(llp2) > 1e-9)

    print()
    print("| 검사 | 결과 |")
    print("|---|---|")
    for name, ok, detail in ROWS:
        tail = f" ({detail})" if detail else ""
        print(f"| {name} | {'통과' if ok else '실패'}{tail} |")
    failed = [name for name, ok, _ in ROWS if not ok]
    if failed:
        raise SystemExit("실패: " + ", ".join(failed))


if __name__ == "__main__":
    main()
