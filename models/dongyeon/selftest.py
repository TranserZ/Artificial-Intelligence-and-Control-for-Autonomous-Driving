"""GRU-PPO 정확성 검사 (SUMO 없이, 가짜 환경). 저장소 루트에서: python models/dongyeon/selftest.py

1) rollout 때 한 스텝씩 계산한 log π·V == 학습 때 조각(저장한 은닉 상태 + 판 경계 리셋)으로 다시 계산한 값
2) 시간 초과: done으로 판을 끊고 γV(s')만 더함, GAE가 다음 판으로 새지 않음
3) 저장 → load_policy → 같은 출력, 이어 학습 상태 복원
4) 평가용 runner: env.reset 때 기억이 지워지고, 같은 판을 두 번 돌리면 같은 행동
5) 업데이트가 유한한 손실로 돌고 정책이 바뀜
"""
import os, sys, tempfile
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.dongyeon.gru_ppo import GRUPPO, RecurrentRunner, attach_reset, load_policy, _reset_stats

class Box:
    def __init__(s, n): s.shape = (n,)

class FakeEnv:
    """관측 67, 판 길이 37 스텝에서 시간 초과, 그 밖에 2% 확률로 종료."""
    def __init__(s, seed=0):
        s.observation_space, s.action_space = Box(67), Box(2)
        s.rng = np.random.default_rng(seed); s.t = 0
    def reset(s, seed=None):
        if seed is not None: s.rng = np.random.default_rng(seed)
        s.t = 0
        return s.rng.uniform(-1, 1, 67).astype(np.float32), {}
    def step(s, a):
        s.t += 1
        obs = s.rng.uniform(-1, 1, 67).astype(np.float32)
        term = bool(s.rng.random() < 0.02); trunc = (not term) and s.t >= 37
        return obs, float(a[0]) + s.rng.normal(), term, trunc, {"speed": 1.0}

def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond: raise SystemExit(1)

torch.manual_seed(0)
ag = GRUPPO(67, n_steps=256, seq_len=64, seqs_per_batch=2, n_epochs=2, hidden=32, gru_hidden=16, seed=0)
env = FakeEnv()
obs, _ = env.reset(); ag._new_episode_state()
stats = _reset_stats({})
obs, last_v, buf, fin = ag.collect_rollout(env, obs, stats)
check("rollout에 판 경계가 여러 번 있음", buf["start"].sum() >= 3 and buf["done"].sum() >= 3)

# 1) 조각 재계산 == rollout 값
L, S = ag.seq_len, ag.n_steps // ag.seq_len
def seq(a):
    t = torch.as_tensor(a); return t.view(S, L, *t.shape[1:]).transpose(0, 1)
with torch.no_grad():
    logp, ent, val, _, _ = ag.policy.evaluate_sequence(
        seq(buf["obs"]), seq(buf["prev"]), seq(buf["start"]),
        torch.as_tensor(buf["h_a"][::L]).unsqueeze(0), torch.as_tensor(buf["h_c"][::L]).unsqueeze(0), seq(buf["act"]))
check("log π 일치 (max |차| < 1e-4)", float((logp - seq(buf["logp"])).abs().max()) < 1e-4)
check("V 일치 (max |차| < 1e-4)", float((val - seq(buf["val"])).abs().max()) < 1e-4)
# 판 경계 바로 다음 스텝은 은닉 상태 0, 직전 행동 0
st = np.where(buf["start"] > 0)[0]
check("판 시작 스텝 은닉 상태·직전 행동 = 0", np.abs(buf["h_a"][st]).max() == 0 and np.abs(buf["prev"][st]).max() == 0)
check("판 시작 스텝은 직전 스텝이 done", all(buf["done"][i - 1] == 1 for i in st if i > 0))
# 리셋 마스크가 실제로 일을 하는지: 저장 은닉 상태 대신 조각 시작 은닉 상태만 주고 판 경계 리셋을 빼면 달라져야 함
with torch.no_grad():
    logp_nomask, *_ = ag.policy.evaluate_sequence(
        seq(buf["obs"]), seq(buf["prev"]), torch.zeros_like(seq(buf["start"])),
        torch.as_tensor(buf["h_a"][::L]).unsqueeze(0), torch.as_tensor(buf["h_c"][::L]).unsqueeze(0), seq(buf["act"]))
check("판 경계 리셋이 결과에 반영됨", float((logp_nomask - seq(buf["logp"])).abs().max()) > 1e-3)

# 2) 시간 초과 처리와 GAE 경계
adv, ret = ag.compute_gae(buf, last_v)
d = np.where(buf["done"] > 0)[0]
check("판 끝 스텝 advantage = r - V (다음 판으로 안 샘)",
      all(abs(adv[i] - (buf["rew"][i] - buf["val"][i])) < 1e-5 for i in d))

# 3) 저장·불러오기
tmp = tempfile.mkdtemp()
ag.save(os.path.join(tmp, "m.pt")); pol = load_policy(os.path.join(tmp, "m.pt"))
x = torch.randn(1, 67); pa = torch.zeros(1, 2); h = pol.initial_state(1)
with torch.no_grad():
    a1 = ag.policy.step(x, pa, *ag.policy.initial_state(1))[0]; a2 = pol.step(x, pa, *h)[0]
check("저장 후 같은 출력", torch.allclose(a1, a2))
ag.save_training_state(os.path.join(tmp, "r.pt"))
ag2 = GRUPPO(67, n_steps=256, seq_len=64, hidden=32, gru_hidden=16, seed=5)
ag2.load_training_state(os.path.join(tmp, "r.pt"))
check("이어 학습 상태 복원 (step·가중치)", ag2.num_timesteps == ag.num_timesteps and all(
    torch.equal(p, q) for p, q in zip(ag.policy.state_dict().values(), ag2.policy.state_dict().values())))

# 4) 평가용 runner
runner = RecurrentRunner(pol, torch.device("cpu"))
e2 = FakeEnv(); orig = attach_reset(e2, runner)
def episode(seed):
    o, _ = e2.reset(seed=seed); acts = []
    for _ in range(30):
        a = runner.predict(o); acts.append(a); o, *_ = e2.step(a)
    return np.array(acts)
A, B = episode(7), episode(7)
check("같은 판 두 번 → 같은 행동 (reset이 기억을 지움)", np.allclose(A, B))
check("기억이 행동에 영향 (스텝마다 다른 은닉 상태)", np.abs(np.diff(A[:, 0])).max() > 0)
e2.reset = orig

# 5) 업데이트
before = [p.clone() for p in ag.policy.parameters()]
out = ag.update(buf, adv, ret)
check("업데이트 손실 유한", all(np.isfinite(v) for k, v in out.items() if k != "explained_variance"))
check("업데이트 후 가중치 변화", any(not torch.equal(a, b) for a, b in zip(before, ag.policy.parameters())))
print("ALL PASS", out)
