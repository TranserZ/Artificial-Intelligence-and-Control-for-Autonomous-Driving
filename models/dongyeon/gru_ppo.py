"""GRU-PPO — 기억이 있는 PPO (강동연 실험, 바꾸는 축: 알고리즘/네트워크).

베이스라인(공용 algorithms/ppo.py, MLP 256×256)과 같은 PPO 손실·하이퍼파라미터를 쓰고,
정책·가치 네트워크만 "한 스텝 관측 → 행동"에서 "지금까지의 관측 기록 → 행동"으로 바꾼다.

왜 기억이 필요한가 (부분 관측):
    env-v1부터 신호는 버스가 맨 앞 차일 때만 보인다. 뒤에 서 있으면 신호가 가려지고,
    보행자도 보이는 사람만, 신호 잔여시간도 없다. 한 스텝 관측만 보는 MLP는
    "방금 녹색으로 바뀌었다", "앞차가 서기 전에 신호가 무슨 색이었다", "여기서 얼마나 서 있었다"를 모른다.

설계 (근거)
  · GRU: POPGym(Morad et al., ICLR 2023)에서 13개 기억 모델 중 가장 좋은 범용 기억 모델.
  · 행동(actor)과 가치(critic)에 GRU를 따로 둔다: Ni et al.(ICML 2022) — 공유 인코더는
    critic gradient가 actor를 압도해 학습을 방해한다.
  · 입력 = 관측 67 + 직전 행동 2: Ni et al. — 과거 행동을 넣으면 숨은 상태 추정이 쉬워진다.
    직전 행동은 버스가 직접 낸 명령이라 새로운 외부 정보가 아니다. 보상은 넣지 않는다(운전 입력이 아님).
  · 학습: truncated BPTT. rollout을 길이 seq_len 조각으로 자르고, 각 조각 시작의 은닉 상태를
    rollout 때 저장해 두었다가 그대로 쓴다 (SB3-contrib RecurrentPPO, CleanRL ppo_lstm,
    MarcoMeter recurrent-ppo-truncated-bptt와 같은 방식). 판이 바뀌는 스텝에서는 은닉 상태를 0으로 만든다.
  · 실행할 때는 은닉 상태를 판 내내 이어 가므로 기억 길이는 seq_len으로 제한되지 않는다
    (gradient만 seq_len에서 끊긴다).

공용 PPO와 다른 점 (버그 수정 포함)
  · 시간 초과(truncated) 처리: 공용 PPO는 buffer에 done=terminated만 넣어서, 시간 초과 스텝에서
    γV(s')를 보상에 더한 뒤 다음 판 첫 상태의 V까지 한 번 더 더하고 GAE가 다음 판으로 새어 나간다.
    여기서는 done=terminated or truncated로 판을 끊고, 시간 초과는 γV(s')만 더한다.
"""
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

LANE_CMDS = (1.0, 0.0, -1.0)   # 차선 인덱스 0=왼쪽(+1), 1=유지(0), 2=오른쪽(-1) — 공용 hybrid와 같다
FORMAT = "gru_ppo_v1"


def _ortho(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class GRUEncoder(nn.Module):
    """[관측 + 직전 행동] → Linear+ReLU → GRU → 특징."""

    def __init__(self, in_dim, hidden, gru_hidden):
        super().__init__()
        self.pre = _ortho(nn.Linear(in_dim, hidden), np.sqrt(2))
        self.gru = nn.GRU(hidden, gru_hidden)            # 입력 (T, B, H)
        for name, p in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            else:
                nn.init.constant_(p, 0.0)

    def step(self, x, h):
        """한 스텝. x (B, in), h (1, B, Hg) → (특징 (B, Hg), 새 h)."""
        z = torch.relu(self.pre(x)).unsqueeze(0)
        out, h = self.gru(z, h)
        return out.squeeze(0), h

    def sequence(self, x, h0, starts):
        """조각 전체. x (T, B, in), h0 (1, B, Hg), starts (T, B) 1이면 그 스텝 전에 h=0.

        판이 바뀌는 지점마다 GRU를 끊어서 돌린다 (그 지점에서 해당 조각만 h를 0으로).
        조각 대부분은 판 경계가 없어 GRU 한 번으로 끝난다.
        """
        T = x.shape[0]
        z = torch.relu(self.pre(x))
        cut = sorted(set([0] + torch.nonzero((starts > 0.5).any(dim=1)).flatten().tolist() + [T]))
        outs, h = [], h0
        for a, b in zip(cut[:-1], cut[1:]):
            h = h * (1.0 - starts[a]).view(1, -1, 1)
            out, h = self.gru(z[a:b], h)
            outs.append(out)
        return torch.cat(outs, dim=0)                     # (T, B, Hg)


class GRUActorCritic(nn.Module):
    """가감속 Normal + 차선 Categorical(3) 정책, 가치 — actor와 critic이 GRU를 따로 가진다."""

    def __init__(self, obs_dim, hidden=256, gru_hidden=256, init_log_std=-0.5,
                 lane_keep_bias=0.5, prev_action=True):
        super().__init__()
        self.obs_dim, self.prev_action = obs_dim, prev_action
        in_dim = obs_dim + (2 if prev_action else 0)
        self.arch = dict(obs_dim=obs_dim, hidden=hidden, gru_hidden=gru_hidden,
                         init_log_std=float(init_log_std), lane_keep_bias=float(lane_keep_bias),
                         prev_action=bool(prev_action))
        self.actor = GRUEncoder(in_dim, hidden, gru_hidden)
        self.critic = GRUEncoder(in_dim, hidden, gru_hidden)
        self.mu_head = _ortho(nn.Linear(gru_hidden, 1), 0.01)
        self.lane_head = _ortho(nn.Linear(gru_hidden, 3), 0.01)
        self.v_head = _ortho(nn.Linear(gru_hidden, 1), 1.0)
        with torch.no_grad():
            self.lane_head.bias[1] = lane_keep_bias
        self.log_std = nn.Parameter(torch.ones(1) * float(init_log_std))
        self.gru_hidden = gru_hidden

    # ---- 입력·상태 ----------------------------------------------------
    def inputs(self, obs, prev_act):
        return torch.cat([obs, prev_act], dim=-1) if self.prev_action else obs

    def initial_state(self, batch=1, device="cpu"):
        z = torch.zeros(1, batch, self.gru_hidden, device=device)
        return z, z.clone()

    def _heads(self, za, zc):
        mu = torch.tanh(self.mu_head(za))
        return mu, self.lane_head(za), self.v_head(zc).squeeze(-1)

    # ---- 한 스텝 (rollout·실행) ----------------------------------------
    def step(self, obs, prev_act, h_a, h_c):
        x = self.inputs(obs, prev_act)
        za, h_a = self.actor.step(x, h_a)
        zc, h_c = self.critic.step(x, h_c)
        mu, logits, value = self._heads(za, zc)
        return mu, logits, value, h_a, h_c

    def value_step(self, obs, prev_act, h_c):
        zc, _ = self.critic.step(self.inputs(obs, prev_act), h_c)
        return self.v_head(zc).squeeze(-1)

    # ---- 조각 (학습) ---------------------------------------------------
    def evaluate_sequence(self, obs, prev_act, starts, h_a0, h_c0, actions):
        """obs (T,B,D), actions (T,B,2) → log_prob, 엔트로피, 가치, 차선 엔트로피, 변경 확률 (T,B)."""
        x = self.inputs(obs, prev_act)
        za = self.actor.sequence(x, h_a0, starts)
        zc = self.critic.sequence(x, h_c0, starts)
        mu, logits, value = self._heads(za, zc)
        std = torch.exp(self.log_std)
        nd, cd = Normal(mu, std), Categorical(logits=logits)
        lane_idx = (1.0 - torch.round(actions[..., 1])).long().clamp(0, 2)
        log_prob = nd.log_prob(actions[..., :1]).sum(-1) + cd.log_prob(lane_idx)
        lane_ent = cd.entropy()
        entropy = nd.entropy().sum(-1) + lane_ent
        p_change = 1.0 - torch.softmax(logits, dim=-1)[..., 1]
        return log_prob, entropy, value, lane_ent, p_change

    @property
    def current_std(self):
        return float(torch.exp(self.log_std).mean().item())


def lane_cmd_of(idx):
    return LANE_CMDS[int(idx)]


class RecurrentRunner:
    """실행·평가용 정책 (predict(obs, deterministic)). 판이 시작되면 reset()을 불러야 한다."""

    def __init__(self, policy, device):
        self.policy, self.device = policy, device
        self.reset()

    def reset(self):
        self.h_a, self.h_c = self.policy.initial_state(1, self.device)
        self.prev = torch.zeros(1, 2, device=self.device)

    @torch.no_grad()
    def predict(self, obs, deterministic=True):
        o = torch.as_tensor(np.asarray(obs, np.float32), device=self.device).unsqueeze(0)
        mu, logits, _, self.h_a, self.h_c = self.policy.step(o, self.prev, self.h_a, self.h_c)
        if deterministic:
            accel, idx = mu[0, 0], int(torch.argmax(logits[0]))
        else:
            accel = Normal(mu, torch.exp(self.policy.log_std)).sample()[0, 0]
            idx = int(Categorical(logits=logits).sample()[0])
        a = np.array([float(accel), lane_cmd_of(idx)], np.float32)
        self.prev = torch.as_tensor(a, device=self.device).unsqueeze(0)
        return a


def attach_reset(env, runner):
    """env.reset을 감싸서 판이 시작될 때마다 runner.reset()을 부른다. 원래 함수를 돌려준다.

    공용 평가 코드(utils/evaluator.py, scripts/common_eval.py)는 predict만 부르고 판 시작을
    알려 주지 않으므로, 공용 파일을 고치지 않고 판 경계를 잡는 방법이다.
    """
    original = env.reset

    def reset(*args, **kwargs):
        out = original(*args, **kwargs)
        runner.reset()
        return out

    env.reset = reset
    return original


# ======================================================================
class GRUPPO:
    def __init__(self, obs_dim, act_dim=2, lr=1e-4, n_steps=2048, n_epochs=10,
                 seq_len=64, seqs_per_batch=2, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 value_coef=0.5, entropy_coef=0.0, lane_entropy_coef=0.01, max_grad_norm=0.5,
                 normalize_advantage=True, hidden=256, gru_hidden=256, init_log_std=-0.5,
                 lane_keep_bias=0.5, prev_action=True, seed=0, device="auto"):
        assert act_dim == 2, "[가감속, 차선] 2차원 행동 전용"
        assert n_steps % seq_len == 0, "n_steps는 seq_len의 배수여야 한다"
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.policy = GRUActorCritic(obs_dim, hidden, gru_hidden, init_log_std,
                                     lane_keep_bias, prev_action).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.n_steps, self.n_epochs = n_steps, n_epochs
        self.seq_len, self.seqs_per_batch = seq_len, seqs_per_batch
        self.gamma, self.gae_lambda, self.clip_eps = gamma, gae_lambda, clip_eps
        self.value_coef, self.entropy_coef = value_coef, entropy_coef
        self.lane_entropy_coef, self.max_grad_norm = lane_entropy_coef, max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.num_timesteps = 0
        self.eval_runner = RecurrentRunner(self.policy, self.device)
        self._new_episode_state()

    # ---- 학습 중 상태 (판이 끝나면 0) ---------------------------------
    def _new_episode_state(self):
        self.h_a, self.h_c = self.policy.initial_state(1, self.device)
        self.prev = torch.zeros(1, 2, device=self.device)
        self.ep_start = True

    def predict(self, obs, deterministic=True):
        """utils/evaluator.py 인터페이스. 판 경계는 attach_reset으로 잡는다."""
        return self.eval_runner.predict(obs, deterministic)

    # ---- 1) rollout ----------------------------------------------------
    @torch.no_grad()
    def collect_rollout(self, env, obs, ep_stats, logger=None):
        N, dev = self.n_steps, self.device
        D = self.policy.obs_dim
        G = self.policy.gru_hidden
        buf = dict(obs=np.zeros((N, D), np.float32), prev=np.zeros((N, 2), np.float32),
                   act=np.zeros((N, 2), np.float32), logp=np.zeros(N, np.float32),
                   rew=np.zeros(N, np.float32), val=np.zeros(N, np.float32),
                   done=np.zeros(N, np.float32), start=np.zeros(N, np.float32),
                   h_a=np.zeros((N, G), np.float32), h_c=np.zeros((N, G), np.float32))
        finished = []
        for t in range(N):
            buf["start"][t] = float(self.ep_start)
            buf["h_a"][t], buf["h_c"][t] = self.h_a[0, 0].cpu().numpy(), self.h_c[0, 0].cpu().numpy()
            buf["prev"][t] = self.prev[0].cpu().numpy()
            buf["obs"][t] = obs
            o = torch.as_tensor(np.asarray(obs, np.float32), device=dev).unsqueeze(0)
            mu, logits, value, self.h_a, self.h_c = self.policy.step(o, self.prev, self.h_a, self.h_c)
            nd, cd = Normal(mu, torch.exp(self.policy.log_std)), Categorical(logits=logits)
            accel = nd.sample()
            idx = cd.sample()
            logp = nd.log_prob(accel).sum(-1) + cd.log_prob(idx)
            action = np.array([float(accel[0, 0]), lane_cmd_of(idx[0])], np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action)
            self.num_timesteps += 1
            self.ep_start = False
            self.prev = torch.as_tensor(action, device=dev).unsqueeze(0)

            _accumulate(ep_stats, reward, info)
            stored = float(reward)
            if truncated and not terminated:
                # 시간 초과는 MDP 종료가 아니다: 이어졌다면 받았을 가치를 부트스트랩
                no = torch.as_tensor(np.asarray(next_obs, np.float32), device=dev).unsqueeze(0)
                stored += self.gamma * float(self.policy.value_step(no, self.prev, self.h_c)[0])
            buf["act"][t], buf["logp"][t], buf["val"][t] = action, float(logp[0]), float(value[0])
            buf["rew"][t], buf["done"][t] = stored, float(terminated or truncated)

            if terminated or truncated:
                finished.append(_finish(ep_stats, truncated and not terminated))
                if logger is not None:
                    logger.log_episode(self.num_timesteps, ep_stats["return"], int(ep_stats["length"]))
                _reset_stats(ep_stats)
                obs, _ = env.reset()
                self._new_episode_state()
            else:
                obs = next_obs

        # rollout 끝 상태 가치 (은닉 상태는 진행시키지 않음 — 다음 rollout이 같은 상태에서 이어 감)
        o = torch.as_tensor(np.asarray(obs, np.float32), device=dev).unsqueeze(0)
        last_value = float(self.policy.value_step(o, self.prev, self.h_c)[0])
        return obs, last_value, buf, finished

    # ---- 2) GAE ---------------------------------------------------------
    def compute_gae(self, buf, last_value):
        rew, val, done = buf["rew"], buf["val"], buf["done"]
        N = len(rew)
        adv = np.zeros(N, np.float32)
        gae = 0.0
        for t in reversed(range(N)):
            next_v = last_value if t == N - 1 else val[t + 1]
            nonterm = 1.0 - done[t]          # 판이 끝났으면(충돌·도착·시간 초과) 다음 판으로 넘기지 않음
            delta = rew[t] + self.gamma * next_v * nonterm - val[t]
            gae = delta + self.gamma * self.gae_lambda * nonterm * gae
            adv[t] = gae
        return adv, adv + val

    # ---- 3) 업데이트 ----------------------------------------------------
    def update(self, buf, adv, ret):
        dev, L = self.device, self.seq_len
        S = self.n_steps // L
        def seq(a):     # (N, ...) → (L, S, ...): 조각 s = 스텝 [s*L, (s+1)*L)
            t = torch.as_tensor(a, device=dev)
            return t.view(S, L, *t.shape[1:]).transpose(0, 1).contiguous()
        obs, prev, act = seq(buf["obs"]), seq(buf["prev"]), seq(buf["act"])
        old_logp, starts = seq(buf["logp"]), seq(buf["start"])
        adv_t, ret_t = seq(adv), seq(ret)
        if self.normalize_advantage:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        h_a0 = torch.as_tensor(buf["h_a"][::L], device=dev).unsqueeze(0)   # (1, S, G)
        h_c0 = torch.as_tensor(buf["h_c"][::L], device=dev).unsqueeze(0)

        stats = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, approx_kl=0.0, clip_fraction=0.0)
        nb = 0
        for _ in range(self.n_epochs):
            order = np.random.permutation(S)
            for i in range(0, S, self.seqs_per_batch):
                mb = torch.as_tensor(order[i:i + self.seqs_per_batch], device=dev)
                logp, ent, value, lane_ent, _ = self.policy.evaluate_sequence(
                    obs[:, mb], prev[:, mb], starts[:, mb], h_a0[:, mb], h_c0[:, mb], act[:, mb])
                ratio = torch.exp(logp - old_logp[:, mb])
                a = adv_t[:, mb]
                policy_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - self.clip_eps,
                                                                1 + self.clip_eps) * a).mean()
                value_loss = ((value - ret_t[:, mb]) ** 2).mean()
                loss = (policy_loss + self.value_coef * value_loss
                        - self.entropy_coef * ent.mean() - self.lane_entropy_coef * lane_ent.mean())
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    stats["policy_loss"] += policy_loss.item()
                    stats["value_loss"] += value_loss.item()
                    stats["entropy"] += ent.mean().item()
                    stats["approx_kl"] += (old_logp[:, mb] - logp).mean().item()
                    stats["clip_fraction"] += ((ratio - 1).abs() > self.clip_eps).float().mean().item()
                nb += 1
        out = {k: v / max(nb, 1) for k, v in stats.items()}
        with torch.no_grad():
            _, _, value, _, p_change = self.policy.evaluate_sequence(obs, prev, starts, h_a0, h_c0, act)
            var = torch.var(ret_t)
            out["explained_variance"] = float(1 - torch.var(ret_t - value) / var) if var > 1e-12 else float("nan")
            out["p_lane_change"] = float(p_change.mean())
        out["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        return out

    # ---- 메인 루프 -------------------------------------------------------
    def learn(self, env, total_timesteps, logger=None, log_interval=1, eval_interval=0,
              eval_episodes=20, checkpoint_interval=0, checkpoint_fn=None, eval_fn=None,
              eval_seed_base=None, save_steps=(), save_fn=None, should_stop=None):
        """should_stop(): True면 업데이트 경계에서 멈추고 "time_limit"를 돌려준다."""
        from utils.evaluator import evaluate_policy
        pending = sorted(int(s) for s in save_steps if int(s) > self.num_timesteps)
        obs, _ = env.reset()
        self._new_episode_state()
        ep_stats = _reset_stats({})
        recent = deque(maxlen=100)
        n_updates = 0
        while self.num_timesteps < total_timesteps:
            obs, last_value, buf, finished = self.collect_rollout(env, obs, ep_stats, logger)
            recent.extend(finished)
            adv, ret = self.compute_gae(buf, last_value)
            stats = self.update(buf, adv, ret)
            n_updates += 1

            def rm(k):
                return float(np.mean([e[k] for e in recent])) if recent else float("nan")
            metrics = dict(update=n_updates, ep_rew_mean=rm("return"), ep_len_mean=rm("length"),
                           completion_rate=rm("completed"), official_completion_rate=rm("official"),
                           collision_rate=rm("collided"), red_light_rate=rm("red_light"),
                           lane_error_rate=rm("lane_error"), timeout_rate=rm("timeout"),
                           missed_stop_rate=rm("missed_stop"), avg_speed=rm("avg_speed"),
                           lane_changes_per_episode=rm("lane_changes"), std=self.policy.current_std,
                           **stats)
            if logger is not None:
                logger.log_train(self.num_timesteps, metrics, verbose=(n_updates % log_interval == 0))

            while pending and self.num_timesteps >= pending[0]:
                if save_fn is not None:
                    save_fn(pending[0])
                pending.pop(0)

            if eval_interval > 0 and n_updates % eval_interval == 0:
                original = attach_reset(env, self.eval_runner)
                try:
                    m = evaluate_policy(self, env, n_episodes=eval_episodes,
                                        seed_base=eval_seed_base, shield_mode="shadow")
                finally:
                    env.reset = original
                if logger is not None:
                    logger.log_eval(self.num_timesteps, m.as_dict(), m.summary(), eval_episodes)
                if eval_fn is not None:
                    eval_fn(m)
                obs, _ = env.reset()          # 평가가 학습 판을 끊었으므로 새 판
                self._new_episode_state()
                _reset_stats(ep_stats)

            if checkpoint_fn is not None and checkpoint_interval > 0 and n_updates % checkpoint_interval == 0:
                checkpoint_fn()
            if should_stop is not None and self.num_timesteps < total_timesteps and should_stop():
                return "time_limit"
        return "completed"

    # ---- 저장 ------------------------------------------------------------
    def save(self, path):
        torch.save({"format": FORMAT, "arch": self.policy.arch,
                    "state_dict": self.policy.state_dict()}, path)

    def save_training_state(self, path):
        state = {"format": FORMAT + "_train", "arch": self.policy.arch,
                 "policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "num_timesteps": self.num_timesteps, "numpy_rng_state": np.random.get_state(),
                 "torch_rng_state": torch.get_rng_state()}
        torch.save(state, path)

    def load_training_state(self, path):
        state = _torch_load(path, self.device)
        if state.get("format") != FORMAT + "_train":
            raise ValueError(f"GRU-PPO 이어 학습 파일이 아닙니다: {path}")
        if state["arch"] != self.policy.arch:
            raise ValueError(f"네트워크 구성이 다릅니다: {state['arch']} ≠ {self.policy.arch}")
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.num_timesteps = int(state["num_timesteps"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_policy(path, device="cpu"):
    """save()한 .pt → GRUActorCritic (평가용)."""
    ck = _torch_load(path, torch.device(device))
    if ck.get("format") != FORMAT:
        raise ValueError(f"GRU-PPO 모델 파일이 아닙니다: {path}")
    policy = GRUActorCritic(**ck["arch"]).to(device)
    policy.load_state_dict(ck["state_dict"])
    policy.eval()
    return policy


# ---- 판 통계 (공용 PPO와 같은 항목 + 공식 완주) -------------------------
def _reset_stats(s):
    s.update({"return": 0.0, "length": 0, "collided": False, "red_light": False,
              "lane_error": False, "arrived": False, "missed_stops": 0, "lane_changes": 0,
              "speed_sum": 0.0, "speed_count": 0})
    return s


def _accumulate(s, reward, info):
    s["return"] += float(reward)
    s["length"] += 1
    s["collided"] |= bool(info.get("collided"))
    s["red_light"] |= bool(info.get("red_violation"))
    s["lane_error"] |= bool(info.get("wrong_lane"))
    s["arrived"] |= bool(info.get("arrived"))
    s["missed_stops"] = int(info.get("missed_stops", 0) or 0)
    s["lane_changes"] += int(info.get("lane_change", 0) != 0 and bool(info.get("lane_change_applied")))
    if info.get("speed") is not None:
        s["speed_sum"] += float(info["speed"])
        s["speed_count"] += 1


def _finish(s, timeout):
    official = bool(s["arrived"] and not (s["collided"] or s["red_light"] or s["lane_error"]))
    return {**s, "completed": bool(official and s["missed_stops"] == 0), "official": official,
            "missed_stop": bool(s["missed_stops"] > 0), "timeout": bool(timeout),
            "avg_speed": s["speed_sum"] / s["speed_count"] if s["speed_count"] else 0.0}
