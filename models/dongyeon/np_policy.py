"""torch 없이 GRU-PPO 정책 실행 (numpy). torch가 없는 PC에서 화면 재생·확인용.

PyTorch nn.GRU 식 그대로 (게이트 순서 r, z, n):
    r = σ(W_ir x + b_ir + W_hr h + b_hr)
    z = σ(W_iz x + b_iz + W_hz h + b_hz)
    n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn))
    h' = (1 − z) ⊙ n + z ⊙ h
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptload  # noqa: E402

LANE_CMDS = (1.0, 0.0, -1.0)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class _Enc:
    def __init__(self, sd, p):
        self.W, self.b = sd[f"{p}.pre.weight"], sd[f"{p}.pre.bias"]
        self.Wi, self.Wh = sd[f"{p}.gru.weight_ih_l0"], sd[f"{p}.gru.weight_hh_l0"]
        self.bi, self.bh = sd[f"{p}.gru.bias_ih_l0"], sd[f"{p}.gru.bias_hh_l0"]
        self.H = self.Wh.shape[1]

    def step(self, x, h):
        z0 = np.maximum(self.W @ x + self.b, 0.0)
        gi, gh = self.Wi @ z0 + self.bi, self.Wh @ h + self.bh
        H = self.H
        r = _sig(gi[:H] + gh[:H])
        z = _sig(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        return (1.0 - z) * n + z * h


class NumpyGRUPolicy:
    """predict(obs) — 판이 시작되면 reset()."""

    def __init__(self, path):
        ck = ptload.load(path)
        if ck.get("format") != "gru_ppo_v1":
            raise ValueError(f"GRU-PPO 모델 파일이 아닙니다: {path}")
        self.arch, sd = ck["arch"], ck["state_dict"]
        self.actor, self.critic = _Enc(sd, "actor"), _Enc(sd, "critic")
        self.mu = (sd["mu_head.weight"], sd["mu_head.bias"])
        self.lane = (sd["lane_head.weight"], sd["lane_head.bias"])
        self.v = (sd["v_head.weight"], sd["v_head.bias"])
        self.obs_dim = self.arch["obs_dim"]
        self.reset()

    def reset(self):
        self.h_a = np.zeros(self.actor.H, np.float32)
        self.h_c = np.zeros(self.critic.H, np.float32)
        self.prev = np.zeros(2, np.float32)
        self.last = {}

    def predict(self, obs, deterministic=True):
        o = np.asarray(obs, np.float32)
        x = np.concatenate([o, self.prev]) if self.arch.get("prev_action", True) else o
        self.h_a = self.actor.step(x, self.h_a)
        self.h_c = self.critic.step(x, self.h_c)
        mu = float(np.tanh(self.mu[0] @ self.h_a + self.mu[1])[0])
        logits = self.lane[0] @ self.h_a + self.lane[1]
        p = np.exp(logits - logits.max()); p /= p.sum()
        idx = int(np.argmax(logits))
        a = np.array([mu, LANE_CMDS[idx]], np.float32)
        self.prev = a
        self.last = {"value": float(self.v[0] @ self.h_c + self.v[1]), "p_lane": p.tolist(),
                     "h_norm": float(np.linalg.norm(self.h_a))}
        return a
