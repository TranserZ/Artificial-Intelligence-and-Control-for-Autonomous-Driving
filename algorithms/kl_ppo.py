"""가속만 고치는 KL-PPO.

차선 행동은 학습 롤아웃에서도 차선 갈래의 argmax다.
PPO 확률 비율, 직전 정책 KL early stop, 기준 묶음 벌점은 가속만 쓴다.
--unfreeze-lane일 때만 차선 분포에 별도의 PPO 항을 더한다.
기준 묶음을 끄면(--no-anchor) 그 벌점 계수는 0이다.
"""
from __future__ import annotations

import math
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from utils.buffer import RolloutBuffer
from utils.networks import AccelSplitPolicy, HybridActorCritic, SplitCritic


ANCHOR_STD = 0.079
ANCHOR_TARGET_RMS = 0.1
BETA_MIN = 0.01


def explained_variance(y_pred, y_true) -> float:
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    var_y = float(np.var(true))
    if var_y < 1e-8:
        return float("nan")
    return float(1.0 - np.var(true - pred) / var_y)


def adjust_beta(beta: float, rms: float, target: float,
                beta_min: float = BETA_MIN) -> float:
    """RMS가 목표의 1.5배를 넘으면 2배, 1/1.5보다 작으면 절반. 절반은 beta_min 아래로 내리지 않는다."""
    beta = float(beta)
    floor = float(beta_min)
    if beta <= 0:
        return beta
    if rms > target * 1.5:
        return beta * 2.0
    if rms < target / 1.5:
        return max(beta / 2.0, floor)
    return beta


def _hidden_sizes(state_dict):
    h1 = int(state_dict["trunk.0.weight"].shape[0])
    h2 = int(state_dict["trunk.2.weight"].shape[0])
    return (h1, h2)


def student_from_state_dict(state_dict, activation: str = "relu") -> HybridActorCritic:
    obs_dim = int(state_dict["trunk.0.weight"].shape[1])
    student = HybridActorCritic(
        obs_dim, hidden_sizes=_hidden_sizes(state_dict),
        activation=activation, init_log_std=-0.5)
    student.load_state_dict(state_dict)
    student.eval()
    return student


class KLPPO:
    """별도 critic + 가속 PPO + 평균 차이 제곱 벌점."""

    def __init__(
        self,
        student: HybridActorCritic,
        explore_std: float,
        *,
        anchor_std: float = ANCHOR_STD,
        anchor_target_rms: float = ANCHOR_TARGET_RMS,
        beta: float = 0.1,
        beta_min: float = BETA_MIN,
        no_anchor: bool = False,
        unfreeze_lane: bool = False,
        lane_temp: float = 1.0,
        lane_kl_target: float = 0.01,
        lane_beta: float = 0.0,
        lane_beta_min: float = 0.1,
        policy_lr: float = 1e-5,
        critic_lr: float = 1e-4,
        n_steps: int = 2048,
        n_epochs: int = 10,
        minibatch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = True,
        target_kl: float = 0.02,
        warmup_max_updates: int = 40,
        warmup_ev: float = 0.8,
        seed: int = 0,
        device: str = "cpu",
        buffer: RolloutBuffer = None,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.policy = AccelSplitPolicy(
            student, explore_std, unfreeze_lane=unfreeze_lane,
            lane_temp=lane_temp).to(self.device)
        self.critic = SplitCritic(student).to(self.device)
        self.policy_optim = torch.optim.Adam(
            [p for p in self.policy.parameters() if p.requires_grad], lr=policy_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.explore_std = float(explore_std)
        self.anchor_std = float(anchor_std)
        self.anchor_target_rms = float(anchor_target_rms)
        self.no_anchor = bool(no_anchor)
        self.unfreeze_lane = bool(unfreeze_lane)
        self.lane_increases_only = False
        self.lane_kl_target = float(lane_kl_target)
        self.lane_beta_min = float(lane_beta_min)
        self.lane_beta = float(lane_beta) if self.unfreeze_lane else 0.0
        self.beta_min = float(beta_min)
        self.beta = 0.0 if self.no_anchor else float(beta)
        self.n_steps = int(n_steps)
        self.n_epochs = int(n_epochs)
        self.minibatch_size = int(minibatch_size)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.normalize_advantage = bool(normalize_advantage)
        self.target_kl = float(target_kl)
        self.warmup_max_updates = int(warmup_max_updates)
        self.warmup_ev = float(warmup_ev)
        self.buffer = buffer if buffer is not None else RolloutBuffer()
        self._lane_log_probs = []
        self.num_timesteps = 0
        self.warmup_done = False
        self.warmup_updates = 0
        self.ev_start = None
        self.ev_end = None
        self.activation = "relu"
        self.obs_dim = int(student.trunk[0].in_features)

    @torch.no_grad()
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, log_prob, lane_log_prob = self.policy.sample_accel(obs_t)
        value = self.critic(obs_t)
        total_lp = log_prob + lane_log_prob if self.unfreeze_lane else log_prob
        return (action.squeeze(0).cpu().numpy().astype(np.float32),
                float(total_lp.item()), float(lane_log_prob.item()),
                float(value.item()))

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True):
        """시험 주행은 가속 평균. 차선은 항상 argmax."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            action = self.policy.det_action(obs_t)
        else:
            action, _, _ = self.policy.sample_accel(obs_t)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def get_value(self, obs) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.critic(obs_t).item())

    def collect_rollout(self, env, state, logger=None, ep_stats=None):
        obs = state
        finished_returns = []
        self.buffer.clear()
        self._lane_log_probs = []
        lane_hit = 0
        use_labels = os.environ.get("HOTSPOT_LABELS") == "1"
        self._labels = []
        label_stats = dict(left_labels=0, keep_labels=0, near_left_attempts=0, near_unsafe_changes=0)
        penalties = dict.fromkeys(("lane_reversal_penalty", "unsafe_lane_change_penalty",
                                   "hotspot_stall_penalty", "unsafe_accel_penalty"), 0.0)
        for _ in range(self.n_steps):
            action, log_prob, lane_log_prob, value = self.act(obs)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            ref_lane = float(self.policy.student_lane_command(obs_t).item())
            if abs(float(action[1]) - ref_lane) < 1e-5:
                lane_hit += 1
            if use_labels:
                label, site = env.hotspot_training_label()
                self._labels.append(label)
                label_stats["left_labels"] += int(label == 0)
                label_stats["keep_labels"] += int(label == 1)
                label_stats["near_left_attempts"] += int(bool(site) and action[1] > 0.5)
            trace_dir = os.environ.get("TRAIN_TRACE_DIR")
            if trace_dir:
                import traci
                st = env._current_stop()
                trace_row = dict(step=env.step_count + 1, time=traci.simulation.getTime(),
                                 edge=traci.vehicle.getRoadID(env.ego_id),
                                 lane=traci.vehicle.getLaneIndex(env.ego_id),
                                 pos=traci.vehicle.getLanePosition(env.ego_id),
                                 speed=traci.vehicle.getSpeed(env.ego_id),
                                 accel=float(action[0]), lane_cmd=float(action[1]),
                                 bus_stop=env._stop_obs(), stop=st["id"] if st else "",
                                 site=site if use_labels else "", label=label if use_labels else -1)
            next_obs, reward, terminated, truncated, info = env.step(action)
            if trace_dir:
                from pathlib import Path
                trace_row.update(lane_after=info.get("lane_after") if info.get("lane_after") is not None else -1,
                                 lane_before=info.get("lane_before") if info.get("lane_before") is not None else -1,
                                 shadow=bool(info.get("shield_active")), gate=bool(info.get("shield_lc_gate")),
                                 risk_penalty=info.get("unsafe_lane_change_penalty", 0))
                if not hasattr(self, "_episode_records"):
                    self._episode_records = []
                self._episode_records.append(trace_row)
                if terminated or truncated:
                    from evaluation.drive_parallel import end_reason
                    path = Path(trace_dir)
                    path.mkdir(parents=True, exist_ok=True)
                    records = self._episode_records
                    filename = f"{self.num_timesteps+1}_{env._ep_seed}.npz"
                    temp = path / ("tmp_" + filename)
                    np.savez_compressed(temp, end_reason=end_reason(info, terminated, truncated),
                                        **{k:np.asarray([r[k] for r in records]) for k in trace_row})
                    temp.replace(path / filename)
                    self._episode_records = []
            if use_labels and site:
                label_stats["near_unsafe_changes"] += int(info.get("unsafe_lane_change_penalty", 0) > 0)
            for key in penalties:
                penalties[key] += info.get(key, 0.0)
            self.num_timesteps += 1
            ep_stats[0] += reward
            ep_stats[1] += 1
            stored_reward = reward
            if truncated and not terminated:
                stored_reward += self.gamma * self.get_value(next_obs)
            self.buffer.add(obs, action, log_prob, stored_reward, value, terminated)
            self._lane_log_probs.append(lane_log_prob)
            if terminated or truncated:
                finished_returns.append(ep_stats[0])
                if logger is not None:
                    logger.log_episode(self.num_timesteps, ep_stats[0], int(ep_stats[1]))
                ep_stats[0], ep_stats[1] = 0.0, 0
                obs, _ = env.reset()
            else:
                obs = next_obs
        self.lane_penalty_means = {key: value / self.n_steps for key, value in penalties.items()}
        if use_labels:
            self.lane_penalty_means.update(label_stats)
        self.last_lane_match = lane_hit / max(self.n_steps, 1)
        return obs, self.get_value(obs), finished_returns

    def _batches(self, indices):
        T = len(indices)
        np.random.shuffle(indices)
        for start in range(0, T, self.minibatch_size):
            yield indices[start:start + self.minibatch_size]

    def update_critic_only(self, returns) -> dict:
        """정책은 그대로 두고 가치만 맞춘다."""
        obs_t, _, _ = self.buffer.to_tensors(self.device)
        ret_t = torch.as_tensor(returns, device=self.device)
        before = explained_variance(self.buffer.values, returns)
        policy_before = {k: v.detach().clone() for k, v in self.policy.state_dict().items()}
        indices = np.arange(len(self.buffer))
        loss_sum, n_batches = 0.0, 0
        for _ in range(self.n_epochs):
            for mb in self._batches(indices):
                value = self.critic(obs_t[mb])
                loss = ((value - ret_t[mb]) ** 2).mean()
                self.critic_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()
                loss_sum += float(loss.item())
                n_batches += 1
        with torch.no_grad():
            after_pred = self.critic(obs_t).cpu().numpy()
        after = explained_variance(after_pred, returns)
        for k, v in self.policy.state_dict().items():
            if not torch.equal(v, policy_before[k]):
                raise RuntimeError(f"critic 전용 업데이트인데 정책이 바뀜: {k}")
        return {"value_loss": loss_sum / max(n_batches, 1),
                "explained_variance": after, "ev_before": before}

    def update(self, advantages, returns) -> dict:
        obs_t, act_t, old_log_prob_t = self.buffer.to_tensors(self.device)
        old_lane = torch.as_tensor(self._lane_log_probs, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, device=self.device)
        ret_t = torch.as_tensor(returns, device=self.device)
        if self.normalize_advantage:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        indices = np.arange(len(self.buffer))
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0}
        n_batches = 0
        epochs_ran = 0
        stopped = False
        for _ in range(self.n_epochs):
            if stopped:
                break
            epochs_ran += 1
            epoch_kl, epoch_n = 0.0, 0
            for mb in self._batches(indices):
                if self.unfreeze_lane:
                    log_prob, entropy = self.policy.full_log_prob(obs_t[mb], act_t[mb])
                    old_mb = old_log_prob_t[mb] + old_lane[mb]
                else:
                    log_prob, entropy = self.policy.accel_log_prob(obs_t[mb], act_t[mb])
                    old_mb = old_log_prob_t[mb]
                ratio = torch.exp(log_prob - old_mb)
                s1 = ratio * adv_t[mb]
                s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                policy_loss = -torch.min(s1, s2).mean()
                value = self.critic(obs_t[mb])
                value_loss = ((value - ret_t[mb]) ** 2).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                if not self.no_anchor and self.beta > 0:
                    _, kl = self.policy.anchor_stats(obs_t[mb], self.anchor_std)
                    loss = loss + self.beta * kl
                if self.unfreeze_lane and self.lane_beta > 0:
                    loss = loss + self.lane_beta * self.policy.lane_kl_to_ref(obs_t[mb], self.lane_increases_only)
                if getattr(self, "_labels", None):
                    labels = torch.as_tensor(self._labels, device=self.device)[mb]
                    selected = labels >= 0
                    if selected.any():
                        logits = self.policy.lane_logits(obs_t[mb][selected])
                        loss = loss + float(os.environ.get("HOTSPOT_LABEL_WEIGHT", "0.1")) * nn.functional.cross_entropy(logits, labels[selected])
                self.policy_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in self.policy.parameters() if p.requires_grad], self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.policy_optim.step()
                self.critic_optim.step()
                with torch.no_grad():
                    approx_kl = (old_mb - log_prob).mean()
                    clip_frac = ((ratio - 1).abs() > self.clip_eps).float().mean()
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.mean().item())
                stats["approx_kl"] += float(approx_kl.item())
                stats["clip_frac"] += float(clip_frac.item())
                epoch_kl += float(approx_kl.item())
                epoch_n += 1
                n_batches += 1
            if epoch_n and (epoch_kl / epoch_n) > self.target_kl:
                stopped = True
        out = {k: v / max(n_batches, 1) for k, v in stats.items()}
        with torch.no_grad():
            rms_t, kl_t = self.policy.anchor_stats(obs_t, self.anchor_std)
            lane_kl = float(self.policy.lane_kl_to_ref(obs_t, self.lane_increases_only).item()) if self.unfreeze_lane else 0.0
            pred = self.critic(obs_t).cpu().numpy()
        rms = float(rms_t.item())
        if not self.no_anchor:
            self.beta = adjust_beta(
                self.beta, rms, self.anchor_target_rms, self.beta_min)
        if self.unfreeze_lane and self.lane_beta > 0:
            self.lane_beta = adjust_beta(
                self.lane_beta, math.sqrt(max(lane_kl, 0.0)),
                self.lane_kl_target, self.lane_beta_min)
        out.update({
            "anchor_rms": rms,
            "lane_delta": self.lane_delta(),
            "beta": float(self.beta),
            "kl_ref": float(kl_t.item()),
            "lane_kl": lane_kl,
            "lane_beta": float(self.lane_beta),
            "explained_variance": explained_variance(pred, returns),
            "epochs": epochs_ran,
            "std": self.policy.current_std,
        })
        return out

    def learn(self, env, total_timesteps: int, logger=None, log_interval: int = 1,
              eval_interval: int = 0, eval_episodes: int = 5, checkpoint_interval: int = 0,
              checkpoint_fn=None, eval_fn=None, eval_seed_base=None, save_steps=(), save_fn=None):
        pending = sorted(int(s) for s in save_steps)
        obs, _ = env.reset()
        ep_stats = [0.0, 0]
        recent = deque(maxlen=20)
        n_updates = 0
        self.policy_updates = 0
        self.last_lane_match = 1.0
        self._lane_snap = {name: param.detach().clone()
                           for name, param in self.policy.named_parameters()
                           if name.startswith("lane_")}
        clock = time.perf_counter()
        while self.num_timesteps < total_timesteps:
            obs, last_value, finished = self.collect_rollout(
                env, obs, logger=logger, ep_stats=ep_stats)
            recent.extend(finished)
            advantages, returns = self.buffer.compute_gae(
                last_value, self.gamma, self.gae_lambda)
            if not self.warmup_done:
                stats = self.update_critic_only(returns)
                self.warmup_updates += 1
                if self.ev_start is None:
                    self.ev_start = stats["ev_before"]
                    print(f"warmup 시작 explained_variance={self.ev_start}", flush=True)
                self.ev_end = stats["explained_variance"]
                done = (self.ev_end >= self.warmup_ev
                        or self.warmup_updates >= self.warmup_max_updates
                        or self.num_timesteps >= total_timesteps)
                if done:
                    self.warmup_done = True
                    why = "explained_variance" if self.ev_end >= self.warmup_ev else "한도"
                    print(f"warmup 끝 explained_variance={self.ev_end} "
                          f"updates={self.warmup_updates} ({why})", flush=True)
            else:
                stats = self.update(advantages, returns)
                self.policy_updates += 1
            if self.lane_increases_only and (
                    not all(math.isfinite(float(stats[key])) for key in ("value_loss", "anchor_rms", "lane_kl", "policy_loss")
                            if key in stats)
                    or not all(bool(torch.isfinite(parameter).all())
                               for parameter in self.policy.parameters())):
                raise FloatingPointError("Non-finite stage-6 PPO metrics or policy weights")
            n_updates += 1
            metrics = {
                "update": n_updates,
                "ep_rew_mean": float(np.mean(recent)) if recent else float("nan"),
                "lane_match": float(self.last_lane_match),
                "anchor_rms": 0.0 if not self.warmup_done else stats.get("anchor_rms", 0.0),
                "lane_delta": float(self.lane_delta()),
                "beta": float(self.beta),
                "kl_ref": 0.0 if not self.warmup_done else stats.get("kl_ref", 0.0),
                "explained_variance": stats.get("explained_variance", float("nan")),
                "value_loss": stats.get("value_loss", float("nan")),
                "std": self.policy.current_std,
                "lane_kl": stats.get("lane_kl", 0.0),
                "lane_beta": self.lane_beta,
                **self.lane_penalty_means,
            }
            if self.warmup_done and "policy_loss" in stats:
                metrics.update({k: stats[k] for k in
                                ("policy_loss", "approx_kl", "clip_frac", "entropy")})
            if logger is not None:
                logger.log_train(self.num_timesteps, metrics,
                                 verbose=(n_updates % log_interval == 0))
            while pending and self.num_timesteps >= pending[0]:
                if save_fn is not None:
                    save_fn(pending[0])
                pending.pop(0)
            if eval_interval > 0 and n_updates % eval_interval == 0 and eval_fn is not None:
                from utils.evaluator import evaluate_policy
                m = evaluate_policy(self, env, n_episodes=eval_episodes,
                                    seed_base=eval_seed_base, shield_mode="shadow")
                if logger is not None:
                    logger.log_eval(self.num_timesteps, m.as_dict(), m.summary(), eval_episodes)
                eval_fn(m)
                obs, _ = env.reset()
                ep_stats[0], ep_stats[1] = 0.0, 0
            if (checkpoint_fn is not None and checkpoint_interval > 0
                    and n_updates % checkpoint_interval == 0):
                checkpoint_fn()
        if not self.warmup_done:
            print(f"warmup 끝 explained_variance={self.ev_end} "
                  f"updates={self.warmup_updates} (스텝 한도)", flush=True)
        elapsed = max(time.perf_counter() - clock, 1e-6)
        rate = self.num_timesteps / elapsed
        print(f"학습 속도 train_steps={self.num_timesteps} train_seconds={elapsed:.1f} "
              f"steps_per_sec={rate:.3f}", flush=True)
        print(f"요약 policy_updates={self.policy_updates} "
              f"lane_match={self.last_lane_match:.4f} lane_delta={self.lane_delta():.6g} "
              f"beta={self.beta}", flush=True)
        return {"ev_start": self.ev_start, "ev_end": self.ev_end,
                "policy_updates": self.policy_updates, "steps_per_sec": rate}

    def lane_delta(self) -> float:
        """학습 시작 대비 차선 갈래 파라미터 L2."""
        snap = getattr(self, "_lane_snap", None)
        if not snap:
            return 0.0
        total = 0.0
        for name, param in self.policy.named_parameters():
            if name in snap:
                total += float((param.detach() - snap[name]).pow(2).sum().item())
        return math.sqrt(total)

    def save(self, path: str):
        torch.save({
            "format": "kl_ppo",
            "activation": self.activation,
            "explore_std": self.explore_std,
            "anchor_std": self.anchor_std,
            "unfreeze_lane": self.unfreeze_lane,
            "lane_temp": self.policy.lane_temp,
            "lane_beta": self.lane_beta,
            "lane_increases_only": self.lane_increases_only,
            "lane_kl_target": self.lane_kl_target,
            "lane_beta_min": self.lane_beta_min,
            "no_anchor": self.no_anchor,
            "obs_dim": self.obs_dim,
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load_weights(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("format") != "kl_ppo":
            raise ValueError("KL-PPO 체크포인트가 아닙니다.")
        if any(key.startswith("ref_lane_trunk.") for key in ckpt["policy"]):
            self.policy.snapshot_reference()
        else:
            self.policy.ref_lane_trunk = None
        self.lane_increases_only = bool(ckpt.get("lane_increases_only"))
        self.policy.load_state_dict(ckpt["policy"])
        self.critic.load_state_dict(ckpt["critic"])
        return self

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(ckpt, dict) or ckpt.get("format") != "kl_ppo":
            raise ValueError("KL-PPO 체크포인트가 아닙니다.")
        ref_sd = {k[len("ref."):]: v for k, v in ckpt["policy"].items() if k.startswith("ref.")}
        student = student_from_state_dict(ref_sd, ckpt.get("activation", "relu"))
        agent = cls(
            student,
            explore_std=float(ckpt["explore_std"]),
            anchor_std=float(ckpt.get("anchor_std", ANCHOR_STD)),
            no_anchor=bool(ckpt.get("no_anchor", False)),
            unfreeze_lane=bool(ckpt.get("unfreeze_lane", False)),
            lane_temp=float(ckpt.get("lane_temp", 1.0)),
            lane_beta=float(ckpt.get("lane_beta", 0.0)),
            lane_kl_target=float(ckpt.get("lane_kl_target", 0.01)),
            lane_beta_min=float(ckpt.get("lane_beta_min", 0.1)),
            device=device,
        )
        agent.activation = ckpt.get("activation", "relu")
        agent.load_weights(path)
        agent.policy.eval()
        return agent


def load_eval_policy(path, obs_dim, act_dim, hparams):
    """PPO state_dict와 KL-PPO 체크포인트를 구분해 predict()가 있는 객체를 돌려준다."""
    from algorithms.ppo import PPO
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and ckpt.get("format") == "kl_ppo":
        agent = KLPPO.from_checkpoint(path, device="cpu")
        if agent.obs_dim != obs_dim:
            raise ValueError(f"KL-PPO 관측 {agent.obs_dim} != 환경 {obs_dim}")
        return agent
    agent = PPO(obs_dim=obs_dim, act_dim=act_dim, **{**hparams, "device": "cpu"})
    agent.policy.load_state_dict(ckpt)
    agent.policy.eval()
    return agent
