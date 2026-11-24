"""신경망 부품 모음 — Domain-Grouped Entity Self-Attention 아키텍처.

기존 단순 1D MLP(규칙 근사 한계)를 대체하여, 67차원 관측 벡터를
자차·5개 차선·신호·횡단보도·정류장 9개 의미 엔티티(Entity Token)로 분리한 뒤
Multi-Head Self-Attention으로 객체 간 공간·규제 상호작용을 학습한다.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def orthogonal_init(module: nn.Module, gain: float = 1.0):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)
    return module


class DrivingEntityAttentionTrunk(nn.Module):
    """자율주행 관측(67차원) 전용 Entity Tokenizer + Multi-Head Self-Attention 트렁크.

    67차원 관측 블록 매핑 (env/mdp_config.py 기준):
      - speed (0:1), turn (32:35), school_zone (56:58), lc_zone (61:62) -> Ego/Reg Token (7d)
      - neighbors (1:21 -> 5x4), lane_end (21:26 -> 5x1),
        density (26:31 -> 5x1), lane_arrows (35:50 -> 5x3)             -> 5 Lane Tokens (5 x 9d)
      - signal (31:32? 주의: 정확한 순서는 speed[1], neighbors[20], lane_end[5],
        density[5], signal[6], turn[3], lane_arrows[15], crossings[6],
        school_zone[2], bus_stop[3], lc_zone[1])
    """

    def __init__(self, obs_dim: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, out_dim: int = 256, activation: str = "relu"):
        super().__init__()
        self.obs_dim = obs_dim
        self.d_model = d_model
        act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation]

        # 67차원 표준 청계천/고속도로 관측일 때 9개 의미 엔티티로 분해
        self.use_entity_attn = (obs_dim == 67)
        if self.use_entity_attn:
            self.ego_enc = nn.Sequential(
                orthogonal_init(nn.Linear(7, d_model)), act_cls(),
                orthogonal_init(nn.Linear(d_model, d_model)),
            )
            self.lane_enc = nn.Sequential(
                orthogonal_init(nn.Linear(9, d_model)), act_cls(),
                orthogonal_init(nn.Linear(d_model, d_model)),
            )
            self.signal_enc = nn.Sequential(
                orthogonal_init(nn.Linear(6, d_model)), act_cls(),
                orthogonal_init(nn.Linear(d_model, d_model)),
            )
            self.cross_enc = nn.Sequential(
                orthogonal_init(nn.Linear(6, d_model)), act_cls(),
                orthogonal_init(nn.Linear(d_model, d_model)),
            )
            self.stop_enc = nn.Sequential(
                orthogonal_init(nn.Linear(3, d_model)), act_cls(),
                orthogonal_init(nn.Linear(d_model, d_model)),
            )
            # 9개 토큰(자차 1 + 차선 5 + 신호 1 + 횡단보도 1 + 정류장 1) 고유 역할 임베딩
            self.token_type_emb = nn.Parameter(torch.randn(1, 9, d_model) * 0.02)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

            # 자차 중심 어텐션 풀링 + 전체 토큰 평균 결합 -> out_dim
            self.readout = nn.Sequential(
                orthogonal_init(nn.Linear(d_model * 2, out_dim)),
                nn.LayerNorm(out_dim),
                act_cls(),
            )
        else:
            # 비표준 obs_dim 폴백용
            self.fallback = nn.Sequential(
                orthogonal_init(nn.Linear(obs_dim, out_dim)),
                nn.LayerNorm(out_dim),
                act_cls(),
            )

    def _tokenize(self, obs: torch.Tensor) -> torch.Tensor:
        # obs shape: (B, 67)
        # 인덱스 구성:
        #   0:1   speed (1)
        #   1:21  neighbors (5 lanes x 4)
        #   21:26 lane_end (5 lanes x 1)
        #   26:31 density (5 lanes x 1)
        #   31:37 signal (6)
        #   37:40 turn (3)
        #   40:55 lane_arrows (5 lanes x 3)
        #   55:61 crossings (6)
        #   61:63 school_zone (2)
        #   63:66 bus_stop (3)
        #   66:67 lc_zone (1)
        B = obs.shape[0]
        ego_feat = torch.cat([
            obs[:, 0:1],    # speed
            obs[:, 37:40],  # turn
            obs[:, 61:63],  # school_zone
            obs[:, 66:67],  # lc_zone
        ], dim=-1)          # (B, 7)

        nb = obs[:, 1:21].view(B, 5, 4)
        le = obs[:, 21:26].view(B, 5, 1)
        dn = obs[:, 26:31].view(B, 5, 1)
        ar = obs[:, 40:55].view(B, 5, 3)
        lane_feat = torch.cat([nb, le, dn, ar], dim=-1)  # (B, 5, 9)

        sig_feat = obs[:, 31:37]    # (B, 6)
        crs_feat = obs[:, 55:61]    # (B, 6)
        stp_feat = obs[:, 63:66]    # (B, 3)

        t_ego = self.ego_enc(ego_feat).unsqueeze(1)      # (B, 1, D)
        t_lanes = self.lane_enc(lane_feat)               # (B, 5, D)
        t_sig = self.signal_enc(sig_feat).unsqueeze(1)   # (B, 1, D)
        t_crs = self.cross_enc(crs_feat).unsqueeze(1)    # (B, 1, D)
        t_stp = self.stop_enc(stp_feat).unsqueeze(1)     # (B, 1, D)

        tokens = torch.cat([t_ego, t_lanes, t_sig, t_crs, t_stp], dim=1)  # (B, 9, D)
        return tokens + self.token_type_emb

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        squeeze = (obs.dim() == 1)
        if squeeze:
            obs = obs.unsqueeze(0)

        if self.use_entity_attn:
            tokens = self._tokenize(obs)
            attended = self.transformer(tokens)          # (B, 9, D)
            ego_ctx = attended[:, 0, :]                  # 자차 중심 상호작용 표현
            scene_ctx = attended.mean(dim=1)             # 전체 장면 요약 표현
            z = self.readout(torch.cat([ego_ctx, scene_ctx], dim=-1))
        else:
            z = self.fallback(obs)

        return z.squeeze(0) if squeeze else z


# ======================================================================
# 연속 행동용 Actor-Critic (Gaussian 정책)
# ======================================================================
class GaussianActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256),
                 activation: str = "relu", init_log_std=-0.5,
                 squash_mean: bool = True):
        super().__init__()
        self.squash_mean = squash_mean
        out_dim = hidden_sizes[-1] if hidden_sizes else 256
        self.trunk = DrivingEntityAttentionTrunk(
            obs_dim, d_model=64, n_heads=4, n_layers=2,
            out_dim=out_dim, activation=activation,
        )
        self.mu_head = orthogonal_init(nn.Linear(out_dim, act_dim), gain=0.01)
        self.v_head = orthogonal_init(nn.Linear(out_dim, 1), gain=1.0)

        if np.isscalar(init_log_std):
            init = torch.ones(act_dim) * float(init_log_std)
        else:
            init = torch.as_tensor([float(x) for x in init_log_std])
        self.log_std = nn.Parameter(init)

    def forward(self, obs):
        z = self.trunk(obs)
        mu = self.mu_head(z)
        if self.squash_mean:
            mu = torch.tanh(mu)
        std = torch.exp(self.log_std)
        return mu, std, self.v_head(z).squeeze(-1)

    def distribution(self, obs) -> Normal:
        mu, std, _ = self.forward(obs)
        return Normal(mu, std)

    def evaluate_actions(self, obs, actions):
        mu, std, value = self.forward(obs)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value

    @property
    def current_std(self) -> float:
        return float(torch.exp(self.log_std).mean().item())


# ======================================================================
# 이산 행동용 Actor-Critic (Categorical 정책)
# ======================================================================
class CategoricalActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(256, 256),
                 activation: str = "relu"):
        super().__init__()
        out_dim = hidden_sizes[-1] if hidden_sizes else 256
        self.trunk = DrivingEntityAttentionTrunk(
            obs_dim, d_model=64, n_heads=4, n_layers=2,
            out_dim=out_dim, activation=activation,
        )
        self.pi_head = orthogonal_init(nn.Linear(out_dim, n_actions), gain=0.01)
        self.v_head = orthogonal_init(nn.Linear(out_dim, 1), gain=1.0)

    def forward(self, obs):
        z = self.trunk(obs)
        return self.pi_head(z), self.v_head(z).squeeze(-1)

    def distribution(self, obs) -> Categorical:
        logits, _ = self.forward(obs)
        return Categorical(logits=logits)

    def evaluate_actions(self, obs, actions):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions.long()), dist.entropy(), value

    @property
    def current_std(self) -> float:
        return float("nan")


# ======================================================================
# 하이브리드 Actor-Critic: Entity Self-Attention + 연속 가감속 + 이산 차선변경
# ======================================================================
class HybridActorCritic(nn.Module):
    LANE_CMDS = (1.0, 0.0, -1.0)

    def __init__(self, obs_dim: int, hidden_sizes=(256, 256),
                 activation: str = "relu", init_log_std=-0.5,
                 lane_keep_bias: float = 1.0):
        super().__init__()
        out_dim = hidden_sizes[-1] if hidden_sizes else 256
        self.trunk = DrivingEntityAttentionTrunk(
            obs_dim, d_model=64, n_heads=4, n_layers=2,
            out_dim=out_dim, activation=activation,
        )
        self.mu_head = orthogonal_init(nn.Linear(out_dim, 1), gain=0.01)
        self.lane_head = orthogonal_init(nn.Linear(out_dim, 3), gain=0.01)
        self.v_head = orthogonal_init(nn.Linear(out_dim, 1), gain=1.0)
        with torch.no_grad():
            self.lane_head.bias[1] = lane_keep_bias

        if not np.isscalar(init_log_std):
            init_log_std = float(init_log_std[0])
        self.log_std = nn.Parameter(torch.ones(1) * float(init_log_std))

    def forward(self, obs):
        z = self.trunk(obs)
        mu = torch.tanh(self.mu_head(z))
        std = torch.exp(self.log_std)
        lane_logits = self.lane_head(z)
        return mu, std, lane_logits, self.v_head(z).squeeze(-1)

    def sample(self, obs):
        mu, std, logits, value = self.forward(obs)
        nd = Normal(mu, std)
        accel = nd.sample()
        lane_idx = Categorical(logits=logits).sample()
        log_prob = (nd.log_prob(accel).sum(-1)
                    + Categorical(logits=logits).log_prob(lane_idx))
        lane_cmd = torch.as_tensor(
            [self.LANE_CMDS[int(i)] for i in lane_idx], device=accel.device)
        action = torch.cat([accel, lane_cmd.unsqueeze(-1)], dim=-1)
        return action, log_prob, value

    def det_action(self, obs):
        mu, _, logits, _ = self.forward(obs)
        lane_idx = torch.argmax(logits, dim=-1)
        lane_cmd = torch.as_tensor(
            [self.LANE_CMDS[int(i)] for i in lane_idx], device=mu.device)
        return torch.cat([mu, lane_cmd.unsqueeze(-1)], dim=-1)

    def evaluate_actions(self, obs, actions):
        mu, std, logits, value = self.forward(obs)
        nd = Normal(mu, std)
        cd = Categorical(logits=logits)
        accel = actions[:, :1]
        lane_idx = (1.0 - torch.round(actions[:, 1])).long().clamp(0, 2)
        log_prob = nd.log_prob(accel).sum(-1) + cd.log_prob(lane_idx)
        entropy = nd.entropy().sum(-1) + cd.entropy()
        return log_prob, entropy, value

    def lane_entropy(self, obs) -> torch.Tensor:
        _, _, logits, _ = self.forward(obs)
        return Categorical(logits=logits).entropy()

    def lane_change_prob(self, obs) -> torch.Tensor:
        _, _, logits, _ = self.forward(obs)
        p = torch.softmax(logits, dim=-1)
        return 1.0 - p[:, 1]

    @property
    def current_std(self) -> float:
        return float(torch.exp(self.log_std).mean().item())
