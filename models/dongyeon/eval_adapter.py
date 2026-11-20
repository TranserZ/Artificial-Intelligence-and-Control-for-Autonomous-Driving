"""공통 평가 스크립트용 정책 로더 (experiment_protocol.md 2-6 policy factory).

    python scripts/evaluate_all.py --ckpt models/dongyeon/checkpoints/ts0/step500k.pt \
        --policy-factory models.dongyeon.eval_adapter:make_policy ...

GRU는 판마다 기억을 0에서 시작해야 한다. 공통 평가 코드는 판 시작을 알려 주지 않으므로
env.reset을 감싸서 판이 시작될 때마다 기억을 지운다 (공용 파일은 고치지 않음).
"""
import torch

from models.dongyeon.gru_ppo import RecurrentRunner, attach_reset, load_policy


def make_policy(checkpoint_path, env, config_path=None):
    device = torch.device("cpu")
    policy = load_policy(checkpoint_path, device)
    if policy.obs_dim != env.observation_space.shape[0]:
        raise ValueError(f"모델 입력 {policy.obs_dim}차원 ≠ 환경 관측 {env.observation_space.shape[0]}차원")
    runner = RecurrentRunner(policy, device)
    attach_reset(env, runner)
    return runner
