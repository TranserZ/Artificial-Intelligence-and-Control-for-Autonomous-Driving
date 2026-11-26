"""DAgger 데이터에서 "학생이 틀린 장면"만 골라 새 npz로 저장한다 (강동연, 6단계 실패 집중 학습용).

    OBS_SCAN=1 CGC_STAGE=6 python models/dongyeon_bc/hardmine.py STUDENT_CKPT OUT.npz IN1.npz IN2.npz ...

고르는 기준 (둘 중 하나):
  1. 선생님 정답과 학생 행동이 어긋난 step (가감속 차이 > ACC_TOL, 또는 차선 명령이 다름)과 그 앞뒤 PAD step
  2. 실패(충돌·적색·시간초과·차로 오류)로 끝난 판의 마지막 FAIL_TAIL step (= 실패 직전 30초)
s6_student_v5 180판 진단: 실패 48판 중 약 25판이 e908646367_3(정류장 앞 정지)·_4(적색)·_6(세워진 오토바이 옆 차선변경) 한 구간,
나머지는 주교 앞(e218924602_4/6)·보행자. 전체 DAgger 데이터를 고르게 쓰면 이런 장면은 1~2%라 BC 항에서 묻힌다.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import teacher56  # noqa: E402

bcppo = teacher56.bcppo
ACC_TOL = float(os.environ.get("HARD_ACC_TOL", "0.3"))
PAD = int(os.environ.get("HARD_PAD", "3"))
FAIL_TAIL = int(os.environ.get("HARD_FAIL_TAIL", "60"))
FAIL = ("collision", "red_light", "timeout", "wrong_lane")
LANE_CMDS = np.array([1.0, 0.0, -1.0], np.float32)

ckpt, out, inputs = sys.argv[1], sys.argv[2], sys.argv[3:]
env = bcppo.make_env("shadow")
agent = bcppo.make_agent(env, 0)
bcppo.load_weights(agent, ckpt)
pol = agent.policy.eval()

obs_o, act_o, rew_o, ep_o, ends_o = [], [], [], [], []
ep_off = 0
for path in inputs:
    d = np.load(path)
    obs, act, ep, ends = d["obs"], d["act"], d["ep"], d["end_reason"]
    with torch.no_grad():
        mu, _, logits, _ = pol.forward(torch.as_tensor(obs, dtype=torch.float32))
    s_acc = mu[:, 0].numpy()
    s_lane = LANE_CMDS[logits.argmax(-1).numpy()]
    mis = (np.abs(act[:, 0] - s_acc) > ACC_TOL) | (act[:, 1] != s_lane)
    keep = np.zeros(len(ep), bool)
    for k in np.flatnonzero(mis):
        lo, hi = max(0, k - PAD), min(len(ep), k + PAD + 1)
        keep[lo:hi] |= ep[lo:hi] == ep[k]
    fail_eps = 0
    for e in np.unique(ep):
        if ends[e] in FAIL:
            idx = np.flatnonzero(ep == e)
            keep[idx[-FAIL_TAIL:]] = True
            fail_eps += 1
    print(f"  {path}: {len(ep)} step 중 어긋남 {mis.sum()} → 고름 {keep.sum()} "
          f"(실패 판 {fail_eps}/{len(np.unique(ep))})", flush=True)
    obs_o.append(obs[keep]); act_o.append(act[keep]); rew_o.append(d["rew"][keep])
    ep_o.append(ep[keep] + ep_off); ends_o.append(ends)
    ep_off += len(ends)
env.close()
np.savez_compressed(out, obs=np.concatenate(obs_o), act=np.concatenate(act_o), rew=np.concatenate(rew_o),
                    ep=np.concatenate(ep_o).astype(np.int32), dagger=np.array(True),
                    end_reason=np.concatenate(ends_o))
print(f"저장: {out} — {sum(len(x) for x in obs_o)} step", flush=True)
