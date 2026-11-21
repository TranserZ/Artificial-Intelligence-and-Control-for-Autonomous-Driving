"""선생님(PPO + shield 켬) → 시범 데이터 → 모방학습(BC) → 학생 PPO(shield 끔) 파이프라인.

공용 파일(env/, train.py, algorithms/)은 고치지 않고 가져다 쓴다. 보상·하이퍼파라미터는
train.py의 REWARD / HPARAMS 그대로다 (바꾸는 축은 "학습 방식" 하나).
도로 단계는 환경변수 CGC_STAGE로 정한다 (env/curriculum.py).

모드
  expert   : 선생님 학습. 학습·검증 모두 shield "on" (위험한 명령을 shield가 덮어씀)
  collect  : 선생님이 shield "on"으로 운전한 시범 데이터 저장.
             정답 행동 = shield가 고친 뒤 "실제로 적용된" 가감속과 실제 실행된 차선변경
  bc       : HybridActorCritic(PPO와 같은 구조)으로 시범을 흉내 낸다.
             가감속은 정규분포 log-likelihood, 차선은 cross-entropy,
             가치 함수는 시범 판의 할인 수익으로 미리 맞춘다 (PPO 첫 업데이트 붕괴 방지)
  finetune : 학생 PPO. 학습 shield "off", 검증 "shadow".
             PPO 손실 + λ·BC 손실. λ는 bc_coef에서 decay_steps 동안 0으로 줄어든다.
             처음 warmup_steps 동안은 가치 출력층(v_head)만 학습한다 (행동은 BC 그대로)
  eval     : 체크포인트를 개발 판에서 평가 (기본 900100~900129, shield shadow)

예시
  CGC_STAGE=1 python models/bc_ppo/bcppo.py expert --seed 0 --steps 300000 --out models/bc_ppo/runs/s1_expert_ts0
  CGC_STAGE=1 python models/bc_ppo/bcppo.py collect --ckpt .../best_model.pt --episodes 100 --data models/bc_ppo/data/s1.npz
  CGC_STAGE=1 python models/bc_ppo/bcppo.py bc --data models/bc_ppo/data/s1.npz --out models/bc_ppo/runs/s1_bc
  CGC_STAGE=1 python models/bc_ppo/bcppo.py finetune --init .../bc.pt --data ... --out models/bc_ppo/runs/s1_ft_ts0
  CGC_STAGE=1 python models/bc_ppo/bcppo.py eval --ckpt .../final.pt --shield shadow
"""
import argparse
import json
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

# 여러 학습을 동시에 돌리면 PyTorch가 프로세스마다 코어 전부를 쓰려 해서 서로 막힌다
# (학습 6개 동시 실행 시 부하 47 / 16코어, 2단계 선생님 45분 동안 업데이트 0회). 기본 1 스레드.
torch.set_num_threads(int(os.environ.get("BCPPO_THREADS", "1")))

import train as T                                # REWARD, HPARAMS (공용 기본값)
import algorithms.ppo as ppo_module
from algorithms.ppo import PPO
from env import road_builder
from env.mdp_config import ACTION, OBSERVATION, SIMULATION
from env.road_config import EGO, ROAD, TRAFFIC
from env.sumo_env import SumoHighwayEnv
from utils.buffer import RolloutBuffer
from utils.evaluator import evaluate_policy
from utils.logger import RunLogger

VAL_SEED_BASE = 900_000      # 학습 중 best 고르기 (protocol 3-2)
DEV_SEED_BASE = 900_100      # 개발·튜닝 확인 (900100~900129)
COLLECT_SEED_BASE = 10_000   # 시범 데이터 판 (학습 판 범위, 테스트·검증 구간 밖)


# ──────────────────────────────────────────────────────────────
# 공통
# ──────────────────────────────────────────────────────────────
# 선생님 전용 차선변경 게이트 (팀 기본값보다 엄격). 팀 기본값(rear_tau 0.3 s, approach_horizon 0.6 s)은
# 버스가 거의 서 있을 때 뒤차(8 m/s)와 5.4 m만 벌어져도 변경을 허용해, shield on 선생님 학습 3만 step에서
# 충돌 48~61%가 전부 "차선변경 직후 뒤차가 버스를 받음"이었다 (뒤차가 4.5 m/s²로 서려면 7 m 이상 필요).
# 선생님은 시범을 만드는 도구일 뿐이라 여기서만 강화한다. 학생 학습·평가(감시 모드)는 팀 기본값.
EXPERT_LC_GATE = {"rear_tau": 1.0, "approach_horizon": 2.0}
# 엄격한 간격으로도 충돌이 남았다: 거의 전부 교차로를 막 지난 엣지 0~3 m에서 바꾼 직후 옆 차로 차가 받음.
# 1단계는 신호가 없어 팀 코드의 "교차로 직후 금지 구간"이 꺼져 있고, 게이트는 교차로 안을 지나는 차를
# 못 본다. 그래서 선생님 게이트는 엣지 시작·끝 EXPERT_JUNCTION_GAP m 안의 변경을 막는다.
EXPERT_JUNCTION_GAP = 20.0
# 선생님 학습 초반 탐색: 기본 lane_keep_bias 0.5는 step마다 약 55% 차선변경 시도 → 2.0 (약 14%)
EXPERT_LANE_KEEP_BIAS = 2.0


class ExpertGateEnv(SumoHighwayEnv):
    """선생님 전용: 교차로 앞뒤 EXPERT_JUNCTION_GAP m에서는 차선변경 게이트가 막는다 (shield on일 때)."""

    def _lane_change_is_safe(self, direction: int) -> bool:
        import traci
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            pos = traci.vehicle.getLanePosition(self.ego_id)
            length = traci.lane.getLength(traci.vehicle.getLaneID(self.ego_id))
        except traci.TraCIException:
            return False
        if edge.startswith(":") or pos < EXPERT_JUNCTION_GAP or length - pos < EXPERT_JUNCTION_GAP:
            return False
        return super()._lane_change_is_safe(direction)


# 선생님 정류장 정차 규칙 (규칙 + PPO = Optimal Bus). shield만 켠 선생님은 5만 step 동안 정차 성공 0회:
# 정류장 차로·15 m 구간·10초 연속 정지를 탐색으로 우연히 해내기 어렵고, 놓침 −20 × 5 때문에
# "첫 정류장 앞에 서 있기"(시드 1) 또는 "전부 지나치기"(시드 0)로 갔다.
STOP_RULE_RANGE = 150.0    # m. 이 안에 다음 정류장이 있으면 정류장 차로로 옮긴다
STOP_RULE_DECEL = 1.0      # m/s². 편한 감속 곡선 (ROii 감속 한계 2.5)
STOP_RULE_TARGET = 2.0     # m. 앞범퍼를 정류장 끝보다 이만큼 앞에 세운다
# 선생님 신호 접근 규칙 (2단계~)
SIGNAL_RULE_DECEL = 1.5         # m/s². 적색 앞 감속 곡선
SIGNAL_RULE_YELLOW_DECEL = 2.0  # m/s². 황색에서 이 감속으로 설 수 있을 때만 선다
SIGNAL_RULE_MARGIN = 1.0        # m. 정지선 앞 여유


class TeacherEnv(ExpertGateEnv):
    """선생님 = PPO + shield on + 정류장 정차 규칙. 규칙은 모델 명령보다 느리게만 바꾼다.

    info["teacher_rule"]: 이번 step에 정차 규칙이 가감속·차선을 바꿨는가.
    """

    def _stop_rule(self, accel_raw, lane_raw):
        import traci
        st = self._current_stop()
        ns = self._next_stop()
        if st is None or ns is None or self.ego_id not in traci.vehicle.getIDList():
            return accel_raw, lane_raw, False
        dist_end, _, _ = ns
        if dist_end < 0 or dist_end > STOP_RULE_RANGE:
            return accel_raw, lane_raw, False
        v = traci.vehicle.getSpeed(self.ego_id)
        changed = False
        # 차선: 정류장 차로 쪽으로 (관측 bus_stop.lane_diff: +왼쪽, −오른쪽). 반대쪽 변경은 막는다
        diff = self._stop_obs()[1]
        want = 0 if abs(diff) < 1e-6 else (1 if diff > 0 else -1)
        cmd = self._quantize_lane_change(lane_raw)
        if cmd != want and (want != 0 or cmd != 0):
            lane_raw, changed = float(want), True
        # 속도: 정류장 차로면 정류장 끝 2 m 앞에 선다. 아직 다른 차로면 정류장 "시작점" 앞에 설 수 있게
        #   감속하며 차로 틈을 기다린다 (2단계: 오른쪽 차로가 신호 대기줄로 막혀 게이트가 계속 막고,
        #   감속 없이 왼쪽 차로로 지나쳐 정류장 놓침 55%)
        if True:
            if want == 0:
                d = max(dist_end - STOP_RULE_TARGET, 0.0)
                hold = self._in_stop_area(st) and (self._dwell_steps > 0 or d <= 0.5)
            else:
                d = max(dist_end - (st["end"] - st["start"]), 0.0)
                hold = False
            if hold:
                target_v = 0.0                      # 목표 지점 도착 또는 정차 중: 10초 채울 때까지 정지
            else:
                target_v = float(np.sqrt(2.0 * STOP_RULE_DECEL * d))
            if target_v < 0.05:
                rule_a = -1.0
            else:
                need = (target_v - v) / self.step_length
                rule_a = need / self.max_accel if need >= 0 else need / self.max_decel
            rule_a = float(np.clip(rule_a, -1.0, 1.0))
            if rule_a < accel_raw - 1e-6:
                accel_raw, changed = rule_a, True
        return accel_raw, lane_raw, changed

    def _signal_rule(self, accel_raw):
        """적색(또는 편하게 설 수 있는 황색·멈추기 전 일시정지)이면 미리 감속해 정지선 앞에 선다.

        신호 shield는 "멈출 수 있는 마지막 순간"에 최대 감속으로 세워서, 그 시범을 흉내 낸 2단계 BC가
        정지선 26 m 앞까지 30 km/h로 달리다 한 박자 늦어 적색 위반 50% (개발 30판)."""
        import traci
        sig = self._next_signal()
        if sig is None or self.ego_id not in traci.vehicle.getIDList():
            return accel_raw, False
        _, dist, state, _ = sig
        if dist > self.signal_visibility:
            return accel_raw, False
        v = traci.vehicle.getSpeed(self.ego_id)
        d = max(dist - SIGNAL_RULE_MARGIN, 0.0)
        if state == "s" and self._stopped_at == sig[0]:
            return accel_raw, False                 # 일시정지 후 진행: 이미 멈췄음
        if state in ("y",) and v * v > 2.0 * SIGNAL_RULE_YELLOW_DECEL * d:
            return accel_raw, False                 # 황색인데 편하게 못 섬 → 통과
        if state not in ("r", "u", "y", "s"):
            return accel_raw, False
        target_v = float(np.sqrt(2.0 * SIGNAL_RULE_DECEL * d))
        if target_v < 0.05:
            rule_a = -1.0
        else:
            need = (target_v - v) / self.step_length
            rule_a = need / self.max_accel if need >= 0 else need / self.max_decel
        rule_a = float(np.clip(rule_a, -1.0, 1.0))
        if rule_a < accel_raw - 1e-6:
            return rule_a, True
        return accel_raw, False

    def teacher_rules(self, accel_raw, lane_raw):
        """선생님 규칙 전부: 정류장 정차 → 신호 접근 (둘 다 모델 명령보다 느리게만 바꿈)."""
        accel_raw, lane_raw, used_stop = self._stop_rule(accel_raw, lane_raw)
        if self.signal_obs:
            accel_raw, used_sig = self._signal_rule(accel_raw)
        else:
            used_sig = False
        return accel_raw, lane_raw, used_stop or used_sig

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        a[0], a[1], used = self.teacher_rules(float(np.clip(a[0], -1, 1)), float(a[1]))
        obs, r, te, tr, info = super().step(a)
        info["teacher_rule"] = used
        return obs, r, te, tr, info


class DaggerEnv(TeacherEnv):
    """DAgger 수집: 운전은 학생 행동 그대로, 매 step "선생님이라면 무엇을 했을지"를 info에 남긴다.

    선생님 행동 = 선생님 PPO(결정적) → 정류장 정차 규칙 → shield(보행자·차량·신호, 선생님 게이트).
    학생 행동에는 규칙·shield를 적용하지 않는다 (shield는 감시 모드로 기록만).
    """
    teacher = None          # PPO 에이전트 (cmd_dagger가 붙인다)
    _last_obs = None

    def reset(self, *a, **kw):
        obs, info = super().reset(*a, **kw)
        self._last_obs = obs
        return obs, info

    def _teacher_label(self):
        import traci
        if self.ego_id not in traci.vehicle.getIDList() or self._last_obs is None:
            return 0.0, 0.0
        t = self.teacher.predict(self._last_obs, deterministic=True)
        acc, lane, _ = self.teacher_rules(float(np.clip(t[0], -1, 1)), float(t[1]))
        v = traci.vehicle.getSpeed(self.ego_id)
        out = acc
        p_cmd, p_hit = self.ped_filter.apply(acc, self.ego_id)
        v_cmd, v_hit = self._shield_accel(acc, v)
        s_cmd, s_hit = self._signal_shield_accel(acc, v, self._next_signal())
        for cmd, hit in ((p_cmd, p_hit), (v_cmd, v_hit), (s_cmd, s_hit)):
            if hit:
                out = min(out, cmd)
        lc = self._quantize_lane_change(lane)
        if lc != 0:
            try:
                edge = traci.vehicle.getRoadID(self.ego_id)
                ok = (not edge.startswith(":")
                      and self._is_car_lane(edge, traci.vehicle.getLaneIndex(self.ego_id) + lc)
                      and self._lane_change_is_safe(lc))
            except traci.TraCIException:
                ok = False
            lc = lc if ok else 0
        return float(np.clip(out, -1.0, 1.0)), float(lc)

    def step(self, action):
        label = self._teacher_label()
        obs, r, te, tr, info = ExpertGateEnv.step(self, action)   # 정차 규칙 없이 학생 행동 그대로
        info["teacher_label"] = label
        self._last_obs = obs
        return obs, r, te, tr, info


def make_env(shield_mode: str, replay_prob: float = 0.0, lc_gate: dict = None, teacher: bool = False,
             env_cls=None):
    work = tempfile.TemporaryDirectory(prefix=f"bcppo_{os.getpid()}_")
    cfg = road_builder.build(ROAD, EGO, TRAFFIC, work.name)
    sim = {**SIMULATION,
           "shield": {**SIMULATION["shield"], "mode": shield_mode},
           "replay": {"prob": replay_prob, "size": 200}}
    action = ACTION
    cls = SumoHighwayEnv
    if lc_gate:
        action = {**ACTION, "lane_change_safety": {**ACTION["lane_change_safety"], **lc_gate}}
        cls = TeacherEnv if teacher else ExpertGateEnv
    if env_cls is not None:
        cls = env_cls
    env = cls(cfg_path=cfg, road=ROAD, ego=EGO, mdp_sim=sim,
                         mdp_obs=OBSERVATION, action=action, reward=T.REWARD,
                         traffic=TRAFFIC, gui=False)
    env._workdir = work          # 환경과 수명을 같이 하도록 붙여 둔다
    return env


def make_agent(env, seed: int, cls=PPO, **extra):
    hp = {**T.HPARAMS, "seed": seed}
    for k in list(extra):
        if k in hp:
            hp[k] = extra.pop(k)
    return cls(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0],
               buffer=RolloutBuffer(), **hp, **extra)


def load_weights(agent, path):
    sd = torch.load(path, map_location=agent.device, weights_only=True)
    agent.policy.load_state_dict(sd)


def force_eval_shield(mode: str):
    """PPO.learn 안의 검증 평가가 쓸 shield 모드를 바꾼다 (기본은 shadow로 고정돼 있음)."""
    def _ev(policy, env, **kw):
        kw["shield_mode"] = mode
        return evaluate_policy(policy, env, **kw)
    ppo_module.evaluate_policy = _ev


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")


def run_meta(args):
    return {"stage": ROAD.get("stage"), "elements": ROAD.get("elements"),
            "args": vars(args), "hparams": T.HPARAMS, "reward": T.REWARD,
            "git": T._git("rev-parse", "HEAD"),
            "git_dirty": bool(T._git("status", "--porcelain", "--untracked-files=no")),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def metric_line(m):
    return (f"완주(정류장 포함) {m.completion_rate:.0%} · 도착 {m.success_rate:.0%} · "
            f"충돌 {m.collision_rate:.0%} · 적색 {m.red_violation_rate:.0%} · "
            f"시간초과 {m.timeout_rate:.0%} · 정류장놓침 {m.missed_stop_rate:.0%} · "
            f"수동개입률 {m.intervention_rate:.1%} · 평균속도 {m.mean_speed:.2f} m/s")


def score_of(m):
    """best 기준: 완벽 완주율(정류장 포함) → 충돌 적음 → 도착률 → 주행 거리."""
    return (m.completion_rate, -m.collision_rate, m.success_rate, m.mean_speed * m.ep_length)


def train_loop(agent, env, args, eval_shield: str):
    """train.py와 같은 산출물(model/best_model/final.pt, 로그, config)을 남기며 학습한다."""
    os.makedirs(args.out, exist_ok=True)
    logger = RunLogger(args.out)
    write_json(os.path.join(args.out, "config.json"), {**run_meta(args), "eval_shield": eval_shield})
    best = {"score": None}
    model_path = os.path.join(args.out, "model.pt")

    def on_eval(m):
        print(f"    [검증 {eval_shield}] {metric_line(m)}", flush=True)
        s = score_of(m)
        if best["score"] is None or s > best["score"]:
            best["score"] = s
            agent.save(os.path.join(args.out, "best_model.pt"))
            write_json(os.path.join(args.out, "best.json"),
                       {"steps": agent.num_timesteps, **m.as_dict()})
            print(f"    [best @{agent.num_timesteps}] → best_model.pt", flush=True)

    def checkpoint():
        agent.save(model_path)
        agent.save_training_state(os.path.join(args.out, "model.resume.pt"))
        logger.save_csv()

    def save_step(step):
        agent.save(os.path.join(args.out, f"step{step // 1000}k.pt"))

    force_eval_shield(eval_shield)
    done = False
    try:
        agent.learn(env, total_timesteps=args.steps, logger=logger, log_interval=1,
                    eval_interval=args.eval_interval, eval_episodes=args.eval_episodes,
                    checkpoint_interval=10, checkpoint_fn=checkpoint, eval_fn=on_eval,
                    eval_seed_base=VAL_SEED_BASE,
                    save_steps=tuple(int(s) for s in args.save_steps.split(",") if s),
                    save_fn=save_step)
        done = True
    finally:
        env.close()
        logger.close()
        agent.save(model_path)
        agent.save_training_state(os.path.join(args.out, "model.resume.pt"))
        if done:
            agent.save(os.path.join(args.out, "final.pt"))
            agent.save_training_state(os.path.join(args.out, "final.resume.pt"))
        print(f"저장: {args.out} ({'완료' if done else '중단'}, {agent.num_timesteps} step)")


# ──────────────────────────────────────────────────────────────
# expert: 선생님 PPO (shield on)
# ──────────────────────────────────────────────────────────────
def cmd_expert(args):
    env = make_env("on", replay_prob=args.replay_prob, lc_gate=EXPERT_LC_GATE,
                   teacher=not args.no_stop_rule)
    agent = make_agent(env, args.seed, lane_keep_bias=EXPERT_LANE_KEEP_BIAS)
    if args.init:
        load_weights(agent, args.init)
        print(f"초기 가중치: {args.init}")
    if args.lr:
        for g in agent.optimizer.param_groups:
            g["lr"] = args.lr
    train_loop(agent, env, args, eval_shield="on")


# ──────────────────────────────────────────────────────────────
# collect: 시범 데이터 (shield on, 실제 적용된 행동이 정답)
# ──────────────────────────────────────────────────────────────
def end_reason(info, terminated, truncated):
    if info.get("collided"):
        return "collision"
    if info.get("red_violation") and terminated:
        return "red_light"
    if info.get("wrong_lane"):
        return "lane_error"
    if info.get("arrived"):
        return "missed_stop" if int(info.get("missed_stops", 0) or 0) else "completed"
    return "timeout" if truncated else "vanished"


def cmd_dagger(args):
    """학생(BC)이 운전한 상태에 선생님 정답을 붙여 저장 (DAgger 1회분)."""
    env = make_env("shadow", lc_gate=EXPERT_LC_GATE, env_cls=DaggerEnv)
    env.eval_mode = True
    student = make_agent(env, 0)
    load_weights(student, args.ckpt)
    teacher = make_agent(env, 0)
    load_weights(teacher, args.teacher)
    env.teacher = teacher
    obs_l, act_l, rew_l, ep_l, episodes = [], [], [], [], []
    t0 = time.time()
    try:
        for i in range(args.episodes):
            seed = args.seed_base + i
            obs, _ = env.reset(seed=seed)
            done, n = False, 0
            while not done:
                a = student.predict(obs, deterministic=not args.stochastic)
                nobs, r, te, tr, info = env.step(a)
                obs_l.append(obs.astype(np.float32))
                act_l.append(info["teacher_label"])
                rew_l.append(r)
                ep_l.append(i)
                obs, done, n = nobs, te or tr, n + 1
            why = end_reason(info, te, tr)
            episodes.append({"episode": i, "seed": seed, "steps": n, "end_reason": why,
                             "stops_done": info.get("stops_done")})
            print(f"[{i + 1}/{args.episodes}] seed {seed}: 학생 {why}, {n} step, "
                  f"정류장 {info.get('stops_done')}/{info.get('stops_total')} ({time.time() - t0:.0f}s)",
                  flush=True)
    finally:
        env.close()
    act = np.array(act_l, dtype=np.float32)
    np.savez_compressed(args.data, obs=np.array(obs_l), act=act, rew=np.array(rew_l, dtype=np.float32),
                        ep=np.array(ep_l, dtype=np.int32), dagger=np.array(True),
                        end_reason=np.array([e["end_reason"] for e in episodes]))
    reasons = {}
    for e in episodes:
        reasons[e["end_reason"]] = reasons.get(e["end_reason"], 0) + 1
    write_json(args.data.replace(".npz", ".json"),
               {**run_meta(args), "student": args.ckpt, "teacher": args.teacher,
                "transitions": len(obs_l), "student_end_reasons": reasons,
                "label_lane_change_ratio": float(np.mean(act[:, 1] != 0)), "episodes": episodes})
    print(f"저장: {args.data} — {len(obs_l)} step, 학생 종료 {reasons}, "
          f"선생님 차선변경 정답 {np.mean(act[:, 1] != 0):.2%}")


def cmd_collect(args):
    env = make_env("on", lc_gate=EXPERT_LC_GATE, teacher=not args.no_stop_rule)
    env.eval_mode = True
    agent = make_agent(env, 0)
    load_weights(agent, args.ckpt)
    obs_l, act_l, rew_l, term_l, trunc_l, ep_l, shield_l = [], [], [], [], [], [], []
    episodes = []
    t0 = time.time()
    try:
        for i in range(args.episodes):
            seed = args.seed_base + i
            obs, _ = env.reset(seed=seed)
            done, n, ret = False, 0, 0.0
            while not done:
                a = agent.predict(obs, deterministic=True)
                nobs, r, te, tr, info = env.step(a)
                # 정답 = 실제로 적용된 행동 (shield가 고쳤으면 고친 값, 막힌 차선변경은 유지 0)
                lane = int(info["lane_change"]) if (info["lane_change"] != 0
                                                   and info["lane_change_applied"]) else 0
                obs_l.append(obs.astype(np.float32))
                act_l.append((float(info["accel_raw"]), float(lane)))
                rew_l.append(r)
                term_l.append(te)
                trunc_l.append(tr)
                ep_l.append(i)
                shield_l.append(bool(info["shield_active"]))
                obs, done, n, ret = nobs, te or tr, n + 1, ret + r
            why = end_reason(info, te, tr)
            episodes.append({"episode": i, "seed": seed, "steps": n, "return": round(ret, 3),
                             "end_reason": why, "stops_done": info.get("stops_done"),
                             "stops_total": info.get("stops_total")})
            print(f"[{i + 1}/{args.episodes}] seed {seed}: {why}, {n} step, "
                  f"정류장 {info.get('stops_done')}/{info.get('stops_total')}, 보상 {ret:.1f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        env.close()
    os.makedirs(os.path.dirname(os.path.abspath(args.data)), exist_ok=True)
    np.savez_compressed(args.data, obs=np.array(obs_l), act=np.array(act_l, dtype=np.float32),
                        rew=np.array(rew_l, dtype=np.float32), term=np.array(term_l),
                        trunc=np.array(trunc_l), ep=np.array(ep_l, dtype=np.int32),
                        shield=np.array(shield_l),
                        end_reason=np.array([e["end_reason"] for e in episodes]))
    reasons = {}
    for e in episodes:
        reasons[e["end_reason"]] = reasons.get(e["end_reason"], 0) + 1
    write_json(args.data.replace(".npz", ".json"),
               {**run_meta(args), "ckpt": args.ckpt, "transitions": len(obs_l),
                "shield_step_ratio": float(np.mean(shield_l)) if shield_l else 0.0,
                "end_reasons": reasons, "episodes": episodes})
    print(f"저장: {args.data} — {len(obs_l)} step, 종료 {reasons}, "
          f"shield 개입 step {np.mean(shield_l):.1%}")


# ──────────────────────────────────────────────────────────────
# bc: 모방학습 (데이터 읽기 · 손실)
# ──────────────────────────────────────────────────────────────
GOOD_ENDS = ("completed", "missed_stop")   # 충돌·적색·차로오류·시간초과 판은 흉내 내지 않는다


def load_demos(paths, gamma, only_good=True):
    """시범 데이터 → (obs, act, 할인 수익). 여러 파일을 이어 붙인다."""
    obs_all, act_all, ret_all, ep_all = [], [], [], []
    ep_offset = 0
    for path in paths:
        d = np.load(path)
        ep, ends = d["ep"], d["end_reason"]
        dagger = bool(d["dagger"]) if "dagger" in d.files else False
        keep = (np.isin(ends[ep], GOOD_ENDS) if only_good and not dagger
                else np.ones(len(ep), bool))
        # 판마다 뒤에서 앞으로 할인 수익 (시간 초과로 잘린 판은 만들지 않으므로 끝 가치 0)
        ret = np.zeros(len(ep), np.float32)
        g = 0.0
        for t in range(len(ep) - 1, -1, -1):
            if t == len(ep) - 1 or ep[t + 1] != ep[t]:
                g = 0.0
            g = d["rew"][t] + gamma * g
            ret[t] = g
        obs_all.append(d["obs"][keep])
        act_all.append(d["act"][keep])
        ret_all.append(ret[keep])
        ep_all.append(ep[keep] + ep_offset)
        ep_offset += int(ep.max()) + 1
        print(f"  {path}: {keep.sum()} / {len(ep)} step 사용 "
              f"(좋은 판 {len(set(ep[keep]))} / {len(set(ep))})")
    return (np.concatenate(obs_all), np.concatenate(act_all),
            np.concatenate(ret_all), np.concatenate(ep_all))


def bc_losses(policy, obs, act):
    """HybridActorCritic의 BC 손실: 가감속 정규분포 -log p, 차선 cross-entropy."""
    mu, std, logits, value = policy(obs)
    nll = -Normal(mu, std).log_prob(act[:, :1]).sum(-1).mean()
    lane_idx = (1.0 - torch.round(act[:, 1])).long().clamp(0, 2)   # +1→0(좌), 0→1(유지), −1→2(우)
    ce = F.cross_entropy(logits, lane_idx)
    return nll, ce, value, mu, logits, lane_idx


def cmd_bc(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    obs, act, ret, ep = load_demos(args.data, T.HPARAMS["gamma"], only_good=not args.all_episodes)
    # 판 단위로 검증 10% 분리 (같은 판의 이웃 step이 양쪽에 섞이지 않게)
    eps = np.unique(ep)
    rng = np.random.default_rng(args.seed)
    val_eps = rng.choice(eps, size=max(1, len(eps) // 10), replace=False)
    val = np.isin(ep, val_eps)
    print(f"학습 {int((~val).sum())} step / 검증 {int(val.sum())} step, "
          f"차선변경 비율 {np.mean(act[:, 1] != 0):.2%}")

    from utils.networks import HybridActorCritic
    hp = T.HPARAMS
    policy = HybridActorCritic(obs.shape[1], hidden_sizes=hp["hidden_sizes"],
                               activation=hp["activation"], init_log_std=hp["init_log_std"],
                               lane_keep_bias=hp["lane_keep_bias"])
    if args.init:
        policy.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True))
        print(f"초기 가중치: {args.init}")
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    to = lambda x: torch.as_tensor(x)
    O, A, R = to(obs), to(act), to(ret)
    tr_idx, va_idx = np.where(~val)[0], np.where(val)[0]
    # 차선변경은 드문 행동이라 가중치를 줘서 "항상 유지"로 무너지지 않게 한다 (lane_weight 배)
    lane_w = torch.ones(3)
    lane_w[[0, 2]] = args.lane_weight
    os.makedirs(args.out, exist_ok=True)
    log = []
    for epoch in range(1, args.epochs + 1):
        policy.train()
        rng.shuffle(tr_idx)
        tot = {"nll": 0.0, "ce": 0.0, "v": 0.0}
        nb = 0
        for s in range(0, len(tr_idx), args.batch):
            b = tr_idx[s:s + args.batch]
            mu, std, logits, value = policy(O[b])
            nll = -Normal(mu, std).log_prob(A[b, :1]).sum(-1).mean()
            lane_idx = (1.0 - torch.round(A[b, 1])).long().clamp(0, 2)
            ce = F.cross_entropy(logits, lane_idx, weight=lane_w)
            v_loss = F.mse_loss(value, R[b])
            loss = nll + ce + args.value_coef * v_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            tot["nll"] += nll.item(); tot["ce"] += ce.item(); tot["v"] += v_loss.item()
            nb += 1
        policy.eval()
        with torch.no_grad():
            mu, std, logits, value = policy(O[va_idx])
            lane_idx = (1.0 - torch.round(A[va_idx, 1])).long().clamp(0, 2)
            pred = logits.argmax(-1)
            chg = lane_idx != 1
            row = {"epoch": epoch, **{k: v / nb for k, v in tot.items()},
                   "val_accel_mae": float((mu.squeeze(-1) - A[va_idx, 0]).abs().mean()),
                   "val_lane_acc": float((pred == lane_idx).float().mean()),
                   "val_change_recall": float((pred[chg] == lane_idx[chg]).float().mean())
                   if chg.any() else float("nan"),
                   "val_value_mae": float((value - R[va_idx]).abs().mean()),
                   "std": float(std.mean())}
        log.append(row)
        print(f"[epoch {epoch:>3}] nll {row['nll']:.3f} ce {row['ce']:.3f} v {row['v']:.2f} | "
              f"검증 가감속 오차 {row['val_accel_mae']:.3f} · 차선 정확도 {row['val_lane_acc']:.3f} · "
              f"변경 재현 {row['val_change_recall']:.2f} · 가치 오차 {row['val_value_mae']:.2f}",
              flush=True)
    torch.save(policy.state_dict(), os.path.join(args.out, "bc.pt"))
    write_json(os.path.join(args.out, "bc_log.json"),
               {**run_meta(args), "train_steps": int(len(tr_idx)), "val_steps": int(len(va_idx)),
                "log": log})
    print(f"저장: {os.path.join(args.out, 'bc.pt')}")


# ──────────────────────────────────────────────────────────────
# finetune: 학생 PPO + BC 보조 손실 + 분리한 가치 신경망
# ──────────────────────────────────────────────────────────────
class BCPPO(PPO):
    """PPO 손실 + λ(t)·BC 손실, 가치 함수는 정책과 몸통(trunk)을 나누지 않는 별도 신경망.

    λ(t) = bc_coef × max(0, 1 − (t − warmup) / decay)  (t = 누적 학습 step)

    왜 가치 신경망을 나눴나: HybridActorCritic은 정책·가치가 trunk을 같이 쓴다. warmup이 끝나자
    가치 오차가 0.7 → 11로 뛰고, 그 기울기가 trunk을 바꿔 정차 행동이 무너졌다 (lr 1e-4, lr 3e-5 +
    KL 제한, BC 가중치 5 모두 5만 step에서 완주 87% → 0~5%). 정책의 v_head는 저장 형식 호환용으로
    남기고 쓰지 않는다. 그래서 저장된 .pt는 test.py·평가에서 그대로 읽힌다.
    warmup 동안은 정책을 고정하고 가치 신경망만 학생 수익에 맞춘다.
    """

    def __init__(self, *a, demos=None, demo_returns=None, bc_coef=1.0, warmup_steps=10_000,
                 decay_steps=150_000, bc_batch=64, target_kl=None, critic_lr=3e-4,
                 lane_bc_coef=None, **kw):
        super().__init__(*a, **kw)
        # 차선 BC 항 가중치. None이면 가감속과 같이 λ(t)로 줄고, 값을 주면 끝까지 그 값으로 유지한다
        # (학생이 속도 보상 때문에 왼쪽 차로로 나갔다가 정류장 11 m 앞에서야 돌아와 놓침)
        self.lane_bc_coef = lane_bc_coef
        from utils.networks import build_mlp, orthogonal_init
        hp = T.HPARAMS
        obs_dim = self.policy.trunk[0].in_features
        self.critic = build_mlp(obs_dim, hp["hidden_sizes"], 1, hp["activation"]).to(self.device)
        orthogonal_init(self.critic[-1], gain=1.0)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        # KL 조기 종료: 한 업데이트에서 approx_kl이 1.5 × target_kl을 넘으면 남은 epoch를 건너뛴다
        self.target_kl = target_kl
        obs, act = demos
        self.demo_obs = torch.as_tensor(obs, device=self.device)
        self.demo_act = torch.as_tensor(act, device=self.device)
        self.demo_ret = None if demo_returns is None else torch.as_tensor(demo_returns, device=self.device)
        self.bc_coef, self.warmup, self.decay, self.bc_batch = bc_coef, warmup_steps, decay_steps, bc_batch

    def value(self, obs_t):
        return self.critic(obs_t).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, log_prob, _ = self.policy.sample(obs_t)
        return (action.squeeze(0).cpu().numpy().astype(np.float32),
                float(log_prob.item()), float(self.value(obs_t).item()))

    @torch.no_grad()
    def get_value(self, obs) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.value(obs_t).item())

    def pretrain_critic(self, epochs=5, batch=256):
        """시범 판 할인 수익으로 가치 신경망을 먼저 맞춘다 (학생 수익에는 warmup에서 다시 맞춤)."""
        if self.demo_ret is None:
            return
        n = len(self.demo_obs)
        for ep in range(epochs):
            perm = torch.randperm(n, device=self.device)
            tot, nb = 0.0, 0
            for s in range(0, n, batch):
                j = perm[s:s + batch]
                loss = F.mse_loss(self.value(self.demo_obs[j]), self.demo_ret[j])
                self.critic_opt.zero_grad()
                loss.backward()
                self.critic_opt.step()
                tot += loss.item(); nb += 1
            print(f"  가치 신경망 사전학습 {ep + 1}/{epochs}: MSE {tot / nb:.2f}", flush=True)

    def bc_lambda(self):
        t = self.num_timesteps
        if t < self.warmup:
            return self.bc_coef
        return self.bc_coef * max(0.0, 1.0 - (t - self.warmup) / max(self.decay, 1))

    def update(self, advantages, returns) -> dict:
        obs_t, act_t, old_log_prob_t = self.buffer.to_tensors(self.device)
        adv_t = torch.as_tensor(advantages, device=self.device)
        ret_t = torch.as_tensor(returns, device=self.device)
        if self.normalize_advantage:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        lam = self.bc_lambda()
        warm = self.num_timesteps < self.warmup
        T_ = len(self.buffer)
        idx = np.arange(T_)
        keys = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "bc_loss")
        stats = {k: 0.0 for k in keys}
        nb = 0
        actor_on = not warm
        stopped_epoch = self.n_epochs
        for epoch in range(self.n_epochs):
            np.random.shuffle(idx)
            epoch_kl, epoch_nb = 0.0, 0
            for s in range(0, T_, self.minibatch_size):
                mb = idx[s:s + self.minibatch_size]
                # ---- 가치 신경망 (정책과 따로) ----
                value_loss = F.mse_loss(self.value(obs_t[mb]), ret_t[mb])
                self.critic_opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()
                # ---- 정책 (warmup 동안은 고정) ----
                log_prob, entropy, _ = self.policy.evaluate_actions(obs_t[mb], act_t[mb])
                ratio = torch.exp(log_prob - old_log_prob_t[mb])
                s1 = ratio * adv_t[mb]
                s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                policy_loss = -torch.min(s1, s2).mean()
                bc_loss = torch.zeros((), device=self.device)
                if actor_on:
                    loss = policy_loss - self.entropy_coef * entropy.mean()
                    if self.lane_entropy_coef > 0:
                        loss = loss - self.lane_entropy_coef * self.policy.lane_entropy(obs_t[mb]).mean()
                    lam_lane = lam if self.lane_bc_coef is None else self.lane_bc_coef
                    if lam > 0 or lam_lane > 0:
                        j = torch.randint(0, len(self.demo_obs), (self.bc_batch,), device=self.device)
                        nll, ce, *_ = bc_losses(self.policy, self.demo_obs[j], self.demo_act[j])
                        bc_loss = nll + ce
                        loss = loss + lam * nll + lam_lane * ce
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                with torch.no_grad():
                    kl = (old_log_prob_t[mb] - log_prob).mean().item()
                    stats["approx_kl"] += kl
                    epoch_kl += kl
                    epoch_nb += 1
                    stats["clip_fraction"] += ((ratio - 1).abs() > self.clip_eps).float().mean().item()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["bc_loss"] += bc_loss.item()
                nb += 1
            # KL이 넘치면 이번 업데이트의 남은 epoch는 가치 신경망만 학습한다
            if (actor_on and self.target_kl
                    and epoch_kl / max(epoch_nb, 1) > 1.5 * self.target_kl):
                actor_on = False
                stopped_epoch = epoch + 1
        out = {k: v / max(nb, 1) for k, v in stats.items()}
        with torch.no_grad():
            var = torch.var(ret_t)
            out["explained_variance"] = (float(1 - torch.var(ret_t - self.value(obs_t)) / var)
                                         if var > 1e-12 else float("nan"))
            out["p_lane_change"] = float(self.policy.lane_change_prob(obs_t).mean())
        out["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        out["bc_lambda"] = lam
        out["epochs_run"] = 0 if warm else stopped_epoch
        out["warmup"] = int(warm)
        return out


def cmd_finetune(args):
    obs, act, ret, _ = load_demos(args.data, T.HPARAMS["gamma"], only_good=not args.all_episodes)
    env = make_env("off", replay_prob=args.replay_prob)
    agent = make_agent(env, args.seed, cls=BCPPO, demos=(obs, act), demo_returns=ret,
                       bc_coef=args.bc_coef, warmup_steps=args.warmup_steps,
                       decay_steps=args.decay_steps, target_kl=args.target_kl,
                       lane_bc_coef=args.lane_bc_coef)
    load_weights(agent, args.init)
    print(f"초기 가중치: {args.init} · BC λ {args.bc_coef} → 0 ({args.warmup_steps} + "
          f"{args.decay_steps} step) · 시범 {len(obs)} step · 가치 신경망 분리")
    if args.lr:
        for g in agent.optimizer.param_groups:
            g["lr"] = args.lr
    agent.pretrain_critic()
    train_loop(agent, env, args, eval_shield="shadow")

# ──────────────────────────────────────────────────────────────
# eval: 개발 판 평가
# ──────────────────────────────────────────────────────────────
def cmd_eval(args):
    env = make_env(args.shield, lc_gate=EXPERT_LC_GATE if args.expert_gate else None,
                   teacher=args.expert_gate)
    agent = make_agent(env, 0)
    load_weights(agent, args.ckpt)
    t0 = time.time()
    try:
        m = evaluate_policy(agent, env, n_episodes=args.episodes, seed_base=args.seed_base,
                            shield_mode=args.shield)
    finally:
        env.close()
    print(f"[{args.ckpt} · {ROAD.get('stage')}단계 · shield {args.shield} · "
          f"판 {args.seed_base}~{args.seed_base + args.episodes - 1}] {metric_line(m)} "
          f"({time.time() - t0:.0f}s)")
    if args.json:
        write_json(args.json, {**run_meta(args), **m.as_dict()})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def train_opts(q):
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--steps", type=int, default=300_000)
        q.add_argument("--out", required=True)
        q.add_argument("--init", default=None)
        q.add_argument("--lr", type=float, default=None)
        q.add_argument("--replay-prob", type=float, default=0.3)
        q.add_argument("--eval-interval", type=int, default=25)
        q.add_argument("--eval-episodes", type=int, default=20)
        q.add_argument("--save-steps", default="100000,200000,300000")

    q = sub.add_parser("expert"); train_opts(q)
    q.add_argument("--no-stop-rule", action="store_true", help="정류장 정차 규칙 없이 (shield만)")
    q = sub.add_parser("finetune"); train_opts(q)
    q.add_argument("--data", nargs="+", required=True)
    q.add_argument("--bc-coef", type=float, default=1.0)
    q.add_argument("--warmup-steps", type=int, default=10_000)
    q.add_argument("--decay-steps", type=int, default=150_000)
    q.add_argument("--all-episodes", action="store_true", help="실패한 판도 흉내 낸다 (스모크용)")
    q.add_argument("--target-kl", type=float, default=0.02, help="KL 조기 종료 기준 (0이면 끔)")
    q.add_argument("--lane-bc-coef", type=float, default=None,
                   help="차선 BC 항을 이 가중치로 끝까지 유지 (없으면 가감속과 같이 줄어듦)")

    q = sub.add_parser("collect")
    q.add_argument("--ckpt", required=True)
    q.add_argument("--episodes", type=int, default=100)
    q.add_argument("--seed-base", type=int, default=COLLECT_SEED_BASE)
    q.add_argument("--data", required=True)
    q.add_argument("--no-stop-rule", action="store_true")

    q = sub.add_parser("dagger")
    q.add_argument("--ckpt", required=True, help="운전할 학생 (BC) 모델")
    q.add_argument("--teacher", required=True, help="정답을 낼 선생님 모델")
    q.add_argument("--episodes", type=int, default=50)
    q.add_argument("--seed-base", type=int, default=20_000)
    q.add_argument("--data", required=True)
    q.add_argument("--stochastic", action="store_true", help="학생 행동을 확률적으로 (상태를 더 다양하게)")

    q = sub.add_parser("bc")
    q.add_argument("--data", nargs="+", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--init", default=None)
    q.add_argument("--epochs", type=int, default=30)
    q.add_argument("--batch", type=int, default=256)
    q.add_argument("--lr", type=float, default=3e-4)
    q.add_argument("--value-coef", type=float, default=0.5)
    q.add_argument("--lane-weight", type=float, default=3.0)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--all-episodes", action="store_true", help="실패한 판도 흉내 낸다")

    q = sub.add_parser("eval")
    q.add_argument("--ckpt", required=True)
    q.add_argument("--shield", choices=["shadow", "on", "off"], default="shadow")
    q.add_argument("--episodes", type=int, default=30)
    q.add_argument("--seed-base", type=int, default=DEV_SEED_BASE)
    q.add_argument("--json", default=None)
    q.add_argument("--expert-gate", action="store_true", help="선생님 전용(엄격한) 차선변경 게이트로 평가")

    args = p.parse_args()
    if args.cmd in ("collect", "dagger") and SumoHighwayEnv.is_held_out_seed(args.seed_base):
        p.error("시범 데이터에 테스트·검증 판 시드를 쓸 수 없습니다")
    {"expert": cmd_expert, "collect": cmd_collect, "dagger": cmd_dagger, "bc": cmd_bc,
     "finetune": cmd_finetune, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
