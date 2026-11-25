# 6단계 PPO 미세조정 3 (강동연): 최고 s6_student_ppo(70.0%)에서 이어서, 충돌 벌점 20→40·적색 60, best = 완주 - 충돌, 검증 30판 → 90판
import os, subprocess, sys, time, glob, json, shutil
REPO = "https://github.com/TranserZ/Artificial-Intelligence-and-Control-for-Autonomous-Driving.git"
COMMIT = "cd993e1186b90ee49741231ed95aeae8fbc710b3"
STAGE = 6
INIT = None   # 아래에서 cgc-s5-v2 출력의 s5_bc0/bc.pt로
TEACHER = "models/bc_ppo/teachers/s4_teacher.pt"
def sh(cmd, **kw):
    print("$", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)
sh([sys.executable, "-m", "pip", "install", "-q", "eclipse-sumo==1.27.1", "traci==1.28.0", "gymnasium==1.4.0"])
import sumo
os.environ["SUMO_HOME"] = sumo.SUMO_HOME
os.environ["PATH"] = os.path.join(sumo.SUMO_HOME, "bin") + os.pathsep + os.environ["PATH"]
os.environ["CGC_STAGE"] = str(STAGE)
os.environ["TEACHER_AVOID"] = "1"
os.environ["OBS_SCAN"] = "1"
os.environ["REWARD_RED"] = "60"
os.environ["REWARD_COLL"] = "40"
os.environ["BEST_COLL_W"] = "1.0"
os.environ["BCPPO_YELLOW_DECEL"] = "2.4"
os.environ["BCPPO_LC_GATE"] = json.dumps({"rear_tau": 3.0, "approach_horizon": 4.0, "min_rear_gap": 8.0})
D = "/kaggle/tmp/repo"; os.makedirs("/kaggle/tmp", exist_ok=True)
sh(["git", "clone", "-q", REPO, D]); sh(["git", "checkout", "-q", COMMIT], cwd=D)
W = "/kaggle/working"; P = "models/dongyeon_bc/teacher56.py"; T6 = "models/dongyeon_bc/teacher56.py"; t0 = time.time()
os.makedirs(f"{W}/data", exist_ok=True)
INIT = os.path.join(D, "models/dongyeon_bc/s6_student_ppo.pt")
def grab(kernel, pat, prefix, n):
    fs = sorted(glob.glob(f"/kaggle/input/**/{kernel}/data/{pat}", recursive=True)); assert len(fs) == n, (kernel, pat, fs)
    out = []
    for k, f in enumerate(fs):
        dst = f"{W}/data/{prefix}{k}.npz"; shutil.copy(f, dst); out.append(dst)
    return out
OLD = grab("cgc-s6-obsfix2", "s6_part[0-7].npz", "old_part", 8) + grab("cgc-s6-obsfix2", "d[12]_[0-3].npz", "old_d", 8)
print("시작", INIT, len(OLD), flush=True)
DEMO = OLD
def par(cmds, tag):
    procs = []
    for i, c in enumerate(cmds):
        log = open(f"{W}/{tag}_{i}.log", "w")
        procs.append((subprocess.Popen(c, cwd=D, stdout=log, stderr=subprocess.STDOUT), log))
    for p, log in procs:
        p.wait(); log.close()
        if p.returncode != 0: print(open(log.name).read()[-2500:]); raise SystemExit(1)
    return [open(f"{W}/{tag}_{i}.log").read() for i in range(len(cmds))]
def bc(out, init, data):
    env = {**os.environ, "BCPPO_THREADS": "4"}
    sh([sys.executable, "-u", P, "bc", "--data", *data, "--init", init, "--out", out, "--epochs", "30",
        "--brake-weight", "10", "--lane-weight", "3"], cwd=D, env=env)
    print(f"BC {out} 끝 {(time.time()-t0)/60:.0f}분", flush=True)
def evaluate(out, tag):
    # 개발 900100~129 (30판) + 학습 범위 570000~059 (60판) = 90판, 15판씩 6개 동시
    seeds = [900100, 900115, 670000, 670015, 670030, 670045]
    cmds = [[sys.executable, "-u", P, "eval", "--ckpt", f"{out}/bc.pt", "--shield", "shadow", "--episodes", "15",
             "--seed-base", str(s), "--json", f"{out}/eval_{s}.json"] for s in seeds]
    par(cmds, f"eval_{tag}")
    tot, n, sh_sum = {}, 0, 0.0
    for s in seeds:
        j = json.load(open(f"{out}/eval_{s}.json"))
        print(" ", s, {k: j.get(k) for k in ("completion_rate", "success_rate", "collision_rate", "red_violation_rate", "timeout_rate", "intervention_rate")}, flush=True)
        for k in ("completion_rate", "success_rate", "collision_rate", "red_violation_rate", "timeout_rate", "intervention_rate"):
            tot[k] = tot.get(k, 0.0) + float(j.get(k) or 0.0) * 15
        n += 15
    print(f"### {tag} 90판:", {k: round(v / n, 3) for k, v in tot.items()}, f"({(time.time()-t0)/60:.0f}분)", flush=True)
env2 = {**os.environ, "BCPPO_THREADS": "2"}
procs = []
for sd in (1, 2):
    log = open(f"{W}/ft_{sd}.log", "w")
    procs.append((subprocess.Popen([sys.executable, "-u", P, "finetune", "--data", *DEMO, "--init", INIT, "--out", f"{W}/ft_s{sd}",
        "--seed", str(sd), "--steps", "200000", "--bc-coef", "1.0", "--warmup-steps", "10000", "--decay-steps", "100000000",
        "--lane-bc-coef", "1.0", "--lr", "0.00003", "--replay-prob", "0.3", "--eval-interval", "25", "--eval-episodes", "30",
        "--save-steps", "100000,200000"], cwd=D, stdout=log, stderr=subprocess.STDOUT, env=env2), log))
for p, log in procs:
    p.wait(); log.close()
    if p.returncode != 0: print(open(log.name).read()[-3000:], flush=True)
print(f"학습 끝 {(time.time()-t0)/60:.0f}분", flush=True)
for sd in (1, 2):
    for l in open(f"{W}/ft_{sd}.log"):
        if "검증" in l or "best @" in l: print(sd, l.rstrip()[:220], flush=True)
for sd in (1, 2):
    out = f"{W}/ft_s{sd}"
    for name in ("best_model", "step200k"):
        if os.path.exists(f"{out}/{name}.pt"):
            os.makedirs(f"{out}/{name}", exist_ok=True); shutil.copy(f"{out}/{name}.pt", f"{out}/{name}/bc.pt")
            evaluate(f"{out}/{name}", f"FT{sd}_{name}")
