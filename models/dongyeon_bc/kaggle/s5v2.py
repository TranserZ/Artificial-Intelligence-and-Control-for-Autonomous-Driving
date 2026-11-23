# 5단계 본 학습 (강동연): 민기 님 4단계 선생(s4_teacher, 보호구역 규칙·황색 2.4)으로 시범 100판 → s4_student에서 BC₀ → 90판 평가 → DAgger 1회 → BC₁ → 90판 평가
import os, subprocess, sys, time, glob, json, shutil
REPO = "https://github.com/TranserZ/Artificial-Intelligence-and-Control-for-Autonomous-Driving.git"
COMMIT = "444017e27aaab062877ead67013df672a764ffa8"
STAGE = 5
INIT = "models/bc_ppo/s4_student.pt"
TEACHER = "models/bc_ppo/teachers/s4_teacher.pt"
def sh(cmd, **kw):
    print("$", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)
sh([sys.executable, "-m", "pip", "install", "-q", "eclipse-sumo==1.27.1", "traci==1.28.0", "gymnasium==1.4.0"])
import sumo
os.environ["SUMO_HOME"] = sumo.SUMO_HOME
os.environ["PATH"] = os.path.join(sumo.SUMO_HOME, "bin") + os.pathsep + os.environ["PATH"]
os.environ["CGC_STAGE"] = str(STAGE)
os.environ["BCPPO_YELLOW_DECEL"] = "2.4"
os.environ["BCPPO_LC_GATE"] = json.dumps({"rear_tau": 3.0, "approach_horizon": 4.0, "min_rear_gap": 8.0})
D = "/kaggle/tmp/repo"; os.makedirs("/kaggle/tmp", exist_ok=True)
sh(["git", "clone", "-q", REPO, D]); sh(["git", "checkout", "-q", COMMIT], cwd=D)
W = "/kaggle/working"; P = "models/bc_ppo/bcppo.py"; t0 = time.time()
os.makedirs(f"{W}/data", exist_ok=True)
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
    seeds = [900100, 900115, 570000, 570015, 570030, 570045]
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
cmds = [[sys.executable, "-u", P, "collect", "--episodes", "25", "--seed-base", str(500000 + 25 * k),
         "--data", f"{W}/data/s5_part{k}.npz", "--ckpt", TEACHER] for k in range(4)]
for l in par(cmds, "collect"):
    print([x for x in l.split("\n") if "저장" in x][-1][:200], flush=True)
data = [f"{W}/data/s5_part{k}.npz" for k in range(4)]
bc(f"{W}/s5_bc0", INIT, data)
evaluate(f"{W}/s5_bc0", "BC0")
cmds = [[sys.executable, "-u", P, "dagger", "--episodes", "25", "--seed-base", str(510000 + 25 * k),
         "--data", f"{W}/data/s5_dagger1_part{k}.npz", "--ckpt", f"{W}/s5_bc0/bc.pt", "--teacher", TEACHER] for k in range(4)]
for l in par(cmds, "dagger1"):
    print([x for x in l.split("\n") if "저장" in x][-1][:200], flush=True)
data += [f"{W}/data/s5_dagger1_part{k}.npz" for k in range(4)]
bc(f"{W}/s5_bc1", f"{W}/s5_bc0/bc.pt", data)
evaluate(f"{W}/s5_bc1", "BC1")
