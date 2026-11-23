# 6단계 관측 보강 (강동연): OBS_SCAN=1 (옆 차로 차를 차선 스캔으로 보완)으로 선생님 시범 200판 → s5_student에서 BC₀ → DAgger 1회 → BC1 → 평가
import os, subprocess, sys, time, glob, json, shutil
REPO = "https://github.com/TranserZ/Artificial-Intelligence-and-Control-for-Autonomous-Driving.git"
COMMIT = "8a00bbca2227cf1c453348504235b7d5ea41ac79"
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
os.environ["BCPPO_YELLOW_DECEL"] = "2.4"
os.environ["BCPPO_LC_GATE"] = json.dumps({"rear_tau": 3.0, "approach_horizon": 4.0, "min_rear_gap": 8.0})
D = "/kaggle/tmp/repo"; os.makedirs("/kaggle/tmp", exist_ok=True)
sh(["git", "clone", "-q", REPO, D]); sh(["git", "checkout", "-q", COMMIT], cwd=D)
W = "/kaggle/working"; P = "models/dongyeon_bc/teacher56.py"; T6 = "models/dongyeon_bc/teacher56.py"; t0 = time.time()
os.makedirs(f"{W}/data", exist_ok=True)
INIT = "models/dongyeon_bc/s5_student.pt"
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
def collect(k0):
    cmds = [[sys.executable, "-u", T6, "collect", "--episodes", "25", "--seed-base", str(600000 + 25 * k),
             "--data", f"{W}/data/s6_part{k}.npz", "--ckpt", TEACHER] for k in range(k0, k0 + 4)]
    for l in par(cmds, f"collect{k0}"):
        print([x for x in l.split("\n") if "저장" in x][-1][:200], flush=True)
collect(0); collect(4)
print(f"시범 끝 {(time.time()-t0)/60:.0f}분", flush=True)
data = [f"{W}/data/s6_part{k}.npz" for k in range(8)]
bc(f"{W}/s6_bc0", INIT, data)
cmds = [[sys.executable, "-u", T6, "dagger", "--ckpt", f"{W}/s6_bc0/bc.pt", "--teacher", TEACHER, "--episodes", "25",
         "--seed-base", str(610000 + 25 * k), "--data", f"{W}/data/d1_{k}.npz"] for k in range(4)]
for l in par(cmds, "d1"):
    print([x for x in l.split("\n") if "저장" in x][-1][:200], flush=True)
data += [f"{W}/data/d1_{k}.npz" for k in range(4)]
bc(f"{W}/s6_bc1", f"{W}/s6_bc0/bc.pt", data)
evaluate(f"{W}/s6_bc1", "BC1")
evaluate(f"{W}/s6_bc0", "BC0")
