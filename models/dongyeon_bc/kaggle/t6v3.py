# 6단계 보강 선생님 v2(회피 차선 명령 + 막혔을 때 v4 간격 게이트, s4_teacher) 시범 100판 (강동연)
import os, subprocess, sys, time, glob, json
REPO = "https://github.com/TranserZ/Artificial-Intelligence-and-Control-for-Autonomous-Driving.git"
COMMIT = "020a6177feca5964800d6849c4a305fed3c53852"
def sh(cmd, **kw):
    print("$", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)
sh([sys.executable, "-m", "pip", "install", "-q", "eclipse-sumo==1.27.1", "traci==1.28.0", "gymnasium==1.4.0"])
import sumo
os.environ["SUMO_HOME"] = sumo.SUMO_HOME
os.environ["PATH"] = os.path.join(sumo.SUMO_HOME, "bin") + os.pathsep + os.environ["PATH"]
os.environ["CGC_STAGE"] = "6"
os.environ["BCPPO_YELLOW_DECEL"] = "2.4"
os.environ["BCPPO_LC_GATE"] = json.dumps({"rear_tau": 3.0, "approach_horizon": 4.0, "min_rear_gap": 8.0})
D = "/kaggle/tmp/repo"; os.makedirs("/kaggle/tmp", exist_ok=True)
sh(["git", "clone", "-q", REPO, D]); sh(["git", "checkout", "-q", COMMIT], cwd=D)
W = "/kaggle/working"; T = "models/dongyeon_bc/teacher56.py"; CK = "models/bc_ppo/teachers/s4_teacher.pt"
def par(jobs):
    procs = []
    for name, cmd, env in jobs:
        log = open(f"{W}/{name}.log", "w")
        procs.append((name, subprocess.Popen(cmd, cwd=D, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env}), log))
    for name, p, log in procs:
        p.wait(); log.close()
        print(f"----- {name} (exit {p.returncode})\n" + open(log.name).read()[-4000:], flush=True)
# 2) 보강 선생님 6단계 시범 100판
par([(f"s6_part{k}", [sys.executable, "-u", T, "collect", "--episodes", "25", "--seed-base", str(600000 + 25 * k),
      "--data", f"{W}/s6_part{k}.npz", "--ckpt", CK], {"TEACHER_AVOID": "1"}) for k in range(4)])
tot = {}
for k in range(4):
    j = json.load(open(f"{W}/s6_part{k}.json"))
    for r, n in j["end_reasons"].items(): tot[r] = tot.get(r, 0) + n
print("### 보강 선생님 6단계 100판:", tot, flush=True)
