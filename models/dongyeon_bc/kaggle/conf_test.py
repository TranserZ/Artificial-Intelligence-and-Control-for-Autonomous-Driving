# 6단계 최종 후보 확인 (강동연): 처음 보는 테스트 판 1000~1089 (90판)으로 다시 평가. 후보 고를 때 쓴 90판과 겹치지 않는다
import os, subprocess, sys, time, glob, json
REPO = "https://github.com/TranserZ/Artificial-Intelligence-and-Control-for-Autonomous-Driving.git"
COMMIT = "b96e289dd3ef754cc410c7d9eed271178044094e"
def sh(cmd, **kw):
    print("$", " ".join(cmd), flush=True); subprocess.run(cmd, check=True, **kw)
sh([sys.executable, "-m", "pip", "install", "-q", "eclipse-sumo==1.27.1", "traci==1.28.0", "gymnasium==1.4.0"])
import sumo
os.environ["SUMO_HOME"] = sumo.SUMO_HOME
os.environ["PATH"] = os.path.join(sumo.SUMO_HOME, "bin") + os.pathsep + os.environ["PATH"]
os.environ.update({"CGC_STAGE": "6", "OBS_SCAN": "1", "TEACHER_AVOID": "1", "BCPPO_YELLOW_DECEL": "2.4", "BCPPO_THREADS": "1",
                   "BCPPO_LC_GATE": json.dumps({"rear_tau": 3.0, "approach_horizon": 4.0, "min_rear_gap": 8.0})})
D = "/kaggle/tmp/repo"; os.makedirs("/kaggle/tmp", exist_ok=True)
sh(["git", "clone", "-q", REPO, D]); sh(["git", "checkout", "-q", COMMIT], cwd=D)
W = "/kaggle/working"; P = "models/dongyeon_bc/teacher56.py"; t0 = time.time()
def find(p):
    if p.startswith("repo:"): return os.path.join(D, p[5:])
    return glob.glob(f"/kaggle/input/**/{p}", recursive=True)[0]
MODELS = [('CUR', 'repo:models/dongyeon_bc/s6_student_ppo.pt'), ('DAG0b', 'cgc-s6-dagppo/ft_s0/best_model/bc.pt')]
seeds = [1000, 1015, 1030, 1045, 1060, 1075]
for tag, p in MODELS:
    ck = find(p); os.makedirs(f"{W}/{tag}", exist_ok=True)
    procs = []
    for s in seeds:
        log = open(f"{W}/{tag}/eval_{s}.log", "w")
        procs.append((subprocess.Popen([sys.executable, "-u", P, "eval", "--ckpt", ck, "--shield", "shadow", "--episodes", "15",
                      "--seed-base", str(s), "--json", f"{W}/{tag}/eval_{s}.json"], cwd=D, stdout=log, stderr=subprocess.STDOUT), log))
    for pr, log in procs:
        pr.wait(); log.close()
        if pr.returncode != 0: print(open(log.name).read()[-2000:], flush=True)
    tot = {}
    for s in seeds:
        j = json.load(open(f"{W}/{tag}/eval_{s}.json"))
        for k in ("completion_rate", "collision_rate", "red_violation_rate", "timeout_rate", "intervention_rate"):
            tot[k] = tot.get(k, 0.0) + float(j.get(k) or 0.0) / len(seeds)
    print(f"### {tag} 테스트 90판:", {k: round(v, 3) for k, v in tot.items()}, ck, f"({(time.time()-t0)/60:.0f}분)", flush=True)
