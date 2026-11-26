# 6단계 사후 보정·모델 평균 (강동연): 추가 학습 없이 (1) 차선 head "유지" 편향 +δ (확실할 때만 차선변경) (2) 같은 출발점 미세조정 모델들의 가중치 평균(model soup) → 180판
import os, subprocess, sys, time, glob, json
REPO = "https://github.com/TranserZ/Artificial-Intelligence-and-Control-for-Autonomous-Driving.git"
COMMIT = "6803606"
def sh(cmd, **kw):
    print("$", " ".join(cmd), flush=True); subprocess.run(cmd, check=True, **kw)
sh([sys.executable, "-m", "pip", "install", "-q", "eclipse-sumo==1.27.1", "traci==1.28.0", "gymnasium==1.4.0"])
import sumo, torch
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
def load(p): return torch.load(find(p), map_location="cpu", weights_only=True)
def soup(ps):
    sds = [load(p) for p in ps]
    return {k: sum(sd[k].float() for sd in sds) / len(sds) for k in sds[0]}
def keep_bias(sd, d):
    sd = {k: v.clone() for k, v in sd.items()}
    key = [k for k in sd if k.endswith("lane_head.bias")][0]
    sd[key][1] += d
    return sd
V5 = "repo:models/dongyeon_bc/s6_student_v5.pt"
POOL = [V5, "repo:models/dongyeon_bc/s6_student_ppo.pt", "cgc-s6-sig/ft_s1/best_model/bc.pt", "cgc-s6-dagppo/ft_s1/best_model/bc.pt"]
BUILD = [('SOUP4', 'soup(POOL)'), ('SOUP2', 'soup(POOL[:2])'), ('SOUP4_KEEP05', 'keep_bias(soup(POOL), 0.5)')]
seeds = [900100, 900115, 670000, 670015, 670030, 670045, 1000, 1015, 1030, 1045, 1060, 1075]
for tag, make in BUILD:
    sd = eval(make)
    os.makedirs(f"{W}/{tag}", exist_ok=True); ck = f"{W}/{tag}/bc.pt"; torch.save(sd, ck)
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
    print(f"### {tag} 180판(개발 90 + 테스트 90):", {k: round(v, 3) for k, v in tot.items()}, f"({(time.time()-t0)/60:.0f}분)", flush=True)
