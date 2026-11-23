# 5·6단계 최종 학생 실패 위치 진단 (강동연): s5_student(5단계) / s6 obsfix BC1(6단계, OBS_SCAN=1)을 shield shadow로 돌려 충돌 순간(상대 차·엣지·차선 명령 이력)을 찍는다
import os, subprocess, sys, time, glob, json
REPO = "https://github.com/TranserZ/Artificial-Intelligence-and-Control-for-Autonomous-Driving.git"
COMMIT = "96f18bfccf3f26f32d309da4d7bfba8ddf6c4138"
def sh(cmd, **kw):
    print("$", " ".join(cmd), flush=True); subprocess.run(cmd, check=True, **kw)
sh([sys.executable, "-m", "pip", "install", "-q", "eclipse-sumo==1.27.1", "traci==1.28.0", "gymnasium==1.4.0"])
import sumo
os.environ["SUMO_HOME"] = sumo.SUMO_HOME
os.environ["PATH"] = os.path.join(sumo.SUMO_HOME, "bin") + os.pathsep + os.environ["PATH"]
os.environ["CGC_STAGE"] = "6"; os.environ["BCPPO_YELLOW_DECEL"] = "2.4"; os.environ["BCPPO_THREADS"] = "1"
D = "/kaggle/tmp/repo"; os.makedirs("/kaggle/tmp", exist_ok=True)
sh(["git", "clone", "-q", REPO, D]); sh(["git", "checkout", "-q", COMMIT], cwd=D)
W = "/kaggle/working"
DIAG = r'''
import os, sys, collections
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "models/bc_ppo")
import bcppo, traci
if os.environ.get("OBS_SCAN") == "1":
    sys.path.insert(0, "models/dongyeon_bc"); import obsfix; obsfix.apply()
ckpt, s0, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
env = bcppo.make_env("shadow"); env.eval_mode = True
agent = bcppo.make_agent(env, 0); bcppo.load_weights(agent, ckpt)
for seed in range(s0, s0 + n):
    obs, _ = env.reset(seed=seed); hist = collections.deque(maxlen=12); t = 0
    while True:
        act = agent.predict(obs, deterministic=True)
        ego = env.ego_id
        if ego in traci.vehicle.getIDList():
            ld = traci.vehicle.getLeader(ego, 60.0)
            hist.append((t, round(traci.vehicle.getSpeed(ego), 1), round(float(act[0]), 2), int(act[1]) if len(act) > 1 else 0,
                         traci.vehicle.getRoadID(ego), traci.vehicle.getLaneIndex(ego), round(traci.vehicle.getLanePosition(ego)),
                         (ld[0][:16], round(ld[1], 1), round(traci.vehicle.getSpeed(ld[0]), 1)) if ld and ld[0] else None))
        obs, r, te, tr, info = env.step(act); t += 1
        cs = [(c.collider, c.victim, c.type) for c in traci.simulation.getCollisions()]
        if te or tr:
            why = bcppo.end_reason(info, te, tr)
            print(f"== seed {seed}: {why} t {t} 정차 {info.get('stops_done')} 충돌 {cs}", flush=True)
            if why != "completed" or cs:
                for h in hist: print("    ", h, flush=True)
            break
env.close()
'''
open(f"{D}/diag6.py", "w").write(DIAG)
s6 = glob.glob("/kaggle/input/**/s6_bc1/bc.pt", recursive=True)[0]
s5 = os.path.join(D, "models/dongyeon_bc/s5_student.pt")
jobs = [("s6", s6, 900100, "6", "1"), ("s6", s6, 900115, "6", "1"), ("s6", s6, 670000, "6", "1"), ("s6", s6, 670015, "6", "1"),
        ("s5", s5, 900100, "5", "0"), ("s5", s5, 900115, "5", "0"), ("s5", s5, 570000, "5", "0"), ("s5", s5, 570015, "5", "0")]
procs = []
for name, ck, s, stg, ob in jobs:
    log = open(f"{W}/diag_{name}_{s}.log", "w")
    procs.append((subprocess.Popen([sys.executable, "-u", "diag6.py", ck, str(s), "15"], cwd=D, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "CGC_STAGE": stg, "OBS_SCAN": ob}), log))
for p, log in procs: p.wait(); log.close()
for name, ck, s, stg, ob in jobs:
    txt = open(f"{W}/diag_{name}_{s}.log").read()
    ends = [l for l in txt.split("\n") if l.startswith("== seed")]
    print(f"### {name} {s}:", collections.Counter(l.split(":")[1].split()[0] for l in ends) if (collections := __import__("collections")) else "", flush=True)
