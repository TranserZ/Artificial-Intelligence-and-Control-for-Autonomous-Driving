"""numpy 정책(np_policy) == torch 정책(RecurrentRunner) 확인. torch가 있는 곳에서: python models/dongyeon/check_numpy_parity.py <model.pt>"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.dongyeon.gru_ppo import RecurrentRunner, load_policy
from np_policy import NumpyGRUPolicy
path = sys.argv[1]
tr, nr = RecurrentRunner(load_policy(path), torch.device("cpu")), NumpyGRUPolicy(path)
rng = np.random.default_rng(0); worst = 0.0
for ep in range(3):
    tr.reset(); nr.reset()
    for t in range(300):
        o = rng.uniform(-1, 1, nr.obs_dim).astype(np.float32)
        a, b = tr.predict(o), nr.predict(o)
        worst = max(worst, float(np.abs(a - b).max()))
        if a[1] != b[1]:
            raise SystemExit(f"FAIL 차선 명령 다름 ep{ep} t{t}: {a} vs {b}")
print(("PASS" if worst < 1e-4 else "FAIL"), "numpy == torch, 최대 차", worst)
sys.exit(0 if worst < 1e-4 else 1)
