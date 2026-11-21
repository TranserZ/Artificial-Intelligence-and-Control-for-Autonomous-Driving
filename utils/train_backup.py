"""train.py 버전 백업 (backup/train_file_backup/train_<날짜_시각>.py).

train.py는 하나뿐이라 학습 방식을 바꾸면 예전 내용이 덮어써진다.
그래서 학습을 시작할 때 train.py가 지난 백업과 달라졌으면 새 파일로 저장하고,
파일 맨 위에 "무엇을 왜 바꿨는지" 머리말을 붙인다.

머리말 내용
  - 목적: train.py의 TRAIN_PURPOSE (사람이 적는 메모)
  - 바뀐 설정: 이전 백업과 비교해 자동으로 찾은 값 변화
               (TOTAL_TIMESTEPS, REWARD_OVERRIDES, HPARAMS 등 대문자 설정, 함수 추가·삭제)
  - 첫 run: 이 버전으로 처음 돌린 results/ 폴더
"""
import ast
import os
from datetime import datetime

HEADER_END = "# ══════ 백업 설명 끝 — 아래는 실행된 train.py 원본 ══════\n"


def strip_header(src: str) -> str:
    """백업 파일에서 머리말을 떼고 train.py 원본만 돌려준다."""
    i = src.find(HEADER_END)
    return src[i + len(HEADER_END):] if i >= 0 else src


def _settings(src: str):
    """모듈 최상위의 대문자 설정값과 함수 이름. dict(...)/{...}는 키 단위로 펼친다."""
    vals, funcs = {}, set()
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            name, v = n.targets[0].id, n.value
            if not name.isupper():
                continue
            if isinstance(v, ast.Call) and getattr(v.func, "id", "") == "dict":
                for kw in v.keywords:
                    if kw.arg:
                        vals[f"{name}.{kw.arg}"] = ast.unparse(kw.value)
            elif isinstance(v, ast.Dict):
                for k, x in zip(v.keys, v.values):
                    if k is not None:
                        vals[f"{name}.{ast.unparse(k).strip(chr(39))}"] = ast.unparse(x)
            else:
                vals[name] = ast.unparse(v)
        elif isinstance(n, ast.FunctionDef):
            funcs.add(n.name)
    return vals, funcs


def setting_changes(prev_src: str, cur_src: str) -> list:
    """이전 → 현재 설정 변화를 사람이 읽을 문장 목록으로."""
    pv, pf = _settings(prev_src)
    cv, cf = _settings(cur_src)
    out = []
    for k in sorted(set(pv) | set(cv)):
        if k == "TRAIN_PURPOSE" or pv.get(k) == cv.get(k):
            continue
        if k not in pv:
            out.append(f"{k} = {cv[k]} (새로 추가)")
        elif k not in cv:
            out.append(f"{k} 삭제 (이전 {pv[k]})")
        else:
            out.append(f"{k}: {pv[k]} → {cv[k]}")
    out += [f"함수 추가: {f}()" for f in sorted(cf - pf)]
    out += [f"함수 삭제: {f}()" for f in sorted(pf - cf)]
    return out


def make_header(cur_src: str, purpose: str, saved_at: str, prev_name: str = None,
                prev_src: str = None, run_name: str = None) -> str:
    L = ["# ══════ train.py 백업 ══════",
         f"# 저장: {saved_at}" + (f"   (이전 백업: {prev_name})" if prev_name else "   (첫 백업)")]
    if run_name:
        L.append(f"# 이 버전으로 처음 돌린 run: results/{run_name}/")
    L.append("#")
    L.append("# [목적]")
    L += [f"#   {line}".rstrip() for line in purpose.strip().splitlines()] or ["#   (적지 않음)"]
    if prev_src is not None:
        L.append("#")
        L.append("# [바뀐 설정] 이전 백업과 자동 비교")
        ch = setting_changes(prev_src, cur_src)
        L += [f"#   - {c}" for c in ch] or ["#   - 설정값 변화 없음 (주석·코드만 바뀜)"]
    return "\n".join(L) + "\n" + HEADER_END


def backup(train_path: str, backup_dir: str, purpose: str, run_name: str = None):
    """train.py가 지난 백업과 다르면 머리말을 붙여 저장하고 경로를 돌려준다. 같으면 None."""
    os.makedirs(backup_dir, exist_ok=True)
    with open(train_path, encoding="utf-8") as f:
        cur = f.read()
    names = sorted(n for n in os.listdir(backup_dir) if n.startswith("train_") and n.endswith(".py"))
    prev_name = prev_src = None
    if names:
        prev_name = names[-1]
        with open(os.path.join(backup_dir, prev_name), encoding="utf-8") as f:
            prev_src = strip_header(f.read())
        if prev_src == cur:
            return None
        prev_purpose = _settings(prev_src)[0].get("TRAIN_PURPOSE")
        if prev_purpose is not None and prev_purpose == _settings(cur)[0].get("TRAIN_PURPOSE"):
            print("  ⚠ train.py가 바뀌었는데 TRAIN_PURPOSE는 이전 백업과 같습니다. "
                  "이번 학습 목적을 적었는지 확인하세요.")
    now = datetime.now()
    path = os.path.join(backup_dir, now.strftime("train_%Y%m%d_%H%M%S.py"))
    header = make_header(cur, purpose, now.strftime("%Y-%m-%d %H:%M"), prev_name, prev_src, run_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + cur)
    return path
