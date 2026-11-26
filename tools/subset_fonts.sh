#!/bin/bash
# 페이지에 실제로 쓰인 글자 + 자주 쓰는 한글 2,350자만 남겨 woff2로 만든다.
#   _src/subset_fonts.sh <fonttools가 깔린 python>
PY=${1:-python3}
cd "$(dirname "$0")/.."
$PY - <<'PYEOF'
import re, subprocess, sys
txt = "".join(open(f, encoding="utf-8").read() for f in ("site/index.html", "site/app.js"))
ks = bytes(range(0x20, 0x7f)).decode() + "".join(
    bytes([a, b]).decode("euc-kr", "ignore") for a in range(0xB0, 0xC9) for b in range(0xA1, 0xFF))
chars = "".join(sorted(set(txt + ks + "·→←↑↓–—“”‘’…×²₀₁₂°%㎞㎡")))
open("_src/fonts/chars.txt", "w", encoding="utf-8").write(chars)
for name in ("SeoulNamsanM", "SeoulHangangM"):
    subprocess.check_call([sys.executable, "-m", "fontTools.subset", f"_src/fonts/{name}.woff",
        "--text-file=_src/fonts/chars.txt", "--flavor=woff2", f"--output-file=site/fonts/{name}.woff2",
        "--layout-features=*", "--no-hinting"])
PYEOF
ls -la site/fonts
