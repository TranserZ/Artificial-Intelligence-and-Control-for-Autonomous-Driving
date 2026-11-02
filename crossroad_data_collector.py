"""
실제 교차로 데이터 수집기 (서울 T-DATA)

  - 교차로 MAP 정보 API  → 교차로 ID·기반정보(제한속도 등)
  - 신호정보 API         → 방향별 신호 상태·잔여시간을 일정 간격으로 반복 수집
  - 수집 결과를 분석해 신호 단계(phase) 순서·단계별 시간·주기를 추정
  - 교통량은 API 가 아니라 2025 서울시 교통량 조사자료(TOPIS)에서 정리한 값을 저장

저장 위치: data/road_data/<교차로 이름>/
  map_info.json        MAP API 에서 찾은 교차로 기반정보
  signal_raw.jsonl     신호 API 원본 (호출 1회 = 1줄)
  signal_timeline.csv  시각 × 방향별 신호 상태·잔여시간
  signal_summary.json  추정 단계 순서·시간·주기
  traffic_volume.json  주변 도로 일평균 통행량
  meta.json            수집 일시·설정 (인증키 제외)

설정: crossroad_api_config.py (처음 실행 시 템플릿으로 자동 생성, git 미추적)

API 제한 (실측):
  - API 키당 하루 1000회 (실패 호출도 차감) → 남은 호출 수에 맞춰 수집 시간을 자동으로 줄인다
  - MAP API 는 호출 후 6시간 재호출 불가 → 성공한 목록을 data/road_data/_map_cache.json 에 캐시
  - 신호 API 는 itstId 또는 sggCd 필수 → itstId 를 자동으로 붙인다

사용법:
  python crossroad_data_collector.py --name 양평동사거리
  python crossroad_data_collector.py --name 양평동사거리 천호사거리 --duration 1200
  python crossroad_data_collector.py --name 천호사거리 --id 1234        # ID 직접 지정
  python crossroad_data_collector.py --search 양평                      # MAP 에서 후보 검색
  python crossroad_data_collector.py --name 양평동사거리 --analyze-only # 기존 원본 재분석
"""

import argparse
import csv
import importlib.util
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "data" / "road_data"
MAP_CACHE = OUT_ROOT / "_map_cache.json"
CONFIG_PATH = ROOT / "crossroad_api_config.py"
CONFIG_TEMPLATE = '''"""
crossroad_data_collector.py 설정 (git 미추적 — 인증키 포함)

URL 작성 규칙:
  - 인증키를 포함한 전체 URL 을 그대로 넣는다.
  - 신호 API 는 itstId 가 필수이며 프로그램이 자동으로 붙인다.
  - 교차로 ID 자리를 직접 지정하려면 {itst_id} 를 넣는다.
"""

# 서울 T-DATA 교차로 MAP 정보 API URL (호출 후 6시간 재호출 불가)
MAP_API_URL = ""

# 서울 T-DATA 신호정보 API URL (방향별 신호 상태·잔여시간)
SIGNAL_API_URL = ""

# 신호 잔여시간 API URL (상태와 잔여시간이 별도 API 인 경우만, 아니면 비워 둠)
SIGNAL_TIMING_API_URL = ""

# 교차로 이름 → 교차로 ID 수동 지정
MANUAL_ITST_ID = {
    # "천호사거리": "2157",
}

# 잔여시간 단위 (초). T-DATA 는 1/10초 단위(값 100 = 10초)
REMAINING_TIME_UNIT = 0.1
'''

# 수집 기본값: 1초 간격(신호 API 갱신 주기 수준) × 20분(주기 160~200초 기준 5주기 이상).
# 단계 경계 시각은 폴링 간격이 아니라 '잔여시간'으로 보정하므로 0.1초 수준까지 추정된다.
DEFAULT_DURATION = 1200
DEFAULT_INTERVAL = 1.0

# ── 교통량 (2025 서울시 교통량 조사자료, 주중 일평균, 단위 대/일) ──
# 주변 도로 측정지점 통행량이며 사거리 자체의 진입량·회전량은 아님.
TRAFFIC_VOLUME = {
    "양평동사거리": [
        {"road": "양화대교", "point_id": "C-05",
         "directions": {"남단→북단": 58435, "북단→남단": 54939},
         "total": 113374, "page": 94},
        {"road": "노들로", "point_id": "D-41", "section": "여의하류IC–여의2교",
         "directions": {"여의하류IC→여의2교": 22683, "여의2교→여의하류IC": 29011},
         "total": 51694, "page": 151},
    ],
    "천호사거리": [
        {"road": "천호대교", "point_id": "C-20",
         "directions": {"남단→북단": 30345, "북단→남단": 35003},
         "total": 65348, "page": 109,
         "note": "북단→남단 방향은 1·2·8월 유효자료 없음"},
    ],
}
TRAFFIC_SOURCE = {
    "source": "TOPIS 교통량 정보 - 2025년 서울특별시 교통량 조사자료",
    "period": "2025년 주중(월~금) 일평균",
    "unit": "대/일",
    "note": "주변 도로 측정지점 통행량이며, 사거리 전체 진입량이나 좌·직·우 회전량이 아님",
}

# ── 응답 필드 해석 ──
# 실제 응답과 다르면 여기만 수정한다.
ID_KEYS = ("itstId", "ITST_ID", "crsrdId")
NAME_KEYS = ("itstNm", "ITST_NM", "crsrdNm")
# 신호 필드: <방향><종류>sg<StatNm|RmdrCs>  예) ntStsgStatNm(북 직진 상태), ntPdsgRmdrCs(북 보행 잔여시간)
SIGNAL_FIELD_RE = re.compile(r"^(nt|et|st|wt|ne|nw|se|sw)([A-Z][a-z])sg(StatNm|RmdrCs)$")
DIRECTIONS = {"nt": "북", "et": "동", "st": "남", "wt": "서",
              "ne": "북동", "nw": "북서", "se": "남동", "sw": "남서"}
MOVEMENTS = {"St": "직진", "Lt": "좌회전", "Rt": "우회전", "Pd": "보행",
             "Ut": "유턴", "Bc": "버스", "Bs": "버스"}
# 신호 상태명(SAE J2735 MovementPhaseState) → SUMO 상태 문자
STATE_CODES = {
    "protected-movement-allowed": "G",
    "permissive-movement-allowed": "g",
    "protected-clearance": "y",
    "permissive-clearance": "y",
    "stop-and-remain": "r",
    "stop-then-proceed": "s",
    "pre-movement": "u",
    "caution-conflicting-traffic": "o",
    "dark": "O",
    "unavailable": "O",
}
CODE_NAMES = {"G": "녹색", "g": "비보호녹색", "y": "황색/점멸", "r": "적색", "s": "적색후진행",
              "u": "적황", "o": "황색점멸", "O": "소등", "?": "미확인"}
SECRET_PARAMS = ("apikey", "api_key", "key", "servicekey", "authkey", "token")


# ───────────────────────────── 설정·HTTP ─────────────────────────────

def load_config():
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(CONFIG_TEMPLATE, encoding="utf-8")
        sys.exit(f"[설정] {CONFIG_PATH.name} 을 만들었습니다. API URL 을 채운 뒤 다시 실행하세요.")
    spec = importlib.util.spec_from_file_location("crossroad_api_config", CONFIG_PATH)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def mask_url(url):
    """인증키 쿼리값을 *** 로 가린 URL (meta.json 기록용)."""
    if not url:
        return ""
    parts = urllib.parse.urlsplit(url)
    query = [(k, "***" if k.lower() in SECRET_PARAMS else v)
             for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query, safe="{}*")))


class ApiError(RuntimeError):
    """API 가 4xx 를 돌려준 경우. 재시도해도 같은 결과이므로 재시도하지 않는다."""

    def __init__(self, status, body):
        self.status = status
        try:
            info = json.loads(body)
        except json.JSONDecodeError:
            info = {}
        self.code = info.get("code")
        self.next_call = info.get("nextCallAvailableAt")
        super().__init__(f"HTTP {status} {info.get('code') or ''} {info.get('message') or body[:200]}".strip())


# API 키당 일일 호출 한도 (실패 호출도 차감됨). 응답 헤더 X-RateLimit-* 로 갱신
RATE = {"limit": None, "remaining": None}
QUOTA_MARGIN = 20  # 수집 후에도 남겨 둘 호출 수


def _update_rate(headers):
    for key, name in (("limit", "X-RateLimit-Limit"), ("remaining", "X-RateLimit-Remaining")):
        if headers.get(name, "").isdigit():
            RATE[key] = int(headers[name])


def fetch_json(url, retries=3, timeout=5.0):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                _update_rate(resp.headers)
                body = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                raise ValueError(f"JSON 이 아닌 응답 (type=json 파라미터 확인): {body[:200]!r}")
        except urllib.error.HTTPError as e:
            _update_rate(e.headers)
            if 400 <= e.code < 500:
                raise ApiError(e.code, e.read().decode("utf-8", errors="replace"))
            last = e
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            last = e
        time.sleep(0.3 * (attempt + 1))
    raise RuntimeError(f"요청 실패: {last}")


def find_records(obj):
    """응답 JSON 안에서 레코드(dict) 리스트를 찾는다. 가장 긴 dict 리스트를 반환."""
    best = []
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            best = obj
        for x in obj:
            sub = find_records(x)
            if len(sub) > len(best):
                best = sub
    elif isinstance(obj, dict):
        if any(k in obj for k in ID_KEYS):
            best = [obj]
        for v in obj.values():
            sub = find_records(v)
            if len(sub) > len(best):
                best = sub
    return best


def _get(rec, keys):
    for k in keys:
        if rec.get(k) not in (None, ""):
            return str(rec[k])
    return None


def with_param(url, key, value):
    sep = "&" if urllib.parse.urlsplit(url).query else "?"
    return f"{url}{sep}{key}={urllib.parse.quote(str(value))}"


def fetch_records(url, itst_id=None, add_id_param=False):
    """
    URL 의 {itst_id} 를 채워 레코드를 받는다. {page} 가 있으면 빈 페이지까지 순회.
    add_id_param: URL 에 {itst_id}/itstId 가 없으면 itstId 파라미터를 붙인다 (신호 API 는 필수).
    """
    if add_id_param and itst_id and "{itst_id}" not in url and "itstId=" not in url:
        url = with_param(url, "itstId", itst_id)
    url = url.replace("{itst_id}", urllib.parse.quote(itst_id or ""))
    if "{page}" not in url:
        records = find_records(fetch_json(url))
    else:
        records, page = [], 1
        while True:
            chunk = find_records(fetch_json(url.replace("{page}", str(page))))
            if not chunk:
                break
            records += chunk
            page += 1
    if itst_id:
        records = [r for r in records if _get(r, ID_KEYS) in (None, itst_id)]
    return records


# ───────────────────────────── MAP 정보 ─────────────────────────────

def _norm(s):
    return re.sub(r"\s+", "", s or "")


def load_map_records(cfg, refresh=False):
    """
    MAP 전체 목록. MAP API 는 '호출 후 6시간 재호출 불가'(V2X_REPEAT_CALL_LIMIT)이므로
    한 번 받은 목록을 MAP_CACHE 에 저장하고 이후에는 캐시를 쓴다.
    """
    if MAP_CACHE.exists() and not refresh:
        return json.loads(MAP_CACHE.read_text(encoding="utf-8"))["records"]
    if not cfg.MAP_API_URL:
        sys.exit("[설정] MAP_API_URL 이 비어 있습니다.")
    if "{itst_id}" in cfg.MAP_API_URL:
        sys.exit("[설정] MAP_API_URL 에 {itst_id} 가 있으면 이름 검색이 불가합니다. --id 로 지정하세요.")
    try:
        records = fetch_records(cfg.MAP_API_URL)
    except ApiError as e:
        msg = f"[MAP] {e}"
        if e.next_call:
            msg += f"\n  → MAP API 다음 호출 가능: {e.next_call}. 그 전에는 --id 로 교차로 ID 를 지정하세요."
        sys.exit(msg)
    MAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    save_json(MAP_CACHE, {"fetched_at": datetime.now().isoformat(timespec="seconds"), "records": records})
    print(f"[MAP] 교차로 {len(records)}개 → {MAP_CACHE} 에 캐시")
    return records


def search_map(cfg, keyword, refresh=False):
    records = load_map_records(cfg, refresh)
    key = _norm(keyword)
    exact = [r for r in records if _norm(_get(r, NAME_KEYS)) == key]
    return exact or [r for r in records if key in _norm(_get(r, NAME_KEYS))]


def resolve_intersection(cfg, name, itst_id=None, refresh=False):
    """교차로 이름/ID → (ID, MAP 레코드). ID 를 알면 MAP 은 캐시가 있을 때만 참고한다."""
    itst_id = itst_id or getattr(cfg, "MANUAL_ITST_ID", {}).get(name) or None
    if itst_id:
        if not MAP_CACHE.exists() or refresh:
            return itst_id, None
        recs = [r for r in load_map_records(cfg) if _get(r, ID_KEYS) == itst_id]
        return itst_id, (recs[0] if recs else None)

    cands = search_map(cfg, name, refresh)
    if not cands:
        sys.exit(f"[MAP] '{name}' 교차로를 찾지 못했습니다. --search 로 후보를 확인하거나 --id 로 지정하세요.")
    if len(cands) > 1:
        print(f"[MAP] '{name}' 후보 {len(cands)}개:")
        for r in cands:
            print(f"   ID={_get(r, ID_KEYS)}  이름={_get(r, NAME_KEYS)}")
        sys.exit("  → --id 로 하나를 지정하세요.")
    return _get(cands[0], ID_KEYS), cands[0]


# ───────────────────────────── 신호 수집 ─────────────────────────────

# 신호 데이터 갱신 정지 판단 (실측: 갱신이 멈춘 교차로는 같은 dataId·trsmUtcTime 을 계속 돌려줌)
DATA_ID_KEY = "dataId"
TIME_KEY = "trsmUtcTime"  # 전송 시각 (epoch ms)
STALE_AGE_S = 600         # 첫 데이터가 이보다 오래됐으면 수집하지 않음
STALE_REPEAT = 10         # 같은 dataId 가 이만큼 연속되면 수집 중단


def data_age_s(record):
    try:
        return time.time() - float(record[TIME_KEY]) / 1000
    except (KeyError, TypeError, ValueError):
        return None


def collect_signal(cfg, itst_id, duration, interval, raw_path, share=1):
    """share: 남은 호출 한도를 나눠 쓸 교차로 수 (이번 교차로 포함)."""
    urls = [u for u in (cfg.SIGNAL_API_URL, getattr(cfg, "SIGNAL_TIMING_API_URL", "")) if u]
    if not urls:
        sys.exit("[설정] SIGNAL_API_URL 이 비어 있습니다.")

    # 첫 호출로 응답·남은 호출 수를 확인한 뒤, 한도를 넘지 않도록 수집 시간을 줄인다
    for u in urls:
        try:
            recs = fetch_records(u, itst_id, add_id_param=True)
        except ApiError as e:
            sys.exit(f"[신호] {e}")
        if not recs:
            sys.exit(f"[신호] 교차로 {itst_id} 의 신호 레코드가 없습니다 (신호 API 제공 대상인지 확인).")
        age = data_age_s(recs[0])
        if age is not None:
            print(f"[신호] 최신 데이터 전송 시각: {age:.0f}초 전")
            if age > STALE_AGE_S:
                sys.exit(f"[신호] 데이터가 {age / 3600:.1f}시간 전 것입니다. 실시간 갱신이 멈춘 교차로라 수집하지 않습니다.")
    if RATE["remaining"] is not None:
        budget = (RATE["remaining"] - QUOTA_MARGIN) // len(urls) // share
        if budget < duration / interval:
            new_duration = max(budget, 0) * interval
            print(f"[호출 한도] 남은 호출 {RATE['remaining']}회 → 수집 시간 {duration:.0f}s 를 {new_duration:.0f}s 로 줄입니다.")
            duration = new_duration
    print(f"[신호] {duration:.0f}초 동안 {interval}초 간격 수집 "
          f"(예상 {int(duration / interval)}회 × API {len(urls)}개, 남은 호출 {RATE['remaining']})")

    t0 = time.monotonic()
    n = n_ok = same_count = 0
    last_id = None
    with open(raw_path, "w", encoding="utf-8") as f:
        while time.monotonic() - t0 < duration:
            t_req = time.monotonic()
            merged, errors = {}, []
            for u in urls:
                try:
                    recs = fetch_records(u, itst_id, add_id_param=True)
                    if recs:
                        merged.update(recs[0])
                    else:
                        errors.append(f"교차로 {itst_id} 레코드 없음")
                except ApiError as e:
                    errors.append(str(e))
                    if e.status == 429:
                        print(f"[신호] 호출 한도 초과로 중단: {e}")
                        duration = 0
                except Exception as e:
                    errors.append(str(e))
            if RATE["remaining"] is not None and RATE["remaining"] <= QUOTA_MARGIN:
                print(f"[호출 한도] 남은 호출 {RATE['remaining']}회 → 수집 중단")
                duration = 0
            # 같은 메시지가 계속 오면 갱신이 멈춘 것이므로 호출을 아끼기 위해 중단
            data_id = merged.get(DATA_ID_KEY)
            same_count = same_count + 1 if data_id and data_id == last_id else 0
            last_id = data_id or last_id
            if same_count >= STALE_REPEAT:
                print(f"[신호] 같은 데이터({data_id})가 {same_count + 1}회 연속 → 갱신 정지로 판단, 수집 중단")
                duration = 0
            # 응답 시각은 요청 전후 중간값으로 기록
            t_mid = (t_req + time.monotonic()) / 2 - t0
            line = {"t": round(t_mid, 3), "wall": datetime.now().isoformat(timespec="milliseconds"),
                    "record": merged or None}
            if errors:
                line["errors"] = errors
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()

            n += 1
            n_ok += bool(merged)
            if n == 1 and not merged:
                print(f"   첫 호출 실패: {errors}")
            if n % 60 == 0:
                print(f"   {t_mid:6.0f}s  수집 {n_ok}/{n}")
            time.sleep(max(0.0, t0 + n * interval - time.monotonic()))
    print(f"[신호] 완료: 성공 {n_ok}/{n}")
    return n, n_ok


def parse_signal(record, unit):
    """레코드 → {이동류 키: (상태 문자, 잔여시간 s)}. 키 예) 'nt_St'."""
    out = {}
    for field, val in (record or {}).items():
        m = SIGNAL_FIELD_RE.match(field)
        if not m or val in (None, ""):
            continue
        key = f"{m.group(1)}_{m.group(2)}"
        code, rem = out.get(key, (None, None))
        if m.group(3) == "StatNm":
            code = STATE_CODES.get(str(val).strip().lower(), "?")
        else:
            try:
                rem = float(val) * unit
            except (TypeError, ValueError):
                pass
        out[key] = (code, rem)
    return out


def movement_label(key):
    d, m = key.split("_")
    return f"{DIRECTIONS.get(d, d)} {MOVEMENTS.get(m, m)}"


def load_samples(raw_path, unit):
    samples = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("record"):
                samples.append((row["t"], row["wall"], parse_signal(row["record"], unit)))
    return samples


def write_timeline(samples, csv_path):
    keys = sorted({k for _, _, s in samples for k in s})
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["t", "wall"] + [f"{movement_label(k)}_{c}" for k in keys for c in ("상태", "잔여s")])
        for t, wall, s in samples:
            row = [t, wall]
            for k in keys:
                code, rem = s.get(k, (None, None))
                row += [code or "", "" if rem is None else round(rem, 1)]
            w.writerow(row)
    return keys


# ───────────────────────────── 단계 분석 ─────────────────────────────

def analyze(samples):
    """
    모든 이동류의 상태 조합(signature)이 같은 구간을 하나의 단계로 본다.
    단계 종료 시각은 '바뀌는 이동류의 (시각 + 잔여시간)' 중앙값으로 보정한다.
    처음·마지막 단계는 잘려 있으므로 통계에서 제외한다.
    """
    keys = sorted({k for _, _, s in samples for k in s if s[k][0] not in (None, "O")})
    if not keys:
        return {"error": "해석 가능한 신호 필드가 없습니다. SIGNAL_FIELD_RE / STATE_CODES 를 확인하세요."}

    # 누락값은 직전 값으로 채운다
    seq, last = [], {}
    for t, _, s in samples:
        for k in keys:
            if s.get(k, (None, None))[0] is not None:
                last[k] = s[k]
        if len(last) == len(keys):
            seq.append((t, {k: last[k] for k in keys}))
    if not seq:
        return {"error": "모든 이동류가 채워진 샘플이 없습니다."}

    # 상태 조합이 같은 연속 구간
    segs = []
    for t, st in seq:
        sig = tuple(st[k][0] for k in keys)
        if segs and segs[-1]["sig"] == sig:
            segs[-1]["samples"].append((t, st))
        else:
            segs.append({"sig": sig, "samples": [(t, st)]})

    # 경계 시각 보정
    for i, seg in enumerate(segs[:-1]):
        nxt = segs[i + 1]
        changed = [k for j, k in enumerate(keys) if seg["sig"][j] != nxt["sig"][j]]
        est = [t + st[k][1] for t, st in seg["samples"] for k in changed if st[k][1] is not None]
        t_last, t_next = seg["samples"][-1][0], nxt["samples"][0][0]
        end = statistics.median(est) if est else (t_last + t_next) / 2
        seg["end"] = min(max(end, t_last), t_next)  # 관측 범위 밖으로 벗어나지 않게
    for i in range(1, len(segs) - 1):
        segs[i]["start"] = segs[i - 1]["end"]
        segs[i]["duration"] = segs[i]["end"] - segs[i]["start"]
    full = segs[1:-1]

    def describe(sig):
        groups = {}
        for k, c in zip(keys, sig):
            groups.setdefault(CODE_NAMES.get(c, c), []).append(movement_label(k))
        return groups

    # 같은 상태 조합별 시간 통계
    by_sig = {}
    for s in full:
        by_sig.setdefault(s["sig"], []).append(s["duration"])

    # 주기: 첫 완전 단계와 같은 조합이 다시 나타나는 간격
    order, cycles = [], []
    if full:
        first = full[0]["sig"]
        starts = [s["start"] for s in full if s["sig"] == first]
        cycles = [b - a for a, b in zip(starts, starts[1:])]
        order = [first]
        for s in full[1:]:
            if s["sig"] == first:
                break
            order.append(s["sig"])

    phases = []
    for i, sig in enumerate(order):
        d = by_sig[sig]
        phases.append({
            "index": i,
            "duration_s": round(statistics.median(d), 1),
            "duration_min_max_s": [round(min(d), 1), round(max(d), 1)],
            "observed": len(d),
            "state": "".join(sig),
            "signals": describe(sig),
        })

    movement_green = {}
    for j, k in enumerate(keys):
        g = sum(p["duration_s"] for p, sig in zip(phases, order) if sig[j] in ("G", "g"))
        movement_green[movement_label(k)] = round(g, 1)

    return {
        "movements": [{"key": k, "label": movement_label(k)} for k in keys],
        "state_legend": CODE_NAMES,
        "cycle_s": round(statistics.median(cycles), 1) if cycles else None,
        "cycles_observed": len(cycles),
        "cycle_list_s": [round(c, 1) for c in cycles],
        "phase_sum_s": round(sum(p["duration_s"] for p in phases), 1),
        "phases": phases,
        "green_time_per_movement_s": movement_green,
        "other_states_seen": [
            {"state": "".join(sig), "signals": describe(sig), "durations_s": [round(x, 1) for x in d]}
            for sig, d in by_sig.items() if sig not in order
        ],
        "segments_total": len(segs),
    }


def print_summary(name, summary):
    print(f"\n─── {name} 신호 분석 ───")
    if "error" in summary:
        print("  " + summary["error"])
        return
    print(f"  주기: {summary['cycle_s']} s (관측 {summary['cycles_observed']}회), 단계 합계 {summary['phase_sum_s']} s")
    for p in summary["phases"]:
        sig = ", ".join(f"{state}: {'/'.join(m)}" for state, m in p["signals"].items() if state != "적색")
        print(f"  [{p['index']}] {p['duration_s']:5.1f}s  {p['state']}  {sig}")
    if summary["other_states_seen"]:
        print(f"  ※ 주기 밖 상태 조합 {len(summary['other_states_seen'])}개 (signal_summary.json 참고)")


# ───────────────────────────── 실행 ─────────────────────────────

def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def run(cfg, name, itst_id, duration, interval, analyze_only, refresh_map=False, share=1):
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "signal_raw.jsonl"
    unit = getattr(cfg, "REMAINING_TIME_UNIT", 0.1)

    if not analyze_only:
        itst_id, map_rec = resolve_intersection(cfg, name, itst_id, refresh_map)
        print(f"[MAP] {name} → 교차로 ID {itst_id}")
        save_json(out_dir / "map_info.json", {
            "name": name, "itst_id": itst_id, "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "record": map_rec,
        })
        save_json(out_dir / "traffic_volume.json", {
            "name": name, **TRAFFIC_SOURCE, "roads": TRAFFIC_VOLUME.get(name, []),
        })

        started = datetime.now()
        n, n_ok = collect_signal(cfg, itst_id, duration, interval, raw_path, share)
        save_json(out_dir / "meta.json", {
            "name": name, "itst_id": itst_id,
            "started_at": started.isoformat(timespec="seconds"),
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "weekday": "월화수목금토일"[started.weekday()],
            "duration_s": duration, "interval_s": interval,
            "requests": n, "successful": n_ok,
            "remaining_time_unit_s": unit,
            "map_api_url": mask_url(cfg.MAP_API_URL),
            "signal_api_url": mask_url(cfg.SIGNAL_API_URL),
            "signal_timing_api_url": mask_url(getattr(cfg, "SIGNAL_TIMING_API_URL", "")),
        })
    elif not raw_path.exists():
        sys.exit(f"[분석] {raw_path} 가 없습니다.")

    samples = load_samples(raw_path, unit)
    write_timeline(samples, out_dir / "signal_timeline.csv")
    summary = analyze(samples)
    summary = {"name": name, "samples": len(samples), **summary}
    save_json(out_dir / "signal_summary.json", summary)
    print_summary(name, summary)
    print(f"  저장: {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="서울 T-DATA 교차로 MAP·신호 데이터 수집")
    ap.add_argument("--name", nargs="+", help="교차로 이름 (여러 개 가능, 순서대로 수집)")
    ap.add_argument("--id", help="교차로 ID 직접 지정 (--name 이 하나일 때)")
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="교차로당 신호 수집 시간 (초)")
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="신호 API 호출 간격 (초)")
    ap.add_argument("--analyze-only", action="store_true", help="수집 없이 기존 signal_raw.jsonl 재분석")
    ap.add_argument("--search", help="MAP 정보에서 이름에 키워드가 들어간 교차로 검색")
    ap.add_argument("--refresh-map", action="store_true",
                    help="MAP 캐시를 무시하고 API 를 다시 호출 (6시간에 1회만 가능)")
    args = ap.parse_args()

    cfg = load_config()
    if args.search:
        for r in search_map(cfg, args.search, args.refresh_map):
            print(f"ID={_get(r, ID_KEYS)}  이름={_get(r, NAME_KEYS)}")
        return
    if not args.name:
        ap.error("--name 또는 --search 가 필요합니다.")
    if args.id and len(args.name) > 1:
        ap.error("--id 는 --name 이 하나일 때만 사용할 수 있습니다.")
    for i, name in enumerate(args.name):
        run(cfg, name, args.id, args.duration, args.interval, args.analyze_only,
            args.refresh_map and i == 0, share=len(args.name) - i)


if __name__ == "__main__":
    main()
