"""SUMO 고속도로 RL 환경 — Gymnasium 인터페이스 + TraCI.

┌─────────────────────────────────────────────────────────────┐
│  강화학습 루프에서 이 클래스의 역할                              │
│                                                             │
│    obs, _ = env.reset()          # 에피소드 시작, 초기 관측     │
│    obs, r, term, trunc, _ = env.step(action)  # 1스텝 진행    │
│                                                             │
│  내부적으로는 TraCI(Traffic Control Interface)라는 소켓 API로  │
│  SUMO 시뮬레이터를 원격 조종한다:                               │
│    traci.simulationStep()  → 시뮬레이션을 1스텝 진행            │
│    traci.vehicle.setSpeed() → 차량 속도 명령                   │
│    traci.vehicle.getSpeed() 등 → 상태 조회                     │
└─────────────────────────────────────────────────────────────┘

관측/행동/보상의 구체적인 정의는 configs/mdp_config.py 에 있고,
이 파일은 그 설정을 받아서 실행만 한다. (MDP를 바꿀 때 이 파일 수정 불필요)
"""
import os
import sys
import csv
import json

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# SUMO 설치 폴더의 tools/ 안에 traci, sumolib 파이썬 모듈이 들어있다.
# pip로도 설치했지만, 버전 불일치를 피하려면 SUMO_HOME 쪽을 우선 경로에 추가.
#
# SUMO_HOME이 설정 안 돼 있으면 OS별 기본 설치 경로를 순서대로 탐색해서
# 자동으로 잡아준다 (환경변수 설정을 깜빡해도 동작하도록).
if "SUMO_HOME" not in os.environ:
    _candidates = [
        # macOS (Homebrew) — Apple Silicon / Intel
        "/opt/homebrew/opt/sumo/share/sumo",
        "/usr/local/opt/sumo/share/sumo",
        # Linux (apt)
        "/usr/share/sumo",
        # Windows (기본 인스톨러 경로)
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
    ]
    for _p in _candidates:
        if os.path.isdir(_p):
            os.environ["SUMO_HOME"] = _p
            print(f"[sumo_env] SUMO_HOME 자동 감지: {_p}")
            break
    else:
        raise RuntimeError(
            "SUMO_HOME 환경변수가 없고 기본 경로에서도 SUMO를 찾지 못했습니다.\n"
            "  Windows: 인스톨러 재설치(SUMO_HOME 체크) 또는 환경변수 수동 설정\n"
            "  macOS  : ./setup_mac.sh 실행 또는 README_MAC.md 참고\n"
            "  Linux  : sudo apt install sumo sumo-tools 후 "
            "export SUMO_HOME=/usr/share/sumo")

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci
from sumolib import checkBinary

from env.safety_filter import PedSafetyFilter


def _load_zone_rows(filename, columns):
    """data/road_data의 CSV를 읽어 columns 순서의 튜플 목록으로 만든다.

    숫자로 적힌 칸은 float이 된다. 파일이 없으면 빈 목록.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "road_data", filename)
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            item = []
            for col in columns:
                val = (row.get(col) or "").strip()
                try:
                    val = float(val)
                except ValueError:
                    pass
                item.append(val)
            rows.append(tuple(item))
    return rows


# 노선별 정차 순서는 시나리오 생성 시 cgc.line_meta.json으로 저장된다 (config/lines.yaml).
# 아래는 메타 파일이 없는 구버전 시나리오용 fallback (청계A01 동행 5곳).
BUS_STOPS_ORDER = ["bs_01", "bs_02", "bs_03", "bs_04", "bs_05"]

# 학습 판에서 빼는 판 시드 구간 (테스트·검증용, rules/team/experiment_protocol.md 3-2)
#   1000~1099: 팀 비교, 2000~2999: 최종 후보 평가,
#   900000~999999: 학습 중 best 고르기·튜닝.
# 1100~1999도 이전 프로토콜에서 테스트로 예약했으므로 계속 제외한다.
HELD_OUT_SEEDS = ((1000, 2999), (900_000, 999_999))

# 팀원 BC 수집·평가 판. HELD_OUT_SEEDS와 별도이고, 학습 판 샘플링에서만 거른다.
BC_RESERVED_SEEDS = ((400_000, 400_099), (470_000, 470_059))

# 신호색 one-hot 순서 (관측·데이터 공통): 적색 / 황색 / 녹색 / 일시정지 후 진행(s)
SIGNAL_COLORS = ("red", "yellow", "green", "stop")
_COLOR_OF_STATE = {"r": "red", "u": "red", "y": "yellow",
                   "G": "green", "g": "green", "o": "green", "O": "green", "s": "stop"}
# 데이터셋 state.sig{k}.* 칸 수 (경로 r0의 신호 정지선 수, 청계2가는 정지선 2개).
# 신호가 없는 단계(1단계)에서도 칸을 만들고 -1로 채워, 모든 단계의 기록 파일 칸 구성을 같게 한다.
STATE_SIGNAL_SLOTS = 7

# SUMO 링크 방향 → (좌, 직, 우) 인덱스. 유턴(t)은 좌로 본다
_DIR_INDEX = {"l": 0, "L": 0, "t": 0, "s": 1, "r": 2, "R": 2}

# 프리필: 같은 차로에 두 차를 이 거리(m)보다 가깝게 배치하지 않는다
# (_prefill_traffic의 최소 배치 간격과 같은 값)
_PREFILL_MIN_GAP = 10.0


class SumoHighwayEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg_path, road, ego, mdp_sim, mdp_obs, action, reward,
                 traffic=None, gui=False, gui_autostart=True):
        """
        Args:
            cfg_path: road_builder.build()가 만든 .sumocfg 경로
            road:     configs.road_config.ROAD
            ego:      configs.road_config.EGO (가속/감속 한계가 여기서 옴)
            mdp_sim:  configs.mdp_config.SIMULATION
            mdp_obs:  configs.mdp_config.OBSERVATION
            action:   configs.mdp_config.ACTION (행동 차원 수)
            reward:   configs.mdp_config.REWARD
            traffic:  configs.road_config.TRAFFIC (프리필/컨트롤러 정보).
                None이면 프리필 없이 flow만으로 동작 (하위호환)
            gui:      True면 sumo-gui(화면 있는 버전)로 실행
            gui_autostart: (gui=True일 때만 의미)
                True  → 창이 뜨면 시뮬레이션이 자동으로 시작
                False → 사용자가 ▶(플레이) 버튼을 누를 때까지 대기.
                        sumo-gui는 --start 옵션이 없으면 TraCI의
                        simulationStep() 호출도 플레이 전까지 블록되므로,
                        에피소드 전체가 "▶를 눌러야 시작"되는 효과가 난다.
        """
        super().__init__()
        self.cfg_path = cfg_path
        self.gui = gui
        self.gui_autostart = gui_autostart
        # 노선 메타 (route id → 정차할 정류장 순서). make_cgc_scenario가 만든다.
        self._line_stops = {}
        meta_path = os.path.join(os.path.dirname(os.path.abspath(cfg_path)),
                                 "cgc.line_meta.json")
        self._line_names = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                lines_meta = json.load(f)["lines"]
            self._line_stops = {route_id: d.get("stops", [])
                                for route_id, d in lines_meta.items()}
            self._line_names = {route_id: d.get("line")
                                for route_id, d in lines_meta.items()}

        # ----- 도로/차량 설정에서 필요한 값들 -----
        self.vmax = float(road["speed_limit"])   # 속도 정규화 기준
        # 단계별 맵 (env/curriculum.py): 이번 맵에서 켜진 요소. 없으면(예전 설정) 전부 켜짐
        from env.curriculum import has as _has
        self.stage = road.get("stage")
        self.has_parking = _has(road, "illegal_parking")
        self.has_school = _has(road, "school_zone")
        self.num_lanes = int(road["num_lanes"])
        # 시작 차선: "center"면 중앙 차선(차선수//2)으로 자동 결정.
        #   3차선 → 1번(가운데), 5차선 → 2번. 차선 수를 바꿔도 항상 가운데라
        #   좌/우 양쪽으로 변경 여지가 대칭으로 생긴다 (탐험에 유리).
        _dl = ego["depart_lane"]
        if str(_dl).lower() == "center":
            self.ego_depart_lane = str(int(road["num_lanes"]) // 2)
        else:
            self.ego_depart_lane = str(_dl)
        # ---- 프리필(도로 전체 사전 배치) 설정 ----
        self.traffic = traffic or {}
        pf = (self.traffic.get("prefill") or {})
        self.prefill_enabled = bool(pf.get("enabled", False))
        self.prefill_start = float(pf.get("start_offset", 80.0))
        self.prefill_end_margin = float(pf.get("end_margin", 50.0))
        self.prefill_jitter = float(pf.get("spacing_jitter", 0.3))
        # ---- 차선변경 안전 게이트 설정 (mdp_config.ACTION["lane_change_safety"]) ----
        lcs = (action.get("lane_change_safety") or {})
        self.lc_front_min = float(lcs.get("min_front_gap", 5.0))   # m
        self.lc_rear_min = float(lcs.get("min_rear_gap", 5.0))     # m
        self.lc_front_tau = float(lcs.get("front_tau", 0.3))       # s
        self.lc_rear_tau = float(lcs.get("rear_tau", 0.5))         # s
        self.lc_spawn_protect = float(lcs.get("spawn_protect", 60.0))  # m
        self.lc_horizon = float(lcs.get("approach_horizon", 1.0))       # s
        self.lc_signal_zone = float(lcs.get("signal_no_change_zone", 0.0))  # m
        self.road_length = float(road["length"])
        self.traffic_max_speed = float(self.traffic.get("max_speed", self.vmax))
        # 차선당 배치 간격: 지정값이 없으면 교통류 이론대로 자동 계산
        #   차선당 유량 q_lane = (vehs/h) / 3600 / 차선수  [대/초]
        #   간격 = 속도 / q_lane   (그 속도로 흐를 때 차 사이 평균 거리)
        if pf.get("spacing"):
            self.prefill_spacing = float(pf["spacing"])
        else:
            vph = float(self.traffic.get("vehs_per_hour", 1800))
            q_lane = max(vph / 3600.0 / max(self.num_lanes, 1), 1e-6)
            self.prefill_spacing = self.traffic_max_speed / q_lane
        # 컨트롤러 타입 샘플링 준비 (rou.xml의 car_<이름> vType과 대응)
        ctrls = self.traffic.get("controllers") or {}
        if ctrls:
            names = list(ctrls.keys())
            probs = np.array([float(ctrls[n].get("probability", 1.0)) for n in names])
            self._prefill_types = [f"car_{n}" for n in names]
            self._prefill_probs = probs / probs.sum()
        else:
            self._prefill_types, self._prefill_probs = ["car"], np.array([1.0])
        self.ego_depart_speed = str(ego["depart_speed"])
        # 행동 a ∈ [-1,1] 을 실제 가속도로 변환할 때 쓰는 스케일:
        #   a = +1 → +max_accel (풀 액셀),  a = -1 → -max_decel (풀 브레이크)
        self.max_accel = float(ego["accel"])   # m/s^2
        self.max_decel = float(ego["decel"])   # m/s^2 (양수로 저장, 부호는 계산 시)
        # ego가 낼 수 있는 최고속도 (m/s). speedMode=0이면 SUMO가 vType maxSpeed를 지켜 주지 않으므로
        # setSpeed 상한을 여기서 건다. 관측 정규화 기준(vmax)과는 별개다.
        self.ego_vmax = min(self.vmax, float(ego.get("max_speed", self.vmax)))

        # ----- MDP 설정 -----
        self.step_length = float(mdp_sim["step_length"])
        self.max_steps = int(mdp_sim["max_steps"])
        self.warmup_steps = int(mdp_sim["warmup_steps"])
        # 보행자 안전 필터: shield의 한 종류. 판정용 계산(보행자 앞 대기 여부)에도 쓰므로 항상 만든다
        self.ped_filter = PedSafetyFilter(self)
        self.visibility = float(mdp_obs["visibility"])   # 관측 거리 W
        # 밀도 정규화용 "차량 1대가 점유하는 길이" (차체 + 최소 간격)
        self.density_veh_len = float(mdp_obs.get("density_veh_length", 7.5))
        # 전방 신호 관측 (mdp_config.OBSERVATION["signal"])
        sig = (mdp_obs.get("signal") or {})
        self.signal_obs = bool(sig.get("enabled", False))
        self.signal_visibility = float(sig.get("visibility", 150.0))
        # 경로 안내 관측: 다음 교차로 회전 방향 + 차로별 진행 가능 방향(노면 화살표)
        self.route_dir_obs = bool((mdp_obs.get("route_dir") or {}).get("enabled", False))
        # 지금 위치가 차선변경 금지 구간인지 (막지 않고 감점하므로 보여 준다)
        self.lc_zone_obs = bool((mdp_obs.get("lc_zone") or {}).get("enabled", False))
        # 경로상 횡단보도 관측 (mdp_config.OBSERVATION["pedestrian"])
        ped = (mdp_obs.get("pedestrian") or {})
        self.ped_obs = bool(ped.get("enabled", False))
        self.ped_visibility = float(ped.get("visibility", 50.0))
        self.ped_num_crossings = int(ped.get("num_crossings", 2))
        # 어린이 보호구역 관측 (mdp_config.OBSERVATION["school_zone"])
        szo = (mdp_obs.get("school_zone") or {})
        self.zone_obs = bool(szo.get("enabled", False))
        self.zone_visibility = float(szo.get("visibility", 100.0))
        # 버스정류장 (mdp_config.OBSERVATION["bus_stop"]): 관측 3개. 정차는 모델이 직접 한다.
        #   정차 성공 = 정류장 차로·정류장 구간 안에서 stop_speed 이하로 dwell_time 초 이상 정지
        #   (SUMO 정류장 등록·속도 위임·정류장 규칙은 쓰지 않는다: rules/team/experiment_protocol.md A1, A5)
        bso = (mdp_obs.get("bus_stop") or {})
        self.stop_obs = bool(bso.get("enabled", False))
        self.stop_visibility = float(bso.get("visibility", 150.0))
        self.stop_dwell_time = float(bso.get("dwell_time", 10.0))
        self.stop_speed = float(bso.get("stop_speed", 0.1))
        self.stop_overshoot = float(bso.get("overshoot", 1.0))   # 정류장 끝을 앞범퍼가 넘어도 되는 여유 m
        # ---- shield (수동개입) — 차량·신호 shield, 보행자 필터, 차선변경 게이트를 한 스위치로 ----
        #   "off"    : 계산하지 않는다 (학습 기본값)
        #   "on"     : 위험한 명령을 덮어쓴다 (참고용 평가)
        #   "shadow" : 덮어쓰지 않고 "개입했을지"만 info에 기록한다 (팀 비교 테스트)
        sh = (mdp_sim.get("shield") or {})
        self.shield_mode = str(sh.get("mode", "off"))
        if self.shield_mode not in ("off", "on", "shadow"):
            raise ValueError(f"shield mode는 off/on/shadow 중 하나: {self.shield_mode}")
        # 안전 속도 계산에 쓰는 제동 m/s². 차가 낼 수 있는 감속(max_decel)보다 크면 안 된다
        self.shield_decel = min(float(sh.get("decel", 3.0)), self.max_decel)
        self.shield_margin = float(sh.get("margin", 2.5))   # 정지 후 남길 간격 m
        self.signal_margin = float(sh.get("signal_margin", 1.0))   # 정지선 앞 여유 m
        # 이 감속 이상이 필요하면(급정거) 신호 shield는 개입하지 않는다
        self.signal_max_decel = min(float(sh.get("signal_max_decel", self.max_decel)), self.max_decel)
        # 평가 전용 모드: True면 실패 판 다시 열기 목록을 바꾸지 않는다 (utils/evaluator.py가 켠다)
        self.eval_mode = False
        # 실패한 판 다시 연습 (Prioritized Level Replay를 단순화: Jiang et al. 2021).
        # 학습 중 완주 못 한 판의 시드를 모아 두고 reset 때 replay_prob 확률로 그 판을 다시 연다.
        # 완주하면 목록에서 뺀다. 같은 시드 = 같은 초기 교통·신호·경로라 어려운 상황을 반복 연습.
        rp = (mdp_sim.get("replay") or {})
        self.replay_prob = float(rp.get("prob", 0.0))
        self.replay_size = int(rp.get("size", 200))
        self._replay_seeds = []
        self._ep_seed = None
        self._ep_replayed = False
        # ego 경로 후보와 확률 (road_config.EGO["routes"]). 네트워크에 없는 route는 reset에서 제외
        self.ego_routes = dict(ego.get("routes") or {"r0": 1.0})
        self.route_depart_lane = dict(ego.get("route_depart_lane") or {})
        self._crossings, self._edge_ends = self._load_net_crossings()
        self._zones = []           # 어린이 보호구역 [(시작, 끝, 제한속도)] — reset에서 노면을 읽어 채움
        self.act_dim = int(action["dim"])
        self.accel_index = int(action.get("accel_index", 0))
        self.lane_change_index = int(action.get("lane_change_index", 1))
        self.lane_change_threshold = float(action.get("lane_change_threshold", 0.5))
        self.lane_change_duration = float(action.get("lane_change_duration", 1.0))
        self.rw = reward
        # 정차금지지대 (edge, 시작 m, 끝 m). 파일은 data/road_data/no_stop_zones.csv
        # 행마다 엣지 범위(edge_id, start_pos, end_pos) 또는 교차로 전체(junction_id: 내부 차로 전부)
        _ns_rows = _load_zone_rows(
            "no_stop_zones.csv", ("edge_id", "start_pos", "end_pos", "junction_id"))
        self._no_stop_zones = [(e, a, b) for e, a, b, _ in _ns_rows if e]
        self._no_stop_junctions = {str(j) for _, _, _, j in _ns_rows if j}
        # 불법주정차 다발지점 (차로 점유). reset마다 지점별 prob 확률로 0초부터 정차 상태로 세운다.
        self._hotspots = _load_zone_rows(
            "illegal_parking_hotspots.csv",
            ("hotspot_id", "edge_id", "start_pos", "end_pos", "vehicles", "prob")
        ) if self.has_parking else []
        self._hotspot_taken = {}   # (엣지, 차선 번호) → 다발지점 차량 위치 (프리필이 피함)
        self._hotspot_placed = []  # 이번 판에 생성된 다발지점 (id, 엣지, 차량 수)
        # 어린이보호구역 edge → 제한속도 m/s
        self._school_limits = {
            edge: speed for edge, speed in _load_zone_rows(
                "school_zones.csv", ("edge_id", "speed_limit_ms"))
        } if self.has_school else {}

        # GUI는 ego를 따라가며 가까이 보이도록 기본값을 둔다.
        self.gui_zoom = float(mdp_sim.get("gui_zoom", 1200.0))
        self.gui_delay = float(mdp_sim.get("gui_delay", 100.0))
        self.gui_track_ego = bool(mdp_sim.get("gui_track_ego", True))

        self.ego_id = "ego"        # SUMO 안에서 ego 차량을 식별하는 ID
        self.step_count = 0
        self._previous_policy_lane_change = None
        # 정류장 진행 상태 (reset에서 채움): 이번 판 정류장 목록, 다음 정류장 번호, 연속 정지 스텝
        self._stops = []
        self._stop_i = 0
        self._dwell_steps = 0
        self._missed_stops = 0
        self._stops_done = 0
        self._stop_errors = []     # 정차 성공 때 앞범퍼 ↔ 정류장 끝 거리 (m, 정차 위치 오차)
        self._onboard = []         # 버스에 탄 승객 [(내릴 정류장, 내린 뒤 이어갈 stage들)]
        self._pax_seq = 0
        self._started = False      # SUMO 프로세스가 떠 있는지 여부

        # Gymnasium 행동 공간:
        #   action[0] = 종방향 가감속 raw ∈ [-1,1]
        #   action[1] = 차선변경 raw ∈ [-1,1], 환경 내부에서 {-1,0,+1}로 양자화
        # PPO의 Gaussian 정책과 호환하기 위해 Box를 유지한다. 실제 적용된
        # 차선변경 명령은 info["lane_change"]에 정확한 -1/0/+1로 반환한다.
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.act_dim,), dtype=np.float32)
        # 관측: 범위 [-1,1]. 레이아웃은 env/mdp_config.py 주석과 self.obs_blocks 참고.
        #   시야 차선: 좌2, 좌1, 현재, 우1, 우2 (5개) — 3차선 도로에서
        #   맨 왼쪽에 있어도 오프셋 -2로 맨 오른쪽 차선까지 보인다.
        self.obs_offsets = (+2, +1, 0, -1, -2)
        n_lanes_view = len(self.obs_offsets)
        # 관측 블록 (이름, 차원) — 순서가 곧 관측 벡터 순서. 인덱스는 self.obs_index(이름)로 찾는다
        blocks = [("speed", 1), ("neighbors", n_lanes_view * 4),
                  ("lane_end", n_lanes_view), ("density", n_lanes_view)]
        if self.signal_obs:
            blocks.append(("signal", 2 + len(SIGNAL_COLORS)))   # 거리 + 색 4 + 보이는지
        if self.route_dir_obs:
            blocks.append(("turn", 3))                          # 다음 교차로 좌/직/우
            blocks.append(("lane_arrows", n_lanes_view * 3))    # 차로별 좌/직/우 화살표
        if self.ped_obs:
            blocks.append(("crossings", 3 * self.ped_num_crossings))
        if self.zone_obs:
            blocks.append(("school_zone", 2))
        if self.stop_obs:
            blocks.append(("bus_stop", 3))   # 거리 / 정류장 차로까지 차로 차이 / 정지 시간 진행도
        if self.lc_zone_obs:
            blocks.append(("lc_zone", 1))
        self.obs_blocks, start = {}, 0
        for name, n in blocks:
            self.obs_blocks[name] = (start, start + n)
            start += n
        self.obs_dim = start
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.obs_dim,),
                                            dtype=np.float32)

    def obs_index(self, name: str) -> slice:
        """관측 블록 이름 → 관측 벡터 slice (예: obs[env.obs_index("signal")]).

        관측 차원이 바뀌어도 스크립트가 고정 인덱스(obs[40] 등) 대신 이름으로 찾게 한다."""
        lo, hi = self.obs_blocks[name]
        return slice(lo, hi)

    _LANE_TAGS = ("L2", "L1", "C", "R1", "R2")   # obs_offsets(+2,+1,0,-1,-2) 순서의 차로 이름

    def obs_names(self) -> list:
        """관측 벡터 각 칸의 컬럼 이름 (데이터셋 obs.* 컬럼, 길이 = obs_dim).

        차로 이름: L2/L1 = 왼쪽 두/한 칸, C = 현재 차로, R1/R2 = 오른쪽 한/두 칸."""
        names = []
        for block in self.obs_blocks:
            if block == "speed":
                names.append("obs.speed")
            elif block == "neighbors":
                for tag in self._LANE_TAGS:
                    names += [f"obs.{tag}.lead_gap", f"obs.{tag}.lead_dv",
                              f"obs.{tag}.follow_gap", f"obs.{tag}.follow_dv"]
            elif block in ("lane_end", "density"):
                names += [f"obs.{tag}.{block}" for tag in self._LANE_TAGS]
            elif block == "signal":
                names += (["obs.signal.dist"] + [f"obs.signal.{c}" for c in SIGNAL_COLORS]
                          + ["obs.signal.visible"])
            elif block == "turn":
                names += ["obs.turn.left", "obs.turn.straight", "obs.turn.right"]
            elif block == "lane_arrows":
                for tag in self._LANE_TAGS:
                    names += [f"obs.{tag}.arrow_left", f"obs.{tag}.arrow_straight",
                              f"obs.{tag}.arrow_right"]
            elif block == "crossings":
                for k in range(self.ped_num_crossings):
                    names += [f"obs.crossing{k}.dist", f"obs.crossing{k}.ped_gap",
                              f"obs.crossing{k}.ped_toward"]
            elif block == "school_zone":
                names += ["obs.school_zone.dist", "obs.school_zone.limit"]
            elif block == "bus_stop":
                names += ["obs.bus_stop.dist", "obs.bus_stop.lane_diff", "obs.bus_stop.dwell"]
            elif block == "lc_zone":
                names.append("obs.lc_zone")
        assert len(names) == self.obs_dim
        return names

    def _load_net_crossings(self):
        """net.xml에서 (횡단보도 → 가로지르는 엣지들, 일반 엣지 → (from, to) 노드)를 읽는다.

        traci로는 횡단보도가 어느 엣지를 가로지르는지와 엣지의 양 끝 노드를
        알 수 없어서 (이 SUMO 버전엔 edge.getToJunction이 없음) 파일에서 직접 읽는다.
        """
        import xml.etree.ElementTree as ET
        # 함께 읽는 것 (net.xml에만 있는 정보):
        #   _no_change_edges : 승용차 차로변경이 금지된 엣지 (어린이 보호구역 지그재그 구간)
        #   _junction_type   : 노드 → 종류 (dead_end / traffic_light / priority ...)
        #   _stop_nodes      : 정지선이 있는 노드 = 신호 교차로 + 횡단보도가 있는 노드
        self._no_change_edges, self._junction_type, self._stop_nodes = set(), {}, set()
        cfg_dir = os.path.dirname(os.path.abspath(self.cfg_path))
        net_el = ET.parse(self.cfg_path).getroot().find("input/net-file")
        if net_el is None:
            return {}, {}
        crossings, ends = {}, {}
        root = ET.parse(os.path.join(cfg_dir, net_el.get("value"))).getroot()
        for el in root.iter("edge"):
            if el.get("function") == "crossing":
                crossings[el.get("id")] = el.get("crossingEdges", "").split()
                self._stop_nodes.add(el.get("id")[1:].rsplit("_c", 1)[0])
            elif el.get("function") is None:
                ends[el.get("id")] = (el.get("from"), el.get("to"))
                for lane in el.findall("lane"):
                    ch = lane.get("changeLeft")
                    if ch is not None and "passenger" not in ch.split() and lane.get("allow") != "pedestrian":
                        self._no_change_edges.add(el.get("id"))
        for j in root.iter("junction"):
            self._junction_type[j.get("id")] = j.get("type")
            if j.get("type") == "traffic_light":
                self._stop_nodes.add(j.get("id"))
        return crossings, ends

    def is_stop_junction(self, internal_lane: str) -> bool:
        """교차로 내부 차로 id(":<노드>_<i>_<j>")의 노드에 정지선이 있는가 (신호 또는 횡단보도).

        엣지를 나누기만 한 지점(좌회전 포켓 시작, 보호구역 경계)은 False.
        """
        return internal_lane[1:].rsplit("_", 2)[0] in self._stop_nodes

    # ==================================================================
    # 어린이 보호구역 인식 — 노면 색으로
    #
    # road_builder가 그린 노면 다각형 중 ego 진행 방향 차로 위에 있는 것만 본다.
    #   적색(채움) 포장       → 보호구역 시작 (시점부 미끄럼방지포장)
    #   노란(채움) 긴 선      → 보호구역 범위 (노란 이중 가장자리선, 주정차 금지)
    # 제한속도는 적색 포장에 붙은 speed_limit 값((30) 노면표시와 같은 값)을 읽는다.
    # 위치는 ego 경로상 주행거리(getDistance 기준)로 바꿔 저장한다.
    #
    # 다각형에 type이 붙어 있으면 type으로 고른다 (road_builder, tools/zone_markings.py):
    #   school_zone = 적색 포장, school_zone_line = 노란선, 그 밖의 type(정차금지 빗금·
    #   정류장 그림 등)은 무시. 노란선에 zone_id가 있으면 간격과 상관없이 같은 구역으로 잇는다
    #   (청계천로는 구역 사이에 100 m 넘는 교차로가 있다). type이 하나도 없으면 예전처럼 색으로.
    # ==================================================================
    @staticmethod
    def _is_red(c):
        return c[0] >= 110 and c[1] <= 80 and c[2] <= 80

    @staticmethod
    def _is_yellow(c):
        return c[0] >= 200 and c[1] >= 150 and c[2] <= 90

    def _route_lane_geometry(self):
        """ego 경로의 일반 엣지 차로들: [(엣지 시작 주행거리, 차로 shape, 폭/2)]."""
        odo = traci.vehicle.getDistance(self.ego_id)
        out = []
        for edge in traci.vehicle.getRoute(self.ego_id):
            lo, hi = self._car_lanes(edge)
            if lo >= hi:
                continue
            try:
                start = odo + traci.vehicle.getDrivingDistance(self.ego_id, edge, 0.0, lo)
            except traci.TraCIException:
                continue
            if start < odo - 1e5:      # 경로상 도달 불가 (INVALID)
                continue
            for i in range(lo, hi):
                lane_id = f"{edge}_{i}"
                out.append((start, np.array(traci.lane.getShape(lane_id), dtype=float),
                            float(traci.lane.getWidth(lane_id)) / 2.0))
        return out

    @staticmethod
    def _project(geom, x, y, tol_lat=0.1, tol_along=3.0):
        """점 (x, y)가 경로 차로 위(가로 tol_lat, 세로 tol_along 여유)면 경로상 주행거리, 아니면 None."""
        for start, shape, half in geom:
            along = 0.0
            for (ax, ay), (bx, by) in zip(shape[:-1], shape[1:]):
                seg = float(np.hypot(bx - ax, by - ay))
                if seg < 1e-6:
                    continue
                ux, uy = (bx - ax) / seg, (by - ay) / seg
                t = (x - ax) * ux + (y - ay) * uy
                lat = abs(-(x - ax) * uy + (y - ay) * ux)
                if -tol_along <= t <= seg + tol_along and lat <= half + tol_lat:
                    return start + along + t
                along += seg
        return None

    def _setup_school_zones(self):
        """ego 경로상 어린이 보호구역 [(시작, 끝, 제한속도)] (주행거리 m)를 노면에서 읽는다."""
        self._zones = []
        try:
            poly_ids = traci.polygon.getIDList()
        except traci.TraCIException:
            return
        if not poly_ids:
            return
        geom = self._route_lane_geometry()
        reds, lines = [], []
        types = {pid: traci.polygon.getType(pid) for pid in poly_ids}
        typed = any(t in ("school_zone", "school_zone_line") for t in types.values())
        for pid in poly_ids:
            color = traci.polygon.getColor(pid)
            if not traci.polygon.getFilled(pid):
                continue
            if typed:
                red, yellow = types[pid] == "school_zone", types[pid] == "school_zone_line"
            else:
                red, yellow = self._is_red(color), self._is_yellow(color)
            if not (red or yellow):
                continue
            shape = traci.polygon.getShape(pid)
            xs, ys = [p[0] for p in shape], [p[1] for p in shape]
            if (yellow and not typed
                    and max(max(xs) - min(xs), max(ys) - min(ys)) < 15.0):
                continue                                   # 횡단보도 줄무늬 등 짧은 노란 표시 제외
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            if self._project(geom, cx, cy) is None:        # 내 진행 방향 차로 위가 아님
                continue
            s = [self._project(geom, x, y, tol_lat=0.6) for x, y in shape]
            s = [v for v in s if v is not None]
            if not s:
                continue
            if red:
                try:
                    limit = float(traci.polygon.getParameter(pid, "speed_limit") or 0.0)
                except (traci.TraCIException, ValueError):
                    limit = 0.0
                reds.append((min(s), limit))
            else:
                try:
                    zid = traci.polygon.getParameter(pid, "zone_id") or None
                except traci.TraCIException:
                    zid = None
                lines.append((min(s), max(s), zid))
        # 노란 이중선 조각들을 이어 붙여 구간을 만든다 (횡단보도·지그재그 구간에서 끊겨 있음)
        # zone_id가 있으면 같은 id끼리, 없으면 40 m 이내 간격끼리
        by_id = {}
        for a, b, zid in lines:
            if zid is not None:
                lo, hi = by_id.get(zid, (a, b))
                by_id[zid] = (min(lo, a), max(hi, b))
        merged = [[a, b] for a, b in by_id.values()]
        for a, b, zid in sorted(lines):
            if zid is not None:
                continue
            if merged and a - merged[-1][1] <= 40.0:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        merged.sort()
        for a, b in merged:
            inside = [(s, lim) for s, lim in reds if a - 5.0 <= s <= b]
            if not inside:
                continue
            start = min(s for s, _ in inside)
            limit = next((lim for s, lim in sorted(inside) if lim > 0), self.vmax)
            self._zones.append((min(start, a), b, limit))

    def _zone_state(self):
        """(보호구역까지 거리 m — 안이면 0, 제한속도 m/s) 또는 None (앞에 없음)."""
        if not self._zones or self.ego_id not in traci.vehicle.getIDList():
            return None
        odo = traci.vehicle.getDistance(self.ego_id)
        for s0, s1, limit in self._zones:
            if odo < s0:
                return s0 - odo, limit
            if odo <= s1:
                return 0.0, limit
        return None

    def _zone_obs(self):
        """관측 [40..41]: 보호구역까지 거리 / 적용 제한속도 (mdp_config 참고)."""
        zs = self._zone_state()
        if zs is None or zs[0] > self.zone_visibility:
            return [1.0, 1.0]
        return [float(np.clip(zs[0] / self.zone_visibility, 0.0, 1.0)),
                float(np.clip(zs[1] / self.vmax, 0.0, 1.0))]

    def _setup_route_crossings(self):
        """ego 경로가 지나는 횡단보도를 지나는 순서대로 준비한다 (reset에서 1회).

        각 횡단보도: 진입 지점의 경로상 위치(주행거리계 기준, m), 폭, 중심선,
        "내 경로 구간"(횡단보도 중 내 진행 방향 차로들 위의 부분)과 보행 신호 링크.
        """
        self._route_crossings = []
        if not self._crossings:
            return
        route = traci.vehicle.getRoute(self.ego_id)
        odo = traci.vehicle.getDistance(self.ego_id)
        ped_links = {}
        for tl_id in traci.trafficlight.getIDList():
            for idx, links in enumerate(traci.trafficlight.getControlledLinks(tl_id)):
                for link in links:
                    ped_links[link[1]] = (tl_id, idx)
        for e_in, e_out in zip(route[:-1], route[1:]):
            node = self._edge_ends.get(e_in, (None, None))[1]
            # 같은 교차로에서는 진입 쪽 횡단보도를 먼저 지난다
            for edge, side in ((e_in, "in"), (e_out, "out")):
                for cid, crossed in self._crossings.items():
                    if edge not in crossed or cid[1:].rsplit("_c", 1)[0] != node:
                        continue
                    lane_id = f"{cid}_0"
                    (ax, ay), (bx, by) = traci.lane.getShape(lane_id)[:2]
                    length = float(np.hypot(bx - ax, by - ay))
                    ux, uy = (bx - ax) / length, (by - ay) / length
                    width = float(traci.lane.getWidth(lane_id))
                    lo, hi = self._car_lanes(edge)
                    proj = []
                    for i in range(lo, hi):
                        shape = traci.lane.getShape(f"{edge}_{i}")
                        px, py = shape[-1] if side == "in" else shape[0]
                        half = float(traci.lane.getWidth(f"{edge}_{i}")) / 2.0
                        t = (px - ax) * ux + (py - ay) * uy
                        proj += [t - half, t + half]
                    if side == "in":
                        target = traci.lane.getLength(f"{edge}_{lo}")
                        ahead = traci.vehicle.getDrivingDistance(self.ego_id, edge, target, lo)
                    else:
                        ahead = traci.vehicle.getDrivingDistance(self.ego_id, edge, 0.0, lo) - width
                    self._route_crossings.append({
                        "id": cid, "s": odo + ahead, "width": width, "length": length,
                        "a": (ax, ay), "u": (ux, uy), "zone": (min(proj), max(proj)),
                        "link": ped_links.get(lane_id), "judged": False})

    # 관측용 보행자 범위: 횡단보도 양 끝에서 이만큼(m) 바깥 보도 모서리까지 서 있는 사람도 본다
    _PED_EDGE_MARGIN = 2.0

    def _crossing_ped_state(self, c, intent: bool = True):
        """횡단보도 c의 (가장 가까운 보행자 ↔ 내 경로 구간 거리 m, 다가오는 중인가).

        intent=True (판정용 — 양보 위반·정당한 대기):
            횡단보도 위의 보행자 + 보행 녹색일 때 진입 대기 중인 보행자
            (보도 모서리에서 다음 엣지가 이 횡단보도인 보행자 — SUMO 경로를 직접 읽음).
        intent=False (관측용 — 버스가 눈으로 볼 수 있는 것만):
            횡단보도 위 + 횡단보도 양 끝 _PED_EDGE_MARGIN m 안 보도 모서리에 서 있는 보행자.
            다가오는지는 보행자가 향한 방향(각도)으로만 판단하고, 다음 경로는 읽지 않는다.
        없으면 (None, False).
        """
        cache = self.__dict__.setdefault("_ped_cache", {})
        now = traci.simulation.getTime()
        if cache.get("t") != now:
            cache.clear()
            cache["t"] = now
            cache["roads"] = {pid: traci.person.getRoadID(pid)
                              for pid in traci.person.getIDList()}
        key = (c["id"], intent)
        if key in cache:
            return cache[key]
        green = True
        if c["link"] is not None:
            tl_id, idx = c["link"]
            green = traci.trafficlight.getRedYellowGreenState(tl_id)[idx] in "Gg"
        (ax, ay), (ux, uy), (zlo, zhi) = c["a"], c["u"], c["zone"]
        node = c["id"][1:].rsplit("_c", 1)[0]
        best = (None, False)
        for pid, road in cache["roads"].items():
            on = road == c["id"]
            x = y = t = None
            if not on:
                if not road.startswith(":"):
                    continue
                if intent:
                    if not (green and traci.person.getNextEdge(pid) == c["id"]):
                        continue
                else:
                    # 같은 교차로 보도 모서리(walkingarea)에서 횡단보도 끝 근처에 서 있는 사람만
                    if not road.startswith(f":{node}_w"):
                        continue
                    x, y = traci.person.getPosition(pid)
                    t = (x - ax) * ux + (y - ay) * uy
                    lat = abs(-(x - ax) * uy + (y - ay) * ux)
                    m = self._PED_EDGE_MARGIN
                    if not (-m <= t <= c["length"] + m and lat <= c["width"] / 2.0 + m):
                        continue
            if t is None:
                x, y = traci.person.getPosition(pid)
                t = (x - ax) * ux + (y - ay) * uy
            gap = max(0.0, zlo - t, t - zhi)
            toward = True if (intent and not on) else False   # 판정용: 진입 대기 = 다가옴
            if gap > 0.0 and (on or not intent):
                a = np.radians(traci.person.getAngle(pid))   # SUMO 각도: 북=0, 시계방향
                along = np.sin(a) * ux + np.cos(a) * uy
                toward = (t < zlo and along > 0.0) or (t > zhi and along < 0.0)
            elif gap <= 0.0:
                toward = True                  # 이미 내 경로 구간 위
            if best[0] is None or (gap, not toward) < (best[0], not best[1]):
                best = (gap, toward)
        cache[key] = best
        return best

    def _ped_obs(self):
        """관측 [34..39]: 경로상 다음 횡단보도들의 거리 / 보행자 거리 / 접근 (mdp_config 참고)."""
        odo = traci.vehicle.getDistance(self.ego_id)
        # 차체 뒤끝까지 횡단보도를 벗어나기 전까지는 "앞에 있는" 횡단보도로 본다
        ahead = [c for c in self._route_crossings
                 if odo <= c["s"] + c["width"] + self.ego_length]
        out = []
        for k in range(self.ped_num_crossings):
            if k >= len(ahead) or ahead[k]["s"] - odo > self.ped_visibility:
                out += [1.0, 1.0, 0.0]         # 없음/멀리: 제약 없음
                continue
            c = ahead[k]
            gap, toward = self._crossing_ped_state(c, intent=False)   # 관측: 보이는 것만
            out += [float(np.clip((c["s"] - odo) / self.ped_visibility, 0.0, 1.0)),
                    1.0 if gap is None else float(np.clip(gap / c["length"], 0.0, 1.0)),
                    float(toward)]
        return out

    def _waiting_for_ped(self) -> bool:
        """앞 ped_wait_zone 안의 횡단보도에 보행자가 "있음"이면 True (정당한 대기)."""
        margin = float(self.rw.get("yield_margin", 3.0))
        zone = float(self.rw.get("ped_wait_zone", 15.0))
        odo = traci.vehicle.getDistance(self.ego_id)
        for c in self._route_crossings:
            if c["judged"] or not (0.0 <= c["s"] - odo <= zone):
                continue
            gap, toward = self._crossing_ped_state(c)
            if gap is not None and gap <= margin and toward:
                return True
        return False

    def _check_yield(self, odo_before: float, odo_after: float) -> bool:
        """이번 스텝에 앞범퍼가 횡단보도에 들어섰는데 보행자가 있었으면 True.

        "있음" = 보행자가 내 경로 구간 위에 있거나, yield_margin 안에서 다가오는 중.
        횡단보도마다 진입하는 스텝에 한 번만 판정한다.
        """
        margin = float(self.rw.get("yield_margin", 3.0))
        violated = False
        for c in self._route_crossings:
            if c["judged"] or not (odo_before < c["s"] <= odo_after):
                continue
            c["judged"] = True
            gap, toward = self._crossing_ped_state(c)
            if gap is not None and gap <= margin and toward:
                violated = True
        return violated

    # ==================================================================
    # SUMO 프로세스 시작
    # ==================================================================
    def _start_sumo(self):
        # gui=True면 화면이 있는 sumo-gui, 아니면 화면 없는 sumo (학습용, 훨씬 빠름)
        binary = checkBinary("sumo-gui" if self.gui else "sumo")
        cmd = [
            binary, "-c", self.cfg_path,
            "--step-length", str(self.step_length),
            # 충돌한 차량을 시뮬레이션에서 제거. (기본값 teleport는 순간이동시켜버려서
            # RL 학습에는 부적합. remove로 해야 "충돌 → 에피소드 종료"가 깔끔함)
            "--collision.action", "remove",
            # 교차로 내부(다른 방향 차량·횡단보도 보행자)와의 충돌도 검사한다.
            # 기본값 false면 교차로 안에서는 차량이 겹쳐도 충돌로 안 잡힌다.
            "--collision.check-junctions", "true",
            # 보행자가 차와 유지하는 안전거리 (기본 0.25 m). 기본값이면 보행 녹색에
            # 우회전 차가 횡단보도 입구에 있어도 걸어 들어와 배경 충돌이 시간당 ~7건.
            # 1.5 m로 ~1.5건 (2 m는 서 있는 보행자와 충돌이 생김)
            "--pedestrian.striping.mingap-to-vehicle", "1.5",
            # 300초(기본값) 서 있는 차를 앞으로 순간이동시키지 않는다. ego가 정지선을
            # 건너뛰어 신호위반으로 잘못 기록되고, 평가 기준(Teleport 없음)에도 어긋난다
            "--time-to-teleport", "-1",
            "--no-warnings", "true",   # 콘솔 경고 끄기
            "--no-step-log", "true",   # 스텝 로그 끄기
            # 매 에피소드 다른 시드 → 배경 교통 패턴이 달라져서 과적합 방지
            "--seed", str(int(self._scn_rng.integers(0, 100000))),
        ]
        if self.gui:
            cmd += ["--quit-on-end"]      # 에피소드 종료 시 창 정리
            cmd += ["--delay", str(self.gui_delay)]
            if self.gui_autostart:
                cmd += ["--start"]        # 자동 재생
            # --start가 없으면: 창이 뜬 뒤 ▶(플레이)를 눌러야 진행된다.
            #   매 에피소드가 reset()에서 SUMO를 새로 띄우므로,
            #   에피소드마다 ▶를 눌러 시작하게 된다.
        traci.start(cmd)   # SUMO를 자식 프로세스로 띄우고 소켓 연결
        self._started = True

    # ==================================================================
    # 에피소드 초기화
    # ==================================================================
    # 다발지점 차량 종류 → (vType, 차체 길이 m). vType은 make_cgc_scenario가 rou.xml에 정의
    _HOTSPOT_TYPES = {"moto": ("parked_moto", 2.2), "truck": ("parked_truck", 6.0),
                      "car": ("parked_car", 5.0),
                      "big_truck": ("illegal_truck_big", 10.0)}
    # 다발지점 차량 사이 간격 (m). SUMO는 앞차와 minGap(기본 2.5 m)보다 가까우면 삽입을 거부해
    # 1.0으로 두면 맨 앞 1대만 들어가고 나머지는 삽입 대기열에 영원히 남았다
    _HOTSPOT_GAP = 2.6

    def _place_illegal_parking(self):
        """불법주정차 다발지점 (illegal_parking_hotspots.csv)에 차로를 막는 정차 차량을 세운다.

        (env-v1: 한빛공원 앞 대형 화물차(big_truck)는 1차로 구간을 완전히 막아 버스가 끝까지 서
        있었다 — 시드 1029. CSV에서 승용차(car) 1대로 바꿨다.)

        - 지점마다 prob 확률로만 생성 (에피소드마다 다양성). 시나리오 난수(_scn_rng)를 써서
          reset(seed=k)면 같은 판이 된다.
        - vehicles 예: "moto:3-5;truck:1" → 이륜차 3~5대(무작위) + 화물차 1대
        - 맨 오른쪽 승용차 차로, end_pos에서 뒤쪽으로 촘촘히, 에피소드 끝까지 정차(parking=false)
        """
        self._hotspot_taken = {}
        self._experimental_prefill_slots = {}
        self._hotspot_placed = []          # (지점 id, 엣지, 차량 수) — 확인·평가용
        rng = self._scn_rng
        for hid, edge, a, b, spec, prob in self._hotspots:
            if not edge or rng.random() >= float(prob or 0.0):
                continue
            try:
                lane = self._car_lanes(edge)[0]
                length = traci.lane.getLength(f"{edge}_{lane}")
            except traci.TraCIException:
                continue
            rid = f"hs_route_{edge}"
            if rid not in traci.route.getIDList():
                traci.route.add(rid, [edge])
            kinds = []
            for part in str(spec).split(";"):
                kind, _, cnt = part.strip().partition(":")
                lo, _, hi = cnt.partition("-")
                n = int(rng.integers(int(lo), int(hi or lo) + 1))
                kinds += [kind] * n
            pos = min(float(b), length - 0.5)   # 앞범퍼 위치. 뒤로 차곡차곡
            n_ok = 0
            for k, kind in enumerate(kinds):
                vtype, vlen = self._HOTSPOT_TYPES.get(kind, self._HOTSPOT_TYPES["moto"])
                if pos - vlen < float(a) - 0.1:
                    break                      # 지점 범위를 넘으면 그만
                if hid == "samil_east" and os.environ.get("DISABLE_SAMIL_EAST") == "1":
                    # Preserve the baseline prefill positions and all scenario RNG draws.
                    self._experimental_prefill_slots.setdefault((edge, lane), []).append(pos - vlen / 2)
                    pos -= vlen + self._HOTSPOT_GAP
                    continue
                vid = f"hs_{hid}_{k}"
                try:
                    traci.vehicle.add(vid, rid, typeID=vtype, departLane=str(lane),
                                      departPos=f"{pos - 0.1:.1f}", departSpeed="0")
                    traci.vehicle.setStop(vid, edge, pos=pos, laneIndex=lane,
                                          duration=1.0e6, flags=0)
                except traci.TraCIException:
                    continue
                self._hotspot_taken.setdefault((edge, lane), []).append(pos - vlen / 2)
                n_ok += 1
                pos -= vlen + self._HOTSPOT_GAP
            if n_ok:
                self._hotspot_placed.append((hid, edge, n_ok))

    def _prefill_traffic(self):
        """도로 전 구간에 배경차를 일정 간격(±jitter)으로 미리 배치한다.

        flow는 도로 입구에서만 차를 만들기 때문에, 프리필이 없으면
        에피소드 초반의 긴 도로가 텅 비어 "빈 도로 주행"만 잔뜩 학습된다.
        프리필은 첫 스텝부터 ego가 실제 교통류 한가운데서 시작하게 만든다.

        - 위치: start_offset ~ (길이 - end_margin), 차선별 독립 배치
        - 간격: prefill_spacing × (1 ± jitter) 무작위
        - 타입: controllers 확률 비율로 샘플링 (rou.xml의 vType과 동일)
        - 속도: departSpeed="max" → 각 개체가 "자기 허용 최고속도"로 시작.
          컨트롤러별 max_speed/speedDev 차등이 프리필에도 그대로 반영되고,
          개체별 무작위 speedFactor 때문에 숫자 속도를 주면 SUMO가
          "허용속도 초과"로 거부할 수 있는 문제도 함께 피한다.
        배치된 차들은 SUMO의 각자 컨트롤러(IDM 등)가 정상적으로 운전한다.
        """
        rng = self._scn_rng
        count = 0
        # 차선별 간격 배율 (없으면 전 차선 동일). 인덱스 0 = 맨 오른쪽.
        factors = (self.traffic.get("prefill", {})
                   .get("lane_spacing_factor")
                   or [1.0] * self.num_lanes)
        # ── 여러 엣지로 나뉜 route 대응 (예: 신호등으로 e0/e1 분할) ──
        # departPos는 "출발 엣지" 기준 위치라서, 첫 엣지 길이를 넘는 값을
        # 주면 SUMO가 그 차를 삽입 대기열에 계속 붙잡아 두고 같은 엣지의
        # 뒤 삽입(ego 포함)까지 막힌다. 그래서 도로 전체 좌표를 엣지별
        # (route, 엣지 내 위치)로 바꿔서 넣는다. k번째 엣지에서 출발하는
        # 차는 route r0의 k번째 엣지부터 시작하는 하위 route를 쓴다.
        # 각 엣지 끝 end_margin 구간도 비워둔다 (도로 끝 / 신호 정지선 앞).
        # 끝 여유(end_margin)는 신호 정지선 앞 엣지와 마지막 엣지에만 둔다. 보호구역·포켓
        # 때문에 나눈 짧은 엣지마다 비우면 그 구간에 차가 거의 깔리지 않는다.
        spans, start = [], 0.0
        edges = traci.route.getEdges("r0")
        for k, edge in enumerate(edges):
            length = traci.lane.getLength(f"{edge}_0")
            route_id = "r0" if k == 0 else f"r0_from{k}"
            if k > 0:
                traci.route.add(route_id, list(edges[k:]))
            to_node = self._edge_ends.get(edge, (None, None))[1]
            margin = (self.prefill_end_margin
                      if k == len(edges) - 1 or self._junction_type.get(to_node) == "traffic_light"
                      else 1.0)
            spans.append((route_id, start, length, margin))
            start += length
        # (엣지, 차선 번호) → 이미 배치한 엣지 내 위치들. 불법주정차 다발지점 자리도 피한다
        placed = {k: list(v) for k, v in self._hotspot_taken.items()}
        for key, positions in getattr(self, "_experimental_prefill_slots", {}).items():
            placed.setdefault(key, []).extend(positions)
        for lane in range(self.num_lanes):
            f = float(factors[lane]) if lane < len(factors) else 1.0
            lane_spacing = self.prefill_spacing * f
            pos = self.prefill_start + rng.uniform(0, lane_spacing)
            while pos < self.road_length - self.prefill_end_margin:
                route_id, local = None, 0.0
                for rid, edge_start, length, margin in spans:
                    if edge_start <= pos < edge_start + length:
                        local = pos - edge_start
                        if local <= length - margin:
                            route_id = rid
                        break
                vid = f"pre_{lane}_{count}"
                vtype = str(rng.choice(self._prefill_types, p=self._prefill_probs))
                try:
                    if route_id is None:
                        raise traci.TraCIException("엣지 끝 여유 구간")
                    first_edge = traci.route.getEdges(route_id)[0]
                    # 이 엣지에 해당 차선이 없으면(차로 1개 구간 등) 맨 오른쪽 차로로
                    # 대신 넣는다. 없는 차선 번호를 주면 SUMO가 "Invalid departlane"으로
                    # 거부해서 좁은 구간의 프리필이 빠졌다.
                    lo, hi = self._car_lanes(first_edge)
                    depart_lane = lo + lane if lo + lane < hi else lo
                    # 대체 차로에 이미 배치한 차와 겹치면 건너뛴다 (겹친 차는 삽입
                    # 대기열에 남아 같은 엣지의 다른 삽입까지 막을 수 있다)
                    taken = placed.setdefault((first_edge, depart_lane), [])
                    if any(abs(local - p) < _PREFILL_MIN_GAP for p in taken):
                        raise traci.TraCIException("대체 차로 위치 겹침")
                    traci.vehicle.add(
                        vid, route_id, typeID=vtype,
                        departLane=str(depart_lane),
                        departPos=f"{local:.1f}",
                        departSpeed="max")
                    taken.append(local)
                    count += 1
                except traci.TraCIException:
                    pass  # 겹침 / 엣지 끝 여유 구간이면 건너뜀
                step = lane_spacing * rng.uniform(
                    1.0 - self.prefill_jitter, 1.0 + self.prefill_jitter)
                pos += max(step, 10.0)
        return count

    def _randomize_signal_phase(self):
        """각 신호를 주기 안의 무작위 시점에서 시작시킨다.

        매 에피소드 같은 위상으로 시작하면 ego가 신호에 도착하는 시점의
        신호색이 거의 고정되어, 신호를 "보는" 대신 외워서 통과하게 된다.
        """
        for tl_id in traci.trafficlight.getIDList():
            program = traci.trafficlight.getProgram(tl_id)
            logic = next(l for l in traci.trafficlight.getAllProgramLogics(tl_id)
                         if l.programID == program)
            durations = [p.duration for p in logic.phases]
            t = self._scn_rng.uniform(0.0, sum(durations))
            for idx, d in enumerate(durations):
                if t < d:
                    break
                t -= d
            traci.trafficlight.setPhase(tl_id, idx)
            traci.trafficlight.setPhaseDuration(tl_id, durations[idx] - t)

    def start_scene(self, seed):
        """ego 없이 판 시작 장면까지만 만든다 — view_road.py --live(GUI 구경)용.

        reset(seed=k)와 같은 순서·같은 난수로 SUMO 실행(옵션·SUMO 시드) → setup_scene()을
        하므로, GUI에 보이는 시작 장면이 학습/평가 판 seed=k의 시작 장면과 같다.
        """
        self._scn_rng = np.random.default_rng(seed)
        if self._started:
            traci.close()
            self._started = False
        self._start_sumo()
        self.setup_scene()

    def setup_scene(self):
        """판 시작 장면을 만든다: 신호 시작 위상, 불법주정차 다발지점, 배경차 프리필, 정착.

        reset()과 start_scene()(GUI 구경)이 같이 쓴다. SUMO(TraCI)가 이미 떠 있어야 한다.
        """
        self._randomize_signal_phase()

        # 불법주정차 다발지점: 프리필보다 먼저 세워 배경차가 그 자리를 피하게 한다
        self._place_illegal_parking()
        # 도로 전 구간에 배경차를 미리 배치 (첫 스텝부터 실제 교통류 상태)
        if self.prefill_enabled:
            self._prefill_traffic()
        # 배치된 차들이 자리를 잡도록 몇 스텝만 정착시킨다.
        for _ in range(self.warmup_steps):
            traci.simulationStep()
        # 끝내 못 들어간 다발지점 차량은 지운다 (삽입 대기열에 남으면 같은 엣지 삽입을 막는다)
        for vid in traci.simulation.getPendingVehicles():
            if vid.startswith("hs_"):
                try:
                    traci.vehicle.remove(vid)
                except traci.TraCIException:
                    pass
        self._hotspot_placed = [
            (hid, edge, sum(1 for v in traci.vehicle.getIDList() if v.startswith(f"hs_{hid}_")))
            for hid, edge, _ in self._hotspot_placed]

    @staticmethod
    def is_held_out_seed(seed: int) -> bool:
        """테스트·검증용 판 시드인가 (학습 판으로 쓰면 안 됨)."""
        return any(lo <= int(seed) <= hi for lo, hi in HELD_OUT_SEEDS)

    @staticmethod
    def is_bc_reserved_seed(seed: int) -> bool:
        """팀원 BC가 수집·평가에 쓴 판인가. HELD_OUT과 별도 목록."""
        return any(lo <= int(seed) <= hi for lo, hi in BC_RESERVED_SEEDS)

    @classmethod
    def sample_train_seed(cls) -> int:
        """학습 판 시드. HELD_OUT_SEEDS와 BC_RESERVED_SEEDS를 모두 뺀다."""
        while True:
            seed = int(np.random.randint(0, 2**31 - 1))
            if not cls.is_held_out_seed(seed) and not cls.is_bc_reserved_seed(seed):
                return seed

    def reset(self, seed=None, options=None):
        self._ep_replayed = False
        # 시뮬레이션 시각 기준 캐시는 판이 바뀌면 시각이 0부터 다시 시작하므로 비운다
        self._sig_cache = None
        self.__dict__.pop("_ped_cache", None)
        if seed is None:
            # 학습 판. 실패한 판 다시 열기(replay) 목록도 학습 판 시드만 담는다
            if (not self.eval_mode and self._replay_seeds
                    and np.random.random() < self.replay_prob):
                seed = int(self._replay_seeds[np.random.randint(len(self._replay_seeds))])
                if self.is_held_out_seed(seed) or self.is_bc_reserved_seed(seed):
                    seed = self.sample_train_seed()
                else:
                    self._ep_replayed = True
            else:
                seed = self.sample_train_seed()
        self._ep_seed = seed
        super().reset(seed=seed)
        # 시나리오 난수 (SUMO 시드·신호 시작 위상·프리필·ego 경로): reset(seed=k)면 k로 고정해
        # 같은 시드 = 같은 판이 되게 한다 (평가끼리 같은 판으로 비교). 학습(seed=None)은 전역
        # np.random에서 뽑아 PPO 시드만 같으면 재현된다.
        self._scn_rng = np.random.default_rng(seed)

        # SUMO 상태를 완전히 초기화하는 가장 확실한 방법은 껐다 켜는 것.
        if self._started:
            traci.close()
            self._started = False
        self._start_sumo()
        # 신호 위상·불법주정차 다발지점·프리필·정착 (view_road.py --live도 같은 코드: start_scene)
        self.setup_scene()

        # ego 경로 선택 (EGO["routes"] 확률, 네트워크에 있는 route만)
        known = set(traci.route.getIDList())
        routes = {r: float(p) for r, p in self.ego_routes.items() if r in known} or {"r0": 1.0}
        probs = np.array(list(routes.values()))
        self.ego_route = str(self._scn_rng.choice(list(routes), p=probs / probs.sum()))

        # ego 차량 투입 — 반드시 "도로 시작점(0m)"에서 출발
        lane_base = self._car_lanes(traci.route.getEdges(self.ego_route)[0])[0]
        depart_lane = str(self.route_depart_lane.get(self.ego_route, self.ego_depart_lane))
        traci.vehicle.add(self.ego_id, self.ego_route, typeID="ego_type",
                          departLane=(str(int(depart_lane) + lane_base)
                                      if depart_lane.isdigit() else depart_lane),
                          departPos="0",
                          departSpeed=self.ego_depart_speed)
        # 이 route의 노선 정류장 (cgc.line_meta.json). SUMO에 정류장 정차를 등록하지 않는다:
        # 등록하면 SUMO가 버스를 세우거나 정류장 앞에 붙잡아 두므로 "모델이 직접 서기"가 아니다.
        # 정차 성공·놓침 판정과 승객 승하차는 이 클래스가 한다 (_update_stops).
        line_stops = self._line_stops.get(self.ego_route)
        if line_stops is None and self.ego_route in ("r0", "r0_loop"):
            line_stops = BUS_STOPS_ORDER   # 구버전 시나리오(메타 없음) 호환
        self._line_stop_ids = list(line_stops or [])
        # 노선 이름 (승객이 기다리는 노선과 맞춰 태운다 — _exchange_passengers)
        self._line_name = self._line_names.get(self.ego_route)

        # ★ 중요: speedMode=0 → SUMO의 내장 안전장치(자동 안전제동 등)를 전부 끔.
        #   이걸 꺼야 RL이 속도를 온전히 제어하고, 잘못 배우면 실제로 충돌한다.
        #   (켜두면 SUMO가 알아서 브레이크를 밟아줘서 충돌이 거의 안 나고,
        #    그러면 "충돌 회피"라는 학습 신호 자체가 사라진다). 판 끝까지 0으로 유지한다.
        traci.vehicle.setSpeedMode(self.ego_id, 0)
        # SUMO의 자율 차선변경은 끄고, RL이 내리는 TraCI 차선변경 명령만 사용한다.
        traci.vehicle.setLaneChangeMode(self.ego_id, 0)

        traci.simulationStep()  # ego가 실제로 도로에 나타나도록 1스텝 진행

        # ── ego 삽입 보증 ──
        # 도로 입구는 flow가 차를 스폰하는 자리라, 시작 차선 입구가 잠시
        # 막혀 있으면 SUMO가 삽입을 "지연"시킨다. 그 경우 ego 없이
        # 에피소드가 진행되어 즉시 종료되는 버그가 되므로, 삽입될 때까지
        # (최대 200스텝=100초, 청계천로는 입구가 1차로라 자전거·오토바이에 자주 막힘) 기다린다.
        _wait = 0
        while self.ego_id not in traci.vehicle.getIDList() and _wait < 200:
            traci.simulationStep()
            _wait += 1
        if self.ego_id not in traci.vehicle.getIDList():
            raise RuntimeError(
                "ego 삽입 실패: 도로 입구가 200스텝 이상 막혀 있습니다. "
                "TRAFFIC['vehs_per_hour']를 낮추거나 depart_speed를 확인하세요.")

        # GUI 기본 시점: ego를 화면 중앙에서 추적하고 충분히 가까이 확대한다.
        # 긴 직선도로 전체를 한 화면에 맞추는 기본 뷰는 차량이 점/삼각형처럼
        # 보이기 쉬우므로 학습/평가 관찰용으로는 이 설정이 더 직관적이다.
        if self.gui and self.ego_id in traci.vehicle.getIDList():
            try:
                view_ids = traci.gui.getIDList()
                if view_ids:
                    view_id = view_ids[0]
                    # tracking을 먼저 켜고 zoom을 마지막에 적용해야 SUMO가
                    # 차량 추적을 시작하면서 viewport를 다시 덮어쓰지 않는다.
                    if self.gui_track_ego:
                        traci.gui.trackVehicle(view_id, self.ego_id)
                    traci.gui.setZoom(view_id, self.gui_zoom)
            except traci.TraCIException:
                pass

        self._setup_stops()
        self.step_count = 0
        # 진행 보상용: 직전 스텝까지의 주행거리 (getDistance = 출발 후 누적 m)
        self._prev_dist = traci.vehicle.getDistance(self.ego_id)
        # 진행 보상 정규화: 이번 경로의 총 길이 (경로마다 길이가 다르다)
        last = traci.vehicle.getRoute(self.ego_id)[-1]
        last_lane = f"{last}_{self._car_lanes(last)[0]}"
        self._route_length = max(self._prev_dist + traci.vehicle.getDrivingDistance(
            self.ego_id, last, traci.lane.getLength(last_lane), self._car_lanes(last)[0]), 1.0)
        self.ego_length = float(traci.vehicle.getLength(self.ego_id))
        self._setup_route_crossings()
        self._setup_school_zones()
        self._setup_route_signals()
        self._stopped_at = None    # 일시정지(s) 신호 앞에서 멈췄던 신호 id
        return self._get_obs(), {}

    # ==================================================================
    # 관측 만들기 (레이아웃은 env/mdp_config.py 주석 참고)
    #
    # 핵심 설계: 특정 도로에 대한 하드코딩 없이, 매 스텝 SUMO 네트워크를
    # 실시간 조회해서 "지금 내 주변"을 계산한다. 그래서 도로 구조가
    # 어떻게 바뀌어도(차선 수, 합류, 차선 감소 등) 이 코드는 그대로 동작한다.
    #
    # 사용하는 TraCI API:
    #   getLeader / getFollower : 같은 차선의 앞/뒤 차량 (edge 경계도 넘어 탐색)
    #   getNeighbors(mode)      : 옆 차선의 앞/뒤 차량.
    #       mode 비트: bit0 = 0:왼쪽 / 1:오른쪽,  bit1 = 0:후행 / 1:선행
    #   lane.getLinks           : 차선이 다음 edge로 이어지는지 (연결성 판단)
    # ==================================================================

    # (차선 오프셋, 선행차 getNeighbors mode, 후행차 mode)
    #   오프셋 +1 = 왼쪽 차선 (SUMO는 차선 index가 왼쪽으로 갈수록 커짐)
    _NEIGHBOR_MODES = {+1: (0b10, 0b00),   # 왼쪽: 선행=2, 후행=0
                       -1: (0b11, 0b01)}   # 오른쪽: 선행=3, 후행=1

    def _car_lanes(self, edge: str):
        """edge에서 승용차가 다닐 수 있는 차선 번호 범위 (lo, hi) — hi는 미포함.

        교차로 layout은 보도가 차선 0번에 들어가서 차로가 1번부터 시작한다.
        차선 범위 검사와 출발 차선은 모두 이 범위를 기준으로 한다.
        """
        cache = self.__dict__.setdefault("_car_lane_cache", {})
        if edge not in cache:
            ok = []
            for i in range(traci.edge.getLaneNumber(edge)):
                lane_id = f"{edge}_{i}"
                allowed = traci.lane.getAllowed(lane_id)
                disallowed = traci.lane.getDisallowed(lane_id)
                if (not allowed or "passenger" in allowed) and "passenger" not in disallowed:
                    ok.append(i)
            cache[edge] = (min(ok), max(ok) + 1) if ok else (0, 0)
        return cache[edge]

    def _is_car_lane(self, edge: str, index: int) -> bool:
        lo, hi = self._car_lanes(edge)
        return lo <= index < hi

    def _car_lane_index(self, vid: str) -> int:
        """보도를 뺀 차로 번호 (0 = 맨 오른쪽 차로)."""
        edge = traci.vehicle.getRoadID(vid)
        index = int(traci.vehicle.getLaneIndex(vid))
        if edge.startswith(":"):
            return index
        return index - self._car_lanes(edge)[0]

    def _neighbor(self, offset: int, leader: bool):
        """오프셋 차선의 가장 가까운 선행/후행. OBS_SCAN=1일 때만 ±1을 보강한다.

        보강 본문은 강동연 obsfix.py (model/dongyeon-bc56, d54a761)와 같다.
        ±1은 getNeighbors와 차선 스캔 중 가까운 쪽, 겹친 차(gap < 0)는 gap 0.
        OBS_SCAN이 1이 아니면 _neighbor_base만 반환한다 (기본 0).
        """
        if os.environ.get("OBS_SCAN", "0") != "1":
            return self._neighbor_base(offset, leader)
        if abs(offset) != 1:
            return self._neighbor_base(offset, leader)
        n = self._neighbor_base(offset, leader)
        front, rear = self._scan_lane_neighbors(offset)
        s = front if leader else rear
        if s is not None:
            gap = max(float(s[0]), 0.0)
            if gap <= self.visibility and (n is None or gap < n[0]):
                n = (gap, float(s[1]))
        return n

    def _neighbor_base(self, offset: int, leader: bool):
        """오프셋 차선(-1=오른쪽, 0=현재, +1=왼쪽)의 가장 가까운
        선행(leader=True)/후행 차량을 (거리 m, 그 차량 속도)로 반환.
        visibility 밖이거나 없으면 None."""
        if offset == 0:
            if leader:
                res = traci.vehicle.getLeader(self.ego_id, self.visibility)
                if res is None:
                    return None
                vid, gap = res
            else:
                vid, gap = traci.vehicle.getFollower(self.ego_id, self.visibility)
                if vid == "":
                    return None
        elif abs(offset) >= 2:
            # 두 칸 이상 옆 차선: SUMO getNeighbors는 "바로 옆"만 지원하므로
            # 차선 직접 스캔(_scan_lane_neighbors — 같은 edge의 차량 목록을
            # 훑어 가장 가까운 앞/뒤를 찾음)으로 감지한다.
            front, rear = self._scan_lane_neighbors(offset)
            res = front if leader else rear
            if res is None or res[0] < 0 or res[0] > self.visibility:
                return None
            return float(res[0]), float(res[1])
        else:
            mode_leader, mode_follower = self._NEIGHBOR_MODES[offset]
            pairs = traci.vehicle.getNeighbors(
                self.ego_id, mode_leader if leader else mode_follower)
            if not pairs:
                return None
            vid, gap = min(pairs, key=lambda p: p[1])   # 가장 가까운 차량

        if gap < 0 or gap > self.visibility:
            return None
        return float(gap), traci.vehicle.getSpeed(vid)

    def _lane_leads_on_route(self, edge: str, lane_index: int) -> bool:
        """그 차선이 ego route의 다음 엣지로 이어지는가 (마지막 엣지면 True).

        교차로 앞 좌회전 포켓처럼 다른 방향으로만 이어지는 차선은 ego에게는
        "끝나는 차선"이다. 들어가면 route를 따라갈 수 없으므로 관측과
        차선변경 판단에서 이어지지 않는 차선으로 취급한다.
        """
        try:
            route = traci.vehicle.getRoute(self.ego_id)
            if edge not in route or route.index(edge) == len(route) - 1:
                return True
            nxt = route[route.index(edge) + 1]
            return any(link[0].rsplit("_", 1)[0] == nxt
                       for link in traci.lane.getLinks(f"{edge}_{lane_index}"))
        except traci.TraCIException:
            return True

    def _lane_route_link(self, lane_id: str):
        """그 차선에서 ego route의 다음 엣지로 가는 링크 (getLinks 항목), 없으면 None.

        항목: (다음 차선, 우선, 열림, 접근 차량, 교차로 내부 차선, 상태, 방향, 길이)
        """
        try:
            route = traci.vehicle.getRoute(self.ego_id)
            edge = lane_id.rsplit("_", 1)[0]
            if edge not in route or route.index(edge) == len(route) - 1:
                return None
            nxt = route[route.index(edge) + 1]
            return next((link for link in traci.lane.getLinks(lane_id)
                         if link[0].rsplit("_", 1)[0] == nxt), None)
        except traci.TraCIException:
            return None

    def _lane_connectivity(self, offset: int) -> float:
        """오프셋 차선의 '끊김 정도'를 반환 (도로 모양만 본다).

        -1 : 그 차선이 존재하지 않음
         1 : visibility 이내에서 계속 이어짐 (어느 방향으로든 다음 차로가 있음)
        0~1: (차선이 끝나는 지점까지의 거리 / visibility)
             → 값이 작을수록 곧 차선이 사라진다는 경고

        차선 끝에서 다음 edge로 가는 연결(link)이 하나도 없으면 "끝나는 차선"이다.
        env-v1부터 "내 경로로 이어지는가"는 보지 않는다 (차선 지시라서 과제 규칙 위반).
        어느 차로가 내 방향으로 가는지는 노면 화살표 관측(lane_arrows)과 회전 방향(turn)으로 안다.
        """
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):          # 교차로(junction) 내부 통과 중
                return 1.0
            lane_index = traci.vehicle.getLaneIndex(self.ego_id) + offset
            if not self._is_car_lane(edge, lane_index):
                return -1.0                   # 그런 차선이 없음 (보도 포함)
            lane_id = f"{edge}_{lane_index}"
            remaining = (traci.lane.getLength(lane_id)
                         - traci.vehicle.getLanePosition(self.ego_id))
            if remaining > self.visibility:
                return 1.0                    # 시야 내에선 계속 이어짐
            if traci.lane.getLinks(lane_id):
                return 1.0                    # edge가 끝나도 다음 차로로 이어짐
            return float(np.clip(remaining / self.visibility, 0.0, 1.0))
        except traci.TraCIException:
            return 1.0                        # 조회 실패 시 중립값

    def _lane_density(self, offset: int) -> float:
        """오프셋 차선의 '전방 차량 밀도'를 0~1로 반환.

        밀도 = (시야 W 이내 전방 차량 수) / (그 구간의 최대 수용 대수)
               최대 수용 대수 = 사용가능거리 / density_veh_length

        인코딩 관례 (막힘 정도로 읽히도록):
            1.0 : 차선이 없거나 사실상 진입 불가  ← "꽉 찬 것"처럼 취급해서
                  에이전트가 자연스럽게 회피하게 만든다
            0.0 : 텅 빈 차선
        일반화: 차선이 시야보다 짧으면 link를 따라 다음 차선까지 이어서 센다.
        곧 끊기는 차선(link 없음)은 남은 거리만큼만 세므로 분모가 줄어
        같은 대수여도 밀도가 높게 나온다 → "끊기는 차선에 차가 몰려있음" 신호.
        """
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):          # 교차로 내부: 중립값
                return 0.0
            lane_index = traci.vehicle.getLaneIndex(self.ego_id) + offset
            if not self._is_car_lane(edge, lane_index):
                return 1.0                    # 차선 없음(보도 포함) = 꽉 참 취급
            lane_id = f"{edge}_{lane_index}"
            ego_pos = traci.vehicle.getLanePosition(self.ego_id)
            remaining = traci.lane.getLength(lane_id) - ego_pos
            links = traci.lane.getLinks(lane_id)

            # 이 차선에서 앞으로 실제로 쓸 수 있는 거리
            usable = self.visibility if links else min(self.visibility, remaining)
            if usable <= 0:
                return 1.0

            count = 0
            # (a) 현재 차선 구간의 전방 차량
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                if vid == self.ego_id:
                    continue
                d = traci.vehicle.getLanePosition(vid) - ego_pos
                if 0.0 < d <= min(self.visibility, remaining):
                    count += 1
            # (b) 시야가 edge 끝을 넘고 다음 차선으로 이어지면 이어서 센다
            if links and remaining < self.visibility:
                next_lane_id = links[0][0]    # 첫 번째 연결 차선
                for vid in traci.lane.getLastStepVehicleIDs(next_lane_id):
                    if traci.vehicle.getLanePosition(vid) <= \
                            self.visibility - remaining:
                        count += 1

            max_count = usable / self.density_veh_len
            return float(np.clip(count / max_count, 0.0, 1.0))
        except traci.TraCIException:
            return 0.0

    def get_detected_vehicles(self):
        """현재 관측 시야에 들어온(=감지된) 배경차들의 좌표 목록을 반환.

        데이터셋 수집용 부가 정보 — 관측 벡터에는 들어가지 않는다.
        "감지"의 정의는 관측과 동일: 관측 대상 차선(obs_offsets 범위)에서
        ego 기준 종방향 ±visibility(W) 이내에 있는 차량.

        Returns: list[dict]  각 원소 =
            {"id": 차량 id, "x": 절대 x, "y": 절대 y,
             "rel_pos": ego 기준 종방향 상대거리(m, 앞 +), "speed": m/s,
             "lane_offset": ego 차선 기준 오프셋(+왼쪽)}
        ego가 없으면 빈 리스트.
        """
        out = []
        if self.ego_id not in traci.vehicle.getIDList():
            return out
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                return out
            ego_lane = traci.vehicle.getLaneIndex(self.ego_id)
            ego_pos = traci.vehicle.getLanePosition(self.ego_id)
            for offset in self.obs_offsets:
                li = ego_lane + offset
                if not self._is_car_lane(edge, li):
                    continue
                for vid in traci.lane.getLastStepVehicleIDs(f"{edge}_{li}"):
                    if vid == self.ego_id:
                        continue
                    rel = traci.vehicle.getLanePosition(vid) - ego_pos
                    if abs(rel) > self.visibility:
                        continue
                    x, y = traci.vehicle.getPosition(vid)
                    out.append({"id": vid, "x": float(x), "y": float(y),
                                "rel_pos": float(rel),
                                "speed": float(traci.vehicle.getSpeed(vid)),
                                "lane_offset": int(offset)})
        except traci.TraCIException:
            pass
        return out

    # 신호색 → 정지 필요도 (관측 [32])
    _SIGNAL_STOP_LEVEL = {"r": 1.0, "u": 1.0,             # 적색 계열
                          "s": 0.75,                       # 일시정지 후 진행 (우회전)
                          "y": 0.5,                        # 황색
                          "G": 0.0, "g": 0.0, "o": 0.0, "O": 0.0}

    # 일시정지(s) 신호: 정지선 이 거리(m) 안에서 멈췄다가 지나가야 위반이 아니다
    _STOP_ZONE = 5.0

    def _tls_remaining(self, tl_id, link, state):
        """신호 tl_id의 link가 지금 색(같은 정지 필요도)을 유지하는 남은 시간 s."""
        remaining = traci.trafficlight.getNextSwitch(tl_id) - traci.simulation.getTime()
        # 교차로 신호는 ego 링크 기준으로 같은 색이 여러 단계에 걸쳐 이어진다
        # (예: 기본 프로그램의 적색 37+5+3초). "ego 신호색이 바뀔 때까지"의
        # 시간이 되도록, 같은 정지 필요도가 이어지는 다음 단계들을 더한다.
        level = self._SIGNAL_STOP_LEVEL.get(state, 1.0)
        program = traci.trafficlight.getProgram(tl_id)
        logic = next(l for l in traci.trafficlight.getAllProgramLogics(tl_id)
                     if l.programID == program)
        phases = logic.phases
        k = traci.trafficlight.getPhase(tl_id)
        for _ in range(len(phases) - 1):
            k = (k + 1) % len(phases)
            if self._SIGNAL_STOP_LEVEL.get(phases[k].state[link], 1.0) != level:
                break
            remaining += phases[k].duration
        return float(max(remaining, 0.0))

    def _next_signal(self):
        """ego 경로상 다음 신호를 (신호 id, 정지선까지 거리 m, 신호색, 남은 시간 s)로 반환.

        traci getNextTLS가 돌려주는 가장 가까운 신호 그대로다 (거리는 ego 앞범퍼 → 정지선).
        env-v1부터 가공하지 않는다: 황색을 "지나갈 수 있으면 녹색"으로 바꾸던 것과 연달아 있는
        정지선의 더 엄격한 색을 앞당겨 쓰던 것은 규칙이라 지웠다 (rules/team/experiment_protocol.md 4-1).
        남은 시간은 판정·보상·데이터용이고 관측에는 넣지 않는다.
        신호를 이미 지났거나 ego가 없으면 None.
        """
        # 같은 시뮬레이션 시각엔 결과가 같으므로 한 번만 계산한다 (한 스텝에 여러 번 불림)
        now = traci.simulation.getTime()
        cached = getattr(self, "_sig_cache", None)
        if cached is not None and cached[0] == now:
            return cached[1]
        out = None
        if self.ego_id in traci.vehicle.getIDList():
            try:
                upcoming = traci.vehicle.getNextTLS(self.ego_id)
                if upcoming:
                    tl_id, link, dist, state = upcoming[0]
                    out = (tl_id, float(dist), state, self._tls_remaining(tl_id, link, state))
            except traci.TraCIException:
                out = None
        self._sig_cache = (now, out)
        return out

    def _first_in_line(self, dist: float) -> bool:
        """ego와 정지선(dist m 앞) 사이 같은 진행 차로에 다른 차가 없는가 = 신호를 직접 보는 선두 차량."""
        try:
            lead = traci.vehicle.getLeader(self.ego_id, max(dist, 0.0))
        except traci.TraCIException:
            return True
        return not (lead and lead[0] and lead[1] < dist)

    def _signal_obs(self):
        """관측 signal: [정지선 거리/visibility, 적, 황, 녹, 일시정지, 보이는지].

        버스가 맨 앞 차일 때만(정지선까지 같은 차로에 앞차가 없을 때) 값을 넣는다. 뒤에 서 있으면
        신호를 가린다: [1, 0, 0, 0, 0, 0] (실제 운전자도 앞차에 가려 신호보다 앞차를 보고 판단).
        신호 없음 / 시야 밖도 같은 값이다.
        """
        hidden = [1.0] + [0.0] * len(SIGNAL_COLORS) + [0.0]
        nxt = self._next_signal()
        if nxt is None or nxt[1] > self.signal_visibility:
            return hidden
        _, dist, state, _ = nxt
        if not self._first_in_line(dist):
            return hidden
        color = _COLOR_OF_STATE.get(state, "red")
        return ([float(np.clip(dist / self.signal_visibility, 0.0, 1.0))]
                + [1.0 if c == color else 0.0 for c in SIGNAL_COLORS] + [1.0])

    # ---------------- 경로 안내 (다음 교차로 회전 방향 + 노면 화살표) ----------------
    def _lane_dirs(self, lane_id: str):
        """그 차로에서 갈 수 있는 방향 [좌, 직, 우] (0/1) — 노면 화살표에 해당."""
        out = [0.0, 0.0, 0.0]
        try:
            for link in traci.lane.getLinks(lane_id):
                k = _DIR_INDEX.get(link[6])
                if k is not None:
                    out[k] = 1.0
        except traci.TraCIException:
            pass
        return out

    def _next_turn(self):
        """경로상 다음 교차로에서 할 회전 [좌, 직, 우] one-hot (내비게이션 안내).

        지금 엣지에서 경로의 다음 엣지로 가는 링크 방향으로 정한다. 교차로 안이거나
        마지막 엣지면 [0, 0, 0]."""
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                return [0.0, 0.0, 0.0]
            route = traci.vehicle.getRoute(self.ego_id)
            ri = traci.vehicle.getRouteIndex(self.ego_id)
            if ri < 0 or ri >= len(route) - 1:
                return [0.0, 0.0, 0.0]
            nxt = route[ri + 1]
            for i in range(traci.edge.getLaneNumber(edge)):
                for link in traci.lane.getLinks(f"{edge}_{i}"):
                    if link[0].rsplit("_", 1)[0] == nxt:
                        k = _DIR_INDEX.get(link[6])
                        if k is not None:
                            return [1.0 if j == k else 0.0 for j in range(3)]
        except traci.TraCIException:
            pass
        return [0.0, 0.0, 0.0]

    def _route_dir_obs(self):
        """관측 turn(3) + lane_arrows(5×3): 다음 회전 방향, 시야 5개 차로의 화살표 (없는 차로 0)."""
        out = self._next_turn()
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            cur = traci.vehicle.getLaneIndex(self.ego_id)
        except traci.TraCIException:
            return out + [0.0] * (3 * len(self.obs_offsets))
        for offset in self.obs_offsets:
            li = cur + offset
            if edge.startswith(":") or not self._is_car_lane(edge, li):
                out += [0.0, 0.0, 0.0]
            else:
                out += self._lane_dirs(f"{edge}_{li}")
        return out

    # ---------------- 차선변경 금지 구간 (막지 않고 감점) ----------------
    def _lc_forbidden(self) -> bool:
        """지금 위치가 차선변경 금지 구간인가 (도로교통 규칙. env-v1부터 막지 않고 감점만 한다).

        - 신호 정지선 앞 signal_no_change_zone m (교차로 앞 실선)
        - 교차로 직후 spawn_protect m (신호 교차로·도로 시작에서 나온 엣지)
        - 지그재그 구간 (어린이 보호구역 횡단보도 앞, net.xml changeLeft로 승용차 변경 금지인 엣지)
        """
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                return False
            if edge in self._no_change_edges:
                return True
            pos = traci.vehicle.getLanePosition(self.ego_id)
            from_node = self._edge_ends.get(edge, (None, None))[0]
            if (pos < self.lc_spawn_protect
                    and self._junction_type.get(from_node, "dead_end") in ("dead_end", "traffic_light")):
                return True
            if self.lc_signal_zone > 0.0:
                nxt = self._next_signal()
                if nxt is not None and nxt[1] < self.lc_signal_zone:
                    return True
        except traci.TraCIException:
            pass
        return False

    # ---------------- 버스정류장 (모델이 직접 서기) ----------------
    def _setup_stops(self):
        """이번 판 정류장 목록을 만든다 (reset에서 ego 투입 뒤 1회).

        각 정류장: id, 엣지, 차로 번호, 정류장 구간(시작·끝 m), 경로상 엣지 순번.
        경로에 없는 정류장은 뺀다."""
        self._stops, self._stop_i, self._dwell_steps = [], 0, 0
        self._missed_stops, self._stops_done, self._stop_errors = 0, 0, []
        self._stop_status = []     # 정류장별 0 남음 / 1 정차 성공 / -1 놓침 (데이터 컬럼용)
        self._onboard = []
        route = list(traci.vehicle.getRoute(self.ego_id))
        for bs in self._line_stop_ids:
            try:
                lane = traci.busstop.getLaneID(bs)
                edge, idx = lane.rsplit("_", 1)
                self._stops.append({
                    "id": bs, "edge": edge, "lane": int(idx),
                    "start": float(traci.busstop.getStartPos(bs)),
                    "end": float(traci.busstop.getEndPos(bs)),
                    "ri": route.index(edge)})
            except (traci.TraCIException, ValueError):
                print("정류장 정보 없음 또는 경로 밖:", bs)
        self._stop_status = [0] * len(self._stops)

    def _current_stop(self):
        return self._stops[self._stop_i] if self._stop_i < len(self._stops) else None

    def _next_stop(self):
        """다음 정류장 (정류장 끝까지 주행거리 m, 정류장 차로 index, 엣지 id). 없으면 None."""
        st = self._current_stop()
        if st is None:
            return None
        try:
            d = traci.vehicle.getDrivingDistance(self.ego_id, st["edge"], st["end"], st["lane"])
            if d < -1e5:          # 현재 차로에서 갈 수 없음 → 엣지 기준 거리로 대신
                d = traci.vehicle.getDrivingDistance(self.ego_id, st["edge"], st["end"])
        except traci.TraCIException:
            return None
        if d < -1e5:
            return None
        return float(d), st["lane"], st["edge"]

    def _in_stop_area(self, st) -> bool:
        """ego 앞범퍼가 정류장 st의 정류장 차로·정류장 구간 안에 있는가."""
        try:
            if traci.vehicle.getRoadID(self.ego_id) != st["edge"]:
                return False
            if traci.vehicle.getLaneIndex(self.ego_id) != st["lane"]:
                return False
            pos = traci.vehicle.getLanePosition(self.ego_id)
        except traci.TraCIException:
            return False
        return st["start"] <= pos <= st["end"] + self.stop_overshoot

    def _stop_obs(self):
        """관측 bus_stop 3개:
        [0] 다음 정류장(끝)까지 거리 / visibility (없거나 멀면 1)
        [1] 정류장 차로까지 차로 차이 / 2 (+면 왼쪽으로, -면 오른쪽으로 가야 함. 맨 오른쪽 차로부터 센 번호로 비교)
        [2] 정류장 구간 안에서 연속으로 서 있은 시간 / dwell_time (정차 성공까지 진행도, 0~1)
        """
        ns = self._next_stop()
        dwell = float(np.clip(self._dwell_steps * self.step_length / self.stop_dwell_time, 0.0, 1.0))
        if ns is None or ns[0] < 0 or ns[0] > self.stop_visibility:
            return [1.0, 0.0, dwell]
        d, lane, edge = ns
        try:
            cur_edge = traci.vehicle.getRoadID(self.ego_id)
            if cur_edge.startswith(":"):
                diff = 0.0
            else:
                want = lane - self._car_lanes(edge)[0]
                have = traci.vehicle.getLaneIndex(self.ego_id) - self._car_lanes(cur_edge)[0]
                diff = float(np.clip((want - have) / 2.0, -1.0, 1.0))
        except traci.TraCIException:
            diff = 0.0
        return [float(np.clip(d / self.stop_visibility, 0.0, 1.0)), diff, dwell]

    def _update_stops(self, alive: bool, arrived: bool):
        """스텝 끝에 정류장 진행을 판정한다. 반환: (정차 성공 수, 놓친 수) — 이번 스텝.

        - 성공: 정류장 차로·구간 안에서 stop_speed 이하로 dwell_time 초 연속 정지
          → 승객 승하차(_exchange_passengers), 다음 정류장으로
        - 놓침: 정류장 엣지를 지나쳤거나, 정류장 끝 + overshoot를 넘어감 (되돌아갈 수 없음)
          도착(arrived)하면 남은 정류장은 모두 놓친 것으로 센다
        """
        ok, missed = 0, 0
        if arrived:
            missed = len(self._stops) - self._stop_i
            for k in range(self._stop_i, len(self._stops)):
                self._stop_status[k] = -1
            self._stop_i = len(self._stops)
        if not alive:
            self._missed_stops += missed
            return ok, missed
        st = self._current_stop()
        while st is not None:
            try:
                ri = traci.vehicle.getRouteIndex(self.ego_id)
                edge = traci.vehicle.getRoadID(self.ego_id)
                pos = traci.vehicle.getLanePosition(self.ego_id)
                v = traci.vehicle.getSpeed(self.ego_id)
            except traci.TraCIException:
                break
            passed = ri > st["ri"] or (edge == st["edge"] and pos > st["end"] + self.stop_overshoot)
            if passed:
                missed += 1
                self._stop_status[self._stop_i] = -1
                self._stop_i += 1
                self._dwell_steps = 0
                st = self._current_stop()
                continue
            if self._in_stop_area(st) and v <= self.stop_speed:
                self._dwell_steps += 1
                if self._dwell_steps * self.step_length >= self.stop_dwell_time - 1e-9:
                    ok += 1
                    self._stop_errors.append(float(st["end"] - pos))
                    self._exchange_passengers(st["id"], edge, pos)
                    self._stop_status[self._stop_i] = 1
                    self._stop_i += 1
                    self._dwell_steps = 0
            else:
                self._dwell_steps = 0
            break
        self._missed_stops += missed
        self._stops_done += ok
        return ok, missed

    def _exchange_passengers(self, bs_id: str, edge: str, pos: float):
        """정차 성공 때 승객을 내리고 태운다 (SUMO 정류장 정차 대신 직접).

        - 하차: 이 정류장에서 내릴 승객을 정류장 위치에 새 보행자로 만들고, 타기 전 계획에 있던
          다음 단계(길 건너기 등)를 그대로 이어 붙인다.
        - 승차: 이 정류장에서 이 노선을 기다리는 승객을 정원(personCapacity)까지 태운다.
          탄 승객은 시뮬레이션에서 빼고 (내릴 정류장, 이어갈 단계들)만 기억한다.
        승객 처리 실패는 주행에 영향이 없도록 조용히 넘긴다.
        """
        # 하차
        stay = []
        for dest, stages in self._onboard:
            if dest != bs_id:
                stay.append((dest, stages))
                continue
            if not stages:
                continue
            self._pax_seq += 1
            pid = f"pax_off_{self._pax_seq}"
            try:
                traci.person.add(pid, edge, pos, depart=-3, typeID="pax")
                for stage in stages:
                    if stage.type == traci.constants.STAGE_TRIP and stage.edges:
                        # 목적지까지 "이동"(personTrip) 단계는 그대로 붙이면 걷지 않아서,
                        # 내린 자리에서 목적지까지 걷는 경로를 다시 찾아 붙인다
                        for walk in traci.simulation.findIntermodalRoute(
                                edge, stage.edges[-1], arrivalPos=stage.arrivalPos,
                                pType="pax", modes=""):
                            traci.person.appendStage(pid, walk)
                    else:
                        traci.person.appendStage(pid, stage)
            except traci.TraCIException:
                try:
                    traci.person.remove(pid)
                except traci.TraCIException:
                    pass
        self._onboard = stay
        # 승차
        try:
            cap = int(traci.vehicletype.getPersonCapacity(traci.vehicle.getTypeID(self.ego_id)))
        except traci.TraCIException:
            cap = 0
        try:
            waiting = list(traci.busstop.getPersonIDs(bs_id))
        except traci.TraCIException:
            waiting = []
        for pid in waiting:
            if len(self._onboard) >= cap:
                break
            try:
                ride = traci.person.getStage(pid, 0)
                lines = str(getattr(ride, "line", "") or "").split()
                if ride.type != traci.constants.STAGE_DRIVING or (
                        self._line_name and lines and self._line_name not in lines):
                    continue
                n = traci.person.getRemainingStages(pid)
                rest = [traci.person.getStage(pid, k) for k in range(1, n)]
                self._onboard.append((ride.destStop, rest))
                traci.person.remove(pid)
            except traci.TraCIException:
                continue

    def _shield_accel(self, accel_raw: float, v: float):
        """앞차 기준 안전 속도(정지거리 모델)를 넘는 명령이면 제동으로 바꾼다.

        v_safe = sqrt(v_lead² + 2·b·(gap − margin)) − b·dt  (b = shield_decel)
        다음 스텝 속도가 v_safe를 넘으면 v_safe로 맞추는 가속을 낸다.
        """
        lead = traci.vehicle.getLeader(self.ego_id, 120.0)
        if not lead or not lead[0]:
            return accel_raw, False
        gap = max(float(lead[1]) - self.shield_margin, 0.0)
        vl = traci.vehicle.getSpeed(lead[0])
        b, dt = self.shield_decel, self.step_length
        v_safe = max(np.sqrt(vl * vl + 2.0 * b * gap) - b * dt, 0.0)
        cmd = accel_raw * (self.max_accel if accel_raw >= 0 else self.max_decel)
        if v + cmd * dt <= v_safe:
            return accel_raw, False
        need = (v_safe - v) / dt
        new = need / self.max_accel if need >= 0 else need / self.max_decel
        return float(np.clip(min(new, accel_raw), -1.0, 1.0)), True

    def _signal_shield_accel(self, accel_raw: float, v: float, sig):
        """앞 신호가 적색(r/u) 또는 편하게 설 수 있는 황색(y)이면 정지선을 장애물로 보고
        _shield_accel과 같은 정지거리 모델로 속도를 제한한다. sig = _next_signal() 결과."""
        if sig is None:
            return accel_raw, False
        _, dist, state, _ = sig
        if state not in ("r", "u", "y"):
            return accel_raw, False
        room = dist - self.signal_margin
        b, dt = self.shield_decel, self.step_length
        if state == "y" and v * v > 2.0 * b * max(room, 0.0):
            return accel_raw, False                 # 황색인데 편하게 못 섬 → 통과
        if v * v > 2.0 * self.signal_max_decel * max(room, 0.0) + 1e-6:
            return accel_raw, False                 # 허용 제동으로 못 섬 → 개입하지 않음
        v_safe = max(np.sqrt(2.0 * b * max(room, 0.0)) - b * dt, 0.0)
        cmd = accel_raw * (self.max_accel if accel_raw >= 0 else self.max_decel)
        if v + cmd * dt <= v_safe:
            return accel_raw, False
        need = (v_safe - v) / dt
        new = need / self.max_accel if need >= 0 else need / self.max_decel
        return float(np.clip(min(new, accel_raw), -1.0, 1.0)), True

    def _get_obs(self):
        # ego가 이미 사라졌다면(충돌/도착) 0벡터 반환 — 종료 직전 마지막 관측용
        if self.ego_id not in traci.vehicle.getIDList():
            return np.zeros(self.obs_dim, dtype=np.float32)

        v = traci.vehicle.getSpeed(self.ego_id)
        obs = [v / self.vmax]                      # [0] 내 절대속도

        # ----- 5개 차선(좌2→좌1→현재→우1→우2)의 선행/후행 상대거리·상대속도 -----
        for offset in self.obs_offsets:
            lead = self._neighbor(offset, leader=True)
            if lead is None:
                obs += [1.0, 1.0]                  # 없음: 멀고 + 위협 없음
            else:
                gap, lv = lead
                obs += [gap / self.visibility,
                        float(np.clip((lv - v) / self.vmax, -1.0, 1.0))]

            foll = self._neighbor(offset, leader=False)
            if foll is None:
                obs += [-1.0, -1.0]                # 없음: 멀고 + 위협 없음
            else:
                gap, fv = foll
                obs += [-gap / self.visibility,    # 후행은 음수 부호
                        float(np.clip((fv - v) / self.vmax, -1.0, 1.0))]

        # ----- 5개 차선의 연결성 (같은 순서) -----
        for offset in self.obs_offsets:
            obs.append(self._lane_connectivity(offset))

        # ----- 5개 차선의 전방 차량 밀도 (같은 순서) -----
        for offset in self.obs_offsets:
            obs.append(self._lane_density(offset))

        # ----- 전방 신호 (OBSERVATION["signal"] 사용 시) — 선두 차량일 때만 보임 -----
        if self.signal_obs:
            obs += self._signal_obs()

        # ----- 경로 안내: 다음 회전 방향 + 차로별 화살표 (OBSERVATION["route_dir"]) -----
        if self.route_dir_obs:
            obs += self._route_dir_obs()

        # ----- 경로상 횡단보도 (OBSERVATION["pedestrian"] 사용 시) -----
        if self.ped_obs:
            obs += self._ped_obs()

        # ----- 어린이 보호구역 (OBSERVATION["school_zone"] 사용 시) -----
        if self.zone_obs:
            obs += self._zone_obs()

        # ----- 버스정류장 (OBSERVATION["bus_stop"] 사용 시) -----
        if self.stop_obs:
            obs += self._stop_obs()

        # ----- 차선변경 금지 구간 (OBSERVATION["lc_zone"]) -----
        if self.lc_zone_obs:
            obs.append(1.0 if self._lc_forbidden() else 0.0)

        assert len(obs) == self.obs_dim, f"관측 길이 {len(obs)} != obs_dim {self.obs_dim}"
        return np.clip(np.array(obs, dtype=np.float32), -1.0, 1.0)

    # ==================================================================
    # 1 스텝 진행: 행동 적용 → 시뮬레이션 1스텝 → 보상/종료 판정
    # ==================================================================
    def _quantize_lane_change(self, raw: float) -> int:
        """연속 policy 출력을 실제 차선 명령 {-1,0,+1}로 변환한다.

        +1 = 왼쪽, 0 = 유지, -1 = 오른쪽. SUMO 차선 index가 왼쪽으로
        갈수록 증가하는 규약과 동일해서 해석이 직관적이다.
        """
        raw = float(np.clip(raw, -1.0, 1.0))
        if raw >= self.lane_change_threshold:
            return 1
        if raw <= -self.lane_change_threshold:
            return -1
        return 0

    def _scan_lane_neighbors(self, direction: int):
        """목표 차선을 직접 스캔해 (가장 가까운 앞차 gap·속도, 뒷차 gap·속도) 반환.

        getNeighbors의 백업. getNeighbors는 SUMO 차선변경모델의 이웃 캐시를
        읽는데, "방금 스폰된 차량"은 다음 스텝까지 이 캐시에 없어서
        도로 입구 근처에서 사각지대가 생긴다. 차선의 차량 목록을 직접
        훑으면 이미 삽입된 차량은 전부 잡힌다.
        """
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            li = traci.vehicle.getLaneIndex(self.ego_id) + direction
            # 존재하지 않는 차선이면 조회하지 않는다.
            # (없는 lane id를 traci.lane.*로 조회하면 SUMO가 매번
            #  에러 로그를 찍으므로, 범위 검사를 먼저 해서 조용히 건너뛴다)
            if edge.startswith(":") or not self._is_car_lane(edge, li):
                return None, None
            my_pos = traci.vehicle.getLanePosition(self.ego_id)
            # 같은 엣지 + 경로상 바로 앞/뒤 엣지의 같은 번호 차선도 훑는다. 보호구역 때문에
            # 도로를 여러 엣지로 나눈 뒤, 이음매 바로 너머 옆 차선 차량을 못 봐서
            # 이음매(x≈100 m) 직전 차선변경 측면충돌이 났다 (1M run 충돌의 절반).
            scan = [(f"{edge}_{li}", 0.0)]
            route = traci.vehicle.getRoute(self.ego_id)
            ri = traci.vehicle.getRouteIndex(self.ego_id)
            if 0 <= ri < len(route) - 1 and self._is_car_lane(route[ri + 1], li):
                scan.append((f"{route[ri + 1]}_{li}",
                             traci.lane.getLength(f"{edge}_{li}")))
            if ri >= 1 and self._is_car_lane(route[ri - 1], li):
                prev_lane = f"{route[ri - 1]}_{li}"
                scan.append((prev_lane, -traci.lane.getLength(prev_lane)))
            front, rear = None, None
            for lane_id, off in scan:
                for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                    d = traci.vehicle.getLanePosition(vid) + off - my_pos
                    if d >= 0:
                        g = d - traci.vehicle.getLength(vid)
                        if front is None or g < front[0]:
                            front = (g, traci.vehicle.getSpeed(vid))
                    else:
                        g = -d - traci.vehicle.getLength(self.ego_id)
                        if rear is None or g < rear[0]:
                            rear = (g, traci.vehicle.getSpeed(vid))
            return front, rear
        except traci.TraCIException:
            return None, None

    def hotspot_training_label(self):
        """Supervised labels on training states only. Lane class 0=left, 1=keep."""
        ego = self.ego_id
        if ego not in traci.vehicle.getIDList():
            return -1, ""
        edge = traci.vehicle.getRoadID(ego)
        pos = traci.vehicle.getLanePosition(ego)
        lane = traci.vehicle.getLaneIndex(ego)
        for site, road, lo, hi, *_ in self._hotspots:
            if site not in ("samil_east", "hanbit") or road != edge or not lo - 60 <= pos <= hi:
                continue
            if not any(item[0] == site for item in self._hotspot_placed):
                return -1, ""
            st = self._current_stop()
            if (st and st["edge"] == edge and st["lane"] == lane
                    and st["start"] - 2 <= pos <= st["end"] + self.stop_overshoot):
                return 1, site
            lead = traci.vehicle.getLeader(ego, 50)
            if not lead or not lead[0].startswith("hs_" + site + "_"):
                return -1, site
            if (self._is_car_lane(edge, lane + 1) and self._lane_leads_on_route(edge, lane + 1)
                    and not self._lc_forbidden() and self._lane_change_is_safe(1)):
                return 0, site
            return -1, site
        return -1, ""

    def _lane_change_is_safe(self, direction: int) -> bool:
        """차선변경 게이트 (shield의 한 종류): 목표 차선 앞뒤 간격이 충분한가.

          (1) 앞: gap > min_front_gap + front_tau × 내속도
              + 접근 여유: gap > (내속도 − 앞차속도)⁺ × approach_horizon   (내가 더 빠를 때)
          (2) 뒤: gap > min_rear_gap + rear_tau × 뒤차속도
              + 접근 여유: gap > (뒤차속도 − 내속도)⁺ × approach_horizon   (뒤가 더 빠를 때)
        이웃은 getNeighbors 결과와 차선 직접 스캔 결과 중 "더 가까운 쪽"을
        쓴다 (스폰 직후 차량 사각지대 보완).
        env-v1부터 교차로 직후·정지선 앞 금지는 여기서 막지 않는다 (_lc_forbidden 감점으로).
        """
        v = traci.vehicle.getSpeed(self.ego_id)
        scan_front, scan_rear = self._scan_lane_neighbors(direction)

        def closer(a, b):
            if a is None:
                return b
            if b is None:
                return a
            return a if a[0] <= b[0] else b

        lead = closer(self._neighbor(direction, leader=True), scan_front)
        if lead is not None:
            gap, lv = lead
            if gap <= self.lc_front_min + self.lc_front_tau * v:
                return False
            if gap <= max(v - lv, 0.0) * self.lc_horizon:  # 시간여유 내 접촉 위험
                return False

        foll = closer(self._neighbor(direction, leader=False), scan_rear)
        if foll is not None:
            gap, fv = foll
            if gap <= self.lc_rear_min + self.lc_rear_tau * fv:
                return False
            if gap <= max(fv - v, 0.0) * self.lc_horizon:  # 시간여유 내 접촉 위험
                return False
        return True

    def _apply_lane_change(self, lane_change: int):
        """차선변경 명령을 적용한다. 반환: (적용됐는가, 게이트가 막았을/막은 위험한 변경인가).

        물리적으로 불가능한 명령(교차로 안, 없는 차로)만 실행하지 않는다 (invalid).
        경로 밖 차로·금지 구간이어도 막지 않는다 (env-v1: 규칙 대신 차로 오류 종료·금지 구간 감점).
        shield가 켜져 있으면(on) 차선변경 게이트가 위험한 변경을 막고, 감시 모드(shadow)면
        막지 않고 "막았을 것"만 알린다.
        """
        if self.ego_id not in traci.vehicle.getIDList():
            return False, False
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                return False, False
            current = traci.vehicle.getLaneIndex(self.ego_id)
            target = current + lane_change
            if not self._is_car_lane(edge, target):
                return False, False
            unsafe = False
            if lane_change != 0 and self.shield_mode != "off":
                unsafe = not self._lane_change_is_safe(lane_change)
                if unsafe and self.shield_mode == "on":
                    return False, True
            # 유지(0)도 changeLane(현재 차로)로 차로를 고정한다 (SUMO가 차로를 옮기지 않게)
            traci.vehicle.changeLane(
                self.ego_id, target, self.lane_change_duration)
            return True, unsafe
        except traci.TraCIException:
            return False, False

    def _setup_route_signals(self):
        """ego 경로가 지나는 신호 링크를 지나는 순서대로 모은다 (reset에서 1회).

        [(신호 id, 링크 번호, 들어오는 엣지)] — 데이터의 신호 컬럼(신호등 하나당 독립 컬럼)용.
        같은 신호(tls.join으로 묶인 교차로)를 두 번 지나면 링크마다 따로 센다."""
        self._route_signals = []
        try:
            route = list(traci.vehicle.getRoute(self.ego_id))
            for tl_id in traci.trafficlight.getIDList():
                for idx, links in enumerate(traci.trafficlight.getControlledLinks(tl_id)):
                    for in_lane, out_lane, _ in links:
                        e_in = in_lane.rsplit("_", 1)[0]
                        e_out = out_lane.rsplit("_", 1)[0]
                        if e_in in route and route.index(e_in) + 1 < len(route) \
                                and route[route.index(e_in) + 1] == e_out:
                            self._route_signals.append((route.index(e_in), tl_id, idx, e_in))
            # 같은 (신호, 들어오는 엣지)는 한 번만 (차로마다 링크가 따로 있어 겹친다)
            seen, out = set(), []
            for ri, tl_id, idx, e_in in sorted(self._route_signals):
                if (tl_id, e_in) in seen:
                    continue
                seen.add((tl_id, e_in))
                out.append((tl_id, idx, e_in))
            self._route_signals = out
        except traci.TraCIException:
            self._route_signals = []

    def _signal_states(self) -> list:
        """경로 위 신호 상태 (privileged state, 3인칭). 신호등(통과 지점) 하나당 한 항목.

        색은 문자열 대신 숫자 one-hot으로 나눈다: red / yellow / green / stop(일시정지 후 진행).
        time_to_switch는 데이터·분석용이고 관측에는 넣지 않는다."""
        out = []
        now = traci.simulation.getTime()
        for k, (tl_id, idx, e_in) in enumerate(getattr(self, "_route_signals", [])):
            try:
                state = traci.trafficlight.getRedYellowGreenState(tl_id)[idx]
                color = _COLOR_OF_STATE.get(state, "red")
                rec = {"k": k, "id": tl_id, "edge": e_in}
                rec.update({c: 1 if c == color else 0 for c in SIGNAL_COLORS})
                rec["time_to_switch"] = float(traci.trafficlight.getNextSwitch(tl_id) - now)
                out.append(rec)
            except traci.TraCIException:
                continue
        return out

    def get_privileged_state(self) -> dict:
        """POMDP 데이터셋용 simulator-side state.

        agent가 실제로 받는 입력은 _get_obs()의 부분관측이고, 이 메서드는
        분석/지도신호용 privileged state만 수집한다. 정책 입력으로 쓰지 않는다.
        """
        if self.ego_id not in traci.vehicle.getIDList():
            return {"ego_alive": False}
        try:
            x, y = traci.vehicle.getPosition(self.ego_id)
            vehicles = []
            for vid in traci.vehicle.getIDList():
                try:
                    vx, vy = traci.vehicle.getPosition(vid)
                    vehicles.append({
                        "id": vid,
                        "x": float(vx),
                        "y": float(vy),
                        "speed": float(traci.vehicle.getSpeed(vid)),
                        "lane": self._car_lane_index(vid),
                        "lane_id": traci.vehicle.getLaneID(vid),
                        "road_id": traci.vehicle.getRoadID(vid),
                        "lane_pos": float(traci.vehicle.getLanePosition(vid)),
                    })
                except traci.TraCIException:
                    continue
            persons = []   # 교차로 layout의 보행자 (road_id가 ":..._c*"면 횡단보도 위)
            for pid in traci.person.getIDList():
                try:
                    px, py = traci.person.getPosition(pid)
                    persons.append({
                        "id": pid,
                        "type": traci.person.getTypeID(pid),   # ped / child (어린이)
                        "x": float(px),
                        "y": float(py),
                        "speed": float(traci.person.getSpeed(pid)),
                        "road_id": traci.person.getRoadID(pid),
                    })
                except traci.TraCIException:
                    continue
            return {
                "ego_alive": True,
                "ego_x": float(x),
                "ego_y": float(y),
                "ego_speed": float(traci.vehicle.getSpeed(self.ego_id)),
                "ego_lane": self._car_lane_index(self.ego_id),
                "ego_lane_id": traci.vehicle.getLaneID(self.ego_id),
                "ego_road_id": traci.vehicle.getRoadID(self.ego_id),
                "ego_lane_pos": float(traci.vehicle.getLanePosition(self.ego_id)),
                "vehicles": vehicles,
                "persons": persons,
                "signals": self._signal_states(),
            }
        except traci.TraCIException:
            return {"ego_alive": False}

    def get_state_columns(self) -> dict:
        """데이터셋 state.* 컬럼: 3인칭(시뮬레이터) 상태를 숫자 컬럼 하나씩으로 편 것.

        도로 상황을 알 수 있을 정도로만 모은다 (모든 차 좌표 같은 재현용 목록은 state 딕셔너리에).
        학습 입력이 아니다 — 학습에는 obs.* 컬럼만 쓴다 (docs/data_columns.md).
        - 신호등은 경로가 지나는 순서대로 정지선 하나당 독립 컬럼: sig{k}.red/yellow/green/stop (0/1),
          sig{k}.time_to_switch (s), sig{k}.dist (정지선까지 경로상 거리 m, 지났으면 -1).
          칸은 항상 STATE_SIGNAL_SLOTS개 — 신호가 없는 단계는 모든 값이 -1
        - 정류장은 하나당: stop{k}.status (0 남음 / 1 정차 성공 / -1 놓침), stop{k}.dist (m, 지났으면 -1)
        - 값이 없으면 -1 (예: 앞차 없음)
        ego가 없으면(종료 뒤) 빈 dict.
        """
        e = self.ego_id
        if e not in traci.vehicle.getIDList():
            return {}
        col = {}
        try:
            x, y = traci.vehicle.getPosition(e)
            col.update({
                "state.ego.x": float(x), "state.ego.y": float(y),
                "state.ego.speed": float(traci.vehicle.getSpeed(e)),
                "state.ego.accel": float(traci.vehicle.getAcceleration(e)),
                "state.ego.angle": float(traci.vehicle.getAngle(e)),
                "state.ego.lane": int(self._car_lane_index(e)),
                "state.ego.lane_pos": float(traci.vehicle.getLanePosition(e)),
                "state.ego.route_index": int(traci.vehicle.getRouteIndex(e)),
                "state.ego.odometer": float(traci.vehicle.getDistance(e)),
                "state.ego.in_junction": int(traci.vehicle.getRoadID(e).startswith(":")),
            })
            lead = traci.vehicle.getLeader(e, 100.0)
            col["state.lead.gap"] = float(lead[1]) if lead and lead[0] else -1.0
            col["state.lead.speed"] = (float(traci.vehicle.getSpeed(lead[0]))
                                       if lead and lead[0] else -1.0)
            near = 0
            for vid in traci.vehicle.getIDList():
                if vid == e:
                    continue
                vx, vy = traci.vehicle.getPosition(vid)
                near += int((vx - x) ** 2 + (vy - y) ** 2 <= 50.0 ** 2)
            col["state.traffic.n_vehicles"] = int(traci.vehicle.getIDCount()) - 1
            col["state.traffic.n_near50"] = near
            col["state.traffic.n_persons"] = int(traci.person.getIDCount())
            now = traci.simulation.getTime()
            for k, (tl_id, idx, e_in) in enumerate(getattr(self, "_route_signals", [])):
                state = traci.trafficlight.getRedYellowGreenState(tl_id)[idx]
                color = _COLOR_OF_STATE.get(state, "red")
                for c in SIGNAL_COLORS:
                    col[f"state.sig{k}.{c}"] = int(c == color)
                col[f"state.sig{k}.time_to_switch"] = float(traci.trafficlight.getNextSwitch(tl_id) - now)
                lo = self._car_lanes(e_in)[0]
                d = traci.vehicle.getDrivingDistance(e, e_in, traci.lane.getLength(f"{e_in}_{lo}"), lo)
                col[f"state.sig{k}.dist"] = float(d) if d > -1e5 else -1.0
            # 신호가 없거나 적은 단계: 남은 칸을 -1(없음)로 채운다
            for k in range(len(getattr(self, "_route_signals", [])), STATE_SIGNAL_SLOTS):
                for c in (*SIGNAL_COLORS, "time_to_switch", "dist"):
                    col[f"state.sig{k}.{c}"] = -1
            for k, st in enumerate(self._stops):
                col[f"state.stop{k}.status"] = int(self._stop_status[k])
                d = -1.0
                if self._stop_status[k] == 0:
                    d = traci.vehicle.getDrivingDistance(e, st["edge"], st["end"], st["lane"])
                    d = float(d) if d > -1e5 else -1.0
                col[f"state.stop{k}.dist"] = d
            zs = self._zone_state()
            col["state.school_zone.inside"] = int(zs is not None and zs[0] <= 0.0)
            col["state.school_zone.dist"] = -1.0 if zs is None else float(zs[0])
            col["state.pax.onboard"] = len(self._onboard)
            st = self._current_stop()
            col["state.pax.waiting_next_stop"] = (len(traci.busstop.getPersonIDs(st["id"]))
                                                  if st is not None else 0)
        except traci.TraCIException:
            pass
        return col

    def _shield_check(self, accel_raw: float, v: float):
        """shield 네 종류(보행자 필터·차량·신호·차선변경 게이트 중 앞의 셋)를 모델 명령에 적용해 본다.

        반환: (shield 적용 뒤 가속 명령, {"ped","vehicle","signal": 개입 여부}, 개입 세기 0~1).
        각 종류의 개입 여부는 "모델 원래 명령"에 대해 따로 판단한다 (감시 모드에서 유형별로 센다).
        보행자 필터는 판정용 계산(conflict_now)이 필요해 shield가 꺼져 있어도 이 함수 밖에서 돈다."""
        hits = {"ped": False, "vehicle": False, "signal": False}
        out = accel_raw
        p_cmd, hits["ped"] = self._ped_cmd
        v_cmd, hits["vehicle"] = self._shield_accel(accel_raw, v)
        s_cmd, hits["signal"] = self._signal_shield_accel(accel_raw, v, self._next_signal())
        for cmd, hit in ((p_cmd, hits["ped"]), (v_cmd, hits["vehicle"]), (s_cmd, hits["signal"])):
            if hit:
                out = min(out, cmd)
        f = lambda r: r * (self.max_accel if r >= 0 else self.max_decel)
        severity = max(0.0, f(accel_raw) - f(out)) / self.max_decel
        return out, hits, severity

    def step(self, action):
        # action = [가감속 raw, 차선변경 raw]
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size < self.act_dim:
            raise ValueError(
                f"action은 {self.act_dim}차원이어야 합니다. 받은 shape={np.asarray(action).shape}")

        accel_raw = float(np.clip(action_arr[self.accel_index], -1.0, 1.0))
        accel_model = accel_raw
        lane_raw = float(np.clip(action_arr[self.lane_change_index], -1.0, 1.0))
        lane_change = self._quantize_lane_change(lane_raw)
        lane_change_applied = False
        hits = {"ped": False, "vehicle": False, "signal": False, "lc_gate": False}
        shield_severity = shadow_severity = 0.0
        lc_forbidden = False
        lane_before = None
        lane_edge = None
        train_lane_penalties = "lane_reversal_penalty" in self.rw
        odo_before = None

        # 행동 결정 시점의 전방 신호 (신호 위반 판정 기준 — 정책이 본 관측과 같은 시점)
        signal_before = self._next_signal()

        # ----- (1) 종방향 + 횡방향 행동 적용 -----
        if self.ego_id in traci.vehicle.getIDList():
            v = traci.vehicle.getSpeed(self.ego_id)
            lane_before = self._car_lane_index(self.ego_id)
            if train_lane_penalties:
                lane_edge = traci.vehicle.getRoadID(self.ego_id)
            odo_before = traci.vehicle.getDistance(self.ego_id)

            # 보행자 필터는 판정(보행자 앞 정당한 대기)에도 쓰므로 항상 계산한다
            self._ped_cmd = self.ped_filter.apply(accel_raw, self.ego_id)
            if self.shield_mode != "off":
                shielded, sh_hits, sev = self._shield_check(accel_raw, v)
                hits.update(sh_hits)
                shadow_severity = sev
                if self.shield_mode == "on":
                    accel_raw, shield_severity = shielded, sev

            accel_cmd = (accel_raw * self.max_accel
                         if accel_raw >= 0.0 else accel_raw * self.max_decel)
            dv = accel_cmd * self.step_length
            traci.vehicle.setSpeed(
                self.ego_id, float(np.clip(v + dv, 0.0, self.ego_vmax)))

            # 금지 구간은 변경 "명령 시점" 위치로 판정한다
            lc_forbidden = lane_change != 0 and self._lc_forbidden()
            lane_change_applied, hits["lc_gate"] = self._apply_lane_change(lane_change)

        shield_any = any(hits.values())

        # ----- (2) 시뮬레이션 1스텝 -----
        traci.simulationStep()
        self.step_count += 1

        # ----- (3) 이벤트/관측 -----
        collided = self.ego_id in traci.simulation.getCollidingVehiclesIDList()
        ped_collision = collided and any(
            self.ego_id in (c.collider, c.victim)
            and "pedestrian" in (traci.vehicletype.getVehicleClass(c.colliderType),
                                 traci.vehicletype.getVehicleClass(c.victimType))
            for c in traci.simulation.getCollisions())
        arrived = self.ego_id in traci.simulation.getArrivedIDList()
        alive = self.ego_id in traci.vehicle.getIDList()
        lane_after = self._car_lane_index(self.ego_id) if alive else None

        terminated, truncated = False, False
        ego_speed, ego_gap = None, None
        approach_pen, stop_pen, zone_pen = 0.0, 0.0, 0.0   # 진단용 (info에 기록)

        # ---- 신호 위반: 결정 시점에 적색이던 신호의 정지선을 이번 스텝에 넘음 ----
        # 넘었는지는 "다음 신호 목록에서 그 신호가 사라졌는지"로 본다.
        # 황색 진입은 위반으로 보지 않는다. (ego는 speedMode=0이라 SUMO가
        # 대신 멈춰주지 않으므로, 이 판정과 보상으로 신호 준수를 학습한다)
        # 일시정지(s)는 정지선 앞 _STOP_ZONE 안에서 한 번 멈췄으면 지나가도 된다.
        signal_after = self._next_signal() if alive else None
        # 같은 신호(tls.join으로 묶인 교차로)의 정지선이 연달아 있으면 id가 같으므로,
        # 다음 정지선까지 거리가 늘어난 것도 "지났다"로 본다.
        passed = bool(alive and signal_before is not None
                      and (signal_after is None or signal_after[0] != signal_before[0]
                           or signal_after[1] > signal_before[1] + 0.5))
        before_level = (self._SIGNAL_STOP_LEVEL.get(signal_before[2], 1.0)
                        if signal_before is not None else 0.0)
        red_violation = passed and (
            before_level >= 1.0
            or (signal_before[2] == "s" and self._stopped_at != signal_before[0]))
        if alive and signal_after is not None:
            if (signal_after[1] <= self._STOP_ZONE
                    and traci.vehicle.getSpeed(self.ego_id) < 0.1):
                self._stopped_at = signal_after[0]
        # ---- 정류장: 정차 성공(정류장 구간에서 dwell_time 연속 정지) / 놓침 판정 ----
        # 충돌로 사라진 판은 남은 정류장을 놓침으로 세지 않는다 (SUMO는 충돌 제거도 arrived로 올린다)
        stop_ok, stop_missed = self._update_stops(alive, arrived and not collided)
        st_now = self._current_stop() if alive else None
        in_stop_area = bool(st_now is not None and self._in_stop_area(st_now))
        # 일시정지 완료·정차 진행도가 이번 스텝 관측에 바로 반영되도록 판정 뒤에 만든다
        obs = self._get_obs()
        # ---- 횡단보도 양보 위반: 보행자가 있는데 앞범퍼가 횡단보도에 들어섬 ----
        yield_violation = bool(alive and odo_before is not None and self._check_yield(
            odo_before, traci.vehicle.getDistance(self.ego_id)))
        # ---- 차로 오류: 경로로 이어지지 않는 차로 끝에서 멈춤 (갇혀서 못 나감) ----
        wrong_lane = False
        if alive and not arrived:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if not edge.startswith(":"):
                lane_index = traci.vehicle.getLaneIndex(self.ego_id)
                remaining = (traci.lane.getLength(f"{edge}_{lane_index}")
                             - traci.vehicle.getLanePosition(self.ego_id))
                wrong_lane = (remaining <= self._STOP_ZONE
                              and traci.vehicle.getSpeed(self.ego_id) < 0.1
                              and not self._lane_leads_on_route(edge, lane_index))

        # ----- (4) 기존 보상 유지 -----
        if collided:
            reward = -float(self.rw["collision_penalty"])
            terminated = True
        elif red_violation and self.rw.get("red_light_terminate", True):
            reward = -float(self.rw.get("red_light_penalty", 0.0))
            terminated = True
        elif wrong_lane:
            reward = -float(self.rw.get("wrong_lane_penalty", 0.0))
            terminated = True
        elif arrived:
            reward = float(self.rw["arrival_bonus"])
            terminated = True
        elif not alive:
            reward = 0.0
            terminated = True
        else:
            ego_speed = traci.vehicle.getSpeed(self.ego_id)
            same_lane_leader = self._neighbor(0, leader=True)
            if same_lane_leader is not None:
                ego_gap = same_lane_leader[0]
            reward = float(self.rw["speed_weight"]) * ego_speed / self.vmax
            if (ego_gap is not None and
                    ego_gap < float(self.rw["close_gap_threshold"])):
                reward -= float(self.rw["close_gap_penalty"])

            # ---- 막힘 페널티: 느린 앞차 뒤에 갇혀 있는 동안 매 스텝 감점 ----
            # "차선을 유지한 채 막혀 있기"와 "변경으로 탈출하기" 사이에
            # 즉각적인 보상 차등을 만들어 추월 학습 신호를 조밀하게 한다.
            # (자세한 설계 이유는 env/mdp_config.py의 REWARD 주석 참고)
            # 단, 전방 신호가 적/황색이면 앞차가 신호 대기 중인 것이므로
            # 막힘으로 보지 않는다 (합법적 대기를 벌주면 대기열을 피하려는
            # 차선변경을 배우게 된다).
            waiting_for_signal = (
                signal_after is not None
                and signal_after[1] <= self.signal_visibility
                and self._SIGNAL_STOP_LEVEL.get(signal_after[2], 1.0) > 0.0)
            # 횡단보도 앞에서 보행자를 기다리는 것도 정당한 대기다
            waiting_for_signal = waiting_for_signal or self._waiting_for_ped()
            # blocked_gap=None이면 거리 조건 없이 "느리면 감점" (앞차와 거리를
            # 벌려 감점을 피하는 캠핑 정책 차단).
            bp = float(self.rw.get("blocked_penalty", 0.0))
            blocked_gap = self.rw.get("blocked_gap", 20.0)
            near_leader = blocked_gap is None or (
                ego_gap is not None and ego_gap < float(blocked_gap))
            if (bp > 0.0 and not waiting_for_signal and near_leader
                    and ego_speed < float(self.rw.get("blocked_speed_frac", 0.7))
                        * self.vmax):
                reward -= bp

            # ---- 진행 보상: 이번 스텝 이동거리 / 도로 길이 × progress_weight ----
            # 완주까지 누적하면 정확히 progress_weight. 속도 보상과 달리 상한이
            # 고정이라 "빨리 완주해서 남은 스텝을 잃는" 손해가 없다.
            dist = traci.vehicle.getDistance(self.ego_id)
            pw = float(self.rw.get("progress_weight", 0.0))
            if pw > 0.0:
                reward += pw * max(dist - self._prev_dist, 0.0) / self._route_length
            self._prev_dist = dist

            # ---- 차선변경 보상 항 ----
            # lane_change        : 이번 스텝에 policy가 낸 양자화 명령 (-1/0/+1)
            # lane_change_applied: 그 명령이 실제로 실행됐는지
            # 주의: _apply_lane_change는 유지 명령(0)에도 현재 차선으로
            # changeLane을 호출해 True를 반환한다(차선 고정 용도).
            # 따라서 "실제 변경"은 lane_change != 0 과 함께 판정해야 한다.
            if lane_change != 0 and lane_change_applied:
                # 실행된 변경마다 소액 감점 → 이득 없는 지그재그 억제.
                # 변경 덕에 속도 보상이 그 이상 늘면 순이득이므로,
                # "필요할 때만 바꾸는" 정책으로 수렴하게 만드는 항이다.
                reward -= float(self.rw.get("lane_change_penalty", 0.0))
            elif lane_change != 0:
                # 명령은 냈는데 실행 불가(차선 없음/교차로)였던 경우 감점
                # → 의미 없는 행동 출력을 줄이는 정칙화 (rl_action_penalty 대응)
                reward -= float(self.rw.get("invalid_action_penalty", 0.0))

            # 종료하지 않는 설정(red_light_terminate=False)이면 감점만 하고 계속 주행
            if red_violation:
                reward -= float(self.rw.get("red_light_penalty", 0.0))

            # ---- 교차로 안 회전 속도: 내부 차로 제한속도 초과 시 스텝당 감점 ----
            lane_id = traci.vehicle.getLaneID(self.ego_id)
            if (lane_id.startswith(":")
                    and ego_speed > traci.lane.getMaxSpeed(lane_id) + 0.1):
                reward -= float(self.rw.get("turn_overspeed_penalty", 0.0))

            # ---- 신호 접근 감속: 정지선에서 멈출 수 있는 속도보다 빠르면 스텝당 감점 ----
            # 허용 속도 v_ref = sqrt(2 · approach_decel · (정지선 거리 - 1 m)), 즉 편한 감속으로
            # 정지선 1 m 앞에 멈출 수 있는 속도. 적용 대상: 적색·황색·멈추기 전 일시정지(s),
            # 그리고 곧 끝나는 녹색(남은 시간 < 지금 속도로 정지선까지 걸리는 시간).
            # 황색인데 최대 감속으로도 못 멈추면 통과가 맞으므로 감점하지 않는다.
            # (멀리 서던 이유: 다가가다 못 멈추면 -20이라, 안전한 접근법을 모르고 미리 섰다)
            ap = float(self.rw.get("approach_overspeed_penalty", 0.0))
            if ap > 0.0 and signal_after is not None and signal_after[1] <= self.signal_visibility:
                _, sdist, sstate, sremain = signal_after
                level = self._SIGNAL_STOP_LEVEL.get(sstate, 1.0)
                if sstate == "s" and self._stopped_at == signal_after[0]:
                    level = 0.0
                ending_green = level == 0.0 and sstate in "Gg" and \
                    sremain < sdist / max(ego_speed, 1.0)
                stoppable = ego_speed ** 2 / (2.0 * self.max_decel) <= sdist
                if (level >= 1.0 or ending_green
                        or (level > 0.0 and (sstate != "y" or stoppable))):
                    v_ref = float(np.sqrt(2.0 * float(self.rw.get("approach_decel", 2.0))
                                          * max(sdist - 1.0, 0.0)))
                    if ego_speed > v_ref:
                        approach_pen = min(ap * (ego_speed - v_ref),
                                           float(self.rw.get("approach_penalty_cap", 0.2)))
                        reward -= approach_pen

            # ---- 도로 중간 정지: 멈춰야 할 이유 없이 서 있거나 아주 느리면 스텝당 감점 ----
            # 앞이 빈 거리 F = min(정지 신호의 정지선 거리, 앞차 거리). F가 green_stop_gap
            # 이하면 정당한 대기. 넘으면 base에서 시작해 F가 멀수록 커진다 (경계 15 m에
            # 맞춰 16~20 m에 서는 습관을 막으려고 계단 대신 경사). 보행자 대기는 제외.
            # 정지선 앞에서 양보 신호(g, 또는 이미 멈춘 일시정지 s)를 받고 있으면 교차·대향
            # 차량 틈을 기다리는 것일 수 있어 base만 준다 (세게 밀면 끼어들기 충돌 위험).
            gsp = float(self.rw.get("green_stop_penalty", 0.0))
            stop_gap = float(self.rw.get("green_stop_gap", 15.0))
            if gsp > 0.0 and ego_speed < float(self.rw.get("green_stop_speed", 1.5)):
                free = np.inf
                s_done = False
                if signal_after is not None and signal_after[1] <= self.signal_visibility:
                    if signal_after[2] == "g" or (signal_after[2] == "s"
                                                  and self._stopped_at == signal_after[0]):
                        s_done = signal_after[1] <= stop_gap
                    elif self._SIGNAL_STOP_LEVEL.get(signal_after[2], 1.0) > 0.0:
                        free = signal_after[1]
                if ego_gap is not None:
                    free = min(free, ego_gap)
                ped_ahead = (self._waiting_for_ped()
                             or (self.ped_filter is not None and self.ped_filter.conflict_now))
                # 정류장 구간 안에서 서 있는 것은 정차 중이라 정당하다
                if not ped_ahead and not in_stop_area:
                    if s_done and free > stop_gap:
                        stop_pen = gsp
                    elif free > stop_gap:
                        ramp = float(self.rw.get("green_stop_ramp", 30.0))
                        extra = float(self.rw.get("green_stop_extra", 0.0))
                        stop_pen = gsp + extra * min((free - stop_gap) / ramp, 1.0)
                    reward -= stop_pen

            # ---- 어린이 보호구역 속도 ----
            # 안: 제한속도 + tol 초과 1 m/s당 스텝 감점 (cap까지)
            # 앞: 편한 감속(zone_approach_decel)으로 입구에서 제한속도가 될 수 있는 속도
            #     v_ref = sqrt(limit² + 2·a·(거리 - 1))보다 빠르면 초과 1 m/s당 감점 (cap까지)
            zone_now = self._zone_state()
            if zone_now is not None:
                zdist, zlimit = zone_now
                if zdist <= 0.0:
                    over = ego_speed - zlimit - float(self.rw.get("zone_overspeed_tol", 0.3))
                    if over > 0.0:
                        zone_pen = min(float(self.rw.get("zone_overspeed_penalty", 0.0)) * over,
                                       float(self.rw.get("zone_overspeed_cap", 0.5)))
                elif zdist <= self.zone_visibility:
                    v_ref = float(np.sqrt(zlimit ** 2 + 2.0 * float(self.rw.get("zone_approach_decel", 2.0))
                                          * max(zdist - 1.0, 0.0)))
                    if ego_speed > v_ref:
                        zone_pen = min(float(self.rw.get("zone_approach_penalty", 0.0))
                                       * (ego_speed - v_ref),
                                       float(self.rw.get("zone_approach_cap", 0.2)))
                reward -= zone_pen

        # ---- 정류장: 정차 성공 +, 놓침 − (도착 스텝에 남은 정류장을 놓친 것도 포함) ----
        reward += float(self.rw.get("stop_success_bonus", 0.0)) * stop_ok
        reward -= float(self.rw.get("missed_stop_penalty", 0.0)) * stop_missed
        dwelling = bool(in_stop_area and ego_speed is not None and ego_speed <= self.stop_speed)
        # ---- 차선변경 금지 구간에서 변경 실행 (막지 않고 감점) ----
        if lc_forbidden and lane_change_applied:
            reward -= float(self.rw.get("lc_forbidden_penalty", 0.0))
        # shield가 실제로 개입한 스텝 감점 (shield on 모드에서만. 학습 기본은 off라 0)
        if self.shield_mode == "on" and shield_any and not terminated:
            reward -= (float(self.rw.get("shield_penalty", 0.0))
                       + float(self.rw.get("shield_penalty_scale", 0.0)) * shield_severity)

        # 양보 위반은 종료하지 않고 감점만 (충돌·완주와 같은 스텝이어도 적용)
        if yield_violation:
            reward -= float(self.rw.get("yield_violation_penalty", 0.0))

        if self.step_count >= self.max_steps:
            truncated = True
            if not terminated:
                reward -= float(self.rw.get("timeout_penalty", 0.0))

        zone_info = self._zone_state() if alive and not arrived else None
        info = {
            "collided": collided,
            "arrived": arrived,
            "speed": ego_speed,
            "gap": ego_gap,
            "accel_model": accel_model,      # 모델이 낸 가감속 명령
            "accel_raw": accel_raw,          # 실제 적용 값 (shield on이 바꿨으면 바뀐 값)
            # shield (수동개입): 모드, 이번 스텝 유형별로 "개입했을/개입한" 여부, 하나라도
            "shield_mode": self.shield_mode,
            "shield_vehicle": hits["vehicle"],
            "shield_signal": hits["signal"],
            "shield_ped": hits["ped"],
            "shield_lc_gate": hits["lc_gate"],
            "shield_active": shield_any,
            "filter_active": hits["ped"],    # 예전 이름 (보행자 필터) 호환
            "dwelling": dwelling,            # 정류장 구간에서 정지 중
            "in_stop_area": in_stop_area,
            "dwell_time": self._dwell_steps * self.step_length,
            "stop_success": int(stop_ok),    # 이번 스텝 정차 성공 수
            "stop_missed": int(stop_missed), # 이번 스텝 놓친 정류장 수
            "stops_done": self._stops_done,
            "missed_stops": self._missed_stops,
            "stops_total": len(self._stops),
            "stop_pos_errors": list(self._stop_errors),
            "lc_forbidden": bool(lc_forbidden),
            "approach_pen": approach_pen,
            "stop_pen": stop_pen,
            "lane_change_raw": lane_raw,
            "lane_change": lane_change,
            "lane_change_applied": lane_change_applied,
            "lane_before": lane_before,
            "lane_after": lane_after,
            "red_violation": red_violation,
            "ped_collision": ped_collision,
            "yield_violation": yield_violation,
            "wrong_lane": wrong_lane,
            "route": self.ego_route,
            # 어린이 보호구역 (스텝 끝 기준): 안인가 / 입구까지 거리 m (안이면 0, 없으면 None) /
            # 적용 제한속도 m/s (앞이나 안에 없으면 None) / 이번 스텝 보호구역 감점
            "in_school_zone": bool(zone_info is not None and zone_info[0] <= 0.0),
            "zone_dist": None if zone_info is None else float(zone_info[0]),
            "zone_limit": None if zone_info is None else float(zone_info[1]),
            "zone_pen": zone_pen,
            # 결정 시점의 전방 신호 (없으면 None). 색은 SIGNAL_COLORS 이름 (red/yellow/green/stop)
            "signal_state": signal_before[2] if signal_before else None,
            "signal_color": (_COLOR_OF_STATE.get(signal_before[2], "red")
                             if signal_before else None),
            "signal_dist": signal_before[1] if signal_before else None,
        }
        # 정차금지지대 안에서 거의 서 있으면 스텝마다 감점.
        # 어린이보호구역은 제한을 넘긴 1 m/s당 감점, 상한 0.5.
        if alive and not traci.vehicle.getRoadID(self.ego_id).startswith(":"):
            edge = traci.vehicle.getRoadID(self.ego_id)
            pos = traci.vehicle.getLanePosition(self.ego_id)
            spd = traci.vehicle.getSpeed(self.ego_id)
            if spd < 0.5 and any(
                    z[0] == edge and z[1] <= pos <= z[2] for z in self._no_stop_zones):
                reward -= float(self.rw.get("no_stop_zone_penalty", 0.0))
            # 노면에서 보호구역을 읽었으면(self._zones) 과속은 구간 기준 zone_overspeed_penalty가
            # 맡는다 (교차로 안까지 포함). 같은 과속을 두 번 깎지 않도록 엣지 기준 감점은 건너뛴다.
            limit = None if self._zones else self._school_limits.get(edge)
            if limit is not None and spd > limit:
                rate = float(self.rw.get("school_zone_speeding_penalty", 0.0))
                reward -= min(rate * (spd - limit), 0.5)
        # 교차로 단위 정차금지지대 (no_stop_zones.csv의 junction_id): 그 교차로 내부 차로
        # (":<교차로 id>_<번호>")에서 거의 서 있으면 스텝마다 같은 감점.
        if alive and self._no_stop_junctions:
            rid = traci.vehicle.getRoadID(self.ego_id)
            if (rid.startswith(":") and rid[1:rid.rfind("_")] in self._no_stop_junctions
                    and traci.vehicle.getSpeed(self.ego_id) < 0.5):
                reward -= float(self.rw.get("no_stop_zone_penalty", 0.0))
        if (terminated or truncated) and not self.eval_mode and self.replay_prob > 0:
            self._update_replay(bool(arrived))
        if train_lane_penalties:
            reversal_penalty = unsafe_penalty = 0.0
            if self.step_count == 1:
                self._previous_policy_lane_change = None
            if (alive and lane_change and lane_change_applied
                    and lane_before is not None and lane_after is not None
                    and lane_after - lane_before == lane_change
                    and lane_edge == traci.vehicle.getRoadID(self.ego_id)):
                previous = self._previous_policy_lane_change
                if previous and previous[1] == -lane_change and self.step_count - previous[0] <= 10:
                    reversal_penalty = self.rw["lane_reversal_penalty"]
                if hits["lc_gate"]:
                    unsafe_penalty = self.rw["unsafe_lane_change_penalty"]
                self._previous_policy_lane_change = (self.step_count, lane_change)
            reward -= reversal_penalty + unsafe_penalty
            info["lane_reversal_penalty"] = reversal_penalty
            info["unsafe_lane_change_penalty"] = unsafe_penalty
        if "hotspot_stall_penalty" in self.rw:
            recovery = 0.0
            if (alive and not terminated and ego_speed is not None and ego_speed < 1.5
                    and not in_stop_area and not self._waiting_for_ped()
                    and not self._lc_forbidden()):
                signal_blocks = (signal_after is not None and signal_after[1] <= 25.0
                                 and self._SIGNAL_STOP_LEVEL.get(signal_after[2], 1.0) > 0.0)
                leader = traci.vehicle.getLeader(self.ego_id, 20.0)
                if not signal_blocks and leader and leader[0].startswith("hs_") and leader[1] <= 20.0:
                    edge = traci.vehicle.getRoadID(self.ego_id)
                    current = traci.vehicle.getLaneIndex(self.ego_id)
                    if not edge.startswith(":") and any(
                            self._is_car_lane(edge, current + direction)
                            and self._lane_leads_on_route(edge, current + direction)
                            and self._lane_change_is_safe(direction) for direction in (1, -1)):
                        recovery = self.rw["hotspot_stall_penalty"]
            risk = self.rw["unsafe_accel_penalty"] * shadow_severity
            reward -= recovery + risk
            info["hotspot_stall_penalty"] = recovery
            info["unsafe_accel_penalty"] = risk
        info["replayed"] = self._ep_replayed
        return obs, float(reward), terminated, truncated, info

    def _update_replay(self, arrived: bool):
        """학습 판이 끝날 때: 완주 못 했으면 그 시드를 다시 연습 목록에, 완주했으면 목록에서 뺀다."""
        s = self._ep_seed
        if s is None:
            return
        if arrived:
            if s in self._replay_seeds:
                self._replay_seeds.remove(s)
        elif s not in self._replay_seeds:
            self._replay_seeds.append(s)
            if len(self._replay_seeds) > self.replay_size:
                self._replay_seeds.pop(0)      # 오래된 것부터 버림

    # ==================================================================
    def close(self):
        """환경 정리. 학습이 끝나면 반드시 호출해서 SUMO 프로세스를 종료할 것."""
        if self._started:
            traci.close()
            self._started = False
