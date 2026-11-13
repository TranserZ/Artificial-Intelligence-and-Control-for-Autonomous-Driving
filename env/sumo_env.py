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


# 편도(동행) route에서 ego가 서는 순서. 정차 시간은 reset에서 15초.
# bs_06(광장시장)은 서행 엣지(e1212424785_1)에 있어 편도 동행 route(r0)에서는 제외한다.
BUS_STOPS_ORDER = ["bs_01", "bs_02", "bs_03", "bs_04", "bs_05"]

# 어린이보호구역(세운상가, 3차로) 구간 — 버스 노선은 가운데 차로를 유지한다.
SCHOOL_ZONE_CENTER_LANE_EDGES = ("e218924602_1", "e218924602_2", "e218924602_3")


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

        # ----- 도로/차량 설정에서 필요한 값들 -----
        self.vmax = float(road["speed_limit"])   # 속도 정규화 기준
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
        self.lc_safety_enabled = bool(lcs.get("enabled", True))
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

        # ----- MDP 설정 -----
        self.step_length = float(mdp_sim["step_length"])
        self.max_steps = int(mdp_sim["max_steps"])
        self.warmup_steps = int(mdp_sim["warmup_steps"])
        self.ped_filter = PedSafetyFilter(self) if mdp_sim.get("ped_filter") else None
        self.visibility = float(mdp_obs["visibility"])   # 관측 거리 W
        # 밀도 정규화용 "차량 1대가 점유하는 길이" (차체 + 최소 간격)
        self.density_veh_len = float(mdp_obs.get("density_veh_length", 7.5))
        # 전방 신호 관측 (mdp_config.OBSERVATION["signal"])
        sig = (mdp_obs.get("signal") or {})
        self.signal_obs = bool(sig.get("enabled", False))
        self.signal_visibility = float(sig.get("visibility", 150.0))
        self.signal_time_scale = float(sig.get("time_scale", 30.0))
        # 경로상 횡단보도 관측 (mdp_config.OBSERVATION["pedestrian"])
        ped = (mdp_obs.get("pedestrian") or {})
        self.ped_obs = bool(ped.get("enabled", False))
        self.ped_visibility = float(ped.get("visibility", 50.0))
        self.ped_num_crossings = int(ped.get("num_crossings", 2))
        # ego 경로 후보와 확률 (road_config.EGO["routes"]). 네트워크에 없는 route는 reset에서 제외
        self.ego_routes = dict(ego.get("routes") or {"r0": 1.0})
        self.route_depart_lane = dict(ego.get("route_depart_lane") or {})
        self._crossings, self._edge_ends = self._load_net_crossings()
        self.act_dim = int(action["dim"])
        self.accel_index = int(action.get("accel_index", 0))
        self.lane_change_index = int(action.get("lane_change_index", 1))
        self.lane_change_threshold = float(action.get("lane_change_threshold", 0.5))
        self.lane_change_duration = float(action.get("lane_change_duration", 1.0))
        self.rw = reward
        # 정차금지지대 (edge, 시작 m, 끝 m). 파일은 data/road_data/no_stop_zones.csv
        self._no_stop_zones = _load_zone_rows(
            "no_stop_zones.csv", ("edge_id", "start_pos", "end_pos"))
        # 어린이보호구역 edge → 제한속도 m/s
        self._school_limits = {
            edge: speed for edge, speed in _load_zone_rows(
                "school_zones.csv", ("edge_id", "speed_limit_ms"))
        }

        # GUI는 ego를 따라가며 가까이 보이도록 기본값을 둔다.
        self.gui_zoom = float(mdp_sim.get("gui_zoom", 1200.0))
        self.gui_delay = float(mdp_sim.get("gui_delay", 100.0))
        self.gui_track_ego = bool(mdp_sim.get("gui_track_ego", True))

        self.ego_id = "ego"        # SUMO 안에서 ego 차량을 식별하는 ID
        self.step_count = 0
        self._started = False      # SUMO 프로세스가 떠 있는지 여부

        # Gymnasium 행동 공간:
        #   action[0] = 종방향 가감속 raw ∈ [-1,1]
        #   action[1] = 차선변경 raw ∈ [-1,1], 환경 내부에서 {-1,0,+1}로 양자화
        # PPO의 Gaussian 정책과 호환하기 위해 Box를 유지한다. 실제 적용된
        # 차선변경 명령은 info["lane_change"]에 정확한 -1/0/+1로 반환한다.
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.act_dim,), dtype=np.float32)
        # 관측: 31차원(신호 관측 사용 시 34차원), 범위 [-1,1].
        #   레이아웃은 env/mdp_config.py 주석 참고.
        #   [0] 내 속도, [1..20] 5개 차선 × (선행/후행 상대거리·상대속도),
        #   [21..25] 연결성, [26..30] 밀도, [31..33] 전방 신호 (선택),
        #   [34..39] 경로상 횡단보도 2개 × (거리, 보행자 거리, 접근) (선택)
        #   시야 차선: 좌2, 좌1, 현재, 우1, 우2 (5개) — 3차선 도로에서
        #   맨 왼쪽에 있어도 오프셋 -2로 맨 오른쪽 차선까지 보인다.
        self.obs_offsets = (+2, +1, 0, -1, -2)
        n_lanes_view = len(self.obs_offsets)
        self.obs_dim = 1 + n_lanes_view * 4 + n_lanes_view + n_lanes_view
        if self.signal_obs:
            self.obs_dim += 3      # [31..33] 전방 신호 거리 / 정지 필요도 / 남은 시간
        if self.ped_obs:
            self.obs_dim += 3 * self.ped_num_crossings   # 횡단보도별 거리 / 보행자 거리 / 접근
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.obs_dim,),
                                            dtype=np.float32)

    def _load_net_crossings(self):
        """net.xml에서 (횡단보도 → 가로지르는 엣지들, 일반 엣지 → (from, to) 노드)를 읽는다.

        traci로는 횡단보도가 어느 엣지를 가로지르는지와 엣지의 양 끝 노드를
        알 수 없어서 (이 SUMO 버전엔 edge.getToJunction이 없음) 파일에서 직접 읽는다.
        """
        import xml.etree.ElementTree as ET
        cfg_dir = os.path.dirname(os.path.abspath(self.cfg_path))
        net_el = ET.parse(self.cfg_path).getroot().find("input/net-file")
        if net_el is None:
            return {}, {}
        crossings, ends = {}, {}
        for el in ET.parse(os.path.join(cfg_dir, net_el.get("value"))).getroot().iter("edge"):
            if el.get("function") == "crossing":
                crossings[el.get("id")] = el.get("crossingEdges", "").split()
            elif el.get("function") is None:
                ends[el.get("id")] = (el.get("from"), el.get("to"))
        return crossings, ends

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

    def _crossing_ped_state(self, c):
        """횡단보도 c의 (가장 가까운 보행자 ↔ 내 경로 구간 거리 m, 다가오는 중인가).

        대상: 횡단보도 위의 보행자 + 보행 녹색일 때 진입 대기 중인 보행자
        (보도 모서리에서 다음 엣지가 이 횡단보도인 보행자). 없으면 (None, False).
        """
        cache = self.__dict__.setdefault("_ped_cache", {})
        now = traci.simulation.getTime()
        if cache.get("t") != now:
            cache.clear()
            cache["t"] = now
            cache["roads"] = {pid: traci.person.getRoadID(pid)
                              for pid in traci.person.getIDList()}
        if c["id"] in cache:
            return cache[c["id"]]
        green = True
        if c["link"] is not None:
            tl_id, idx = c["link"]
            green = traci.trafficlight.getRedYellowGreenState(tl_id)[idx] in "Gg"
        (ax, ay), (ux, uy), (zlo, zhi) = c["a"], c["u"], c["zone"]
        best = (None, False)
        for pid, road in cache["roads"].items():
            on = road == c["id"]
            if not on and not (green and road.startswith(":")
                               and traci.person.getNextEdge(pid) == c["id"]):
                continue
            x, y = traci.person.getPosition(pid)
            t = (x - ax) * ux + (y - ay) * uy
            gap = max(0.0, zlo - t, t - zhi)
            toward = True                      # 진입 대기 중이면 다가오는 것으로 본다
            if on and gap > 0.0:
                a = np.radians(traci.person.getAngle(pid))   # SUMO 각도: 북=0, 시계방향
                along = np.sin(a) * ux + np.cos(a) * uy
                toward = (t < zlo and along > 0.0) or (t > zhi and along < 0.0)
            if best[0] is None or (gap, not toward) < (best[0], not best[1]):
                best = (gap, toward)
        cache[c["id"]] = best
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
            gap, toward = self._crossing_ped_state(c)
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
            "--seed", str(np.random.randint(0, 100000)),
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
        rng = np.random.default_rng()
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
        spans, start = [], 0.0
        edges = traci.route.getEdges("r0")
        for k, edge in enumerate(edges):
            length = traci.lane.getLength(f"{edge}_0")
            route_id = "r0" if k == 0 else f"r0_from{k}"
            if k > 0:
                traci.route.add(route_id, list(edges[k:]))
            spans.append((route_id, start, length))
            start += length
        for lane in range(self.num_lanes):
            f = float(factors[lane]) if lane < len(factors) else 1.0
            lane_spacing = self.prefill_spacing * f
            pos = self.prefill_start + rng.uniform(0, lane_spacing)
            while pos < self.road_length - self.prefill_end_margin:
                route_id, local = None, 0.0
                for rid, edge_start, length in spans:
                    if edge_start <= pos < edge_start + length:
                        local = pos - edge_start
                        if local <= length - self.prefill_end_margin:
                            route_id = rid
                        break
                vid = f"pre_{lane}_{count}"
                vtype = str(rng.choice(self._prefill_types, p=self._prefill_probs))
                try:
                    if route_id is None:
                        raise traci.TraCIException("엣지 끝 여유 구간")
                    first_edge = traci.route.getEdges(route_id)[0]
                    traci.vehicle.add(
                        vid, route_id, typeID=vtype,
                        departLane=str(lane + self._car_lanes(first_edge)[0]),
                        departPos=f"{local:.1f}",
                        departSpeed="max")
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
            t = np.random.uniform(0.0, sum(durations))
            for idx, d in enumerate(durations):
                if t < d:
                    break
                t -= d
            traci.trafficlight.setPhase(tl_id, idx)
            traci.trafficlight.setPhaseDuration(tl_id, durations[idx] - t)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # SUMO 상태를 완전히 초기화하는 가장 확실한 방법은 껐다 켜는 것.
        if self._started:
            traci.close()
            self._started = False
        self._start_sumo()
        self._randomize_signal_phase()

        # 도로 전 구간에 배경차를 미리 배치 (첫 스텝부터 실제 교통류 상태)
        if self.prefill_enabled:
            self._prefill_traffic()
        # 배치된 차들이 자리를 잡도록 몇 스텝만 정착시킨다.
        for _ in range(self.warmup_steps):
            traci.simulationStep()

        # ego 경로 선택 (EGO["routes"] 확률, 네트워크에 있는 route만)
        known = set(traci.route.getIDList())
        routes = {r: float(p) for r, p in self.ego_routes.items() if r in known} or {"r0": 1.0}
        probs = np.array(list(routes.values()))
        self.ego_route = str(np.random.choice(list(routes), p=probs / probs.sum()))

        # ego 차량 투입 — 반드시 "도로 시작점(0m)"에서 출발
        lane_base = self._car_lanes(traci.route.getEdges(self.ego_route)[0])[0]
        depart_lane = str(self.route_depart_lane.get(self.ego_route, self.ego_depart_lane))
        traci.vehicle.add(self.ego_id, self.ego_route, typeID="ego_type",
                          departLane=(str(int(depart_lane) + lane_base)
                                      if depart_lane.isdigit() else depart_lane),
                          departPos="0",
                          departSpeed=self.ego_depart_speed)
        # 편도(동행) route면 정류장 5곳에 정차. 정차 시간 15초 (TODO: 노션 조사값 들어오면 교체)
        if self.ego_route in ("r0", "r0_loop"):
            for bs_id in BUS_STOPS_ORDER:
                traci.vehicle.setBusStop(self.ego_id, bs_id, duration=15)

        # ★ 중요: speedMode=0 → SUMO의 내장 안전장치(자동 안전제동 등)를 전부 끔.
        #   이걸 꺼야 RL이 속도를 온전히 제어하고, 잘못 배우면 실제로 충돌한다.
        #   (켜두면 SUMO가 알아서 브레이크를 밟아줘서 충돌이 거의 안 나고,
        #    그러면 "충돌 회피"라는 학습 신호 자체가 사라진다)
        traci.vehicle.setSpeedMode(self.ego_id, 0)
        # SUMO의 자율 차선변경은 끄고, RL이 내리는 TraCI 차선변경 명령만 사용한다.
        traci.vehicle.setLaneChangeMode(self.ego_id, 0)

        traci.simulationStep()  # ego가 실제로 도로에 나타나도록 1스텝 진행

        # ── ego 삽입 보증 ──
        # 도로 입구는 flow가 차를 스폰하는 자리라, 시작 차선 입구가 잠시
        # 막혀 있으면 SUMO가 삽입을 "지연"시킨다. 그 경우 ego 없이
        # 에피소드가 진행되어 즉시 종료되는 버그가 되므로, 삽입될 때까지
        # (최대 30스텝) 기다린다. 지연되어도 요청한 차선은 유지된다.
        _wait = 0
        while self.ego_id not in traci.vehicle.getIDList() and _wait < 30:
            traci.simulationStep()
            _wait += 1
        if self.ego_id not in traci.vehicle.getIDList():
            raise RuntimeError(
                "ego 삽입 실패: 도로 입구가 30스텝 이상 막혀 있습니다. "
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
        """오프셋 차선의 '끊김 정도'를 반환.

        -1 : 그 차선이 존재하지 않음
         1 : visibility 이내에서 계속 이어짐
        0~1: (차선이 끝나는 지점까지의 거리 / visibility)
             → 값이 작을수록 곧 차선이 사라진다는 경고

        일반화 원리: 차선의 남은 길이와, 차선 끝에서 다음 edge로의
        연결(link) 존재 여부를 SUMO 네트워크에서 직접 조회한다.
        차선 감소/합류 지점은 "링크 없는 차선의 끝"으로 자연히 감지된다.
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
            if self._lane_leads_on_route(edge, lane_index):
                return 1.0                    # edge가 끝나도 route의 다음 엣지로 연결됨
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

    def _next_signal(self):
        """ego 경로상 다음 신호를 (신호 id, 정지선까지 거리 m, 신호색, 남은 시간 s)로 반환.

        traci getNextTLS는 ego가 앞으로 지날 신호들을 가까운 순으로 돌려준다
        (거리는 ego 앞범퍼 → 정지선). 신호를 이미 지났거나 ego가 없으면 None.
        """
        if self.ego_id not in traci.vehicle.getIDList():
            return None
        try:
            upcoming = traci.vehicle.getNextTLS(self.ego_id)
            if not upcoming:
                return None
            tl_id, link, dist, state = upcoming[0]
            remaining = (traci.trafficlight.getNextSwitch(tl_id)
                         - traci.simulation.getTime())
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
            return tl_id, float(dist), state, float(max(remaining, 0.0))
        except traci.TraCIException:
            return None

    def _signal_obs(self):
        """관측 [31..33]: 정지선 거리 / 정지 필요도 / 남은 시간 (mdp_config 참고)."""
        nxt = self._next_signal()
        if nxt is None or nxt[1] > self.signal_visibility:
            return [1.0, 0.0, 1.0]                 # 신호 없음/멀리: 제약 없음
        tl_id, dist, state, remaining = nxt
        level = self._SIGNAL_STOP_LEVEL.get(state, 1.0)
        # 일시정지(s): 정지선 앞에서 멈추기 전엔 적색, 멈춘 뒤엔 녹색으로 보인다.
        # (0.75 하나로만 보여주면 대기열에서 멈춘 것과 구분하지 못해 그냥 통과했다)
        if state == "s":
            level = 0.0 if self._stopped_at == tl_id else 1.0
        return [float(np.clip(dist / self.signal_visibility, 0.0, 1.0)),
                level,
                float(np.clip(remaining / self.signal_time_scale, 0.0, 1.0))]

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

        # ----- 전방 신호 (OBSERVATION["signal"] 사용 시) -----
        if self.signal_obs:
            obs += self._signal_obs()

        # ----- 경로상 횡단보도 (OBSERVATION["pedestrian"] 사용 시) -----
        if self.ped_obs:
            obs += self._ped_obs()

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
            lane_id = f"{edge}_{li}"
            my_pos = traci.vehicle.getLanePosition(self.ego_id)
            front, rear = None, None
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                d = traci.vehicle.getLanePosition(vid) - my_pos
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

    def _lane_change_is_safe(self, direction: int) -> bool:
        """목표 차선으로 지금 이동해도 측면충돌이 나지 않는지 검사한다.

        3중 안전 규칙:
          (0) 입구 보호구간: ego가 도로 시작 spawn_protect(m) 이내면 변경 금지.
              flow가 입구(0m)에서 차를 새로 만드는데, 생성 "직전" 차량은
              어떤 API로도 미리 볼 수 없어 원천적으로 사각지대이기 때문.
          (1) 앞: gap > min_front_gap + front_tau × 내속도
              + 접근 여유: gap > (내속도 − 앞차속도)⁺ × 1s   (내가 더 빠를 때)
          (2) 뒤: gap > min_rear_gap + rear_tau × 뒤차속도
              + 접근 여유: gap > (뒤차속도 − 내속도)⁺ × 1s   (뒤가 더 빠를 때)
        이웃은 getNeighbors 결과와 차선 직접 스캔 결과 중 "더 가까운 쪽"을
        쓴다 (스폰 직후 차량 사각지대 보완).
        """
        my_pos = traci.vehicle.getLanePosition(self.ego_id)
        if my_pos < self.lc_spawn_protect:
            return False
        # 신호 정지선 앞 구간: 차선변경 금지 (교차로 앞 실선).
        # 엣지 경계 너머(교차로 건너편) 차량은 차선 스캔에 안 잡히므로
        # 사각지대 보완 효과도 있다. 교차로 직후 구간은 위의 spawn_protect가
        # 엣지 내 위치 기준이라 이미 막혀 있다.
        nxt = self._next_signal()
        if nxt is not None and nxt[1] < self.lc_signal_zone:
            return False

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

    def _apply_lane_change(self, lane_change: int) -> bool:
        """가능하고 "안전한" 경우에만 차선변경을 적용하고 성공 여부를 반환.

        안전 게이트(_lane_change_is_safe)를 통과하지 못한 명령은 실행하지
        않고 False를 반환한다 → step()에서 "불가능한 명령"과 동일하게
        invalid_action_penalty가 부과된다. 즉 에이전트 입장에서
        "위험한 변경 = 실행도 안 되고 감점"이라, 측면충돌로 즉사하며
        배우는 대신 안전한 타이밍을 고르는 쪽으로 학습이 유도된다.
        """
        if self.ego_id not in traci.vehicle.getIDList():
            return False
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                return False
            current = traci.vehicle.getLaneIndex(self.ego_id)
            target = current + lane_change
            if not self._is_car_lane(edge, target):
                return False
            # route로 이어지지 않는 차선(좌회전 포켓 등)으로는 들어가지 않는다
            if lane_change != 0 and not self._lane_leads_on_route(edge, target):
                return False
            # ★ 측면충돌 방지: 실제 이동 명령(0이 아닌 변경)에만 안전 검사
            if (lane_change != 0 and self.lc_safety_enabled
                    and not self._lane_change_is_safe(lane_change)):
                return False
            traci.vehicle.changeLane(
                self.ego_id, target, self.lane_change_duration)
            return True
        except traci.TraCIException:
            return False

    def _signal_states(self) -> list:
        """도로 위 모든 신호의 상태 (privileged state용)."""
        out = []
        now = traci.simulation.getTime()
        for tl_id in traci.trafficlight.getIDList():
            out.append({
                "id": tl_id,
                "state": traci.trafficlight.getRedYellowGreenState(tl_id),
                "phase": int(traci.trafficlight.getPhase(tl_id)),
                "time_to_switch": float(traci.trafficlight.getNextSwitch(tl_id) - now),
            })
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

    def step(self, action):
        # action = [가감속 raw, 차선변경 raw]
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size < self.act_dim:
            raise ValueError(
                f"action은 {self.act_dim}차원이어야 합니다. 받은 shape={np.asarray(action).shape}")

        accel_raw = float(np.clip(action_arr[self.accel_index], -1.0, 1.0))
        lane_raw = float(np.clip(action_arr[self.lane_change_index], -1.0, 1.0))
        lane_change = self._quantize_lane_change(lane_raw)
        lane_change_applied = False
        filter_active = False
        lane_before = None
        odo_before = None

        # 행동 결정 시점의 전방 신호 (신호 위반 판정 기준 — 정책이 본 관측과 같은 시점)
        signal_before = self._next_signal()

        # ----- (1) 종방향 + 횡방향 행동 적용 -----
        if self.ego_id in traci.vehicle.getIDList():
            v = traci.vehicle.getSpeed(self.ego_id)
            lane_before = self._car_lane_index(self.ego_id)
            odo_before = traci.vehicle.getDistance(self.ego_id)

            # 보행자 안전 필터: 부딪힐 것 같으면 PPO 명령을 제동으로 바꾼다
            if self.ped_filter is not None:
                accel_raw, filter_active = self.ped_filter.apply(accel_raw, self.ego_id)

            accel_cmd = (accel_raw * self.max_accel
                         if accel_raw >= 0.0 else accel_raw * self.max_decel)
            dv = accel_cmd * self.step_length
            traci.vehicle.setSpeed(
                self.ego_id, float(np.clip(v + dv, 0.0, self.vmax)))

            lane_change_applied = self._apply_lane_change(lane_change)

        # ----- (2) 시뮬레이션 1스텝 -----
        traci.simulationStep()
        self.step_count += 1

        # 어린이보호구역(3차로) 가운데 차로 유지: RL의 차선변경 명령과 별개로,
        # 이 구간에 있으면 매 스텝 가운데 차로로 변경을 "권고"한다
        # (이미 그 차로면 SUMO가 조용히 무시하므로 매 스텝 호출해도 안전하다).
        if self.ego_id in traci.vehicle.getIDList():
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge in SCHOOL_ZONE_CENTER_LANE_EDGES:
                lo, hi = self._car_lanes(edge)
                if hi - lo >= 3:
                    try:
                        traci.vehicle.changeLane(self.ego_id, lo + 1, 10.0)
                    except traci.TraCIException:
                        pass

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
        approach_pen, stop_pen = 0.0, 0.0   # 진단용 (info에 기록)

        # ---- 신호 위반: 결정 시점에 적색이던 신호의 정지선을 이번 스텝에 넘음 ----
        # 넘었는지는 "다음 신호 목록에서 그 신호가 사라졌는지"로 본다.
        # 황색 진입은 위반으로 보지 않는다. (ego는 speedMode=0이라 SUMO가
        # 대신 멈춰주지 않으므로, 이 판정과 보상으로 신호 준수를 학습한다)
        # 일시정지(s)는 정지선 앞 _STOP_ZONE 안에서 한 번 멈췄으면 지나가도 된다.
        signal_after = self._next_signal() if alive else None
        passed = bool(alive and signal_before is not None
                      and (signal_after is None or signal_after[0] != signal_before[0]))
        before_level = (self._SIGNAL_STOP_LEVEL.get(signal_before[2], 1.0)
                        if signal_before is not None else 0.0)
        red_violation = passed and (
            before_level >= 1.0
            or (signal_before[2] == "s" and self._stopped_at != signal_before[0]))
        if alive and signal_after is not None:
            if (signal_after[1] <= self._STOP_ZONE
                    and traci.vehicle.getSpeed(self.ego_id) < 0.1):
                self._stopped_at = signal_after[0]
        # 일시정지 완료가 이번 스텝 관측에 바로 반영되도록 _stopped_at 갱신 뒤에 만든다
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
                if not ped_ahead:
                    if s_done and free > stop_gap:
                        stop_pen = gsp
                    elif free > stop_gap:
                        ramp = float(self.rw.get("green_stop_ramp", 30.0))
                        extra = float(self.rw.get("green_stop_extra", 0.0))
                        stop_pen = gsp + extra * min((free - stop_gap) / ramp, 1.0)
                    reward -= stop_pen

        # 양보 위반은 종료하지 않고 감점만 (충돌·완주와 같은 스텝이어도 적용)
        if yield_violation:
            reward -= float(self.rw.get("yield_violation_penalty", 0.0))

        if self.step_count >= self.max_steps:
            truncated = True
            if not terminated:
                reward -= float(self.rw.get("timeout_penalty", 0.0))

        info = {
            "collided": collided,
            "arrived": arrived,
            "speed": ego_speed,
            "gap": ego_gap,
            "accel_raw": accel_raw,          # 실제 적용 값 (필터가 바꿨으면 바뀐 값)
            "filter_active": filter_active,
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
            # 결정 시점의 전방 신호 (없으면 None)
            "signal_state": signal_before[2] if signal_before else None,
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
            limit = self._school_limits.get(edge)
            if limit is not None and spd > limit:
                rate = float(self.rw.get("school_zone_speeding_penalty", 0.0))
                reward -= min(rate * (spd - limit), 0.5)
        return obs, float(reward), terminated, truncated, info

    # ==================================================================
    def close(self):
        """환경 정리. 학습이 끝나면 반드시 호출해서 SUMO 프로세스를 종료할 것."""
        if self._started:
            traci.close()
            self._started = False
