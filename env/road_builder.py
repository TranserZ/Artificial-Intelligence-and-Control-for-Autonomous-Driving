"""configs/road_config.py 를 읽어 SUMO 도로 파일들을 자동 생성하는 모듈.

SUMO는 도로를 여러 XML 파일로 정의한다:
    highway.nod.xml  : 노드(교차점/도로 끝점)의 좌표
    highway.edg.xml  : 엣지(도로 구간) — 어느 노드에서 어느 노드로, 차선 수, 제한속도
    highway.net.xml  : 위 두 파일을 netconvert 도구로 컴파일한 "완성된 도로망"
                       (SUMO가 실제로 읽는 파일. 직접 작성하기엔 복잡해서 도구로 생성)
    highway.rou.xml  : 차량 타입(vType)과 교통 수요(flow/route) 정의
    highway.sumocfg  : 위 파일들을 묶어주는 설정 파일 (시뮬레이션 실행의 진입점)

train.py / test.py 가 시작할 때마다 build()를 호출하므로,
road_config.py만 수정하면 도로가 자동으로 다시 만들어진다.
"""
import os
import subprocess
import sys

from sumolib import checkBinary  # SUMO 실행파일(sumo, netconvert 등) 경로를 찾아주는 유틸
from env.mdp_config import SIMULATION
try:
    from env.road_config import SCENERY
except ImportError:      # 구버전 설정 하위호환
    SCENERY = {"trees": False}
try:
    from env.road_config import TRAFFIC_LIGHT
except ImportError:      # 신호등 설정이 없는 구버전 설정 하위호환
    TRAFFIC_LIGHT = {"enabled": False}
try:
    from env.road_config import INTERSECTION, PEDESTRIANS
except ImportError:      # 교차로 설정이 없는 구버전 설정 하위호환
    INTERSECTION, PEDESTRIANS = {}, {"enabled": False}
try:
    from env.road_config import SIGNAL_PLAN
except ImportError:      # 실제 교차로 신호 계획이 없는 구버전 설정 하위호환
    SIGNAL_PLAN = {}
try:
    from env.road_config import SCHOOL
except ImportError:      # 학교/어린이 보호구역 설정이 없는 구버전 설정 하위호환
    SCHOOL = {"enabled": False}

TL_ID = "n_tl"   # 신호 노드 id = SUMO 신호(tlLogic) id. sumo_env.py도 이 값을 쓴다.
LANE_WIDTH = 3.2  # SUMO 기본 차로 폭 (m)


def tl_enabled(tl: dict = None) -> bool:
    tl = TRAFFIC_LIGHT if tl is None else tl
    return bool(tl and tl.get("enabled", False))


def is_intersection(road: dict) -> bool:
    return road.get("layout", "straight") == "intersection"


def is_cheonggyecheon(road: dict) -> bool:
    return road.get("layout", "straight") == "cheonggyecheon"


def _build_cheonggyecheon(ego: dict, out_dir: str) -> str:
    """청계천로(OSM 실제 도로) 시나리오를 out_dir에 만들고 sumocfg 경로를 반환한다.

    1) data/osm/cheonggyecheon.osm → cgc.nod/edg.xml (tools/osm_to_sumo.py, PROJ 불필요)
    2) netconvert → cgc.net.xml (보도·횡단보도 자동 생성, 좌표 평행이동 끔)
    3) 경로·교통·보행자·신호 → cgc.rou.xml (tools/make_cgc_scenario.py)
       버스(ego) 경로는 "r0" — sumo_env가 ego 투입·배경차 prefill·경로상 횡단보도 탐색에 쓴다.
       데모 셔틀은 끈다 (ego는 sumo_env가 TraCI로 직접 투입).
    근거·가정: docs/cheonggyecheon_research.md
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tools = os.path.join(root, "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import osm_to_sumo
    import make_cgc_scenario

    osm_to_sumo.convert(os.path.join(root, "data", "osm", "cheonggyecheon.osm"), out_dir)
    netconvert_cmd = (
        [checkBinary("netconvert"),
         "--node-files", os.path.join(out_dir, "cgc.nod.xml"),
         "--edge-files", os.path.join(out_dir, "cgc.edg.xml"),
         "-o", os.path.join(out_dir, "cgc.net.xml"),
         "--no-turnarounds", "true", "--junctions.join", "true", "--tls.join", "true", "--tls.join-dist", "30",
         "--junctions.corner-detail", "5", "--output.street-names", "true",
         "--offset.disable-normalization", "true",
         "--sidewalks.guess", "true", "--sidewalks.guess.max-speed", "14",
         "--crossings.guess", "true", "--crossings.guess.speed-threshold", "14"])
    subprocess.run(netconvert_cmd, check=True, capture_output=True)
    # 겹친 교차로 사이에 남은 아주 짧은 엣지는 교차로를 합쳐 다시 만든다
    if osm_to_sumo.tiny_edge_joins(os.path.join(out_dir, "cgc.net.xml"),
                                   os.path.join(out_dir, "cgc.nod.xml")):
        subprocess.run(netconvert_cmd, check=True, capture_output=True)
    args = make_cgc_scenario.parse_args(
        ["--net", os.path.join(out_dir, "cgc.net.xml"), "--out", out_dir,
         "--route-id", "r0", "--no-demo-ego",
         "--ego-accel", str(ego["accel"]), "--ego-decel", str(ego["decel"]),
         "--ego-max-speed", str(ego["max_speed"])])
    make_cgc_scenario.build(args.net, out_dir, args)
    # 버스 경로 위 오른쪽 차로가 직진으로 안 이어지는 교차로 보정 (도로망만 다시 쓰고, 경로는 그대로)
    # r0_loop의 청계5가·청계광장 좌회전은 오른쪽 승용차 차로(lane 1)에 따로 붙인다.
    net_path = os.path.join(out_dir, "cgc.net.xml")
    rou_path = os.path.join(out_dir, "cgc.rou.xml")
    con_path = os.path.join(out_dir, "cgc.fix.con.xml")
    n_fix = osm_to_sumo.route_straight_connections(net_path, rou_path, "r0", con_path)
    n_fix += osm_to_sumo.route_straight_connections(
        net_path, rou_path, "r0_loop", con_path, append=True)
    n_fix += osm_to_sumo.loop_left_connections(net_path, con_path, append=True)
    if n_fix:
        subprocess.run([checkBinary("netconvert"), "--sumo-net-file", net_path,
                        "--connection-files", con_path, "-o", net_path,
                        "--offset.disable-normalization", "true",
                        "--tls.rebuild", "true"],
                       check=True, capture_output=True)
    return os.path.join(out_dir, "cgc.sumocfg")


def _left_bay(road: dict) -> float:
    """교차로 앞 좌회전 전용차로(포켓) 길이 m. 0이면 포켓 없음 (좌회전은 맨 왼쪽 직진 차로 공용)."""
    return float(INTERSECTION.get("left_bay_length", 0.0)) if is_intersection(road) else 0.0


def school_layout(road: dict):
    """어린이 보호구역 배치 (본선 x 좌표, m). 학교가 꺼져 있으면 None.

    lo/hi   : 보호구역 시작/끝 (정문 중심 반경 원 ∩ 본선, 하한·포켓 앞으로 자름)
    gate    : 정문 = 횡단보도 중심
    zw / ze : 동쪽/서쪽 방향 지그재그 구간이 시작하는 노드 (횡단보도에서 zigzag + 여유만큼)
    """
    import math
    if not (SCHOOL.get("enabled", False) and is_intersection(road)):
        return None
    bay = _left_bay(road)
    median = LANE_WIDTH if bay else 0.0
    curb = median / 2 + LANE_WIDTH * int(road["num_lanes"]) + 2.0   # 남쪽 보도 바깥 (|y|)
    gate = float(SCHOOL["gate_x"])
    radius = float(SCHOOL.get("radius", 200.0))
    gate_y = curb + 1.0                                               # 정문은 보도 바로 뒤
    if radius <= gate_y:
        raise ValueError(f"SCHOOL radius {radius} m가 너무 작습니다.")
    half = math.sqrt(radius ** 2 - gate_y ** 2)
    pos = float(INTERSECTION.get("position", float(road["length"]) / 2))
    lo = max(gate - half, float(SCHOOL.get("zone_min_x", 50.0)))
    hi = min(gate + half, pos - bay - float(SCHOOL.get("zone_bay_margin", 20.0)))
    if (lo, hi) != (gate - half, gate + half):
        print(f"[school] 보호구역 반경 {radius:g} m 범위 {gate - half:.0f}~{gate + half:.0f} m를 "
              f"{lo:.0f}~{hi:.0f} m로 자름 (시작 하한·좌회전 포켓)")
    cw = float(SCHOOL.get("crosswalk_width", 4.0))
    zz = float(SCHOOL.get("zigzag_length", 20.0))
    reach = cw / 2 + 2.0 + zz        # 횡단보도 중심 → 지그재그 시작 (정지선 여유 2 m)
    zw, ze = gate - reach, gate + reach
    if not (lo + 30.0 < zw and ze < hi - 30.0):
        raise ValueError(f"보호구역({lo:.0f}~{hi:.0f} m)이 정문({gate:g} m) 앞뒤 지그재그 구간을 "
                         f"담기에 너무 짧습니다. gate_x / radius를 조정하세요.")
    return {"lo": lo, "hi": hi, "gate": gate, "zw": zw, "ze": ze, "curb": curb,
            "limit": float(SCHOOL.get("speed_limit", 8.33)), "cw": cw, "zz": zz}


def main_chains(road: dict):
    """본선 (동쪽 진입 엣지들, 서쪽 진출 엣지들). 보호구역이 있으면 구간마다 엣지를 나눈다.

    동쪽: n0 → (포켓 노드 또는 교차로), 서쪽: 교차로 → n0. 포켓 엣지(e0b)는 포함하지 않는다.
    """
    if school_layout(road) is None:
        return ["e0"], ["e0_r"]
    return (["e0", "e0_sz1", "e0_szz", "e0_sz2", "e0_sz3", "e0_a"],
            ["e0_r", "e0_r_sz3", "e0_r_szz", "e0_r_sz2", "e0_r_sz1", "e0_r_out"])


def route_edges(road: dict, tl: dict = None) -> list:
    """ego/배경차 route r0이 지나가는 엣지 순서."""
    if is_intersection(road):
        east = main_chains(road)[0]
        return east + (["e0b", "e1"] if _left_bay(road) > 0 else ["e1"])
    if tl_enabled(tl):
        return ["e0", "e1"]
    return ["e0"]


def _custom_program(tl: dict = None) -> bool:
    """True면 highway.tll.xml로 신호를 덮어쓰고 ("custom"/"plan"), False면 netconvert 기본 신호."""
    tl = TRAFFIC_LIGHT if tl is None else tl
    return tl_enabled(tl) and tl.get("program", "custom") in ("custom", "plan")


def build(road: dict, ego: dict, traffic: dict, out_dir: str) -> str:
    """SUMO 파일들을 out_dir에 생성하고, sumocfg 파일 경로를 반환한다.

    Args:
        road:    configs.road_config.ROAD    (도로 형상)
        ego:     configs.road_config.EGO     (RL 차량 속성)
        traffic: configs.road_config.TRAFFIC (배경 교통류)
        out_dir: 생성 위치 (보통 프로젝트의 sumo/ 폴더)
    신호등 설정은 configs.road_config.TRAFFIC_LIGHT 에서 읽는다.
    """
    os.makedirs(out_dir, exist_ok=True)
    if is_cheonggyecheon(road):
        return _build_cheonggyecheon(ego, out_dir)
    use_tl = tl_enabled()
    cross = is_intersection(road)
    if cross:
        _write_intersection_geometry(road, out_dir, use_tl)
    elif use_tl:
        tl_pos = float(TRAFFIC_LIGHT["position"])
        if not 0.0 < tl_pos < float(road["length"]):
            raise ValueError(
                f"TRAFFIC_LIGHT['position']={tl_pos}는 0과 도로 길이"
                f"({road['length']}) 사이여야 합니다.")

    # ──────────────────────────────────────────────────────
    # 1) 노드 파일: 직선 도로의 양 끝점 2개
    #    n0 = 원점(출발), n1 = length만큼 떨어진 지점(도착)
    #    신호등 사용 시 position 지점에 신호 노드 n_tl 추가.
    #    type="traffic_light" 노드에 netconvert가 신호(tlLogic)를 만든다.
    # ──────────────────────────────────────────────────────
    if not cross:
        tl_node = (f'    <node id="{TL_ID}" x="{tl_pos}" y="0" type="traffic_light"/>\n'
                   if use_tl else "")
        with open(os.path.join(out_dir, "highway.nod.xml"), "w") as f:
            f.write(f"""<nodes>
    <node id="n0" x="0" y="0"/>
{tl_node}    <node id="n1" x="{road['length']}" y="0"/>
</nodes>
""")

    # ──────────────────────────────────────────────────────
    # 2) 엣지 파일: n0 → n1 을 잇는 도로 구간 하나
    #    numLanes = 차선 수, speed = 제한속도(m/s)
    #    신호등 사용 시 n0 → n_tl (e0) / n_tl → n1 (e1) 두 구간으로 분할
    # ──────────────────────────────────────────────────────
    if not cross:
        lane_attrs = f'numLanes="{road["num_lanes"]}" speed="{road["speed_limit"]}"'
        if use_tl:
            edge_lines = (f'    <edge id="e0" from="n0" to="{TL_ID}" {lane_attrs}/>\n'
                          f'    <edge id="e1" from="{TL_ID}" to="n1" {lane_attrs}/>\n')
        else:
            edge_lines = f'    <edge id="e0" from="n0" to="n1" {lane_attrs}/>\n'
        with open(os.path.join(out_dir, "highway.edg.xml"), "w") as f:
            f.write(f"""<edges>
{edge_lines}</edges>
""")

    # ──────────────────────────────────────────────────────
    # 3) 라우트 파일: 차량 타입 2종 + 배경 교통 flow
    #
    #    <vType>  : 차량의 물리 속성 템플릿
    #       ego_type : RL이 조종할 차 (빨간색, color="1,0,0" = RGB)
    #    <vTypeDistribution> : 배경차 컨트롤러 믹스.
    #       TRAFFIC["controllers"]에 정의된 여러 차량추종모델(Krauss/IDM/
    #       EIDM/ACC 등)을 probability 비율로 섞는다. flow가 이 분포를
    #       type으로 쓰면, 투입되는 차량마다 SUMO가 확률적으로 타입을 뽑는다.
    #    <route>  : 지나갈 엣지들의 나열. 여기선 e0 하나뿐.
    #    <flow>   : 교통 수요 정의. period="X"는 "X초마다 정확히 1대씩"
    #               → 등간격(uniform) 생성. (vehsPerHour와 등가지만
    #               "비슷한 간격으로 새로 생성"을 명시적으로 보장)
    #               ego는 여기에 없고, 환경(sumo_env.py)이 reset마다
    #               traci.vehicle.add()로 직접 투입한다.
    # ──────────────────────────────────────────────────────
    controllers = traffic.get("controllers")
    if controllers:
        # probability 합이 1이 아니어도 되도록 자동 정규화
        total_p = sum(float(c.get("probability", 1.0))
                      for c in controllers.values())
        vtype_lines, free_lines = [], []
        for name, c in controllers.items():
            prob = float(c.get("probability", 1.0)) / total_p
            attrs = [
                f'id="car_{name}"',
                f'probability="{prob:.4f}"',
                f'carFollowModel="{c["carFollowModel"]}"',
                f'accel="{c.get("accel", 2.5)}"',
                f'decel="{c.get("decel", 4.5)}"',
                f'tau="{c.get("tau", 1.0)}"',
                f'minGap="{c.get("min_gap", 2.5)}"',
                # 컨트롤러별 희망속도 오버라이드 (없으면 공통 max_speed).
                # 컨트롤러마다 속도를 다르게 주면 차선 간 속도차가 생겨
                # "추월할 이유"가 자연스럽게 만들어진다.
                f'maxSpeed="{c.get("max_speed", traffic["max_speed"])}"',
                # speedDev: 같은 타입 안에서도 개체별 희망속도 편차 (0~)
                f'speedDev="{c.get("speed_dev", 0.1)}"',
                f'color="{c.get("color", "1,1,0")}"',
                'vClass="passenger"', 'guiShape="passenger/sedan"',
            ]
            # sigma는 Krauss 계열에만 의미가 있으므로 지정된 경우에만 넣는다
            if "sigma" in c:
                attrs.append(f'sigma="{c["sigma"]}"')
            # ── 배경차 차선 유지 (keep_lane, 기본 True) ──
            # 배경차의 자율 차선변경(전략/협조/속도이득/우측유지)을 전부 꺼서
            # 각자 자기 차선을 지키게 한다. 이유:
            #   배경차가 자유롭게 돌아다니면 (측정: 150스텝에 477회 변경)
            #   차선 간 틈이 수시로 닫혀서, ego가 틈을 노려 파고드는
            #   "칼치기 추월"을 학습할 안정적인 기회 자체가 사라진다.
            #   배경차가 차선을 유지하면 "느린 차 뒤 → 옆 틈으로 변경 →
            #   가속"이라는 추월 패턴이 재현 가능한 형태로 나타난다.
            # 특정 컨트롤러만 돌아다니게 하려면 그 항목에 "keep_lane": False.
            # 교차로의 회전 차량용 분포(carDistFree)는 차선을 바꿔야 회전
            # 차로로 갈 수 있으므로 차선 유지 옵션 없이 따로 만든다.
            free_lines.append("        <vType " + " ".join(attrs).replace(
                f'id="car_{name}"', f'id="free_{name}"') + "/>")
            if c.get("keep_lane", True):
                attrs += ['lcStrategic="0"', 'lcCooperative="0"',
                          'lcSpeedGain="0"', 'lcKeepRight="0"']
            vtype_lines.append("        <vType " + " ".join(attrs) + "/>")
        vtypes_block = ('    <vTypeDistribution id="carDist">\n'
                        + "\n".join(vtype_lines)
                        + "\n    </vTypeDistribution>")
        if cross:
            vtypes_block += ('\n    <vTypeDistribution id="carDistFree">\n'
                             + "\n".join(free_lines)
                             + "\n    </vTypeDistribution>")
        flow_type = "carDist"
    else:
        # controllers 설정이 없으면 예전 방식(단일 Krauss)으로 동작 (하위호환)
        vtypes_block = (f'    <vType id="car" accel="2.5" decel="4.5" '
                        f'maxSpeed="{traffic["max_speed"]}" '
                        f'sigma="{traffic.get("sigma", 0.5)}" '
                        f'vClass="passenger" guiShape="passenger/sedan"/>')
        flow_type = "car"
        if cross:
            raise ValueError("교차로 layout은 TRAFFIC['controllers'] 설정이 필요합니다.")

    with open(os.path.join(out_dir, "highway.rou.xml"), "w") as f:
        f.write(f"""<routes>
    <vType id="ego_type" accel="{ego['accel']}" decel="{ego['decel']}" maxSpeed="{ego['max_speed']}" color="1,0,0" vClass="passenger" guiShape="passenger/sedan"/>
{vtypes_block}
    <route id="r0" edges="{' '.join(route_edges(road))}"/>
    <flow id="traffic" type="{flow_type}" route="r0" begin="0" end="100000"
          period="{3600.0 / traffic['vehs_per_hour']:.2f}"
          departLane="{traffic['depart_lane']}" departSpeed="{traffic['depart_speed']}"/>
{_intersection_demand(traffic, road) if cross else ""}</routes>
""")

    # ──────────────────────────────────────────────────────
    # 4) GUI 기본 시각화 설정
    #    GUI 관련 값의 source of truth는 env/mdp_config.py의 SIMULATION 하나뿐이다.
    #    view_road.py / train.py / test.py / 데이터 수집 모두 build()를 거치므로
    #    gui_zoom/gui_delay를 바꾸면 생성되는 highway.gui.xml에도 즉시 반영된다.
    # ──────────────────────────────────────────────────────
    # GUI 시작 화면: "도로 시작 구간"이 차선까지 보이게 자동 확대해둔다.
    #   (12km 전체 맞춤으로 시작하면 도로가 가느다란 선으로만 보여 직관성이
    #    떨어지므로, ego가 등장할 시작부를 확대해 대기 화면으로 삼는다.
    #    ▶ 재생 후에는 env가 ego를 추적(trackVehicle)하며 시점을 가져간다.)
    # 좌표/배율은 도로 길이·차선 수에서 "계산"하므로 설정을 바꿔도 안 어긋난다:
    #   zoom(%) = 도로길이 / 보고싶은폭(gui_view_width) × 100
    #     (SUMO zoom은 "네트워크 전체 = 100%" 기준)
    #   y 중심 = -차선폭(3.2m) × 차선수 / 2   (SUMO는 차선을 y<0 쪽으로 그림)
    gui_delay = float(SIMULATION.get("gui_delay", 100.0))
    view_width = float(SIMULATION.get("gui_view_width", 500.0))   # 시작 화면 시야(m)
    zoom = float(road["length"]) / view_width * 100.0
    center_x = view_width * 0.4                                   # 시작점이 화면 왼쪽에 오게
    center_y = -3.2 * int(road["num_lanes"]) / 2.0
    gui_settings_path = os.path.join(out_dir, "highway.gui.xml")
    with open(gui_settings_path, "w") as f:
        f.write(f"""<viewsettings>
    <scheme name="real world"/>
    <viewport zoom="{zoom:.0f}" x="{center_x:.1f}" y="{center_y:.1f}" angle="0"/>
    <delay value="{gui_delay}"/>
</viewsettings>
""")

    # ──────────────────────────────────────────────────────
    # 4.7) 도로변 장식(나무) — 순수 시각 요소
    #
    # SUMO의 <poly>(다각형)로 "초록 수관 + 갈색 줄기"를 그려 도로
    # 양옆에 심는다. additional 파일로 로드되며 차량 물리와 완전히
    # 무관하다 (충돌/관측/보상 어디에도 안 잡힘).
    # 좌표계: 차선은 y ∈ [0, -3.2×차선수] 영역 → 왼쪽 숲은 y > 0,
    # 오른쪽 숲은 y < -3.2×차선수 - side_offset 에 배치.
    # ──────────────────────────────────────────────────────
    _write_scenery(out_dir, road)

    # ──────────────────────────────────────────────────────
    # 5) sumocfg: "이 net 파일과 이 route 파일로 시뮬레이션해라"
    #    신호등 사용 시 highway.tll.xml(신호 주기)도 additional로 로드
    # ──────────────────────────────────────────────────────
    use_school = school_layout(road) is not None
    additional = ("highway.scenery.xml"
                  + (",highway.school.xml" if use_school else "")
                  + (",highway.tll.xml" if _custom_program() else ""))
    cfg_path = os.path.join(out_dir, "highway.sumocfg")
    with open(cfg_path, "w") as f:
        f.write(f"""<configuration>
    <input>
        <net-file value="highway.net.xml"/>
        <route-files value="highway.rou.xml"/>
        <additional-files value="{additional}"/>
    </input>
    <gui_only>
        <gui-settings-file value="highway.gui.xml"/>
    </gui_only>
</configuration>
""")

    # ──────────────────────────────────────────────────────
    # 6) netconvert 실행: nod + edg → net.xml 컴파일
    #    (SUMO 설치 시 함께 깔리는 커맨드라인 도구)
    # ──────────────────────────────────────────────────────
    netconvert = checkBinary("netconvert")
    subprocess.run(
        [netconvert,
         "--node-files", os.path.join(out_dir, "highway.nod.xml"),
         "--edge-files", os.path.join(out_dir, "highway.edg.xml"),
         "-o", os.path.join(out_dir, "highway.net.xml")]
        + (_intersection_netconvert_args(road, out_dir) if cross else []),
        check=True,           # 실패하면 예외를 던져서 바로 알 수 있게
        capture_output=True,  # 출력은 조용히 삼킴
    )

    # ──────────────────────────────────────────────────────
    # 7) 신호 주기 파일 (highway.tll.xml)
    #    netconvert가 만든 기본 신호 대신 TRAFFIC_LIGHT["phases"]의 고정 주기를
    #    쓴다. 신호 state 문자열 길이(= 신호가 제어하는 연결 수)는 net.xml에서
    #    읽어 맞추므로 차선 수를 바꿔도 어긋나지 않는다.
    #    같은 신호 id에 새 programID를 additional로 로드하면 SUMO는 마지막에
    #    로드된 프로그램을 사용한다.
    # ──────────────────────────────────────────────────────
    if _custom_program():
        if TRAFFIC_LIGHT.get("program") == "plan":
            if not cross:
                raise ValueError("TRAFFIC_LIGHT['program']='plan'은 교차로 layout 전용입니다.")
            _write_plan_tll(out_dir)
        else:
            _write_tll(out_dir, full_state=cross)
    elif os.path.exists(os.path.join(out_dir, "highway.tll.xml")):
        os.remove(os.path.join(out_dir, "highway.tll.xml"))   # 이전 custom 신호 잔재 제거

    # ──────────────────────────────────────────────────────
    # 8) 학교 + 어린이 보호구역 노면 (highway.school.xml)
    #    차로 좌표가 필요하므로 netconvert 뒤에 net.xml을 읽어 만든다.
    # ──────────────────────────────────────────────────────
    school_path = os.path.join(out_dir, "highway.school.xml")
    if use_school:
        from env.school_assets import make_assets
        layout = school_layout(road)
        make_assets(SCHOOL.get("name", "학교"), int(round(layout["limit"] * 3.6)),
                    SCHOOL.get("line_color", "255,200,0"))
        _write_school(out_dir, road, layout)
    elif os.path.exists(school_path):
        os.remove(school_path)
    return cfg_path


# ══════════════════════════════════════════════════════════════════
#  교차로 layout (ROAD["layout"] = "intersection")
#
#  본선(동서)과 교차 도로(남북)가 만나는 4지 교차로:
#
#                 n_n
#                  │  side_in / side_out
#   n0 ── e0 ──▶ n_tl ── e1 ──▶ n1      (ego: n0 → n1 직진, route r0 = e0 e1)
#        ◀─ e0_r ─    ◀─ e1_r ─         (반대 방향 본선)
#                  │
#                 n_s
#
#  보도/횡단보도는 netconvert가 자동 생성(--sidewalks.guess --crossings.guess)하고,
#  신호는 TRAFFIC_LIGHT["program"]에 따라
#    "default": netconvert 기본 프로그램 / "plan": SIGNAL_PLAN(실제 교차로 현시) /
#    "custom" : TRAFFIC_LIGHT["phases"]의 전체 state 문자열
#  을 쓴다.
# ══════════════════════════════════════════════════════════════════
def _write_intersection_geometry(road: dict, out_dir: str, use_tl: bool):
    length = float(road["length"])
    pos = float(INTERSECTION.get("position", length / 2))
    if not 0.0 < pos < length:
        raise ValueError(f"INTERSECTION['position']={pos}는 0과 도로 길이({length}) 사이여야 합니다.")
    arm = float(INTERSECTION.get("side_length", 150.0))
    bay = _left_bay(road)
    if bay and not (0 < bay < min(pos, length - pos, arm) - 10):
        raise ValueError(f"INTERSECTION['left_bay_length']={bay}는 교차로 팔 길이보다 짧아야 합니다.")
    node_type = "traffic_light" if use_tl else "priority"
    lanes = _approach_lanes(road)
    speed = {"main": road["speed_limit"],
             "side": INTERSECTION.get("side_speed", road["speed_limit"])}

    # 진출 엣지와 (포켓이 없을 때의) 진입 엣지
    nodes = [("n0", 0, 0), (TL_ID, pos, 0), ("n1", length, 0), ("n_n", pos, arm), ("n_s", pos, -arm)]
    edges = [("e1", TL_ID, "n1", lanes["e1"], "main"), ("e0_r", TL_ID, "n0", lanes["e0_r"], "main"),
             ("n_out", TL_ID, "n_n", lanes["n_out"], "side"), ("s_out", TL_ID, "n_s", lanes["s_out"], "side")]
    # 진입 엣지: 포켓이 있으면 교차로 앞 bay m 구간을 "<id>b" 엣지로 떼어내고
    # 그 구간에만 왼쪽에 좌회전 전용차로 1개를 더한다.
    approach = {"e0": ("n0", (pos - bay, 0)), "e1_r": ("n1", (pos + bay, 0)),
                "n_in": ("n_n", (pos, bay)), "s_in": ("n_s", (pos, -bay))}
    for eid, (start, (bx, by)) in approach.items():
        kind = "main" if eid in ("e0", "e1_r") else "side"
        if bay:
            nodes.append((f"{eid}_bay", bx, by))
            edges.append((eid, start, f"{eid}_bay", lanes[eid], kind))
            edges.append((f"{eid}b", f"{eid}_bay", TL_ID, lanes[eid] + 1, kind))
        else:
            edges.append((eid, start, TL_ID, lanes[eid], kind))

    # 어린이 보호구역: 본선 양방향을 시점·지그재그 시작·횡단보도(정문)·지그재그 시작·종점에서
    # 나눈다. 제한속도와 차로변경 금지는 엣지(차로) 단위로만 걸 수 있기 때문이다.
    #   동쪽 e0 → e0_sz1 → e0_szz(지그재그) → [횡단보도] → e0_sz2 → e0_sz3 → e0_a
    #   서쪽 e0_r → e0_r_sz3 → e0_r_szz(지그재그) → [횡단보도] → e0_r_sz2 → e0_r_sz1 → e0_r_out
    sz = school_layout(road)
    if sz:
        speed["zone"] = speed["zone_nolc"] = sz["limit"]
        cuts = [("sz_s", sz["lo"]), ("sz_zw", sz["zw"]), ("sz_c", sz["gate"]),
                ("sz_ze", sz["ze"]), ("sz_e", sz["hi"])]
        nodes += [(nid, x, 0) for nid, x in cuts]
        kinds = ["main", "zone", "zone_nolc", "zone", "zone", "main"]
        east, west = main_chains(road)

        def split(eid, chain, order):
            i = next(k for k, e in enumerate(edges) if e[0] == eid)
            _, a, b, n, _ = edges[i]
            seq = [a] + order + [b]
            edges[i:i + 1] = [(chain[k], seq[k], seq[k + 1], n, kinds[k]) for k in range(len(chain))]

        split("e0", east, [nid for nid, _ in cuts])
        split("e0_r", west, [nid for nid, _ in reversed(cuts)])

    # 포켓이 있으면 양방향 사이에 차로 1개 폭의 중앙분리대를 두고, 포켓 구간에서만
    # 그 자리를 좌회전 차로로 쓴다 (실제 도로처럼 직진 차로는 꺾이지 않고 곧게 이어짐).
    # SUMO는 차로를 엣지 중심선의 오른쪽에 쌓으므로
    #   일반 엣지: 진행 방향 오른쪽으로 median/2 이동 → 양방향 사이에 median 폭의 틈
    #   포켓 엣지: 진행 방향 왼쪽으로 median/2 이동 → 늘어난 차로 1개가 틈을 채움
    median = LANE_WIDTH if bay else 0.0
    xy = {nid: (x, y) for nid, x, y in nodes}

    def shape(eid, a, b):
        if not median:
            return ""
        (x1, y1), (x2, y2) = xy[a], xy[b]
        d = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        rx, ry = (y2 - y1) / d, -(x2 - x1) / d          # 진행 방향 오른쪽 단위벡터
        k = (-1 if eid.endswith("b") else 1) * median / 2
        return (f' shape="{x1 + k * rx:.2f},{y1 + k * ry:.2f} '
                f'{x2 + k * rx:.2f},{y2 + k * ry:.2f}"')

    with open(os.path.join(out_dir, "highway.nod.xml"), "w") as f:
        f.write("<nodes>\n" + "".join(
            f'    <node id="{nid}" x="{x}" y="{y}"'
            + (f' type="{node_type}"' if nid == TL_ID else "") + "/>\n"
            for nid, x, y in nodes) + "</nodes>\n")
    def edge_xml(eid, a, b, n, kind):
        head = (f'    <edge id="{eid}" from="{a}" to="{b}" numLanes="{n}" speed="{speed[kind]}"'
                f'{shape(eid, a, b)}')
        if kind != "zone_nolc":
            return head + "/>\n"
        # 지그재그 구간: 차로변경 금지 (긴급차량만 허용)
        return (head + ">\n" + "".join(
            f'        <lane index="{i}" changeLeft="emergency" changeRight="emergency"/>\n'
            for i in range(n)) + "    </edge>\n")

    with open(os.path.join(out_dir, "highway.edg.xml"), "w") as f:
        f.write("<edges>\n" + "".join(edge_xml(*e) for e in edges) + "</edges>\n")
    _write_intersection_connections(out_dir, lanes, bay, road)


def _approach_lanes(road: dict) -> dict:
    """엣지별 차로 수 (보도 제외, 포켓 구간 제외)."""
    main = int(road["num_lanes"])
    opp = int(INTERSECTION.get("opposite_lanes", main))
    side = int(INTERSECTION.get("side_lanes", 2))
    return {"e0": main, "e1": main, "e1_r": opp, "e0_r": opp,
            "n_in": side, "n_out": side, "s_in": side, "s_out": side}


# 교차로 진입 엣지 → (우회전, 직진, 좌회전) 진출 엣지
_TURNS = {"e0": ("s_out", "e1", "n_out"), "e1_r": ("n_out", "e0_r", "s_out"),
          "n_in": ("e0_r", "s_out", "e1"), "s_in": ("e1", "n_out", "e0_r")}


def _write_intersection_connections(out_dir: str, lanes: dict, bay: float, road: dict):
    """좌회전 포켓 차로 연결과 어린이 보호구역 횡단보도를 정의한다 (highway.con.xml).

    포켓 시작점: 직진 차로는 그대로 이어지고, 맨 왼쪽 차로에서 포켓으로 진입 가능.
    교차로     : 맨 오른쪽 차로 = 직진+우회전, 나머지 = 직진, 포켓 = 좌회전 전용.
    차로 번호는 보도(0번)를 뺀 번호이며, netconvert가 보도를 넣으면서 1씩 밀린다.
    보호구역   : 정문 노드(sz_c)에 본선 전체를 건너는 보행자 우선 횡단보도.
    """
    path = os.path.join(out_dir, "highway.con.xml")
    sz = school_layout(road)
    if not bay and not sz:
        if os.path.exists(path):
            os.remove(path)
        return
    lines = ["<connections>"]
    east, west = main_chains(road)
    if sz:
        # 보호구역이 있으면 횡단보도 자동 생성(--crossings.guess)을 끈다. 켜 두면 본선을 나눈
        # 모든 지점에 횡단보도가 생긴다. 대신 교차로 4개 + 정문 1개를 직접 정의한다.
        # 교차로 횡단보도 순서(c0 북, c1 동, c2 남, c3 서)는 자동 생성 때와 같다.
        if INTERSECTION.get("crosswalks", True):
            arm = lambda e: f"{e}b" if bay else e
            for out_e, in_e in (("n_out", arm("n_in")), ("e1", arm("e1_r")),
                                ("s_out", arm("s_in")), (west[0], "e0b" if bay else east[-1])):
                lines.append(f'    <crossing node="{TL_ID}" edges="{out_e} {in_e}"/>')
        # 정문 횡단보도 (보행자 우선, 신호 없음): 노드 서쪽에 붙은 양방향 엣지를 건넌다
        lines.append(f'    <crossing node="sz_c" edges="{east[2]} {west[3]}" '
                     f'priority="true" width="{sz["cw"]:g}"/>')
    for eid, (right, straight, left) in (_TURNS.items() if bay else ()):
        n = lanes[eid]
        src = east[-1] if eid == "e0" else eid      # 포켓 바로 앞 엣지
        for i in range(n):
            lines.append(f'    <connection from="{src}" to="{eid}b" fromLane="{i}" toLane="{i}"/>')
        lines.append(f'    <connection from="{src}" to="{eid}b" fromLane="{n - 1}" toLane="{n}"/>')
        out_n = lanes[straight]
        for i in range(n):
            lines.append(f'    <connection from="{eid}b" to="{straight}" fromLane="{i}" '
                         f'toLane="{min(i, out_n - 1)}"/>')
        lines.append(f'    <connection from="{eid}b" to="{right}" fromLane="0" toLane="0"/>')
        lines.append(f'    <connection from="{eid}b" to="{left}" fromLane="{n}" '
                     f'toLane="{lanes[left] - 1}"/>')
    lines.append("</connections>")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _intersection_netconvert_args(road: dict, out_dir: str) -> list:
    # 보도·횡단보도 자동 생성은 기본적으로 13.89 m/s(50 km/h) 이하 도로에만
    # 적용되므로, 기준을 이 도로의 최고 제한속도에 맞춰 둔다.
    vmax = max(float(road["speed_limit"]),
               float(INTERSECTION.get("side_speed", road["speed_limit"]))) + 0.1
    # 교차 도로가 y<0까지 내려가므로 netconvert가 좌표를 원점 기준으로 옮기는데(netOffset),
    # 그러면 원래 좌표로 그린 도로변 나무·GUI 시점이 어긋난다 → 좌표 이동을 끈다.
    args = ["--no-turnarounds", "--offset.disable-normalization", "true"]
    con = os.path.join(out_dir, "highway.con.xml")
    if os.path.exists(con):
        args += ["--connection-files", con]
    if INTERSECTION.get("crosswalks", True):
        # SUMO는 보도를 각 엣지의 차선 0번으로 넣는다 → 본선 차로는 1번부터.
        # (본선에 보도가 없으면 본선 횡단보도가 생성되지 않는다.)
        # env(sumo_env.py)는 차로 번호를 "보도를 뺀 번호"로 보정해서 쓴다.
        args += ["--sidewalks.guess", "--sidewalks.guess.max-speed", f"{vmax:.2f}"]
        if school_layout(road) is None:   # 보호구역이 있으면 횡단보도는 con.xml에 직접 정의
            args += ["--crossings.guess", "--crossings.guess.speed-threshold", f"{vmax:.2f}"]
    elif school_layout(road) is not None:
        raise ValueError("SCHOOL은 보도가 필요합니다: INTERSECTION['crosswalks']=True로 두세요.")
    return args


def _intersection_demand(traffic: dict, road: dict) -> str:
    """반대 방향 본선 / 본선 회전 / 교차 도로 차량 flow + 보행자 personFlow."""
    # 포켓이 있으면 진입 엣지 뒤에 포켓 엣지("<id>b")를 끼운다.
    # 본선은 보호구역 때문에 여러 엣지로 나뉠 수 있다 → 동쪽 진입 / 서쪽 진출 엣지 묶음.
    east, west = main_chains(road)
    b = {e: (f"{e} {e}b" if _left_bay(road) > 0 else e) for e in _TURNS}
    b["e0"] = " ".join(east) + (" e0b" if _left_bay(road) > 0 else "")
    w_out = " ".join(west)   # 교차로 → n0
    lines = []
    opp_vph = float(INTERSECTION.get("opposite_vehs_per_hour", 0))
    if opp_vph > 0:
        lines.append(f'    <flow id="opp" type="carDist" route="r_opp" begin="0" end="100000" '
                     f'period="{3600.0 / opp_vph:.2f}" departLane="random" '
                     f'departSpeed="{traffic["depart_speed"]}"/>')
        lines.insert(0, f'    <route id="r_opp" edges="{b["e1_r"]} {w_out}"/>')
    # 회전 차량은 차선변경 가능한 분포(carDistFree)로 포켓/우측 차로로 들어간다
    turn_routes = {}
    # 본선 좌/우회전 (ego 방향 + 반대 방향)
    if float(INTERSECTION.get("main_turn_vehs_per_hour", 0)) > 0:
        turn_routes["main"] = (float(INTERSECTION["main_turn_vehs_per_hour"]), {
            "er": f'{b["e0"]} s_out', "el": f'{b["e0"]} n_out',
            "wr": f'{b["e1_r"]} n_out', "wl": f'{b["e1_r"]} s_out'})
    # 교차 도로: 남북 직진 + 본선으로 좌/우회전
    if float(INTERSECTION.get("side_vehs_per_hour", 0)) > 0:
        turn_routes["side"] = (float(INTERSECTION["side_vehs_per_hour"]), {
            "ns": f'{b["n_in"]} s_out', "sn": f'{b["s_in"]} n_out',
            "ne": f'{b["n_in"]} e1', "nw": f'{b["n_in"]} {w_out}',
            "se": f'{b["s_in"]} e1', "sw": f'{b["s_in"]} {w_out}'})
    for group, (vph, routes) in turn_routes.items():
        per = 3600.0 / (vph / len(routes))
        for rid, edges in routes.items():
            lines.append(f'    <route id="r_{rid}" edges="{edges}"/>')
            lines.append(f'    <flow id="{group}_{rid}" type="carDistFree" route="r_{rid}" '
                         f'begin="0" end="100000" period="{per:.2f}" departLane="best" '
                         f'departSpeed="{traffic["depart_speed"]}"/>')
    # ego 우회전 경로 (서→남). sumo_env가 EGO["routes"] 확률로 r0과 섞어 고른다
    lines.append(f'    <route id="r_ego_right" edges="{b["e0"]} s_out"/>')
    # 보행자: 남북 보행자는 본선 횡단보도(서·동), 동서 보행자는 교차 도로
    # 횡단보도(북·남)를 건넌다. 동서는 포켓 엣지에서 출발해 교차로 근처에서만 걷는다
    if PEDESTRIANS.get("enabled", False):
        lines.append(f'    <vType id="ped" vClass="pedestrian" '
                     f'speed="{PEDESTRIANS.get("speed", 1.3)}" color="0,0.4,1"/>')
        e0_last = b["e0"].split()[-1]
        e1r_last = b["e1_r"].split()[-1]
        groups = [(float(PEDESTRIANS.get("persons_per_hour", 300)),
                   {"ns": ("n_in", "s_out"), "sn": ("s_in", "n_out")}),
                  (float(PEDESTRIANS.get("ew_persons_per_hour", 0)),
                   {"we": (e0_last, "e1"), "ew": (e1r_last, "e0_r")})]
        for pph, walks in groups:
            if pph <= 0:
                continue
            for pid, (frm, to) in walks.items():
                lines.append(f'    <personFlow id="ped_{pid}" type="ped" begin="0" end="100000" '
                             f'period="{3600.0 / (pph / 2):.2f}" departPos="random">\n'
                             f'        <walk from="{frm}" to="{to}" arrivalPos="random"/>\n'
                             f'    </personFlow>')
    # 어린이: 북쪽 보도(서쪽 방향 엣지의 보도)의 동네 ↔ 남쪽 정문. 정문 앞 노란 횡단보도를 건넌다.
    #   등교: 북쪽 보도 양옆(정문 서쪽·동쪽 구간)에서 출발해 정문(e0_sz2 시작 = 횡단보도 바로 옆) 도착
    #   하교: 정문에서 출발해 북쪽 보도 양옆으로
    cph = float(SCHOOL.get("children_per_hour", 0)) if school_layout(road) else 0.0
    if cph > 0:
        lines.append(f'    <vType id="child" vClass="pedestrian" speed="{SCHOOL.get("child_speed", 1.1)}" '
                     f'width="0.4" length="0.2" color="255,200,0"/>')
        gate_edge = east[3]                       # e0_sz2: 남쪽 보도, 정문 바로 동쪽
        homes = [west[4], west[1]]                # e0_r_sz1(정문 서쪽) / e0_r_sz3(정문 동쪽) 북쪽 보도
        per = 3600.0 / (cph / 4)
        for k, home in enumerate(homes):
            lines.append(f'    <personFlow id="child_in{k}" type="child" begin="0" end="100000" '
                         f'period="{per:.2f}" departPos="random">\n'
                         f'        <walk from="{home}" to="{gate_edge}" arrivalPos="1"/>\n'
                         f'    </personFlow>')
            lines.append(f'    <personFlow id="child_out{k}" type="child" begin="0" end="100000" '
                         f'period="{per:.2f}" departPos="1">\n'
                         f'        <walk from="{gate_edge}" to="{home}" arrivalPos="random"/>\n'
                         f'    </personFlow>')
    return "\n".join(lines) + ("\n" if lines else "")


_STATE_CHARS = set("GgyrsuoO")


def _load_net_tl(out_dir: str):
    """(net.xml 루트, 신호 링크 수)."""
    import xml.etree.ElementTree as ET
    net = ET.parse(os.path.join(out_dir, "highway.net.xml")).getroot()
    logic = net.find(f"tlLogic[@id='{TL_ID}']")
    if logic is None:
        raise RuntimeError(f"net.xml에 신호 '{TL_ID}'가 생성되지 않았습니다.")
    return net, len(logic.find("phase").get("state"))


def _save_tll(out_dir: str, program_id: str, phases: list, header: str = ""):
    """phases: [(지속시간 s, state 문자열, 주석)] → highway.tll.xml."""
    lines = [f"<additional>{header}",
             f'    <tlLogic id="{TL_ID}" type="static" programID="{program_id}" offset="0">']
    for duration, state, note in phases:
        comment = f"  <!-- {note} -->" if note else ""
        lines.append(f'        <phase duration="{float(duration):g}" state="{state}"/>{comment}')
    lines += ["    </tlLogic>", "</additional>"]
    with open(os.path.join(out_dir, "highway.tll.xml"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_tll(out_dir: str, full_state: bool = False):
    """TRAFFIC_LIGHT["phases"] 고정 주기.
    full_state=False: 한 글자 신호를 모든 링크에 적용 (straight layout)
    full_state=True : 링크 수만큼의 state 문자열을 그대로 사용 (교차로 layout)
    """
    _, n_links = _load_net_tl(out_dir)
    phases = TRAFFIC_LIGHT.get("phases") or [("G", 40), ("y", 4), ("r", 25)]
    out = []
    for signal, duration in phases:
        if full_state:
            if len(signal) != n_links or set(signal) - _STATE_CHARS:
                raise ValueError(f"교차로 phase state는 {n_links}글자({''.join(sorted(_STATE_CHARS))})"
                                 f"여야 합니다: {signal!r}")
            out.append((duration, signal, ""))
        else:
            if signal not in ("G", "g", "y", "r"):
                raise ValueError(f"TRAFFIC_LIGHT phase 신호는 G/g/y/r 중 하나여야 합니다: {signal}")
            out.append((duration, signal * n_links, ""))
    _save_tll(out_dir, "custom", out)


# 접근 엣지 → 축 (본선 = 동서, 교차 도로 = 남북)
_APPROACH_AXIS = {"e0": "EW", "e1_r": "EW", "n_in": "NS", "s_in": "NS",
                  "e0b": "EW", "e1_rb": "EW", "n_inb": "NS", "s_inb": "NS"}   # 포켓 엣지
_MAIN_EDGES = {"e0", "e1", "e0_r", "e1_r", "e0b", "e1_rb"}
# SUMO 연결 방향 → 이동류 (s 직진 / l·L·t 좌회전·유턴 / r·R 우회전)
_DIR_MOVE = {"s": "S", "l": "L", "L": "L", "t": "L", "r": "R", "R": "R"}


def _tl_links(net) -> dict:
    """신호 링크 번호 → ("veh", 축, 이동류) 또는 ("ped", 나란한 직진 축, 횡단보도 길이 m)."""
    crossings = {}
    for e in net.findall("edge[@function='crossing']"):
        crossed = set(e.get("crossingEdges", "").split())
        # 본선을 건너는 횡단보도는 남북 직진과 나란하다 (그 반대도 마찬가지)
        axis = "NS" if crossed & _MAIN_EDGES else "EW"
        crossings[e.get("id")] = (axis, float(e.find("lane").get("length")))
    links = {}
    for c in net.findall(f"connection[@tl='{TL_ID}']"):
        idx = int(c.get("linkIndex"))
        if c.get("to") in crossings:
            links[idx] = ("ped",) + crossings[c.get("to")]
        else:
            links[idx] = ("veh", _APPROACH_AXIS[c.get("from")], _DIR_MOVE[c.get("dir")])
    return links


def _write_plan_tll(out_dir: str):
    """SIGNAL_PLAN(현시별 녹색 이동류·녹색·황색 시간) → 링크별 state 문자열.

    각 현시는 최대 3단계로 펼친다:
      보행 녹색 → 보행 점멸(보행 적색, 차량은 녹색 유지) → 황색
    """
    import math
    if not SIGNAL_PLAN.get("phases"):
        raise ValueError("TRAFFIC_LIGHT['program']='plan'에는 SIGNAL_PLAN['phases']가 필요합니다.")
    net, n_links = _load_net_tl(out_dir)
    links = _tl_links(net)
    if sorted(links) != list(range(n_links)):
        raise RuntimeError(f"신호 링크 해석 실패: {len(links)}개 해석 / {n_links}개 링크")
    walk_speed = float(SIGNAL_PLAN.get("ped_check_speed", 1.0))
    min_walk = float(SIGNAL_PLAN.get("ped_min_walk", 7))

    def state(moves: set, stage: str) -> str:
        s = []
        for i in range(n_links):
            kind, axis, x = links[i]
            if kind == "ped":
                s.append("G" if stage == "walk" and f"{axis}_S" in moves else "r")
            elif x == "R":   # 우회전: 같은 축 직진 녹색이면 양보 진행, 아니면 일시정지 후 진행
                # 직진 황색 동안은 우회전도 황색 (없으면 g→s가 예고 없이 바뀌어
                # 정지선 바로 앞에서 멈출 수 없는 신호위반이 생겼다)
                if f"{axis}_S" in moves:
                    s.append("y" if stage == "yellow" else "g")
                else:
                    s.append("s")
            elif f"{axis}_{x}" in moves:
                s.append("y" if stage == "yellow" else "G")
            else:
                s.append("r")
        return "".join(s)

    out = []
    for k, (moves, green, yellow) in enumerate(SIGNAL_PLAN["phases"], 1):
        moves = set(moves)
        name = f"{k}현시 {'+'.join(sorted(moves))}"
        flash = max((math.ceil(length / walk_speed) for kind, axis, length in links.values()
                     if kind == "ped" and f"{axis}_S" in moves), default=0)
        flash = min(flash, max(green - min_walk, 0))
        if flash > 0:
            out.append((green - flash, state(moves, "walk"), f"{name} 녹색 (보행 녹색)"))
            out.append((flash, state(moves, "flash"), f"{name} 녹색 (보행 점멸)"))
        else:
            out.append((green, state(moves, "walk"), f"{name} 녹색"))
        out.append((yellow, state(moves, "yellow"), f"{name} 황색"))
    cycle = sum(d for d, _, _ in out)
    header = (f"\n    <!-- {SIGNAL_PLAN.get('name', '실제 교차로')} 신호 계획, 주기 {cycle:g} s. "
              f"링크 순서: "
              + " ".join(f"{i}:{a}_{x}" if kind == "veh" else f"{i}:보행({a})"
                         for i, (kind, a, x) in sorted(links.items()))
              + " -->")
    _save_tll(out_dir, "plan", out, header)


# ══════════════════════════════════════════════════════════════════
#  학교 + 어린이 보호구역 노면 (SCHOOL)  →  highway.school.xml
#
#  한국 법정 표준형을 "멀리서 보면 그대로" 수준으로 단순화해 그린다.
#  진행 방향마다 (입구부터):
#    적색 미끄럼방지포장 10 m × 2줄 (10 m 간격) → "어린이/보호구역" 글자 → (30)
#    → ◇ 횡단보도 예고 → (30) → 적색포장 + 노란 지그재그 (zigzag_length) → 정지선
#    → 노란 횡단보도 (정문 앞)
#  구간 전체: 오른쪽 가장자리에 노란 이중선 (주정차 금지). 시점/종점 길가에 표지판.
#  ego 인식용: 적색 포장 = type "school_zone"(제한속도 param), 노란 이중선 =
#  type "school_zone_line". sumo_env가 이 두 노면 색으로 보호구역 시작·범위를 읽는다.
#
#  층(layer): sumo-gui는 차로 면을 낮은 layer의 다각형 위에 그려서, 도로 위 표시는
#  20 이상이어야 보인다 (실측: 0.5·2는 가려짐). 차량은 그보다 위에 그려진다.
# ══════════════════════════════════════════════════════════════════
SCHOOL_ZONE_TYPE = "school_zone"         # 적색 포장 (인식: 보호구역 시작)
SCHOOL_LINE_TYPE = "school_zone_line"    # 노란 이중 가장자리선 (인식: 보호구역 범위)
SCHOOL_MARK_TYPE = "school_zone_mark"    # 그 밖의 노면표시 (보기용)
SCHOOL_SITE = (140.0, 90.0)              # 학교 부지 (본선 방향 폭, 깊이) m
_ROAD_LAYER = 20.0                       # 도로 위 표시의 기본 층
# 엣지 분할 지점의 정지선·화살표(SUMO link rule)는 그보다 훨씬 위에 그려져서, 덮개는 300에
# 둔다 (실측: 100은 정지선이 보이고 300은 가려짐, 차량은 300보다 위). 덮개 위로 이어져야 하는
# 점선·노란 이중선은 _COVER_LAYER + 0.1.
_COVER_LAYER = 300.0
_ASSET_REL = "../assets"                 # highway.school.xml(env/sumo) 기준 그림 경로


def _school_site(sz: dict):
    """학교 부지 (x0, x1, 위쪽 y(도로 쪽), 아래쪽 y)."""
    w, depth = SCHOOL_SITE
    top = -(sz["curb"] + 4.0)
    return sz["gate"] - w / 2, sz["gate"] + w / 2, top, top - depth


def _write_school(out_dir: str, road: dict, sz: dict):
    import math
    import xml.etree.ElementTree as ET
    from env.school_assets import FILES, SIZE

    pave = SCHOOL.get("pave_color", "150,35,35")
    yellow = SCHOOL.get("line_color", "255,200,0")
    white = "255,255,255"
    limit_param = f'        <param key="speed_limit" value="{sz["limit"]:g}"/>\n'
    net = ET.parse(os.path.join(out_dir, "highway.net.xml")).getroot()
    lines = ["<additional>",
             f'    <!-- {SCHOOL.get("name", "학교")} 어린이 보호구역 x {sz["lo"]:.0f}~{sz["hi"]:.0f} m, '
             f'제한 {sz["limit"] * 3.6:.0f} km/h -->']
    count = [0]

    def nid(prefix):
        count[0] += 1
        return f"{prefix}_{count[0]}"

    def poly(ptype, color, layer, pts, fill=True, width=None, extra=""):
        shape = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        attrs = (f'id="{nid(ptype)}" type="{ptype}" color="{color}" fill="{int(fill)}" '
                 f'layer="{layer}" shape="{shape}"' + (f' lineWidth="{width}"' if width else ""))
        lines.append(f"    <poly {attrs}" + (f">\n{extra}    </poly>" if extra else "/>"))

    def rect(ptype, color, layer, x0, x1, y0, y1, extra=""):
        poly(ptype, color, layer, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], extra=extra)

    def image(key, x, y, angle, layer, ptype=SCHOOL_MARK_TYPE):
        w, h = SIZE[key]
        lines.append(f'    <poi id="{nid("szi")}" type="{ptype}" x="{x:.2f}" y="{y:.2f}" '
                     f'width="{w}" height="{h}" angle="{angle:g}" layer="{layer}" '
                     f'imgFile="{_ASSET_REL}/{FILES[key]}"/>')

    # ---- net.xml에서 본선 차로 읽기: 엣지 → [(차로 중심 y, 폭/2, x 시작, x 끝)] (오른쪽 차로부터) ----
    lanes = {}
    for edge in net.iter("edge"):
        if edge.get("function"):
            continue
        out = []
        for lane in edge.findall("lane"):
            if lane.get("allow") == "pedestrian":
                continue
            pts = [tuple(map(float, p.split(","))) for p in lane.get("shape").split()]
            out.append((pts[0][1], float(lane.get("width", LANE_WIDTH)) / 2.0, pts[0][0], pts[-1][0]))
        lanes[edge.get("id")] = out

    east, west = main_chains(road)
    # 방향별: (진행 방향 d, 보호구역 엣지들, 입구 x, 출구 x, 지그재그 엣지, 횡단보도 뒤 엣지)
    dirs = [(+1.0, east[1:5], sz["lo"], sz["hi"], east[2], east[3]),
            (-1.0, west[1:5], sz["hi"], sz["lo"], west[2], west[3])]
    angle_of = {+1.0: 90.0, -1.0: 270.0}    # POI 그림 위쪽을 진행 방향으로 (SUMO: 시계방향, 북=0)

    def dashes(car, d, phase_x, xa, xb, layer=_ROAD_LAYER + 0.1):
        """차로 사이 흰 점선 (SUMO와 같은 3 m 선 / 6 m 틈, phase_x = 차로 시작점) 을 xa→xb 구간에."""
        lo_x, hi_x = sorted((xa, xb))
        for k in range(len(car) - 1):
            (ya, ha, _, _), (yb, hb, _, _) = car[k], car[k + 1]
            y = (ya + yb) / 2.0
            t = phase_x + d * 9.0 * math.floor(((xa - phase_x) * d) / 9.0)
            while (t < hi_x) if d > 0 else (t > lo_x):
                a, b = sorted((t, t + d * 3.0))
                a, b = max(a, lo_x), min(b, hi_x)
                if b - a > 0.1:
                    rect(SCHOOL_MARK_TYPE, white, layer, a, b, y - 0.06, y + 0.06)
                t += d * 9.0

    # 0) 엣지를 나눈 지점: SUMO가 그리는 정지선·화살표를 도로색으로 덮고 점선을 잇는다
    #    (실제 도로에는 없는 표시). 정문 횡단보도 지점은 정지선이 필요하므로 그대로 둔다.
    for d, chain in ((+1.0, east), (-1.0, west)):
        for e_in, e_out in zip(chain[:-1], chain[1:]):
            if e_in in (east[2], west[2]):
                continue
            car = lanes[e_in]
            end_x, start_x = car[0][3], lanes[e_out][0][2]
            ys = [y + s * h for y, h, _, _ in car for s in (-1, 1)]
            a, b = sorted((end_x - d * 7.0, start_x))
            rect(SCHOOL_MARK_TYPE, "0,0,0", _COVER_LAYER, a, b, min(ys), max(ys))
            walk_y = (min(ys) - 2.0, min(ys)) if d > 0 else (max(ys), max(ys) + 2.0)   # 오른쪽 보도
            a, b = sorted((end_x - d * 1.0, start_x))
            rect(SCHOOL_MARK_TYPE, "128,128,128", _COVER_LAYER, a, b, *walk_y)
            dashes(car, d, car[0][2], end_x - d * 7.0, start_x, _COVER_LAYER + 0.1)
            a, b = sorted((end_x - d * 7.0, start_x))
            for yb in (min(ys), max(ys)):                # 덮개가 가린 흰 차도 가장자리선
                rect(SCHOOL_MARK_TYPE, white, _COVER_LAYER + 0.1, a, b, yb - 0.05, yb + 0.05)

    for d, zone_edges, x_in, x_out, zz_edge, after_edge in dirs:
        car = lanes[zone_edges[0]]                       # 차로 배치는 구간 내내 같다
        right = [y - d * h for y, h, _, _ in car]        # 차로별 오른쪽 경계 y
        left = [y + d * h for y, h, _, _ in car]
        x_at = lambda s: x_in + d * s                    # 입구에서 진행 방향으로 s m
        stop_x = lanes[zz_edge][0][3]                    # 지그재그 엣지 차로 끝 = 정지선 위치
        s_stop = (stop_x - x_in) * d
        s_after = (lanes[after_edge][0][2] - x_in) * d   # 횡단보도 건너편 차로 시작
        s_end = (x_out - x_in) * d
        s0_lane = (car[0][2] - x_in) * d                 # 첫 차로 시작 (분할 지점 덮개 뒤)
        zz = sz["zz"]

        def lane_rect(ptype, color, layer, s0, s1, extra=""):
            for y, h, _, _ in car:
                a, b = sorted((x_at(s0), x_at(s1)))
                rect(ptype, color, layer, a, b, y - h, y + h, extra)

        def across(ptype, color, layer, s0, s1):        # 그 방향 전 차로를 가로지르는 띠
            a, b = sorted((x_at(s0), x_at(s1)))
            ys = right + left
            rect(ptype, color, layer, a, b, min(ys), max(ys))

        # 1) 적색 미끄럼방지포장: 시점부 2줄 + 횡단보도 앞 지그재그 구간
        bands = [(s0_lane, s0_lane + 10.0), (s0_lane + 20.0, s0_lane + 30.0)]
        for s0, s1 in bands + [(s_stop - zz, s_stop)]:
            lane_rect(SCHOOL_ZONE_TYPE, pave, _ROAD_LAYER, s0, s1, limit_param)
        # 적색 포장이 가린 흰 차선 점선을 다시 그린다 (시점부만; 지그재그 구간은 지그재그가 대신)
        for s0, s1 in bands:
            dashes(car, d, car[0][2], x_at(s0), x_at(s1))

        # 2) 노면 글자·기호 (차로마다, 그림 위쪽 = 진행 방향)
        marks = [("text", s0_lane + 35.5), ("speed", s0_lane + 41.5), ("speed", s_stop - zz - 18.0)]
        for key, s in marks:
            for y, _, _, _ in car:
                image(key, x_at(s), y, angle_of[d], _ROAD_LAYER + 0.3)
        for y, h, _, _ in car:                           # ◇ 횡단보도 예고 (흰 마름모 테두리)
            cx = x_at(s_stop - zz - 38.0)
            poly(SCHOOL_MARK_TYPE, white, _ROAD_LAYER + 0.2,
                 [(cx - 1.6, y), (cx, y + 0.7), (cx + 1.6, y), (cx, y - 0.7), (cx - 1.6, y)],
                 fill=False, width=0.15)

        # 3) 노란 지그재그 (차로 경계마다) + 흰 정지선
        bounds = [right[0]] + [(left[k] + right[k + 1]) / 2.0 for k in range(len(car) - 1)] + [left[-1]]
        for yb in bounds:
            pts, s, k = [], s_stop - zz, 0
            while s <= s_stop - 0.5 + 1e-6:
                pts.append((x_at(s), yb + (0.3 if k % 2 else -0.3)))
                s += 1.5
                k += 1
            poly(SCHOOL_MARK_TYPE, yellow, _ROAD_LAYER + 0.2, pts, fill=False, width=0.15)
        across(SCHOOL_MARK_TYPE, white, _ROAD_LAYER + 0.2, s_stop - 0.45, s_stop)

        # 4) 노란 이중 가장자리선 (주정차 금지): 입구 ~ 지그재그 앞, 횡단보도 뒤 ~ 출구
        yr = right[0]
        for s0, s1 in ((0.0, s_stop - zz), (s_after, s_end)):
            for inset in (0.15, 0.45):
                yl = yr + d * inset                      # 차로 안쪽으로 inset
                a, b = sorted((x_at(s0), x_at(s1)))
                rect(SCHOOL_LINE_TYPE, yellow, _COVER_LAYER + 0.1, a, b, yl - 0.06, yl + 0.06)

        # 5) 길가 표지판 (오른쪽 보도 바깥, 위에서 본 아이콘이라 글자는 북쪽이 위)
        side_y = -d * (sz["curb"] + 1.8)
        image("sign_start", x_at(1.0), side_y - d * 0.6, 0.0, 6.0)
        image("sign_end", x_at(s_end), side_y - d * 0.4, 0.0, 6.0)

    # 6) 노란 횡단보도: SUMO 흰 줄무늬를 도로색으로 덮고 노란 줄무늬를 다시 그린다
    for edge in net.iter("edge"):
        if edge.get("function") != "crossing" or not edge.get("id").startswith(":sz_c_"):
            continue
        lane = edge.find("lane")
        (ax, ay), (bx, by) = [tuple(map(float, p.split(","))) for p in lane.get("shape").split()][:2]
        hw = float(lane.get("width", sz["cw"])) / 2.0
        length = math.hypot(bx - ax, by - ay)
        ux, uy = (bx - ax) / length, (by - ay) / length   # 횡단 방향
        nx, ny = -uy, ux                                  # 차량 진행 방향
        def quad(t0, t1, w):
            return [(ax + ux * t + nx * s * w, ay + uy * t + ny * s * w)
                    for t, s in ((t0, -1), (t1, -1), (t1, 1), (t0, 1))]
        poly(SCHOOL_MARK_TYPE, "0,0,0", _ROAD_LAYER, quad(0.0, length, hw + 0.05))
        t = 0.4
        while t + 0.45 <= length - 0.2:
            poly(SCHOOL_MARK_TYPE, yellow, _ROAD_LAYER + 0.1, quad(t, t + 0.45, hw - 0.2))
            t += 0.9

    # 7) 학교 (남쪽 길가): 부지·운동장·트랙·교사동·담장·정문 진입로·학교 이름
    x0, x1, top, bot = _school_site(sz)
    g = sz["gate"]
    rect("school", "196,190,172", 0.6, x0, x1, bot, top)                     # 부지 바닥
    rect("school", "176,174,168", 0.7, g - 4.0, g + 4.0, top, -sz["curb"])   # 정문 → 보도 진입로
    fx0, fx1, fy0, fy1 = g + 2.0, x1 - 8.0, bot + 10.0, top - 8.0            # 운동장 (동쪽)
    rect("school", "201,160,105", 0.7, fx0, fx1, fy0, fy1)
    cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2                                # 트랙 (흰 타원)
    rx, ry = (fx1 - fx0) / 2 - 5.0, (fy1 - fy0) / 2 - 5.0
    poly("school", white, 0.8, [(cx + rx * math.cos(a), cy + ry * math.sin(a))
                                for a in [2 * math.pi * k / 48 for k in range(49)]],
         fill=False, width=0.35)
    rect("school", "176,174,168", 0.7, g - 4.0, g + 2.0, bot + 8.0, top)    # 정문 → 현관 길
    # 교사동: 뒤쪽 본관 + 서쪽 별관 (ㄱ자), 그림자는 남동쪽 (나무와 같은 해 방향)
    blocks = [(x0 + 8.0, g - 8.0, bot + 8.0, bot + 24.0), (x0 + 8.0, x0 + 24.0, bot + 24.0, top - 16.0)]
    for bx0, bx1, by0, by1 in blocks:
        rect("school", "70,70,64,110", 0.8, bx0 + 1.5, bx1 + 1.5, by0 - 1.5, by1 - 1.5)
    for bx0, bx1, by0, by1 in blocks:
        rect("school", "214,201,168", 0.9, bx0, bx1, by0, by1)
        rect("school", "186,170,135", 0.95, bx0 + 1.0, bx1 - 1.0, (by0 + by1) / 2 - 0.3,
             (by0 + by1) / 2 + 0.3)                                            # 지붕 용마루
    image("name", (blocks[0][0] + blocks[0][1]) / 2, blocks[0][2] + 5.0, 0.0, 6.0, "school")
    # 담장 (정문 자리 8 m만 비움)
    fence, t = "45,95,55", 0.35
    for a, b in ((x0, g - 4.0), (g + 4.0, x1)):
        rect("school", fence, 1.0, a, b, top - t, top)
    rect("school", fence, 1.0, x0, x1, bot, bot + t)
    rect("school", fence, 1.0, x0, x0 + t, bot, top)
    rect("school", fence, 1.0, x1 - t, x1, bot, top)

    lines.append("</additional>")
    with open(os.path.join(out_dir, "highway.school.xml"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _blob(cx: float, cy: float, r: float, rng, n: int = 14,
          roughness: float = 0.35) -> str:
    """유기적인(울퉁불퉁한) 수관 실루엣 다각형을 만든다.

    실제 나무를 위에서 내려다보면 완전한 원이 아니라 잎 뭉치들이
    만드는 불규칙한 테두리다. 꼭짓점마다 반지름을 무작위로 흔들되,
    이웃 꼭짓점과 평균을 내서(smoothing) 뾰족하지 않고 부드럽게
    울퉁불퉁한 실루엣을 얻는다.
    """
    import math
    raw = [1.0 + rng.uniform(-roughness, roughness) for _ in range(n)]
    sm = [(raw[i - 1] + raw[i] + raw[(i + 1) % n]) / 3.0 for i in range(n)]
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        rr = r * sm[i]
        pts.append(f"{cx + rr * math.cos(a):.2f},{cy + rr * math.sin(a):.2f}")
    return " ".join(pts)


def _write_scenery(out_dir: str, road: dict):
    """도로변 나무를 highway.scenery.xml (additional 파일)로 생성.

    항공뷰(top-down) 조경처럼 보이도록 나무 한 그루를 여러 겹으로 그린다:
      layer 1  그림자  : 수관을 남동쪽으로 살짝 밀어낸 반투명 검정 blob
                         (해가 북서쪽에 있는 항공사진 느낌)
      layer 2  수관 몸통: 중간톤 초록의 불규칙 실루엣
      layer 3  잎 뭉치  : 수관 안에 어두운/짙은 클럼프 1~2개
                         → 캐노피의 입체적인 질감
      layer 4  하이라이트: 북서쪽으로 치우친 밝은 초록 작은 blob
    배치도 실제 조경처럼: 단독목 / 2~3그루 군락 / 낮은 관목이 섞인다.
    SCENERY["trees"]가 False면 빈 additional 파일 생성.
    """
    import random
    path = os.path.join(out_dir, "highway.scenery.xml")
    cfg = SCENERY if isinstance(SCENERY, dict) else {}
    if not cfg.get("trees", False):
        with open(path, "w") as f:
            f.write("<additional/>\n")
        return

    rng = random.Random(int(cfg.get("seed", 7)))
    spacing = float(cfg.get("tree_spacing", 45.0))
    jitter = float(cfg.get("tree_jitter", 0.5))
    side = float(cfg.get("side_offset", 6.0))
    length = float(road["length"])
    road_bottom = -3.2 * int(road["num_lanes"])
    road_top, gap = 0.0, None
    if is_intersection(road):
        # 반대 방향 차선(y > 0)과 보도 바깥에 심고, 교차 도로 주변은 비운다
        road_top = 3.2 * int(INTERSECTION.get("opposite_lanes", road["num_lanes"])) + 2.0
        road_bottom -= 2.0
        half = 3.2 * int(INTERSECTION.get("side_lanes", 2)) + 2.0 + 12.0
        center = float(INTERSECTION.get("position", length / 2))
        gap = (center - half, center + half)

    # (몸통, 어두운 클럼프, 밝은 하이라이트) 초록 팔레트 — 수종별 톤
    palettes = [
        ("46,110,44", "32,84,32", "96,158,84"),
        ("38,98,40",  "26,72,28", "84,146,74"),
        ("56,122,50", "40,92,38", "110,168,92"),
        ("30,88,36",  "20,64,26", "76,138,70"),
    ]
    bush_greens = ["70,124,58", "84,136,64", "62,112,52"]

    lines = ["<additional>"]
    tid = [0]

    def draw_tree(x, y, r):
        body, dark, light = rng.choice(palettes)
        i = tid[0]; tid[0] += 1
        # 그림자 (남동쪽 +x, -y 로 오프셋, 반투명)
        lines.append(f'    <poly id="sh_{i}" color="20,26,18,70" fill="1" '
                     f'layer="1" shape="{_blob(x + r*0.35, y - r*0.35, r*1.02, rng)}"/>')
        # 수관 몸통
        lines.append(f'    <poly id="tr_{i}" color="{body}" fill="1" '
                     f'layer="2" shape="{_blob(x, y, r, rng)}"/>')
        # 잎 뭉치 1~2개 (어두운 톤, 수관 내부 무작위 위치)
        for k in range(rng.randint(1, 2)):
            ox = rng.uniform(-0.35, 0.35) * r
            oy = rng.uniform(-0.35, 0.35) * r
            lines.append(f'    <poly id="cl_{i}_{k}" color="{dark}" fill="1" '
                         f'layer="3" shape="{_blob(x + ox, y + oy, r * rng.uniform(0.4, 0.6), rng, n=10)}"/>')
        # 하이라이트 (북서쪽 -x, +y — 그림자 반대 방향)
        lines.append(f'    <poly id="hl_{i}" color="{light}" fill="1" '
                     f'layer="4" shape="{_blob(x - r*0.28, y + r*0.28, r*0.34, rng, n=8, roughness=0.25)}"/>')

    def draw_bush(x, y):
        i = tid[0]; tid[0] += 1
        g = rng.choice(bush_greens)
        r = rng.uniform(1.0, 1.8)
        lines.append(f'    <poly id="bsh_{i}" color="20,26,18,55" fill="1" '
                     f'layer="1" shape="{_blob(x + 0.3, y - 0.3, r, rng, n=8)}"/>')
        lines.append(f'    <poly id="bs_{i}" color="{g}" fill="1" '
                     f'layer="2" shape="{_blob(x, y, r, rng, n=8)}"/>')

    sz = school_layout(road)
    for side_sign, base_y in [(+1, road_top + side), (-1, road_bottom - side)]:
        gaps = [gap] if gap is not None else []
        if sz and side_sign < 0:          # 남쪽 길가의 학교 부지는 비운다 (학교가 따로 그림)
            gaps.append((sz["gate"] - SCHOOL_SITE[0] / 2, sz["gate"] + SCHOOL_SITE[0] / 2))
        x = rng.uniform(0, spacing)
        while x < length:
            hit = next((g for g in gaps if g[0] - 6.0 < x < g[1] + 6.0), None)
            if hit is not None:
                x = hit[1] + 6.0
                continue
            y = base_y + side_sign * rng.uniform(0, 4.0)
            roll = rng.random()
            if roll < 0.12:
                draw_bush(x, y)                        # 15%: 낮은 관목
            elif roll < 0.67:
                draw_tree(x, y, rng.uniform(2.6, 4.6))  # 55%: 단독목 (실제 수관 스케일)
            else:
                # 25%: 2~3그루 군락 (겹치듯 붙여 심어 숲 덩어리 느낌)
                n = rng.randint(2, 4)
                for _ in range(n):
                    gx = x + rng.uniform(-5.5, 5.5)
                    gy = y + side_sign * rng.uniform(-2.0, 4.5)
                    draw_tree(gx, gy, rng.uniform(2.2, 3.8))
                x += spacing * 0.4                     # 군락 뒤엔 간격 조금 더
            x += spacing * rng.uniform(1.0 - jitter, 1.0 + jitter)
    if sz:   # 학교 담장 안쪽 둘레에 나무 (정문 앞은 비움)
        x0, x1, top, bot = _school_site(sz)
        x = x0 + 6.0
        while x < x1 - 4.0:
            if abs(x - sz["gate"]) > 10.0:
                draw_tree(x + rng.uniform(-1.5, 1.5), top - 3.5, rng.uniform(2.2, 3.2))
            x += 11.0
        y = top - 14.0
        while y > bot + 12.0:
            for tx in (x0 + 3.5, x1 - 3.5):
                draw_tree(tx, y + rng.uniform(-1.5, 1.5), rng.uniform(2.2, 3.2))
            y -= 12.0
    lines.append("</additional>")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
