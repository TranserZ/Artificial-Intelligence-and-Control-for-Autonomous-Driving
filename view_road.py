"""도로 구조 확인 스크립트 — 학습 없이 도로만 눈으로 확인한다.

env/road_config.py 를 수정한 뒤, 학습을 돌리기 전에
"도로가 의도대로 만들어졌는지 / 교통 밀도가 적당한지"를
sumo-gui 화면으로 빠르게 확인하는 용도.

사용법:
    python view_road.py             # sumo-gui로 도로를 띄운다 (정적 — 자동 재생 안 함)
    python view_road.py --netedit   # netedit(SUMO 도로 편집기)로 도로망 열기
    python view_road.py --nogui     # 화면 없이 설정 검증만 (GUI 없는 서버용)

기본 모드(sumo-gui) 동작:
    창이 뜨면 도로만 정적으로 보인다. 차를 움직이려면 상단의 ▶(플레이) 버튼을
    직접 누르면 되고, 시뮬레이션이 끝나도 창이 저절로 닫히지 않는다.
    (일시정지/한 스텝씩 진행 버튼으로 원하는 만큼 관찰 가능)

관찰 포인트:
    - 차선 수 / 도로 길이가 road_config.ROAD 대로인지  ← 재생 없이 바로 확인 가능
    - (▶ 재생 후) 교통량(TRAFFIC["vehs_per_hour"])이 너무 한산하거나 막히지 않는지
      → 정체가 심하면 학습 난이도가 급상승하니 여기서 미리 조정
    - 재생 속도는 상단 Delay(ms) 값으로 조절 (크게 잡을수록 느리게)

※ netedit 관련 주의:
    netedit에서 도로를 수정하고 저장해도, 다음번 train.py 실행 시
    road_builder가 env/road_config.py 기준으로 파일을 **덮어쓴다**.
    이 프로젝트에서 도로의 원본(source of truth)은 어디까지나
    env/road_config.py 이므로, 영구적인 수정은 그 파일에서 할 것.
    (netedit은 구조를 뜯어보거나 좌표를 확인하는 용도로 쓰면 좋다)
"""
import argparse
import os
import subprocess

from env.road_config import ROAD, EGO, TRAFFIC
from env.mdp_config import SIMULATION
from env import road_builder
# sumo_env를 import하면 SUMO_HOME 자동 감지 로직이 함께 실행된다
from env.sumo_env import SumoHighwayEnv  # noqa: F401  (자동 감지 목적의 import)
from sumolib import checkBinary

BASE = os.path.dirname(os.path.abspath(__file__))


def print_summary():
    """현재 road_config 내용을 표로 요약 출력."""
    print("─" * 46)
    print(" 현재 도로 설정 (env/road_config.py)")
    print("─" * 46)
    print(f"  도로 길이      : {ROAD['length']} m")
    print(f"  차선 수        : {ROAD['num_lanes']}")
    print(f"  제한속도       : {ROAD['speed_limit']} m/s "
          f"(≈ {float(ROAD['speed_limit']) * 3.6:.0f} km/h)")
    print(f"  배경 교통량    : {TRAFFIC['vehs_per_hour']} 대/시간")
    print(f"  배경차 최고속도: {TRAFFIC['max_speed']} m/s")
    print(f"  ego 가속/감속  : +{EGO['accel']} / -{EGO['decel']} m/s²")
    sz = road_builder.school_layout(ROAD)
    if sz:
        print(f"  어린이 보호구역: x {sz['lo']:.0f}~{sz['hi']:.0f} m (정문 {sz['gate']:.0f} m), "
              f"{sz['limit'] * 3.6:.0f} km/h, 지그재그 {sz['zz']:.0f} m 차로변경 금지")
    print("─" * 46)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="도로 구조를 GUI로 확인")
    parser.add_argument("--netedit", action="store_true",
                        help="sumo-gui 대신 netedit(도로 편집기)로 열기")
    parser.add_argument("--nogui", action="store_true",
                        help="화면 없이 설정 검증만 (GUI 없는 환경용)")
    parser.add_argument("--live", action="store_true",
                        help="(청계천) sumo-gui와 함께 '정류장 대기 현황' 표 창(브라우저 새 창)을 "
                             "띄운다. 정류장별 대기 인원·버스 탑승 인원이 1초마다 갱신된다")
    parser.add_argument("--seconds", type=int, default=120,
                        help="--nogui 검증 시 시뮬레이션 길이(초). 기본 120초. "
                             "(GUI 모드는 시간 제한 없이 사용자가 직접 제어)")
    args = parser.parse_args()

    # 1) road_config 기준으로 SUMO 파일 (재)생성 — 수정사항이 즉시 반영된다
    #    demo_ego=True: 구경용이므로 SUMO가 스스로 모는 데모 버스를 넣는다 (청계천 한정).
    #    (train/test는 ego를 TraCI로 직접 투입하므로 데모 버스 없음 — 여기서만 켠다)
    sumocfg = road_builder.build(ROAD, EGO, TRAFFIC, os.path.join(BASE, "env", "sumo"),
                                 demo_ego=True)
    net_file = os.path.join(BASE, "env", "sumo", "highway.net.xml")
    print(f"도로 생성 완료: {sumocfg}")
    print_summary()

    # 2) 요청한 모드로 열기
    if args.netedit:
        # netedit: SUMO에 포함된 도로망 편집기. 노드/엣지/차선을 시각적으로 확인.
        print("netedit 실행 중... (창을 닫으면 종료)")
        subprocess.run([checkBinary("netedit"), net_file])

    elif args.nogui:
        # 화면 없이 짧게 돌려서 "설정이 유효하고 차가 실제로 투입되는지"만 검증
        result = subprocess.run(
            [checkBinary("sumo"), "-c", sumocfg,
             "--step-length", str(SIMULATION["step_length"]),
             "--end", str(args.seconds),
             "--no-warnings", "true", "--no-step-log", "true",
             "--duration-log.statistics", "true"],   # 종료 시 차량 통계 출력
            capture_output=True, text=True)
        # SUMO의 통계 요약(투입/도착 차량 수 등)을 그대로 보여준다
        print(result.stderr.strip() or result.stdout.strip())
        print("설정 검증 OK" if result.returncode == 0 else "설정 오류!")

    elif args.live:
        # 실시간 대기 현황: TraCI로 sumo-gui를 띄우고 우리가 스텝을 돌리면서,
        # 1초(시뮬레이션 시간)마다 정류장별 대기 인원과 버스 상태를 읽어
        # 두 번째 창(브라우저 새 창)의 표에 보낸다 — tools/live_board.py.
        # (TraCI가 재생을 주도하므로 ▶ 버튼 없이 바로 재생된다. Delay로 속도 조절)
        import json
        import sys
        import traci
        sys.path.insert(0, os.path.join(BASE, "tools"))
        from live_board import LiveBoard

        meta_path = os.path.join(BASE, "env", "sumo", "cgc.line_meta.json")
        if not os.path.exists(meta_path):
            raise SystemExit("--live는 청계천 시나리오(정류장이 있는 도로)에서만 쓸 수 있습니다.")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        # 표 순서 = 정류장이 가장 많은 노선의 정차 순서 (보통 왕복 A01_loop), 나머지는 뒤에
        order = list(max((ln["stops"] for ln in meta["lines"].values()), key=len, default=[]))
        order += [s for s in meta["stops"] if s not in order]
        names = {s: meta["stops"][s].get("name", s) for s in order}

        board = LiveBoard(order, names)
        url = board.start()
        print("sumo-gui + 대기 현황 표 창 실행 중 ... sumo-gui 창을 닫거나 Ctrl+C로 종료")
        print(f"  · 표 창이 안 뜨면 브라우저에서 직접 여세요: {url}")

        traci.start([checkBinary("sumo-gui"), "-c", sumocfg, "--start",
                     "--step-length", str(SIMULATION["step_length"]),
                     "--delay", str(SIMULATION["gui_delay"])])
        capacity = {}

        def bus_rows():
            rows = []
            for vid in traci.vehicle.getIDList():
                vtype = traci.vehicle.getTypeID(vid)
                if not vtype.startswith("ego_type"):
                    continue
                if vtype not in capacity:
                    capacity[vtype] = traci.vehicletype.getPersonCapacity(vtype)
                upcoming = [s.stoppingPlaceID for s in traci.vehicle.getStops(vid, 2)
                            if s.stoppingPlaceID]
                at_stop = upcoming[0] if traci.vehicle.isAtBusStop(vid) and upcoming else None
                nxt = upcoming[1:2] if at_stop else upcoming[:1]
                speed = traci.vehicle.getSpeed(vid)
                if at_stop:
                    state = f"정차 중 · {names.get(at_stop, at_stop)}"
                elif speed < 0.5:
                    state = "신호·정체로 멈춤"
                else:
                    state = "주행 중"
                rows.append({"id": vid, "onboard": traci.vehicle.getPersonNumber(vid),
                             "capacity": capacity[vtype], "state": state,
                             "next_stop": names.get(nxt[0], nxt[0]) if nxt else "종점",
                             "speed_kmh": speed * 3.6, "at_stop": at_stop})
            return rows

        next_t = 0.0
        try:
            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                t = traci.simulation.getTime()
                if t >= next_t:                        # 1초마다 표 갱신
                    next_t = t + 1.0
                    waits = {}
                    for sid in order:
                        try:
                            waits[sid] = traci.busstop.getPersonCount(sid)
                        except traci.TraCIException:
                            pass
                    board.update(t, waits, bus_rows())
        except (KeyboardInterrupt, traci.FatalTraCIError):
            pass                                       # 창을 닫았거나 Ctrl+C
        finally:
            try:
                traci.close()
            except Exception:
                pass
            board.stop()

    else:
        # sumo-gui: 도로를 "정적으로" 띄운다.
        #   --start, --quit-on-end, --end 를 모두 주지 않는 것이 핵심:
        #     --start 없음      → 자동 재생 안 함. 창이 뜨면 도로만 보이고,
        #                         사용자가 ▶(플레이) 버튼을 눌러야 차가 움직인다.
        #     --quit-on-end 없음 → 시뮬레이션이 끝나도 창이 저절로 닫히지 않는다.
        #     --end 없음        → 시간 제한 없음. 원하는 만큼 돌려볼 수 있다.
        #   --delay 100 : 재생 시 스텝당 100ms (0이면 눈으로 못 따라갈 만큼 빠름).
        #                 GUI 상단 Delay 입력칸에서 실행 중에도 바꿀 수 있다.
        print("sumo-gui 실행 중...")
        print("  · 창이 뜨면 도로만 보입니다. ▶(플레이) 버튼을 눌러야 차가 움직입니다.")
        print("  · (청계천) 데모 버스는 60초에 처음 출발, 이후 5분마다 1대씩 노선을 돕니다.")
        print("    주황색 사람 = 버스 승객, 파란색 사람 = 일반 보행자.")
        print(f"  · 확대: env/mdp_config.py의 gui_zoom={SIMULATION['gui_zoom']}")
        print("  · 속도 조절: 상단 Delay(ms) 값을 조정 (크게 = 느리게)")
        print("  · 종료하려면 창을 닫으세요.")
        subprocess.run(
            [checkBinary("sumo-gui"), "-c", sumocfg,
             "--step-length", str(SIMULATION["step_length"]),
             "--delay", str(SIMULATION["gui_delay"])])
