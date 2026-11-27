# 청계로드 프로젝트 페이지

팀 저장소의 `gh-pages` 브랜치입니다. 이 브랜치에는 페이지 파일만 있습니다 (코드는 `main` 과 각 작업 브랜치).

## 페이지 켜기 (저장소 관리자가 한 번)

Settings → Pages → Build and deployment → Source: **Deploy from a branch** → Branch: **gh-pages**, 폴더 **/ (root)** → Save.
몇 분 뒤 `https://transerz.github.io/Artificial-Intelligence-and-Control-for-Autonomous-Driving/` 에서 열립니다.

## 발표 PDF, 데이터셋 링크 넣기

`app.js` 맨 위 `LINKS` 두 줄만 채우면 상단, 데이터셋 구간, 하단 버튼이 모두 바뀝니다. 비워 두면 "준비 중"으로 보입니다.

```js
const LINKS = {
  slides: "slides.pdf",                                   // 이 폴더에 slides.pdf 를 같이 올린다 (로그인 없이 열림)
  dataset: "https://huggingface.co/datasets/<계정>/<이름>",
};
```

## 내 컴퓨터에서 보기

```bash
python3 -m http.server 8000     # 이 폴더에서 실행 → http://localhost:8000
```

## 파일

| 파일 | 내용 |
|---|---|
| `index.html`, `style.css`, `app.js` | 페이지 (빌드 과정 없음) |
| `run-data.js` | 상단 영상과 노선도를 맞추는 주행 로그 (5단계 모델, seed 1000, 스텝별 거리·속도·정차 수) |
| `fail-data.js` | 테스트 90판(seed 1000~1089) 중 실패한 판의 종료 위치 |
| `media/` | SUMO 화면 녹화 (H.264, 소리 없음) |
| `fonts/` | 서울남산체, 서울한강체에서 페이지에 쓰는 글자만 남긴 woff2 |
| `tools/` | 영상과 수치를 만든 스크립트 |

## 영상과 테스트 판 수치를 만든 방법

`model/dongyeon-bc56` 브랜치의 `models/dongyeon_bc/s5_student.pt` 를 `CGC_STAGE=5`, 안전 필터 감시 모드, 결정적 행동으로 실행했습니다 (SUMO 1.27.1).

- `tools/render_run.py`: 학생 모델을 numpy 로 계산해 주행시키고, sumo-gui 화면을 스텝마다 저장하고 로그를 남깁니다. `--nogui` 면 로그만 남깁니다.
- `tools/eval5.sh`: seed 1000~1089 를 화면 없이 실행합니다.
- `tools/analyze5.py`: 판별 종료 사유와 위치를 집계합니다. 결과: 완주 80, 시간초과 7, 신호위반 2, 충돌 1.
- `tools/enc.sh`: 저장한 화면을 mp4 로 묶습니다 (배경색 교체, 12배속).

스크립트 안의 경로는 만든 사람 컴퓨터 기준이라 실행하려면 고쳐야 합니다.

## 출처

도로 형상 © OpenStreetMap contributors (ODbL). 서체: 서울남산체, 서울한강체 (서울특별시).
