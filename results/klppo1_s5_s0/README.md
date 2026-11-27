# 청계천로 5단계 최종 모델

평가에는 step250k.pt를 사용한다. model.pt는 30만 스텝 모델이다. 기본 PPO 정책의 클래스는 utils/networks.py의 HybridActorCritic이다. 이 체크포인트는 분리된 KL-PPO 정책을 담으므로 algorithms/kl_ppo.py의 load_eval_policy 또는 KLPPO.from_checkpoint로 복원한다.

|시드|충돌|공식 완주|완벽 완주|장기대기 또는 시간초과|이동 중 속도|
|---|---:|---:|---:|---:|---:|
|개발 900100–900129|1/30 (3.3%)|28/30 (93.3%)|26/30 (86.7%)|미측정|미측정|
|독립 900300–900399|3/100 (3%)|87/100 (87%)|84/100 (84%)|63/100 (63%)|19.6 km/h|

PowerShell에서 저장소 루트를 기준으로 실행한다.

~~~powershell
$env:CGC_STAGE='5'
$env:OBS_SCAN='0'
$env:STOP_FIX='0'
$env:PLACE_FIX='0'
$env:CGC_THROUGH_VPH='400'
py -3.13 evaluation/drive_parallel.py --model results/klppo1_s5_s0/step250k.pt --seeds 900300-900399 --shield shadow --workers 4 --json-out results/local_evaluation/stage5.json
~~~
