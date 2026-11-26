# τ 시험만 (2차 학습·감시는 설계 확정 후 수동)
$ErrorActionPreference = "Stop"
$Root = "C:\Users\USER\Desktop\청계로드-kl-ppo"
$Init = Join-Path $Root "results\klppo1_s6_s1\step200k.pt"
$LogDir = Join-Path $Root "results\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$env:CGC_STAGE = "6"
$env:OBS_SCAN = "1"
Set-Location $Root

Write-Host "=== τ 시험 (workers 3) ==="
py -3.13 scripts/lane_temp_probe.py --workers 3 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "lane_temp_probe.out")

$tauFile = Join-Path $Root "results\lane_temp_probe\chosen_tau.txt"
if (-not (Test-Path $tauFile)) {
    Write-Host "기준 미달: chosen_tau.txt 없음. τ=1.0 으로 진행"
    $tau = "1.0"
} else {
    $tau = Get-Content $tauFile -Raw
}
Write-Host "lane-temp=$tau"
Write-Host "=== τ 시험만 끝. 2차 학습은 설계 확정 후 수동 실행 ==="