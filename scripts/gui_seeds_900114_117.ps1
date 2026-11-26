# bs_03 / ipjeong 충돌 장면 GUI (klppo1_s6_s1 step200k, shadow)
$ErrorActionPreference = "Stop"
$Root = "C:\Users\USER\Desktop\청계로드-kl-ppo"
$env:CGC_STAGE = "6"
$env:OBS_SCAN = "1"
Set-Location $Root
$seeds = @(900114, 900115, 900117)
foreach ($s in $seeds) {
    Write-Host "`n======== seed $s (Enter after watching) ========" -ForegroundColor Cyan
    py -3.13 scripts/gui_drive_seed.py --seed $s --model klppo --shield shadow --delay 120 --pause
}