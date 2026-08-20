# 启动 serve.py（M1 推理服务）
# 前置：
#   1. 已开 TUN/全局代理（siglip2 首次下载需要，之后走本地缓存可离线）
#   2. ng.pt 在仓库外 _models/（相对本脚本为 ../../_models/ng.pt）
# 启动后监听 tcp://*:5555。用 ModelClient 请求 reset/info/predict。
param(
    [int]$Port = 5555,
    [string]$Ckpt = $null
)
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $Ckpt) { $Ckpt = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path '_models\ng.pt') }
$nitrogenRepo = (Resolve-Path (Join-Path $repo '..\..\NitroGen')).Path

if (-not (Test-Path $Ckpt)) { Write-Error "ng.pt not found: $Ckpt"; exit 1 }

Write-Host "Starting serve.py with ckpt=$Ckpt port=$Port"
Write-Host "TUN/代理需已开启（siglip2 首次下载），之后可离线。"
& (Join-Path $repo '.venv\Scripts\python.exe') (Join-Path $nitrogenRepo 'scripts\serve.py') $Ckpt --port $Port
