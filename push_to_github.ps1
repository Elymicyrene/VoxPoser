# push_to_github.ps1 - VoxPoser auto-push to GitHub
# Usage: Right-click -> Run with PowerShell, or run in PowerShell

Set-Location $PSScriptRoot
$ErrorActionPreference = "Continue"

# Check remote
$remote = git remote get-url origin 2>$null
if (-not $remote) {
    Write-Host "[ERROR] No origin remote configured!" -ForegroundColor Red
    Write-Host "Run: git remote add origin https://github.com/Elymicyrene/VoxPoser.git"
    exit 1
}

Write-Host "=== VoxPoser -> GitHub Auto Push ===" -ForegroundColor Cyan
Write-Host "Remote: $remote"
Write-Host ""

# ── Proxy selection ────────────────────────────────────────────────────────
# The global git proxy may point to a stale port (e.g. 7890) while the running
# client (Clash Verge) listens elsewhere (e.g. 7897), and a direct connection is
# often reset. Probe common local proxy ports and use the first reachable one;
# if none respond, fall back to a direct connection.
function Test-LocalPort([int]$Port) {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $iar = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if ($iar.AsyncWaitHandle.WaitOne(300, $false) -and $client.Connected) {
            $client.EndConnect($iar)
            $client.Close()
            return $true
        }
        $client.Close()
        return $false
    } catch {
        return $false
    }
}

$proxyPort = $null
foreach ($p in 7897, 7890, 7891, 10809, 10808, 1080, 8080, 8888) {
    if (Test-LocalPort $p) { $proxyPort = $p; break }
}

if ($proxyPort) {
    $proxyUrl = "http://127.0.0.1:$proxyPort"
    $gitArgs = @("-c", "http.proxy=$proxyUrl", "-c", "https.proxy=$proxyUrl")
    Write-Host "Proxy: $proxyUrl (auto-detected)" -ForegroundColor Gray
} else {
    $gitArgs = @("-c", "http.proxy=", "-c", "https.proxy=")
    Write-Host "Proxy: none (direct)" -ForegroundColor Gray
}

# ── API Key Protection ─────────────────────────────────────────────────────
# Replace any hardcoded API keys in tracked files with a placeholder reminder
# BEFORE staging, then RESTORE the real key locally AFTER pushing.
# This prevents credential leakage to GitHub while keeping local use intact.
#
# The real key is stored in the gitignored file ".api_key" (single line).
# It is NEVER hardcoded in any tracked file.

$API_KEY_FILE = Join-Path $PSScriptRoot ".api_key"
if (-not (Test-Path $API_KEY_FILE)) {
    Write-Host "[ERROR] .api_key file not found! Create it with your real DeepSeek key (one line)." -ForegroundColor Red
    exit 1
}
$REAL_API_KEY = (Get-Content -LiteralPath $API_KEY_FILE -Raw -Encoding UTF8).Trim()
$PLACEHOLDER = "YOUR_DEEPSEEK_API_KEY_HERE  Set OPENAI_API_KEY env var"

# Collect ALL tracked files that contain the real key
$trackedFiles = git ls-files
$modifiedFiles = @()
foreach ($f in $trackedFiles) {
    if (Test-Path $f) {
        $content = Get-Content -LiteralPath $f -Raw -Encoding UTF8
        if ($content -and $content.Contains($REAL_API_KEY)) {
            $modifiedFiles += $f
            $newContent = $content.Replace($REAL_API_KEY, $PLACEHOLDER)
            Set-Content -LiteralPath $f -Value $newContent -Encoding UTF8 -NoNewline
            Write-Host "[SECURE] Masked API key in: $f" -ForegroundColor Yellow
        }
    }
}

try {
    # 1. Show status
    Write-Host "--- Changes ---" -ForegroundColor Yellow
    $status = git status --short
    if ($status) {
        Write-Host $status
    } else {
        Write-Host "  (no changes)" -ForegroundColor Gray
    }
    Write-Host ""

    # 2. Add all changes
    Write-Host "--- Staging ---" -ForegroundColor Yellow
    git add -A
    $staged = git diff --cached --stat
    if ($staged) {
        Write-Host $staged
        Write-Host ""

        # 3. Commit
        $hasChanges = git diff --cached --quiet
        if ($LASTEXITCODE -eq 0) {
            Write-Host "No new changes to commit" -ForegroundColor Gray
        } else {
            $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            Write-Host "--- Commit ---" -ForegroundColor Yellow
            git commit -m "auto: update at $timestamp"
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[ERROR] Commit failed!" -ForegroundColor Red
                exit 1
            }
        }
    }

    # 4. Push to GitHub
    Write-Host "--- Pushing to GitHub ---" -ForegroundColor Yellow
    & git @gitArgs push -u origin main 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "SUCCESS! Pushed to GitHub." -ForegroundColor Green
        Write-Host "  URL: https://github.com/Elymicyrene/VoxPoser"
    } else {
        Write-Host "[ERROR] Push failed!" -ForegroundColor Red
        Write-Host "Try running manually or check your network."
        exit 1
    }
}
finally {
    # ── Restore real API key locally so the project still works ─────────
    foreach ($f in $modifiedFiles) {
        if (Test-Path $f) {
            $content = Get-Content -LiteralPath $f -Raw -Encoding UTF8
            if ($content -and $content.Contains($PLACEHOLDER)) {
                $restored = $content.Replace($PLACEHOLDER, $REAL_API_KEY)
                Set-Content -LiteralPath $f -Value $restored -Encoding UTF8 -NoNewline
                Write-Host "[RESTORE] API key restored locally in: $f" -ForegroundColor Green
            }
        }
    }
}
