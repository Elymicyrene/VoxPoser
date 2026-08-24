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

# Override global proxy for GitHub access
$gitArgs = @("-c", "http.proxy=", "-c", "https.proxy=")

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
