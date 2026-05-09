# First-time push to https://github.com/siddhartha4u2c/Decision_tree
# Run from the project root in PowerShell: .\push-to-github.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$remote = "https://github.com/siddhartha4u2c/Decision_tree.git"

if (-not (Test-Path .git)) {
    git init
    git branch -M main
    git remote add origin $remote
} else {
    git remote get-url origin 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        git remote add origin $remote
    } else {
        git remote set-url origin $remote
    }
}

git add .
git status
$msg = "Initial commit: Streamlit decision tree demo (Gini + entropy)"
$pending = git status --porcelain
if ($pending) {
    git commit -m $msg
} else {
    Write-Host "No changes to commit; pushing current branch."
}
git push -u origin main
