# Pull lap timing for races that have run since the last pull, test, and push.
#
# Run this from a home connection, not CI: F1's timing archive needs an F1 TV
# account, and it blocks datacenter IPs (GitHub's runners included) outright.
# The weekly GitHub job keeps results, sprints and pre-registration current;
# this keeps the lap datasets current (strategy, degradation, safety car, pit
# loss).
#
# FastF1 asks for your F1 TV email and password the first time, in this
# terminal, and stores only the resulting token, which it reuses until it
# expires. Races already pulled are skipped, so a run after each race weekend
# takes a few minutes.
#
#     .\scripts\refresh_laps.ps1

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = ".venv\Scripts\python.exe"
$year = (Get-Date).ToUniversalTime().Year

git pull --rebase --autostash
if ($LASTEXITCODE) { throw "git pull failed" }

& $py -m f1se.data.ingest $year
if ($LASTEXITCODE) { throw "lap ingest failed" }
& $py -m f1se.data.ingest status $year
if ($LASTEXITCODE) { throw "track-status ingest failed" }
& $py -m f1se.data.ingest racelaps $year
if ($LASTEXITCODE) { throw "race-laps ingest failed" }
# Practice long runs: the engine's wear evidence for a track before its race
# (METHODOLOGY section 22). Run this script again after Friday practice to use them.
& $py analysis/practice_longruns.py $year
if ($LASTEXITCODE) { throw "practice pull failed" }
& $py -m f1se.standalone.results $year
if ($LASTEXITCODE) { throw "results pull failed" }

& $py -m pytest -q
if ($LASTEXITCODE) { throw "tests failed; nothing committed" }

git add data/processed/*.parquet predictions/
git diff --cached --quiet
if (-not $LASTEXITCODE) { Write-Host "no new data"; exit 0 }
git commit -m "Data: lap timing refresh $(Get-Date -Format yyyy-MM-dd)"
git push
