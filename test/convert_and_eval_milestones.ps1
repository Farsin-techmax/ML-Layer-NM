# convert_and_eval_milestones.ps1
# Automates the end-to-end candidate-mode conversion and evaluation for milestones 50k to 100k
# (or any subset specified via parameters).

[CmdletBinding()]
param (
    [Parameter(Position = 0)]
    [int[]]$Milestones = @(50, 60, 70, 80, 90, 100),

    [string]$PythonExe = "venv\Scripts\python.exe",
    [string]$TrainFrom = "2023",
    [string]$RunTag = "_cand_wins_2023",
    [int]$GraceDays = 15,

    [ValidateSet("All", "FeatureEng", "TestSets", "Train", "Eval")]
    [string]$Step = "All"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python executable not found at '$PythonExe'. Please verify virtual environment."
    exit 1
}

$startTime = Get-Date

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "CANDIDATE-MODE PIPELINE RUNNER (50k - 100k)" -ForegroundColor Cyan
Write-Host "Milestones : $($Milestones -join ', ')k" -ForegroundColor Cyan
Write-Host "Step Mode  : $Step" -ForegroundColor Cyan
Write-Host "Train From : $TrainFrom" -ForegroundColor Cyan
Write-Host "Run Tag    : $RunTag" -ForegroundColor Cyan
Write-Host "Grace Days : $GraceDays" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

foreach ($m in $Milestones) {
    $mStartTime = Get-Date
    Write-Host "`n>>> Processing Milestone: ${m}k <<<" -ForegroundColor Green

    # -------------------------------------------------------------
    # 1. Feature Engineering (Candidate Matrix Generation)
    # -------------------------------------------------------------
    if ($Step -in @("All", "FeatureEng")) {
        Write-Host "`n[Step 1/4] Building candidate training matrix for ${m}k..." -ForegroundColor Yellow
        $cmd = "$PythonExe pmstrainfeatureeng_refactored.py 2026 Q1 $m --label-mode candidates --grace-days $GraceDays"
        Write-Host "Running: $cmd" -ForegroundColor Gray
        & $PythonExe pmstrainfeatureeng_refactored.py 2026 Q1 $m --label-mode candidates --grace-days $GraceDays
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Feature engineering failed for ${m}k (exit code $LASTEXITCODE)."
            exit $LASTEXITCODE
        }
    }

    # -------------------------------------------------------------
    # 2. Test Set Generation (Q1 & Q2 Candidate Test Cohorts)
    # -------------------------------------------------------------
    if ($Step -in @("All", "TestSets")) {
        Write-Host "`n[Step 2/4] Building candidate test sets (Q1 and Q2) for ${m}k..." -ForegroundColor Yellow
        
        Write-Host "Generating Q1 2026 test set..." -ForegroundColor Gray
        & $PythonExe prepare_test_set.py `
            --milestone $m `
            --window-start 2026-01-01 `
            --window-end 2026-03-31 `
            --train-cutoff 2025-12-31 `
            --label-mode candidates `
            --grace-days $GraceDays
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to build Q1 candidate test set for ${m}k."
            exit $LASTEXITCODE
        }

        Write-Host "Generating Q2 2026 test set..." -ForegroundColor Gray
        & $PythonExe prepare_test_set.py `
            --milestone $m `
            --window-start 2026-04-01 `
            --window-end 2026-06-30 `
            --train-cutoff 2025-12-31 `
            --label-mode candidates `
            --grace-days $GraceDays
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to build Q2 candidate test set for ${m}k."
            exit $LASTEXITCODE
        }
    }

    # -------------------------------------------------------------
    # 3. Model Retraining (retrain.py)
    # -------------------------------------------------------------
    if ($Step -in @("All", "Train")) {
        Write-Host "`n[Step 3/4] Retraining candidate model for ${m}k..." -ForegroundColor Yellow
        $env:PMS_LABEL_MODE = "candidates"
        $env:PMS_TRAIN_FROM = $TrainFrom
        $env:PMS_RUN_TAG = $RunTag

        Write-Host "Environment set: PMS_LABEL_MODE=$env:PMS_LABEL_MODE, PMS_TRAIN_FROM=$env:PMS_TRAIN_FROM, PMS_RUN_TAG=$env:PMS_RUN_TAG" -ForegroundColor Gray
        & $PythonExe retrain.py $m
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Model training failed for ${m}k."
            exit $LASTEXITCODE
        }
    }

    # -------------------------------------------------------------
    # 4. Clean Evaluation on Expected Cohorts (No Early Completers)
    # -------------------------------------------------------------
    if ($Step -in @("All", "Eval")) {
        Write-Host "`n[Step 4/4] Evaluating expected cohorts for ${m}k (early completers excluded)..." -ForegroundColor Yellow
        $modelDir = "models/${m}k$RunTag"
        $featFile = "models/selected_features_${m}k$RunTag.json"

        $evalScript = if (Test-Path "test/eval_expected_cohort.py") { "test/eval_expected_cohort.py" } elseif (Test-Path "eval_expected_cohort.py") { "eval_expected_cohort.py" } else { Join-Path $PSScriptRoot "eval_expected_cohort.py" }

        Write-Host "Evaluating Q1 2026 Expected Cohort..." -ForegroundColor Gray
        & $PythonExe $evalScript $m `
            --quarter Q1 `
            --model-dir $modelDir `
            --features $featFile
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Q1 evaluation failed for ${m}k."
            exit $LASTEXITCODE
        }

        Write-Host "Evaluating Q2 2026 Expected Cohort..." -ForegroundColor Gray
        & $PythonExe $evalScript $m `
            --quarter Q2 `
            --model-dir $modelDir `
            --features $featFile
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Q2 evaluation failed for ${m}k."
            exit $LASTEXITCODE
        }
    }

    $mDuration = (Get-Date) - $mStartTime
    Write-Host "`nCompleted Milestone ${m}k in $($mDuration.ToString('mm\:ss'))" -ForegroundColor Green
}

$totalDuration = (Get-Date) - $startTime
Write-Host "`n=================================================================" -ForegroundColor Cyan
Write-Host "ALL SPECIFIED MILESTONES COMPLETED IN $($totalDuration.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
