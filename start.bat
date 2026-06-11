@echo off
echo 🚀 Starting MetaShield Production Server
echo ========================================

REM Check if model exists
if not exist "checkpoints\best_model.pt" (
    echo ⚠️ No trained model found. Training first...
    python main.py
)

REM Start API server
uvicorn api:app --host 0.0.0.0 --port 5000 --reload --log-level info
pause