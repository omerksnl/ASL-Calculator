@echo off
echo ============================================================
echo Starting Federated Learning Client (PC)
echo ============================================================
echo.

REM Activate virtual environment
call venv311\Scripts\activate.bat

echo Choose experiment type:
echo   1^) IID (balanced data)
echo   2^) Non-IID (skewed data)
set /p EXPERIMENT="Choice [1-2]: "

if "%EXPERIMENT%"=="1" (
    set "DATA_DIR=data/master_dataset/master_iid/client1"
    echo Starting IID Client 1 on PC...
) else (
    set "DATA_DIR=data/master_dataset/master_noniid/client1"
    echo Starting Non-IID Client 1 on PC - digits...
)

echo.
echo Connecting to localhost:8080
echo Data: %DATA_DIR%
echo.

REM Start client
python src/client.py --server localhost:8080 --data-dir %DATA_DIR%

pause

