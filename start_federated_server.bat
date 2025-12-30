@echo off
echo ============================================================
echo Starting Federated Learning Server
echo ============================================================
echo.

REM Activate virtual environment
call venv311\Scripts\activate.bat

REM Get local IP
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    set IP=%%a
    goto :found
)
:found

echo Your PC IP: %IP%
echo.
echo IMPORTANT: Both clients run on Raspberry Pi
echo.
echo Pi Terminal 1: python src/client.py --server %IP:~1%:8080 --data-dir data/master_dataset/master_iid/client1
echo Pi Terminal 2: python src/client.py --server %IP:~1%:8080 --data-dir data/master_dataset/master_iid/client2
echo.
echo ============================================================
echo.

REM Start server (waits for 2 Pi clients)
python src/server.py --host 0.0.0.0 --port 8080 --rounds 20 --min-clients 2

pause

