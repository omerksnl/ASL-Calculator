@echo off
echo ============================================================
echo ASL Calculator App
echo ============================================================
echo.

REM Activate virtual environment
call venv311\Scripts\activate.bat

REM Start calculator
python asl_calculator_app.py

pause

