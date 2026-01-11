@echo off
echo Running Satellite Examples...
echo.

echo [Example 1] Solving Simple 3-SAT (Advanced-CNF)
..\target\release\satellite.exe solve simple_3sat.json
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Please calculate release build first: cargo build --release --workspace
    ..\target\debug\satellite.exe solve simple_3sat.json
)
echo.

echo [Example 2] Solving Pigeonhole (DIMACS)
..\target\debug\satellite.exe solve pigeonhole.cnf
echo.

echo [Example 3] Batch Mode (Simulated)
echo Running batch solver on current directory...
..\target\debug\satellite.exe batch --input-dir . --output-dir results --workers 2
echo.

echo Done.
pause
