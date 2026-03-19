@echo off
cd /d "%~dp0"

REM Path to spark-submit (adjust if needed)
set "SPARK_SUBMIT=C:\Spark\spark-4.1.1-bin-hadoop3\bin\spark-submit.cmd"
set "SPARK_SUBMIT=%SPARK_SUBMIT:"=%"

REM Detect the real Python executable via the Windows 'py' launcher (force Python 3.10)
for /f "delims=" %%P in ('py -3.10 -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%P"
if not defined PYTHON_EXE (
  echo ERROR: Could not detect Python 3.10 using the 'py' launcher.
  echo Fix: ensure "py -3.10 --version" works in CMD.
  pause
  exit /b 1
)

REM Tell PySpark to use that Python instead of the default "python3"
set "PYSPARK_PYTHON=%PYTHON_EXE%"
set "PYSPARK_DRIVER_PYTHON=%PYTHON_EXE%"

if not exist "%SPARK_SUBMIT%" (
  echo ERROR: %SPARK_SUBMIT% not found.
  echo Fix: update SPARK_SUBMIT in run.bat to your spark-submit.cmd path.
  pause
  exit /b 1
)

echo SPARK_SUBMIT=%SPARK_SUBMIT%
echo PYTHON_EXE=%PYTHON_EXE%
echo Starting PySpark server...
set "SPARK_ARGS=--conf spark.pyspark.driver.python=%PYTHON_EXE% --conf spark.pyspark.python=%PYTHON_EXE% --conf spark.python.worker.faulthandler.enabled=true --conf spark.sql.execution.pyspark.udf.faulthandler.enabled=true --conf spark.sql.execution.arrow.pyspark.enabled=false --conf spark.python.use.daemon=false --conf spark.python.worker.reuse=false spark\pyspark_server.py"
echo Running: "%SPARK_SUBMIT%" %SPARK_ARGS%
start "PySpark Genre Server" "%ComSpec%" /k ""%SPARK_SUBMIT%" %SPARK_ARGS%"

echo Waiting for backend health...
set "URL=http://127.0.0.1:8081/api/health"
for /l %%i in (1,1,60) do (
  powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 '%URL%'; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
  if %errorlevel%==0 goto OPEN
  powershell -NoProfile -Command "Start-Sleep -Seconds 1"
)

echo Backend did not become ready in time.
echo You can still open http://127.0.0.1:8081/ after Spark finishes starting.
goto :EOF

:OPEN
echo Opening browser at http://127.0.0.1:8081/ ...
start "" "http://127.0.0.1:8081/"

echo If the page is blank, wait a bit and refresh.
