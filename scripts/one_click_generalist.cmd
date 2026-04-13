@echo off
setlocal
if "%~1"=="" (
  powershell -ExecutionPolicy Bypass -File "%~dp0one_click_generalist.ps1" -RunTests
) else (
  powershell -ExecutionPolicy Bypass -File "%~dp0one_click_generalist.ps1" %*
)
exit /b %ERRORLEVEL%
