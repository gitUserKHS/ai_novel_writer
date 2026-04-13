@echo off
setlocal
if "%~1"=="" (
  powershell -ExecutionPolicy Bypass -File "%~dp0one_click_loop.ps1" -RunTests
) else (
  powershell -ExecutionPolicy Bypass -File "%~dp0one_click_loop.ps1" %*
)
exit /b %ERRORLEVEL%
