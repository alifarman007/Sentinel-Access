@echo off
title Face Re-ID Access Management System
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python main.py
pause
