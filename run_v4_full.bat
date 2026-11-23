@echo off
rem StudyMate v4 full run: the whole Maarek deck
cd /d C:\Users\Colli\PycharmProjects\StudyMate
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OLLAMA_NUM_PARALLEL=2
call .venv\Scripts\activate.bat
echo [%date% %time%] setup start > Engine\Output\v4_full_console.log
python -m pip install -q pypdfium2 piper-tts imageio-ffmpeg >> Engine\Output\v4_full_console.log 2>&1
python -m playwright install chromium >> Engine\Output\v4_full_console.log 2>&1
echo [%date% %time%] run start >> Engine\Output\v4_full_console.log
python main_v4.py "C:\Users\Colli\Downloads\AWS Certified Solutions Architect Slides v49 (reduced).pdf"  >> Engine\Output\v4_full_console.log 2>&1
echo [%date% %time%] run end, exit code %errorlevel% >> Engine\Output\v4_full_console.log
