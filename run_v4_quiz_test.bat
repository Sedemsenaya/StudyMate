@echo off
rem StudyMate v4: AI illustration test on the first 10 practice-test questions (CPU settings)
cd /d C:\Users\Colli\PycharmProjects\StudyMate
set PYTHONUTF8=1
set OLLAMA_NUM_PARALLEL=1
call .venv\Scripts\activate.bat
python main_v4.py "uploads\Practice_Test#3-AWS_Certified_Solutions_Architect_Associate-Results.pdf" --questions 1-10 --llm-parallel 1
