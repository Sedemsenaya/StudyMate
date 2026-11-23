@echo off
rem StudyMate CPU run: interview prep (Python PDF) with the small qwen3:1.7b model.
rem Double-click to run. Stop the old run (Ctrl+C in its window) first.
cd /d C:\Users\Colli\PycharmProjects\StudyMate
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OLLAMA_NUM_PARALLEL=1
call .venv\Scripts\activate.bat

echo.
echo === 1/4  Unloading qwen3:8b to free RAM
ollama stop qwen3:8b

echo.
echo === 2/4  Downloading qwen3:1.7b (about 1.4 GB, only the first time)
ollama pull qwen3:1.7b
if errorlevel 1 (
    echo Could not download the model. Is Ollama running?
    pause
    exit /b 1
)

echo.
echo === 3/4  Speed test: look at "eval rate" below (20+ tokens/s is good)
ollama run qwen3:1.7b "Explain a hash map in 3 sentences." --verbose

echo.
echo === 4/4  Making the videos (one AI call at a time, 300 s limit)
python main_v4.py uploads\DSA_20_Patterns_Python.pdf --model qwen3:1.7b --llm-parallel 1 --ai-timeout 300

echo.
echo Done. Videos are in Engine\Output
pause
