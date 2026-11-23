"""
StudyMate v4: PDFs (100+ at a time) -> narrated, hand-drawn, animated lesson videos.

    python main_v4.py "C:\\Users\\Colli\\Downloads\\AWS Certified Solutions Architect Slides v49 (reduced).pdf"
    python main_v4.py "C:\\Users\\Colli\\Downloads\\AWS Certified Solutions Architect Slides v49 (reduced).pdf" --pages 194-267
    python main_v4.py uploads\\ --planner llm --model qwen3:8b --llm-parallel 2
    python main_v4.py some.pdf --tts none                 # captions only, fastest

See Engine/v4/README_v4.md for every option and the log files.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "Engine" / "v4"))

from batch import main  # noqa: E402

if __name__ == "__main__":
    main()
