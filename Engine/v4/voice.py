"""
StudyMate v4 - narration audio (local, offline) with a cache.

  piper   (default) fast on CPU, ~0.1-0.3x real time.  pip install piper-tts
          voice: en_US-lessac-medium.onnx (+ .onnx.json). v4 looks in
          Engine/voices/, then ~/Downloads, or pass --voice path\\to\\voice.onnx
          get one: python -m piper.download_voices en_US-lessac-medium
  kokoro  nicer voice, slower on CPU.   pip install kokoro soundfile  (+ espeak-ng)
  none    silent video (captions only)

Each text is synthesised once: wav files are cached by a hash of
(engine, voice, speed, text) in Engine/Output/.tts_cache, so re-runs and
repeated slides cost nothing. One voice model is loaded per worker thread.
"""
import hashlib
import subprocess
import threading
import wave
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent


def find_piper_voice(name="en_US-lessac-medium"):
    for d in (ENGINE_DIR / "voices", Path.home() / "Downloads", Path.cwd()):
        p = d / f"{name}.onnx"
        if p.exists():
            return p
    return None


class Voice:
    def __init__(self, engine="piper", voice=None, speed=1.0, cache_dir=None):
        self.engine = engine
        self.speed = speed
        if engine == "piper":
            self.voice = Path(voice) if voice else find_piper_voice()
            if not self.voice or not Path(self.voice).exists():
                raise SystemExit("Piper voice not found. Put en_US-lessac-medium.onnx (+ .onnx.json) "
                                 "in Engine/voices or pass --voice. "
                                 "Get it: python -m piper.download_voices en_US-lessac-medium")
        else:
            self.voice = voice or "af_heart"
        self.cache = Path(cache_dir or ENGINE_DIR / "Output" / ".tts_cache")
        self.cache.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()

    def _model(self):
        m = getattr(self._local, "m", None)
        if m is None:
            if self.engine == "piper":
                from piper import PiperVoice
                m = PiperVoice.load(str(self.voice))
            elif self.engine == "kokoro":
                from kokoro import KPipeline
                m = KPipeline(lang_code="a")
            self._local.m = m
        return m

    def key(self, text):
        h = hashlib.sha1(f"{self.engine}|{self.voice}|{self.speed}|{text}".encode()).hexdigest()
        return self.cache / f"{h[:20]}.wav"

    def say(self, text, out_wav=None):
        """text -> wav path (cached). Returns (path, seconds, cache_hit)."""
        if self.engine == "none" or not text.strip():
            return None, 0.0, True
        wav = self.key(text)
        hit = wav.exists() and wav.stat().st_size > 1000
        if not hit:
            tmp = wav.with_suffix(f".{threading.get_ident()}.tmp")
            if self.engine == "piper":
                from piper import SynthesisConfig
                cfg = SynthesisConfig(length_scale=1.0 / self.speed)
                m = self._model()
                rate = int(getattr(getattr(m, "config", None), "sample_rate", 22050) or 22050)
                say = speakable(text)
                try:
                    with wave.open(str(tmp), "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(rate)
                        if say:
                            m.synthesize_wav(say, w, syn_config=cfg, set_wav_format=False)
                        else:
                            w.writeframes(b"\x00\x00" * int(rate * 0.4))
                except TypeError:                      # older piper without set_wav_format
                    with wave.open(str(tmp), "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(rate)
                        if say:
                            m.synthesize_wav(say, w, syn_config=cfg)
                        if w.getnframes() == 0:
                            w.writeframes(b"\x00\x00" * int(rate * 0.4))
            elif self.engine == "kokoro":
                import numpy as np
                import soundfile as sf
                chunks = [a for _, _, a in self._model()(text, voice=self.voice, speed=self.speed)]
                sf.write(str(tmp), np.concatenate(chunks), 24000, format="WAV")
            tmp.replace(wav)
        return wav, duration(wav), hit


SAY = {"≈": " about ", "≤": " at most ", "≥": " at least ", "≠": " not equal to ", "→": " to ", "←": " from ",
       "×": " times ", "÷": " divided by ", "√": " square root of ", "π": " pi ", "∞": " infinity ", "Σ": " sum ",
       "θ": " theta ", "λ": " lambda ", "μ": " mu ", "σ": " sigma ", "Δ": " change in ", "°": " degrees ",
       "²": " squared ", "³": " cubed ", "±": " plus or minus ", "%": " percent", "&": " and ", "=": " equals ", "+": " plus "}


def speakable(text):
    """Text a TTS voice can say: symbols spelled out, emoji and stray marks dropped."""
    s = str(text)
    for k, v in SAY.items():
        s = s.replace(k, v)
    s = "".join(ch if ch.isalnum() or ch in " .,;:!?'\"-()/\n" else " " for ch in s)
    s = " ".join(s.split())
    return s if any(c.isalpha() for c in s) else ""


def duration(wav):
    try:
        with wave.open(str(wav)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(wav)], capture_output=True, text=True)
        return float(out.stdout.strip() or 0)
