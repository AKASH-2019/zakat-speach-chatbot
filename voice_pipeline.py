"""
voice_pipeline.py  (fixed)
--------------------------
English-only voice pipeline.

BUGS FIXED:
1. process_voice_query() now accepts audio_suffix and force_language kwargs
   so main.py can call it without a TypeError.
2. text_to_speech() now accepts a language param (ignored for English but
   stops main.py from crashing with an unexpected keyword argument).
3. TTS output_filename is now treated as a bare filename, not a full path,
   so the final path is always TTS_OUTPUT_DIR/filename with no duplication.
"""

import os
import re
import tempfile
import time
from pathlib import Path

import ffmpeg
import torch
import whisper
from gtts import gTTS

# ── Constants ─────────────────────────────────────────────────────────────────

SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}
DEFAULT_LANGUAGE = "en"

TTS_OUTPUT_DIR = "./tts_output"
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)

NOT_FOUND_MSG = "I don't know based on provided data."

# ── Text normalisation ────────────────────────────────────────────────────────

ALIAS = {
    "nisap" : "nisab",
    "zuckat": "zakat",
    "zakkat": "zakat",
}

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def correct_aliases(text: str) -> str:
    return " ".join(ALIAS.get(w, w) for w in text.split())

# ── Model Loading ─────────────────────────────────────────────────────────────

def load_whisper(model_size: str = "base") -> whisper.Whisper:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper '{model_size}' on {device}...")
    model = whisper.load_model(model_size, device=device)
    print("Whisper ready.")
    return model

# ── Audio Preprocessing ───────────────────────────────────────────────────────

def preprocess_audio(input_path: str, output_path: str) -> str:
    try:
        (
            ffmpeg.input(input_path)
            .output(output_path, ar=16000, ac=1, acodec="pcm_s16le")
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(
            f"ffmpeg preprocessing failed: {e.stderr.decode() if e.stderr else 'unknown'}"
        )
    return output_path

def validate_audio_file(file_path: str) -> None:
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"File not found: {file_path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: {path.suffix}. "
            f"Supported: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}"
        )
    if path.stat().st_size == 0:
        raise ValueError("Audio file is empty (0 bytes).")

# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe(audio_path: str, model: whisper.Whisper, language: str = DEFAULT_LANGUAGE) -> dict:
    validate_audio_file(audio_path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        clean_wav = tmp.name
    try:
        preprocess_audio(audio_path, clean_wav)
        start = time.time()
        result = model.transcribe(
            clean_wav,
            language=language,
            task="transcribe",
            fp16=torch.cuda.is_available(),
            verbose=False,
        )
        elapsed = round(time.time() - start, 2)
    finally:
        if os.path.exists(clean_wav):
            os.remove(clean_wav)

    segments = result.get("segments", [])
    duration = segments[-1]["end"] if segments else 0.0
    text     = result["text"].strip()

    if not text:
        raise ValueError("Whisper returned an empty transcription.")

    # Apply alias correction (e.g. "zakkat" → "zakat")
    text = correct_aliases(text)

    return {
        "text"          : text,
        "language"      : language,
        "language_name" : "English",
        "duration_sec"  : round(duration, 2),
        "segments"      : segments,
        "processing_sec": elapsed,
    }

def transcribe_bytes(
    audio_bytes: bytes,
    model: whisper.Whisper,
    suffix: str = ".webm",
    language: str = DEFAULT_LANGUAGE,
) -> dict:
    """Transcribe raw bytes (from browser MediaRecorder or file upload)."""
    if not audio_bytes:
        raise ValueError("Received empty audio data.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        return transcribe(tmp_path, model, language=language)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ── Text-to-Speech ────────────────────────────────────────────────────────────

def text_to_speech(
    text: str,
    language: str = "en",          # BUG FIX: accept language param (was missing)
    output_filename: str | None = None,
) -> str:
    """
    Convert text to speech with gTTS. Returns the path to the saved .mp3.

    BUG FIX: output_filename must be a bare filename (e.g. "tts_abc.mp3"),
    NOT a full path. The function always writes to TTS_OUTPUT_DIR/filename.
    Passing a full path caused double-path concatenation and a FileNotFoundError.
    """
    clean_text = re.sub(r"\[Source \d+\]", "", text).strip()
    if not clean_text:
        raise ValueError("No speakable text after stripping source markers.")

    words = clean_text.split()
    if len(words) > 500:
        clean_text = " ".join(words[:500]) + "..."

    # Always use English TTS (language param is accepted but ignored here)
    gtts_lang = "en"

    if output_filename is None:
        output_filename = f"tts_{int(time.time())}.mp3"

    # BUG FIX: strip any directory component — filename only
    safe_filename = Path(output_filename).name
    output_path   = os.path.join(TTS_OUTPUT_DIR, safe_filename)

    tts = gTTS(text=clean_text, lang=gtts_lang, slow=False)
    tts.save(output_path)
    return output_path

# ── Convenience Wrappers ──────────────────────────────────────────────────────

def process_voice_query(
    audio_bytes: bytes,
    whisper_model: whisper.Whisper,
    audio_suffix: str = ".webm",        # BUG FIX: added missing param
    force_language: str | None = None,  # BUG FIX: added missing param
) -> dict:
    """
    End-to-end: bytes → transcription dict.
    Called by main.py's /upload-audio endpoint.

    BUG FIX: original signature only accepted (audio_bytes, whisper_model).
    main.py passes audio_suffix and force_language — those caused TypeError.
    """
    lang = force_language or DEFAULT_LANGUAGE
    try:
        result = transcribe_bytes(audio_bytes, whisper_model, suffix=audio_suffix, language=lang)
        return {
            "query"          : result["text"],
            "language"       : result["language"],
            "language_name"  : result["language_name"],
            "duration_sec"   : result["duration_sec"],
            "processing_sec" : result["processing_sec"],
            "error"          : None,
        }
    except Exception as e:
        return {
            "query"          : "",
            "language"       : "unknown",
            "language_name"  : "Unknown",
            "duration_sec"   : 0.0,
            "processing_sec" : 0.0,
            "error"          : str(e),
        }

def process_audio_file(audio_path: str, whisper_model: whisper.Whisper) -> dict:
    """Transcribe from a file path (helper for CLI testing)."""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    return process_voice_query(audio_bytes, whisper_model)

# ── CLI Test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python voice_pipeline.py <audio_file>")
        sys.exit(1)

    model  = load_whisper("base")
    result = process_audio_file(sys.argv[1], model)

    print("\n" + "=" * 52)
    print(f"Language   : {result['language_name']} ({result['language']})")
    print(f"Duration   : {result['duration_sec']}s")
    print(f"Processed  : {result['processing_sec']}s")
    print(f"Transcript : {result['query']}")
