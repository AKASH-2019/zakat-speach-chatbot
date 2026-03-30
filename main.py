"""
main.py  (fixed)
----------------
"""

import os
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import app_state
from rag_pipeline import generate_answer
from voice_pipeline import process_voice_query, text_to_speech

# ── Paths ─────────────────────────────────────────────────────────────────────
# All paths are relative to the working directory (where you run uvicorn from).
# In Colab Cell 10 that is the folder where you uploaded your files.
# Do NOT use absolute Google Drive paths here — they break when Colab restarts.

TTS_DIR = Path("./tts_output")
TTS_DIR.mkdir(exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Voice RAG Assistant",
    description="English-only RAG Assistant with Voice + MP3 upload",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State ─────────────────────────────────────────────────────────────────────

conversation_history: list[dict] = []
MAX_HISTORY = 50

# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_models():
    if app_state.vectorstore is None or app_state.llm_pipeline is None:
        raise HTTPException(503, detail="Models not loaded. Run Colab Cell 8 first.")

def _add_to_history(role: str, content: str):
    conversation_history.append({
        "id"     : str(uuid.uuid4())[:8],
        "role"   : role,
        "content": content,
        "time"   : time.strftime("%H:%M:%S"),
    })
    while len(conversation_history) > MAX_HISTORY:
        conversation_history.pop(0)

def _build_response(
    query: str,
    rag_result: dict,
    tts_path: str | None,
    language: str = "en",   # BUG FIX: was missing — caused TypeError in upload_audio
) -> dict:
    tts_url = None
    if tts_path:
        # Return only the filename part so the URL is always /tts/<filename>
        tts_url = f"/tts/{Path(tts_path).name}"

    return {
        "query"     : query,
        "answer"    : rag_result["answer"],
        "found"     : rag_result["found"],
        "confidence": rag_result["confidence"],
        "sources"   : [
            {
                "label"  : f"[{s['index']}] {s['file']}  p.{s['page']}",
                "snippet": s["snippet"],
                "score"  : s["score"],
            }
            for s in rag_result.get("sources", [])
        ],
        "language"  : language,
        "tts_url"   : tts_url,
    }

# ── Request models ────────────────────────────────────────────────────────────

class TextQueryRequest(BaseModel):
    query   : str
    tts     : Optional[bool] = True
    language: Optional[str]  = "en"

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def serve_ui():
    index = Path("index.html")
    if not index.exists():
        raise HTTPException(404, detail="index.html not found. Upload it to the same folder as main.py.")
    return FileResponse(str(index), media_type="text/html")


@app.get("/health")
def health():
    return {
        "status"    : "ok",
        "vectorstore": app_state.vectorstore  is not None,
        "llm"       : app_state.llm_pipeline  is not None,
        "whisper"   : app_state.whisper_model is not None,
        "history"   : len(conversation_history),
    }


@app.post("/ask")
def ask(request: TextQueryRequest):
    _check_models()

    query = request.query.strip()
    if not query:
        raise HTTPException(400, detail="Query must not be empty.")

    _add_to_history("user", query)

    rag_result = generate_answer(query, app_state.vectorstore, app_state.llm_pipeline)

    tts_path = None
    if request.tts and rag_result["found"]:
        try:
            # BUG FIX: pass bare filename, not full path
            tts_filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"
            tts_path = text_to_speech(
                rag_result["answer"],
                language=request.language,
                output_filename=tts_filename,
            )
        except Exception as e:
            print(f"[TTS warning] {e}")

    _add_to_history("assistant", rag_result["answer"])

    return JSONResponse(content=_build_response(query, rag_result, tts_path, request.language))


@app.post("/upload-audio")
async def upload_audio(
    audio   : UploadFile = File(...),
    language: str = Form("en"),
    tts     : str = Form("true"),
):
    _check_models()
    if app_state.whisper_model is None:
        raise HTTPException(503, detail="Whisper model not loaded.")

    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise ValueError("Uploaded audio file is empty.")

        suffix      = Path(audio.filename or "audio.webm").suffix.lower() or ".webm"
        force_lang  = None if language in ("auto", "") else language

        # BUG FIX: pass audio_suffix and force_language — these params were
        # missing from the old voice_pipeline.process_voice_query signature
        stt_result = process_voice_query(
            audio_bytes,
            app_state.whisper_model,
            audio_suffix=suffix,
            force_language=force_lang,
        )

        if stt_result["error"]:
            raise ValueError(f"Transcription failed: {stt_result['error']}")

        query         = stt_result["query"]
        detected_lang = stt_result["language"]

        _add_to_history("user", query)

        rag_result = generate_answer(query, app_state.vectorstore, app_state.llm_pipeline)

        tts_path = None
        if tts.lower() == "true" and rag_result["found"]:
            try:
                # BUG FIX: bare filename only
                tts_filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"
                tts_path = text_to_speech(
                    rag_result["answer"],
                    language=detected_lang,
                    output_filename=tts_filename,
                )
            except Exception as e:
                print(f"[TTS warning] {e}")

        _add_to_history("assistant", rag_result["answer"])

        # BUG FIX: pass detected_lang as 4th arg — _build_response now accepts it
        response = _build_response(query, rag_result, tts_path, detected_lang)
        response.update({
            "transcript"    : query,
            "detected_lang" : detected_lang,
            "audio_duration": stt_result["duration_sec"],
            "stt_time"      : stt_result["processing_sec"],
        })
        return JSONResponse(content=response)

    except Exception as e:
        # BUG FIX: return the real error message, not just a generic 500
        print(f"[upload-audio ERROR] {e}")
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/tts/{filename}")
def serve_tts(filename: str):
    safe_name = Path(filename).name   # strip any path traversal
    if not safe_name.endswith(".mp3"):
        raise HTTPException(400, detail="Invalid filename.")
    file_path = TTS_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(404, detail="TTS file not found.")
    return FileResponse(str(file_path), media_type="audio/mpeg")


@app.post("/ask-mcq")
def ask_mcq(request: TextQueryRequest):
    """Bonus: generate an MCQ from the RAG answer."""
    _check_models()

    query = request.query.strip()
    if not query:
        raise HTTPException(400, detail="Empty query.")

    rag_result = generate_answer(query, app_state.vectorstore, app_state.llm_pipeline)
    if not rag_result["found"]:
        return {"query": query, "answer": rag_result["answer"], "mcq": None}

    mcq_prompt = (
        f"<|system|>\nCreate one multiple-choice question with 1 correct and 3 wrong answers "
        f"based on the following text. Format:\nQ: ...\nA) ...\nB) ...\nC) ...\nD) ...\nAnswer: ...</s>\n"
        f"<|user|>\n{rag_result['answer']}</s>\n<|assistant|>\n"
    )
    raw = app_state.llm_pipeline(mcq_prompt)
    mcq = raw[0].get("generated_text", "") if isinstance(raw, list) else ""

    return {"query": query, "answer": rag_result["answer"], "mcq": mcq}


@app.get("/history")
def get_history():
    return {"history": conversation_history, "count": len(conversation_history)}


@app.delete("/history")
def clear_history():
    conversation_history.clear()
    return {"status": "cleared", "count": 0}
