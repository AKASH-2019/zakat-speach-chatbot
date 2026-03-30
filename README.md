# 🕌 Zakat Voice RAG Assistant

> English-only RAG assistant with voice input/output.  
> Built with **FastAPI**, **LangChain**, **Whisper**, and **gTTS**, with cloud access via **ngrok**.
> Drive link - https://drive.google.com/drive/folders/1zBzP2r8NNGbNsQYohUUTzfLoZOz7pS05?usp=sharing
> video: https://drive.google.com/file/d/1Uhob-l-Lf7KkmTogWxpa-5bCdH6dynij/view?usp=sharing
---

## 📂 Project Structure

```
Voice-RAG-Assistant/
│
├── main.py                 # FastAPI server: text/audio query endpoints, TTS serving
├── voice_pipeline.py       # Audio processing, Whisper transcription, gTTS text-to-speech
├── rag_pipeline.py         # RAG pipeline: embeddings, FAISS vectorstore, LLM answer generation
├── app_state.py            # Global shared state (vectorstore, LLM, Whisper)
├── tts_output/             # Generated .mp3 files from TTS
├── docs/                   # Knowledge base PDFs / TXT files for RAG
├── faiss_index/            # Prebuilt FAISS vectorstore (optional, for reuse)
├── env                     # Environment variables (HF_TOKEN, NGROK_TOKEN)
└── index.html              # Optional frontend UI
```

---

## ⚙️ Setup Instructions

### 1. Enable GPU

* Go to **Colab → Runtime → Change runtime type → GPU (T4)**
* Verify GPU availability:

```python
import torch
print(torch.cuda.is_available())  # should be True
```

---

### 2. Install Dependencies

* Install required Python packages:

```python
!pip install \
  langchain==0.1.16 \
  langchain-community==0.0.34 \
  faiss-cpu \
  sentence-transformers \
  transformers \
  bitsandbytes \
  accelerate \
  openai-whisper \
  ffmpeg-python \
  fastapi uvicorn python-multipart aiofiles \
  gtts \
  pyngrok nest-asyncio \
  huggingface_hub
```

* Install system `ffmpeg` for audio processing:

```python
!apt-get install -y ffmpeg
```

---

### 3. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/ColabNotebooks/as-sunnah
```

---

### 4. Load Environment Variables

* Your `env` file should contain:

```
HF_TOKEN=<your_HuggingFace_token>
NGROK_TOKEN=<your_Ngrok_token>
```

* Load them in Python:

```python
import os
from dotenv import load_dotenv
load_dotenv("env")
```

---

### 5. Authenticate Services

```python
from huggingface_hub import login
from pyngrok import conf

login(os.getenv("HF_TOKEN"))
conf.get_default().auth_token = os.getenv("NGROK_TOKEN")
```

---

### 6. Prepare Knowledge Base

* Place your PDF or TXT files inside the `docs/` folder.
* Build a new FAISS index, or load an existing one:

```python
from rag_pipeline import build_index_from_docs, load_vectorstore

# Build fresh index from docs/
vectorstore = build_index_from_docs("docs")

# OR load a previously saved index
# vectorstore = load_vectorstore("faiss_index")
```

---

### 7. Load Models

* Load **Whisper** for speech-to-text:

```python
from voice_pipeline import load_whisper
whisper_model = load_whisper("base")
```

* Load **TinyLlama** LLM for RAG answer generation:

```python
from rag_pipeline import load_llm
llm_pipeline = load_llm()
```

---

### 8. Inject Shared State

```python
import app_state

app_state.vectorstore  = vectorstore
app_state.llm_pipeline = llm_pipeline
app_state.whisper_model = whisper_model
```

---

### 9. Start FastAPI Server

```python
import nest_asyncio
import uvicorn

nest_asyncio.apply()
uvicorn.run("main:app", host="0.0.0.0", port=8000)
```

---

### 10. Expose Public URL via ngrok

```python
from pyngrok import ngrok

tunnel = ngrok.connect(8000)
print("Public URL:", tunnel.public_url)
```

* Open the printed URL in your browser to access the assistant UI.

---

## 🔄 Request Flow

```
User Input (text / voice / mp3)
        │
        ▼
[ FastAPI — main.py ]
        │
        ├─── Voice/MP3 ──► [ Whisper STT — voice_pipeline.py ]
        │                          │
        │                    Transcribed Text
        │                          │
        └─── Text Query ───────────┘
                    │
                    ▼
        [ RAG Pipeline — rag_pipeline.py ]
                    │
           ┌────────┴────────┐
           ▼                 ▼
    [ FAISS Vector      [ Topic Gate +
      Similarity ]        Confidence
                           Check ]
                    │
                    ▼
             [ TinyLlama LLM ]
                    │
             [ Hallucination
               Detector ]
                    │
                    ▼
              Final Answer
                    │
                    ├──► JSON Response (text)
                    └──► [ gTTS — voice_pipeline.py ] ──► MP3 Audio
```

---

---

## 🧠 `rag_pipeline.py` — RAG Pipeline Reference

> Handles the full RAG pipeline: **Load docs → chunk → embed → retrieve → validate → generate answer**

---

### 📂 Document Processing

#### `load_documents(docs_dir)`

* Loads all PDF + TXT files from the given directory
* Adds metadata to every page:
  * `source_file` — original filename
  * `page` — page number
* Returns a list of `Document` objects

#### `chunk_documents(documents)`

* Splits documents into smaller, searchable chunks
* Settings:
  * `CHUNK_SIZE = 600`
  * `CHUNK_OVERLAP = 80`
* Smaller chunks improve retrieval accuracy

---

### 🧠 Embedding & Index

#### `_make_embeddings()`

* Loads the embedding model:
  * `sentence-transformers/all-MiniLM-L6-v2`
* Uses GPU if available, otherwise CPU
* Normalizes embeddings — required for correct cosine similarity scores

#### `build_vectorstore(chunks, save_path)`

* Converts document chunks → embedding vectors
* Stores them in a FAISS index
* Uses `MAX_INNER_PRODUCT` distance strategy (cosine similarity)
* Saves the index locally for reuse

#### `load_vectorstore(save_path)`

* Loads a previously saved FAISS index from disk
* Ready for similarity search immediately

---

### 🔍 Retrieval

#### `retrieve_chunks(query, vectorstore, top_k)`

* Finds the top-K most relevant chunks for the query
* Returns a list of `(document, score)` tuples
* Prints debug info: score + content preview per chunk

#### `build_context(results)`

* Combines retrieved chunks into a single LLM-ready context string
* Also builds:
  * `sources` — list of `{ file, page, snippet, score }`
  * `top_score` — the highest similarity score across all results
* Output is passed directly to the LLM prompt

---

### 🛑 Hallucination Prevention

#### `_content_words(text)`

* Extracts meaningful words from any text string:
  * Lowercased
  * Stop-words removed
  * Only keeps words with length ≥ 3
* Used as input by both gate functions below

#### `_topic_gate(query, context_text)`

> 👉 Prevents off-topic questions **BEFORE** the LLM is called

* Computes the overlap between query words and context words:

```
query_words ∩ context_words
```

* If overlap is empty → **BLOCK**, return `NOT_FOUND` immediately
* Examples:
  * ❌ `"Do you know about Donald Trump"` — not in docs → blocked
  * ✅ `"What is the nisab for zakat"` — words found in context → allowed

#### `_hallucination_check(query, answer, context_text)`

> 👉 Detects hallucination **AFTER** the LLM generates an answer

* Finds words from the query that have **no grounding** in the retrieved context (`foreign words`)
* If any foreign word appears in the generated answer → hallucination detected → reject
* Example:
  * Query contains: `"trump"`, `"donald"`
  * Context contains: zakat, nisab, gold, silver — no Trump
  * Answer mentions `"trump"` → ❌ rejected

---

### 🤖 LLM

#### `load_llm()`

* Loads `TinyLlama/TinyLlama-1.1B-Chat-v1.0` in **4-bit quantized** mode
* Uses HuggingFace `pipeline("text-generation")`
* Key settings:
  * `do_sample=False` — deterministic output
  * `max_new_tokens=512`
  * `repetition_penalty=1.15`

---

### 🧾 Prompt Engineering

#### `SYSTEM_PROMPT`

* Strict numbered rules given to the LLM:
  * Answer **only** from the provided context
  * **Never** invent or combine unrelated topics
  * If the question is off-topic, return exactly:

```
I don't know based on provided data.
```

#### `build_prompt(query, context)`

* Assembles the full prompt for TinyLlama:
  * System rules
  * Retrieved context (capped at 1800 chars)
  * User question
  * Reminder to refuse if off-topic

#### `_clean_answer(raw_text)`

* Strips leaked TinyLlama control tokens from generated output:
  * `<|system|>`, `<|user|>`, `<|assistant|>`, `</s>`
* Returns a clean, readable answer string

---

### 🎯 `generate_answer()` — Main Function

> 👉 Runs the full RAG pipeline end-to-end

```python
generate_answer(query, vectorstore, llm_pipeline)
```

**Step-by-step flow:**

```
1. retrieve_chunks()         → find top-K relevant chunks
        ↓
2. Confidence Check          → score < 0.28 → ❌ NOT_FOUND
        ↓
3. Topic Gate (Fix 1)        → no keyword overlap → ❌ block
        ↓
4. build_prompt() + LLM      → generate raw answer
        ↓
5. _clean_answer()           → strip control tokens
        ↓
6. Length Check              → answer < 8 chars → ❌ NOT_FOUND
        ↓
7. Refusal Detection         → LLM said "I don't know" → return standard message
        ↓
8. Hallucination Check (Fix 3) → foreign words in answer → ❌ block
        ↓
9. ✅ Return answer, sources, confidence, found=True
```

---

### 🏗️ Index Builder

#### `build_index_from_docs(docs_dir, index_path)`

* Convenience function that runs the full indexing pipeline in one call:

```
load_documents() → chunk_documents() → build_vectorstore()
```

* Use this **once during setup** to build and save your FAISS index

---

### ⚠️ Why You Get `"I don't know based on provided data."`

There are **three layers** that can return this message. Check the server logs to find which one fired:

| Log Line | Cause | Fix |
|----------|-------|-----|
| `[generate] below threshold` | Similarity score < 0.28 — query too distant from docs | Lower `CONFIDENCE_THRESH` or improve docs coverage |
| `[topic_gate] overlap=set()` | No query words found in retrieved context | Query is genuinely off-topic — expected behaviour |
| `[hallucination_check] foreign words` | LLM used query terms absent from context | Expected — hallucination correctly blocked |
| `[generate] answer is a refusal` | LLM correctly refused on its own | Expected — model followed instructions |
| `[generate] retrieval returned 0 results` | Vectorstore is empty or not loaded | Rebuild index with `build_index_from_docs()` |

---

## 🎤 `voice_pipeline.py` — Voice Input / Output Reference

> Handles the full voice I/O layer: **Audio → Text (STT) → (RAG in main.py) → Text → Audio (TTS)**

---

### 🔤 Text Processing

#### `normalize_text(text)`

* Lowercases text
* Removes punctuation
* Cleans extra whitespace

#### `correct_aliases(text)`

* Fixes common mispronunciations before RAG lookup:
  * `"zakkat"` → `"zakat"`
  * `"nisap"` → `"nisab"`
* Improves retrieval accuracy for domain-specific terms

---

### 🤖 Model

#### `load_whisper(model_size)`

* Loads the OpenAI Whisper speech-to-text model
* Uses GPU if available, falls back to CPU
* Returns the loaded model object

---

### 🎧 Audio Processing

#### `preprocess_audio(input_path, output_path)`

* Converts any input audio to a Whisper-compatible format:
  * WAV format
  * 16kHz sample rate
  * Mono channel
* Uses `ffmpeg` under the hood
* Required before every transcription call

#### `validate_audio_file(file_path)`

* Guards against runtime errors by checking:
  * File exists on disk
  * Format is in the supported list (`.mp3`, `.wav`, `.webm`, etc.)
  * File is not empty (0 bytes)

---

### 📝 Transcription (Speech → Text)

#### `transcribe(audio_path, model, language)`

* Full STT pipeline from a file path:
  1. Validate audio file
  2. Convert to clean WAV via `preprocess_audio()`
  3. Run Whisper transcription
  4. Extract: `text`, `duration`, `segments`, `processing_sec`
* Applies alias correction after transcription

#### `transcribe_bytes(audio_bytes, model, suffix, language)`

* Same pipeline as `transcribe()`, but accepts raw bytes
* Writes bytes to a temp file, then calls `transcribe()`
* Used for browser uploads and API audio endpoints

---

### 🔊 Text-to-Speech

#### `text_to_speech(text, language, output_filename)`

* Converts answer text → `.mp3` using gTTS
* Pre-processing:
  * Strips `[Source X]` citation markers from text
  * Truncates to 500 words to avoid gTTS timeouts
* Always saves output into:

```
./tts_output/
```

* Key fixes applied:
  * Accepts `language` param (previously caused a crash)
  * Treats `output_filename` as a bare filename only — prevents double-path bug

---

### 🔁 `process_voice_query()` — Main Wrapper

> 👉 Called by `/upload-audio` in `main.py`

```python
process_voice_query(audio_bytes, whisper_model, audio_suffix, force_language)
```

**Flow:**

```
Receive audio bytes
      ↓
transcribe_bytes()   →   Whisper STT
      ↓
Alias correction     →   "zakkat" → "zakat"
      ↓
Return structured result:
  • query          (transcribed text)
  • language       (detected/forced)
  • duration_sec
  • processing_sec
  • error          (None if success)
```

* Fix applied: added `audio_suffix` and `force_language` params — previously caused `TypeError` when called from `main.py`

---

### 📁 File Helper

#### `process_audio_file(audio_path, whisper_model)`

* Reads an audio file from disk → converts to bytes → calls `process_voice_query()`
* Used for local CLI testing only

---

### 🧪 CLI Mode

* Run directly from terminal to test transcription:

```bash
python voice_pipeline.py audio.mp3
```

* Prints: language, duration, and full transcript

---

### 🧠 Full Voice Flow

```
Audio Input (mic / file / upload)
         ↓
process_voice_query()
         ↓
preprocess_audio()    →   WAV 16kHz mono
         ↓
Whisper STT           →   Raw transcript
         ↓
correct_aliases()     →   "nisap" → "nisab"
         ↓
Return query text
         ↓
(main.py passes to RAG pipeline)
         ↓
text_to_speech()      →   Answer → MP3
         ↓
MP3 served via /tts/{filename}
```

---

### ⚠️ Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `TypeError: process_voice_query() got unexpected keyword argument` | Old signature missing `audio_suffix` / `force_language` | Use the fixed `voice_pipeline.py` |
| `FileNotFoundError` on TTS save | `output_filename` was a full path, causing double-path concat | Fixed: only bare filename is used now |
| Empty transcription from Whisper | Audio file is silent or corrupt | Validate with `validate_audio_file()` first |
| Wrong words in transcript | Domain terms mispronounced | Add entries to the `ALIAS` dict in `voice_pipeline.py` |

---

## 🌐 `main.py` — FastAPI Server Reference

> Ties everything together: receives requests, calls voice/RAG pipelines, returns answers + audio

---

### 🔧 Helpers

#### `_check_models()`

* Verifies that both the vectorstore and LLM pipeline are loaded in `app_state`
* If either is missing → raises `503 Service Unavailable`

#### `_add_to_history(role, content)`

* Appends a message to the in-memory conversation history
* Keeps the last **50 messages** only (older ones are dropped)

#### `_build_response(query, rag_result, tts_path, language)`

* Formats the final API response dict:
  * `answer`, `confidence`, `found`
  * `sources` — each with label, snippet, score
  * `tts_url` — relative path `/tts/filename.mp3`
* Fix applied: `language` param was missing, causing `TypeError` in `upload_audio()`

---

### 📦 Request Model

#### `TextQueryRequest`

* Pydantic model used by `/ask`
* Fields:
  * `query` — the question string
  * `tts` — `True/False`, whether to generate audio
  * `language` — defaults to `"en"`

---

### 🌐 Endpoints

#### `GET /`  →  `serve_ui()`

* Serves `index.html` as the frontend
* Returns `404` if the file is not present in the working directory

#### `GET /health`  →  `health()`

* Returns current system status:
  * vectorstore loaded?
  * LLM loaded?
  * Whisper loaded?
  * conversation history count

#### `POST /ask`  →  `ask(request)`

> 👉 Text query → RAG → optional TTS response

**Flow:**

```
1. Validate query (not empty)
2. Save user message to history
3. generate_answer()    →   RAG result
4. text_to_speech()     →   MP3 (if tts=True and answer found)
5. Save assistant reply to history
6. _build_response()    →   Return JSON
```

#### `POST /upload-audio`  →  `upload_audio(...)`

> 👉 Audio file → Whisper STT → RAG → TTS response

**Flow:**

```
1. Read uploaded audio bytes
2. process_voice_query()   →   transcribed text + language
3. generate_answer()       →   RAG result
4. text_to_speech()        →   MP3 (if tts=true and answer found)
5. _build_response()       →   Return JSON with:
     • transcript
     • detected_lang
     • audio_duration
     • stt_time
```

* Fixes applied:
  * Passes `audio_suffix` and `force_language` to `process_voice_query()`
  * Returns `422` with real error message instead of a generic `500`

#### `GET /tts/{filename}`  →  `serve_tts(filename)`

* Serves generated `.mp3` files from `./tts_output/`
* Security:
  * Only `.mp3` extensions accepted
  * Path traversal stripped via `Path(filename).name`

#### `POST /ask-mcq`  →  `ask_mcq(request)`

> 👉 Generate a multiple-choice question from the RAG answer

**Flow:**

```
1. Run generate_answer()
2. If no answer found → return null MCQ
3. Prompt LLM to create:
     Q: ...
     A) ...  B) ...  C) ...  D) ...
     Answer: ...
```

#### `GET /history`  →  `get_history()`

* Returns full conversation history with timestamps

#### `DELETE /history`  →  `clear_history()`

* Wipes all stored conversation history

---

### 🧠 Full System Flow

```
Text path:
User → POST /ask → generate_answer() → answer → (TTS) → JSON response

Voice path:
User → POST /upload-audio → Whisper STT → generate_answer() → answer → TTS → JSON response
```

---

### ⚠️ Key Fixes in `main.py`

| Bug | Symptom | Fix |
|-----|---------|-----|
| `_build_response()` missing `language` param | `TypeError` on every audio upload | Added `language="en"` as 4th parameter |
| `text_to_speech()` called with full path as filename | `FileNotFoundError` — double-path concat | Now passes bare filename: `tts_{uuid}.mp3` |
| `process_voice_query()` called without `audio_suffix` / `force_language` | `TypeError` crash on audio upload | Now passes both kwargs explicitly |
| Generic `500` error on audio upload failure | No useful error shown in UI | Returns `422` with `detail=str(e)` |

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| API Server | FastAPI + Uvicorn |
| Speech-to-Text | OpenAI Whisper |
| Text-to-Speech | gTTS |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (IndexFlatIP — cosine similarity) |
| LLM | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| RAG Framework | LangChain |
| Public Tunnel | ngrok |
| Audio Processing | ffmpeg |
