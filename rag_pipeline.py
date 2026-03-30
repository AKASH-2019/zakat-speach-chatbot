"""
rag_pipeline.py  (fixed v3 — anti-hallucination)
-------------------------------------------------
English-only RAG pipeline.

"""

import os
import re
import uuid
from pathlib import Path

import numpy as np
import torch

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL         = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

CHUNK_SIZE        = 600
CHUNK_OVERLAP     = 80
TOP_K             = 4
CONFIDENCE_THRESH = 0.28
FAISS_INDEX_PATH  = "faiss_index"
NOT_FOUND_MSG     = "I don't know based on provided data."

# Common English stop-words excluded from topic-gate and hallucination checks
_STOP = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should",
    "may","might","shall","can","need","dare","ought","used",
    "i","you","he","she","it","we","they","me","him","her","us","them",
    "my","your","his","its","our","their","mine","yours","ours","theirs",
    "this","that","these","those","what","which","who","whom","whose",
    "when","where","why","how","all","each","every","both","few","more",
    "most","other","some","such","no","not","only","same","so","than",
    "too","very","just","but","and","or","nor","for","yet","if","of",
    "at","by","in","on","to","up","as","into","through","about","with",
    "from","then","there","here","any","also","know","based","provided",
    "data","tell","give","me","please","explain","describe","information",
    "question","answer","context","do","u","your","can","something",
}

# ── Document Loading ──────────────────────────────────────────────────────────

def load_documents(docs_dir: str) -> list[Document]:
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    documents = []

    for pdf_file in docs_path.glob("**/*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            pages  = loader.load()
            for page in pages:
                page.metadata["source_file"] = pdf_file.name
            documents.extend(pages)
            print(f"  Loaded PDF: {pdf_file.name} ({len(pages)} pages)")
        except Exception as e:
            print(f"  Warning: Could not load {pdf_file.name}: {e}")

    for txt_file in docs_path.glob("**/*.txt"):
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            pages  = loader.load()
            for page in pages:
                page.metadata["source_file"] = txt_file.name
            documents.extend(pages)
            print(f"  Loaded TXT: {txt_file.name}")
        except Exception as e:
            print(f"  Warning: Could not load {txt_file.name}: {e}")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents

# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

# ── Embedding & Indexing ──────────────────────────────────────────────────────

def _make_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(
    chunks: list[Document],
    save_path: str = FAISS_INDEX_PATH,
) -> FAISS:
    embeddings = _make_embeddings()
    print("Building FAISS index (MAX_INNER_PRODUCT / cosine similarity)...")
    vs = FAISS.from_documents(
        chunks,
        embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    vs.save_local(save_path)
    print(f"Index saved to: {save_path}/")
    return vs


def load_vectorstore(save_path: str = FAISS_INDEX_PATH) -> FAISS:
    embeddings = _make_embeddings()
    print(f"Loading FAISS index from: {save_path}/")
    return FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )

# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_chunks(query: str, vectorstore: FAISS, top_k: int = TOP_K):
    results = vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
    for doc, score in results:
        src = doc.metadata.get("source_file", "?")
        print(f"  [retrieval] score={score:.3f}  src={src}  preview={doc.page_content[:60]!r}")
    return results


def build_context(results):
    context_parts = []
    sources       = []
    top_score     = 0.0

    for i, (doc, score) in enumerate(results, 1):
        top_score   = max(top_score, score)
        source_file = doc.metadata.get("source_file", "Unknown")
        page_num    = doc.metadata.get("page", "?")
        chunk_text  = doc.page_content.strip()

        context_parts.append(
            f"[Source {i}: {source_file}, page {page_num}]\n{chunk_text}"
        )
        sources.append({
            "index"  : i,
            "file"   : source_file,
            "page"   : page_num,
            "score"  : round(score, 3),
            "snippet": chunk_text[:160] + ("..." if len(chunk_text) > 160 else ""),
        })

    return "\n\n---\n\n".join(context_parts), sources, top_score

# ── Topic Gate — HALLUCINATION FIX 1 ─────────────────────────────────────────

def _content_words(text: str) -> set[str]:
    """Lowercase alpha words, length >= 3, minus stop-words."""
    return {
        w for w in re.findall(r"[a-z]{3,}", text.lower())
        if w not in _STOP
    }


def _topic_gate(query: str, context_text: str) -> bool:
    """
    Return True (allow LLM call) only when the query shares at least one
    content word with the retrieved context.

    TinyLlama hallucinates when the question topic has zero grounding in the
    context — it invents a connection instead of refusing.  Blocking here
    prevents the LLM from ever seeing the off-topic query.

    Examples:
      "do you know about donald trump"  → overlap={} → BLOCKED
      "what is the nisab for gold"      → overlap={"nisab","gold"} → ALLOWED
    """
    query_words   = _content_words(query)
    context_words = _content_words(context_text)
    overlap       = query_words & context_words

    print(f"  [topic_gate] query_words={query_words}  overlap={overlap}")

    # If query has no content words at all (all stop-words), let LLM handle
    if not query_words:
        return True

    return len(overlap) >= 1


# ── Hallucination Detector — HALLUCINATION FIX 3 ─────────────────────────────

def _hallucination_check(query: str, answer: str, context_text: str) -> bool:
    """
    Return True (hallucinated) if the answer contains query words that have
    no grounding in the retrieved context.

    Logic:
      foreign = content_words(query) - content_words(context)
      if any foreign word appears in the answer → hallucinated

    Example:
      query foreign words: {"trump", "donald"}
      answer contains "trump" → return True → block answer
    """
    query_words   = _content_words(query)
    context_words = _content_words(context_text)
    answer_words  = _content_words(answer)

    foreign      = query_words - context_words
    hallucinated = foreign & answer_words

    if hallucinated:
        print(f"  [hallucination_check] foreign words in answer: {hallucinated}")
        return True

    return False

# ── LLM ───────────────────────────────────────────────────────────────────────

def load_llm():
    from transformers import BitsAndBytesConfig
    print(f"\nLoading LLM: {LLM_MODEL}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.15,
        return_full_text=False,
    )
    print("LLM loaded.")
    return llm_pipeline

# ── Prompt — HALLUCINATION FIX 2 ─────────────────────────────────────────────

# Numbered rules work better than a single soft instruction for TinyLlama.
SYSTEM_PROMPT = """\
You are a strict question-answering assistant. Follow these rules without exception:

1. Read the CONTEXT carefully.
2. Answer ONLY using information that is explicitly stated in the CONTEXT.
3. If the question asks about a person, place, event, or topic that does NOT appear anywhere in the CONTEXT, you MUST reply with this exact sentence and nothing else:
   I don't know based on provided data.
4. NEVER combine topics from the CONTEXT with topics from the question that are not in the CONTEXT.
5. NEVER invent, guess, or assume any information not present in the CONTEXT.
6. When the CONTEXT does contain the answer, be concise and cite sources like [Source 1].\
"""

MAX_CONTEXT_CHARS = 1800


def build_prompt(query: str, context: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n"
        f"CONTEXT:\n{context[:MAX_CONTEXT_CHARS]}\n\n"
        f"QUESTION: {query}\n\n"
        f"Remember: if the question mentions anything not in the CONTEXT, "
        f"reply with exactly: I don't know based on provided data.</s>\n"
        f"<|assistant|>\n"
    )


def _clean_answer(raw_text: str) -> str:
    """Strip leaked TinyLlama prompt tokens from generated output."""
    cleaned = re.sub(r"<\|system\|>.*?</s>", "", raw_text, flags=re.S)
    cleaned = re.sub(r"<\|user\|>.*?</s>",   "", cleaned,  flags=re.S)
    cleaned = re.sub(r"<\|assistant\|>",      "", cleaned)
    cleaned = re.sub(r"</s>",                 "", cleaned)
    return cleaned.strip()

# ── Answer Generation ─────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    vectorstore: FAISS,
    llm_pipeline,
    top_k: int = TOP_K,
    confidence_threshold: float = CONFIDENCE_THRESH,
) -> dict:
    try:
        results = retrieve_chunks(query, vectorstore, top_k)
        if not results:
            print("  [generate] retrieval returned 0 results")
            return {"answer": NOT_FOUND_MSG, "sources": [], "confidence": 0.0, "found": False}

        context_text, sources, top_score = build_context(results)
        print(f"  [generate] top_score={top_score:.3f}  threshold={confidence_threshold}")

        # ── Confidence threshold ──────────────────────────────────────────────
        if top_score < confidence_threshold:
            print("  [generate] below threshold — NOT_FOUND")
            return {
                "answer"    : NOT_FOUND_MSG,
                "sources"   : [],
                "confidence": round(top_score, 3),
                "found"     : False,
            }

        # ── FIX 1: Topic gate (pre-LLM) ──────────────────────────────────────
        if not _topic_gate(query, context_text):
            print("  [generate] topic_gate blocked — NOT_FOUND")
            return {
                "answer"    : NOT_FOUND_MSG,
                "sources"   : [],
                "confidence": round(top_score, 3),
                "found"     : False,
            }

        # ── Call LLM ─────────────────────────────────────────────────────────
        prompt = build_prompt(query, context_text)
        raw    = llm_pipeline(prompt)

        if isinstance(raw, list) and len(raw) > 0:
            answer = raw[0].get("generated_text", "").strip()
        else:
            answer = str(raw).strip()

        answer = _clean_answer(answer)
        print(f"  [generate] raw answer: {answer[:150]!r}")

        if not answer or len(answer) < 8:
            return {
                "answer"    : NOT_FOUND_MSG,
                "sources"   : sources,
                "confidence": round(top_score, 3),
                "found"     : False,
            }

        # ── Refusal pattern check ─────────────────────────────────────────────
        first_line = answer.split("\n")[0]
        if re.search(
            r"i (don't|dont|do not) know based on|"
            r"not (in|available in) (the |this )?context|"
            r"cannot answer (this|the) question",
            first_line,
            re.I,
        ):
            print("  [generate] answer is a refusal — NOT_FOUND")
            return {
                "answer"    : NOT_FOUND_MSG,
                "sources"   : [],
                "confidence": round(top_score, 3),
                "found"     : False,
            }

        # ── FIX 3: Post-generation hallucination check ────────────────────────
        if _hallucination_check(query, answer, context_text):
            print("  [generate] hallucination detected — NOT_FOUND")
            return {
                "answer"    : NOT_FOUND_MSG,
                "sources"   : [],
                "confidence": round(top_score, 3),
                "found"     : False,
            }

        return {
            "answer"    : answer,
            "sources"   : sources,
            "confidence": round(top_score, 3),
            "found"     : True,
        }

    except Exception as e:
        print(f"  [generate] ERROR: {e}")
        return {
            "answer"    : NOT_FOUND_MSG,
            "sources"   : [],
            "confidence": 0.0,
            "found"     : False,
            "error"     : str(e),
        }

# ── Index Builder ─────────────────────────────────────────────────────────────

def build_index_from_docs(docs_dir: str, index_path: str = FAISS_INDEX_PATH) -> FAISS:
    print("=" * 52)
    print("  Building RAG index")
    print("=" * 52)
    docs   = load_documents(docs_dir)
    chunks = chunk_documents(docs)
    return build_vectorstore(chunks, save_path=index_path)
