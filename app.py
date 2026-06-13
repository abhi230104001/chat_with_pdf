import os
import math
import json
from pathlib import Path
from collections import defaultdict

import google.generativeai as genai
from pypdf import PdfReader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 60
TOP_K         = 4
UPLOAD_DIR    = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI(title="RAG Chatbot — Gemini")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── In-memory document store ───────────────────────────────────────────────────
document_chunks: dict[str, list[dict]] = {}


# ── RAG utilities ──────────────────────────────────────────────────────────────
def chunk_text(text: str, source: str) -> list[dict]:
    words  = text.split()
    step   = CHUNK_SIZE - CHUNK_OVERLAP
    result = []
    for i in range(0, len(words), step):
        snippet = " ".join(words[i : i + CHUNK_SIZE])
        if snippet.strip():
            result.append({"id": f"{source}::{i}", "text": snippet, "source": source})
    return result


def extract_text_from_file(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        try:
            reader = PdfReader(file_path)
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF: {e}")
    else:
        # text files (.txt, .md, .csv)
        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise RuntimeError(f"Failed to read text file: {e}")


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def tfidf_embed(query: str, corpus: list[str]) -> tuple[list[float], list[list[float]]]:
    import re
    tokenize = lambda t: re.sub(r'[^\w\s]', ' ', t.lower()).split()
    vocab    = sorted({w for doc in [query] + corpus for w in tokenize(doc)})
    w_idx    = {w: i for i, w in enumerate(vocab)}
    N        = len(corpus)
    df       = defaultdict(int)
    for doc in corpus:
        for w in set(tokenize(doc)):
            df[w] += 1

    def embed(text: str) -> list[float]:
        words = tokenize(text)
        n     = len(words) or 1
        tf    = defaultdict(int)
        for w in words:
            tf[w] += 1
        vec = [0.0] * len(vocab)
        for w, cnt in tf.items():
            if w in w_idx:
                vec[w_idx[w]] = (cnt / n) * (math.log((N + 1) / (df[w] + 1)) + 1.0)
        return vec

    return embed(query), [embed(c) for c in corpus]


def retrieve(query: str) -> list[dict]:
    all_chunks = [c for cs in document_chunks.values() for c in cs]
    if not all_chunks:
        return []
    corpus    = [c["text"] for c in all_chunks]
    qv, cvecs = tfidf_embed(query, corpus)
    scored    = sorted(zip(all_chunks, cvecs),
                       key=lambda x: cosine_sim(qv, x[1]), reverse=True)
    
    matches = [c for c, v in scored if cosine_sim(qv, v) > 0]
    if matches:
        return matches[:TOP_K]
        
    return [c for c, v in scored[:TOP_K]]


# ── Pydantic models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer:      str
    sources:     list[str]
    chunks_used: int


# ── API routes ─────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed = {".txt", ".md", ".csv", ".pdf"}
    ext     = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported type '{ext}'. Allowed: {allowed}")
    content = await file.read()
    target_path = UPLOAD_DIR / file.filename
    target_path.write_bytes(content)
    try:
        text = extract_text_from_file(target_path)
    except Exception as e:
        if target_path.exists():
            target_path.unlink()
        raise HTTPException(400, str(e))
    chunks  = chunk_text(text, file.filename)
    document_chunks[file.filename] = chunks
    return {"filename": file.filename, "chunks": len(chunks), "words": len(text.split())}


@app.delete("/document/{filename}")
async def delete_doc(filename: str):
    if filename not in document_chunks:
        raise HTTPException(404, "Document not found")
    del document_chunks[filename]
    p = UPLOAD_DIR / filename
    if p.exists():
        p.unlink()
    return {"deleted": filename}


@app.get("/documents")
async def list_documents():
    return [{"filename": n, "chunks": len(cs)} for n, cs in document_chunks.items()]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    retrieved = retrieve(req.question)

    if retrieved:
        context = "\n\n".join(
            f"[{i+1}] (source: {c['source']})\n{c['text']}"
            for i, c in enumerate(retrieved)
        )
    else:
        context = "No relevant documents found in the knowledge base."

    prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "context provided below. If the answer is not found in the context, say so clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {req.question}"
    )

    response = model.generate_content(prompt)
    answer   = response.text
    sources  = list({c["source"] for c in retrieved})
    return ChatResponse(answer=answer, sources=sources, chunks_used=len(retrieved))


@app.on_event("startup")
def startup_event():
    if UPLOAD_DIR.exists():
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                allowed = {".txt", ".md", ".csv", ".pdf"}
                if file_path.suffix.lower() in allowed:
                    try:
                        text = extract_text_from_file(file_path)
                        chunks = chunk_text(text, file_path.name)
                        document_chunks[file_path.name] = chunks
                        print(f"Indexed startup file: {file_path.name} ({len(chunks)} chunks)")
                    except Exception as e:
                        print(f"Failed to index startup file {file_path.name}: {e}")

# ── Serve frontend ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    p = Path("static/index.html")
    return p.read_text() if p.exists() else "<h1>RAG Chatbot (Gemini)</h1>"
