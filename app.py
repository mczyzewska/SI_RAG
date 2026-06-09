"""
Motorcycle RAG — FastAPI backend
Używa: Ollama (lokalny LLM), FAISS (baza wektorowa), sentence-transformers (embeddingi)
"""

import os
import json
import shutil

import faiss
import numpy as np
import pymupdf
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------
KNOWLEDGE_DIR = "knowledge/"
VEC_DB_DIR    = "vec_db/"
CHUNK_SIZE    = 600           # znaki na chunk (więcej = więcej kontekstu)
TOP_K         = 5             # ile chunków pobieramy z FAISS
EMBED_MODEL   = "paraphrase-multilingual-mpnet-base-v2"
OLLAMA_URL    = "http://localhost:11434/api/chat"
OLLAMA_MODEL  = "qwen2.5"    # zmień na dowolny model zainstalowany w Ollama

os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(VEC_DB_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Inicjalizacja modelu embeddingowego + FAISS
# ---------------------------------------------------------------------------
print("⏳ Ładowanie modelu embeddingowego...")
embedder = SentenceTransformer(EMBED_MODEL)
DIM = embedder.get_embedding_dimension()

index_path    = os.path.join(VEC_DB_DIR, "vector_database.index")
metadata_path = os.path.join(VEC_DB_DIR, "metadata.json")

# Wczytaj istniejący indeks lub utwórz nowy
if os.path.exists(index_path) and os.path.exists(metadata_path):
    print("✅ Wczytywanie istniejącego indeksu FAISS...")
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
else:
    print("🆕 Tworzenie nowego indeksu FAISS...")
    index    = faiss.IndexFlatL2(DIM)
    metadata = []

print(f"📚 Chunks w bazie: {index.ntotal}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    """Zwraca listę (numer_strony, tekst_strony)."""
    result = []
    with pymupdf.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            text = page.get_text().replace("\n", " ").strip()
            if text:
                result.append((page_num, text))
    return result


def chunk_text(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Tnie strony na chunki o długości CHUNK_SIZE znaków."""
    chunks = []
    for page_num, text in pages:
        for i in range(0, len(text), CHUNK_SIZE):
            chunks.append((page_num, text[i : i + CHUNK_SIZE]))
    return chunks


def save_index():
    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def already_indexed(filename: str) -> bool:
    return any(m["filename"] == filename for m in metadata)


def add_pdf_to_index(file_path: str, filename: str) -> int:
    pages  = extract_text_from_pdf(file_path)
    chunks = chunk_text(pages)
    for chunk_num, (page_num, chunk) in enumerate(tqdm(chunks, desc=f"Indeksowanie {filename}")):
        emb = embedder.encode(chunk, show_progress_bar=False)
        index.add(np.array([emb], dtype=np.float32))
        metadata.append({
            "filename":    filename,
            "page_number": page_num,
            "chunk_num":   chunk_num,
            "chunk":       chunk,
        })
    save_index()
    return len(chunks)


def retrieve_chunks(query: str, k: int = TOP_K) -> list[dict]:
    if index.ntotal == 0:
        return []
    q_emb = embedder.encode(query, show_progress_bar=False)
    _, I  = index.search(np.array([q_emb], dtype=np.float32), k)
    return [metadata[i] for i in I[0] if i < len(metadata)]


def ask_ollama(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model":    OLLAMA_MODEL,
        "messages": [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Nie można połączyć się z Ollama. Upewnij się, że działa: `ollama serve`",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="Moto RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


# ── Status ──────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    files = list({m["filename"] for m in metadata})
    return {"chunks": index.ntotal, "files": files, "ollama_model": OLLAMA_MODEL}


# ── Upload PDF ───────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Tylko pliki PDF.")

    if already_indexed(file.filename):
        return {"message": f"'{file.filename}' już jest w bazie.", "chunks_added": 0}

    dest = os.path.join(KNOWLEDGE_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    n = add_pdf_to_index(dest, file.filename)
    return {"message": f"Dodano '{file.filename}' ({n} chunków).", "chunks_added": n}


# ── Usuń plik ────────────────────────────────────────────────────────────────

@app.delete("/files/{filename}")
def delete_file(filename: str):
    global index, metadata

    if not any(m["filename"] == filename for m in metadata):
        raise HTTPException(status_code=404, detail="Plik nie znaleziony w bazie.")

    # Przebuduj indeks bez chunków tego pliku
    new_meta = [m for m in metadata if m["filename"] != filename]
    new_index = faiss.IndexFlatL2(DIM)

    if new_meta:
        vecs = []
        for m in new_meta:
            emb = embedder.encode(m["chunk"], show_progress_bar=False)
            vecs.append(emb)
        new_index.add(np.array(vecs, dtype=np.float32))

    index    = new_index
    metadata = new_meta
    save_index()

    # Usuń plik z dysku
    fpath = os.path.join(KNOWLEDGE_DIR, filename)
    if os.path.exists(fpath):
        os.remove(fpath)

    return {"message": f"Usunięto '{filename}' z bazy."}


# ── Pytanie ──────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    query: str

SYSTEM_PROMPT = """Jesteś ekspertem mechaniki motocyklowej i diagnostyki. 
Odpowiadasz WYŁĄCZNIE na podstawie dostarczonego kontekstu z dokumentacji.
NIE ODPOWIADASZ na pytania nie związane z mechaniką motocyklową.
Jeśli kontekst nie zawiera odpowiedzi — powiedz to wprost.
Odpowiadaj po polsku, konkretnie i technicznie. 
Nie wymyślaj informacji których nie ma w kontekście."""

PROMPT_TEMPLATE = """Kontekst z dokumentacji motocyklowej:
{context}

Pytanie użytkownika: {query}

Odpowiedz na podstawie powyższego kontekstu. Jeśli to kod błędu — wyjaśnij co oznacza i jak go naprawić. 
Jeśli to pytanie o wymianę części — podaj kroki. Bądź precyzyjny."""


@app.post("/ask")
def ask(req: QuestionRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Pytanie nie może być puste.")

    chunks = retrieve_chunks(req.query)

    if not chunks:
        return {
            "answer": "Baza wiedzy jest pusta. Wgraj najpierw dokumentację PDF.",
            "chunks": [],
        }

    context = "\n\n".join(
        f"[{i+1}] (plik: {c['filename']}, str. {c['page_number']+1})\n{c['chunk']}"
        for i, c in enumerate(chunks)
    )

    prompt  = PROMPT_TEMPLATE.format(context=context, query=req.query)
    answer  = ask_ollama(SYSTEM_PROMPT, prompt)

    return {
        "answer": answer,
        "chunks": [
            {
                "filename":    c["filename"],
                "page_number": c["page_number"] + 1,
                "excerpt":     c["chunk"][:300] + ("..." if len(c["chunk"]) > 300 else ""),
            }
            for c in chunks
        ],
    }
