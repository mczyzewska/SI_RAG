# 🏍️ MotoRAG — Asystent Mechanika

Lokalny system RAG do odpowiadania na pytania o motocykle na podstawie własnych PDFów (instrukcje serwisowe, DTCs, schematy elektryczne itp.).

---

## Jak to działa?

```
PDF → wyciąg tekstu → chunki → embeddingi (sentence-transformers) → FAISS
Pytanie → embedding pytania → top-5 chunków z FAISS → prompt → Ollama (LLM) → odpowiedź
```

Komponenty:
- **FastAPI** — serwer HTTP + REST API
- **sentence-transformers** (`paraphrase-multilingual-mpnet-base-v2`) — model embeddingowy, działa lokalnie, obsługuje język polski
- **FAISS** — wektorowa baza danych (wyszukiwanie semantyczne)
- **Ollama** — lokalny LLM (np. llama3.2, mistral, qwen2.5)
- **PyMuPDF** — ekstrakcja tekstu z PDF

---

## Wymagania

- Python 3.10+
- [Ollama](https://ollama.com) zainstalowane i działające

---

## Instalacja

### 1. Sklonuj / rozpakuj projekt

```bash
cd moto_rag
```

### 2. Zainstaluj zależności Pythona

```bash
pip install -r requirements.txt
```

Pierwsze uruchomienie pobierze model embeddingowy (~420 MB) — tylko raz.

### 3. Zainstaluj Ollama i pobierz model LLM

```bash
# Instalacja Ollama (Linux/Mac)
curl -fsSL https://ollama.com/install.sh | sh

# Pobierz model (wybierz jeden):
ollama pull llama3.2        # 2GB, szybki, dobry po angielsku
ollama pull mistral         # 4GB, lepszy kontekst
ollama pull qwen2.5         # świetny dla polskiego
ollama pull gemma3:4b       # lekki, dobry jakościowo
```

### 4. (Opcjonalnie) Zmień model w app.py

W pliku `app.py` zmień linię:
```python
OLLAMA_MODEL = "llama3.2"   # ← wpisz nazwę pobranego modelu
```

---

## Uruchomienie

### Terminal 1 — Ollama
```bash
ollama serve
```

### Terminal 2 — FastAPI
```bash
cd moto_rag
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Otwórz przeglądarkę
```
http://localhost:8000
```

---

## Użytkowanie

1. **Wgraj PDFy** — kliknij strefę upload lub przeciągnij pliki (instrukcje serwisowe, listy DTC, schematy)
2. **Zadaj pytanie** — np. `"Co oznacza błąd P0300?"`, `"Jak wymienić sprzęgło?"`, `"Moment dokręcania śrub głowicy"`
3. **Zobacz źródła** — każda odpowiedź pokazuje fragmenty dokumentacji, na których bazuje

---

## Struktura projektu

```
moto_rag/
├── app.py              ← backend FastAPI (cała logika RAG)
├── requirements.txt
├── static/
│   └── index.html      ← frontend (UI)
├── knowledge/          ← tu trafiają wgrane PDFy
└── vec_db/             ← indeks FAISS + metadane (tworzone automatycznie)
    ├── vector_database.index
    └── metadata.json
```

---

## API (dla zainteresowanych)

| Endpoint | Metoda | Opis |
|---|---|---|
| `GET /status` | GET | Liczba chunków, lista plików, model |
| `POST /upload` | POST | Upload PDF (form-data, pole `file`) |
| `DELETE /files/{filename}` | DELETE | Usuń plik z bazy |
| `POST /ask` | POST | Zadaj pytanie `{"query": "..."}` |

---

## Konfiguracja (app.py)

| Zmienna | Domyślnie | Opis |
|---|---|---|
| `CHUNK_SIZE` | 600 | Znaki na fragment tekstu. Zwiększ jeśli model pomija kontekst |
| `TOP_K` | 5 | Ile fragmentów pobierać z FAISS na zapytanie |
| `OLLAMA_MODEL` | `llama3.2` | Model Ollama do generowania odpowiedzi |
| `EMBED_MODEL` | `paraphrase-multilingual-mpnet-base-v2` | Model embeddingowy |

---

## Wskazówki

- **Jakość PDFów ma znaczenie** — skanowane obrazy nie zadziałają (brak warstwy tekstowej). Używaj PDFów z możliwością zaznaczania tekstu.
- **Język modelu** — modele jak `qwen2.5` lub `mistral` lepiej radzą sobie z polskim.
- **Duże pliki** — indeksowanie kilkuset-stronicowej instrukcji może zająć 1-2 minuty.
- **Baza persystuje** — po restarcie serwera wektory są wczytywane z dysku (`vec_db/`).
