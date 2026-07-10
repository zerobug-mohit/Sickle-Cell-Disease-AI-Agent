# SCD WhatsApp Chatbot

A WhatsApp chatbot for health workers (ASHAs, MPWs, CHOs, Staff Nurses, Medical
Officers) under India's **National Sickle Cell Anaemia Elimination Mission
(NSCAEM)**. It answers questions in **English, Hindi (Devanagari), or Hinglish**,
grounded in the 6 Government of India SCD training documents via retrieval-augmented
generation (RAG). Every grounded answer cites the source document.

## How it works

```
WhatsApp ──▶ Meta webhook ──▶ FastAPI (app/main.py)
                                  │
                          message_router
                                  ├─ new user ▶ onboarding: State ▶ District ▶ Cadre
                                  ├─ greetings / /help / /stop / /profile
                                  │
                              qa_handler  (profile-aware)
                                  ├─ scope gate (gpt-4o-mini)        ── off-topic ▶ fallback (no source)
                                  ├─ multi-query expansion (+ cadre)
                                  ├─ embed (text-embedding-3-small) ▶ FAISS top-k ▶ cadre re-rank
                                  └─ answer (gpt-4o-mini, profile injected)  ──▶ reply + source citation
                                  │
                              meta_client ──▶ WhatsApp reply
```

- **User profile** — on first contact the bot captures **State, District, and Cadre** (ASHA, ANM, MPW, CHO, Staff Nurse, Lab Technician, Medical Officer, Counsellor); `/profile` re-runs it.
- **Cadre-aware retrieval** — every chunk is tagged with the cadre(s) it applies to; retrieval mildly boosts the user's cadre + general content, and answers are framed for that cadre's roles and facility level (grounded in the Roles & Responsibilities matrix + cadre manuals).
- **Source citation** — grounded answers append the real document title(s) used; off-topic/conversational replies do not.
- **Scope safety** — an independent classifier gates out-of-topic questions before answering.
- **No quiz feature** — the bot only answers questions via RAG.

## Project layout

| Path | Purpose |
|------|---------|
| `app/` | FastAPI app: webhook, handlers, RAG, session, utils |
| `scripts/ingest.py` | Build the FAISS index from PDFs (with OCR for image-based pages) |
| `scripts/chat.py` | Local CLI to test the bot without WhatsApp |
| `data/faiss_index/` | Prebuilt vector index (committed; ~2.4 MB) |
| `deploy/` | GCP VM deployment: `DEPLOY.md`, systemd unit, Caddyfile |
| `tests/` | pytest suite |

## Local setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
cp .env.example .env          # then fill in OPENAI_API_KEY and Meta credentials
python scripts/chat.py        # interactive local test (uses the committed index)
```

To rebuild the index from source files (PDF/DOCX/PPTX in `data/training_materials/`;
Tesseract installed for OCR of image-based PDF pages; PPTX SmartArt and per-cadre
tagging handled automatically):

```bash
python scripts/ingest.py
```

## Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Deployment

See **[deploy/DEPLOY.md](deploy/DEPLOY.md)** for the full Google Cloud VM deployment
(DuckDNS hostname + Caddy auto-HTTPS + systemd service).

## Configuration

All via environment variables (see `.env.example`): Meta WhatsApp Cloud API
credentials, `OPENAI_API_KEY`, session backend (memory/redis), index paths, optional
Google Sheets logging, and OCR settings (ingestion only).

> **Secrets** (`.env`, `credentials/`) are git-ignored and must never be committed.
