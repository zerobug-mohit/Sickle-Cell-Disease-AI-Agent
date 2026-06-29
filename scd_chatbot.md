# SCD WhatsApp Chatbot — Build Guide

## What you are building

A WhatsApp chatbot for health workers (ASHAs, MPWs, CHOs, Staff Nurses, Medical Officers) working under India's National Sickle Cell Anaemia Elimination Mission (NSCAEM). Users send questions in English, Hindi (Devanagari), or Hinglish and receive answers grounded in the 6 Government of India SCD training PDFs stored in the `Context data/` folder.

**This bot has NO quiz feature.** Do not build anything quiz-related. The bot does one thing: answer questions using RAG.

---

## Architecture

```
User (WhatsApp)
    down  Meta Webhook POST /webhook
FastAPI (main.py)
    down  route_message()
message_router.py   <- greetings, /help, /stop, else -> QA
    down
qa_handler.py
    |- gpt-4o-mini  ->  3 search queries (multi-query expansion + abbreviation expansion)
    |- text-embedding-3-small  ->  embed all 3 queries in one API call
    |- FAISS (local)  ->  top-5 chunks per query, deduplicated, re-ranked, top-3 passed to LLM
    +- gpt-4o-mini  ->  answer grounded in retrieved context
    down
meta_client.py  ->  WhatsApp reply sent back to user
    down
sheets_logger.py  ->  Google Sheets interaction log (optional, async, non-blocking)
```

Session state (language + conversation history) is stored in Redis or an in-memory dict. There is no mode switching — every message routes to QA unless it is a greeting, `/help`, or `/stop`.

---

## Directory structure to create

```
scd_whatsapp_chatbot/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── message_router.py
│   │   ├── menu_handler.py
│   │   └── qa_handler.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── vector_store.py
│   │   └── prompts.py
│   ├── session/
│   │   ├── __init__.py
│   │   └── store.py
│   └── utils/
│       ├── __init__.py
│       ├── language.py
│       ├── meta_client.py
│       └── sheets_logger.py
├── Context data/               <- PDFs already here, do NOT move or rename this folder
│   ├── GoI_CHO SCD Training Manual (English).pdf
│   ├── GoI_Counseling Module (English).pdf
│   ├── GoI_MO SCD Training Manual (English).pdf
│   ├── GoI_MPW and Asha SCD Training Manual (English).pdf
│   ├── GoI_NSCAEM Operational Guidelines.pdf
│   └── GoI_Staff Nurse SCD Training Manual (English).pdf
├── data/
│   ├── faiss_index/            <- created by ingest.py (do not create manually)
│   │   ├── index.faiss
│   │   └── metadata.json
│   └── training_materials/     <- copy the 6 PDFs here before running ingest.py
├── credentials/
│   └── google_service_account.json   <- copy from CHO bot project credentials/
├── scripts/
│   ├── ingest.py
│   └── chat.py
├── tests/
│   ├── __init__.py
│   └── test_rag.py
├── .env
├── .env.example
└── requirements.txt
```

---

## Build order

Follow this order exactly.

1. Create all `__init__.py` files (empty)
2. Write `requirements.txt`
3. Write `app/config.py`
4. Write `app/rag/vector_store.py`
5. Write `app/session/store.py`
6. Write `app/utils/language.py`
7. Write `app/utils/meta_client.py`
8. Write `app/utils/sheets_logger.py`
9. Write `app/rag/prompts.py`
10. Write `app/handlers/menu_handler.py`
11. Write `app/handlers/qa_handler.py`
12. Write `app/handlers/message_router.py`
13. Write `app/main.py`
14. Write `scripts/ingest.py`
15. Write `scripts/chat.py`
16. Write `tests/test_rag.py`
17. Write `.env.example` and `.env`
18. Copy the 6 PDFs from `Context data/` into `data/training_materials/`
19. Run `python scripts/ingest.py` to build the FAISS index
20. Test locally with `python scripts/chat.py`

---

## File specifications — write these exactly as shown

### `requirements.txt`

```
fastapi
uvicorn[standard]
httpx
openai>=1.0.0
faiss-cpu
numpy
pypdf
python-docx
langchain
langchain-community
langchain-text-splitters
tiktoken
python-dotenv
redis
gspread
google-auth
pytest
pytest-asyncio
```

---

### `app/config.py`

```python
from dotenv import load_dotenv
import os

load_dotenv()

META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "")
META_PHONE_NUMBER_ID = os.getenv("META_PHONE_NUMBER_ID", "")
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "")
META_APP_SECRET = os.getenv("META_APP_SECRET", "")
WHATSAPP_API_VERSION = os.getenv("WHATSAPP_API_VERSION", "v19.0")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

PORT = int(os.getenv("PORT", "8000"))
SESSION_BACKEND = os.getenv("SESSION_BACKEND", "memory")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
TRAINING_MATERIALS_DIR = os.getenv("TRAINING_MATERIALS_DIR", "./data/training_materials")

# Google Sheets logging (optional — leave blank to disable)
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "")
INTERACTION_SHEET_ID = os.getenv("INTERACTION_SHEET_ID", "")
```

There is no `QUIZ_QUESTIONS_PATH` — this project has no quiz.

---

### `app/rag/vector_store.py`

```python
import json
import os
from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class SearchResult:
    text: str
    source: str
    chunk_id: str
    score: float


class VectorStore:
    def __init__(self) -> None:
        self._index = None
        self._metadata: list[dict] = []

    def load(self, index_dir: str) -> None:
        index_path = os.path.join(index_dir, "index.faiss")
        meta_path = os.path.join(index_dir, "metadata.json")
        self._index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

    def search(self, embedding: list[float], top_k: int = 3) -> list[SearchResult]:
        if self._index is None:
            raise RuntimeError("Vector store not loaded. Call load() at startup.")
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            results.append(SearchResult(
                text=meta["text"],
                source=meta["source"],
                chunk_id=meta["chunk_id"],
                score=float(score),
            ))
        return results


# Module-level singleton — loaded once at startup, shared across all handlers
vector_store = VectorStore()
```

---

### `app/session/store.py`

```python
import json
from app.config import SESSION_BACKEND, REDIS_URL

_memory: dict[str, dict] = {}
_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is None:
        import redis
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def get_session(phone: str) -> dict:
    if SESSION_BACKEND == "redis":
        data = _get_redis().get(f"session:{phone}")
        return json.loads(data) if data else {"lang": "en"}
    if phone not in _memory:
        _memory[phone] = {"lang": "en"}
    return _memory[phone]


def save_session(phone: str, data: dict) -> None:
    if SESSION_BACKEND == "redis":
        _get_redis().set(f"session:{phone}", json.dumps(data), ex=86400)
        return
    _memory[phone] = data


def clear_session(phone: str) -> None:
    if SESSION_BACKEND == "redis":
        _get_redis().delete(f"session:{phone}")
        return
    _memory.pop(phone, None)
```

Note: default session is `{"lang": "en"}` only — no `"mode"` key needed since there is no mode switching.

---

### `app/utils/language.py`

```python
import re

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")

_HINGLISH_MARKERS = {
    "hai", "hain", "nahi", "nahin", "kya", "kab", "kaise", "kaisa",
    "aur", "se", "mein", "ke", "ka", "ki", "ko", "par",
    "tha", "thi", "the", "hoga", "hogi",
    "batao", "bolo", "dena", "karo", "karna", "kijiye", "dijiye", "lijiye",
    "chahiye", "lagti", "lagta", "milta", "milti",
    "wala", "wali", "raha", "rahi", "sakte", "sakti",
    "bahut", "thoda", "accha", "theek", "jaldi", "abhi",
    "poochho", "poochh", "samjho", "samajh",
    "mujhe", "aapko", "unko", "humko", "tumhe",
    "bimari", "dawai", "dard", "khoon", "jaanch",
}


def detect_language(text: str) -> str:
    """Return 'hi', 'hinglish', or 'en' based on the input text."""
    if _DEVANAGARI_RE.search(text):
        return "hi"
    words = set(re.findall(r"\b\w+\b", text.lower()))
    if words & _HINGLISH_MARKERS:
        return "hinglish"
    return "en"
```

---

### `app/utils/meta_client.py`

```python
import logging
import httpx
from app.config import META_ACCESS_TOKEN, META_PHONE_NUMBER_ID, WHATSAPP_API_VERSION

logger = logging.getLogger(__name__)

_BASE_URL = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}"
_MAX_CHUNK = 1500


def _split_text(text: str) -> list[str]:
    if len(text) <= _MAX_CHUNK:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= _MAX_CHUNK:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, _MAX_CHUNK)
        if split_at == -1:
            split_at = text.rfind(" ", 0, _MAX_CHUNK)
        if split_at == -1:
            split_at = _MAX_CHUNK
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    return chunks


async def send_message(to: str, body: str) -> None:
    chunks = _split_text(body)
    async with httpx.AsyncClient(timeout=10.0) as client:
        for chunk in chunks:
            resp = await client.post(
                f"{_BASE_URL}/{META_PHONE_NUMBER_ID}/messages",
                headers={"Authorization": f"Bearer {META_ACCESS_TOKEN}"},
                json={
                    "messaging_product": "whatsapp",
                    "to": to,
                    "type": "text",
                    "text": {"body": chunk},
                },
            )
            if resp.status_code != 200:
                logger.error("Meta API %s: %s", resp.status_code, resp.text)
```

---

### `app/utils/sheets_logger.py`

No quiz logging — only `log_interaction`.

```python
import asyncio
import hashlib
import logging
from datetime import datetime
from functools import lru_cache

from app import config

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

_INTERACTION_HEADERS = [
    "Timestamp (UTC)", "Interaction ID", "User ID", "Username",
    "Success", "Error Message", "Processing Time (ms)", "Good Example",
    "Language", "User Message", "Bot Response",
]


def _phone_hash(phone: str) -> str:
    return hashlib.sha256(phone.encode()).hexdigest()[:12]


@lru_cache(maxsize=1)
def _get_client():
    import gspread
    return gspread.service_account(
        filename=config.GOOGLE_CREDENTIALS_PATH,
        scopes=_SCOPES,
    )


def _ensure_headers(sheet, headers: list[str]) -> None:
    first_row = sheet.row_values(1)
    if not first_row:
        sheet.append_row(headers, value_input_option="USER_ENTERED")


def _append(sheet_id: str, row: list, headers: list[str]) -> None:
    ws = _get_client().open_by_key(sheet_id).sheet1
    _ensure_headers(ws, headers)
    ws.append_row(row, value_input_option="USER_ENTERED")


async def log_interaction(
    phone: str,
    interaction_id: str,
    username: str,
    user_msg: str,
    bot_reply: str,
    lang: str,
    success: bool,
    error_msg: str,
    processing_ms: int,
) -> None:
    if not config.INTERACTION_SHEET_ID or not config.GOOGLE_CREDENTIALS_PATH:
        return
    row = [
        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        interaction_id,
        _phone_hash(phone),
        username,
        "TRUE" if success else "FALSE",
        error_msg,
        processing_ms,
        "",  # Good Example — filled manually for fine-tuning
        lang,
        user_msg,
        bot_reply,
    ]
    try:
        await asyncio.to_thread(_append, config.INTERACTION_SHEET_ID, row, _INTERACTION_HEADERS)
    except Exception as e:
        logger.error("Sheets log_interaction failed: %s", e)
```

---

### `app/rag/prompts.py`

All SCD-specific content. Write this file in full exactly as shown.

```python
QA_SYSTEM_PROMPT = """\
You are a friendly training assistant for health workers (ASHAs, MPWs, CHOs, Staff Nurses, and Medical Officers) \
working under India's National Sickle Cell Anaemia Elimination Mission (NSCAEM).
Your users are health workers learning about Sickle Cell Disease on the job -- they may ask vague, comparative, \
or imprecise questions. Your job is to guide them to the right understanding, not to reject their questions.

How to answer:
- Use the provided context as your primary source. Do not invent medical facts, dosages, clinical criteria, or procedures.
- For vague or ambiguous questions, interpret charitably and answer the most likely intent based on the conversation so far.
- For comparison questions ("is HbSS worse than HbAS?", "what is the difference between a crisis and an infection?") \
explain how the two things relate or differ using what the context tells you. Do not refuse just because the context \
does not directly compare them.
- For follow-up questions with pronouns ("it", "that disease", "the same test", "ye", "woh") \
resolve them from the conversation history before answering.
- If the context only partially covers the question, answer what you can and acknowledge the gap briefly.
- Only decline to answer if the topic is completely unrelated to Sickle Cell Disease, NSCAEM, or related health \
worker duties (e.g. cooking, politics, sports). In that case say clearly that this bot covers SCD training topics only.

Style:
- Keep answers short -- maximum 3 short paragraphs or a brief bullet list.
- Use simple, encouraging language. These are field health workers, not specialists.
- Detect the language of the user's question and reply in the SAME language and script:
    * Hindi (Devanagari script) -> reply in Hindi Devanagari
    * Hinglish (Roman script Hindi) -> reply in Hinglish Roman script
    * English -> reply in English
- Do NOT mix scripts in a single reply.

Follow-up suggestions:
- Only include these when you have given a substantive answer drawn from the training material context.
- Do NOT include them if your response is introductory, meta, or redirecting.
- After a substantive answer, add a blank line, then suggest 2-3 follow-up questions the user could ask next.
- Ground every suggested question in content that is actually present in the retrieved context below.
- Do NOT suggest questions whose answers are not covered by the context.
- Format exactly like this (match the user's language/script):
  English  -> [light bulb] *You could also ask:*\n* Q1\n* Q2\n* Q3
  Hindi    -> [light bulb] *aap yeh bhi poochh sakte hain:*\n* Q1\n* Q2\n* Q3
  Hinglish -> [light bulb] *Aap yeh bhi poochh sakte hain:*\n* Q1\n* Q2\n* Q3

Context from training materials:
{retrieved_chunks}"""


FALLBACK = {
    "en": (
        "This question is outside the scope of the SCD training modules.\n\n"
        "I can only answer questions related to:\n"
        "* Sickle Cell Disease -- basics, genetics, and inheritance\n"
        "* Symptoms, complications, and when to refer\n"
        "* Diagnosis and community screening methods\n"
        "* Treatment and patient management\n"
        "* Counseling, NSCAEM guidelines, and community outreach\n\n"
        "Please ask a question on one of these topics."
    ),
    "hi": (
        "yeh prashn SCD prasikshan samagri ke dayre se bahar hai.\n\n"
        "main keval in vishyon par uttar de sakta hoon:\n"
        "* Sickle Cell Disease -- mool baaten aur aanuvamshikta\n"
        "* Lakshan, jatiltaen aur referral\n"
        "* Nidan aur samudayik screening\n"
        "* Upchar aur rogi prabandhan\n"
        "* Paramarsh aur NSCAEM disha-nirdesh\n\n"
        "Kripya inhi vishyon se sambandhit prashn poochhen."
    ),
    "hinglish": (
        "Yeh sawaal SCD training modules ke bahar ka hai.\n\n"
        "Main sirf in topics par jawaab de sakta hoon:\n"
        "* Sickle Cell Disease -- basics aur genetics\n"
        "* Symptoms, complications aur referral\n"
        "* Diagnosis aur community screening\n"
        "* Treatment aur patient management\n"
        "* Counseling aur NSCAEM guidelines\n\n"
        "Kripya inhi topics se related sawaal poochhen."
    ),
}

GOODBYE = {
    "en":       "Goodbye! Thank you for your commitment to fighting Sickle Cell Disease.\n\nNSCAEM -- National Sickle Cell Anaemia Elimination Mission",
    "hi":       "Alvida! Sickle Cell Disease se ladne ki aapki pratibaddhat ke liye dhanyavaad.\n\nNSCAEM -- Rashtriya Sickle Cell Anaemia Unmulan Mission",
    "hinglish": "Alvida! Sickle Cell Disease se ladne ki aapki commitment ke liye shukriya.\n\nNSCAEM -- National Sickle Cell Anaemia Elimination Mission",
}

EXIT_FOOTER = {
    "en":       "Type /help to return to the main menu anytime.",
    "hi":       "Kabhi bhi mukhya menu par wapas jaane ke liye /help likhein.",
    "hinglish": "Kabhi bhi main menu par jaane ke liye /help likhein.",
}

GREETING_PREFIX = {
    "en":       "Hello! Welcome to the SCD Health Worker Training Bot.\n\nHere is what you can ask about:",
    "hi":       "Namaste! SCD Health Worker Training Bot mein aapka swagat hai.\n\nAap in vishyon par prashn poochh sakte hain:",
    "hinglish": "Namaste! SCD Health Worker Training Bot mein aapka swagat hai.\n\nAap in topics par sawaal poochh sakte hain:",
}

MENU = {
    "en": """\
SCD Health Worker Training Bot
National Sickle Cell Anaemia Elimination Mission (NSCAEM)

Ask any question about Sickle Cell Disease:

---
Disease Basics and Genetics
* What is sickle cell disease and how is it inherited?
* What is the difference between HbSS and HbAS?
* Can two carriers (HbAS) have a child with SCD?
* What is the difference between a carrier and a patient?
* Why does the red blood cell become sickle-shaped?

Symptoms and Complications
* What are the signs of a sickle cell crisis?
* What is acute chest syndrome and how do I recognise it?
* Why are SCD patients more prone to infections?
* What complications should I watch for in a child with SCD?
* When does a patient with SCD need emergency referral?

Diagnosis and Screening
* How is sickle cell disease diagnosed?
* What is HPLC and when is it used?
* What does a point-of-care (Sickle SCAN) test show?
* At what age should newborn screening for SCD be done?
* How is community-level screening organised under NSCAEM?

Treatment and Management
* What is hydroxyurea and who should take it?
* Why do SCD patients need folic acid daily?
* How do I manage a pain crisis at the community level?
* What vaccines are especially important for SCD patients?
* When should I refer a patient to a higher facility?

Counseling and Community
* How do I counsel a couple before marriage about SCD risk?
* How do I explain a new SCD diagnosis to a family?
* What is my role in community-level SCD screening?
* How do I track and follow up SCD patients in my area?
---

Type /help to return here anytime
Type /stop to end this session.
For support, contact: mchaurasiya@wjcf.in""",

    "hi": """\
SCD Health Worker Training Bot
Rashtriya Sickle Cell Anaemia Unmulan Mission (NSCAEM)

Sickle Cell Disease ke baare mein koi bhi prashn poochhen:

---
Rog ki mool baaten aur aanuvamshikta
* Sickle cell disease kya hai aur yeh kaise phailta hai?
* HbSS aur HbAS mein kya antar hai?
* Kya do vahak (HbAS) ke bacche ko SCD ho sakti hai?
* Vahak aur marij mein kya fark hota hai?
* Lal rakt koshika hansia ke aakaar ki kyun ho jaati hai?

Lakshan aur Jatiltaen
* Sickle cell crisis ke kya sanket hain?
* Acute Chest Syndrome kya hai aur ise kaise pehchaanen?
* SCD mareezon ko sankraman adhik kyun hota hai?
* SCD wale bacche mein kaun si jatiltaon par dhyan den?
* SCD mareej ko kab turant refer karna chahiye?

Nidan aur Screening
* SCD ka nidan kaise hota hai?
* HPLC kya hai aur kab upyog hota hai?
* Point-of-care (Sickle SCAN) test kya batata hai?
* Navjat shishu ki SCD screening kab honi chahiye?
* NSCAEM ke tahat samudayik screening kaise hoti hai?

Upchar aur Prabandhan
* Hydroxyurea kya hai aur kise leni chahiye?
* SCD mareezon ko roz Folic Acid kyun chahiye?
* Samudaay star par dard ka sankat kaise sambhaalen?
* SCD mareezon ke liye kaun se tike zaroori hain?
* Mareej ko uchch aspatal kab bhejen?

Paramarsh aur Samudaay
* Shaadi se pehle jode ko SCD jokhim ke baare mein kaise samjayen?
* Naye SCD nidan ko parivar ko kaise batayen?
* Samudaay star par SCD screening mein meri kya bhumika hai?
* Apne kshetr mein SCD mareezon ko kaise track karen?
---

Wapas aane ke liye kabhi bhi /help likhein
Session khatam karne ke liye /stop likhein.
Sahayata ke liye: mchaurasiya@wjcf.in""",

    "hinglish": """\
SCD Health Worker Training Bot
National Sickle Cell Anaemia Elimination Mission (NSCAEM)

Sickle Cell Disease ke baare mein koi bhi sawaal poochhen:

---
Bimari ki basics aur genetics
* Sickle cell disease kya hai aur kaise milti hai?
* HbSS aur HbAS mein kya fark hai?
* Kya do carrier (HbAS) ke bacche ko SCD ho sakti hai?
* Carrier aur patient mein kya fark hota hai?
* Red blood cell haansia jaisi kyun ho jaati hai?

Symptoms aur Complications
* Sickle cell crisis ke kya signs hain?
* Acute Chest Syndrome kya hai aur kaise pehchaanen?
* SCD patients ko infection zyada kyun hota hai?
* SCD wale bacche mein kaunsi complications par dhyan den?
* SCD patient ko kab emergency mein refer karna chahiye?

Diagnosis aur Screening
* SCD ka diagnosis kaise hota hai?
* HPLC kya hai aur kab use hota hai?
* Point-of-care (Sickle SCAN) test kya batata hai?
* Naye shishu ki SCD screening kab honi chahiye?
* NSCAEM ke tahat community screening kaise hoti hai?

Ilaaj aur Management
* Hydroxyurea kya hai aur kise leni chahiye?
* SCD patients ko roz Folic Acid kyun chahiye?
* Community level par dard ka crisis kaise handle karein?
* SCD patients ke liye kaunse vaccine zaroori hain?
* Patient ko uchch hospital kab bhejna chahiye?

Counseling aur Community
* Shaadi se pehle couple ko SCD risk kaise samjayen?
* Naye SCD diagnosis ko family ko kaise batayein?
* Community SCD screening mein meri kya bhumika hai?
* Apne area mein SCD patients ko kaise track karein?
---

Kabhi bhi wapas aane ke liye /help likhein
Session khatam karne ke liye /stop likhein.
Support ke liye: mchaurasiya@wjcf.in""",
}
```

---

### `app/handlers/menu_handler.py`

```python
from app.rag.prompts import MENU


async def handle_menu(session: dict) -> str:
    lang = session.get("lang", "en")
    return MENU.get(lang, MENU["en"])
```

---

### `app/handlers/qa_handler.py`

Multi-query expansion uses SCD-specific abbreviations. All other logic is identical to the reference CHO bot -- gpt-4o-mini for both query expansion and answer generation, top-3 context chunks, 3-turn history.

```python
import json
import time
import uuid

from openai import AsyncOpenAI

from app import config
from app.rag.vector_store import vector_store, SearchResult
from app.rag.prompts import QA_SYSTEM_PROMPT, FALLBACK
from app.utils.language import detect_language
from app.session.store import save_session
from app.utils.sheets_logger import log_interaction

_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

_SIMILARITY_THRESHOLD = 0.10
_TOP_K = 5          # results per query from FAISS
_CONTEXT_CHUNKS = 3 # top chunks passed to the answer model
_HISTORY_TURNS = 3  # full turns (user + assistant pairs) kept in session

_MULTI_QUERY_PROMPT = """\
Generate 3 diverse search queries to find relevant sections in training documents about \
Sickle Cell Disease (SCD) for India's National Sickle Cell Anaemia Elimination Mission (NSCAEM).

Rules:
- Expand abbreviations: SCD -> "Sickle Cell Disease", \
HbSS -> "homozygous sickle cell disease", HbAS -> "sickle cell trait carrier", \
HbSC -> "sickle haemoglobin C disease", HbA -> "normal adult haemoglobin", \
HbF -> "fetal haemoglobin", NSCAEM -> "National Sickle Cell Anaemia Elimination Mission", \
VOC -> "vaso-occlusive crisis", ACS -> "acute chest syndrome", \
HPLC -> "high performance liquid chromatography", CBC -> "complete blood count", \
CHO -> "Community Health Officer", ASHA -> "Accredited Social Health Activist", \
MPW -> "Multi-Purpose Worker", MO -> "Medical Officer", \
POC -> "point-of-care test", SCA -> "sickle cell anaemia"
- Each query must approach the topic from a different angle \
(e.g. one factual, one about clinical procedure, one about signs or numbers)
- Use formal training-document vocabulary
- If the current question uses pronouns or references ("it", "that disease", "the same test", \
"ye bimari", "woh condition") resolve them using the conversation history before forming queries
{history_block}
Return JSON only: {{"queries": ["...", "...", "..."]}}

Current question: {query}"""


def _history_block(history: list[dict]) -> str:
    if not history:
        return ""
    lines = []
    for m in history:
        role = "User" if m["role"] == "user" else "Bot"
        lines.append(f"{role}: {m['content']}")
    return "\nConversation so far:\n" + "\n".join(lines) + "\n"


async def _generate_search_queries(text: str, history: list[dict]) -> list[str]:
    prompt = _MULTI_QUERY_PROMPT.format(
        query=text,
        history_block=_history_block(history),
    )
    resp = await _client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=250,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        queries = json.loads(resp.choices[0].message.content).get("queries", [])
        valid = [q for q in queries if isinstance(q, str) and q.strip()][:3]
        return valid if valid else [text]
    except (json.JSONDecodeError, KeyError, TypeError):
        return [text]


async def _multi_query_search(queries: list[str]) -> list[SearchResult]:
    embed_resp = await _client.embeddings.create(
        model="text-embedding-3-small",
        input=queries,
    )
    embeddings = [d.embedding for d in embed_resp.data]

    seen: set[str] = set()
    combined: list[SearchResult] = []
    for emb in embeddings:
        for result in vector_store.search(emb, top_k=_TOP_K):
            if result.chunk_id not in seen:
                seen.add(result.chunk_id)
                combined.append(result)

    combined.sort(key=lambda r: r.score, reverse=True)
    return combined


async def handle_qa(phone: str, text: str, session: dict, username: str = "") -> str:
    t0 = time.monotonic()
    interaction_id = uuid.uuid4().hex[:8]
    lang = detect_language(text)
    session["lang"] = lang
    error_msg = ""

    history: list[dict] = session.get("history", [])

    queries = await _generate_search_queries(text, history[-(_HISTORY_TURNS * 2):])
    results = await _multi_query_search(queries)

    if not results or results[0].score < _SIMILARITY_THRESHOLD:
        save_session(phone, session)
        reply = FALLBACK.get(lang, FALLBACK["en"])
        processing_ms = int((time.monotonic() - t0) * 1000)
        await log_interaction(phone, interaction_id, username, text, reply, lang, False, "below_threshold", processing_ms)
        return reply

    context = "\n\n---\n\n".join(r.text for r in results[:_CONTEXT_CHUNKS])
    system_prompt = QA_SYSTEM_PROMPT.format(retrieved_chunks=context)

    messages = [
        {"role": "system", "content": system_prompt},
        *history[-(_HISTORY_TURNS * 2):],
        {"role": "user", "content": text},
    ]

    response = await _client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=500,
        messages=messages,
    )
    reply = response.choices[0].message.content

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": reply})
    session["history"] = history[-(_HISTORY_TURNS * 2):]
    save_session(phone, session)

    processing_ms = int((time.monotonic() - t0) * 1000)
    await log_interaction(phone, interaction_id, username, text, reply, lang, True, "", processing_ms)
    return reply
```

---

### `app/handlers/message_router.py`

No quiz routing at all. Three command paths (stop, greeting, help) then QA for everything else.

```python
from app.session.store import get_session, save_session, clear_session
from app.handlers import menu_handler, qa_handler
from app.rag.prompts import EXIT_FOOTER, GOODBYE, GREETING_PREFIX, MENU
from app.utils.language import detect_language

_MENU_TRIGGERS = {"/help"}
_STOP_TRIGGERS = {"/stop"}
_GREETING_TRIGGERS = {
    # English
    "hi", "hey", "hello", "howdy", "greetings", "good morning", "good afternoon",
    "good evening", "good day", "hi there", "hey there",
    # Hindi Devanagari
    "namaste", "namaskar", "helo", "hay", "pranam",
    # Hinglish Roman
    "namaste", "namaskar", "salam", "salaam", "helo", "hii", "heya",
}


def _with_footer(reply: str, lang: str) -> str:
    return reply + "\n\n" + EXIT_FOOTER.get(lang, EXIT_FOOTER["en"])


async def route_message(phone: str, text: str, username: str = "") -> str:
    session = get_session(phone)
    normalized = text.strip().lower()
    lang = session.get("lang", "en")

    if normalized in _STOP_TRIGGERS:
        clear_session(phone)
        return GOODBYE.get(lang, GOODBYE["en"])

    if normalized in _GREETING_TRIGGERS:
        lang = detect_language(text)
        session["lang"] = lang
        save_session(phone, session)
        prefix = GREETING_PREFIX.get(lang, GREETING_PREFIX["en"])
        return prefix + "\n\n" + MENU.get(lang, MENU["en"])

    if normalized in _MENU_TRIGGERS:
        return await menu_handler.handle_menu(session)

    reply = await qa_handler.handle_qa(phone, text, session, username=username)
    return _with_footer(reply, session.get("lang", "en"))
```

---

### `app/main.py`

No quiz imports, no `question_bank.load()`.

```python
import hashlib
import hmac
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse

from app import config
from app.rag.vector_store import vector_store
from app.handlers.message_router import route_message
from app.utils.meta_client import send_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    vector_store.load(config.FAISS_INDEX_PATH)
    logger.info("Vector store loaded. Ready.")
    yield


app = FastAPI(title="SCD WhatsApp Bot", lifespan=lifespan)


def _verify_signature(body: bytes, signature_header: str) -> bool:
    if not config.META_APP_SECRET:
        return True
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(
        config.META_APP_SECRET.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header[7:])


@app.get("/webhook")
async def verify_webhook(request: Request) -> PlainTextResponse:
    params = dict(request.query_params)
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge", "")
    if mode == "subscribe" and token == config.META_VERIFY_TOKEN:
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def receive_webhook(request: Request) -> Response:
    body = await request.body()
    sig = request.headers.get("X-Hub-Signature-256", "")

    if not _verify_signature(body, sig):
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    try:
        entry = data.get("entry", [])
        if not entry:
            return Response(status_code=200)

        changes = entry[0].get("changes", [])
        if not changes:
            return Response(status_code=200)

        value = changes[0].get("value", {})

        if "statuses" in value and "messages" not in value:
            return Response(status_code=200)

        messages = value.get("messages", [])
        if not messages:
            return Response(status_code=200)

        msg = messages[0]
        if msg.get("type") != "text":
            return Response(status_code=200)

        phone: str = msg["from"]
        text: str = msg["text"]["body"]
        contacts = value.get("contacts", [])
        username: str = contacts[0].get("profile", {}).get("name", "") if contacts else ""

        phone_hash = hashlib.sha256(phone.encode()).hexdigest()[:8]
        logger.info("msg from=%s len=%d", phone_hash, len(text))

        reply = await route_message(phone, text, username=username)
        await send_message(phone, reply)

    except Exception:
        logger.exception("Error processing webhook")

    return Response(status_code=200)
```

---

### `scripts/ingest.py`

```python
#!/usr/bin/env python3
"""
Run ONCE to build the FAISS vector store from SCD training materials.
Re-run only when source documents change.

Usage:
    python scripts/ingest.py
"""
import json
import os
import sys

import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import TRAINING_MATERIALS_DIR, FAISS_INDEX_PATH, OPENAI_API_KEY
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 40
EMBED_BATCH_SIZE = 100

client = OpenAI(api_key=OPENAI_API_KEY)


def extract_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def extract_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_documents(directory: str) -> list[tuple[str, str]]:
    docs = []
    for fname in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.lower().endswith(".pdf"):
            print(f"  Reading PDF: {fname}")
            docs.append((fname, extract_pdf(fpath)))
        elif fname.lower().endswith(".docx"):
            print(f"  Reading DOCX: {fname}")
            docs.append((fname, extract_docx(fpath)))
    return docs


def chunk_documents(docs: list[tuple[str, str]]) -> tuple[list[str], list[dict]]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_texts: list[str] = []
    all_meta: list[dict] = []

    for source, text in docs:
        if not text.strip():
            print(f"  WARNING: {source} yielded no extractable text.")
            print(f"  This PDF may be image-based (scanned). See CLAUDE.md for the pymupdf fallback.")
            continue
        chunks = splitter.split_text(text)
        print(f"  {source}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_meta.append({
                "chunk_id": f"{source}_{i}",
                "text": chunk,
                "source": source,
            })

    return all_texts, all_meta


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    total_batches = (len(chunks) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend([d.embedding for d in resp.data])
        batch_num = i // EMBED_BATCH_SIZE + 1
        print(f"  Embedded batch {batch_num}/{total_batches} ({len(all_embeddings)}/{len(chunks)} chunks)")

    return all_embeddings


def build_and_save_index(embeddings: list[list[float]], metadata: list[dict]) -> None:
    emb_array = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(emb_array)

    dim = emb_array.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_array)

    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(FAISS_INDEX_PATH, "index.faiss"))
    with open(os.path.join(FAISS_INDEX_PATH, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nFAISS index saved -> {FAISS_INDEX_PATH}/")
    print(f"  Vectors: {index.ntotal}  Dimensions: {dim}")


def main() -> None:
    print(f"Loading documents from: {TRAINING_MATERIALS_DIR}")
    docs = load_documents(TRAINING_MATERIALS_DIR)
    if not docs:
        print("No PDF or DOCX files found. Copy the PDFs to data/training_materials/ and re-run.")
        sys.exit(1)
    print(f"Loaded {len(docs)} document(s)\n")

    print("Chunking documents...")
    texts, metadata = chunk_documents(docs)
    if not texts:
        print("ERROR: All documents yielded zero text chunks.")
        print("The PDFs are likely scanned/image-based. Install pymupdf and use the fallback in CLAUDE.md.")
        sys.exit(1)
    print(f"Total chunks: {len(texts)}\n")

    print("Embedding chunks (may take several minutes for large documents)...")
    embeddings = embed_chunks(texts)
    print()

    print("Building FAISS index...")
    build_and_save_index(embeddings, metadata)
    print(f"\nDone. {len(docs)} documents -> {len(texts)} chunks indexed.")


if __name__ == "__main__":
    main()
```

---

### `scripts/chat.py`

```python
#!/usr/bin/env python3
"""
Local CLI to test the SCD bot end-to-end without WhatsApp.

Usage:
    python scripts/chat.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.config import FAISS_INDEX_PATH
from app.rag.vector_store import vector_store
from app.handlers.message_router import route_message

TEST_PHONE = "test_user_cli"


def _startup() -> None:
    print("Loading vector store...", end=" ", flush=True)
    vector_store.load(FAISS_INDEX_PATH)
    print("done.\n")


async def _repl() -> None:
    print("=" * 50)
    print("  SCD Bot - local test console")
    print("  Type /help for menu, /stop to end session")
    print("  or ask any question about Sickle Cell Disease")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        try:
            reply = await route_message(TEST_PHONE, user_input)
            print(f"\nBot: {reply}\n")
        except Exception as e:
            print(f"\n[error] {e}\n")


def main() -> None:
    _startup()
    asyncio.run(_repl())


if __name__ == "__main__":
    main()
```

---

### `tests/test_rag.py`

```python
import os
import pytest
from app.rag.vector_store import VectorStore


def test_vector_store_search_returns_list():
    index_path = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
    if not os.path.exists(os.path.join(index_path, "index.faiss")):
        pytest.skip("FAISS index not built yet -- run scripts/ingest.py first")
    vs = VectorStore()
    vs.load(index_path)
    results = vs.search([0.0] * 1536, top_k=3)
    assert isinstance(results, list)


def test_vector_store_score_ordering():
    index_path = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
    if not os.path.exists(os.path.join(index_path, "index.faiss")):
        pytest.skip("FAISS index not built yet -- run scripts/ingest.py first")
    vs = VectorStore()
    vs.load(index_path)
    results = vs.search([0.0] * 1536, top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
```

---

### `.env.example`

```
# Meta WhatsApp Cloud API
META_ACCESS_TOKEN=EAAxxxxxxx
META_PHONE_NUMBER_ID=1234567890
META_VERIFY_TOKEN=your_chosen_secret
META_APP_SECRET=your_facebook_app_secret
WHATSAPP_API_VERSION=v19.0

# OpenAI
OPENAI_API_KEY=sk-...

# App
PORT=8000
SESSION_BACKEND=memory
REDIS_URL=redis://localhost:6379

# Paths
FAISS_INDEX_PATH=./data/faiss_index
TRAINING_MATERIALS_DIR=./data/training_materials

# Google Sheets logging (leave blank to disable)
GOOGLE_CREDENTIALS_PATH=./credentials/google_service_account.json
INTERACTION_SHEET_ID=your_sheet_id_here
```

Create `.env` by copying `.env.example` and filling in real values.

---

## Ingestion — read this before running

### Step 1 — copy PDFs into the training materials folder

Create `data/training_materials/` and copy all 6 PDFs from `Context data/` into it:

```
GoI_CHO SCD Training Manual (English).pdf          -> data/training_materials/
GoI_Counseling Module (English).pdf                -> data/training_materials/
GoI_MO SCD Training Manual (English).pdf           -> data/training_materials/
GoI_MPW and Asha SCD Training Manual (English).pdf -> data/training_materials/
GoI_NSCAEM Operational Guidelines.pdf              -> data/training_materials/
GoI_Staff Nurse SCD Training Manual (English).pdf  -> data/training_materials/
```

### Step 2 — install dependencies

```
pip install -r requirements.txt
```

### Step 3 — run ingestion

```
python scripts/ingest.py
```

Expected output: 1,000 to 4,000 total chunks across all 6 documents. The two largest PDFs (CHO manual ~285 MB, MPW+ASHA manual ~149 MB) will take the most time. Total runtime: 5 to 20 minutes.

### Step 4 — verify the index

```
python -c "
from app.rag.vector_store import VectorStore
vs = VectorStore()
vs.load('./data/faiss_index')
print('Total vectors:', vs._index.ntotal)
"
```

A healthy index shows 1,000+ vectors. If the count is 0 or very low, the PDFs are image-based (scanned) and pypdf could not extract their text. In that case:

1. Install pymupdf: `pip install pymupdf`
2. Replace the `extract_pdf` function in `scripts/ingest.py` with:

```python
def extract_pdf(path: str) -> str:
    import fitz  # pymupdf
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)
```

3. Delete `data/faiss_index/` and re-run `python scripts/ingest.py`

---

## Testing locally

```
python scripts/chat.py
```

Test with these inputs to verify everything works end to end:

- `hello` -- should show the full menu
- `What is sickle cell disease?` -- should answer from context
- `HbSS mein kya hota hai?` -- should detect Hinglish and respond in Hinglish
- `What is hydroxyurea used for?` -- should answer from context
- `/help` -- should show the menu again
- `/stop` -- should end the session with goodbye

---

## Running the server

```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Deployment on Ubuntu VM

1. Clone the repo, create `.env` with real values
2. Copy `credentials/google_service_account.json` (same service account as CHO bot works if the new sheet is shared with it; or create a new one)
3. `pip install -r requirements.txt`
4. Copy PDFs to `data/training_materials/` and run `python scripts/ingest.py`
5. Create `/etc/systemd/system/scd-bot.service`:

```
[Unit]
Description=SCD WhatsApp Bot
After=network.target

[Service]
User=YOUR_USER
WorkingDirectory=/path/to/scd_whatsapp_chatbot
ExecStart=/path/to/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
EnvironmentFile=/path/to/scd_whatsapp_chatbot/.env

[Install]
WantedBy=multi-user.target
```

6. `sudo systemctl enable scd-bot && sudo systemctl start scd-bot`
7. Expose port 8000 via nginx reverse proxy with HTTPS (Meta requires HTTPS for webhooks)
8. Register the webhook in Meta App Dashboard -> WhatsApp -> Configuration -> Webhook URL: `https://yourdomain.com/webhook`

---

## What NOT to build

- No `app/quiz/` directory
- No quiz handler, quiz session, or question bank
- No `/quiz` command or quiz topic selection
- No `QUIZ_SHEET_ID` or `log_quiz_result`
- No session mode switching -- the router has no mode checks; every message that is not a greeting, `/help`, or `/stop` goes directly to `handle_qa`

---

## Google Sheets setup (optional)

1. Create a Google Sheet and copy its ID from the URL
2. Share it with the service account email in `credentials/google_service_account.json`
3. Set `INTERACTION_SHEET_ID` in `.env`

Column headers written automatically on first log entry:
`Timestamp (UTC)` | `Interaction ID` | `User ID` | `Username` | `Success` | `Error Message` | `Processing Time (ms)` | `Good Example` | `Language` | `User Message` | `Bot Response`

If `INTERACTION_SHEET_ID` or `GOOGLE_CREDENTIALS_PATH` is blank, logging is silently skipped with no impact on bot operation.