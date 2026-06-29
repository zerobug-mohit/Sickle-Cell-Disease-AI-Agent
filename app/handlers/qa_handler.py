import asyncio
import json
import re
import time
import uuid

from openai import AsyncOpenAI

from app import config
from app.rag.vector_store import vector_store, SearchResult
from app.rag.prompts import QA_SYSTEM_PROMPT, FALLBACK, SOURCE_TITLES, SOURCE_LABEL
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


_SCOPE_CLASSIFIER_PROMPT = """\
You are a strict topic gate for a chatbot that answers ONLY questions about Sickle Cell Disease (SCD), \
India's National Sickle Cell Anaemia Elimination Mission (NSCAEM), and the SCD-related duties of health workers \
(ASHA, MPW, CHO, Staff Nurse, Medical Officer) -- e.g. genetics, inheritance, carriers, symptoms, complications, \
crises, diagnosis, screening, tests (HPLC, Sickle SCAN), treatment (hydroxyurea, folic acid, transfusion, vaccines), \
referral, patient management, counseling, community outreach, and the NSCAEM programme/guidelines.

Decide whether the user's latest message belongs to this scope. Use the conversation history to resolve short \
follow-ups and pronouns ("what about it?", "aur uska ilaaj?", "ye kaise hota hai?") -- if the prior turn was about \
SCD, such follow-ups are IN scope.

IN scope: any SCD / NSCAEM / SCD-related health-worker topic, including vague, comparative, or one-word follow-ups.
OUT of scope: unrelated topics (cooking, sports, politics, movies, travel, general tech, finance, diseases unrelated \
to SCD) and pure small talk / thanks / chit-chat with no SCD question.

When genuinely uncertain, choose in_scope = true (prefer to help).
{history_block}
Return JSON only: {{"in_scope": true}} or {{"in_scope": false}}

User message: {query}"""


async def _classify_scope(text: str, history: list[dict]) -> bool:
    """Independent gate: is the user's message within SCD scope? Fails open (True)."""
    prompt = _SCOPE_CLASSIFIER_PROMPT.format(
        query=text,
        history_block=_history_block(history),
    )
    try:
        resp = await _client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=10,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = json.loads(resp.choices[0].message.content)
        return bool(parsed.get("in_scope", True))
    except Exception:
        # On any failure, prefer to answer rather than wrongly block a real question.
        return True


def _clean_text(text: str) -> str:
    """Normalize whitespace so replies use single line spacing.

    Strips trailing spaces per line and collapses any run of blank lines into a
    single blank line, so stray model whitespace never reaches the user.
    """
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _format_sources(results: list[SearchResult], lang: str) -> str:
    """Build a localized source line from the chunks actually fed to the model.

    De-duplicates by document title, preserving retrieval order. Returns "" if
    there is nothing to cite (caller then appends nothing).
    """
    titles: list[str] = []
    for r in results:
        title = SOURCE_TITLES.get(r.source, r.source)
        if title not in titles:
            titles.append(title)
    if not titles:
        return ""
    singular, plural = SOURCE_LABEL.get(lang, SOURCE_LABEL["en"])
    label = singular if len(titles) == 1 else plural
    return label + "\n" + "\n".join(f"* {t}" for t in titles)


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
    recent_history = history[-(_HISTORY_TURNS * 2):]

    # Independent scope gate runs concurrently with query expansion so the common
    # (in-scope) path pays no extra latency. This is the authoritative off-topic
    # check: it judges the real user question, before query-expansion can launder
    # an unrelated question into SCD-shaped search queries.
    in_scope_pre, queries = await asyncio.gather(
        _classify_scope(text, recent_history),
        _generate_search_queries(text, recent_history),
    )

    if not in_scope_pre:
        save_session(phone, session)
        reply = FALLBACK.get(lang, FALLBACK["en"])
        processing_ms = int((time.monotonic() - t0) * 1000)
        await log_interaction(phone, interaction_id, username, text, reply, lang, False, "out_of_scope", processing_ms)
        return reply

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
        max_tokens=600,
        response_format={"type": "json_object"},
        messages=messages,
    )
    raw = response.choices[0].message.content or ""

    # The model returns {"refused": bool, "answer": str}. "refused" is true only
    # when it declined an unrelated question (a discrete action it reports
    # reliably), so a genuine answer — even a partial one — always keeps its
    # source. If parsing fails, treat the whole output as the answer and cite
    # (safer than dropping a real source).
    refused = False
    answer = raw
    try:
        parsed = json.loads(raw)
        answer = parsed.get("answer") or raw
        refused = bool(parsed.get("refused", False))
    except (json.JSONDecodeError, AttributeError, TypeError):
        answer = raw
    answer = _clean_text(answer)

    # History keeps the clean answer (no source line) so the model does not learn
    # to echo citations itself — sources are appended deterministically below.
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": answer})
    session["history"] = history[-(_HISTORY_TURNS * 2):]
    save_session(phone, session)

    # Cite the documents whose chunks were actually passed to the model — but only
    # when the bot genuinely answered from them (not a conversational decline).
    reply = answer
    if not refused:
        source_line = _format_sources(results[:_CONTEXT_CHUNKS], lang)
        if source_line:
            reply = answer + "\n\n" + source_line

    processing_ms = int((time.monotonic() - t0) * 1000)
    await log_interaction(phone, interaction_id, username, text, reply, lang, True, "", processing_ms)
    return reply
