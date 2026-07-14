from app.session.store import get_session, save_session, clear_session
from app.handlers import qa_handler
from app.rag.prompts import (
    EXIT_FOOTER, GOODBYE,
    ONBOARD_ASK_STATE, ONBOARD_ASK_DISTRICT, ONBOARD_ASK_CADRE,
    ONBOARD_INVALID_CADRE, parse_cadre,
)
from app.utils.language import detect_language

_MENU_TRIGGERS = {"/help", "/menu"}
_STOP_TRIGGERS = {"/stop"}
_PROFILE_TRIGGERS = {"/profile"}
_GREETING_TRIGGERS = {
    # English
    "hi", "hey", "hello", "howdy", "greetings", "good morning", "good afternoon",
    "good evening", "good day", "hi there", "hey there",
    # Hindi Devanagari
    "namaste", "namaskar", "helo", "hay", "pranam",
    # Hinglish Roman
    "salam", "salaam", "hii", "heya",
}

# Profile fields cleared when (re)starting onboarding — includes the cached welcome.
_PROFILE_KEYS = ("state", "district", "cadre", "cadre_name", "facility", "welcome")


def _with_footer(reply: str, lang: str) -> str:
    return reply + "\n\n" + EXIT_FOOTER.get(lang, EXIT_FOOTER["en"])


def _profile_complete(session: dict) -> bool:
    return bool(session.get("cadre"))


def _start_onboarding(phone: str, session: dict) -> str:
    session["awaiting"] = "state"
    for k in _PROFILE_KEYS:
        session.pop(k, None)
    save_session(phone, session)
    return ONBOARD_ASK_STATE


async def _handle_onboarding(phone: str, text: str, session: dict) -> str:
    awaiting = session.get("awaiting")
    value = text.strip()

    if awaiting == "state":
        session["state"] = value
        session["awaiting"] = "district"
        save_session(phone, session)
        return ONBOARD_ASK_DISTRICT

    if awaiting == "district":
        session["district"] = value
        session["awaiting"] = "cadre"
        save_session(phone, session)
        return ONBOARD_ASK_CADRE

    if awaiting == "cadre":
        cadre = parse_cadre(value)
        if not cadre:
            return ONBOARD_INVALID_CADRE
        session["cadre"] = cadre["code"]
        session["cadre_name"] = cadre["name"]
        session["facility"] = cadre["facility"]
        session.pop("awaiting", None)
        # Cadre-tailored single-message welcome (generated once, cached in session).
        welcome = await qa_handler.build_welcome(session)
        save_session(phone, session)
        return welcome

    # No awaiting set yet -> first contact: greet and ask for the first field.
    return _start_onboarding(phone, session)


async def route_message(phone: str, text: str, username: str = "") -> str:
    session = get_session(phone)
    normalized = text.strip().lower()
    lang = session.get("lang", "en")

    # Global commands (work at any time)
    if normalized in _STOP_TRIGGERS:
        clear_session(phone)
        return GOODBYE.get(lang, GOODBYE["en"])

    if normalized in _PROFILE_TRIGGERS:
        return _start_onboarding(phone, session)

    # Profile capture must complete before normal Q&A.
    if not _profile_complete(session):
        return await _handle_onboarding(phone, text, session)

    # Greetings and /help both show the cadre-tailored welcome (cached).
    if normalized in _GREETING_TRIGGERS or normalized in _MENU_TRIGGERS:
        welcome = await qa_handler.build_welcome(session)
        save_session(phone, session)
        return welcome

    reply = await qa_handler.handle_qa(phone, text, session, username=username)
    return _with_footer(reply, session.get("lang", "en"))
