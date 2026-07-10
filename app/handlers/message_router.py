from app.session.store import get_session, save_session, clear_session
from app.handlers import menu_handler, qa_handler
from app.rag.prompts import (
    EXIT_FOOTER, GOODBYE, GREETING_PREFIX, MENU,
    ONBOARD_ASK_STATE, ONBOARD_ASK_DISTRICT, ONBOARD_ASK_CADRE,
    ONBOARD_INVALID_CADRE, profile_confirm, parse_cadre,
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


def _with_footer(reply: str, lang: str) -> str:
    return reply + "\n\n" + EXIT_FOOTER.get(lang, EXIT_FOOTER["en"])


def _profile_complete(session: dict) -> bool:
    return bool(session.get("cadre"))


def _start_onboarding(phone: str, session: dict) -> str:
    session["awaiting"] = "state"
    session.pop("state", None)
    session.pop("district", None)
    session.pop("cadre", None)
    session.pop("cadre_name", None)
    session.pop("facility", None)
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
        save_session(phone, session)
        lang = session.get("lang", "en")
        confirm = profile_confirm(cadre["name"], session.get("district", ""), session.get("state", ""))
        return confirm + "\n\n" + MENU.get(lang, MENU["en"])

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
