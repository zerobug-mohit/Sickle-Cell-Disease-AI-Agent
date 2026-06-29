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
