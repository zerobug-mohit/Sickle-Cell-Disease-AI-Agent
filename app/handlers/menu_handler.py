from app.rag.prompts import MENU


async def handle_menu(session: dict) -> str:
    lang = session.get("lang", "en")
    return MENU.get(lang, MENU["en"])
