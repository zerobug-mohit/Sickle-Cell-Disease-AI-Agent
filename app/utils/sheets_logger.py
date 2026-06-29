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
