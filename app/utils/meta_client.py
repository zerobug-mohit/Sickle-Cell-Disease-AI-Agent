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
