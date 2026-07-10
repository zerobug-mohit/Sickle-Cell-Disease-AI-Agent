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
            st = value["statuses"][0]
            logger.info("status=%s recipient=%s errors=%s",
                        st.get("status"), st.get("recipient_id"), st.get("errors"))
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
