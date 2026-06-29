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

# OCR (ingestion only) — recovers text baked into images/diagrams/flowcharts
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() in ("1", "true", "yes")
OCR_LANG = os.getenv("OCR_LANG", "eng")
OCR_DPI = int(os.getenv("OCR_DPI", "300"))
# Path to the Tesseract binary. Leave blank to rely on PATH.
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
