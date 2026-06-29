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
