import re

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")

# Roman-script Hindi markers. IMPORTANT: none of these may be common English words
# (e.g. "the", "par") — those would misclassify plain English as Hinglish.
_HINGLISH_MARKERS = {
    "hai", "hain", "nahi", "nahin", "kya", "kyu", "kyun", "kyunki",
    "kab", "kaise", "kaisa", "kaun", "kaunsi", "kaunse", "kitna", "kitni",
    "kahan", "matlab", "aur", "mein", "hota", "hoti", "hote",
    "tha", "thi", "hoga", "hogi",
    "batao", "bolo", "dena", "karo", "karna", "karein", "kare",
    "kijiye", "dijiye", "lijiye", "samjhao",
    "chahiye", "lagti", "lagta", "milta", "milti", "ilaaj",
    "wala", "wali", "waale", "raha", "rahi", "sakta", "sakte", "sakti",
    "bahut", "thoda", "accha", "theek", "jaldi", "abhi",
    "poochho", "poochh", "poochhna", "samjho", "samajh",
    "mujhe", "aapko", "unko", "humko", "tumhe",
    "bimari", "dawai", "dard", "khoon", "jaanch", "mareez", "marij",
}


def detect_language(text: str) -> str:
    """Heuristic language detector: 'hi' (Devanagari), 'hinglish' (Roman Hindi), or 'en'.

    Used as a fallback; the primary detection is the LLM classifier in qa_handler.
    """
    if _DEVANAGARI_RE.search(text):
        return "hi"
    words = set(re.findall(r"\b\w+\b", text.lower()))
    if words & _HINGLISH_MARKERS:
        return "hinglish"
    return "en"
