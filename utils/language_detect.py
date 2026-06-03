from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42

_HINGLISH_MARKERS = ["kya", "hai", "aur", "nahi", "bhi", "toh", "matlab", "yaar", "woh", "iska"]
_HINGLISH_THRESHOLD = 3


def detect_language(text: str) -> str:
    sample = text[:3000].lower()
    hits = sum(1 for marker in _HINGLISH_MARKERS if f" {marker} " in sample)
    if hits >= _HINGLISH_THRESHOLD:
        return "hinglish"
    try:
        return detect(sample)
    except Exception:
        return "en"


def language_instruction(lang_code: str) -> str:
    if lang_code == "hinglish":
        return (
            "The source material is in Hinglish (Hindi-English code-switching). "
            "Understand the content fully and write all output in fluent English."
        )
    if lang_code == "hi":
        return (
            "The source material is in Hindi. "
            "Understand the content fully and write all output in fluent English."
        )
    if lang_code != "en":
        return (
            f"The source material may be in language code '{lang_code}'. "
            "Write all output in fluent English."
        )
    return ""
