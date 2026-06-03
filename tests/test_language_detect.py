from utils.language_detect import detect_language, language_instruction


def test_detects_hinglish_by_marker_count():
    text = "aur yaar matlab kya hai toh bhi nahi samjha yeh concept"
    assert detect_language(text) == "hinglish"


def test_detects_english():
    text = "attention mechanism uses query key and value matrices to compute weighted sums"
    lang = detect_language(text)
    assert lang == "en"


def test_language_instruction_hindi_mentions_english_output():
    instr = language_instruction("hi")
    assert "English" in instr
    assert "Hindi" in instr


def test_language_instruction_hinglish_mentions_english_output():
    instr = language_instruction("hinglish")
    assert "English" in instr
    assert "Hinglish" in instr


def test_language_instruction_english_is_empty():
    assert language_instruction("en") == ""


def test_language_instruction_other_language_mentions_english():
    instr = language_instruction("fr")
    assert "English" in instr
