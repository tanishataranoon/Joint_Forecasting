"""
Grounded LLM advisory layer (Gemini 2.5 Flash), two-step pipeline:
English generation -> separate Bangla translation.

STATUS: this uploaded notebook doesn't yet contain your advisory cells --
per your notes, Sections 15-16 of your v6 notebook already implement this
with GUIDELINE_KB as a placeholder. Port that code in here. This file
just replaces Colab's `userdata.get('GEMINI_API_KEY')` with a .env-based
key, since that Colab-secrets pattern won't exist on either the
university PC or your personal PC.

Remember: GUIDELINE_KB still needs real BRRI/BARC/BARI/DAE excerpts
before final submission -- that's tracked as a TODO, not done here.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env in project root -- see .env.example

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def load_guideline_kb(path: str = "src/advisory_guideline_kb.json") -> dict:
    """
    Placeholder knowledge base -- replace with real BRRI/BARC/BARI/DAE
    guideline excerpts before final submission.
    """
    p = Path(path)
    if not p.exists():
        print(f"WARNING: {path} not found -- using empty placeholder KB.")
        return {}
    with open(p, "r") as f:
        return json.load(f)


def generate_advisory_english(agronomic_summary: str, guideline_kb: dict) -> str:
    """
    TODO: port your existing Gemini prompt + call logic here
    (grounded generation against GUIDELINE_KB, English output).
    """
    raise NotImplementedError("Port your Section 15 Gemini call here.")


def translate_to_bangla(english_text: str) -> str:
    """
    TODO: port your existing second-step Bangla translation call here.
    Kept as a separate call (not one combined prompt) per your existing
    two-step design.
    """
    raise NotImplementedError("Port your Section 16 translation call here.")


def get_advisory(agronomic_summary: str) -> dict:
    kb = load_guideline_kb()
    english = generate_advisory_english(agronomic_summary, kb)
    bangla = translate_to_bangla(english)
    return {"english": english, "bangla": bangla}
