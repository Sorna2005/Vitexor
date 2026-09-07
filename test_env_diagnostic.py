import sys
from pathlib import Path

env_path = Path(".env")

print("=" * 70)
print(".env DIAGNOSTIC (no secrets printed)")
print("=" * 70)

print("exists:", env_path.exists())
print("cwd:", Path.cwd())

if env_path.exists():
    raw = env_path.read_bytes()
    print("byte length:", len(raw))
    print("starts with UTF-16 LE BOM (FF FE):", raw[:2] == b"\xff\xfe")
    print("starts with UTF-8 BOM (EF BB BF):", raw[:3] == b"\xef\xbb\xbf")
    print("contains null bytes (sign of UTF-16):", b"\x00" in raw[:200])

    try:
        text = raw.decode("utf-8")
        print("decodes as utf-8: True")
    except UnicodeDecodeError as e:
        print("decodes as utf-8: False ->", e)
        text = None

    if text is not None:
        found = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("GEMINI_API_KEY"):
                found = True
                key_part, _, value_part = stripped.partition("=")
                print("found line starting with GEMINI_API_KEY:", repr(key_part))
                print("value length (chars):", len(value_part.strip().strip(chr(34)).strip(chr(39))))
        print("GEMINI_API_KEY line found:", found)

print()
print("--- now checking what pydantic-settings actually loads ---")
sys.path.insert(0, ".")
from src.review_engine.config import settings
print("settings.gemini_api_key length:", len(settings.gemini_api_key))
print("settings.gemini_api_key is empty:", settings.gemini_api_key.strip() == "")
