import os
import re
from pathlib import Path
from collections import Counter
from flask import Flask, request, jsonify

# -----------------------------
# CONFIG
# -----------------------------
CORPUS_DIR = Path("./CORPUSES")
LOWERCASE = True
API_KEYS = {"a232jda", "b123xyz", "key3"}  # Add all valid keys

# -----------------------------
# REGEX
# -----------------------------
WORD_RE = re.compile(
    r"[^\w\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+",
    re.UNICODE
)

# -----------------------------
# UTILITIES
# -----------------------------
def clean_words(text: str) -> str:
    text = WORD_RE.sub(" ", text)
    return text.lower() if LOWERCASE else text

def get_lang_from_filename(filename: str):
    if filename.startswith("corpus-") and filename.endswith(".txt"):
        return filename.split("-", 1)[1].split(".")[0]
    return None

# -----------------------------
# BUILD LANGUAGE MODELS
# -----------------------------
def build_language_data(corpus_dir: Path):
    lang_models = {}
    for path in corpus_dir.glob("corpus-*.txt"):
        lang = get_lang_from_filename(path.name)
        if not lang:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[warn] failed to read {path}: {e}")
            continue

        words = clean_words(text).split()
        wc = Counter(words)
        lang_models[lang] = {"words": wc, "total_words": sum(wc.values())}

    return lang_models

# -----------------------------
# DETECTION
# -----------------------------
def detect_language_hits(text, lang_models):
    words = clean_words(text).split()
    if not words:
        return {}
    hits = {lang: 0 for lang in lang_models}
    for w in words:
        for lang, model in lang_models.items():
            if w in model["words"]:
                hits[lang] += 1

    hits = {lang: count for lang, count in hits.items() if count > 0}
    if not hits:
        return {}

    total_hits = sum(hits.values())
    final_probs = {lang: (count / total_hits) * 100.0 for lang, count in hits.items()}
    return dict(sorted(final_probs.items(), key=lambda x: x[1], reverse=True))

# -----------------------------
# FLASK API
# -----------------------------
app = Flask(__name__)
print("Building language models...")
lang_models = build_language_data(CORPUSES)
if not lang_models:
    print("[error] No corpora found. Place corpus-eng.txt, corpus-fra.txt, etc. in ./CORPUSES")
    exit(1)

@app.route("/detect", methods=["GET"])
def detect():
    key = request.headers.get("x-api-key") or request.args.get("api_key")
    if key not in API_KEYS:
        return jsonify({"error": "Unauthorized"}), 401

    text = request.args.get("text", "")
    probs = detect_language_hits(text, lang_models)
    return jsonify(probs)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"data": "running!"})

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
