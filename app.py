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
API_KEYS = {"a232jda", "b123xyz", "key3"}  # Your valid API keys

# -----------------------------
# REGEX
# -----------------------------
WORD_RE = re.compile(r"[^\w\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+", re.UNICODE)

# -----------------------------
# UTILITIES
# -----------------------------
def get_lang_from_filename(filename):
    if filename.startswith("corpus-") and filename.endswith(".txt"):
        return filename.split("-", 1)[1].split(".")[0]
    return None

def clean_words(text):
    text = WORD_RE.sub(" ", text)
    return text.lower() if LOWERCASE else text

# -----------------------------
# BUILD LANGUAGE MODELS
# -----------------------------
def build_language_data(corpus_dir):
    lang_models = {}
    if not corpus_dir.exists():
        print(f"[warn] Corpus folder {corpus_dir} does not exist.")
        return lang_models

    for path in corpus_dir.glob("corpus-*.txt"):
        lang = get_lang_from_filename(path.name)
        if not lang:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[warn] Failed to read {path}: {e}")
            continue

        words = clean_words(text).split()
        wc = Counter(words)
        lang_models[lang] = {"words": wc, "total_words": sum(wc.values())}

    return lang_models

# -----------------------------
# HIT-BASED WORD-LEVEL DETECTION
# -----------------------------
def detect_language_hits(text, lang_models):
    words = clean_words(text).split()
    if not words or not lang_models:
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
lang_models = build_language_data(CORPUS_DIR)
if not lang_models:
    print("[warn] No corpora found. Place files like corpus-english.txt, corpus-french.txt, corpus-arabic.txt in ./CORPUSES")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running!"})

@app.route("/detect", methods=["GET"])
def detect():
    try:
        key = request.headers.get("x-api-key") or request.args.get("api_key")
        if key not in API_KEYS:
            return jsonify({"error": "Unauthorized"}), 401

        text = request.args.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Detect language
        probs = detect_language_hits(text, lang_models)
        return jsonify(probs, "prompt": text)

    except Exception as e:
        # Return JSON instead of crashing Gunicorn
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
