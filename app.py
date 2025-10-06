from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running!"})

if __name__ == "__main__":
    # Render sets PORT in environment, default to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
