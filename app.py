import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

OR_KEY = os.environ.get("OPENROUTER_API_KEY")
OR_URL = "https://openrouter.ai/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OR_KEY}",
    "Content-Type": "application/json"
}

# Adjust this model ID to a free-tier OpenRouter model you prefer:
MODEL_ID = "gpt-3.5-turbo"  

@app.route("/generate-keywords", methods=["POST"])
def generate_keywords():
    data = request.json or {}
    # Build the text blob from survey answers:
    text = " ".join(
        data.get("main_categories", []) +
        data.get("subcategories", []) +
        data.get("sources", []) +
        data.get("language", []) +
        [data.get("location", ""), data.get("tone", "")]
    )

    # Craft a system+user prompt to extract keywords
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system",
             "content": "You are an assistant that extracts the top 15 most relevant keywords (with weights 0.0–1.0) from input text. Output valid JSON."},
            {"role": "user",
             "content": f"Extract keywords from the following user preferences:\n\n\"\"\"\n{text}\n\"\"\"\n\nRespond with:\n```\n[{{\"keyword\": \"...\", \"weight\": 0.85}}, ...]\n```"}
        ],
        "temperature": 0.0,
        "max_tokens": 300
    }

    resp = requests.post(OR_URL, headers=HEADERS, json=payload, timeout=90)
    resp.raise_for_status()
    # Parse the assistant’s reply (assumed to be raw JSON list)
    keywords = resp.json()["choices"][0]["message"]["content"]
    # Convert the JSON string into Python list
    import json as _json
    kw_list = _json.loads(keywords)
    return jsonify({"keywords": kw_list})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
