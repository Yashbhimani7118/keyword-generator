from flask import Flask, request, jsonify
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model)

@app.route("/generate-keywords", methods=["POST"])
def generate_keywords():
    data = request.json
    text_parts = []

    # Combine all relevant user answers into a blob
    categories = data.get("main_categories", [])
    subcategories = data.get("subcategories", [])
    location = data.get("location", "")
    tone = data.get("tone", "")
    sources = data.get("sources", [])
    language = data.get("language", [])

    # Build input text
    text_parts += categories + subcategories + sources + language + [location, tone]
    combined_text = " ".join(text_parts)

    keywords = kw_model.extract_keywords(
        combined_text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=20,
        use_maxsum=True,
        diversity=0.7
    )

    result = [{"keyword": kw, "weight": round(score, 2)} for kw, score in keywords]
    return jsonify({"keywords": result})
