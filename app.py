from flask import Flask, request, jsonify
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model)

@app.route('/generate-keywords', methods=['POST'])
def generate_keywords():
    data = request.json
    combined_text = ' '.join(
        data.get("main_categories", []) +
        data.get("subcategories", []) +
        data.get("sources", []) +
        data.get("language", []) +
        [data.get("location", ""), data.get("tone", "")]
    )
    keywords = kw_model.extract_keywords(
        combined_text, keyphrase_ngram_range=(1, 2),
        stop_words='english', top_n=15,
        use_maxsum=True, diversity=0.7
    )
    return jsonify({"keywords": [{"keyword": kw, "weight": round(score, 2)} for kw, score in keywords]})
