import os
import json
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from typing import List, Dict

# --- CONFIGURATION ---
load_dotenv()
app = Flask(__name__)

OR_API_KEY = os.getenv("OPENROUTER_API_KEY", "<YOUR_OPENROUTER_KEY>")
OR_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
# Using a powerful free model that's good at instruction following
AI_MODEL = "mistralai/mistral-7b-instruct:free"

HEADERS = {
    "Authorization": f"Bearer {OR_API_KEY}",
    "Content-Type": "application/json"
}


# --- MAPPINGS & LOGIC ---

def get_ai_expanded_keywords_for_category(base_keywords: list, category: str) -> list:
    """
    Uses OpenRouter to get semantically related keywords FOR A SPECIFIC CATEGORY.
    """
    if not base_keywords or not OR_API_KEY:
        return []

    # Craft a more specific, category-aware prompt
    prompt = (
        "You are an expert keyword generator for a news feed. Your task is to expand on a list of user-provided keywords for a specific category. "
        "For the given category and base keywords, generate a list of up to 3 additional, highly relevant keywords. \n\n"
        "RULES:\n"
        "1. Each keyword MUST be short and concise (1 to 3 words maximum).\n"
        "2. Do NOT generate long sentences or descriptive phrases.\n"
        "3. The keywords should be specific sub-topics, technologies, or named entities related to the base keywords.\n"
        "4. Respond ONLY with a valid JSON list of strings.\n\n"
        f"Category: \"{category}\"\n"
        f"Base Keywords: {', '.join(base_keywords)}\n\n"
        "Example of good output for Base Keywords ['Stock Markets', 'Venture Capital']:\n"
        "[\"IPO\", \"Angel Investors\", \"Market Analysis\"]\n\n"
        "Example of BAD output:\n"
        "[\"The impact of interest rates on stock market performance\", \"Venture capital funding rounds for tech startups\"]"
    )

    payload = {"model": AI_MODEL, "messages": [{"role": "user", "content": prompt}]}

    try:
        response = requests.post(OR_CHAT_URL, headers=HEADERS, json=payload, timeout=90)
        response.raise_for_status()
        content_str = response.json()["choices"][0]["message"]["content"]
        
        # Clean potential markdown wrapping
        cleaned_str = content_str.strip()
        if cleaned_str.startswith("```json"):
            cleaned_str = cleaned_str.replace("```json", "").replace("```", "").strip()
        
        return json.loads(cleaned_str)
    except Exception as e:
        print(f"Error calling OpenRouter for category {category}: {e}")
        return []


def process_survey_to_preferences(survey_data: dict) -> dict:
    """
    Convert survey answers into structured preferences using a new categorical approach.
    """
    keywords = {}
    keyword_sources = {}

    # --- Step 1: Group User Keywords by Category ---
    
    # Map question IDs to a category name
    category_map = {
        "7": "Technology",
        "8": "Finance",
        "9": "Local News",
        "14": "Professional", # For CEO, etc.
        "16": "Geographical"  # For Ahmedabad, etc.
    }
    
    grouped_keywords = {
        "Technology": [],
        "Finance": [],
        "Local News": [],
        "Professional": [],
        "Geographical": []
    }

    # Populate direct keywords from survey sub-topics
    for qid, category_name in category_map.items():
        answers = survey_data.get(qid, [])
        # Ensure answers are in a list format, even for single entries like Q14/Q16
        if not isinstance(answers, list):
            answers = [answers]
        
        for topic in answers:
            if topic: # Ensure topic is not empty
                keywords[topic] = 1.0  # User-selected topics get highest weight
                keyword_sources[topic] = "user"
                grouped_keywords[category_name].append(topic)

    # Add main ranked categories as lower-weight keywords
    if '5' in survey_data:
        for category, rank in survey_data['5'].items():
            weight = max(0.5, 1.1 - rank * 0.1)
            if category not in keywords:
                keywords[category] = weight
                keyword_sources[category] = "user"

    # --- Step 2: AI Expansion FOR EACH Category ---
    
    all_ai_keywords = []
    for category_name, base_keywords in grouped_keywords.items():
        if base_keywords: # Only call AI if there are keywords for the category
            print(f"Expanding keywords for category: {category_name}...")
            expanded = get_ai_expanded_keywords_for_category(base_keywords, category_name)
            all_ai_keywords.extend(expanded)
            print(f"  > AI suggested: {expanded}")

    # Add the AI-generated keywords with a consistent, moderate weight
    for kw in all_ai_keywords:
        if kw not in keywords:
            keywords[kw] = 0.65
            keyword_sources[kw] = f"ai:{AI_MODEL}"

    # --- Step 3: Assemble Final Output ---
    final_keywords_list = [
        {"keyword": k, "weight": v, "source": keyword_sources.get(k, "user")}
        for k, v in keywords.items()
    ]
    final_keywords_list.sort(key=lambda x: x["weight"], reverse=True)

    return {"keywords": final_keywords_list}

@app.route("/", methods=["POST"])
def generate_preferences_endpoint():
    try:
        survey = request.get_json(force=True)
    except:
        return jsonify({"error": "Invalid JSON input"}), 400

    prefs = process_survey_to_preferences(survey)
    return jsonify(prefs)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9090))
    app.run(host="0.0.0.0", port=port, debug=True)
