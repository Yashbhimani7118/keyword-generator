import os
import json
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
app = Flask(__name__)

# Your OpenRouter API Key, set as an environment variable
OR_API_KEY = os.getenv("OPENROUTER_API_KEY")
OR_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# You can choose a fast and free/low-cost model from OpenRouter
# e.g., "google/gemma-7b-it", "mistralai/mistral-7b-instruct"
AI_MODEL = "google/gemma-7b-it"

HEADERS = {
    "Authorization": f"Bearer {OR_API_KEY}",
    "Content-Type": "application/json"
}


# --- MAPPINGS & LOGIC ---
# These mappings help convert survey answers to concrete data

SOURCE_MAP = {
    "Leading global outlets (e.g., BBC, Reuters)": ["bbc.com", "reuters.com", "nytimes.com", "aljazeera.com"],
    "Industry-specific journals": ["hbr.org", "forbes.com"],
    "Local newspapers/websites": ["divyabhaskar.co.in", "sandesh.com", "gujaratsamachar.com"],
    "Independent blogs/podcasts": [] # Can be expanded later
}

GEO_MAP = {
    "Ahmedabad": ["Ahmedabad", "Gujarat", "Gandhinagar"],
    "Surat": ["Surat", "Gujarat"],
    "Rajkot": ["Rajkot", "Gujarat"]
    # Add other cities as needed
}

def get_ai_expanded_keywords(base_keywords: list) -> list:
    """
    Uses OpenRouter to get semantically related keywords.
    """
    if not OR_API_KEY:
        print("Warning: OPENROUTER_API_KEY is not set. Skipping AI expansion.")
        return []

    # Create a focused prompt for the AI
    prompt = (
        "You are a keyword expansion expert for a news feed. Based on the following user interest keywords, "
        "generate a list of up to 5 additional, highly relevant and specific sub-topics, technologies, or related concepts. "
        "For example, if you see 'Venture Capital', you might add 'Series A Funding'. If you see 'Python', add 'Data Science' or 'FastAPI'.\n\n"
        f"Base keywords: {', '.join(base_keywords)}\n\n"
        "Respond ONLY with a JSON list of strings. For example: [\"keyword1\", \"keyword2\"]"
    )

    payload = {
        "model": AI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"} # Ask for a JSON response
    }

    try:
        response = requests.post(OR_API_URL, headers=HEADERS, json=payload, timeout=90)
        response.raise_for_status()
        
        # The AI's response is a JSON string inside the content
        content_str = response.json()["choices"][0]["message"]["content"]
        # It might be wrapped in ```json ... ```, so we clean it
        cleaned_str = content_str.strip().replace("```json\n", "").replace("\n```", "")
        
        # Safely parse the inner JSON
        # The AI might return a structure like {"keywords": [...]}, so we check for that.
        json_response = json.loads(cleaned_str)
        if isinstance(json_response, dict) and "keywords" in json_response:
             return json_response["keywords"]
        elif isinstance(json_response, list):
             return json_response
        else:
            return []

    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenRouter: {e}")
        return []
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing AI response: {e}")
        return []


def process_survey_to_preferences(survey_data: dict) -> dict:
    """
    Processes the raw survey JSON and returns the full, structured preferences object.
    """
    keywords = {}
    
    # --- 1. Rule-Based Extraction ---

    # Language Filter
    language_filter = [lang.strip() for lang in survey_data.get("15", "English").split(',')]

    # Geo Filter
    primary_location = survey_data.get("16", "Ahmedabad")
    geo_filter = GEO_MAP.get(primary_location, [primary_location])

    # Source Filter
    source_preferences = survey_data.get("11", [])
    source_filter = []
    for pref in source_preferences:
        source_filter.extend(SOURCE_MAP.get(pref, []))

    # --- 2. Keyword & Weight Generation ---

    # From Sub-topics (Q7, Q8, Q9) - Highest weight
    for q_id in ["7", "8", "9"]:
        for topic in survey_data.get(q_id, []):
            keywords[topic] = 1.0

    # From Main Categories (Q5) - Weighted by rank
    if '5' in survey_data:
        for category, rank in survey_data['5'].items():
            weight = max(0.5, 1.1 - (rank * 0.1)) # Rank 1=1.0, 2=0.9, ... 5=0.6
            keywords[category] = max(keywords.get(category, 0), weight)
    
    # From Profession (Q14)
    if survey_data.get("14"):
        keywords[survey_data["14"]] = 0.8

    # --- 3. AI Keyword Expansion ---
    
    # Get the top 5 highest weighted keywords to send to the AI for expansion
    base_keywords_for_ai = sorted(keywords, key=keywords.get, reverse=True)[:5]
    ai_expanded_keywords = get_ai_expanded_keywords(base_keywords_for_ai)

    # Add the AI-generated keywords with a moderate weight
    for kw in ai_expanded_keywords:
        if kw not in keywords: # Only add if it's a new keyword
            keywords[kw] = 0.65 

    # --- 4. Assemble Final Output ---
    
    final_keywords_list = [{"keyword": k, "weight": v} for k, v in keywords.items()]
    # Sort by weight descending
    final_keywords_list.sort(key=lambda x: x['weight'], reverse=True)

    return {
        "keywords": final_keywords_list,
        "language_filter": language_filter,
        "source_filter": list(set(source_filter)), # Remove duplicates
        "geo_filter": list(set(geo_filter))
    }

# --- Main API Endpoint ---

@app.route("/generate-preferences", methods=["POST"])
def generate_preferences_endpoint():
    """
    Main endpoint to receive survey data and return structured preferences.
    """
    survey_data = request.json
    if not survey_data:
        return jsonify({"error": "Invalid JSON input"}), 400

    try:
        preferences = process_survey_to_preferences(survey_data)
        return jsonify(preferences)
    except Exception as e:
        # Log the full error for debugging
        print(f"An error occurred: {e}")
        return jsonify({"error": "Failed to process survey data."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
