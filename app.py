from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
import requests
import base64
import json
import os
import re

load_dotenv()

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def image_url_to_base64(image_url):
    response = requests.get(image_url)
    response.raise_for_status()
    return base64.b64encode(response.content).decode("utf-8")

def extract_json(text):
    """Extract the first valid JSON object found in the model's response."""
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not extract JSON from model response: {text[:300]}")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"success": True, "message": "FasalAI AI Service running with Groq!"})

@app.route("/detect", methods=["POST"])
def detect_disease():
    try:
        data = request.get_json()
        image_url = data.get("image_url")
        crop_type = data.get("crop_type", "Unknown")

        if not image_url:
            return jsonify({"success": False, "error": "image_url is required"}), 400

        print(f"Analyzing {crop_type} crop image...")

        image_base64 = image_url_to_base64(image_url)

        prompt = f"""You are an expert agricultural scientist analyzing a photo of a {crop_type} crop leaf/plant.

Carefully examine the actual visual symptoms in the image (leaf spots, discoloration, wilting, powdery patches, holes, etc.) before deciding.

Respond ONLY with a single JSON object in exactly this format, no extra text, no markdown:
{{
  "disease": "disease name or No Disease Found",
  "confidence": 85,
  "severity": "High or Moderate or Low or None",
  "treatment": ["step 1", "step 2", "step 3", "step 4"],
  "prevention": ["tip 1", "tip 2", "tip 3", "tip 4"]
}}"""

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.2,
        )

        text = response.choices[0].message.content.strip()
        print(f"Groq raw response: {text}")

        parsed = extract_json(text)

        required_keys = {"disease", "confidence", "severity", "treatment", "prevention"}
        if not required_keys.issubset(parsed.keys()):
            raise ValueError(f"Model response missing required fields: {parsed}")

        return jsonify({"success": True, **parsed})

    except ValueError as e:
        print(f"Parsing error: {e}")
        return jsonify({
            "success": False,
            "error": "Could not get a valid analysis from the AI model. Please try again with a clearer image.",
        }), 502

    except requests.RequestException as e:
        print(f"Image download error: {e}")
        return jsonify({
            "success": False,
            "error": "Could not download the provided image.",
        }), 400

    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred during analysis.",
        }), 500

if __name__ == "__main__":
    app.run(port=8000, debug=True)