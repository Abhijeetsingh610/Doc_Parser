from flask import Flask, request, jsonify
import easyocr
import requests
import os
import json
import re
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === CONFIG ===
GEMINI_API_KEY = "AIzaSyDc78hwDAKisI-jHEB11TIMgRSwrHYYNZc"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
HEADERS = {"Content-Type": "application/json"}

# === OCR Setup ===
reader = easyocr.Reader(['en'], gpu=False)

def extract_text_from_image(image_path):
    result = reader.readtext(image_path, detail=0, paragraph=True)
    return "\n".join(result)

def build_prompt(doc_type, text):
    if doc_type == "Driving License":
        return f'''
You are an AI trained to extract structured data from OCR text.

Extract only this JSON format:
{{
  "document_type": "Driving License",
  "name": ...,
  "date_of_birth": ...,
  "license_number": ...,
  "issuing_state": ...,
  "expiry_date": ...
}}

Text:
"""
{text}
"""
Only return JSON. Do not guess. Return null for missing fields.
'''

    elif doc_type == "Shop Receipt":
        return f'''
You are an AI trained to extract structured data from OCR text.

Extract only this JSON format:
{{
  "document_type": "Shop Receipt",
  "merchant_name": ...,
  "total_amount": ...,
  "date_of_purchase": ...,
  "payment_method": ...,
  "items": [
    {{"name": ..., "quantity": ..., "price": ...}},
    ...
  ]
}}

Text:
"""
{text}
"""
Only return JSON. Return null for missing values.
'''

    elif doc_type == "Resume":
        return f'''
You are an AI trained to extract structured resume data from noisy OCR.

Extract only this JSON format:
{{
  "document_type": "Resume",
  "full_name": ..., 
  "email": ..., 
  "phone_number": ..., 
  "skills": [...],
  "work_experience": [{{"company": ..., "role": ..., "dates": ...}}],
  "education": [{{"institution": ..., "degree": ..., "graduation_year": ...}}]
}}

Text:
"""
{text}
"""
Only return JSON. Do not hallucinate. Use null or [] for missing.
'''

    else:
        return ""  # fallback

def call_gemini(prompt):
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    response = requests.post(GEMINI_ENDPOINT, headers=HEADERS, json=data)
    try:
        text = response.json()['candidates'][0]['content']['parts'][0]['text']
        if text.strip().startswith("```"):
            text = re.sub(r"^```(json)?\\s*", "", text.strip())
            text = re.sub(r"\\s*```$", "", text.strip())
        return json.loads(text)
    except Exception as e:
        return {"error": str(e), "raw_output": response.text}

@app.route('/api/parse', methods=['POST'])
def parse_document():
    if 'file' not in request.files or 'doc_type' not in request.form:
        return jsonify({"error": "Missing file or doc_type"}), 400

    file = request.files['file']
    doc_type = request.form['doc_type']

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    text = extract_text_from_image(file_path)
    prompt = build_prompt(doc_type, text)
    result = call_gemini(prompt)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
