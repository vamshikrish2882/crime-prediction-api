import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

def extract_location_time(user_question):
    try:
        payload = {
            "model": "mistralai/mixtral-8x7b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You MUST respond in exactly two lines:\n"
                        "Line 1: 'Location: <location name>'\n"
                        "Line 2: 'Time: <time like 2AM, 10PM>'\n"
                        "If unknown, respond with 'unknown'. No extra text, no quotes, no code."
                    )
                },
                {
                    "role": "user",
                    "content": user_question
                }
            ],
            "temperature": 0.0,
            "max_tokens": 100
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://example.com",  # <- REQUIRED BY OPENROUTER
            "X-Title": "crime-safety-chatbot"       # <- ANY SHORT APP NAME
        }

        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
        response.raise_for_status()

        completion = response.json()
        extracted_text = completion['choices'][0]['message']['content']

        # Extract location and time safely
        location = "unknown"
        time = "unknown"
        for line in extracted_text.splitlines():
            if "location:" in line.lower():
                location = line.split(":", 1)[-1].strip()
            if "time:" in line.lower():
                time = line.split(":", 1)[-1].strip()

        # Extra cleanup: remove unwanted words
        location = location.replace("neighborhood", "").strip()

        return {"location": location, "time": time}

    except Exception as e:
        print(f"❌ Error extracting location and time: {e}")
        return {"location": "unknown", "time": "unknown"}
