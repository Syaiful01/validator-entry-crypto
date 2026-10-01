from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("=== MODEL YANG TERSEDIA ===")
for m in client.models.list():
    name = m.name
    if "flash" in name.lower() or "pro" in name.lower():
        print(name)