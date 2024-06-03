import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model configurations
MODEL_CONFIGS = {
    "gpt-3.5-turbo": {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 1000
    },
    "gpt-4-turbo": {
        "model": "gpt-4-turbo",
        "temperature": 0,
        "max_tokens": 1000
    }
}
