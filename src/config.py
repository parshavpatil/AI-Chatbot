"""
config.py — Central configuration loaded from environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Groq
    GROQ_API_KEY: str   = os.getenv("GROQ_API_KEY", "")
    MODEL: str          = os.getenv("MODEL", "llama-3.3-70b-versatile")
    MAX_TOKENS: int     = int(os.getenv("MAX_TOKENS", "1024"))
    TEMPERATURE: float  = float(os.getenv("TEMPERATURE", "0.7"))

    # App
    SECRET_KEY: str     = os.getenv("SECRET_KEY", "chatbot-secret-key-change-in-prod")
    DEBUG: bool         = os.getenv("DEBUG", "false").lower() == "true"

    @classmethod
    def validate(cls):
        """Raise early if required config is missing."""
        if not cls.GROQ_API_KEY:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Add your Groq API key to the .env file."
            )
