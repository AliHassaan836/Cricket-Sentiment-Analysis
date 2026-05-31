from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
SPACY_MODEL: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
