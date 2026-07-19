"""
Central configuration for the Sports Quiz Agent.
Loads settings from environment variables / .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
CHROMA_COLLECTION_NAME = "sports_knowledge"

SUPPORTED_SPORTS = ["Cricket", "Football", "Tennis", "Badminton", "Basketball"]
DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]
