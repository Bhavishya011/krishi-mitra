"""
config.py — Central configuration loader
=========================================

Reads .env and exposes typed constants used across all modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Load environment variables from .env
load_dotenv(dotenv_path=ENV_PATH)

# ── LLM Configuration ───────────────────────────────────────────────
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b:free")

# ── STT Configuration ───────────────────────────────────────────────
WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_LANGUAGE: str = "hi"  # Hindi — Whisper handles Hinglish well

# ── TTS Configuration ───────────────────────────────────────────────
TTS_VOICE: str = os.getenv("TTS_VOICE", "hi-IN-SwaraNeural")

# ── RAG / Knowledge Base ────────────────────────────────────────────
KNOWLEDGE_BASE_PATH: Path = PROJECT_ROOT / "data" / "knowledge_base"
CHROMA_DB_PATH: Path = PROJECT_ROOT / "data" / "chroma_db"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME: str = "krishi_mitra_kb"

# ── Audio Output ─────────────────────────────────────────────────────
AUDIO_OUTPUT_DIR: Path = PROJECT_ROOT / "assets" / "audio_output"

# ── Disease Detection ────────────────────────────────────────
DISEASE_MODEL_ID: str = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
DISEASE_CONFIDENCE_THRESHOLD: float = 0.50

# ── Weather API ──────────────────────────────────────────────────────
OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODING_URL: str = "https://geocoding-api.open-meteo.com/v1/search"

# ── Ensure directories exist ─────────────────────────────────────────
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_BASE_PATH.mkdir(parents=True, exist_ok=True)
