"""
voice_engine.py — Speech-to-Text & Text-to-Speech Pipeline
============================================================

Handles all voice I/O:
  • STT: faster-whisper for Hindi/Hinglish transcription
  • TTS: edge-tts with hi-IN-SwaraNeural voice
"""

import asyncio
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import AUDIO_OUTPUT_DIR, TTS_VOICE, WHISPER_LANGUAGE, WHISPER_MODEL_SIZE


# ── STT — Speech to Text ───────────────────────────────────────────

_whisper_model = None


def _get_whisper_model():
    """Lazy-load the faster-whisper model (loaded once, reused)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        print(f"[STT] Loading Whisper model '{WHISPER_MODEL_SIZE}'...")
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )
        print("[STT] Whisper model loaded.")
    return _whisper_model


def transcribe_audio(audio_input) -> str:
    """
    Transcribe audio to text using faster-whisper.

    Args:
        audio_input: Either a file path (str) or a tuple of (sample_rate, numpy_array)
                     as provided by Gradio's gr.Audio component.

    Returns:
        Transcribed text string. Empty string if transcription fails or audio is silent.
    """
    model = _get_whisper_model()

    try:
        # Handle Gradio's (sample_rate, numpy_array) tuple format
        if isinstance(audio_input, tuple):
            sample_rate, audio_data = audio_input

            # Validate audio data
            if audio_data is None or len(audio_data) == 0:
                return ""

            # Convert to float32 and normalize
            audio_data = audio_data.astype(np.float32)
            if audio_data.max() > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            # If stereo, convert to mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Save to temp file for faster-whisper (it works with file paths)
            temp_path = AUDIO_OUTPUT_DIR / f"temp_input_{int(time.time())}.wav"
            _save_wav(audio_data, sample_rate, temp_path)
            audio_path = str(temp_path)
        else:
            audio_path = str(audio_input)

        # Transcribe with VAD filter to skip silence
        segments, info = model.transcribe(
            audio_path,
            language=WHISPER_LANGUAGE,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )

        # Concatenate all segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        transcribed = " ".join(text_parts).strip()
        print(f"[STT] Transcribed ({info.language}, "
              f"prob={info.language_probability:.2f}): {transcribed[:100]}...")
        return transcribed

    except Exception as e:
        print(f"[STT] Transcription error: {e}")
        return ""

    finally:
        # Clean up temp file if it was created
        if isinstance(audio_input, tuple):
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


def _save_wav(audio_data: np.ndarray, sample_rate: int, filepath: Path) -> None:
    """Save a numpy array as a WAV file."""
    import wave
    import struct

    # Convert float32 [-1, 1] to int16
    int_data = (audio_data * 32767).astype(np.int16)

    with wave.open(str(filepath), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(int_data)}h", *int_data))


# ── TTS — Text to Speech ───────────────────────────────────────────

def text_to_speech(text: str) -> Optional[str]:
    """
    Convert text to speech using edge-tts.

    Args:
        text: The text to synthesize (supports Hindi, English, Hinglish).

    Returns:
        Path to the generated audio file (MP3), or None on failure.
    """
    if not text or not text.strip():
        return None

    try:
        import edge_tts
        import threading

        # Clean text for TTS — remove markdown formatting
        clean_text = _clean_text_for_tts(text)

        if not clean_text.strip():
            return None

        # Generate unique filename
        timestamp = int(time.time() * 1000)
        output_path = AUDIO_OUTPUT_DIR / f"response_{timestamp}.mp3"

        # Run async TTS in a dedicated thread with its own event loop
        # This avoids conflicts with Gradio's running event loop
        error_holder = [None]

        def _run_tts():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    communicate = edge_tts.Communicate(
                        clean_text,
                        TTS_VOICE,
                        rate="-5%",  # Slightly slower for clarity
                    )
                    loop.run_until_complete(communicate.save(str(output_path)))
                finally:
                    loop.close()
            except Exception as e:
                error_holder[0] = e

        tts_thread = threading.Thread(target=_run_tts)
        tts_thread.start()
        tts_thread.join(timeout=30)  # 30 second timeout

        if error_holder[0]:
            raise error_holder[0]

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"[TTS] Generated: {output_path.name} ({output_path.stat().st_size} bytes)")
            return str(output_path)
        else:
            print("[TTS] Output file is empty or missing.")
            return None

    except Exception as e:
        print(f"[TTS] Error: {e}")
        return None


def _clean_text_for_tts(text: str) -> str:
    """
    Remove markdown formatting and special characters that TTS shouldn't speak.
    """
    import re

    # Remove markdown bold/italic
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)

    # Remove markdown headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

    # Remove markdown links [text](url) → text
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)

    # Remove bullet points
    text = re.sub(r'^[\s]*[-•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Remove emojis (keep Hindi/English text)
    text = re.sub(
        r'[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff'
        r'\U0001f680-\U0001f6ff\U0001f900-\U0001f9ff'
        r'\U00002702-\U000027b0\U0001fa00-\U0001faff]+',
        '', text
    )

    # Collapse multiple spaces/newlines
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def cleanup_old_audio(max_age_seconds: int = 3600) -> int:
    """
    Remove audio files older than max_age_seconds from the output directory.
    Returns the number of files deleted.
    """
    deleted = 0
    now = time.time()
    for f in AUDIO_OUTPUT_DIR.glob("response_*.mp3"):
        if now - f.stat().st_mtime > max_age_seconds:
            f.unlink(missing_ok=True)
            deleted += 1
    return deleted
