"""
disease_detector.py — Crop Disease Detection via Image Classification
======================================================================

Uses a HuggingFace MobileNetV2 model fine-tuned on the PlantVillage dataset
to identify plant diseases from leaf images.

Returns structured detection results that the LLM can use to give
targeted organic remedies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image

from src.config import DISEASE_CONFIDENCE_THRESHOLD, DISEASE_MODEL_ID

# ── Lazy-loaded Pipeline ────────────────────────────────────────────

_classifier = None


def _get_classifier():
    """Lazy-load the image classification pipeline (downloaded once, cached)."""
    global _classifier
    if _classifier is None:
        print(f"[DISEASE] Loading model '{DISEASE_MODEL_ID}'...")
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
            pipeline,
        )

        try:
            # Try loading the pipeline directly first
            _classifier = pipeline(
                "image-classification",
                model=DISEASE_MODEL_ID,
                top_k=5,
            )
        except Exception as e:
            print(f"[DISEASE] Direct pipeline failed ({e}), trying manual load...")
            # Fallback: load model and processor separately
            # Some community models have incomplete configs
            model = AutoModelForImageClassification.from_pretrained(DISEASE_MODEL_ID)
            try:
                processor = AutoImageProcessor.from_pretrained(DISEASE_MODEL_ID)
            except Exception:
                # Use the base MobileNetV2 processor as fallback
                print("[DISEASE] Using base MobileNetV2 processor as fallback.")
                processor = AutoImageProcessor.from_pretrained(
                    "google/mobilenet_v2_1.0_224"
                )
            _classifier = pipeline(
                "image-classification",
                model=model,
                image_processor=processor,
                top_k=5,
            )

        print("[DISEASE] Model loaded successfully.")
    return _classifier


# ── Label Mapping ───────────────────────────────────────────────────

# Map model labels → structured (crop, condition, is_healthy) tuples.
# The PlantVillage model uses labels like "Tomato___Late_blight".
# We normalise these to readable names.

def _parse_label(label: str) -> tuple[str, str, bool]:
    """
    Parse a model label into (crop, condition, is_healthy).

    Handles multiple formats:
        "Tomato with Late Blight"         → ("Tomato", "Late Blight", False)
        "Healthy Corn (Maize) Plant"      → ("Corn (Maize)", "Healthy", True)
        "Corn (Maize) with Common Rust"   → ("Corn (Maize)", "Common Rust", False)
        "Tomato___Late_blight"            → ("Tomato", "Late Blight", False)
        "Potato___healthy"                → ("Potato", "Healthy", True)
    """
    # Format: "Healthy X Plant"
    if label.lower().startswith("healthy "):
        crop = label[len("Healthy "):].strip()
        if crop.lower().endswith(" plant"):
            crop = crop[:-len(" Plant")]
        return crop.strip(), "Healthy", True

    # Format: "Crop with Disease"
    if " with " in label:
        parts = label.split(" with ", 1)
        crop = parts[0].strip()
        condition = parts[1].strip()
        is_healthy = "healthy" in condition.lower()
        return crop, condition, is_healthy

    # Format: "Crop___Disease" (PlantVillage standard)
    for sep in ["___", "__", " - ", " — "]:
        if sep in label:
            parts = label.split(sep, 1)
            crop = parts[0].strip().replace("_", " ").title()
            condition = parts[1].strip().replace("_", " ").title()
            is_healthy = "healthy" in condition.lower()
            return crop, condition, is_healthy

    # Fallback: treat entire label as condition
    is_healthy = "healthy" in label.lower()
    return "Unknown", label.replace("_", " ").title(), is_healthy


# ── Hinglish Label Map ──────────────────────────────────────────────

# Common disease names in Hinglish for the LLM to use naturally.
DISEASE_HINGLISH = {
    "late blight": "Late Blight (Ageti Jhulsa)",
    "early blight": "Early Blight (Pichheti Jhulsa)",
    "bacterial spot": "Bacterial Spot (Jeevaanu Dhabbe)",
    "leaf mold": "Leaf Mold (Patti Ka곰팡이)",
    "septoria leaf spot": "Septoria Leaf Spot (Septoria Dhabbe)",
    "target spot": "Target Spot (Nishana Dhabbe)",
    "yellow leaf curl virus": "Yellow Leaf Curl Virus (Peeli Patti Virus)",
    "mosaic virus": "Mosaic Virus (Mozeik Virus)",
    "black rot": "Black Rot (Kala Sadna)",
    "cedar apple rust": "Cedar Apple Rust (Rust Rog)",
    "common rust": "Common Rust (Aam Rust)",
    "northern leaf blight": "Northern Leaf Blight (Uttar Jhulsa)",
    "cercospora leaf spot": "Cercospora Leaf Spot (Cercospora Dhabbe)",
    "powdery mildew": "Powdery Mildew (Safed Choorni Rog)",
    "downy mildew": "Downy Mildew (Mriduromill Rog)",
    "leaf scorch": "Leaf Scorch (Patti Jhulasna)",
    "black measles": "Black Measles (Kale Dhabbe)",
    "isariopsis leaf spot": "Isariopsis Leaf Spot",
    "haunglongbing": "Citrus Greening (HLB Rog)",
    "huanglongbing": "Citrus Greening (HLB Rog)",
    "healthy": "Swasth (Healthy)",
}


def _get_hinglish_name(condition: str) -> str:
    """Get Hinglish name for a disease condition, or return the original."""
    condition_lower = condition.lower()
    for eng, hinglish in DISEASE_HINGLISH.items():
        if eng in condition_lower:
            return hinglish
    return condition


# ── Main Detection Function ─────────────────────────────────────────

def detect_disease(image_path: str | Path) -> Optional[dict]:
    """
    Detect crop disease from an image file.

    Args:
        image_path: Path to the uploaded image file.

    Returns:
        dict with keys: crop, disease, disease_hinglish, confidence,
        top_3, is_healthy.
        Returns None if confidence is below threshold or on error.
    """
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"[DISEASE] Image file not found: {image_path}")
            return None

        # Open and preprocess image
        image = Image.open(image_path).convert("RGB")

        # Run classification
        classifier = _get_classifier()
        results = classifier(image)

        if not results:
            print("[DISEASE] No predictions returned.")
            return None

        # Parse top result
        top = results[0]
        top_score = top["score"]
        top_label = top["label"]

        print(f"[DISEASE] Top prediction: {top_label} ({top_score:.1%})")

        # Check confidence threshold
        if top_score < DISEASE_CONFIDENCE_THRESHOLD:
            print(
                f"[DISEASE] Confidence {top_score:.1%} below threshold "
                f"{DISEASE_CONFIDENCE_THRESHOLD:.0%}. Returning uncertain."
            )
            return {
                "crop": "Unknown",
                "disease": top_label,
                "disease_hinglish": top_label,
                "confidence": top_score,
                "top_3": [
                    (r["label"], round(r["score"], 3))
                    for r in results[:3]
                ],
                "is_healthy": False,
                "is_uncertain": True,
            }

        # Parse the label
        crop, condition, is_healthy = _parse_label(top_label)
        hinglish = _get_hinglish_name(condition)

        return {
            "crop": crop,
            "disease": condition,
            "disease_hinglish": hinglish,
            "confidence": round(top_score, 3),
            "top_3": [
                (r["label"], round(r["score"], 3))
                for r in results[:3]
            ],
            "is_healthy": is_healthy,
            "is_uncertain": False,
        }

    except Exception as e:
        print(f"[DISEASE] Error during detection: {e}")
        return None
