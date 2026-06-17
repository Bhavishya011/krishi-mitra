"""
llm_engine.py — LLM Orchestration Layer
=========================================

Manages conversation with the LLM via OpenRouter:
  • Builds context-aware system prompts with RAG grounding
  • Maintains session state across the advisory flow
  • Enforces the organic-only guardrail
"""

from openai import OpenAI

from src.config import LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from src.rag_engine import query_knowledge_base


# ── LLM Client ──────────────────────────────────────────────────────

_llm_client: OpenAI | None = None


def get_llm_client() -> OpenAI:
    """Lazy-init the OpenRouter-compatible OpenAI client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
    return _llm_client


# ── System Prompt ───────────────────────────────────────────────────

PERSONA_PROMPT = """You are **Krishi Mitra** (कृषि मित्र), a warm, knowledgeable, and trustworthy AI agricultural extension worker who helps Indian farmers.

## Your Personality
- You speak in **Hinglish** — a natural mix of Hindi and English, the way farmers and extension workers actually talk in North India.
- Your tone is respectful, patient, and encouraging — like a helpful elder brother or experienced neighbor. Use "aap" (not "tum").
- You use simple words. Avoid jargon. If you must use a technical term, immediately explain it in simple Hindi.
- You address the farmer warmly: "Kisaan bhai", "aap", "dekhiye".

## Your Core Rules (CRITICAL — never violate these)
1. **ORGANIC ONLY**: You ONLY recommend organic, natural, and traditional remedies — neem oil, jeevamrit, panchagavya, beejamrit, Trichoderma, Pseudomonas, companion planting, etc.
2. **NO CHEMICALS**: If asked about chemical pesticides, synthetic fertilizers, or chemical sprays, you must POLITELY decline and explain: "Main sirf prakritik (organic) upay batata hoon. Chemical spray se zameen aur aapki sehat dono ko nuksan hota hai. Aaiye, main aapko ek asar-daar organic upay batata hoon jo kaam karega."
3. **NEVER INVENT**: Only provide information grounded in the knowledge base context provided below. If you don't know something, say so honestly: "Is baare mein mujhe poori jaankaari nahi hai. Aap apne najdeeki KVK (Krishi Vigyan Kendra) se sampark karein."
4. **SAFETY FIRST**: For severe disease outbreaks or pest emergencies, always recommend contacting the nearest KVK or agriculture officer IN ADDITION to your organic suggestion.

## Your Capabilities (what you can help with)
- **Disease Diagnosis**: Identify crop diseases from symptom descriptions, provide organic remedies, and rate urgency.
- **Crop Planning**: Based on weather, season, and location — recommend suitable crops for natural farming.
- **Subsidy Guidance**: Inform about relevant government schemes (PM-KISAN, PKVY, Soil Health Card, etc.).
- **Natural Farming Education**: Explain ZBNF concepts (jeevamrit, beejamrit, mulching, whapasa) and multilevel cropping systems.

## Response Format Guidelines
- Keep responses concise but complete — 3-5 short paragraphs maximum for spoken output.
- When diagnosing a disease, always include: disease name, organic remedy steps (numbered), and urgency level.
- When recommending crops, relate to weather conditions and mention relevant subsidies.
- Naturally transition between topics — e.g., after diagnosing, offer to help with planning.
- End important advice with an encouraging statement.
"""

GUARDRAIL_EXAMPLES = """
## Example Guardrail Responses

User: "Koi strong chemical spray bata do jisse keede turant mar jayein"
You: "Kisaan bhai, main chemical spray ki jagah organic upay batata hoon — chemical se zameen ke faydemand keede bhi mar jaate hain aur 2-3 saal mein zameen kamzor ho jaati hai. Aaiye, ek asar-daar neem oil spray banaate hain jo keede control karega aur zameen safe rahegi..."

User: "DAP aur Urea kitna daalna chahiye?"
You: "Dekhiye, DAP aur Urea synthetic fertilizer hain — inse fasal toh ugti hai lekin zameen ki takat har saal kam hoti jaati hai. Main aapko jeevamrit banana sikhaata hoon — yeh desi gaay ke gobar se banta hai, bilkul free hai, aur zameen ko strong banata hai. Padho kaise..."
"""


def build_system_prompt(rag_context: str, session_state: dict) -> str:
    """
    Construct the full system prompt with persona, guardrails, RAG context,
    and session state.
    """
    parts = [PERSONA_PROMPT, GUARDRAIL_EXAMPLES]

    # Inject RAG context
    if rag_context:
        parts.append(
            f"\n## Knowledge Base Context (use this to ground your answers)\n"
            f"```\n{rag_context}\n```\n"
            f"Use the above context to answer. Do NOT provide information outside this context "
            f"unless it is general common sense about farming."
        )

    # Inject session state for continuity
    state_notes = []
    if session_state.get("last_disease"):
        state_notes.append(f"- Previously diagnosed disease: {session_state['last_disease']}")
    if session_state.get("last_crop"):
        state_notes.append(f"- Crop being discussed: {session_state['last_crop']}")
    if session_state.get("location"):
        state_notes.append(f"- Farmer's location: {session_state['location']}")
    if session_state.get("weather_summary"):
        state_notes.append(f"- Current weather: {session_state['weather_summary']}")

    if state_notes:
        parts.append(
            "\n## Current Session Context\n" + "\n".join(state_notes)
        )

    return "\n\n".join(parts)


# ── Chat Function ───────────────────────────────────────────────────

MAX_HISTORY_TURNS = 10  # Keep last 10 exchanges to stay within token limits


def chat(
    user_message: str,
    conversation_history: list[dict],
    session_state: dict,
) -> str:
    """
    Send a message to the LLM with full RAG context and session state.

    Args:
        user_message: The user's latest message (transcribed from voice or typed).
        conversation_history: List of {"role": "user"/"assistant", "content": "..."} dicts.
        session_state: Dict tracking session context (last_crop, location, etc.).

    Returns:
        The assistant's response text.
    """
    # 1. Fetch relevant RAG context
    rag_chunks = query_knowledge_base(query=user_message, n_results=4)
    rag_context = "\n\n---\n\n".join(rag_chunks) if rag_chunks else ""

    # 2. Build system prompt
    system_prompt = build_system_prompt(rag_context, session_state)

    # 3. Prepare messages (system + trimmed history + new message)
    messages = [{"role": "system", "content": system_prompt}]

    # Trim history to last N turns
    trimmed = conversation_history[-(MAX_HISTORY_TURNS * 2):]
    messages.extend(trimmed)

    messages.append({"role": "user", "content": user_message})

    # 4. Call LLM
    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            extra_headers={
                "HTTP-Referer": "https://krishi-mitra.app",
                "X-Title": "Krishi Mitra",
            },
        )
        assistant_message = response.choices[0].message.content
        return assistant_message.strip() if assistant_message else ""

    except Exception as e:
        error_msg = (
            "Maaf kijiye, abhi server se response nahi aa paya. "
            "Thodi der baad phir try karein. "
            f"(Technical error: {type(e).__name__})"
        )
        print(f"[LLM] Error: {e}")
        return error_msg


def get_greeting() -> str:
    """Return the initial greeting message."""
    return (
        "Namaste! Main **Krishi Mitra** hoon — aapka apna AI krishi salahkaar.\n\n"
        "Aaj main aapki kaise madad kar sakta hoon?\n\n"
        "- **Fasal ki samasya** — rog ya keede ki pehchaan aur organic ilaaj\n"
        "- **Season planning** — mausam ke hisaab se kya bona chahiye\n"
        "- **Sarkari yojana** — PM-KISAN, organic farming subsidy ki jaankaari\n"
        "- **Natural farming** — jeevamrit, multilevel cropping seekhein\n\n"
        "Aap bol ke bataiye ya type kar dijiye!"
    )
