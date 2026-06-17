"""
app.py — Krishi Mitra Gradio Application
==========================================

Entry point for the voice-first AI agricultural consultant.
Two-view layout: Hero landing page → Gemini-style consultation interface.
"""

import time
from pathlib import Path

import gradio as gr

from src.config import PROJECT_ROOT
from src.disease_detector import detect_disease
from src.llm_engine import chat, get_greeting
from src.voice_engine import text_to_speech, transcribe_audio
from src.weather_service import (
    format_weather_card,
    get_coordinates,
    get_weather_forecast,
    get_weather_summary_for_llm,
)

# ── Constants ────────────────────────────────────────────────────────
CSS_PATH = PROJECT_ROOT / "assets" / "style.css"
VIDEO_URL = (
    "https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/"
    "hf_20260602_150901_c45b90ec-18d7-42ff-90e2-b95d7109e330.mp4"
)
AVATAR_URL = "https://ui-avatars.com/api/?name=KM&background=16a34a&color=fff&size=128&bold=true&font-size=0.45"


# ── Hero HTML ────────────────────────────────────────────────────────
def _build_hero_html() -> str:
    return f"""
    <div id="hero-section">
        <video id="hero-video-bg" autoplay muted loop playsinline>
            <source src="{VIDEO_URL}" type="video/mp4">
        </video>
        <div id="hero-overlay">
            <div id="hero-navbar" class="liquid-glass">
                <div class="hero-logo">
                    <span class="hero-logo-icon">KM</span>Krishi Mitra
                </div>
                <div class="hero-tagline-nav">AI Agricultural Consultant</div>
            </div>

            <div id="hero-content">
                <div>
                    <h1 class="hero-heading">
                        Aapki Fasal,<br>
                        <em>Hamari Zimmedari.</em>
                    </h1>
                    <p class="hero-subheading">
                        Voice-first AI krishi salahkaar jo aapki fasal ki bimari pehchane,
                        mausam ke hisaab se planning kare, aur organic kheti sikhaye —
                        sab ek baat-cheet mein.
                    </p>
                </div>
                <div style="display:flex; align-items:end; justify-content:end;">
                    <div class="hero-tag-card liquid-glass">
                        Diagnose &middot; Plan &middot; Grow
                    </div>
                </div>
            </div>
        </div>
    </div>
    """


# ── Session State Helpers ────────────────────────────────────────────

def _new_session_state():
    """Create a fresh session state."""
    return {
        "conversation_history": [],
        "last_crop": None,
        "last_disease": None,
        "location": None,
        "latitude": None,
        "longitude": None,
        "weather_summary": None,
        "phase": "greeting",
    }


# ── Core Interaction Logic ───────────────────────────────────────────

def _process_message(
    user_text: str,
    chat_history,
    session_state,
    weather_card_md: str,
) :
    """
    Process a user message through the full pipeline:
    text → LLM → response → TTS → updated state.

    Returns:
        (chat_history, session_state, audio_path, weather_card_md)
    """
    if not user_text or not user_text.strip():
        return chat_history, session_state, None, weather_card_md

    # Add user message to conversation history
    session_state["conversation_history"].append({
        "role": "user",
        "content": user_text,
    })

    # Detect location in user message (simple heuristic)
    _try_detect_location(user_text, session_state)

    # If location detected and no weather yet, fetch it
    if (session_state.get("latitude") and not session_state.get("weather_summary")):
        forecast = get_weather_forecast(
            session_state["latitude"],
            session_state["longitude"],
        )
        if forecast:
            session_state["weather_summary"] = get_weather_summary_for_llm(
                forecast, session_state.get("location", "")
            )
            weather_card_md = format_weather_card(
                forecast, session_state.get("location", "")
            )

    # Get LLM response
    assistant_response = chat(
        user_message=user_text,
        conversation_history=session_state["conversation_history"][:-1],
        session_state=session_state,
    )

    # Add assistant response to history
    session_state["conversation_history"].append({
        "role": "assistant",
        "content": assistant_response,
    })

    # Update chat display
    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": assistant_response})

    # Generate TTS
    audio_path = text_to_speech(assistant_response)

    return chat_history, session_state, audio_path, weather_card_md


def _try_detect_location(text: str, session_state: dict) -> None:
    """
    Attempt to detect a location from user text and resolve coordinates.
    Simple keyword-based heuristic — works for common patterns.
    """
    if session_state.get("latitude"):
        return  # Already have location

    text_lower = text.lower()
    location_triggers = [
        "se hoon", "se hun", "se hu", "mein rehta", "mein rahta",
        "village", "gaon", "district", "jila", "state",
        "haryana", "punjab", "rajasthan", "uttar pradesh",
        "pincode", "pin code",
    ]

    # Check if message likely contains location info
    has_trigger = any(trigger in text_lower for trigger in location_triggers)
    if not has_trigger:
        return

    import re
    words = text.split()
    candidates = []
    for word in words:
        clean = re.sub(r'[,।.!?]', '', word)
        if clean and len(clean) > 2 and clean[0].isupper():
            candidates.append(clean)

    # Also try common city names mentioned in text
    known_cities = [
        "Karnal", "Ludhiana", "Jaipur", "Lucknow", "Delhi", "Chandigarh",
        "Hisar", "Ambala", "Panipat", "Rohtak", "Sonipat", "Kurukshetra",
        "Patiala", "Amritsar", "Jalandhar", "Meerut", "Agra", "Varanasi",
        "Allahabad", "Bareilly", "Moradabad", "Jodhpur", "Udaipur", "Kota",
        "Bhopal", "Indore", "Nagpur", "Pune", "Nashik", "Dehradun",
    ]
    for city in known_cities:
        if city.lower() in text_lower:
            candidates.insert(0, city)

    for candidate in candidates:
        result = get_coordinates(candidate)
        if result:
            session_state["location"] = f"{result['name']}, {result['state']}"
            session_state["latitude"] = result["latitude"]
            session_state["longitude"] = result["longitude"]
            print(f"[APP] Location detected: {session_state['location']}")
            return


# ── Gradio Event Handlers ────────────────────────────────────────────

def on_start_click():
    """Handle 'Start Conversation' button — switch from hero to app view."""
    greeting = get_greeting()
    chat_history = [{"role": "assistant", "content": greeting}]
    state = _new_session_state()
    state["phase"] = "greeting"

    # Generate greeting TTS
    audio_path = text_to_speech(greeting)

    return (
        gr.update(visible=False),   # hero_section → hide
        gr.update(visible=True),    # app_section → show
        gr.update(visible=False),   # explore_section → hide
        chat_history,               # chatbot
        state,                      # session_state
        audio_path,                 # audio_output
        gr.update(visible=False), # image_preview → hide
        None,                       # pending_image state → clear
    )

def on_explore_click():
    """Handle 'Explore Features' button — switch to Explore view."""
    return (
        gr.update(visible=False), # hero_section
        gr.update(visible=False), # app_section
        gr.update(visible=True),  # explore_section
    )

def on_back_to_hero():
    """Handle back button — switch to Hero view."""
    return (
        gr.update(visible=True),  # hero_section
        gr.update(visible=False), # app_section
        gr.update(visible=False), # explore_section
    )


def on_text_submit(
    user_text: str,
    chat_history,
    session_state,
    weather_card_md: str,
    pending_image,
):
    """Handle text input submission, optionally with a pending image."""
    has_text = user_text and user_text.strip()
    has_image = pending_image is not None

    if not has_text and not has_image:
        return "", chat_history, session_state, None, weather_card_md, gr.update(), None

    # If there's a pending image, process it with disease detection
    if has_image:
        # Add image to chat as user message
        chat_history.append({
            "role": "user",
            "content": gr.Image(value=pending_image),
        })

        # Run disease detection on the uploaded image
        detection = detect_disease(pending_image)

        if detection and not detection.get("is_uncertain"):
            # Confident detection — build detailed prompt for LLM
            crop = detection["crop"]
            disease = detection["disease"]
            hinglish = detection["disease_hinglish"]
            confidence = detection["confidence"]
            top_3 = detection["top_3"]

            # Update session state with detected info
            session_state["last_crop"] = crop
            if not detection["is_healthy"]:
                session_state["last_disease"] = disease

            top_3_str = ", ".join(
                f"{label} ({score:.0%})" for label, score in top_3
            )

            if detection["is_healthy"]:
                image_msg = (
                    f"[AI IMAGE ANALYSIS: Farmer ne '{crop}' ki photo upload ki hai. "
                    f"Analysis result: Plant looks HEALTHY ({confidence:.0%} confidence). "
                    f"Farmer ko bataiye ki unki fasal swasth dikh rahi hai. "
                    f"Koi aur madad chahiye toh poochein.]"
                )
            else:
                farmer_note = f" Farmer ka message: '{user_text}'." if has_text else ""
                image_msg = (
                    f"[AI IMAGE ANALYSIS: Farmer ne fasal ki photo upload ki hai.\n"
                    f"- Detected crop: {crop}\n"
                    f"- Detected disease: {disease} / {hinglish}\n"
                    f"- Confidence: {confidence:.0%}\n"
                    f"- Other possibilities: {top_3_str}\n"
                    f"{farmer_note}\n"
                    f"Is disease ke baare mein organic/natural ilaaj bataiye. "
                    f"Disease ka naam, urgency level, aur step-by-step organic "
                    f"remedy dein. ZBNF/natural farming ke tarike se ilaaj bataiye.]"
                )
        elif detection and detection.get("is_uncertain"):
            # Low confidence — ask farmer to describe symptoms
            farmer_note = f" Farmer ne yeh bhi likha: '{user_text}'." if has_text else ""
            image_msg = (
                f"[AI IMAGE ANALYSIS: Farmer ne photo upload ki hai lekin AI model "
                f"ko confident result nahi mila (top prediction: {detection['disease']} "
                f"at {detection['confidence']:.0%}).{farmer_note} "
                f"Farmer se poochein: (1) Yeh konsi fasal hai? (2) Kya symptoms dikh "
                f"rahe hain — patte peele ho rahe hain, dhabbe hain, murjha rahi hai? "
                f"(3) Kitne dinon se yeh problem hai? Taaki aap sahi diagnosis de sakein.]"
            )
        else:
            # Detection failed entirely — fallback to asking farmer
            farmer_note = f" Farmer ka message: '{user_text}'." if has_text else ""
            image_msg = (
                f"[Farmer ne ek photo upload ki hai lekin image analysis available "
                f"nahi hai abhi.{farmer_note} Farmer se poochein ki yeh konsi fasal "
                f"hai, kya dikkat dikh rahi hai, aur kitne dinon se yeh problem hai.]"
            )

        chat_history, session_state, audio_path, weather_card_md = _process_message(
            image_msg, chat_history, session_state, weather_card_md
        )
        return (
            "",                 # clear textbox
            chat_history,
            session_state,
            audio_path,
            weather_card_md,
            gr.update(visible=False),  # hide image preview
            None,               # clear pending_image state
        )

    # Text-only message
    chat_history, session_state, audio_path, weather_card_md = _process_message(
        user_text, chat_history, session_state, weather_card_md
    )
    return (
        "",                 # clear textbox
        chat_history,
        session_state,
        audio_path,
        weather_card_md,
        gr.update(),        # no change to image preview
        None,               # keep pending_image unchanged
    )


def on_audio_record(
    audio_input,
    chat_history,
    session_state,
    weather_card_md: str,
):
    """Handle voice input — transcribe then process."""
    if audio_input is None:
        return chat_history, session_state, None, weather_card_md

    # Transcribe audio
    transcribed_text = transcribe_audio(audio_input)

    if not transcribed_text:
        # Let the user know transcription failed
        chat_history.append({
            "role": "assistant",
            "content": "Maaf kijiye, aapki awaaz samajh nahi aayi. Kripya dobara try karein ya type karke bhejein."
        })
        return chat_history, session_state, None, weather_card_md

    # Process through LLM
    chat_history, session_state, audio_path, weather_card_md = _process_message(
        transcribed_text, chat_history, session_state, weather_card_md
    )

    return chat_history, session_state, audio_path, weather_card_md


def on_image_select(image_path):
    """Handle image file selection — show preview, store in state."""
    if image_path is None:
        return gr.update(visible=False), None
    # Show preview column, store path in pending state
    return gr.update(visible=True), image_path


def on_clear_image():
    """Clear the pending image preview."""
    return gr.update(visible=False), None


# ── Suggestion Chip Handler ─────────────────────────────────────────

def on_explore_location_submit(location_text: str):
    """Handle location submission in Explore Features view."""
    if not location_text or not location_text.strip():
        return gr.update(), gr.update(), gr.update()
        
    result = get_coordinates(location_text)
    if not result:
        return (
            gr.update(visible=True, value="❌ Location not found. Please try again with a valid City or PIN code."),
            gr.update(visible=False),
            gr.update(visible=False)
        )
        
    forecast = get_weather_forecast(result["latitude"], result["longitude"])
    location_str = f"{result['name']}, {result['state']}"
    
    if not forecast:
        return (
            gr.update(visible=True, value=f"❌ Could not fetch weather for {location_str}."),
            gr.update(visible=False),
            gr.update(visible=False)
        )
        
    weather_md = format_weather_card(forecast, location_str)
    
    # Get LLM crop recommendations
    import datetime
    current_month = datetime.datetime.now().strftime("%B")
    
    prompt = (
        f"You are Krishi Mitra, an AI agricultural consultant. A farmer from {location_str} is asking for crop recommendations.\n"
        f"Current month: {current_month}\n"
        f"7-day Weather Forecast Summary:\n"
        f"Max Temp: {max(d['temp_max'] for d in forecast['daily'])}°C, Min Temp: {min(d['temp_min'] for d in forecast['daily'])}°C\n"
        f"Total Rain expected: {sum(d['precipitation'] for d in forecast['daily']):.1f}mm\n\n"
        f"Based on this region, current season, and upcoming weather, recommend 2-3 suitable crops for natural/organic farming.\n"
        f"Format as Markdown with a brief explanation and one organic farming tip per crop. Be concise."
    )
    
    # We can use the chat function but bypassing history
    from src.llm_engine import get_llm_client
    from src.config import LLM_MODEL
    
    try:
        client = get_llm_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
            extra_headers={
                "HTTP-Referer": "https://krishi-mitra.app",
                "X-Title": "Krishi Mitra",
            },
        )
        recommendations = response.choices[0].message.content.strip()
    except Exception as e:
        recommendations = f"❌ Error getting recommendations: {e}"
        
    return (
        gr.update(visible=True, value=weather_md),
        gr.update(visible=True, value=f"### 🌾 Crop Recommendations for {location_str}\n\n{recommendations}"),
        gr.update(visible=True) # start chat button
    )

def on_suggestion_click(
    suggestion: str,
    chat_history: list,
    session_state: dict,
    weather_card_md: str,
):
    """Handle suggestion chip click — send as text message."""
    return on_text_submit(suggestion, chat_history, session_state, weather_card_md, None)


# ── Build the Gradio App ─────────────────────────────────────────────

def create_app() -> gr.Blocks:
    """Construct and return the full Gradio Blocks application."""

    global _css_content, _js_init, _js_head
    _css_content = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""

    # JS: Force dark theme + mic recording + read-aloud on every bot response
    _js_init = """
    () => {
        // Force dark theme
        const url = new URL(window.location);
        if (!url.searchParams.has('__theme')) {
            url.searchParams.set('__theme', 'dark');
            window.location.replace(url.toString());
            return;
        }

        // ═══════════════════════════════════════════
        //  UTILITY: Find the real <button> inside a Gradio component
        // ═══════════════════════════════════════════
        function findActualButton(wrapper) {
            if (!wrapper) return null;
            if (wrapper.tagName === 'BUTTON') return wrapper;
            return wrapper.querySelector('button') || wrapper;
        }

        function findTextarea() {
            // Gradio textbox: could be textarea or input inside the #text-input wrapper
            const wrapper = document.getElementById('text-input');
            if (!wrapper) return null;
            return wrapper.querySelector('textarea') || wrapper.querySelector('input[type="text"]') || wrapper.querySelector('input');
        }

        function setGradioTextboxValue(el, value) {
            // Gradio uses Svelte — we need to set the value via native setter
            // and dispatch the right events so Svelte picks up the change
            if (!el) return;
            el.focus();
            // Try native setter for both textarea and input
            const proto = el.tagName === 'TEXTAREA'
                ? window.HTMLTextAreaElement.prototype
                : window.HTMLInputElement.prototype;
            const descriptor = Object.getOwnPropertyDescriptor(proto, 'value');
            if (descriptor && descriptor.set) {
                descriptor.set.call(el, value);
            } else {
                el.value = value;
            }
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
        }

        // ═══════════════════════════════════════════
        //  MIC BUTTON — Web Speech API (with robust fallback)
        // ═══════════════════════════════════════════
        let micRecognition = null;
        let micIsRecording = false;
        let micSetupDone = false;

        function setupMicButton() {
            if (micSetupDone) return;
            const micWrapper = document.getElementById('mic-trigger-btn');
            if (!micWrapper) { setTimeout(setupMicButton, 500); return; }

            const micBtn = findActualButton(micWrapper);
            if (!micBtn) { setTimeout(setupMicButton, 500); return; }

            // Check browser support
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRecognition) {
                console.warn('[KM] Web Speech API not supported in this browser');
                micWrapper.style.opacity = '0.3';
                micWrapper.title = 'Speech recognition not supported in this browser';
                micSetupDone = true;
                return;
            }

            // Visual feedback elements
            const statusBubble = document.createElement('div');
            statusBubble.id = 'km-mic-status';
            statusBubble.style.cssText = 'display:none; position:fixed; bottom:90px; left:50%; transform:translateX(-50%); background:rgba(239,68,68,0.9); color:white; padding:8px 20px; border-radius:20px; font-size:13px; font-family:Inter,sans-serif; z-index:9999; backdrop-filter:blur(8px); box-shadow:0 4px 20px rgba(239,68,68,0.3);';
            document.body.appendChild(statusBubble);

            function showStatus(msg) {
                statusBubble.textContent = msg;
                statusBubble.style.display = 'block';
            }
            function hideStatus() {
                statusBubble.style.display = 'none';
            }

            function toggleRecording(e) {
                if (e) { e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation(); }

                if (micIsRecording) {
                    // Stop
                    if (micRecognition) micRecognition.stop();
                    micWrapper.classList.remove('km-recording');
                    micIsRecording = false;
                    hideStatus();
                    return;
                }

                // Start
                micRecognition = new SpeechRecognition();
                micRecognition.lang = 'hi-IN';
                micRecognition.interimResults = false;
                micRecognition.maxAlternatives = 1;
                micRecognition.continuous = false;

                micRecognition.onstart = () => {
                    micIsRecording = true;
                    micWrapper.classList.add('km-recording');
                    showStatus('🎤 Sun raha hoon... Boliye!');
                    console.log('[KM] Speech recognition started');
                };

                micRecognition.onresult = (event) => {
                    const transcript = event.results[0][0].transcript;
                    console.log('[KM] Transcript:', transcript);
                    if (transcript && transcript.trim()) {
                        showStatus('✓ ' + transcript.substring(0, 40) + (transcript.length > 40 ? '...' : ''));

                        const textEl = findTextarea();
                        if (textEl) {
                            setGradioTextboxValue(textEl, transcript);

                            // Auto-submit after brief delay
                            setTimeout(() => {
                                hideStatus();
                                const sendWrapper = document.getElementById('send-btn');
                                const sendBtn = findActualButton(sendWrapper);
                                if (sendBtn) {
                                    sendBtn.click();
                                }
                            }, 500);
                        } else {
                            console.warn('[KM] Could not find text input');
                            hideStatus();
                        }
                    }
                };

                micRecognition.onerror = (event) => {
                    console.warn('[KM] Speech recognition error:', event.error);
                    micWrapper.classList.remove('km-recording');
                    micIsRecording = false;
                    if (event.error === 'not-allowed') {
                        showStatus('❌ Microphone permission denied');
                    } else if (event.error === 'no-speech') {
                        showStatus('🔇 Koi awaaz nahi sunai di');
                    } else {
                        showStatus('❌ Error: ' + event.error);
                    }
                    setTimeout(hideStatus, 3000);
                };

                micRecognition.onend = () => {
                    micWrapper.classList.remove('km-recording');
                    micIsRecording = false;
                    if (statusBubble.textContent.startsWith('🎤')) {
                        hideStatus();
                    }
                    console.log('[KM] Speech recognition ended');
                };

                try {
                    micRecognition.start();
                } catch(err) {
                    console.error('[KM] Failed to start speech recognition:', err);
                    showStatus('❌ Mic start nahi ho paya');
                    setTimeout(hideStatus, 3000);
                }
            }

            // Attach to both the wrapper and the actual button
            micBtn.addEventListener('click', toggleRecording, true);
            if (micBtn !== micWrapper) {
                micWrapper.addEventListener('click', toggleRecording, true);
            }

            micSetupDone = true;
            console.log('[KM] Mic button setup complete');
        }

        // ═══════════════════════════════════════════
        //  READ-ALOUD BUTTONS on every bot message
        // ═══════════════════════════════════════════
        let currentSpeakingBtn = null;

        function stopAllSpeech() {
            window.speechSynthesis.cancel();
            if (currentSpeakingBtn) {
                currentSpeakingBtn.classList.remove('km-speaking');
                currentSpeakingBtn.querySelector('.km-ra-icon-play').style.display = '';
                currentSpeakingBtn.querySelector('.km-ra-icon-stop').style.display = 'none';
                currentSpeakingBtn = null;
            }
        }

        function createReadAloudBtn(textContent) {
            const btn = document.createElement('button');
            btn.className = 'km-read-aloud-btn';
            btn.title = 'Sunein (Read Aloud)';
            btn.innerHTML = `
                <span class="km-ra-icon-play">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                        <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
                        <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
                    </svg>
                </span>
                <span class="km-ra-icon-stop" style="display:none">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" stroke="none">
                        <rect x="6" y="6" width="12" height="12" rx="2"/>
                    </svg>
                </span>
                <span class="km-ra-label">Sunein</span>
            `;

            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                e.preventDefault();

                const playIcon = btn.querySelector('.km-ra-icon-play');
                const stopIcon = btn.querySelector('.km-ra-icon-stop');
                const label = btn.querySelector('.km-ra-label');

                if (btn.classList.contains('km-speaking')) {
                    // Stop
                    stopAllSpeech();
                    return;
                }

                // Stop any other currently speaking
                stopAllSpeech();

                const utterance = new SpeechSynthesisUtterance(textContent);
                utterance.lang = 'hi-IN';
                utterance.rate = 0.95;
                utterance.pitch = 1.0;

                // Try to find a Hindi voice
                const voices = window.speechSynthesis.getVoices();
                const hindiVoice = voices.find(v => v.lang.startsWith('hi'));
                if (hindiVoice) utterance.voice = hindiVoice;

                utterance.onstart = () => {
                    btn.classList.add('km-speaking');
                    playIcon.style.display = 'none';
                    stopIcon.style.display = '';
                    label.textContent = 'Rok dein';
                    currentSpeakingBtn = btn;
                };

                utterance.onend = () => {
                    btn.classList.remove('km-speaking');
                    playIcon.style.display = '';
                    stopIcon.style.display = 'none';
                    label.textContent = 'Sunein';
                    currentSpeakingBtn = null;
                };

                utterance.onerror = () => {
                    btn.classList.remove('km-speaking');
                    playIcon.style.display = '';
                    stopIcon.style.display = 'none';
                    label.textContent = 'Sunein';
                    currentSpeakingBtn = null;
                };

                window.speechSynthesis.speak(utterance);
            });

            return btn;
        }

        function extractBotText(messageEl) {
            // Get text from markdown/prose content, skip any button text
            const content = messageEl.querySelector('.prose, .markdown, .message-bubble, .chatbot-text');
            const source = content || messageEl;
            // Clone to remove button text from extraction
            const clone = source.cloneNode(true);
            clone.querySelectorAll('.km-read-aloud-btn, button').forEach(b => b.remove());
            return (clone.textContent || clone.innerText || '').trim();
        }

        function addReadAloudButtons() {
            const chatbot = document.getElementById('chatbot');
            if (!chatbot) return;

            // Gradio 6.x uses different class patterns across versions:
            // - Some versions: .message-row.bot-row (role + '-row')
            // - Some versions: .message-row.bot (just role)
            // Try both, plus fallback to avatar-based detection
            let botRows = chatbot.querySelectorAll('.message-row.bot-row, .message-row.bot');

            if (botRows.length === 0) {
                // Fallback: find message rows that contain avatar images (bot messages have avatars)
                const allRows = chatbot.querySelectorAll('.message-row');
                const filtered = [];
                allRows.forEach(row => {
                    if (row.querySelector('img') ||
                        row.querySelector('.avatar-container') ||
                        row.querySelector('button[aria-label*="Copy"]') ||
                        row.querySelector('button[title*="Copy"]') ||
                        row.classList.contains('bot') ||
                        row.classList.contains('bot-row')) {
                        filtered.push(row);
                    }
                });
                botRows = filtered;
            }

            const rows = Array.isArray(botRows) ? botRows : Array.from(botRows);

            rows.forEach(row => {
                // Skip if already has our button
                if (row.querySelector('.km-read-aloud-btn')) return;

                const textContent = extractBotText(row);
                if (!textContent || textContent.length < 10) return;

                const btn = createReadAloudBtn(textContent);

                // Find where to insert — try multiple strategies
                // 1. Try known action bar classes
                let actionBar = row.querySelector('.message-actions, .icon-buttons, .message-buttons-bot');

                // 2. If not found, find the copy button and use its parent
                if (!actionBar) {
                    const copyBtn = row.querySelector('button[aria-label*="Copy"], button[title*="Copy"], .icon-button');
                    if (copyBtn && copyBtn.parentElement) {
                        actionBar = copyBtn.parentElement;
                    }
                }

                if (actionBar) {
                    actionBar.appendChild(btn);
                } else {
                    // Fallback: create our own container below the message
                    const container = document.createElement('div');
                    container.className = 'km-read-aloud-container';
                    container.appendChild(btn);
                    row.appendChild(container);
                }
            });
        }

        // ═══════════════════════════════════════════
        //  OBSERVER — Watch for new messages
        // ═══════════════════════════════════════════
        let debounceTimer = null;
        const observer = new MutationObserver(() => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                addReadAloudButtons();
            }, 200);
        });

        function startObserving() {
            const chatbot = document.getElementById('chatbot');
            if (chatbot) {
                observer.observe(chatbot, { childList: true, subtree: true });
                // Run immediately + after a delay for dynamic content
                addReadAloudButtons();
                setTimeout(addReadAloudButtons, 1000);
                setTimeout(addReadAloudButtons, 3000);
            } else {
                setTimeout(startObserving, 500);
            }
        }

        // Preload voices
        if (window.speechSynthesis) {
            window.speechSynthesis.getVoices();
            window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
        }

        // Initialize both features
        setupMicButton();
        startObserving();

        // Re-check mic button setup when the app section becomes visible
        // (after hero → app transition)
        function setupBodyObserver() {
            const body = document.body;
            if (!body) { setTimeout(setupBodyObserver, 200); return; }

            const appObserver = new MutationObserver(() => {
                if (!micSetupDone) setupMicButton();
                setTimeout(addReadAloudButtons, 500);
            });
            appObserver.observe(body, { childList: true, subtree: true, attributes: true, attributeFilter: ['style', 'class'] });
        }
        setupBodyObserver();

        console.log('[KM] JavaScript initialization complete');
    }
    """

    # Wrap the JS in a self-executing <script> tag for injection into <head>
    # Uses setInterval polling since Gradio components mount async after DOM ready
    _js_head = "<script>(" + _js_init + ")();</script>"

    with gr.Blocks(
        title="Krishi Mitra -- AI Agricultural Consultant",
        css=_css_content,
        js=_js_init,
        head=_js_head,
        theme=gr.themes.Base(),
    ) as app:

        # ── State ──
        session_state = gr.State(value=_new_session_state)
        weather_card_state = gr.State(value="")
        pending_image = gr.State(value=None)

        # ════════════════════════════════════════════
        #  VIEW 1: Hero Landing Page
        # ════════════════════════════════════════════
        with gr.Column(visible=True, elem_id="hero-wrapper") as hero_section:
            gr.HTML(_build_hero_html())
            with gr.Row(elem_classes=["hero-gradio-buttons"]):
                hero_start_btn = gr.Button(
                    "Baat Shuru Karein",
                    elem_id="hero-start-btn",
                    elem_classes=["hero-btn-primary-gradio"],
                )
                hero_explore_btn = gr.Button(
                    "Explore Features",
                    elem_id="hero-explore-btn",
                    elem_classes=["hero-btn-secondary-gradio"],
                )

        # ════════════════════════════════════════════
        #  VIEW 3: Explore Features Interface
        # ════════════════════════════════════════════
        with gr.Column(visible=False, elem_id="explore-container") as explore_section:
            with gr.Row(elem_id="explore-topbar"):
                explore_back_btn = gr.Button("← Back to Home", elem_classes=["back-btn"], size="sm")
                gr.HTML("""
                    <div class="topbar-left" style="margin-left:auto;">
                        <span class="topbar-logo-badge">KM</span>
                        <span class="topbar-title">Krishi Mitra</span>
                    </div>
                """)

            gr.Markdown("## 🌤️ Farming Recommendations\nApna location bataiye aur jaaniye is season mein kya bona chahiye", elem_classes=["explore-title"])
            
            with gr.Row(elem_classes=["explore-input-row"]):
                explore_location_input = gr.Textbox(placeholder="City or PIN code...", show_label=False, container=False, scale=4)
                explore_location_submit = gr.Button("Get Recommendations", elem_classes=["explore-btn"], size="sm", scale=1)
            
            explore_weather_card = gr.Markdown(visible=False, elem_classes=["explore-card"])
            explore_recommendations_card = gr.Markdown(visible=False, elem_classes=["explore-card"])
            
            with gr.Row(elem_classes=["explore-action-row"]):
                explore_start_chat_btn = gr.Button("📞 Start Consultation →", elem_classes=["hero-btn-primary-gradio", "explore-action-btn"], visible=False)

        # ════════════════════════════════════════════
        #  VIEW 2: Consultation Interface
        # ════════════════════════════════════════════
        with gr.Column(visible=False, elem_id="app-container") as app_section:

            # ── Top Bar ──
            with gr.Row(elem_id="app-topbar"):
                gr.HTML("""
                    <div class="topbar-left">
                        <span class="topbar-logo-badge">KM</span>
                        <span class="topbar-title">Krishi Mitra</span>
                    </div>
                """)
                weather_card = gr.Markdown(
                    value="",
                    elem_id="weather-display",
                    elem_classes=["weather-inline"],
                )

            # ── Chat Area (centered, scrollable) ──
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                height=480,
                show_label=False,
                show_copy_button=True,
                layout="bubble",
                avatar_images=(None, AVATAR_URL),
                type="messages",
            )

            # ── Suggestion Chips (shown below greeting) ──
            with gr.Row(elem_id="suggestion-chips", elem_classes=["suggestion-row"]):
                chip1 = gr.Button(
                    "Fasal ki bimari",
                    elem_classes=["chip-btn"],
                    size="sm",
                )
                chip2 = gr.Button(
                    "Mausam planning",
                    elem_classes=["chip-btn"],
                    size="sm",
                )
                chip3 = gr.Button(
                    "Natural farming",
                    elem_classes=["chip-btn"],
                    size="sm",
                )
                chip4 = gr.Button(
                    "Sarkari yojana",
                    elem_classes=["chip-btn"],
                    size="sm",
                )

            # ── Image Preview (hidden by default) ──
            with gr.Column(elem_id="image-preview-row", visible=False) as image_preview_row:
                image_preview = gr.Image(
                    elem_id="image-preview",
                    label="",
                    show_label=False,
                    interactive=False,
                    height=120,
                    elem_classes=["image-preview-box"],
                )
                clear_image_btn = gr.Button(
                    "Remove",
                    elem_id="clear-image-btn",
                    elem_classes=["clear-image-btn"],
                    size="sm",
                )

            # ── Unified Input Bar ──
            with gr.Row(elem_id="input-bar", elem_classes=["input-bar"]):
                image_upload_btn = gr.UploadButton(
                    label="",
                    file_types=["image"],
                    elem_id="img-upload-btn",
                    elem_classes=["img-upload-icon"],
                    size="sm",
                    icon=str(PROJECT_ROOT / "assets" / "paperclip.svg"),
                )
                text_input = gr.Textbox(
                    elem_id="text-input",
                    placeholder="Apna sawaal type karein...",
                    show_label=False,
                    lines=1,
                    scale=6,
                    container=False,
                )
                mic_btn = gr.Button(
                    "",
                    elem_id="mic-trigger-btn",
                    elem_classes=["mic-btn-icon"],
                    size="sm",
                    icon=str(PROJECT_ROOT / "assets" / "mic.svg"),
                )
                send_btn = gr.Button(
                    "",
                    elem_id="send-btn",
                    elem_classes=["send-btn-icon"],
                    size="sm",
                    icon=None,
                )

            # ── Hidden Audio Input (server-side STT fallback) ──
            audio_input = gr.Audio(
                elem_id="mic-btn-hidden",
                sources=["microphone"],
                type="numpy",
                label="",
                show_label=False,
                visible=False,
            )

            # ── Hidden Audio Output (autoplay, no visible player) ──
            audio_output = gr.Audio(
                elem_id="audio-output",
                label="",
                type="filepath",
                autoplay=True,
                interactive=False,
                visible=False,
            )

        # ══════════════════════════════════════════════
        #  EVENT WIRING
        # ══════════════════════════════════════════════

        # Hero buttons
        _hero_outputs = [
            hero_section, app_section, explore_section, chatbot, session_state,
            audio_output, image_preview, pending_image,
        ]
        hero_start_btn.click(
            fn=on_start_click,
            inputs=[],
            outputs=_hero_outputs,
        )
        hero_explore_btn.click(
            fn=on_explore_click,
            inputs=[],
            outputs=[hero_section, app_section, explore_section],
        )

        # Explore view buttons
        explore_back_btn.click(
            fn=on_back_to_hero,
            inputs=[],
            outputs=[hero_section, app_section, explore_section],
        )
        explore_start_chat_btn.click(
            fn=on_start_click,
            inputs=[],
            outputs=_hero_outputs,
        )
        explore_location_submit.click(
            fn=on_explore_location_submit,
            inputs=[explore_location_input],
            outputs=[explore_weather_card, explore_recommendations_card, explore_start_chat_btn],
        )
        explore_location_input.submit(
            fn=on_explore_location_submit,
            inputs=[explore_location_input],
            outputs=[explore_weather_card, explore_recommendations_card, explore_start_chat_btn],
        )

        # Image upload button → preview
        image_upload_btn.upload(
            fn=on_image_select,
            inputs=[image_upload_btn],
            outputs=[image_preview_row, pending_image],
        ).then(
            fn=lambda img: gr.update(value=img),
            inputs=[pending_image],
            outputs=[image_preview],
        )

        # Clear image preview
        clear_image_btn.click(
            fn=on_clear_image,
            inputs=[],
            outputs=[image_preview_row, pending_image],
        )

        # Text submit (button click or Enter key)
        _text_inputs = [text_input, chatbot, session_state, weather_card_state, pending_image]
        _text_outputs = [
            text_input, chatbot, session_state, audio_output,
            weather_card_state, image_preview_row, pending_image,
        ]

        send_btn.click(
            fn=on_text_submit,
            inputs=_text_inputs,
            outputs=_text_outputs,
        ).then(
            fn=lambda wc: wc if wc else "",
            inputs=[weather_card_state],
            outputs=[weather_card],
        )

        text_input.submit(
            fn=on_text_submit,
            inputs=_text_inputs,
            outputs=_text_outputs,
        ).then(
            fn=lambda wc: wc if wc else "",
            inputs=[weather_card_state],
            outputs=[weather_card],
        )

        # Voice input — triggers when recording stops
        audio_input.stop_recording(
            fn=on_audio_record,
            inputs=[audio_input, chatbot, session_state, weather_card_state],
            outputs=[chatbot, session_state, audio_output, weather_card_state],
        ).then(
            fn=lambda wc: wc if wc else "",
            inputs=[weather_card_state],
            outputs=[weather_card],
        )

        # Suggestion chips
        _chip_inputs = [chatbot, session_state, weather_card_state]
        _chip_outputs = [
            text_input, chatbot, session_state, audio_output,
            weather_card_state, image_preview_row, pending_image,
        ]

        chip1.click(
            fn=lambda ch, ss, wc: on_text_submit("Mere tomato ke patte peele ho rahe hain, kya karna chahiye?", ch, ss, wc, None),
            inputs=_chip_inputs,
            outputs=_chip_outputs,
        ).then(fn=lambda wc: wc if wc else "", inputs=[weather_card_state], outputs=[weather_card])

        chip2.click(
            fn=lambda ch, ss, wc: on_text_submit("Is season mein kya bona chahiye? Main Karnal, Haryana se hoon.", ch, ss, wc, None),
            inputs=_chip_inputs,
            outputs=_chip_outputs,
        ).then(fn=lambda wc: wc if wc else "", inputs=[weather_card_state], outputs=[weather_card])

        chip3.click(
            fn=lambda ch, ss, wc: on_text_submit("Mujhe natural farming ke baare mein bataiye — jeevamrit kaise banate hain?", ch, ss, wc, None),
            inputs=_chip_inputs,
            outputs=_chip_outputs,
        ).then(fn=lambda wc: wc if wc else "", inputs=[weather_card_state], outputs=[weather_card])

        chip4.click(
            fn=lambda ch, ss, wc: on_text_submit("PM-KISAN yojana ke baare mein bataiye — kaise apply karein?", ch, ss, wc, None),
            inputs=_chip_inputs,
            outputs=_chip_outputs,
        ).then(fn=lambda wc: wc if wc else "", inputs=[weather_card_state], outputs=[weather_card])

    return app


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Krishi Mitra -- AI Agricultural Consultant")
    print("=" * 60)

    print("\n[INIT] RAG knowledge base will load on first query.")
    print("[INIT] Starting Gradio server...\n")

    # Create and launch app
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )

