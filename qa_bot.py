"""
Slack QA Bot — Always-online bot that:
1. Listens for messages containing .wav URLs in specified channel(s)
2. Downloads and transcribes audio via OpenAI Whisper (local, free, accurate)
3. Analyzes dialogue quality via Ollama LLM
4. Logs the analysis (ANALYZE_ONLY mode) or reacts + replies (production)

Requirements:
    pip install slack_bolt slack_sdk openai-whisper ollama python-dotenv requests

    Whisper also needs ffmpeg:
        brew install ffmpeg

Environment Variables (.env):
    SLACK_BOT_TOKEN=xoxp-...
    SLACK_APP_TOKEN=xapp-...
    OLLAMA_MODEL=hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q8_0
    QA_CHANNEL_ID=C08S6HHRH8F
    WHISPER_MODEL=base       # Options: tiny, base, small, medium, large
"""

import os
import re
import logging
import tempfile
import requests

from dotenv import load_dotenv
import whisper
import ollama

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

# ─── Configuration ───────────────────────────────────────────────────────────

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q8_0",
)
_raw_channel = os.environ.get("QA_CHANNEL_ID", "")
QA_CHANNEL_ID = _raw_channel if _raw_channel.startswith("C") and len(_raw_channel) > 5 else ""

# Whisper model size: tiny (fastest) → base → small → medium → large (most accurate)
# "base" is a good balance for phone calls. Use "small" or "medium" if you have GPU.
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  ANALYZE_ONLY = True  →  downloads, transcribes, runs LLM, LOGS only      │
# │                          NO reactions, NO replies in Slack                  │
# │  ANALYZE_ONLY = False →  full production: reacts + replies in Slack        │
# └─────────────────────────────────────────────────────────────────────────────┘
ANALYZE_ONLY = True

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

file_handler = logging.FileHandler("qa_bot.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(file_handler)

# ─── Load Whisper Model (once at startup) ────────────────────────────────────

logger.info(f"Loading Whisper model '{WHISPER_MODEL_SIZE}'... (first run downloads it)")
whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
logger.info(f"Whisper model '{WHISPER_MODEL_SIZE}' loaded ✅")

# ─── Slack App ───────────────────────────────────────────────────────────────

app = App(token=SLACK_BOT_TOKEN)

GRADE_EMOJI_MAP = {
    "↗️": "arrow_upper_right",
    "💚": "green_heart",
    "🔥": "fire",
    "💛": "yellow_heart",
    "🧡": "orange_heart",
    "❤️": "heart",
    "😡": "rage",
    "👂": "ear_with_hearing_aid",
    "❌": "x",
}

# ─── WAV URL Extraction ─────────────────────────────────────────────────────

WAV_URL_PATTERN = re.compile(r'https?://[^\s<>|]+\.wav', re.IGNORECASE)


def extract_wav_urls(text: str) -> list[str]:
    """Extract .wav URLs from Slack message text (handles <url|label> format)."""
    slack_links = re.findall(r'<(https?://[^|>]+\.wav)[|>]', text, re.IGNORECASE)
    if slack_links:
        return slack_links
    return WAV_URL_PATTERN.findall(text)


# ─── Audio Processing ────────────────────────────────────────────────────────

def download_wav(url: str, dest_path: str):
    """Download a .wav file from a public URL."""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)
    logger.info(f"  Downloaded {dest_path} ({len(resp.content):,} bytes)")


def transcribe_wav(wav_path: str) -> str:
    """Transcribe a WAV file using OpenAI Whisper (local model)."""
    result = whisper_model.transcribe(
        wav_path,
        language="en",       # Force English for HVAC calls
        fp16=False,          # CPU-safe; set True if you have CUDA GPU
    )
    text = result.get("text", "").strip()
    return text


# ─── LLM Analysis ────────────────────────────────────────────────────────────

QA_PROMPT_TEMPLATE = """You are an extremely strict Quality Assurance Auditor for an AI phone agent (voice bot) that handles incoming customer calls for an HVAC/appliance repair company. You evaluate transcribed phone conversations between the AI agent and customers.

GRADING SCALE — select exactly ONE primary grade:

↗️ SHORT CALL / HAND-OFF (no score):
  The agent transferred or attempted to transfer the call to a human operator.
  This includes ANY of these phrases from the agent: "let me transfer you", "I'll connect you",
  "transferring you now", "let me get someone", "hold on while I transfer", "connecting you to",
  "a member of our team", "one of our team members will", or similar transfer language.
  Also applies if the conversation was very short (under ~30 seconds of real content).
  If the agent mentioned transferring — mark ↗️ regardless of everything else. Do NOT grade quality.

💚 5/5 EXCEPTIONAL:
  Flawless call. Agent was natural, empathetic, efficient, gathered all info correctly.
  Customer was clearly satisfied. So good it could be used as a promotional recording on the website.
  → If you give 💚 AND the call is truly ad-worthy, also add 🔥 (this combo is extremely rare).

💛 4/5 GOOD WITH MINOR SLIPS:
  Call succeeded. Agent made small mistakes the customer did NOT notice or care about.
  Examples: re-asked zip code for returning customer, slightly awkward phrasing, minor hesitation.
  Customer left satisfied and unaware of any issues.

🧡 3/5 CUSTOMER NOTICED ISSUES:
  Agent made errors the customer clearly noticed — pauses, confusion, wrong info self-corrected.
  Customer was NOT irritated, just mildly aware something was off. Call still completed.

❤️ 2/5 CUSTOMER IRRITATED:
  Customer became noticeably frustrated, annoyed, or impatient due to agent errors/loops/incompetence.
  Customer has NOT hung up but is at risk. An operator should intervene immediately.

😡 1/5 CUSTOMER LOST:
  Customer was so frustrated they ended the call, refused service, or became hostile.
  Agent completely failed. Operator must follow up immediately.

ADDITIONAL FLAGS (append after primary grade if applicable):
  👂 ECHO: Add if transcript shows echo patterns — agent words repeated back, doubled phrases,
    same sentence appearing twice in a way consistent with audio echo/feedback.

STRICT EVALUATION RULES:
- Start your response with EXACTLY one grade emoji (↗️, 💚, 💛, 🧡, ❤️, or 😡), optionally followed by 🔥 or 👂
- Then write a concise 2-3 sentence justification in English
- TRANSFER DETECTION IS PRIORITY: If the agent mentioned transferring to a human at ANY point in the call, the grade is ↗️. Period. Check for this FIRST before evaluating quality.
- Be HARSH on quality grades. 💚 is extremely rare. Most decent calls deserve 💛 at best.
- If the agent repeated itself even once unnecessarily → 🧡 maximum
- If the agent asked for info the customer already provided → automatic one-level downgrade
- If the agent looped the same question/phrase 2+ times → 🧡 maximum, likely ❤️
- If the agent looped 3+ times or produced nonsense → 😡
- Customer saying "what?", "you already asked that", "hello?", "are you there?" = sign of problems
- Short calls with transfer language = ↗️, do not overthink
- If transcript is mostly empty or unintelligible = ❌ (cannot evaluate)

TRANSCRIPT:
{transcript}

YOUR GRADE AND JUSTIFICATION:"""


def analyze_transcript(transcript: str) -> str:
    """Send transcript to Ollama for QA analysis."""
    prompt = QA_PROMPT_TEMPLATE.format(transcript=transcript)
    response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
    return response["response"]


def extract_grade_emojis(analysis: str) -> list[str]:
    """Extract grade emoji names from the analysis text."""
    found = []
    for icon, emoji_name in GRADE_EMOJI_MAP.items():
        if icon in analysis:
            found.append(emoji_name)
    return found


# ─── Core Processing Pipeline ───────────────────────────────────────────────

def process_wav_url(wav_url: str, message_ts: str = "", channel: str = "",
                    say=None, client=None):
    """Download → Transcribe → Analyze a single .wav URL."""
    logger.info(f"  🎵 Processing WAV: {wav_url}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        download_wav(wav_url, tmp_path)

        # Transcribe with Whisper
        logger.info("  🎙️ Transcribing with Whisper...")
        transcript = transcribe_wav(tmp_path)

        if not transcript or len(transcript.strip()) < 10:
            logger.warning(f"  ❌ No usable speech detected (transcript: '{transcript}')")

            if not ANALYZE_ONLY and client and message_ts and channel:
                try:
                    client.reactions_add(channel=channel, name="x", timestamp=message_ts)
                except Exception:
                    pass

            return "❌ No speech detected"

        logger.info(f"  📝 Transcript ({len(transcript)} chars):")
        logger.info(f"     {transcript[:600]}{'...' if len(transcript) > 600 else ''}")

        # Analyze with LLM
        logger.info("  🤖 Running QA analysis...")
        analysis = analyze_transcript(transcript)

        grade_emojis = extract_grade_emojis(analysis)
        grade_str = " ".join(f":{e}:" for e in grade_emojis) if grade_emojis else "???"

        logger.info(f"  📊 GRADE: {grade_str}")
        logger.info(f"  📊 ANALYSIS: {analysis}")

        # Production mode: react and reply in Slack
        if not ANALYZE_ONLY and client and say and message_ts and channel:
            for emoji_name in grade_emojis:
                try:
                    client.reactions_add(channel=channel, name=emoji_name, timestamp=message_ts)
                    logger.info(f"    ✅ Reacted: :{emoji_name}:")
                except Exception as e:
                    logger.warning(f"    ⚠️ Reaction failed ({emoji_name}): {e}")

            short_url = wav_url.split("/")[-1] if "/" in wav_url else wav_url
            reply_text = (
                f"*QA Analysis for* `{short_url}`\n\n"
                f"*Transcript:*\n>>> {transcript[:1500]}{'...' if len(transcript) > 1500 else ''}\n\n"
                f"*Assessment:*\n{analysis}"
            )
            say(text=reply_text, thread_ts=message_ts)

        return analysis

    except Exception as e:
        logger.exception(f"  ❌ Error processing {wav_url}: {e}")
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ─── Startup: Fetch & Analyze Latest Message ────────────────────────────────

def process_latest_message(channel_id: str):
    """Fetch the most recent message with a .wav URL and analyze it."""
    logger.info(f"\n{'='*60}")
    logger.info(f"STARTUP: Analyzing latest .wav from #{channel_id}")
    logger.info(f"{'='*60}")
    try:
        result = app.client.conversations_history(channel=channel_id, limit=20)
        messages = result.get("messages", [])

        for msg in messages:
            text = msg.get("text", "")
            wav_urls = extract_wav_urls(text)
            if wav_urls:
                user = msg.get("user", "unknown")
                ts = msg.get("ts", "")
                logger.info(f"  Found message from {user} (ts: {ts})")
                process_wav_url(wav_urls[0], message_ts=ts, channel=channel_id)
                return
        logger.info("  No messages with .wav URLs found in recent history")
    except Exception as e:
        logger.error(f"Failed to fetch messages: {e}")


# ─── Event Handler ───────────────────────────────────────────────────────────

@app.event("message")
def handle_message(event, say, client):
    channel = event.get("channel", "")
    user = event.get("user", "unknown")
    text = event.get("text", "")
    ts = event.get("ts", "")

    if QA_CHANNEL_ID and channel != QA_CHANNEL_ID:
        return

    wav_urls = extract_wav_urls(text)

    if not wav_urls:
        logger.info(f"📨 MSG (no wav) | User: {user} | {text[:100]}")
        return

    logger.info(f"\n📨 NEW WAV MESSAGE | User: {user} | ts: {ts}")
    logger.info(f"   Found {len(wav_urls)} .wav URL(s)")

    process_wav_url(
        wav_urls[0],
        message_ts=ts,
        channel=channel,
        say=say,
        client=client,
    )


# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode_label = "🔬 ANALYZE-ONLY (log only)" if ANALYZE_ONLY else "🚀 PRODUCTION"
    logger.info(f"{mode_label} — QA Bot starting...")
    logger.info(f"   Model: {OLLAMA_MODEL}")
    logger.info(f"   Whisper: {WHISPER_MODEL_SIZE}")
    logger.info(f"   Channel: {QA_CHANNEL_ID or 'ALL'}")

    if ANALYZE_ONLY:
        logger.info("\n   ⚠️  ANALYZE_ONLY=True — no reactions/replies, log only")
        logger.info("   ⚠️  Set ANALYZE_ONLY=False for production\n")

    if QA_CHANNEL_ID:
        process_latest_message(QA_CHANNEL_ID)

    logger.info(f"\n🔴 LIVE — waiting for new .wav messages (Ctrl+C to stop)\n")
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()