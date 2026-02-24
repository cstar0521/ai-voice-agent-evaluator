"""
Slack QA Bot — Always-online bot that:
1. Listens for .wav file uploads in specified channel(s)
2. Downloads and transcribes audio (silence-aware chunking)
3. Analyzes dialogue quality via Ollama LLM
4. Reacts on the message with the appropriate grade emoji
5. Posts a threaded reply with the full analysis

Requirements:
    pip install slack_bolt slack_sdk SpeechRecognition pydub ollama

Slack App Setup:
    1. Go to https://api.slack.com/apps → Create New App → From Scratch
    2. Enable Socket Mode (Settings → Socket Mode → Enable)
       - Create an App-Level Token with scope: connections:write
       - Save that token as SLACK_APP_TOKEN (xapp-...)
    3. OAuth & Permissions → Add Bot Token Scopes:
       - channels:history
       - channels:read
       - files:read
       - reactions:write
       - chat:write
       - groups:history  (if you need private channels)
    4. Event Subscriptions → Enable → Subscribe to bot events:
       - message.channels
       - message.groups  (for private channels)
    5. Install App to Workspace → Copy Bot User OAuth Token as SLACK_BOT_TOKEN (xoxb-...)
    6. Invite the bot to your target channel: /invite @YourBotName

Environment Variables:
    export SLACK_BOT_TOKEN="xoxb-your-token"
    export SLACK_APP_TOKEN="xapp-your-app-level-token"
    export OLLAMA_MODEL="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q8_0"  # optional
    export QA_CHANNEL_ID="C0123ABCDEF"  # optional: restrict to one channel
"""

import os
import io
import re
import logging
import tempfile
import requests

import speech_recognition as sr
import ollama
from pydub import AudioSegment
from pydub.silence import split_on_silence

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# ─── Configuration ───────────────────────────────────────────────────────────

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q8_0"
)
# Optional: restrict bot to a single channel (leave empty to listen everywhere)
QA_CHANNEL_ID = os.environ.get("QA_CHANNEL_ID", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Slack App ───────────────────────────────────────────────────────────────

app = App(token=SLACK_BOT_TOKEN)

# Grade icons → Slack emoji names
GRADE_EMOJI_MAP = {
    "💚": "green_heart",
    "💛": "yellow_heart",
    "🧡": "orange_heart",
    "❤️": "heart",
    "😡": "rage",
    "↗️": "arrow_upper_right",
}


# ─── Audio Processing ────────────────────────────────────────────────────────

def transcribe_wav(wav_path: str) -> str:
    """Transcribe a WAV file using Google Speech Recognition with silence splitting."""
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_wav(wav_path)

    speech_chunks = split_on_silence(
        audio,
        min_silence_len=500,
        silence_thresh=audio.dBFS - 14,
        keep_silence=250,
    )

    transcript_parts = []
    for i, chunk in enumerate(speech_chunks):
        buf = io.BytesIO()
        chunk.export(buf, format="wav")
        buf.seek(0)

        with sr.AudioFile(buf) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                transcript_parts.append(text)
                logger.info(f"  Chunk {i+1}/{len(speech_chunks)}: {text[:60]}...")
            except sr.UnknownValueError:
                logger.debug(f"  Chunk {i+1}: (unintelligible, skipped)")

    return " ".join(transcript_parts).strip()


# ─── LLM Analysis ────────────────────────────────────────────────────────────

QA_PROMPT_TEMPLATE = (
    "As a highly critical Quality Assurance Auditor, evaluate the following transcribed dialogue "
    "with an extreme focus on agent logic and conversational efficiency. "
    "You must maintain a high barrier for excellence; the green heart (💚) is reserved only for "
    "flawless, proactive interactions where the agent anticipates needs. "
    "Identify 'dumb' behavior such as non-sequiturs, repetitive loops, or failing to acknowledge "
    "specific client data. "
    "Select exactly one primary grade icon based on these strict definitions:\n"
    "💚 -> Exceptional; the agent was perfectly natural, proactive, and achieved the goal without "
    "any friction or mechanical feel.\n"
    "💛 -> Standard/Functional; the agent followed the script and the client was served, but the "
    "interaction felt robotic or lacked polish.\n"
    "🧡 -> Technical Failure; the agent entered a repetitive loop, repeated a sentence, or showed "
    "a bug (e.g., 'I didn't catch that' multiple times), even if the client stayed.\n"
    "❤️ -> Critical Failure; the client became frustrated by the agent's incompetence, stupidity, "
    "or looping and terminated the call.\n"
    "😡 -> Total Collapse; the agent's responses were nonsensical, hallucinated information, or "
    "completely failed to comprehend basic human speech.\n"
    "Additionally, append the icon ↗️ immediately after your primary grade if the agent successfully "
    "executed a hand-off or stated they were transferring the call to a human without technical errors.\n"
    "Your response must begin with the appropriate grade icon (and the ↗️ icon if applicable), "
    "followed by a concise justification that explicitly identifies any loops or lapses in agent intelligence.\n\n"
    "Dialogue: {transcript}"
)


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


# ─── Slack File Handling ─────────────────────────────────────────────────────

def download_slack_file(url: str, dest_path: str):
    """Download a file from Slack using bot token for auth."""
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    resp = requests.get(url, headers=headers, timeout=120)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)
    logger.info(f"Downloaded file to {dest_path} ({len(resp.content)} bytes)")


# ─── Event Handler ───────────────────────────────────────────────────────────

@app.event("message")
def handle_message(event, say, client):
    """Process any message that contains a .wav file upload."""
    channel = event.get("channel", "")
    ts = event.get("ts", "")
    files = event.get("files", [])

    # Optional: only process messages in the designated channel
    if QA_CHANNEL_ID and channel != QA_CHANNEL_ID:
        return

    # Filter for .wav files
    wav_files = [
        f for f in files
        if f.get("name", "").lower().endswith(".wav")
        or f.get("mimetype", "") == "audio/wav"
    ]

    if not wav_files:
        return

    for file_info in wav_files:
        filename = file_info.get("name", "unknown.wav")
        download_url = file_info.get("url_private_download") or file_info.get("url_private")

        if not download_url:
            logger.warning(f"No download URL for file: {filename}")
            continue

        logger.info(f"📥 Processing: {filename} in channel {channel}")

        # Acknowledge with a ⏳ reaction
        try:
            client.reactions_add(channel=channel, name="hourglass_flowing_sand", timestamp=ts)
        except Exception:
            pass

        try:
            # 1. Download
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            download_slack_file(download_url, tmp_path)

            # 2. Transcribe
            logger.info("🎙️ Transcribing audio...")
            transcript = transcribe_wav(tmp_path)

            if not transcript:
                say(
                    text=f"⚠️ Could not transcribe any speech from `{filename}`.",
                    thread_ts=ts,
                )
                continue

            logger.info(f"📝 Transcript ({len(transcript)} chars): {transcript[:100]}...")

            # 3. Analyze
            logger.info("🤖 Running QA analysis...")
            analysis = analyze_transcript(transcript)
            logger.info(f"📊 Analysis complete.")

            # 4. React with grade emoji(s)
            grade_emojis = extract_grade_emojis(analysis)
            for emoji_name in grade_emojis:
                try:
                    client.reactions_add(channel=channel, name=emoji_name, timestamp=ts)
                    logger.info(f"  ✅ Reacted: :{emoji_name}:")
                except Exception as e:
                    logger.warning(f"  ⚠️ Reaction failed ({emoji_name}): {e}")

            # 5. Post analysis as a threaded reply
            reply_text = (
                f"*QA Analysis for* `{filename}`\n\n"
                f"*Transcript:*\n>>> {transcript[:1500]}{'...' if len(transcript) > 1500 else ''}\n\n"
                f"*Assessment:*\n{analysis}"
            )
            say(text=reply_text, thread_ts=ts)

        except Exception as e:
            logger.exception(f"Error processing {filename}: {e}")
            say(
                text=f"❌ Error analyzing `{filename}`: {e}",
                thread_ts=ts,
            )
        finally:
            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            # Remove hourglass reaction
            try:
                client.reactions_remove(channel=channel, name="hourglass_flowing_sand", timestamp=ts)
            except Exception:
                pass


# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("🚀 QA Bot starting in Socket Mode (always online)...")
    logger.info(f"   Model: {OLLAMA_MODEL}")
    logger.info(f"   Channel filter: {QA_CHANNEL_ID or 'ALL channels'}")
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()  # Blocks forever — stays online