# AI Voice Agent Evaluator

This project is an automated audio analysis tool designed to evaluate customer service interactions between clients and AI voice agents. By analyzing .wav files dropped directly into your Downloads folder, the script slices out dead air, transcribes the spoken dialogue, and uses a locally hosted Hugging Face Llama 3.2 model to grade the interaction's success.

The primary goal is to provide an instant, highly visual assessment of call quality and routing efficiency without relying on expensive, cloud-based LLM APIs.

---
## Core Features

The script relies on pydub to intelligently trim periods of silence exceeding 500 milliseconds. This prevents the speech-to-text engine from processing dead air and avoids artificial cutoffs in the middle of a spoken word.

Once the active audio segments are isolated, they are held in system memory and transcribed sequentially using Google's Web Speech API.

The complete, assembled transcript is then fed to an offline Llama 3.2 model running via Ollama. The model analyzes the flow, tone, and outcome of the conversation to generate a concise review and a visual grade.

---
## The Grading Rubric

The AI evaluator strictly adheres to a highly visual emoji grading system, allowing you to assess the outcome of a call at a single glance.

| Icon | Meaning |
|------|---------|
| 💚 | **Great**: The interaction was smooth and highly successful. |
| 💛 | **Fine**: The call was acceptable without major issues. |
| 🧡 | **Bugged**: The voice agent looped or caused clear client discomfort. |
| ❤️ | **Abandoned**: The call went poorly, resulting in the client leaving or hanging up. |
| 😡 | **Terrible**: An absolute failure in the agent's logic or behavior. |
| ↗️ | **Transferred**: Added alongside the main grade if the agent successfully routed the client to a live human without errors. |

---
## Environment Setup

To get this script running on your local machine, you will need to configure a few dependencies. First, ensure you have Python installed. You must then install the necessary Python libraries by running the following command in your terminal:

```bash
pip install SpeechRecognition pydub ollama
```
---
## Extra for linking it to slack

Always-online Slack bot that automatically analyzes `.wav` call recordings posted in a channel.

## How It Works

```
Someone posts .wav → ⏳ hourglass appears → Bot transcribes → Ollama analyzes → 
💚/💛/🧡/❤️/😡 reaction + threaded reply with full analysis
```

## Setup (10 minutes)

### 1. Create Slack App

1. Go to **https://api.slack.com/apps** → **Create New App** → **From Scratch**
2. Give it a name like "QA Analyzer Bot"

### 2. Enable Socket Mode

- **Settings → Socket Mode → Enable Socket Mode**
- Create an **App-Level Token** with scope `connections:write`
- Copy it — this is your `SLACK_APP_TOKEN` (`xapp-...`)

### 3. Set Bot Permissions

Go to **OAuth & Permissions → Bot Token Scopes**, add:

| Scope | Purpose |
|---|---|
| `channels:history` | Read messages in public channels |
| `groups:history` | Read messages in private channels |
| `channels:read` | List channels |
| `files:read` | Download uploaded .wav files |
| `reactions:write` | Add grade emoji reactions |
| `chat:write` | Post threaded analysis replies |

### 4. Subscribe to Events

Go to **Event Subscriptions → Enable Events → Subscribe to bot events**, add:

- `message.channels`
- `message.groups` *(for private channels)*

### 5. Install to Workspace

- **Install App** → Copy **Bot User OAuth Token** (`xoxb-...`)
- This is your `SLACK_BOT_TOKEN`

### 6. Invite Bot to Channel

In Slack, go to your target channel and type:
```
/invite @YourBotName
```

## Running the Bot

### Install dependencies
```bash
pip install -r requirements.txt
```

### Set environment variables
```bash
export SLACK_BOT_TOKEN="xoxb-your-token-here"
export SLACK_APP_TOKEN="xapp-your-app-level-token-here"

# Optional: restrict to one channel (right-click channel → View details → copy ID)
export QA_CHANNEL_ID="C0123ABCDEF"

# Optional: change Ollama model
export OLLAMA_MODEL="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q8_0"
```

### Start the bot
```bash
python bot.py
```

The bot will stay online until you stop it (`Ctrl+C`).

### Run as a background service (always on)

#### Option A: systemd (Linux)
```bash
sudo tee /etc/systemd/system/qa-bot.service << EOF
[Unit]
Description=Slack QA Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=SLACK_BOT_TOKEN=xoxb-...
Environment=SLACK_APP_TOKEN=xapp-...
Environment=QA_CHANNEL_ID=C0123ABCDEF
ExecStart=$(which python3) bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable qa-bot
sudo systemctl start qa-bot
```

#### Option B: Docker
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY bot.py .
CMD ["python", "bot.py"]
```

#### Option C: tmux/screen (quick & dirty)
```bash
tmux new -s qabot
python bot.py
# Ctrl+B, D to detach — bot keeps running
```

## Grade Reactions

| Emoji | Meaning |
|---|---|
| 💚 `green_heart` | Exceptional — flawless, proactive agent |
| 💛 `yellow_heart` | Standard — functional but robotic |
| 🧡 `orange_heart` | Technical Failure — loops, bugs |
| ❤️ `heart` | Critical Failure — client frustrated |
| 😡 `rage` | Total Collapse — nonsensical |
| ↗️ `arrow_upper_right` | Successful hand-off to human |

## Prerequisites

- **Python 3.9+**
- **ffmpeg** installed (`sudo apt install ffmpeg` / `brew install ffmpeg`)
- **Ollama** running locally with your model pulled
- Internet access (for Google Speech Recognition API)
