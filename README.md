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
