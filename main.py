import os
import io
import speech_recognition as sr
import ollama
from pydub import AudioSegment
from pydub.silence import split_on_silence

default_path = os.path.expanduser("~/Downloads")

def analyze_dialogue(filename, model_name='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q8_0'):
    wav_file_path = os.path.join(default_path, filename)
    recognizer = sr.Recognizer()
    
    print(f"Loading and analyzing audio file: {wav_file_path}")
    
    try:
        print("Detecting and removing silence...")
        audio = AudioSegment.from_wav(wav_file_path)
        
        speech_chunks = split_on_silence(
            audio,
            min_silence_len=500,
            silence_thresh=audio.dBFS - 14,
            keep_silence=250
        )
        
        full_transcript = ""
        print(f"Audio split into active speech segments. Transcribing...")
        
        for chunk in speech_chunks:
            chunk_io = io.BytesIO()
            chunk.export(chunk_io, format="wav")
            chunk_io.seek(0)
            
            with sr.AudioFile(chunk_io) as source:
                audio_data = recognizer.record(source)
                try:
                    chunk_text = recognizer.recognize_google(audio_data)
                    full_transcript += chunk_text + " "
                    print(f"Transcribed segment: {chunk_text[:50]}...")
                except sr.UnknownValueError:
                    pass
                    
        full_transcript = full_transcript.strip()
        
        if not full_transcript:
            print("Error: No speech could be transcribed from the file.")
            return

        print("\n--- Complete Transcript ---")
        print(full_transcript)
        print("---------------------------\n")
        
        prompt = (
            "As a highly critical Quality Assurance Auditor, evaluate the following transcribed dialogue with an extreme focus on agent logic and conversational efficiency. "
            "You must maintain a high barrier for excellence; the green heart (💚) is reserved only for flawless, proactive interactions where the agent anticipates needs. "
            "Identify 'dumb' behavior such as non-sequiturs, repetitive loops, or failing to acknowledge specific client data. "
            "Select exactly one primary grade icon based on these strict definitions:\n"
            "💚 -> Exceptional; the agent was perfectly natural, proactive, and achieved the goal without any friction or mechanical feel.\n"
            "💛 -> Standard/Functional; the agent followed the script and the client was served, but the interaction felt robotic or lacked polish.\n"
            "🧡 -> Technical Failure; the agent entered a repetitive loop, repeated a sentence, or showed a bug (e.g., 'I didn't catch that' multiple times), even if the client stayed.\n"
            "❤️ -> Critical Failure; the client became frustrated by the agent's incompetence, stupidity, or looping and terminated the call.\n"
            "😡 -> Total Collapse; the agent's responses were nonsensical, hallucinated information, or completely failed to comprehend basic human speech.\n"
            "Additionally, append the icon ↗️ immediately after your primary grade if the agent successfully "
            "executed a hand-off or stated they were transferring the call to a human without technical errors.\n"
            "Your response must begin with the appropriate grade icon (and the ↗️ icon if applicable), "
            "followed by a concise justification that explicitly identifies any loops or lapses in agent intelligence.\n\n"
            f"Dialogue: {full_transcript}"
        )
        
        print(f"Analyzing dialogue using Ollama ({model_name})...")
        response = ollama.generate(model=model_name, prompt=prompt)
        
        print("\n--- Dialogue Review ---")
        print(response['response'])
        print("-----------------------\n")
        
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found in {default_path}.")
    except ImportError:
        print("Error: The 'pydub' library is required. Please run: pip install pydub")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
analyze_dialogue("C-3PO Topserv Recording.wav")