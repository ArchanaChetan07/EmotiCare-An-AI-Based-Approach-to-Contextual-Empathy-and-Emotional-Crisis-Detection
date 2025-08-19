import json
import os
import pandas as pd
from datetime import datetime
from src.langgraphagenticai.LLMS.groqllm import GroqLLM

JOURNAL_FILE = "journal_entries.json"

def save_journal_entry(text, emotion_dict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp": timestamp,
        "text": text,
        **emotion_dict  # Each emotion becomes a column
    }

    if os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(JOURNAL_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_journal_data():
    if not os.path.exists(JOURNAL_FILE):
        return pd.DataFrame()
    with open(JOURNAL_FILE, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def detect_emotions_llm(text, llm_model=None):
    """
    Uses your LLM (Groq or similar) to detect emotions in a journal entry.
    Returns: a dictionary like {"joy": 0.7, "sadness": 0.2, ...}
    """
    if not llm_model:
        obj_llm_config = GroqLLM(user_controls_input={"selected_model": "mixtral-8x7b"})
        llm_model = obj_llm_config.get_llm_model()

    prompt = (
        "Read the journal entry and respond in JSON format. "
        "Return the estimated intensity (0.0 to 1.0) for these emotions: joy, sadness, anger, fear.\n\n"
        f"Journal Entry: \"{text}\"\n\n"
        "Respond like this: {\"joy\": float, \"sadness\": float, \"anger\": float, \"fear\": float}"
    )

    try:
        response = llm_model.invoke(prompt)
        return eval(response.content) if hasattr(response, "content") else eval(response)
    except Exception as e:
        print(f"[Emotion detection failed] {e}")
        return {"joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0}
