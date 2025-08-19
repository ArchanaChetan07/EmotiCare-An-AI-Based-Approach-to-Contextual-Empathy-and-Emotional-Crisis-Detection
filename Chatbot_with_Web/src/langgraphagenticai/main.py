import streamlit as st
import pandas as pd
import os
import sys
from prometheus_client import start_http_server, Counter
import threading

# Add 'src' directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# LangGraph Chatbot Components
from src.langgraphagenticai.ui.streamlitui.loadui import LoadStreamlitUI
from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.ui.streamlitui.display_result import DisplayResultStreamlit

# Journal + Emotion Analysis Utils
from src.langgraphagenticai.ui.streamlitui.journal_utils import (
    detect_emotions_llm,
    save_journal_entry,
    load_journal_data
)

# Prometheus Metrics
chat_input_counter = Counter("chat_input_count", "Number of user chat inputs")
journal_entry_counter = Counter("journal_entry_count", "Number of journal entries saved")

def load_langgraph_agenticai_app():
    """
    Loads and runs the LangGraph AgenticAI application with Streamlit UI.
    Includes: chatbot UI, LLM graph execution, emotion journaling and visualization.
    """

    # Step 1: Load Sidebar UI
    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("❌ Error: Failed to load user input from the UI.")
        return

    # Step 2: Load LLM
    obj_llm_config = GroqLLM(user_controls_input=user_input)
    model = obj_llm_config.get_llm_model()

    if not model:
        st.error("❌ Error: LLM model could not be initialized.")
        return

    # Step 3: Use Case Selection
    usecase = user_input.get("selected_usecase")
    if not usecase:
        st.error("❌ Error: No use case selected.")
        return

    # Step 4: Chat Input
    user_message = st.chat_input("💬 Enter your message:")

    if user_message:
        chat_input_counter.inc()  # Track chat interaction
        try:
            graph_builder = GraphBuilder(model)
            graph = graph_builder.setup_graph(usecase)
            DisplayResultStreamlit(usecase, graph, user_message).display_result_on_ui()
        except Exception as e:
            st.error(f"❌ Error running chatbot: {e}")

    # ======================
    # Step 5: Journal Block
    # ======================
    st.markdown("### 📝 Journal Your Emotions")
    journal_text = st.text_area("Write your thoughts here:")

    if st.button("Save Entry") and journal_text.strip():
        emotions = detect_emotions_llm(journal_text, llm_model=model)
        save_journal_entry(journal_text, emotions)
        journal_entry_counter.inc()  # Track journal entries
        st.success("✅ Journal entry saved!")

    # Step 6: Visualize Emotion Trends
    st.markdown("### 📈 Emotion Trends Over Time")
    df = load_journal_data()

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        st.line_chart(df.set_index("timestamp")[["joy", "sadness"]])
    else:
        st.info("No emotion data yet. Start journaling!")


# =========================
# Background metrics server
# =========================
def start_prometheus_metrics_server():
    start_http_server(8000)  # Exposes metrics on http://localhost:8000


# Main app entry
if __name__ == "__main__":
    threading.Thread(target=start_prometheus_metrics_server, daemon=True).start()
    load_langgraph_agenticai_app()
