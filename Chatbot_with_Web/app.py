import sys
import os

# ✅ Dynamically add the absolute path to 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from langgraphagenticai.main import load_langgraph_agenticai_app

if __name__ == "__main__":
    load_langgraph_agenticai_app()