# EmotiCare-An-AI-Based-Approach-to-Contextual-Empathy-and-Emotional-Crisis-Detection

##  Overview
**EmotiCare** is an AI-powered chatbot designed to detect emotional cues in text and respond with **contextual empathy** in real time.  
Unlike conventional mental health bots that rely on static scripts, EmotiCare leverages **multi-label emotion classification**, **transformer-based NLP**, and **graph-driven chatbot architectures** to deliver scalable, empathetic, and ethical emotional support.

The system integrates:
- **GoEmotions**, **CounselChat**, and **EmpatheticDialogues** datasets for diverse emotional understanding.
- **Logistic Regression**, **BERT embeddings**, and **XGBoost** for emotion detection.
- **LangChain LangGraph** for dynamic, tool-augmented conversations.
- **Streamlit** frontend for interactive journaling and chatbot engagement.

---

##  Key Features
- **Multi-label Emotion Classification**  
  Detects multiple emotions in a single message using optimized NLP pipelines.
- **Context-Aware Empathy**  
  Adapts responses to the user's emotional context.
- **Crisis Detection & Escalation**  
  Identifies high-risk emotional language and flags for professional support.
- **Modular Chatbot Architecture**  
  LangGraph-based state management with real-time web search augmentation.
- **User-Friendly Interface**  
  Built with Streamlit for interactive conversation and journaling.

---

##  Tech Stack
- **Languages:** Python 3.x
- **Frameworks & Libraries:**  
  - NLP: Hugging Face Transformers, Sentence-Transformers, SpaCy, NLTK, SymSpell  
  - ML: Scikit-learn, XGBoost, PyTorch  
  - Visualization: Matplotlib, Seaborn, Plotly  
  - Chatbot: LangChain, LangGraph  
  - Web UI: Streamlit  
- **APIs:** Groq API (LLM inference), Tavily Search API
- **Version Control:** Git + GitHub

---

##  Datasets
1. **[GoEmotions](https://arxiv.org/abs/2005.00547)** – 58K Reddit comments labeled with 27 emotions + neutral.
2. **CounselChat** – Therapist responses to real-world mental health questions.
3. **[EmpatheticDialogues](https://arxiv.org/abs/1811.00207)** – Short conversations with explicit empathy annotation.

All datasets undergo:
- Deduplication & noise removal
- Lemmatization & spelling normalization
- Emoji/emoticon conversion to descriptive tokens
- Unified 28-label schema mapping

---

##  Project Structure
```

Capstone project/
│
├── capstone/                  # Capstone report and related documents
├── Chatbot\_with\_Web/          # Main chatbot implementation
│   ├── Data\_final/             # Final cleaned datasets
│   ├── src/                    # Source code
│   │   ├── langgraphagenticai/ # LangGraph-based chatbot logic
│   │   │   ├── graph/          # Conversation state graph
│   │   │   ├── LLMS/           # LLM configuration files
│   │   │   ├── nodes/          # Chatbot node functions
│   │   │   ├── state/          # State management
│   │   │   ├── tools/          # Tool integrations (e.g., web search)
│   │   │   ├── ui/             # UI components
│   │   │   ├── main.py         # Entry point for chatbot
│   │   │   └── **init**.py
│   ├── app.py                  # Streamlit frontend app
│   ├── journal\_entries.json    # User journaling data
│   ├── requirements.txt        # Python dependencies
│   └── README.md               # Module-specific documentation
│
├── Data/                       # Raw and intermediate datasets
├── notebook/                   # Jupyter notebooks for EDA & modeling
├── .env                        # Environment variables
├── requirements.txt            # Project-level dependencies

````

---

##  Modeling Approach
| Model | Macro F1 | Micro F1 | Hamming Loss | Notes |
|-------|----------|----------|--------------|-------|
| Logistic Regression (Weighted, threshold=0.65) | **0.3182** | **0.3532** | **0.0541** | Best balance of accuracy & interpretability |
| BERT Embeddings + Logistic Regression | 0.3071 | 0.3410 | 0.0562 | Good for nuanced emotions |
| DistilBERT Pipeline (Top-2) | 0.2984 | 0.3365 | 0.0578 | Strong for complex emotions like fear/sadness |
| BERT Embeddings + XGBoost | 0.3010 | 0.3398 | 0.0567 | Best for anger detection |

---

### 3. Run the Streamlit App

```bash
streamlit run app.py

Access in Browser

Chatbot UI: http://localhost:8501

Prometheus Metrics: http://localhost:8000

---

## 📈 Results Summary

* **Top model:** Weighted Logistic Regression with threshold tuning.
* **Strengths:** High recall for distress-related emotions (anger, fear, grief).
* **Limitations:** Class imbalance impacts rare emotion detection; deep model fine-tuning limited by compute resources.

---

## 🔮 Future Work

* Fine-tune **DistilBERT/RoBERTa** on the integrated dataset for better contextual sensitivity.
* Implement **ensemble architecture** combining Logistic Regression, transformer models, and XGBoost.
* Partner with mental health organizations for **real-world validation**.
* Expand to **multimodal emotion detection** (voice + text).

---

## 👥 Authors

* **Jason Tong** – Applied Data Science, University of San Diego
* **Archana Suresh Patil** – Applied Data Science, University of San Diego
* **Sahil Wadhwa** – Applied Data Science, University of San Diego

---
