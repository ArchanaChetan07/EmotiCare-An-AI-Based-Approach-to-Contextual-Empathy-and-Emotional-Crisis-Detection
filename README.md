# EmotiCare – An AI-Based Approach to Contextual Empathy and Emotional Crisis Detection  

## 📌 Overview  
**EmotiCare** is an AI-powered chatbot that detects emotional cues in text and responds with **contextual empathy**.  
Unlike conventional mental health bots that rely on static scripts, EmotiCare integrates **multi-label emotion classification**, **transformer-based NLP**, and **graph-driven chatbot architectures** to deliver scalable, empathetic, and ethical emotional support.  

The system leverages:  
- **Datasets:** GoEmotions, CounselChat, EmpatheticDialogues  
- **Models:** Logistic Regression, BERT embeddings, XGBoost, DistilBERT  
- **Frameworks:** LangChain + LangGraph for dynamic, tool-augmented conversations  
- **Deployment:** Streamlit frontend for journaling and chatbot interaction  

---

## 🌟 Key Features  
- **Multi-Label Emotion Classification** – Detects multiple emotions in a single input.  
- **Context-Aware Empathy** – Adapts chatbot responses to nuanced emotional states.  
- **Crisis Detection & Escalation** – Flags high-risk emotional cues for professional support.  
- **Modular Chatbot Architecture** – LangGraph-based, extensible, and fail-safe.  
- **Interactive UI** – Streamlit-powered journaling and chatbot interface.  

---

## 🛠 Tech Stack  
- **Languages:** Python 3.10+  
- **Frameworks & Libraries:**  
  - NLP: Hugging Face Transformers, Sentence-Transformers, SpaCy, NLTK, SymSpell  
  - ML: Scikit-learn, XGBoost, PyTorch  
  - Visualization: Matplotlib, Seaborn, Plotly  
  - Chatbot: LangChain, LangGraph  
  - Web UI: Streamlit  
- **APIs:** Groq (LLM inference), Tavily Search API  
- **Version Control:** Git & GitHub  

---

## 📂 Project Structure  
```bash
Capstone_Project/
│
├── capstone/                     # Capstone report and related docs
├── Chatbot_with_Web/             # Main chatbot implementation
│   ├── Data_final/                # Final cleaned datasets
│   ├── src/                      # Source code
│   │   ├── langgraphagenticai/   # LangGraph-based chatbot logic
│   │   │   ├── graph/            # Conversation state graph
│   │   │   ├── LLMS/             # LLM configuration
│   │   │   ├── nodes/            # Chatbot node functions
│   │   │   ├── state/            # State management
│   │   │   ├── tools/            # Integrated tools (search, classifier, etc.)
│   │   │   ├── ui/               # UI components
│   │   │   ├── main.py           # Entry point
│   │   │   └── __init__.py
│   ├── app.py                     # Streamlit frontend
│   ├── journal_entries.json       # User journaling data
│   └── requirements.txt           # Dependencies
│
├── Data/                         # Raw and intermediate datasets
├── notebook/                     # Jupyter notebooks for EDA & modeling
│   ├── 01_Data Cleaning_Raw_to_Silver.ipynb
│   ├── 02_Data Cleaning_Silver_to_Gold.ipynb
│   ├── 03_EDA_gold_Dataset.ipynb
│   └── 03_Preprocessing_modeling.ipynb
│
├── .env                          # API keys & environment variables
├── requirements.txt              # Project-level dependencies
└── README.md
📊 Modeling Approach
Model	Macro F1	Micro F1	Hamming Loss	Notes
Logistic Regression (Weighted, thr=0.65)	0.3182	0.3532	0.0541	Best overall; interpretable & efficient
BERT Embeddings + Logistic Regression	0.3071	0.3410	0.0562	Better for nuanced emotions
DistilBERT Pipeline (Top-2)	0.2984	0.3365	0.0578	Strong for fear & sadness
BERT Embeddings + XGBoost	0.3010	0.3398	0.0567	Strong for anger detection

🔑 Insight: Simpler models (weighted Logistic Regression) can rival transformer-based models in safety-critical, multi-label emotion tasks.

⚙️ Setup & Installation
Prerequisites
Python 3.10+

Virtual environment recommended

Installation
bash
Copy
Edit
# Clone repo
git clone https://github.com/ArchanaChetan07/EmotiCare-An-AI-Based-Approach-to-Contextual-Empathy-and-Emotional-Crisis-Detection.git
cd EmotiCare-An-AI-Based-Approach-to-Contextual-Empathy-and-Emotional-Crisis-Detection

# Install dependencies
pip install -r requirements.txt
▶️ Running the App
Start Chatbot (Streamlit)
bash
Copy
Edit
cd Chatbot_with_Web
streamlit run app.py
Access in Browser
Chatbot UI → http://localhost:8501

Prometheus Metrics → http://localhost:8000

📈 Results Summary
Top Model: Weighted Logistic Regression w/ threshold tuning (0.65).

Strengths: High recall for distress emotions (anger, fear, grief).

Limitations: Class imbalance + compute constraints limited deep model fine-tuning.

🔮 Future Work
Fine-tune DistilBERT / RoBERTa on integrated dataset.

Implement ensemble modeling (LogReg + Transformers + XGBoost).

Collaborate with mental health organizations for real-world deployment.

Expand to multimodal emotion detection (text + voice).

👥 Authors
Jason Tong – Applied Data Science, University of San Diego

Archana Suresh Patil – Applied Data Science, University of San Diego

Sahil Wadhwa – Applied Data Science, University of San Diego

✨ EmotiCare demonstrates how psychology, ethical AI, and language science can combine to create emotionally intelligent and responsible digital companions.
---
