# 🎯 PlaceCoach – AI Interview Coach & Resume Analyzer

> **Intelligent interview preparation powered by [Endee](https://github.com/endee-io/endee) Vector Database + Groq LLaMA3**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Endee](https://img.shields.io/badge/VectorDB-Endee-6c3fc4?style=flat-square)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA3--70B-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square)
![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)

---


## 📸 Screenshots

### Home & Setup
![PlaceCoach Home](screenshots/home.png)

### CV Report Card
![CV Report Card](screenshots/cv_report.png)

### Interview Session
![Interview Chat](screenshots/interview.png)

### Final Report
![Interview Report](screenshots/report.png)

## 📌 Problem Statement

Students and job seekers often struggle with interview preparation — generic question lists don't account for their specific skills, experience level, or target role. PlaceCoach solves this by:

1. **Parsing the candidate's resume** to extract skills, experience, and domains
2. **Semantically retrieving** the most relevant interview questions from Endee vector DB
3. **Evaluating answers** using Groq LLaMA3-70B with scores, feedback, and improvement tips
4. **Generating a personalised report** with overall recommendations

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📄 **Resume Analysis** | Extracts name, skills, domains, experience using LLM |
| 🔍 **Semantic Question Retrieval** | Finds most relevant questions via Endee vector search |
| 🤖 **AI Answer Evaluation** | Scores answers 1–10 with detailed feedback |
| 📊 **Interview Report** | Full breakdown with per-question feedback + overall advice |
| 🎯 **15+ Job Roles** | SWE, DS, PM, Marketing, Finance, HR, DevOps, and more |
| 💬 **150+ Questions** | Curated question bank covering technical & behavioural areas |

---

## 🏗️ System Design
```
┌──────────────────────────────────────────────────────────────┐
│                     USER (Streamlit UI)                      │
└────────────────────────┬─────────────────────────────────────┘
                         │  Resume PDF + Target Role
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    RESUME PARSER                             │
│   pdfplumber extracts text → Groq LLM structures it into    │
│   {name, skills, domains, experience, education}            │
└────────────────────────┬─────────────────────────────────────┘
                         │  Structured resume data
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                RAG QUESTION RETRIEVAL                        │
│  1. Build semantic query: role + skills + domains           │
│  2. Embed query (MiniLM-L6-v2, 384-dim)                    │
│  3. Search Endee vector DB → top-20 similar questions       │
│  4. LLM selects best N questions for this candidate         │
└────────────────────────┬─────────────────────────────────────┘
                         │  Personalised question list
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              INTERACTIVE INTERVIEW SESSION                   │
│   Candidate answers each question → submitted to evaluator  │
└────────────────────────┬─────────────────────────────────────┘
                         │  Q&A pairs
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                   ANSWER EVALUATOR                           │
│   Groq LLaMA3-70B scores each answer (1-10) and provides:  │
│   → Positive feedback → Improvement tips → Missing keywords │
└────────────────────────┬─────────────────────────────────────┘
                         │  Evaluation results
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                   INTERVIEW REPORT                           │
│   Overall score, per-question breakdown, recommendations    │
└──────────────────────────────────────────────────────────────┘
```

---

## 🗄️ How Endee is Used

Endee is the **semantic search backbone** of PlaceCoach.

### Indexing the Question Bank
```python
from endee import Endee, Precision

client = Endee()
client.set_base_url("http://localhost:8080/api/v1")

client.create_index(
    name="interview_questions",
    dimension=384,
    space_type="cosine",
    precision=Precision.INT8
)

index = client.get_index(name="interview_questions")
index.upsert([
    {
        "id": "se_001",
        "vector": model.encode("Explain SOLID principles").tolist(),
        "meta": {
            "text": "Explain SOLID principles...",
            "role": "Software Engineer",
            "category": "OOP"
        }
    }
])
```

### Semantic Retrieval
```python
query = "Software Engineer interview. Skills: Python, APIs. Domains: Backend."
query_vector = model.encode(query).tolist()
results = index.query(vector=query_vector, top_k=20)
```

---

## 🛠️ Setup & Execution

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- [Free Groq API Key](https://console.groq.com)

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/PlaceCoach
cd PlaceCoach
```

### Step 2: Start Endee Vector Database
```bash
docker compose up -d
```

### Step 3: Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### Step 5: Launch the App
```bash
streamlit run app.py
```
Visit [http://localhost:8501](http://localhost:8501)

### Step 6: Use It
1. Enter your **Groq API Key** in the sidebar
2. Upload your **resume PDF**
3. Select your **target job role**
4. Choose number of questions
5. Click **Start Interview Session**
6. Answer each question and get instant AI feedback
7. Review your full **Interview Report**

---

## 📂 Project Structure
```
PlaceCoach/
├── app.py                  # Streamlit UI & session management
├── rag_engine.py           # RAG retrieval + answer evaluation
├── question_bank.py        # 150+ questions + Endee indexing
├── resume_parser.py        # PDF parsing + LLM structuring
├── requirements.txt
├── docker-compose.yml      # Endee vector DB
├── .env.example
└── README.md
```

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| Vector Database | **Endee** (HNSW, cosine similarity, INT8) |
| Embedding Model | **sentence-transformers/all-MiniLM-L6-v2** (384-dim) |
| LLM | **Groq LLaMA3-70B** (free, ultra-fast) |
| UI | **Streamlit** |
| PDF Parsing | **pdfplumber** |

---

## 📄 License

Apache 2.0 License
```

- Press **Cmd+S** to save

---

All 8 files are now done! 🎉 Your folder should look like this:
```
placecoach/
├── app.py
├── rag_engine.py
├── question_bank.py
├── resume_parser.py
├── requirements.txt
├── docker-compose.yml
├── .env.example
└── README.md