import streamlit as st
import os
from resume_parser import ResumeParser
from question_bank import QuestionBank
from rag_engine import RAGEngine

st.set_page_config(
    page_title="PlaceCoach – AI Interview Coach",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: 900; color: #1a1a2e; }
    .subtitle   { font-size: 1rem; color: #555; margin-bottom: 1.5rem; }
    .score-box  {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border-radius: 16px; padding: 1.5rem;
        text-align: center; font-size: 2.5rem; font-weight: 800;
    }
    .score-label { font-size: 0.9rem; font-weight: 400; opacity: 0.85; }
    .question-box {
        background: #f8f9ff; border-left: 5px solid #667eea;
        padding: 1rem 1.2rem; border-radius: 8px;
        font-size: 1.05rem; font-weight: 500; color: #1a1a2e;
    }
    .feedback-box {
        background: #f0fff4; border-left: 5px solid #38a169;
        padding: 1rem 1.2rem; border-radius: 8px; color: #1a1a2e;
    }
    .improvement-box {
        background: #fffbeb; border-left: 5px solid #d69e2e;
        padding: 1rem 1.2rem; border-radius: 8px; color: #1a1a2e;
    }
    .skill-tag {
        display: inline-block; background: #e9ecff; color: #4361ee;
        border-radius: 20px; padding: 3px 12px; font-size: 0.8rem;
        font-weight: 600; margin: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Session State
defaults = {
    "resume_data": None,
    "questions": [],
    "current_q": 0,
    "answers": [],
    "evaluations": [],
    "session_complete": False,
    "bank_ready": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "parser"  not in st.session_state:
    st.session_state.parser  = ResumeParser()
if "bank"    not in st.session_state:
    st.session_state.bank    = QuestionBank()
if "engine"  not in st.session_state:
    st.session_state.engine  = RAGEngine()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    groq_key = st.text_input("Groq API Key", type="password",
                              value=os.getenv("GROQ_API_KEY", ""),
                              help="Free at console.groq.com")
    endee_url = st.text_input("Endee URL",
                               value=os.getenv("ENDEE_URL", "http://localhost:8080"))

    st.markdown("---")
    st.markdown("### 📄 Upload Resume")
    resume_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

    st.markdown("### 🎯 Select Job Role")
    role = st.selectbox("Target Role", [
        "Software Engineer",
        "Data Scientist",
        "Product Manager",
        "Marketing Manager",
        "Business Analyst",
        "Frontend Developer",
        "Backend Developer",
        "ML Engineer",
        "DevOps Engineer",
        "UI/UX Designer",
        "Finance Analyst",
        "HR Manager",
        "Sales Executive",
        "Cybersecurity Analyst",
        "Cloud Architect",
    ])

    num_questions = st.slider("Number of Questions", 3, 10, 5)

    st.markdown("---")
    if st.button("🚀 Start Interview Session", use_container_width=True):
        if not groq_key:
            st.error("Please enter your Groq API key.")
        elif not resume_file:
            st.error("Please upload your resume.")
        else:
            st.session_state.engine.set_config(groq_key, endee_url)

            with st.spinner("📖 Parsing resume..."):
                st.session_state.resume_data = st.session_state.parser.parse(
                    resume_file, groq_key)

            if not st.session_state.bank_ready:
                with st.spinner("🗄️ Loading question bank into Endee..."):
                    st.session_state.bank.set_config(endee_url)
                    st.session_state.bank.index_all()
                    st.session_state.bank_ready = True

            with st.spinner("🔍 Retrieving tailored questions..."):
                st.session_state.questions = st.session_state.engine.get_questions(
                    role=role,
                    resume_data=st.session_state.resume_data,
                    n=num_questions,
                )

            st.session_state.current_q   = 0
            st.session_state.answers     = []
            st.session_state.evaluations = []
            st.session_state.session_complete = False
            st.rerun()

    if st.button("🔄 Reset Session", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

# Main Area
st.markdown('<div class="main-title">🎯 PlaceCoach</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Personalised AI interview preparation powered by Endee Vector DB + Groq LLM</div>',
            unsafe_allow_html=True)

if not st.session_state.questions:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\n\nEnter your Groq API key in the sidebar")
    with col2:
        st.info("**Step 2**\n\nUpload your resume PDF & select target role")
    with col3:
        st.info("**Step 3**\n\nClick 'Start Interview Session' and answer questions!")
    st.stop()

if st.session_state.resume_data:
    rd = st.session_state.resume_data
    with st.expander("📋 Resume Analysis", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**👤 Name:** {rd.get('name','—')}")
            st.markdown(f"**🎓 Education:** {rd.get('education','—')}")
            st.markdown(f"**💼 Experience:** {rd.get('experience_years','—')} years")
        with col2:
            st.markdown("**🛠️ Skills detected:**")
            for skill in rd.get("skills", []):
                st.markdown(f'<span class="skill-tag">{skill}</span>',
                            unsafe_allow_html=True)

st.markdown("---")

if not st.session_state.session_complete:
    total = len(st.session_state.questions)
    curr  = st.session_state.current_q

    st.markdown(f"**Question {curr + 1} of {total}**")
    st.progress((curr) / total)

    q = st.session_state.questions[curr]
    st.markdown(f'<div class="question-box">💬 {q}</div>', unsafe_allow_html=True)
    st.markdown("")

    answer = st.text_area("Your Answer", height=160,
                           placeholder="Type your answer here...",
                           key=f"answer_{curr}")

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("✅ Submit Answer", use_container_width=True):
            if not answer.strip():
                st.warning("Please write an answer before submitting.")
            else:
                with st.spinner("🤖 Evaluating your answer..."):
                    eval_result = st.session_state.engine.evaluate(
                        question=q,
                        answer=answer,
                        resume_data=st.session_state.resume_data,
                    )
                st.session_state.answers.append(answer)
                st.session_state.evaluations.append(eval_result)

                if curr + 1 >= total:
                    st.session_state.session_complete = True
                else:
                    st.session_state.current_q += 1
                st.rerun()

else:
    total = len(st.session_state.questions)
    evals = st.session_state.evaluations
    avg_score = round(sum(e["score"] for e in evals) / total, 1)

    st.markdown("## 📊 Interview Report")
    col1, col2, col3 = st.columns(3)
    with col1:
        color = "#38a169" if avg_score >= 7 else "#d69e2e" if avg_score >= 5 else "#e53e3e"
        st.markdown(f"""
        <div class="score-box" style="background:{color}">
            {avg_score}/10<br>
            <span class="score-label">Overall Score</span>
        </div>""", unsafe_allow_html=True)
    with col2:
        strong = sum(1 for e in evals if e["score"] >= 7)
        st.metric("💪 Strong Answers", f"{strong}/{total}")
    with col3:
        improve = sum(1 for e in evals if e["score"] < 5)
        st.metric("📈 Need Improvement", f"{improve}/{total}")

    st.markdown("---")
    st.markdown("### 📝 Detailed Feedback")
    for i, (q, a, e) in enumerate(zip(
        st.session_state.questions,
        st.session_state.answers,
        st.session_state.evaluations
    )):
        with st.expander(f"Q{i+1}: {q[:80]}...  |  Score: {e['score']}/10"):
            st.markdown(f'<div class="question-box">💬 {q}</div>',
                        unsafe_allow_html=True)
            st.markdown("")
            st.markdown("**Your Answer:**")
            st.markdown(f"> {a}")
            st.markdown("")
            st.markdown(
                f'<div class="feedback-box">✅ <b>Feedback</b><br>{e["feedback"]}</div>',
                unsafe_allow_html=True)
            st.markdown("")
            st.markdown(
                f'<div class="improvement-box">💡 <b>How to improve</b><br>{e["improvement"]}</div>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Overall Recommendations")
    all_improvements = " ".join(e["improvement"] for e in evals)
    st.info(st.session_state.engine.overall_recommendation(
        avg_score=avg_score,
        resume_data=st.session_state.resume_data,
        improvements=all_improvements,
    ))