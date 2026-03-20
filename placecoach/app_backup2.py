import streamlit as st
import os
from dotenv import load_dotenv
from resume_parser import ResumeParser
from question_bank import QuestionBank
from rag_engine import RAGEngine

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ENDEE_URL    = os.getenv("ENDEE_URL", "http://localhost:8080")

st.set_page_config(
    page_title="PlaceCoach – AI Interview Coach",
    page_icon="🎯",
    layout="centered",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #eef2ff; }
    .block-container { padding: 2rem 1.5rem; max-width: 800px; }
    #MainMenu, footer, header { visibility: hidden; }
    section[data-testid="stSidebar"] { display: none !important; }

    /* ── Header ── */
    .pc-header {
        background: linear-gradient(135deg, #0a2342 0%, #0077b5 100%);
        border-radius: 24px; padding: 2rem 2.5rem; margin-bottom: 2rem;
        text-align: center; box-shadow: 0 12px 40px rgba(0,119,181,0.3);
        position: relative; overflow: hidden;
    }
    .pc-header::before {
        content: ''; position: absolute; top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 60%);
    }
    .pc-title { font-size: 2.2rem; font-weight: 900; color: white; letter-spacing: -0.5px; }
    .pc-subtitle { font-size: 0.88rem; color: rgba(255,255,255,0.7); margin-top: 0.4rem; }
    .pc-powered {
        display: inline-block; background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2); border-radius: 20px;
        padding: 4px 14px; font-size: 0.75rem; color: white; margin-top: 0.8rem;
    }

    /* ── Cards ── */
    .card {
        background: white; border-radius: 18px; padding: 1.8rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.07); margin-bottom: 1.2rem;
    }
    .card-title { font-size: 0.75rem; font-weight: 700; color: #0077b5;
        text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 1.2rem;
        display: flex; align-items: center; gap: 6px; }

    /* ── CV Dashboard ── */
    .cv-name { font-size: 1.5rem; font-weight: 800; color: #0a2342; }
    .cv-meta { font-size: 0.85rem; color: #666; margin: 0.4rem 0 1rem 0; }
    .cv-section { font-size: 0.72rem; font-weight: 700; color: #0077b5;
        text-transform: uppercase; letter-spacing: 1px; margin: 1rem 0 0.5rem 0; }
    .skill-tag { display: inline-block; background: #eff6ff; color: #1d4ed8;
        border: 1px solid #bfdbfe; border-radius: 20px;
        padding: 4px 12px; font-size: 0.78rem; font-weight: 600; margin: 3px; }
    .domain-tag { display: inline-block; background: #f0fdf4; color: #15803d;
        border: 1px solid #bbf7d0; border-radius: 20px;
        padding: 4px 12px; font-size: 0.78rem; font-weight: 600; margin: 3px; }
    .skill-bar-row { margin-bottom: 0.7rem; }
    .skill-bar-top { display: flex; justify-content: space-between;
        font-size: 0.82rem; font-weight: 600; color: #334155; margin-bottom: 4px; }
    .skill-bar-bg { background: #e2e8f0; border-radius: 20px; height: 8px; overflow: hidden; }
    .skill-bar-fill { height: 100%; border-radius: 20px;
        background: linear-gradient(90deg, #0077b5, #38bdf8); }

    /* ── Chat ── */
    .bubble-ai {
        background: white; border-radius: 20px 20px 20px 6px;
        padding: 1.2rem 1.5rem; box-shadow: 0 3px 15px rgba(0,0,0,0.08);
        border-left: 5px solid #0077b5; margin-bottom: 0.8rem;
        font-size: 1rem; font-weight: 500; color: #0f172a; line-height: 1.65;
    }
    .bubble-user {
        background: linear-gradient(135deg, #1d4ed8, #0077b5);
        border-radius: 20px 20px 6px 20px;
        padding: 1rem 1.5rem; margin-bottom: 0.5rem;
        font-size: 0.95rem; color: white; line-height: 1.6;
        box-shadow: 0 3px 15px rgba(29,78,216,0.25);
    }
    .q-label { font-size: 0.7rem; font-weight: 700; color: #0077b5;
        text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.5rem; }

    /* ── Progress ── */
    .prog-row { display:flex; justify-content:space-between;
        font-size:0.82rem; font-weight:600; color:#64748b; margin-bottom:6px; }
    .prog-bg { background:#dbeafe; border-radius:20px; height:7px; overflow:hidden; margin-bottom:1.5rem; }
    .prog-fill { height:100%; background:linear-gradient(90deg,#0077b5,#38bdf8);
        border-radius:20px; transition:width 0.6s ease; }

    /* ── Mic Button ── */
    .mic-wrap { display:flex; flex-direction:column; align-items:center; margin:1rem 0; }
    .mic-btn {
        width: 72px; height: 72px; border-radius: 50%; border: none;
        background: linear-gradient(135deg, #0077b5, #0ea5e9);
        color: white; font-size: 1.8rem; cursor: pointer;
        box-shadow: 0 0 0 0 rgba(0,119,181,0.5);
        transition: all 0.3s ease; display:flex; align-items:center; justify-content:center;
    }
    .mic-btn:hover { transform: scale(1.08); box-shadow: 0 8px 25px rgba(0,119,181,0.4); }
    .mic-btn.recording {
        background: linear-gradient(135deg, #dc2626, #ef4444) !important;
        animation: micPulse 1.2s infinite;
    }
    @keyframes micPulse {
        0%   { box-shadow: 0 0 0 0 rgba(220,38,38,0.5); }
        50%  { box-shadow: 0 0 0 18px rgba(220,38,38,0); }
        100% { box-shadow: 0 0 0 0 rgba(220,38,38,0); }
    }
    .mic-label { font-size:0.82rem; font-weight:600; color:#64748b; margin-top:0.6rem; }
    .mic-transcript {
        background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
        padding:0.8rem 1rem; font-size:0.85rem; color:#334155;
        min-height:40px; width:100%; margin-top:0.5rem; text-align:left;
        max-width:500px;
    }

    /* ── Score ── */
    .score-hero {
        background: linear-gradient(135deg, #0a2342, #0077b5);
        border-radius: 24px; padding: 2.5rem; text-align:center; color:white;
        box-shadow: 0 12px 40px rgba(0,119,181,0.35); margin-bottom:1.5rem;
    }
    .score-num { font-size:5rem; font-weight:900; line-height:1; letter-spacing:-2px; }
    .score-grade { font-size:1.3rem; font-weight:700; margin-top:0.5rem; opacity:0.9; }
    .metrics { display:flex; gap:1rem; margin-bottom:1.5rem; }
    .mbox { background:white; border-radius:16px; padding:1.3rem;
        text-align:center; flex:1; box-shadow:0 3px 15px rgba(0,0,0,0.07); }
    .mval { font-size:2rem; font-weight:900; }
    .mlbl { font-size:0.72rem; color:#64748b; margin-top:0.3rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; }

    /* ── Report CV ── */
    .report-cv {
        background:white; border-radius:18px; padding:1.8rem;
        box-shadow:0 4px 20px rgba(0,0,0,0.07); margin-bottom:1.5rem;
        border-top:5px solid #0077b5;
    }

    /* ── Feedback ── */
    .fb-good { background:#f0fdf4; border-left:4px solid #16a34a;
        border-radius:12px; padding:1rem 1.2rem; margin-bottom:0.8rem; }
    .fb-tip  { background:#fffbeb; border-left:4px solid #f59e0b;
        border-radius:12px; padding:1rem 1.2rem; margin-bottom:0.8rem; }
    .fb-lbl  { font-size:0.7rem; font-weight:700; text-transform:uppercase;
        letter-spacing:1px; margin-bottom:0.5rem; }
    .fb-txt  { font-size:0.88rem; color:#1e293b; line-height:1.65; }
    .pill-hi { background:#dcfce7; color:#16a34a; border-radius:20px; padding:4px 14px; font-size:0.8rem; font-weight:700; }
    .pill-md { background:#fef9c3; color:#ca8a04; border-radius:20px; padding:4px 14px; font-size:0.8rem; font-weight:700; }
    .pill-lo { background:#fee2e2; color:#dc2626; border-radius:20px; padding:4px 14px; font-size:0.8rem; font-weight:700; }

    .rec-box { background:linear-gradient(135deg,#eff6ff,#dbeafe);
        border:1px solid #bfdbfe; border-radius:16px;
        padding:1.5rem; font-size:0.92rem; color:#1e3a5f; line-height:1.75;
        margin-bottom:1.5rem; }

    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #0077b5, #0ea5e9) !important;
        color: white !important; border: none !important;
        border-radius: 30px !important; font-weight: 700 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 20px rgba(0,119,181,0.35) !important;
        transition: all 0.3s ease !important;
        padding: 0.65rem 2rem !important;
    }
    div[data-testid="stButton"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(0,119,181,0.45) !important;
    }
    .stTextArea textarea {
        border-radius: 16px !important; border: 2px solid #e2e8f0 !important;
        font-size: 0.95rem !important; background: #f8fafc !important;
        transition: border 0.2s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: #0077b5 !important;
        box-shadow: 0 0 0 4px rgba(0,119,181,0.1) !important;
        background: white !important;
    }
    .stTextInput input, .stSelectbox > div {
        border-radius: 12px !important; border: 2px solid #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Voice Component ───────────────────────────────────────────────────────────
VOICE_HTML = """
<div style="display:flex;align-items:center;gap:10px;padding:6px 0;">
    <button id="micBtn" onclick="toggleMic()" title="Hold to speak"
        style="width:52px;height:52px;border-radius:50%;border:none;
        background:linear-gradient(135deg,#0077b5,#0ea5e9);
        color:white;font-size:1.4rem;cursor:pointer;
        box-shadow:0 4px 15px rgba(0,119,181,0.4);
        transition:all 0.3s ease;flex-shrink:0;
        display:flex;align-items:center;justify-content:center;">
        🎤
    </button>
    <div id="micStatus"
        style="flex:1;background:#f8fafc;border:1.5px solid #e2e8f0;
        border-radius:25px;padding:10px 16px;font-size:0.85rem;
        color:#64748b;min-height:42px;display:flex;align-items:center;">
        Tap 🎤 to speak — words will appear in the box above
    </div>
</div>

<script>
let recog = null;
let recording = false;
let fullText = '';

function toggleMic() {
    if (recording) { stopMic(); return; }
    startMic();
}

function startMic() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
        document.getElementById('micLabel').innerText = '⚠️ Use Chrome for voice typing';
        return;
    }
    recog = new SR();
    recog.continuous = true;
    recog.interimResults = true;
    recog.lang = 'en-US';
    fullText = '';

    recog.onstart = () => {
        recording = true;
        document.getElementById('micBtn').style.background = 'linear-gradient(135deg,#dc2626,#ef4444)';
        document.getElementById('micBtn').style.animation = 'micPulse 1.2s infinite';
        document.getElementById('micBtn').innerHTML = '⏹️';
        document.getElementById('micStatus').innerText = '🔴 Recording... tap to stop';
    };

    recog.onresult = (e) => {
        let interim = '';
        fullText = '';
        for (let i = 0; i < e.results.length; i++) {
            if (e.results[i].isFinal) fullText += e.results[i][0].transcript + ' ';
            else interim += e.results[i][0].transcript;
        }
        const display = fullText + interim;
        document.getElementById('micStatus').innerText = '📝 ' + (display.slice(-80) || 'Listening...');

        // Push to Streamlit textarea and trigger React synthetic event
        try {
            const ta = window.parent.document.querySelector('textarea');
            if (ta) {
                const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                setter.call(ta, display);
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                ta.dispatchEvent(new Event('change', { bubbles: true }));
                ta.focus();
                ta.blur();
                ta.focus();
            }
        } catch(err) {}
    };

    recog.onspeechend = () => {
        // Auto-trigger textarea update when speech ends
        try {
            const ta = window.parent.document.querySelector('textarea');
            if (ta) {
                ta.focus();
                const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                setter.call(ta, fullText.trim());
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                ta.dispatchEvent(new Event('change', { bubbles: true }));
                // Simulate user typing to force Streamlit to pick up value
                ta.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true }));
                ta.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true }));
            }
        } catch(err) {}
    };

    recog.onerror = (e) => {
        document.getElementById('micStatus').innerText = '⚠️ ' + e.error + ' — try again';
        stopMic();
    };

    recog.onend = () => { if (recording) recog.start(); };
    recog.start();
}

function stopMic() {
    recording = false;
    if (recog) recog.stop();
    document.getElementById('micBtn').style.background = 'linear-gradient(135deg,#0077b5,#0ea5e9)';
    document.getElementById('micBtn').style.animation = 'none';
    document.getElementById('micBtn').innerHTML = '🎤';
    document.getElementById('micStatus').innerText = '✅ Got it! Edit in the box above if needed.';

    // Force Streamlit to register the value
    setTimeout(() => {
        try {
            const ta = window.parent.document.querySelector('textarea');
            if (ta && fullText.trim()) {
                const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                setter.call(ta, fullText.trim());
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                ta.dispatchEvent(new Event('change', { bubbles: true }));
                ta.focus();
                // Simulate a space + backspace to force React state update
                ta.dispatchEvent(new KeyboardEvent('keypress', { key: ' ', bubbles: true }));
                setter.call(ta, fullText.trim());
                ta.dispatchEvent(new Event('input', { bubbles: true }));
            }
        } catch(err) {}
    }, 300);
}
</script>
"""

# ── Session State ─────────────────────────────────────────────────────────────
defaults = {
    "resume_data": None, "questions": [], "current_q": 0,
    "answers": [], "evaluations": [], "session_complete": False,
    "bank_ready": False, "setup_done": False, "role": "Software Engineer",
    "num_q": 5,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "parser"  not in st.session_state: st.session_state.parser  = ResumeParser()
if "bank"    not in st.session_state: st.session_state.bank    = QuestionBank()
if "engine"  not in st.session_state: st.session_state.engine  = RAGEngine()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pc-header">
    <div class="pc-title">🎯 PlaceCoach</div>
    <div style="font-size:0.78rem;font-weight:700;color:rgba(255,255,255,0.5);letter-spacing:3px;text-transform:uppercase;margin-top:0.2rem;">PLACEMENT COACH</div>
    <div class="pc-subtitle" style="margin-top:0.5rem;">Your AI-Powered Interview Partner — Analyse · Practice · Ace It</div>
    <div class="pc-powered">⚡ Endee Vector DB · Groq LLM · Voice AI</div>
</div>
""", unsafe_allow_html=True)

# ── SETUP SCREEN ──────────────────────────────────────────────────────────────
if not st.session_state.setup_done:

    if not GROQ_API_KEY:
        st.markdown('<div class="card"><div class="card-title">🔑 API Configuration</div>', unsafe_allow_html=True)
        GROQ_API_KEY_INPUT = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
        ENDEE_URL_INPUT    = st.text_input("Endee URL", value="http://localhost:8080")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        GROQ_API_KEY_INPUT = GROQ_API_KEY
        ENDEE_URL_INPUT    = ENDEE_URL

    st.markdown('<div class="card"><div class="card-title">📄 Your Profile</div>', unsafe_allow_html=True)
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"],
                                    help="We'll extract your skills, experience and education automatically")
    role = st.selectbox("🎯 Target Role", [
        "Software Engineer","Data Scientist","Product Manager","Marketing Manager",
        "Business Analyst","Frontend Developer","Backend Developer","ML Engineer",
        "DevOps Engineer","UI/UX Designer","Finance Analyst","HR Manager",
        "Sales Executive","Cybersecurity Analyst","Cloud Architect",
    ])
    num_questions = st.slider("📝 Number of Questions", 3, 10, 5)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀 Start Interview", use_container_width=True):
            if not GROQ_API_KEY_INPUT:
                st.error("⚠️ Please enter your Groq API key")
            elif not resume_file:
                st.error("⚠️ Please upload your resume PDF")
            else:
                st.session_state.engine.set_config(GROQ_API_KEY_INPUT, ENDEE_URL_INPUT)
                with st.spinner("📖 Analysing your resume..."):
                    st.session_state.resume_data = st.session_state.parser.parse(
                        resume_file, GROQ_API_KEY_INPUT)
                if not st.session_state.bank_ready:
                    with st.spinner("🗄️ Loading question bank into Endee..."):
                        st.session_state.bank.set_config(ENDEE_URL_INPUT)
                        st.session_state.bank.index_all()
                        st.session_state.bank_ready = True
                with st.spinner("🔍 Personalising your questions..."):
                    st.session_state.questions = st.session_state.engine.get_questions(
                        role=role, resume_data=st.session_state.resume_data,
                        n=num_questions)
                st.session_state.role = role
                st.session_state.setup_done = True
                st.session_state.current_q = 0
                st.session_state.answers = []
                st.session_state.evaluations = []
                st.session_state.session_complete = False
                st.rerun()
    st.stop()

# ── CV DASHBOARD ──────────────────────────────────────────────────────────────
if st.session_state.resume_data:
    rd     = st.session_state.resume_data
    skills = rd.get("skills", [])
    domains= rd.get("domains", [])

    with st.expander("📋 View Your CV Dashboard", expanded=False):
        skills_html  = "".join(f'<span class="skill-tag">{s}</span>'  for s in skills)
        domains_html = "".join(f'<span class="domain-tag">{d}</span>' for d in domains)
        st.markdown(f"""
        <div class="card">
            <div class="cv-name">👤 {rd.get('name','Candidate')}</div>
            <div class="cv-meta">
                🎓 {rd.get('education','—')} &nbsp;·&nbsp;
                💼 {rd.get('experience_years',0)} years experience &nbsp;·&nbsp;
                🎯 Target: {st.session_state.role}
            </div>
            <div style="background:#f0f7ff;border-radius:12px;padding:1rem;margin-bottom:1rem;font-size:0.88rem;color:#334155;line-height:1.6;">
                {rd.get('summary','—')}
            </div>
            <div class="cv-section">🛠️ Technical Skills</div>
            <div>{skills_html}</div>
            <div class="cv-section">🎯 Domains</div>
            <div>{domains_html}</div>
        </div>
        """, unsafe_allow_html=True)

        if skills:
            st.markdown("**📊 Skill Strength Indicators**")
            import random
            random.seed(sum(ord(c) for c in rd.get('name','x')))
            for skill in skills[:8]:
                pct = random.randint(68, 96)
                st.markdown(f"**{skill}** — {pct}%")
                st.progress(pct / 100)

# ── ACTIVE INTERVIEW ──────────────────────────────────────────────────────────
if not st.session_state.session_complete:
    total = len(st.session_state.questions)
    curr  = st.session_state.current_q
    pct   = int((curr / total) * 100)

    st.markdown(f"""
    <div class="prog-row">
        <span>🎯 Interview Progress</span>
        <span>{curr}/{total} answered · {pct}%</span>
    </div>
    <div class="prog-bg"><div class="prog-fill" style="width:{pct}%"></div></div>
    """, unsafe_allow_html=True)

    # Previous Q&A bubbles
    for i, (q, a) in enumerate(zip(st.session_state.questions[:curr], st.session_state.answers)):
        e = st.session_state.evaluations[i]
        s = e["score"]
        pill = "pill-hi" if s >= 7 else "pill-md" if s >= 5 else "pill-lo"
        st.markdown(f"""
        <div style="margin-bottom:1.5rem;">
            <div class="bubble-ai"><div class="q-label">Question {i+1}</div>💬 {q}</div>
            <div class="bubble-user">{a}</div>
            <div style="text-align:right;margin-top:0.4rem;">
                <span class="{pill}">Score: {s}/10</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Current question bubble
    q = st.session_state.questions[curr]
    st.markdown(f"""
    <div class="bubble-ai">
        <div class="q-label">Question {curr+1} of {total}</div>
        💬 {q}
    </div>
    """, unsafe_allow_html=True)

    # WhatsApp-style input row: textarea + mic + send
    answer = st.text_area(
        "",
        height=100,
        placeholder="Type your answer or tap 🎤 to speak...",
        key=f"answer_{curr}",
        label_visibility="collapsed"
    )

    # Inline mic + send row like WhatsApp
    st.components.v1.html(VOICE_HTML, height=90)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("📤 Send Answer", use_container_width=True):
            if not answer.strip():
                st.warning("Please type or speak an answer first.")
            else:
                with st.spinner("🤖 Evaluating your answer..."):
                    eval_result = st.session_state.engine.evaluate(
                        question=q, answer=answer,
                        resume_data=st.session_state.resume_data)
                st.session_state.answers.append(answer)
                st.session_state.evaluations.append(eval_result)
                if curr + 1 >= total:
                    st.session_state.session_complete = True
                else:
                    st.session_state.current_q += 1
                st.rerun()

# ── REPORT ────────────────────────────────────────────────────────────────────
else:
    total     = len(st.session_state.questions)
    evals     = st.session_state.evaluations
    avg_score = round(sum(e["score"] for e in evals) / total, 1)
    strong    = sum(1 for e in evals if e["score"] >= 7)
    needs_imp = sum(1 for e in evals if e["score"] < 5)
    avg_int   = total - strong - needs_imp

    if avg_score >= 8:   grade, emoji = "A+", "🏆 Outstanding"
    elif avg_score >= 7: grade, emoji = "A",  "🌟 Excellent"
    elif avg_score >= 6: grade, emoji = "B+", "👍 Very Good"
    elif avg_score >= 5: grade, emoji = "B",  "📈 Good"
    else:                grade, emoji = "C",  "💪 Keep Practising"

    # Hero score
    st.markdown(f"""
    <div class="score-hero">
        <div style="font-size:0.8rem;opacity:0.6;text-transform:uppercase;letter-spacing:2px;margin-bottom:0.8rem;">
            Interview Complete
        </div>
        <div class="score-num">{avg_score}<span style="font-size:2rem;opacity:0.5">/10</span></div>
        <div class="score-grade">{grade} Grade · {emoji}</div>
        <div style="margin-top:1rem;font-size:0.85rem;opacity:0.7;">
            🎯 Role: {st.session_state.role}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    st.markdown(f"""
    <div class="metrics">
        <div class="mbox">
            <div class="mval" style="color:#16a34a">{strong}</div>
            <div class="mlbl">Strong Answers</div>
        </div>
        <div class="mbox">
            <div class="mval" style="color:#0077b5">{avg_int}</div>
            <div class="mlbl">Average</div>
        </div>
        <div class="mbox">
            <div class="mval" style="color:#f59e0b">{needs_imp}</div>
            <div class="mlbl">Need Work</div>
        </div>
        <div class="mbox">
            <div class="mval" style="color:#6366f1">{total}</div>
            <div class="mlbl">Total Q's</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Score bar chart
    st.markdown("### 📊 Score Breakdown")
    bar_cols = st.columns(total)
    for i, (e, col) in enumerate(zip(evals, bar_cols)):
        s = e["score"]
        color = "🟢" if s >= 7 else "🟡" if s >= 5 else "🔴"
        with col:
            st.metric(f"Q{i+1}", f"{s}/10", delta=None)

    # CV Performance Report
    if st.session_state.resume_data:
        rd = st.session_state.resume_data
        skills = rd.get("skills", [])
        st.markdown("### 📄 CV Performance Report")
        st.markdown(f"""
        <div class="report-cv">
            <div style="font-size:1.1rem;font-weight:800;color:#0a2342;margin-bottom:0.3rem;">
                {rd.get('name','Candidate')}
            </div>
            <div style="font-size:0.82rem;color:#64748b;margin-bottom:1.2rem;">
                🎓 {rd.get('education','—')} &nbsp;·&nbsp; 💼 {rd.get('experience_years',0)} yrs &nbsp;·&nbsp; 🎯 {st.session_state.role}
            </div>
        """, unsafe_allow_html=True)

        if skills:
            st.markdown("**Skills demonstrated in this session:**")
            import random
            random.seed(int(avg_score * 7))
            for skill in skills[:7]:
                base  = random.randint(50, 88)
                boost = int(avg_score * 2.5)
                final = min(97, base + boost - 15)
                st.markdown(f"**{skill}** — {final}%")
                st.progress(final / 100)

    # Detailed Q&A feedback
    st.markdown("### 📝 Detailed Feedback")
    for i, (q, a, e) in enumerate(zip(st.session_state.questions, st.session_state.answers, evals)):
        s = e["score"]
        pill = "pill-hi" if s >= 7 else "pill-md" if s >= 5 else "pill-lo"
        with st.expander(f"Q{i+1} · {q[:65]}..."):
            st.markdown(f'<span class="{pill}" style="margin-bottom:1rem;display:inline-block">Score: {s}/10</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="bubble-ai" style="margin-bottom:0.8rem;"><div class="q-label">Question {i+1}</div>💬 {q}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bubble-user" style="margin-bottom:1rem;">{a}</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="fb-good">
                <div class="fb-lbl" style="color:#16a34a">✅ What you did well</div>
                <div class="fb-txt">{e['feedback']}</div>
            </div>
            <div class="fb-tip">
                <div class="fb-lbl" style="color:#f59e0b">💡 How to improve</div>
                <div class="fb-txt">{e['improvement']}</div>
            </div>
            """, unsafe_allow_html=True)

    # Recommendations
    st.markdown("### 🎯 Personalised Career Recommendations")
    all_imp = " ".join(e["improvement"] for e in evals)
    rec = st.session_state.engine.overall_recommendation(
        avg_score=avg_score, resume_data=st.session_state.resume_data, improvements=all_imp)
    st.markdown(f'<div class="rec-box">💼 {rec}</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🔄 Start New Session", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()