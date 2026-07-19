"""
app.py
------
Streamlit dashboard for the AI-Powered Sports Quiz Generation Agent.

This is a UI layer only -- all the actual agent logic (ChromaDB retrieval,
web search, LLM generation, JSON validation) lives in backend/quiz_agent.py
and is untouched/reused here.

Run with:
    streamlit run app.py
"""
import os
import streamlit as st

# ---------------------------------------------------------------------------
# Bridge Streamlit Cloud's "Secrets" into normal environment variables so
# backend/config.py (which reads via os.getenv) works unchanged whether
# running locally with a .env file or deployed on Streamlit Community Cloud.
# Locally, st.secrets simply won't find a secrets.toml file and this is a
# harmless no-op.
# ---------------------------------------------------------------------------
try:
    for _key in ("LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass  # no secrets.toml present (e.g. local run with .env) -- that's fine

from backend.config import SUPPORTED_SPORTS, DIFFICULTY_LEVELS
from backend.quiz_agent import generate_quiz

st.set_page_config(page_title="Quiz Board — Sports Quiz Agent", page_icon="🏆", layout="centered")

# ---------------------------------------------------------------------------
# Minimal styling to match the scoreboard theme (optional, purely cosmetic)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0B3D2E; }
    h1, h2, h3, p, label, .stMarkdown { color: #F5F3EA !important; }
    .stButton>button {
        background-color: #FFB627;
        color: #14201C;
        font-weight: 700;
        border-radius: 4px;
        border: none;
    }
    .stButton>button:hover { background-color: #ffc95c; color: #14201C; }
    div[data-testid="stMetricValue"] { color: #FFB627; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "answers" not in st.session_state:
    st.session_state.answers = {}  # {question_index: chosen_letter}
if "previous_questions" not in st.session_state:
    st.session_state.previous_questions = []

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("<p style='color:#FFB627; letter-spacing:0.3em; font-size:12px; text-transform:uppercase;'>AI Agent · RAG + Web Search</p>", unsafe_allow_html=True)
st.title("🏆 Quiz Board")
st.caption("Pick a sport and difficulty. Answer each question to reveal if you're right.")

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    sport = st.selectbox("Sport", SUPPORTED_SPORTS)
with col2:
    difficulty = st.selectbox("Difficulty", DIFFICULTY_LEVELS, index=1)
with col3:
    num_questions = st.selectbox("Questions", [4, 5], index=1)

gen_col, regen_col = st.columns(2)
generate_clicked = gen_col.button("Generate Quiz", use_container_width=True)
regenerate_clicked = regen_col.button(
    "Regenerate", use_container_width=True,
    disabled=st.session_state.quiz_data is None,
)


def run_generation(avoid: bool):
    with st.spinner("Fetching facts from ChromaDB + web search, then writing questions…"):
        try:
            avoid_list = st.session_state.previous_questions if avoid else None
            data = generate_quiz(
                sport=sport,
                difficulty=difficulty,
                num_questions=num_questions,
                avoid_questions=avoid_list,
            )
            st.session_state.quiz_data = data
            st.session_state.answers = {}
            st.session_state.previous_questions = [q["question"] for q in data["questions"]]
        except Exception as e:
            st.error(f"Couldn't generate quiz: {e}")


if generate_clicked:
    run_generation(avoid=False)
if regenerate_clicked:
    run_generation(avoid=True)

# ---------------------------------------------------------------------------
# Quiz display
# ---------------------------------------------------------------------------
data = st.session_state.quiz_data

if data:
    sources = data["sources"]
    st.caption(
        f"Grounded on {sources['vector_db_facts_used']} ChromaDB fact(s) "
        f"+ {sources['web_facts_used']} live web result(s)"
    )
    st.divider()

    questions = data["questions"]
    letters = ["A", "B", "C", "D"]

    for i, q in enumerate(questions):
        st.subheader(f"{i + 1:02d}. {q['question']}")

        answered = i in st.session_state.answers

        if not answered:
            choice = st.radio(
                "Pick an answer",
                letters,
                format_func=lambda L, q=q: f"{L}. {q['options'][L]}",
                key=f"radio_{i}_{q['question']}",
                index=None,
                label_visibility="collapsed",
            )
            if st.button("Submit answer", key=f"submit_{i}_{q['question']}"):
                if choice is None:
                    st.warning("Pick an option first.")
                else:
                    st.session_state.answers[i] = choice
                    st.rerun()
        else:
            picked = st.session_state.answers[i]
            correct = q["correct_answer"]
            for L in letters:
                label = f"{L}. {q['options'][L]}"
                if L == correct:
                    st.success(label + ("  ✅ Correct" if L == picked else "  ← Correct answer"))
                elif L == picked:
                    st.error(label + "  ❌ Your pick")
                else:
                    st.markdown(f"<span style='opacity:0.5'>{label}</span>", unsafe_allow_html=True)
            st.info(f"**Why:** {q['explanation']}")

        st.divider()

    answered_count = len(st.session_state.answers)
    if answered_count == len(questions):
        correct_count = sum(
            1 for i, q in enumerate(questions)
            if st.session_state.answers[i] == q["correct_answer"]
        )
        st.markdown(
            f"<h3 style='text-align:center; color:#FFB627;'>You scored {correct_count} / {len(questions)}</h3>",
            unsafe_allow_html=True,
        )
else:
    st.info("Pick a sport and difficulty above, then click **Generate Quiz** to get started.")

st.markdown(
    "<p style='text-align:center; opacity:0.4; font-size:12px; margin-top:32px;'>"
    "Sports Quiz Generation Agent · ChromaDB retrieval + live web search + Gemini/OpenAI</p>",
    unsafe_allow_html=True,
)
