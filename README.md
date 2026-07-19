# 🏆 Quiz Board — AI-Powered Sports Quiz Generation Agent

An AI agent that generates fresh, factually-grounded sports multiple-choice quizzes
for social media content, using **Retrieval-Augmented Generation (RAG)** over a
**ChromaDB** vector store plus **live web search** for recency, with **Gemini**
doing the writing. Built as a **Streamlit** dashboard.

---

## How it works

```
 User picks sport + difficulty
            │
            ▼
   ┌─────────────────────┐        ┌───────────────────────┐
   │   ChromaDB (RAG)     │        │   Live Web Search       │
   │  seeded knowledge     │        │  (DuckDuckGo, no key)   │
   │  base, semantic query │        │  fetches recent events   │
   └──────────┬───────────┘        └───────────┬────────────┘
              │                                 │
              └───────────────┬─────────────────┘
                               ▼
                     Merged grounding context
                               │
                               ▼
                    Gemini (strict JSON prompt)
                               │
                               ▼
              Validated quiz JSON → Streamlit dashboard
```

1. **Retrieve** — the sport + difficulty is used to query a ChromaDB collection
   seeded with sports facts (records, champions, tournament history). ChromaDB
   handles the embeddings itself (`all-MiniLM-L6-v2`), no external embedding API
   needed.
2. **Search** — a web search is run in parallel to catch anything the static
   knowledge base is missing (this season's winners, recent transfers, etc).
   New facts found this way are written back into ChromaDB so future requests
   for the same sport get faster and richer over time.
3. **Generate** — both context sources are merged into a single prompt sent to
   Gemini with a system prompt that explicitly forbids inventing facts not
   present in the retrieved context, and forces strict JSON output.
4. **Validate** — the app parses and validates the JSON shape (exactly 4
   options, a valid correct answer key, required fields) before it ever reaches
   the UI, so malformed generations fail loudly instead of silently corrupting
   the display.
5. **Regenerate** — the "Regenerate" button re-runs generation while passing
   the previous batch of questions back to Gemini with an explicit instruction
   not to repeat them, so users get variety on demand.
6. **Answer & reveal** — each question is answered by picking an option and
   submitting; the dashboard then reveals whether you were right, highlights
   the correct answer, and shows the explanation. A final score is shown once
   all questions are answered.

---

## Project structure

```
sports-quiz-agent/
├── app.py                Streamlit dashboard (the app) — run with `streamlit run app.py`
├── backend/
│   ├── quiz_agent.py      Core agent: RAG + web search + LLM generation + validation
│   ├── vector_store.py    ChromaDB wrapper (seeding, retrieval, re-indexing)
│   ├── web_search.py      DuckDuckGo-based web search (no API key required)
│   └── config.py          Environment/config loading
├── data/
│   └── seed_knowledge.json   Starter sports knowledge base loaded into ChromaDB
├── requirements.txt
├── .env.example
└── README.md
```

`app.py` is a UI layer only — all the actual agent logic (retrieval, search,
generation, validation) lives in `backend/quiz_agent.py`.

---

## Setup

### 1. Prerequisites
- Python 3.10+
- A [Gemini API key](https://aistudio.google.com/apikey) (free tier available)

### 2. Install dependencies
```bash
cd sports-quiz-agent
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
```
Open `.env` and fill in your key:
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_actual_key_here
GEMINI_MODEL=gemini-2.5-flash
```

### 4. Run the app
```bash
streamlit run app.py
```
This opens the dashboard automatically in your browser, usually at
`http://localhost:8501`.

> On first run, ChromaDB will download its default embedding model
> (`all-MiniLM-L6-v2`, ~90MB) and seed itself from `data/seed_knowledge.json`.
> This requires outbound internet access once; after that it's cached locally
> in `./chroma_data`.

---

## Using the dashboard

1. Pick a **sport** and **difficulty** from the dropdowns.
2. Choose how many questions (4 or 5).
3. Click **Generate Quiz**.
4. For each question, pick an option and click **Submit answer** — this
   reveals whether you were correct, highlights the right answer, and shows
   the explanation.
5. Once all questions are answered, your score is shown at the bottom.
6. Click **Regenerate** for a fresh batch on the same sport/difficulty — the
   agent is told to avoid repeating the previous questions.

---

## Design decisions & trade-offs

- **DuckDuckGo over a paid search API** — keeps the project runnable by anyone
  without needing to sign up for a search API key. Swappable for Tavily/Bing/
  Serper by replacing `backend/web_search.py` if higher-quality results are
  needed in production.
- **Grounding via prompt constraints, not just retrieval** — the system prompt
  explicitly instructs Gemini to fall back to generic rules-based questions
  rather than invent statistics when context is thin, which is the main lever
  against hallucination beyond RAG itself.
- **Self-reinforcing knowledge base** — web search results are written back
  into ChromaDB, so the knowledge base organically grows and later requests
  need less live searching.
- **Strict validation before display** — the LLM output is never trusted
  blindly; malformed responses raise a clear error in the UI rather than
  silently breaking the quiz.
- **Streamlit over a custom frontend** — chosen for a fast, purely-Python
  dashboard with built-in state management (`st.session_state`), avoiding a
  separate JS frontend and API layer for a project of this scope.

## Known limitations / next steps

- The seed knowledge base is intentionally small (~22 facts) for demo purposes;
  a production version would ingest a larger structured sports dataset.
- No persistence of quiz history/analytics yet — would be a natural next
  feature for a "content calendar" style dashboard.
- Web search quality depends on DuckDuckGo's result relevance; a dedicated
  sports data API (e.g. a stats provider) would improve accuracy further for
  fast-moving stats like current league standings.
- Currently supports Gemini only; `backend/quiz_agent.py` was previously built
  with an OpenAI fallback path, which can be re-added easily if needed since
  the provider is abstracted behind a single `_call_llm()` function.
