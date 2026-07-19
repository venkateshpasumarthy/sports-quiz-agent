# 🏆 Quiz Board — AI-Powered Sports Quiz Generation Agent

An AI agent that generates fresh, factually-grounded sports multiple-choice quizzes
for social media content, using **Retrieval-Augmented Generation (RAG)** over a
**ChromaDB** vector store plus **live web search** for recency, with **Gemini** doing
the writing.

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
                  Validated quiz JSON → Dashboard
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
4. **Validate** — the backend parses and validates the JSON shape (exactly 4
   options, a valid correct answer key, required fields) before it ever reaches
   the frontend, so malformed generations fail loudly instead of silently
   corrupting the UI.
5. **Regenerate** — the "Regenerate" button re-runs generation while passing
   the previous batch of questions back to Gemini with an explicit instruction
   not to repeat them, so users get variety on demand.

---

## Project structure

```
sports-quiz-agent/
├── backend/
│   ├── main.py          FastAPI app + REST endpoints, serves the frontend
│   ├── quiz_agent.py     Core agent: RAG + web search + LLM generation + validation
│   ├── vector_store.py   ChromaDB wrapper (seeding, retrieval, re-indexing)
│   ├── web_search.py     DuckDuckGo-based web search (no API key required)
│   └── config.py         Environment/config loading
├── data/
│   └── seed_knowledge.json   Starter sports knowledge base loaded into ChromaDB
├── frontend/
│   └── index.html        Single-file dashboard (vanilla JS, no build step)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Prerequisites
- Python 3.10+
- A [Gemini API key](https://aistudio.google.com/apikey)

### 2. Install dependencies
```bash
cd sports-quiz-agent
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# then edit .env and paste your GEMINI_API_KEY
```

### 4. Run the app
```bash
uvicorn backend.main:app --reload --port 8000
```

Open **http://localhost:8000** — the dashboard is served directly by the backend.

> On first run, ChromaDB will download its default embedding model
> (`all-MiniLM-L6-v2`, ~90MB) and seed itself from `data/seed_knowledge.json`.
> This requires outbound internet access once; after that it's cached locally
> in `./chroma_data`.

---

## API Reference

| Method | Endpoint             | Description                                   |
|--------|-----------------------|------------------------------------------------|
| GET    | `/api/meta`            | List supported sports & difficulty levels      |
| GET    | `/api/health`          | Health check + ChromaDB fact count              |
| POST   | `/api/generate-quiz`   | Generate a new quiz (see body below)            |

**POST `/api/generate-quiz`** body:
```json
{
  "sport": "Badminton",
  "difficulty": "Medium",
  "num_questions": 5,
  "avoid_questions": null
}
```

Response:
```json
{
  "sport": "Badminton",
  "difficulty": "Medium",
  "questions": [
    {
      "question": "Which country won the Thomas Cup in 2022?",
      "options": {"A": "Indonesia", "B": "India", "C": "China", "D": "Denmark"},
      "correct_answer": "B",
      "explanation": "India won its first-ever Thomas Cup title in 2022 after defeating Indonesia in the final."
    }
  ],
  "sources": {"vector_db_facts_used": 4, "web_facts_used": 5}
}
```

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
- **Strict server-side JSON validation** — the LLM output is never trusted
  blindly; malformed responses raise a clear 502 error rather than reaching
  the UI in a broken state.

## Known limitations / next steps

- The seed knowledge base is intentionally small (~22 facts) for demo purposes;
  a production version would ingest a larger structured sports dataset.
- No persistence of quiz history/analytics yet — would be a natural next
  feature for a "content calendar" style dashboard.
- Web search quality depends on DuckDuckGo's result relevance; a dedicated
  sports data API (e.g. a stats provider) would improve accuracy further for
  fast-moving stats like current league standings.
