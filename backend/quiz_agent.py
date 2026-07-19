"""
quiz_agent.py
-------------
The core AI agent. For a given sport + difficulty:

  1. Retrieves relevant grounded facts from ChromaDB (RAG).
  2. Runs a web search to pull in fresh / recent information.
  3. Merges both context sources and feeds them to Gemini with a strict
     JSON-only prompt, instructing it to only use the supplied context
     (grounding, to reduce hallucination).
  4. Parses + validates the JSON response into quiz question objects.
  5. Optionally re-indexes new web facts into ChromaDB for future reuse.
"""
import json
import re
from typing import List, Dict, Optional

from google import genai
from google.genai import types

from backend.config import GEMINI_API_KEY, GEMINI_MODEL
from backend.vector_store import vector_store
from backend.web_search import search_recent_sports_info

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

SYSTEM_PROMPT = """You are a sports trivia quiz-writer. You must ONLY use facts present \
in the CONTEXT block provided by the user to write quiz questions. If the context does not \
contain enough information to write a fact, invent a plausible, clearly-labeled generic \
question about the sport's rules instead of inventing a statistic or historical claim.

Never fabricate specific statistics, dates, scores, or names that are not grounded in the \
provided context. Respond with ONLY valid JSON, no markdown fences, no preamble, matching \
exactly this schema:

{
  "questions": [
    {
      "question": "string",
      "options": {"A": "string", "B": "string", "C": "string", "D": "string"},
      "correct_answer": "A" | "B" | "C" | "D",
      "explanation": "string, 1-2 sentences"
    }
  ]
}"""


def _build_user_prompt(sport: str, difficulty: str, num_questions: int,
                        context_snippets: List[str], avoid_questions: Optional[List[str]] = None) -> str:
    context_block = "\n".join(f"- {c}" for c in context_snippets) if context_snippets else \
        "(No specific context retrieved -- use only well-established general knowledge rules of the sport.)"

    avoid_block = ""
    if avoid_questions:
        avoid_list = "\n".join(f"- {q}" for q in avoid_questions)
        avoid_block = f"\n\nDo NOT repeat or closely rephrase any of these previously used questions:\n{avoid_list}"

    return f"""Sport: {sport}
Difficulty: {difficulty}
Number of questions: {num_questions}

CONTEXT (grounded facts you may use):
{context_block}
{avoid_block}

Write {num_questions} unique, factually accurate, engaging multiple-choice quiz questions \
about {sport} at {difficulty} difficulty, following the JSON schema exactly."""


def _extract_json(text: str) -> Dict:
    """Gemini should return raw JSON (response_mime_type='application/json'), but strip
    markdown fences defensively just in case."""
    cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def generate_quiz(sport: str, difficulty: str, num_questions: int = 5,
                   avoid_questions: Optional[List[str]] = None) -> Dict:
    """
    Main entry point for the agent. Returns a dict:
      {sport, difficulty, questions: [...], sources: {"vector_db": n, "web": n}}
    """
    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    # 1. Retrieve grounded facts from ChromaDB
    vector_facts = vector_store.retrieve(sport=sport, query=f"{sport} {difficulty} notable facts records champions", n_results=6)

    # 2. Pull fresh info from the web to reduce staleness
    web_facts_raw = search_recent_sports_info(sport=sport, difficulty=difficulty, max_results=5)
    web_facts_text = [f["text"] for f in web_facts_raw]

    # 3. Re-index new web facts into Chroma so future requests benefit too
    vector_store.add_facts(sport=sport, facts=[{"id": f["id"], "text": f["text"]} for f in web_facts_raw])

    all_context = vector_facts + web_facts_text

    # 4. Build prompt and call the LLM
    user_prompt = _build_user_prompt(sport, difficulty, num_questions, all_context, avoid_questions)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            max_output_tokens=4000,
        ),
    )

    raw_text = response.text
    if not raw_text:
        raise ValueError("Gemini returned an empty response (it may have been blocked by safety filters).")

    try:
        parsed = _extract_json(raw_text)
    except (json.JSONDecodeError, AttributeError) as e:
        raise ValueError(f"Model did not return valid JSON: {e}\nRaw output: {raw_text[:500]}")

    questions = parsed.get("questions", [])
    _validate_questions(questions)

    return {
        "sport": sport,
        "difficulty": difficulty,
        "questions": questions,
        "sources": {
            "vector_db_facts_used": len(vector_facts),
            "web_facts_used": len(web_facts_text),
        },
    }


def _validate_questions(questions: List[Dict]):
    """Basic structural validation so the API never returns malformed quiz data."""
    if not questions:
        raise ValueError("No questions were generated.")
    for q in questions:
        required_keys = {"question", "options", "correct_answer", "explanation"}
        if not required_keys.issubset(q.keys()):
            raise ValueError(f"Malformed question object, missing keys: {q}")
        if set(q["options"].keys()) != {"A", "B", "C", "D"}:
            raise ValueError(f"Question options must be exactly A/B/C/D: {q}")
        if q["correct_answer"] not in {"A", "B", "C", "D"}:
            raise ValueError(f"correct_answer must be one of A/B/C/D: {q}")
