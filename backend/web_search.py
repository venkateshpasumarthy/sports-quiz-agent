"""
web_search.py
-------------
Lightweight web search wrapper used to pull in recent sports facts that
the static ChromaDB knowledge base won't have (this year's tournament
winners, latest transfers, recent records, etc).

Uses the `ddgs` package (DuckDuckGo search) which needs no API key,
making the project runnable out-of-the-box.
"""
import hashlib
from typing import List, Dict

from ddgs import DDGS


def search_recent_sports_info(sport: str, difficulty: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web for recent notable facts/events about a sport.
    Returns a list of {id, text, source} dicts suitable for both prompting
    the LLM and optionally re-indexing into ChromaDB.
    """
    query = f"latest {sport} records, champions, and notable events 2025 2026"

    facts = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                if not body:
                    continue
                fact_id = "web_" + hashlib.md5((title + body).encode()).hexdigest()[:10]
                facts.append({
                    "id": fact_id,
                    "text": f"{title}: {body}",
                    "source": href,
                })
    except Exception as e:
        # Web search is a "nice to have" freshness layer -- if it fails
        # (e.g. no network), we fall back gracefully to ChromaDB-only context.
        print(f"[web_search] search failed: {e}")

    return facts
