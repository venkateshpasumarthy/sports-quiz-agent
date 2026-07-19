"""
vector_store.py
----------------
Wraps ChromaDB: seeds a persistent collection with sports knowledge and
exposes a simple retrieve(sport, query, n_results) method used by the
quiz agent for Retrieval-Augmented Generation.

ChromaDB uses its own default sentence-embedding model under the hood
(all-MiniLM-L6-v2) so no external embedding API key is required.
"""
import json
import os
from typing import List, Dict

import chromadb

from backend.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME

SEED_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "seed_knowledge.json")


class SportsVectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._seed_if_empty()

    def _seed_if_empty(self):
        """Load the seed knowledge base into Chroma the first time the app runs."""
        if self.collection.count() > 0:
            return

        with open(SEED_FILE, "r") as f:
            records = json.load(f)

        self.collection.add(
            ids=[r["id"] for r in records],
            documents=[r["text"] for r in records],
            metadatas=[{"sport": r["sport"]} for r in records],
        )
        print(f"[vector_store] Seeded ChromaDB with {len(records)} sports facts.")

    def retrieve(self, sport: str, query: str, n_results: int = 5) -> List[str]:
        """Retrieve the most relevant stored facts for a sport + topical query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"sport": sport},
        )
        docs = results.get("documents", [[]])[0]
        return docs

    def add_facts(self, sport: str, facts: List[Dict[str, str]]):
        """
        Add freshly retrieved web-search facts back into the vector store so
        future quiz generations for the same sport benefit from them too.
        Each fact dict needs an 'id' and 'text' key.
        """
        if not facts:
            return
        self.collection.add(
            ids=[f["id"] for f in facts],
            documents=[f["text"] for f in facts],
            metadatas=[{"sport": sport} for _ in facts],
        )

    def count(self) -> int:
        return self.collection.count()


# Singleton instance used across the app
vector_store = SportsVectorStore()
