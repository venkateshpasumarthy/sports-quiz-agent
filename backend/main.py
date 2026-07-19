"""
main.py
-------
FastAPI application exposing the Sports Quiz Agent as a REST API, and
serving the static frontend dashboard.
"""
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import SUPPORTED_SPORTS, DIFFICULTY_LEVELS
from backend.quiz_agent import generate_quiz
from backend.vector_store import vector_store

app = FastAPI(title="Sports Quiz Generation Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuizRequest(BaseModel):
    sport: str = Field(..., description="One of the supported sports")
    difficulty: str = Field(..., description="Easy, Medium, or Hard")
    num_questions: int = Field(5, ge=4, le=5)
    avoid_questions: Optional[List[str]] = Field(
        default=None, description="Previously seen questions to avoid repeating (used on Regenerate)"
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "vector_db_facts": vector_store.count()}


@app.get("/api/meta")
def meta():
    return {"sports": SUPPORTED_SPORTS, "difficulties": DIFFICULTY_LEVELS}


@app.post("/api/generate-quiz")
def generate(req: QuizRequest):
    if req.sport not in SUPPORTED_SPORTS:
        raise HTTPException(status_code=400, detail=f"Unsupported sport. Choose from {SUPPORTED_SPORTS}")
    if req.difficulty not in DIFFICULTY_LEVELS:
        raise HTTPException(status_code=400, detail=f"Unsupported difficulty. Choose from {DIFFICULTY_LEVELS}")

    try:
        result = generate_quiz(
            sport=req.sport,
            difficulty=req.difficulty,
            num_questions=req.num_questions,
            avoid_questions=req.avoid_questions,
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"Generation error: {e}")


# Serve the frontend dashboard at /
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
