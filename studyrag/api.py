"""HTTP API. Thin: every decision lives in retrieve.py and modes/.

The browser never talks to Postgres — it talks here, and this holds the only
database credential. That topology is why the tables have RLS enabled with zero
policies: the anon key path is closed, and this backend connects as a role that
bypasses it.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from psycopg import Connection
from pydantic import BaseModel, Field

from studyrag.db.writer import connect
from studyrag.modes.ask import Answer, answer_question
from studyrag.modes.quiz import MAX_QUESTIONS, Quiz, generate_quiz

log = logging.getLogger(__name__)
WEB = Path(__file__).parent / "web"

_conn: Connection | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the connection once, and warm the models before the first request.

    Both models load lazily on first use, which would otherwise put ~10 s of model
    loading into whichever unlucky request arrives first.
    """
    global _conn
    _conn = connect()

    from studyrag.embed import embed_query
    from studyrag.rerank import score

    embed_query("warmup")
    score("warmup", ["warmup"])
    log.info("models warm, database connected")

    yield
    _conn.close()


app = FastAPI(title="StudyRAG", lifespan=lifespan)


def _db() -> Connection:
    """The lifespan connection, or a loud 503 if startup never completed."""
    if _conn is None:
        raise HTTPException(status_code=503, detail="database connection not ready")
    return _conn


class AskRequest(BaseModel):
    course: str = Field(min_length=1)
    question: str = Field(min_length=3, max_length=500)
    # None means the whole course. Narrowing to one lecture is the student's call,
    # not something to infer from the question.
    lecture: str | None = None


@app.get("/api/courses")
def courses() -> list[dict]:
    """Courses with material ingested, and the files behind each one.

    The file list is shown in the UI so a user can see what the answers can possibly be
    drawn from. "Not in your material" is only a useful answer if you know what your
    material is.
    """
    rows = _db().execute(
        """
        SELECT d.course, d.doc_path, d.lecture, count(c.id) AS chunks
        FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
        GROUP BY d.course, d.doc_path, d.lecture
        ORDER BY d.course, d.lecture NULLS LAST, d.doc_path
        """
    ).fetchall()

    by_course: dict[str, dict] = {}
    for course, doc_path, lecture, chunks in rows:
        entry = by_course.setdefault(course, {"course": course, "chunks": 0, "documents": []})
        entry["chunks"] += chunks
        entry["documents"].append({"doc_path": doc_path, "lecture": lecture, "chunks": chunks})
    return list(by_course.values())


@app.post("/api/ask")
def ask(request: AskRequest) -> Answer:
    """Ask mode. `course` is required — retrieval is never unscoped."""
    try:
        return answer_question(_db(), request.course, request.question, request.lecture)
    except Exception as exc:
        # The message can carry a connection string or a provider error body, so it
        # goes to the log and the caller gets nothing but the status.
        log.exception("ask failed")
        raise HTTPException(status_code=502, detail="generation failed") from exc


class QuizRequest(BaseModel):
    course: str = Field(min_length=1)
    lecture: str | None = None
    n_questions: int = Field(default=10, ge=1, le=MAX_QUESTIONS)


@app.post("/api/quiz")
def quiz(request: QuizRequest) -> Quiz:
    """Quiz mode. Scope is metadata, not similarity: no query is embedded."""
    try:
        return generate_quiz(_db(), request.course, request.lecture, request.n_questions)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("quiz failed")
        raise HTTPException(status_code=502, detail="generation failed") from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")
