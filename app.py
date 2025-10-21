import hmac
import logging
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from chatbot.gemini_bot import GeminiBot

# Align logging before heavy imports
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_gemini_microservice")
# Silence noisy third-party info logs that repeat on each load.
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chatbot.retriever").setLevel(logging.INFO)

# Suppress noisy gRPC logs before importing Google SDKs
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "")

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency guard
    def load_dotenv() -> None:
        return None

load_dotenv()

try:  # pragma: no cover - import guard for runtime environments
    import google.generativeai as genai  # type: ignore
    _HAS_GENAI = True
except Exception:  # pragma: no cover - optional dependency guard
    genai = None  # type: ignore
    _HAS_GENAI = False

APP_NAME = os.getenv("ASSISTANT_NAME", "La B\u00e0n AI")
MICROSERVICE_KEY = os.getenv("MICROSERVICE_INTERNAL_KEY") or os.getenv("MICROSERVICE_KEY")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "data/vector_store.pkl")
DOCS_DIR = Path(os.getenv("DOCS_DIRECTORY", "data/docs"))
MEMORY_DIR = Path(os.getenv("MEMORY_DIRECTORY", "data/memory"))
DEFAULT_MODEL = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.5-flash-lite")
DEFAULT_TEMPERATURE = float(os.getenv("GEMINI_DEFAULT_TEMPERATURE", "0.2"))
DEFAULT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "10"))

_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if _HAS_GENAI and _api_key:
    try:
        genai.configure(api_key=_api_key)
    except Exception as exc:  # pragma: no cover - configuration guard
        logger.error("Failed to configure google-generativeai client: %s", exc)
        raise
elif _HAS_GENAI:
    logger.warning("GEMINI_API_KEY not set; /query endpoint will fail until provided")

from chatbot.retriever import RAGRetriever  # noqa: E402

retriever_lock = threading.RLock()
retriever = RAGRetriever(vector_store_path=VECTOR_STORE_PATH)

_bot_cache: Dict[Tuple[str, str], GeminiBot] = {}
_bot_lock = threading.RLock()

app = FastAPI(title="RAG Gemini Microservice", version="1.0.0")


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    request_id = uuid4().hex
    start = time.perf_counter()
    logger.info("Request %s %s %s", request_id, request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Request %s failed", request_id)
        raise
    duration = time.perf_counter() - start
    logger.info(
        "Response %s %s status=%s duration=%.3fs",
        request_id,
        request.url.path,
        getattr(response, "status_code", "unknown"),
        duration,
    )
    return response


class ChatTurn(BaseModel):
    role: str = Field(..., max_length=32)
    content: str

    @field_validator("role")
    @classmethod
    def _role_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("role must not be empty")
        return value

    @field_validator("content")
    @classmethod
    def _content_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("content must not be empty")
        return value


class QueryPayload(BaseModel):
    query: str
    model: Optional[str] = None
    temperature: Optional[float] = None
    prompt_template: Optional[str] = None
    top_k: int = 3
    user_id: Optional[str] = None

    @field_validator("query")
    @classmethod
    def _query_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty")
        return value

    @field_validator("temperature")
    @classmethod
    def _temperature_bounds(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        if value < 0 or value > 2:
            raise ValueError("temperature must be between 0 and 2")
        return value

    @field_validator("top_k")
    @classmethod
    def _top_k_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("top_k must be positive")
        return value

    @field_validator("user_id")
    @classmethod
    def _user_id_normalized(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.strip()
        return value or None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []


def _require_key(x_internal_key: Optional[str] = Header(None)) -> None:
    if not MICROSERVICE_KEY:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Microservice key not configured")
    if not x_internal_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-Internal-Key header")
    if not hmac.compare_digest(x_internal_key, MICROSERVICE_KEY):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid microservice key")


def _sanitize_filename(filename: str) -> str:
    base = Path(filename).name
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return sanitized or f"document_{uuid4().hex}.pdf"


def _copy_source_document(source: Path, suggested_name: Optional[str]) -> Path:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    target_name = _sanitize_filename(suggested_name or source.name)
    target = DOCS_DIR / target_name
    if target.exists():
        target = DOCS_DIR / f"{target.stem}_{uuid4().hex[:8]}{target.suffix}"
    shutil.copy2(source, target)
    return target


def _store_uploaded_file(upload: UploadFile, suggested_name: Optional[str]) -> Path:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    original_name = suggested_name or upload.filename or f"document_{uuid4().hex}"
    target_name = _sanitize_filename(original_name)
    target = DOCS_DIR / target_name
    if target.exists():
        target = DOCS_DIR / f"{target.stem}_{uuid4().hex[:8]}{target.suffix}"
    with target.open("wb") as out_file:
        shutil.copyfileobj(upload.file, out_file)
    upload.file.seek(0)
    return target


def _memory_path(user_id: str) -> Path:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_filename(user_id) or "default"
    return MEMORY_DIR / f"{safe_name}.json"


def _get_bot(user_id: str, model_name: str) -> GeminiBot:
    cache_key = (user_id, model_name)
    with _bot_lock:
        bot = _bot_cache.get(cache_key)
        if bot is None:
            try:
                bot = GeminiBot(
                    name=APP_NAME,
                    model=model_name,
                    memory_file=str(_memory_path(user_id)),
                    retriever=retriever,
                    user_id=user_id,
                )
            except RuntimeError as exc:
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
            _bot_cache[cache_key] = bot
        return bot


@app.get("/healthz", dependencies=[Depends(_require_key)])
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/upload-doc", dependencies=[Depends(_require_key)])
def upload_document(
    file: UploadFile = File(..., description="PDF document to index"),
    filename: Optional[str] = Form(None, description="Optional override for stored filename"),
) -> JSONResponse:
    indexed_chunks = 0
    try:
        target = _store_uploaded_file(file, filename)
        with retriever_lock:
            before = len(retriever.docs)
            retriever.add_documents(str(target))
            indexed_chunks = max(0, len(retriever.docs) - before)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to index uploaded document")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to index document: {exc}")

    payload = {"stored_path": str(target), "indexed_chunks": indexed_chunks}
    return JSONResponse(content=payload)


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(_require_key)])
def run_query(payload: QueryPayload, request: Request) -> QueryResponse:
    user_id = (
        payload.user_id
        or request.headers.get("X-User-Id")
        or request.headers.get("X-WP-User")
        or request.headers.get("X-Forwarded-User")
        or (request.client.host if request.client else None)
        or "default"
    )
    model_name = payload.model or DEFAULT_MODEL
    bot = _get_bot(user_id, model_name)
    logger.info(
        "Running query user_id=%s model=%s top_k=%s query=%r",
        user_id,
        model_name,
        payload.top_k,
        payload.query,
    )
    try:
        with retriever_lock:
            context_chunks = list(bot.retriever.retrieve(payload.query, top_k=payload.top_k))
    except Exception as exc:
        logger.exception("Failed to retrieve documents for query")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve context: {exc}")

    temperature = payload.temperature if payload.temperature is not None else DEFAULT_TEMPERATURE

    try:
        result = bot.ask(
            payload.query,
            use_rag=True,
            history_turns=DEFAULT_HISTORY_TURNS,
            top_k=payload.top_k,
            temperature=temperature,
            prompt_template=payload.prompt_template,
            return_details=False,
            context_chunks=context_chunks,
            user_id=user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    return QueryResponse(
        answer=result,
        sources=[],
    )
