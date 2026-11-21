"""FastAPI service: same-origin frontend plus a small, documented prediction API.

Endpoints (contract in ``docs/API.md``):

* ``GET  /health``   liveness: the process is up. Never touches the model.
* ``GET  /ready``    readiness: the model artifact loaded and verified (503 otherwise).
* ``GET  /model``    model card summary for clients (version, threshold, limits, metrics).
* ``POST /predict``  classify one SMS.

Runtime dependencies are FastAPI/Pydantic only. The model is a verified JSON
artifact executed by :mod:`spam_detector.inference` (no scikit-learn, NumPy,
NLTK, or pickle), which keeps the serverless bundle and cold start small.

Privacy: message text is never logged. Validation errors are returned without
echoing the rejected input.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from spam_detector.artifact import ArtifactError, LoadedArtifact
from spam_detector.inference import SpamClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "spam-v2"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
MAX_MESSAGE_CHARS = 5000
# 5,000 code points, each at most 12 bytes when JSON-escaped as a surrogate pair
# ("🙂"), plus envelope. Bodies larger than this are rejected with 413
# before JSON parsing, so a huge payload cannot consume memory or CPU.
MAX_BODY_BYTES = 64 * 1024

logger = logging.getLogger("spam_detector.api")
# Client-supplied request IDs are echoed in a response header and written to logs, so
# only short, plain tokens are accepted; anything else is replaced by a generated ID.
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")


def _peak_rss_mib() -> float | None:
    """Peak resident memory of this process (Linux/macOS; ``None`` on Windows).

    Vercel's log records carry no memory figure, so the app reports its own.
    """
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    return round(peak / (2**20 if sys.platform == "darwin" else 2**10), 1)


# ------------------------------------------------------------------------ logging
class JsonLogFormatter(logging.Formatter):
    """One JSON object per line: easy to query in Vercel's runtime logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {"level": record.levelname.lower(), "event": record.getMessage()}
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _configure_logging() -> None:
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonLogFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False


def _log(event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={"fields": fields})


# ------------------------------------------------------------------------ schemas
class PredictRequest(BaseModel):
    text: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_MESSAGE_CHARS,
            description="SMS text, 1-5,000 Unicode code points, not only whitespace.",
            examples=["Congratulations! You won a free prize. Call now to claim."],
        ),
    ]

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("Enter an SMS message.")
        return text


class PredictResponse(BaseModel):
    label: Literal["spam", "ham", "inconclusive"] = Field(
        description="spam / ham, or inconclusive when the model abstains (see `reason`)."
    )
    spam_probability: float | None = Field(
        description=(
            "Calibrated estimate of P(spam) on data resembling the training corpus, rounded to 4 decimals. "
            "`null` when inconclusive. Not a guarantee that a message is safe."
        )
    )
    reason: Literal["no_letters", "unsupported_script", "no_recognized_features"] | None = Field(
        default=None, description="Why the prediction is inconclusive; `null` otherwise."
    )
    model_version: str


class ErrorDetail(BaseModel):
    type: str
    msg: str


# -------------------------------------------------------------------------- state
@dataclass
class ModelState:
    classifier: SpamClassifier | None = None
    loaded: LoadedArtifact | None = None
    error: str | None = None
    load_seconds: float | None = None


def _load_model(directory: Path) -> ModelState:
    start = time.perf_counter()
    try:
        classifier, loaded = SpamClassifier.load(directory)
    except ArtifactError as error:
        _log("model_load_failed", logging.ERROR, model_dir=str(directory), error=str(error))
        return ModelState(error=str(error))
    state = ModelState(classifier, loaded, None, time.perf_counter() - start)
    _log(
        "model_loaded",
        model_version=loaded.artifact.model_version,
        sha256=loaded.sha256,
        features=loaded.artifact.feature_space.size,
        load_ms=round(state.load_seconds * 1000, 1),
        peak_rss_mib=_peak_rss_mib(),
    )
    return state


# ---------------------------------------------------------------------- middleware
class BodySizeLimitMiddleware:
    """Reject request bodies over ``max_bytes`` with 413, including chunked uploads."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self.max_bytes):
            await self._reject(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body += message.get("body", b"")
            if len(body) > self.max_bytes:
                await self._reject(scope, receive, send)
                return
            if not message.get("more_body", False):
                break
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            {"detail": {"type": "payload_too_large", "msg": f"Request body exceeds {self.max_bytes} bytes."}},
            status_code=413,
        )
        await response(scope, receive, send)


_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}
_FRONTEND_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
)


# ----------------------------------------------------------------------- the app
def create_app(model_dir: Path | None = None) -> FastAPI:
    """Application factory. ``model_dir`` defaults to ``$SPAM_MODEL_DIR`` or ``models/spam-v2``."""
    _configure_logging()
    directory = model_dir or Path(os.environ.get("SPAM_MODEL_DIR", DEFAULT_MODEL_DIR))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load once per process (per serverless instance), before serving traffic.
        # A broken artifact does not crash the process: /ready and /predict report
        # 503 with a reason, which is far easier to diagnose than a crash loop.
        app.state.model = _load_model(directory)
        yield

    app = FastAPI(
        title="SMS Spam Detection API",
        version="2.0.0",
        summary="Calibrated English SMS spam classification with abstention.",
        lifespan=lifespan,
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        supplied = request.headers.get("x-request-id", "")
        request_id = supplied if _REQUEST_ID_RE.fullmatch(supplied) else uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            _log("request_failed", logging.ERROR, request_id=request_id, path=request.url.path, exc_info=True)
            response = JSONResponse(
                {"detail": {"type": "internal_error", "msg": "The prediction service had a problem."}},
                status_code=500,
            )
        response.headers["X-Request-ID"] = request_id
        response.headers.update(_SECURITY_HEADERS)
        if not request.url.path.startswith(("/docs", "/redoc", "/openapi")):
            response.headers["Content-Security-Policy"] = _FRONTEND_CSP
        if request.url.path in {"/predict", "/ready", "/model", "/health"}:
            response.headers["Cache-Control"] = "no-store"
            _log(
                "request",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
                peak_rss_mib=_peak_rss_mib(),
                **getattr(request.state, "log_fields", {}),
            )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
        # FastAPI's default 422 body echoes the rejected input back ("input"),
        # which may be a private message. Return only type, location and message.
        detail = [
            {"type": e["type"], "loc": list(e["loc"]), "msg": e["msg"]} for e in error.errors()
        ]
        return JSONResponse({"detail": detail}, status_code=422)

    def model_state(request: Request) -> ModelState:
        return request.app.state.model

    def unavailable() -> JSONResponse:
        # A fresh object per request: middleware mutates response headers.
        return JSONResponse(
            {"detail": {"type": "model_unavailable", "msg": "The prediction model is not available."}},
            status_code=503,
        )

    @app.get("/health", tags=["operations"])
    def health() -> dict[str, str]:
        """Liveness probe. Succeeds whenever the process can serve HTTP."""
        return {"status": "ok"}

    @app.get("/ready", tags=["operations"], responses={503: {"model": ErrorDetail}})
    def ready(request: Request) -> Any:
        """Readiness probe. 200 only when a verified model artifact is loaded."""
        state = model_state(request)
        if state.loaded is None:
            return unavailable()
        return {"status": "ready", "model_version": state.loaded.artifact.model_version,
                "model_sha256": state.loaded.sha256}

    @app.get("/model", tags=["model"], responses={503: {"model": ErrorDetail}})
    def model_info(request: Request) -> Any:
        """What the frontend needs to describe the model honestly."""
        state = model_state(request)
        if state.loaded is None:
            return unavailable()
        artifact, metadata = state.loaded.artifact, state.loaded.metadata
        return {
            "model_version": artifact.model_version,
            "model_sha256": state.loaded.sha256,
            "estimator": artifact.estimator,
            "labels": ["spam", "ham", "inconclusive"],
            "decision_threshold": artifact.threshold,
            "max_message_chars": MAX_MESSAGE_CHARS,
            "language": "English SMS",
            "intended_use": metadata.get("intended_use"),
            "evaluation": metadata.get("evaluation_summary"),
        }

    @app.post(
        "/predict",
        response_model=PredictResponse,
        tags=["model"],
        responses={413: {"model": ErrorDetail}, 503: {"model": ErrorDetail}},
    )
    def predict(body: PredictRequest, request: Request) -> Any:
        """Classify one SMS message.

        Returns ``inconclusive`` (with ``spam_probability: null``) when the text
        has no letters, is mostly in a non-Latin script, or contains no n-gram
        the model learned. CPU cost is linear in the text length and bounded by
        the 5,000-character limit (about a millisecond for a typical SMS).
        """
        state = model_state(request)
        if state.classifier is None or state.loaded is None:
            return unavailable()
        result = state.classifier.predict(body.text)
        # Metadata only; never the message itself.
        request.state.log_fields = {"label": result.label, "reason": result.reason, "chars": len(body.text)}
        probability = None if result.spam_probability is None else round(result.spam_probability, 4)
        return PredictResponse(
            label=result.label,
            spam_probability=probability,
            reason=result.reason,
            model_version=state.loaded.artifact.model_version,
        )

    @app.api_route("/predict", methods=["GET", "HEAD", "PUT", "PATCH", "DELETE"], include_in_schema=False)
    def predict_wrong_method() -> JSONResponse:
        # Without this, the static-files mount at "/" fully matches GET /predict and
        # answers 404, hiding the fact that the endpoint exists but needs POST.
        return JSONResponse(
            {"detail": {"type": "method_not_allowed", "msg": "Use POST with a JSON body."}},
            status_code=405,
            headers={"Allow": "POST"},
        )

    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()
