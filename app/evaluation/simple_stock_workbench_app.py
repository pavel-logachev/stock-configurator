from __future__ import annotations

import html
import secrets
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.simple_stock_case_exporter import CaseExportError
from app.evaluation.simple_stock_workbench import LocalCaseStore, ReviewMutation

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "testserver"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


class FinalizeMutation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm_case_id: str = Field(min_length=1, max_length=128)


def create_workbench_app(
    *,
    case_root: Path,
    dataset_path: Path,
    csrf_token: str | None = None,
) -> FastAPI:
    static_root = Path(__file__).with_name("workbench_static").resolve()
    index_template = (static_root / "index.html").read_text(encoding="utf-8")
    token = csrf_token or secrets.token_urlsafe(32)
    store = LocalCaseStore(case_root=case_root, dataset_path=dataset_path)

    app = FastAPI(
        title="Simple Stock Evaluation Workbench",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.workbench_store = store
    app.state.workbench_csrf_token = token
    app.mount("/assets", StaticFiles(directory=static_root), name="workbench-assets")

    @app.middleware("http")
    async def security_boundary(request: Request, call_next):  # type: ignore[no-untyped-def]
        host = _hostname(request.headers.get("host", ""))
        if host not in ALLOWED_HOSTS:
            return _security_error("security.host_denied")
        if request.method not in SAFE_METHODS:
            if request.headers.get("sec-fetch-site", "").lower() == "cross-site":
                return _security_error("security.cross_site_denied")
            origin = request.headers.get("origin")
            if origin and _hostname(urlsplit(origin).netloc) not in ALLOWED_HOSTS:
                return _security_error("security.origin_denied")
            supplied = request.headers.get("x-workbench-csrf", "")
            if not secrets.compare_digest(supplied, token):
                return _security_error("security.csrf_denied")
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; font-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()"
        )
        return response

    @app.exception_handler(CaseExportError)
    async def case_error_handler(_request: Request, exc: CaseExportError) -> JSONResponse:
        status_code = _status_for_error(exc.code)
        return JSONResponse(
            status_code=status_code,
            content={"status": "blocked", "error": exc.code, "details": exc.details},
        )

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        rendered = index_template.replace("__WORKBENCH_CSRF__", html.escape(token, quote=True))
        return HTMLResponse(rendered)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/summary")
    def summary() -> dict[str, object]:
        return store.quality_summary().model_dump(mode="json")

    @app.get("/api/cases")
    def cases() -> dict[str, object]:
        return {"cases": store.list_cases(), "max_cases": 500}

    @app.get("/api/cases/{case_id}")
    def case_detail(case_id: str) -> dict[str, object]:
        return store.case_detail(case_id)

    @app.post("/api/cases/{case_id}/review")
    def save_review(case_id: str, mutation: ReviewMutation) -> dict[str, object]:
        draft = store.save_review(case_id, mutation)
        return {
            "status": "saved",
            "case_id": case_id,
            "review": draft.model_dump(mode="json"),
        }

    @app.post("/api/cases/{case_id}/finalize")
    def finalize_review(case_id: str, mutation: FinalizeMutation) -> dict[str, object]:
        receipt = store.finalize_review(case_id, confirm_case_id=mutation.confirm_case_id)
        return {"status": "finalized", "case_id": case_id, "receipt": receipt}

    return app


def _security_error(code: str) -> JSONResponse:
    return JSONResponse(status_code=403, content={"status": "blocked", "error": code})


def _hostname(value: str) -> str | None:
    if not value:
        return None
    return urlsplit(f"//{value}").hostname


def _status_for_error(code: str) -> int:
    if code in {"case.not_found", "input.file_not_found"}:
        return 404
    if code in {"review.already_finalized", "review.output_exists"}:
        return 409
    if code.startswith("security."):
        return 403
    return 422


__all__ = ["create_workbench_app"]
