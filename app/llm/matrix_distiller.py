from __future__ import annotations

import json
import queue
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from app.llm.base import LlmClient, LlmError, LlmHttpError, LlmInvalidJsonError

FIT_TIER_STRONG = "strong_fit"
FIT_TIER_POSSIBLE = "possible_fit"
FIT_TIER_FALLBACK_UNKNOWN = "fallback_unknown"
FIT_TIER_EXPLICIT_MISMATCH = "explicit_mismatch"
FIT_TIER_WRONG_ROLE = "wrong_role"
SELECTABLE_FIT_TIERS = {
    FIT_TIER_STRONG,
    FIT_TIER_POSSIBLE,
    FIT_TIER_FALLBACK_UNKNOWN,
}
FIT_TIER_RANK = {
    FIT_TIER_STRONG: 0,
    FIT_TIER_POSSIBLE: 1,
    FIT_TIER_FALLBACK_UNKNOWN: 2,
    FIT_TIER_EXPLICIT_MISMATCH: 98,
    FIT_TIER_WRONG_ROLE: 99,
}

DEFAULT_CHUNK_MAX_CHARS = 30000
DEFAULT_FALLBACK_UNKNOWN_LIMIT = 6
DEFAULT_ROLE_CANDIDATE_POOL_LIMIT = 48
DEFAULT_FULL_MATRIX_MAX_SECONDS = 900.0
DEFAULT_FULL_MATRIX_CHUNK_TIMEOUT_SECONDS = 300.0
READY_SERVER_MATRIX_KEY = "ready_server_candidates"
SERVER_ROLE_LIMITS = {
    "server_platform": 36,
    "platform": 36,
    "cpu": 48,
    "ram": 48,
    "storage": 48,
    "drive": 48,
    "ssd": 48,
    "hdd": 48,
    "storage_controller": 32,
    "network_adapter": 48,
    "power_supply": 24,
    "cable": 24,
    "other_accessory": 24,
}

SERVER_MATRIX_KEYS = {
    "server_platform": "platform_candidates",
    "cpu": "cpu_candidates",
    "ram": "ram_candidates",
    "drive": "drive_candidates",
    "ssd": "ssd_candidates",
    "hdd": "hdd_candidates",
    "storage_controller": "storage_controller_candidates",
    "network_adapter": "network_adapter_candidates",
    "power_supply": "power_supply_candidates",
    "cable": "cable_candidates",
    "other_accessory": "other_accessory_candidates",
    "gpu": "gpu_candidates",
    "transceiver": "transceiver_candidates",
    "rail_kit": "rail_kit_candidates",
    "license": "license_candidates",
    "support": "support_candidates",
}

MatrixDistillerProgressCallback = Callable[[str, Mapping[str, Any]], None]

MATRIX_DISTILLER_SYSTEM_PROMPT = """
You are AI Matrix Distiller / AI Role Evaluator for stock configurator candidate matrices.

Return only strict JSON in this shape:
{
  "role": "...",
  "evaluated_candidates": [
    {
      "component_candidate_id": "...",
      "fit_tier": "strong_fit|possible_fit|fallback_unknown|explicit_mismatch|wrong_role",
      "facts": {},
      "matched_constraints": [],
      "missing_facts": [],
      "mismatch_reasons": [],
      "price_stock_notes": [],
      "compatibility_assumptions": [],
      "engineer_checks": [],
      "confidence": "high|medium|low"
    }
  ],
  "shortlist_candidate_ids": {
    "strong_fit": [],
    "possible_fit": [],
    "fallback_unknown": []
  }
}

Rules:
- Return only component_candidate_id values from the input candidates.
- Do not invent product IDs, component IDs, categories, prices, stock, or quantities.
- Do not choose a final BOM or optimize a complete configuration.
- Evaluate each candidate only against the supplied role constraints and candidate facts.
- Evaluate every candidate in the chunk. If facts are incomplete, use fallback_unknown
  and describe missing_facts/engineer_checks instead of discarding the candidate.
- explicit_mismatch and wrong_role are diagnostics only, not selectable.
- Use fallback_unknown when the role looks plausible but facts are incomplete.
""".strip()


MATRIX_ROLE_REDUCER_SYSTEM_PROMPT = """
You are AI Matrix Distiller / AI Role Reducer for stock configurator candidate matrices.

Return only strict JSON in this shape:
{
  "role": "...",
  "selected_candidate_ids": [],
  "role_summary": "...",
  "no_viable_reason": null,
  "rejected_summary": [
    {"fit_tier": "explicit_mismatch", "count": 3, "top_reasons": []}
  ]
}

Rules:
- You receive evaluated_candidates from every chunk for one role. Reduce across the
  whole role, not one chunk.
- Select a role candidate pool, not a final BOM.
- Preserve commercial breadth: best technical matches, cheapest acceptable candidates,
  high-stock candidates, exact model/token matches, good equivalents, useful fallback
  unknowns, and diversity by vendor/form-factor/category when available.
- Return only IDs from the input evaluated_candidates.
- Do not invent product IDs, prices, stock, quantities, compatibility, or facts.
- Do not include explicit_mismatch or wrong_role in selected_candidate_ids unless no
  selectable candidates exist and you are only documenting a diagnostic fallback.
""".strip()


class MatrixDistillerError(LlmError):
    """Raised when the matrix distiller cannot safely return a usable shortlist."""

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        role: str | None = None,
        chunk_index: int | None = None,
        cause_error_type: str | None = None,
        parse_status: str | None = None,
        http_status: int | None = None,
        timeout_kind: str | None = None,
        timeout_seconds: float | None = None,
        deadline_seconds: float | None = None,
        elapsed_seconds: float | None = None,
        failed_chunks: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.role = role
        self.chunk_index = chunk_index
        self.cause_error_type = cause_error_type
        self.parse_status = parse_status
        self.http_status = http_status
        self.timeout_kind = timeout_kind
        self.timeout_seconds = timeout_seconds
        self.deadline_seconds = deadline_seconds
        self.elapsed_seconds = elapsed_seconds
        self.failed_chunks = [dict(row) for row in failed_chunks or []]


class MatrixDistillerTimeoutError(MatrixDistillerError):
    """Raised when full-matrix evaluation exceeds a bounded call or stage timeout."""


@dataclass
class _FullMatrixDeadline:
    max_seconds: float = DEFAULT_FULL_MATRIX_MAX_SECONDS
    chunk_timeout_seconds: float = DEFAULT_FULL_MATRIX_CHUNK_TIMEOUT_SECONDS
    progress_callback: MatrixDistillerProgressCallback | None = None
    started_at: float = field(default_factory=time.monotonic)

    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    def remaining(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed())

    def call_timeout(self) -> float:
        return max(0.0, min(self.chunk_timeout_seconds, self.remaining()))

    def check(self, *, stage: str, role: str | None = None, chunk_index: int | None = None) -> None:
        remaining = self.remaining()
        if remaining > 0:
            return
        raise MatrixDistillerTimeoutError(
            "full_matrix_evaluation_deadline_exceeded",
            stage=stage,
            role=role,
            chunk_index=chunk_index,
            cause_error_type="stage_deadline",
            timeout_kind="stage_deadline",
            timeout_seconds=self.max_seconds,
            deadline_seconds=self.max_seconds,
            elapsed_seconds=self.elapsed(),
        )


@dataclass(frozen=True)
class DistilledRoleResult:
    role: str
    rows: list[dict[str, Any]]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _RoleReducerResult:
    selected_ids: list[str]
    summary: dict[str, Any]
    prompt_chars: int = 0
    response_chars: int = 0
    llm_calls_count: int = 0


@dataclass(frozen=True)
class MatrixDistillationResult:
    component_candidate_matrix: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _emit_progress(
    callback: MatrixDistillerProgressCallback | None,
    event: str,
    **fields: Any,
) -> None:
    if callback is None:
        return
    try:
        callback(event, {key: value for key, value in fields.items() if value is not None})
    except Exception:
        return


def _generate_json_with_timeout(
    llm_client: LlmClient,
    system_prompt: str,
    user_prompt: str,
    *,
    timeout_seconds: float,
    deadline_seconds: float,
    stage: str,
    role: str | None = None,
    chunk_index: int | None = None,
) -> Any:
    timeout_seconds = float(timeout_seconds or 0)
    if timeout_seconds <= 0:
        raise MatrixDistillerTimeoutError(
            "full_matrix_evaluation_deadline_exceeded",
            stage=stage,
            role=role,
            chunk_index=chunk_index,
            cause_error_type="stage_deadline",
            timeout_kind="stage_deadline",
            timeout_seconds=deadline_seconds,
            deadline_seconds=deadline_seconds,
        )

    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def _target() -> None:
        try:
            result_queue.put_nowait(
                ("result", llm_client.generate_json(system_prompt, user_prompt))
            )
        except Exception as exc:
            try:
                result_queue.put_nowait(("error", exc))
            except queue.Full:
                return

    worker = threading.Thread(target=_target, name=f"matrix-distiller-{stage}", daemon=True)
    started_at = time.monotonic()
    worker.start()
    worker.join(timeout_seconds)
    elapsed_seconds = max(0.0, time.monotonic() - started_at)
    if worker.is_alive():
        raise MatrixDistillerTimeoutError(
            "full_matrix_llm_call_timed_out",
            stage=stage,
            role=role,
            chunk_index=chunk_index,
            cause_error_type="chunk_timeout",
            timeout_kind="chunk_timeout",
            timeout_seconds=timeout_seconds,
            deadline_seconds=deadline_seconds,
            elapsed_seconds=elapsed_seconds,
        )

    try:
        kind, value = result_queue.get_nowait()
    except queue.Empty as exc:
        raise MatrixDistillerError(
            "matrix_distiller_llm_call_finished_without_result",
            stage=stage,
            role=role,
            chunk_index=chunk_index,
            cause_error_type="missing_thread_result",
        ) from exc
    if kind == "error":
        if isinstance(value, LlmError):
            raise value
        raise MatrixDistillerError(
            str(value) or type(value).__name__,
            stage=stage,
            role=role,
            chunk_index=chunk_index,
            cause_error_type=type(value).__name__,
        ) from value
    return value


def distill_component_candidate_matrix(
    *,
    product_group: str,
    component_candidate_matrix: Mapping[str, Any],
    constraints_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    llm_client: LlmClient,
    role_limits: Mapping[str, int] | None = None,
    chunk_max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    max_seconds: float = DEFAULT_FULL_MATRIX_MAX_SECONDS,
    chunk_timeout_seconds: float = DEFAULT_FULL_MATRIX_CHUNK_TIMEOUT_SECONDS,
    progress_callback: MatrixDistillerProgressCallback | None = None,
) -> MatrixDistillationResult:
    if product_group != "server":
        return MatrixDistillationResult(
            component_candidate_matrix=dict(component_candidate_matrix),
            diagnostics={
                "matrix_distiller_source": "skipped",
                "reason": "non_server_product_group",
            },
        )

    limits = dict(SERVER_ROLE_LIMITS)
    if role_limits:
        limits.update({str(role): max(1, int(limit)) for role, limit in role_limits.items()})

    deadline = _FullMatrixDeadline(
        max_seconds=max(0.001, float(max_seconds or DEFAULT_FULL_MATRIX_MAX_SECONDS)),
        chunk_timeout_seconds=max(
            0.001,
            float(chunk_timeout_seconds or DEFAULT_FULL_MATRIX_CHUNK_TIMEOUT_SECONDS),
        ),
        progress_callback=progress_callback,
    )
    distilled_matrix = dict(component_candidate_matrix)
    broad_count_by_role = _count_by_role(component_candidate_matrix)
    distilled_count_by_role: dict[str, int] = {}
    role_diagnostics: dict[str, Any] = {}
    role_chunk_count_by_role: dict[str, int] = {}
    evaluated_candidate_count_by_role: dict[str, int] = {}
    selected_candidate_count_by_role: dict[str, int] = {}
    role_reducer_summary: dict[str, Any] = {}
    full_matrix_failed_chunks: list[dict[str, Any]] = []
    llm_calls_count = 0
    approximate_tokens = 0
    evaluated_any = False
    saw_candidate_rows = False

    for role, matrix_key in SERVER_MATRIX_KEYS.items():
        deadline.check(stage="role_evaluator", role=role)
        rows = _mapping_rows(component_candidate_matrix.get(matrix_key))
        if not rows:
            continue
        saw_candidate_rows = True
        role_constraints = _constraints_for_role(constraints_by_role, role)
        result = distill_role_candidates(
            product_group=product_group,
            role=role,
            constraints=role_constraints,
            candidate_rows=rows,
            llm_client=llm_client,
            role_limit=limits.get(role, 8),
            chunk_max_chars=chunk_max_chars,
            deadline=deadline,
            progress_callback=progress_callback,
        )
        distilled_matrix[matrix_key] = result.rows
        distilled_count_by_role[role] = len(result.rows)
        role_diagnostics[role] = result.diagnostics
        role_chunk_count_by_role[role] = int(result.diagnostics.get("chunk_count") or 0)
        evaluated_candidate_count_by_role[role] = int(
            result.diagnostics.get("evaluated_count") or 0
        )
        selected_candidate_count_by_role[role] = len(result.rows)
        role_reducer_summary[role] = dict(
            result.diagnostics.get("role_reducer_summary") or {}
        )
        full_matrix_failed_chunks.extend(
            _mapping_rows(result.diagnostics.get("failed_chunks"))
        )
        cost = _safe_mapping(result.diagnostics.get("llm_cost_diagnostics"))
        llm_calls_count += int(cost.get("llm_calls_count") or 0)
        approximate_tokens += int(cost.get("approximate_tokens") or 0)
        if int(result.diagnostics.get("evaluated_count") or 0) > 0:
            evaluated_any = True

    if not evaluated_any:
        if saw_candidate_rows and full_matrix_failed_chunks:
            first_failure = full_matrix_failed_chunks[0]
            error_cls: type[MatrixDistillerError] = MatrixDistillerError
            timeout_kind = str(first_failure.get("timeout_kind") or "").strip()
            if timeout_kind:
                error_cls = MatrixDistillerTimeoutError
            raise error_cls(
                "matrix_distiller_all_role_evaluator_chunks_failed",
                stage="role_evaluator",
                role=str(first_failure.get("role") or ""),
                chunk_index=int(first_failure.get("chunk_index") or 0),
                cause_error_type=str(
                    first_failure.get("cause_error_type")
                    or first_failure.get("error_type")
                    or ""
                ),
                parse_status=str(first_failure.get("parse_status") or ""),
                http_status=_int_value(first_failure.get("http_status")),
                timeout_kind=timeout_kind or None,
                timeout_seconds=_float_value(first_failure.get("timeout_seconds")),
                deadline_seconds=deadline.max_seconds,
                elapsed_seconds=deadline.elapsed(),
                failed_chunks=full_matrix_failed_chunks,
            )
        return MatrixDistillationResult(
            component_candidate_matrix=distilled_matrix,
            diagnostics={
                "matrix_distiller_source": "skipped",
                "reason": "no_server_candidate_rows",
                "broad_count_by_role": broad_count_by_role,
                "distilled_count_by_role": {},
            },
        )

    if isinstance(distilled_matrix.get(READY_SERVER_MATRIX_KEY), list):
        ready_server_count = broad_count_by_role.get("ready_server", 0)
        distilled_matrix[READY_SERVER_MATRIX_KEY] = []
        distilled_count_by_role["ready_server"] = 0
        role_diagnostics["ready_server"] = {
            "candidate_count": ready_server_count,
            "selected_count": 0,
            "reason": "component_bom_distillation_only",
        }

    diagnostics = {
        "matrix_distiller_source": "llm",
        "full_matrix_evaluation_used": True,
        "broad_count_by_role": broad_count_by_role,
        "distilled_count_by_role": distilled_count_by_role,
        "role_chunk_count_by_role": role_chunk_count_by_role,
        "evaluated_candidate_count_by_role": evaluated_candidate_count_by_role,
        "selected_candidate_count_by_role": selected_candidate_count_by_role,
        "role_reducer_summary": role_reducer_summary,
        "full_matrix_failed_chunks": full_matrix_failed_chunks,
        "no_recommendation_coverage": _coverage_diagnostics(
            considered_count_by_role=evaluated_candidate_count_by_role,
            matrix_count_by_role=broad_count_by_role,
        ),
        "llm_cost_diagnostics": {
            "llm_calls_count": llm_calls_count,
            "approximate_tokens": approximate_tokens,
        },
        "role_diagnostics": role_diagnostics,
    }
    distilled_matrix.update(
        {
            "matrix_distiller_used": True,
            "matrix_distiller_source": "llm",
            "matrix_distiller_diagnostics": diagnostics,
            "full_matrix_evaluation_used": True,
            "role_chunk_count_by_role": role_chunk_count_by_role,
            "evaluated_candidate_count_by_role": evaluated_candidate_count_by_role,
            "selected_candidate_count_by_role": selected_candidate_count_by_role,
            "role_reducer_summary": role_reducer_summary,
            "full_matrix_failed_chunks": full_matrix_failed_chunks,
            "no_recommendation_coverage": diagnostics["no_recommendation_coverage"],
            "llm_cost_diagnostics": diagnostics["llm_cost_diagnostics"],
            "broad_count_by_role": broad_count_by_role,
            "distilled_count_by_role": distilled_count_by_role,
        }
    )
    return MatrixDistillationResult(
        component_candidate_matrix=distilled_matrix,
        diagnostics=diagnostics,
    )


def distill_role_candidates(
    *,
    product_group: str,
    role: str,
    constraints: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    llm_client: LlmClient,
    role_limit: int,
    chunk_max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    deadline: _FullMatrixDeadline | None = None,
    progress_callback: MatrixDistillerProgressCallback | None = None,
) -> DistilledRoleResult:
    if deadline is None:
        deadline = _FullMatrixDeadline(progress_callback=progress_callback)
    elif progress_callback is None:
        progress_callback = deadline.progress_callback
    compact_candidates = [compact_candidate_for_distiller(row) for row in candidate_rows]
    candidate_by_id = {
        str(candidate.get("component_candidate_id") or "").strip(): candidate
        for candidate in compact_candidates
        if str(candidate.get("component_candidate_id") or "").strip()
    }
    if not candidate_by_id:
        return DistilledRoleResult(
            role=role,
            rows=[],
            diagnostics={"candidate_count": len(candidate_rows), "reason": "no_candidate_ids"},
        )

    chunks = split_candidate_chunks(
        product_group=product_group,
        role=role,
        constraints=constraints,
        candidates=compact_candidates,
        max_chars=chunk_max_chars,
    )
    evaluations: dict[str, dict[str, Any]] = {}
    unknown_ids: list[str] = []
    missing_evaluation_ids: list[str] = []
    invalid_rows = 0
    evaluator_prompt_chars = 0
    evaluator_response_chars = 0
    evaluator_calls_count = 0
    retried_chunk_count = 0
    failed_chunks: list[dict[str, Any]] = []

    for chunk_index, chunk in enumerate(chunks):
        deadline.check(stage="role_evaluator", role=role, chunk_index=chunk_index)
        payload = _distiller_payload(
            product_group=product_group,
            role=role,
            constraints=constraints,
            candidates=chunk,
        )
        prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        evaluated: list[Any] | None = None
        chunk_failure: dict[str, Any] | None = None
        for attempt_index in range(2):
            deadline.check(stage="role_evaluator", role=role, chunk_index=chunk_index)
            if attempt_index == 1:
                retried_chunk_count += 1
            evaluator_prompt_chars += len(prompt)
            evaluator_calls_count += 1
            timeout_seconds = deadline.call_timeout()
            _emit_progress(
                progress_callback,
                "role_evaluator_start",
                role=role,
                chunk=chunk_index,
                attempt=attempt_index + 1,
                candidate_count=len(chunk),
                timeout_seconds=round(timeout_seconds, 3),
                remaining_seconds=round(deadline.remaining(), 3),
            )
            try:
                response = _generate_json_with_timeout(
                    llm_client,
                    MATRIX_DISTILLER_SYSTEM_PROMPT,
                    prompt,
                    timeout_seconds=timeout_seconds,
                    deadline_seconds=deadline.max_seconds,
                    stage="role_evaluator",
                    role=role,
                    chunk_index=chunk_index,
                )
                evaluator_response_chars += len(
                    json.dumps(
                        response,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                )
                if not isinstance(response, Mapping):
                    raise MatrixDistillerError(
                        "matrix_distiller_response_not_object",
                        stage="role_evaluator",
                        role=role,
                        chunk_index=chunk_index,
                        cause_error_type="response_not_object",
                        parse_status="response_not_object",
                    )
                raw_evaluated = response.get("evaluated_candidates")
                if not isinstance(raw_evaluated, list):
                    raise MatrixDistillerError(
                        "matrix_distiller_response_missing_evaluated_candidates",
                        stage="role_evaluator",
                        role=role,
                        chunk_index=chunk_index,
                        cause_error_type="missing_evaluated_candidates",
                        parse_status="missing_evaluated_candidates",
                    )
                evaluated = raw_evaluated
                chunk_failure = None
                _emit_progress(
                    progress_callback,
                    "role_evaluator_done",
                    role=role,
                    chunk=chunk_index,
                    attempt=attempt_index + 1,
                    evaluated_count=len(evaluated),
                    remaining_seconds=round(deadline.remaining(), 3),
                )
                break
            except MatrixDistillerTimeoutError as exc:
                _emit_progress(
                    progress_callback,
                    "role_evaluator_timeout",
                    role=role,
                    chunk=chunk_index,
                    attempt=attempt_index + 1,
                    timeout_kind=exc.timeout_kind,
                    timeout_seconds=round(float(exc.timeout_seconds or 0), 3),
                    remaining_seconds=round(deadline.remaining(), 3),
                )
                chunk_failure = _role_evaluator_chunk_failure(
                    exc,
                    role=role,
                    chunk_index=chunk_index,
                    attempt_count=attempt_index + 1,
                )
                if exc.timeout_kind == "stage_deadline":
                    raise
                if attempt_index == 0:
                    continue
            except LlmError as exc:
                chunk_failure = _role_evaluator_chunk_failure(
                    exc,
                    role=role,
                    chunk_index=chunk_index,
                    attempt_count=attempt_index + 1,
                )
                if attempt_index == 0:
                    continue
        if evaluated is None:
            if chunk_failure is None:
                chunk_failure = {
                    "stage": "role_evaluator",
                    "role": role,
                    "chunk_index": chunk_index,
                    "attempt_count": 2,
                    "error_type": "UnknownError",
                }
            failed_chunks.append(chunk_failure)
            missing_evaluation_ids.extend(
                sorted(
                    str(candidate.get("component_candidate_id") or "").strip()
                    for candidate in chunk
                    if str(candidate.get("component_candidate_id") or "").strip()
                )
            )
            continue
        for item in evaluated:
            if not isinstance(item, Mapping):
                invalid_rows += 1
                continue
            component_id = str(item.get("component_candidate_id") or "").strip()
            if component_id not in candidate_by_id:
                if component_id:
                    unknown_ids.append(component_id)
                invalid_rows += 1
                continue
            normalized = _normalized_evaluation(item)
            current = evaluations.get(component_id)
            if current is None or _evaluation_sort_key(normalized) < _evaluation_sort_key(current):
                evaluations[component_id] = normalized

        chunk_ids = {
            str(candidate.get("component_candidate_id") or "").strip()
            for candidate in chunk
            if str(candidate.get("component_candidate_id") or "").strip()
        }
        missing_evaluation_ids.extend(
            sorted(component_id for component_id in chunk_ids if component_id not in evaluations)
        )

    input_ids = [
        str(candidate.get("component_candidate_id") or "").strip()
        for candidate in compact_candidates
        if str(candidate.get("component_candidate_id") or "").strip()
    ]
    reducer_result = _reduce_role_candidate_pool(
        product_group=product_group,
        role=role,
        constraints=constraints,
        candidates=compact_candidates,
        evaluations=evaluations,
        input_ids=input_ids,
        llm_client=llm_client,
        role_limit=role_limit,
        deadline=deadline,
        progress_callback=progress_callback,
    )
    selected_ids = reducer_result.selected_ids
    rows_by_id = {
        str(row.get("component_candidate_id") or row.get("candidate_id") or "").strip(): dict(row)
        for row in candidate_rows
    }
    distilled_rows = [
        _row_with_distiller_evaluation(rows_by_id[component_id], evaluations[component_id])
        for component_id in selected_ids
        if component_id in rows_by_id
    ]
    tier_counts = Counter(
        str(evaluation.get("fit_tier") or "") for evaluation in evaluations.values()
    )
    diagnostics = {
        "candidate_count": len(candidate_rows),
        "evaluated_count": len(evaluations),
        "selected_count": len(distilled_rows),
        "chunk_count": len(chunks),
        "fit_tier_counts": dict(sorted(tier_counts.items())),
        "unknown_component_candidate_ids": unknown_ids[:10],
        "missing_evaluation_candidate_ids": missing_evaluation_ids[:10],
        "missing_evaluation_count": len(set(missing_evaluation_ids)),
        "invalid_evaluation_rows": invalid_rows,
        "retried_chunk_count": retried_chunk_count,
        "failed_chunk_count": len(failed_chunks),
        "failed_chunks": failed_chunks,
        "role_limit": role_limit,
        "role_reducer_summary": reducer_result.summary,
        "llm_cost_diagnostics": {
            "llm_calls_count": evaluator_calls_count + reducer_result.llm_calls_count,
            "approximate_tokens": max(
                1,
                (
                    evaluator_prompt_chars
                    + evaluator_response_chars
                    + reducer_result.prompt_chars
                    + reducer_result.response_chars
                )
                // 4,
            ),
            "evaluator_calls_count": evaluator_calls_count,
            "reducer_calls_count": reducer_result.llm_calls_count,
            "evaluator_prompt_chars": evaluator_prompt_chars,
            "evaluator_response_chars": evaluator_response_chars,
            "reducer_prompt_chars": reducer_result.prompt_chars,
            "reducer_response_chars": reducer_result.response_chars,
        },
    }
    return DistilledRoleResult(role=role, rows=distilled_rows, diagnostics=diagnostics)


def compact_candidate_for_distiller(row: Mapping[str, Any]) -> dict[str, Any]:
    facts = row.get("extracted_facts") if isinstance(row.get("extracted_facts"), Mapping) else {}
    properties = _compact_content_properties(row.get("ocs_content_properties"))
    compact = {
        "component_candidate_id": row.get("component_candidate_id") or row.get("candidate_id"),
        "role": row.get("role"),
        "category_id": row.get("category_id"),
        "category_name": row.get("category_name"),
        "producer": row.get("producer"),
        "part_number": row.get("part_number"),
        "item_name": _short_text(row.get("name") or row.get("item_name")),
        "item_name_rus": _short_text(row.get("item_name_rus")),
        "product_name": _short_text(row.get("product_name")),
        "product_description": _short_text(row.get("product_description"), limit=240),
        "price_value": row.get("price_value"),
        "price_currency": row.get("price_currency"),
        "available_quantity": row.get("available_quantity"),
        "quantity_required": row.get("quantity_required"),
        "facts": _compact_scalar_mapping(facts),
        "content_properties": properties,
        "catalog_path": _compact_catalog_path(
            row.get("catalog_path") or row.get("catalog_path_json")
        ),
    }
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def split_candidate_chunks(
    *,
    product_group: str,
    role: str,
    constraints: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    max_chars: int,
) -> list[list[dict[str, Any]]]:
    max_chars = max(1000, max_chars)
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_dict = dict(candidate)
        trial = [*current, candidate_dict]
        if current and _payload_size(product_group, role, constraints, trial) > max_chars:
            chunks.append(current)
            current = [candidate_dict]
        else:
            current = trial
    if current:
        chunks.append(current)
    return chunks


def _distiller_payload(
    *,
    product_group: str,
    role: str,
    constraints: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "product_group": product_group,
        "role": role,
        "constraints": [dict(row) for row in constraints],
        "candidates": [dict(row) for row in candidates],
        "contract": {
            "selectable_fit_tiers": sorted(SELECTABLE_FIT_TIERS),
            "diagnostic_only_fit_tiers": [FIT_TIER_EXPLICIT_MISMATCH, FIT_TIER_WRONG_ROLE],
            "final_bom_selection": "application_composer_not_distiller",
        },
    }


def _role_evaluator_chunk_failure(
    exc: LlmError,
    *,
    role: str,
    chunk_index: int,
    attempt_count: int,
) -> dict[str, Any]:
    parse_status = ""
    http_status: int | None = None
    cause_error_type = type(exc).__name__
    if isinstance(exc, MatrixDistillerError):
        cause_error_type = str(exc.cause_error_type or cause_error_type)
        parse_status = str(exc.parse_status or "")
        http_status = exc.http_status
    elif isinstance(exc, LlmInvalidJsonError):
        parse_status = str(
            exc.json_extract_status
            or exc.parse_stage
            or exc.invalid_json_reason
            or "parse_error"
        )
    if isinstance(exc, LlmHttpError):
        http_status = exc.status_code
    failure = {
        "stage": "role_evaluator",
        "role": role,
        "chunk_index": chunk_index,
        "attempt_count": attempt_count,
        "error_type": type(exc).__name__,
        "cause_error_type": cause_error_type,
        "message": _short_text(str(exc) or type(exc).__name__, limit=240),
    }
    if parse_status:
        failure["parse_status"] = parse_status
    if http_status is not None:
        failure["http_status"] = http_status
    timeout_kind = str(getattr(exc, "timeout_kind", "") or "").strip()
    if timeout_kind:
        failure["timeout_kind"] = timeout_kind
    timeout_seconds = _float_value(getattr(exc, "timeout_seconds", None))
    if timeout_seconds is not None:
        failure["timeout_seconds"] = timeout_seconds
    return failure


def _payload_size(
    product_group: str,
    role: str,
    constraints: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> int:
    return len(
        json.dumps(
            _distiller_payload(
                product_group=product_group,
                role=role,
                constraints=constraints,
                candidates=candidates,
            ),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )


def _normalized_evaluation(item: Mapping[str, Any]) -> dict[str, Any]:
    fit_tier = str(item.get("fit_tier") or "").strip()
    if fit_tier not in FIT_TIER_RANK:
        fit_tier = FIT_TIER_FALLBACK_UNKNOWN
    confidence = str(item.get("confidence") or "low").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return {
        "component_candidate_id": str(item.get("component_candidate_id") or "").strip(),
        "fit_tier": fit_tier,
        "facts": _compact_scalar_mapping(item.get("facts")),
        "matched_constraints": _string_list(item.get("matched_constraints"))[:12],
        "missing_facts": _string_list(item.get("missing_facts"))[:12],
        "mismatch_reasons": _string_list(item.get("mismatch_reasons"))[:12],
        "price_stock_notes": _string_list(item.get("price_stock_notes"))[:8],
        "compatibility_assumptions": _string_list(
            item.get("compatibility_assumptions")
        )[:8],
        "engineer_checks": _string_list(item.get("engineer_checks"))[:8],
        "evidence": _short_text(
            item.get("evidence") or item.get("notes") or item.get("summary"),
            limit=280,
        ),
        "confidence": confidence,
    }


def _selected_candidate_ids(
    *,
    evaluations: Mapping[str, Mapping[str, Any]],
    input_ids: Sequence[str],
    role_limit: int,
) -> list[str]:
    role_limit = max(1, role_limit)
    order = {component_id: index for index, component_id in enumerate(input_ids)}
    selected: list[str] = []
    for tier in (FIT_TIER_STRONG, FIT_TIER_POSSIBLE):
        tier_ids = [
            component_id
            for component_id, row in evaluations.items()
            if row.get("fit_tier") == tier
        ]
        for component_id in sorted(tier_ids, key=lambda value: order.get(value, 10**9)):
            if len(selected) >= role_limit:
                return selected
            selected.append(component_id)

    fallback_limit = min(
        max(1, DEFAULT_FALLBACK_UNKNOWN_LIMIT),
        max(1, role_limit - len(selected)),
    )
    fallback_ids = [
        component_id
        for component_id, row in evaluations.items()
        if row.get("fit_tier") == FIT_TIER_FALLBACK_UNKNOWN
    ]
    for component_id in sorted(fallback_ids, key=lambda value: order.get(value, 10**9))[
        :fallback_limit
    ]:
        if len(selected) >= role_limit:
            break
        selected.append(component_id)
    return selected


def _reduce_role_candidate_pool(
    *,
    product_group: str,
    role: str,
    constraints: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    evaluations: Mapping[str, Mapping[str, Any]],
    input_ids: Sequence[str],
    llm_client: LlmClient,
    role_limit: int,
    deadline: _FullMatrixDeadline,
    progress_callback: MatrixDistillerProgressCallback | None = None,
) -> _RoleReducerResult:
    role_limit = max(1, int(role_limit or DEFAULT_ROLE_CANDIDATE_POOL_LIMIT))
    candidate_by_id = {
        str(candidate.get("component_candidate_id") or "").strip(): dict(candidate)
        for candidate in candidates
        if str(candidate.get("component_candidate_id") or "").strip()
    }
    fallback_ids = _selected_candidate_ids(
        evaluations=evaluations,
        input_ids=input_ids,
        role_limit=role_limit,
    )
    if not evaluations:
        return _RoleReducerResult(
            selected_ids=[],
            summary={
                "candidate_count_total": len(candidate_by_id),
                "evaluated_count": 0,
                "selected_count": 0,
                "fallback_reason": "no_evaluations",
                "no_viable_reason": "role_evaluator_returned_no_candidates",
            },
        )

    payload = _role_reducer_payload(
        product_group=product_group,
        role=role,
        constraints=constraints,
        candidates=candidate_by_id,
        evaluations=evaluations,
        role_limit=role_limit,
    )
    prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    response: Any
    deadline.check(stage="role_reducer", role=role)
    timeout_seconds = deadline.call_timeout()
    _emit_progress(
        progress_callback,
        "reducer_start",
        role=role,
        timeout_seconds=round(timeout_seconds, 3),
        evaluated_count=len(evaluations),
        remaining_seconds=round(deadline.remaining(), 3),
    )
    try:
        response = _generate_json_with_timeout(
            llm_client,
            MATRIX_ROLE_REDUCER_SYSTEM_PROMPT,
            prompt,
            timeout_seconds=timeout_seconds,
            deadline_seconds=deadline.max_seconds,
            stage="role_reducer",
            role=role,
        )
    except MatrixDistillerTimeoutError as exc:
        if exc.timeout_kind == "stage_deadline":
            raise
        _emit_progress(
            progress_callback,
            "reducer_timeout",
            role=role,
            timeout_kind=exc.timeout_kind,
            timeout_seconds=round(float(exc.timeout_seconds or 0), 3),
            remaining_seconds=round(deadline.remaining(), 3),
        )
        summary = _role_reducer_fallback_summary(
            role=role,
            candidate_count_total=len(candidate_by_id),
            evaluated_count=len(evaluations),
            role_limit=role_limit,
            selected_ids=fallback_ids[:role_limit],
            fallback_reason="role_reducer_timeout",
            error_type=type(exc).__name__,
            cause_error_type=str(exc.cause_error_type or ""),
            timeout_kind=str(exc.timeout_kind or ""),
            timeout_seconds=exc.timeout_seconds,
        )
        _emit_progress(
            progress_callback,
            "reducer_done",
            role=role,
            selected_count=len(fallback_ids[:role_limit]),
            fallback_reason=summary.get("fallback_reason"),
            remaining_seconds=round(deadline.remaining(), 3),
        )
        return _RoleReducerResult(
            selected_ids=fallback_ids[:role_limit],
            summary=summary,
            prompt_chars=len(prompt),
            llm_calls_count=1,
        )
    except LlmError as exc:
        summary = _role_reducer_fallback_summary(
            role=role,
            candidate_count_total=len(candidate_by_id),
            evaluated_count=len(evaluations),
            role_limit=role_limit,
            selected_ids=fallback_ids[:role_limit],
            fallback_reason="role_reducer_failed",
            error_type=type(exc).__name__,
            cause_error_type=type(exc).__name__,
        )
        _emit_progress(
            progress_callback,
            "reducer_done",
            role=role,
            selected_count=len(fallback_ids[:role_limit]),
            fallback_reason=summary.get("fallback_reason"),
            remaining_seconds=round(deadline.remaining(), 3),
        )
        return _RoleReducerResult(
            selected_ids=fallback_ids[:role_limit],
            summary=summary,
            prompt_chars=len(prompt),
            llm_calls_count=1,
        )
    response_chars = len(json.dumps(response, ensure_ascii=False, sort_keys=True, default=str))
    reducer_summary: dict[str, Any] = {
        "candidate_count_total": len(candidate_by_id),
        "evaluated_count": len(evaluations),
        "role_limit": role_limit,
    }
    selected_ids: list[str] = []
    if isinstance(response, Mapping):
        reducer_summary.update(
            {
                "role_summary": _short_text(response.get("role_summary"), limit=360),
                "no_viable_reason": _short_text(response.get("no_viable_reason"), limit=240),
                "rejected_summary": _mapping_rows(response.get("rejected_summary"))[:12],
            }
        )
        selected_ids = [
            component_id
            for component_id in _string_list(response.get("selected_candidate_ids"))
            if component_id in candidate_by_id and component_id in evaluations
        ]
        unknown_selected = [
            component_id
            for component_id in _string_list(response.get("selected_candidate_ids"))
            if component_id and component_id not in candidate_by_id
        ]
        if unknown_selected:
            reducer_summary["unknown_selected_candidate_ids"] = unknown_selected[:10]
    else:
        reducer_summary["fallback_reason"] = "role_reducer_response_not_object"

    selected_ids = _dedupe_ids(selected_ids)
    selected_ids = _selectable_or_diagnostic_ids(selected_ids, evaluations)
    selected_ids = _supplement_reducer_selection(
        selected_ids=selected_ids,
        fallback_ids=fallback_ids,
        candidate_by_id=candidate_by_id,
        evaluations=evaluations,
        role_limit=role_limit,
    )
    if not selected_ids and fallback_ids:
        reducer_summary["fallback_reason"] = reducer_summary.get(
            "fallback_reason",
            "role_reducer_selected_no_usable_ids",
        )
        selected_ids = fallback_ids[:role_limit]

    tier_counts = Counter(str(row.get("fit_tier") or "") for row in evaluations.values())
    reducer_summary.update(
        {
            "selected_count": len(selected_ids),
            "selected_candidate_ids": selected_ids[:role_limit],
            "fit_tier_counts": dict(sorted(tier_counts.items())),
        }
    )
    _emit_progress(
        progress_callback,
        "reducer_done",
        role=role,
        selected_count=len(selected_ids[:role_limit]),
        remaining_seconds=round(deadline.remaining(), 3),
    )
    return _RoleReducerResult(
        selected_ids=selected_ids[:role_limit],
        summary=reducer_summary,
        prompt_chars=len(prompt),
        response_chars=response_chars,
        llm_calls_count=1,
    )


def _role_reducer_fallback_summary(
    *,
    role: str,
    candidate_count_total: int,
    evaluated_count: int,
    role_limit: int,
    selected_ids: Sequence[str],
    fallback_reason: str,
    error_type: str,
    cause_error_type: str = "",
    timeout_kind: str = "",
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "candidate_count_total": candidate_count_total,
        "evaluated_count": evaluated_count,
        "role_limit": role_limit,
        "selected_count": len(selected_ids),
        "selected_candidate_ids": list(selected_ids),
        "fallback_reason": fallback_reason,
        "role_summary": f"{role} reducer fell back to deterministic shortlist.",
        "error_type": error_type,
    }
    if cause_error_type:
        summary["cause_error_type"] = cause_error_type
    if timeout_kind:
        summary["timeout_kind"] = timeout_kind
    if timeout_seconds is not None:
        summary["timeout_seconds"] = timeout_seconds
    return summary


def _role_reducer_payload(
    *,
    product_group: str,
    role: str,
    constraints: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
    evaluations: Mapping[str, Mapping[str, Any]],
    role_limit: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for component_id, evaluation in evaluations.items():
        candidate = candidates.get(component_id, {})
        rows.append(
            {
                "component_candidate_id": component_id,
                "fit_tier": evaluation.get("fit_tier"),
                "confidence": evaluation.get("confidence"),
                "facts": _compact_scalar_mapping(evaluation.get("facts")),
                "matched_constraints": _string_list(evaluation.get("matched_constraints"))[:8],
                "missing_facts": _string_list(evaluation.get("missing_facts"))[:8],
                "mismatch_reasons": _string_list(evaluation.get("mismatch_reasons"))[:8],
                "price_stock_notes": _string_list(evaluation.get("price_stock_notes"))[:6],
                "compatibility_assumptions": _string_list(
                    evaluation.get("compatibility_assumptions")
                )[:6],
                "engineer_checks": _string_list(evaluation.get("engineer_checks"))[:6],
                "candidate": {
                    key: candidate.get(key)
                    for key in (
                        "role",
                        "category_id",
                        "category_name",
                        "producer",
                        "part_number",
                        "item_name",
                        "product_name",
                        "price_value",
                        "price_currency",
                        "available_quantity",
                        "quantity_required",
                    )
                    if candidate.get(key) not in (None, "", [], {})
                },
            }
        )
    return {
        "product_group": product_group,
        "role": role,
        "constraints": [dict(row) for row in constraints],
        "candidate_count_total": len(candidates),
        "evaluated_count": len(evaluations),
        "role_candidate_pool_limit": role_limit,
        "candidates": [dict(candidate) for candidate in candidates.values()],
        "evaluated_candidates": rows,
        "contract": {
            "selectable_fit_tiers": sorted(SELECTABLE_FIT_TIERS),
            "diagnostic_only_fit_tiers": [FIT_TIER_EXPLICIT_MISMATCH, FIT_TIER_WRONG_ROLE],
            "final_bom_selection": "application_composer_not_role_reducer",
        },
    }


def _selectable_or_diagnostic_ids(
    selected_ids: Sequence[str],
    evaluations: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    selectable = [
        component_id
        for component_id in selected_ids
        if str(evaluations.get(component_id, {}).get("fit_tier") or "") in SELECTABLE_FIT_TIERS
    ]
    if selectable:
        return selectable
    return [
        component_id
        for component_id in selected_ids
        if str(evaluations.get(component_id, {}).get("fit_tier") or "")
        == FIT_TIER_FALLBACK_UNKNOWN
    ]


def _supplement_reducer_selection(
    *,
    selected_ids: Sequence[str],
    fallback_ids: Sequence[str],
    candidate_by_id: Mapping[str, Mapping[str, Any]],
    evaluations: Mapping[str, Mapping[str, Any]],
    role_limit: int,
) -> list[str]:
    selected = _dedupe_ids(selected_ids)
    if len(selected) >= role_limit:
        return selected[:role_limit]

    for component_id in fallback_ids:
        if len(selected) >= min(role_limit, max(8, len(fallback_ids))):
            break
        if component_id not in selected:
            selected.append(component_id)

    selectable_ids = [
        component_id
        for component_id, evaluation in evaluations.items()
        if str(evaluation.get("fit_tier") or "") in SELECTABLE_FIT_TIERS
    ]
    supplement_groups = [
        sorted(
            selectable_ids,
            key=lambda component_id: _candidate_price_sort_key(
                candidate_by_id.get(component_id, {})
            ),
        ),
        sorted(
            selectable_ids,
            key=lambda component_id: _candidate_stock_sort_key(
                candidate_by_id.get(component_id, {})
            ),
        ),
        _diverse_candidate_ids(selectable_ids, candidate_by_id),
    ]
    for group in supplement_groups:
        for component_id in group[: max(4, role_limit // 4)]:
            if len(selected) >= role_limit:
                return selected
            if component_id not in selected:
                selected.append(component_id)
    return selected[:role_limit]


def _candidate_price_sort_key(candidate: Mapping[str, Any]) -> tuple[int, Decimal, str]:
    price = _decimal_value(candidate.get("price_value"))
    if price is None:
        return (1, Decimal("0"), _stable_text(candidate.get("component_candidate_id")))
    quantity = _int_value(candidate.get("quantity_required")) or 1
    return (0, price * max(1, quantity), _stable_text(candidate.get("component_candidate_id")))


def _candidate_stock_sort_key(candidate: Mapping[str, Any]) -> tuple[int, str]:
    stock = _int_value(candidate.get("available_quantity"))
    return (
        -(stock if stock is not None else -1),
        _stable_text(candidate.get("component_candidate_id")),
    )


def _diverse_candidate_ids(
    candidate_ids: Sequence[str],
    candidate_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    seen: set[tuple[str, str]] = set()
    result: list[str] = []
    for component_id in candidate_ids:
        candidate = candidate_by_id.get(component_id, {})
        key = (
            _stable_text(candidate.get("producer")),
            _stable_text(candidate.get("category_id") or candidate.get("category_name")),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(component_id)
    return result


def _row_with_distiller_evaluation(
    row: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(row)
    fit_tier = str(evaluation.get("fit_tier") or FIT_TIER_FALLBACK_UNKNOWN)
    result.update(
        {
            "fit_tier": fit_tier,
            "matrix_distiller_fit_tier": fit_tier,
            "matrix_distiller_confidence": evaluation.get("confidence"),
            "matrix_distiller_facts": dict(evaluation.get("facts") or {}),
            "matrix_distiller_matched_constraints": _string_list(
                evaluation.get("matched_constraints")
            ),
            "matrix_distiller_missing_facts": _string_list(evaluation.get("missing_facts")),
            "matrix_distiller_mismatch_reasons": _string_list(
                evaluation.get("mismatch_reasons")
            ),
            "matrix_distiller_price_stock_notes": _string_list(
                evaluation.get("price_stock_notes")
            ),
            "matrix_distiller_compatibility_assumptions": _string_list(
                evaluation.get("compatibility_assumptions")
            ),
            "matrix_distiller_engineer_checks": _string_list(
                evaluation.get("engineer_checks")
            ),
            "matrix_distiller_evidence": _short_text(evaluation.get("evidence"), limit=280),
        }
    )
    return result


def _constraints_for_role(
    constraints_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    role: str,
) -> list[Mapping[str, Any]]:
    rows = list(constraints_by_role.get(role) or [])
    if role in {"drive", "ssd", "hdd"}:
        rows.extend(constraints_by_role.get("storage") or [])
    if role == "server_platform":
        rows.extend(constraints_by_role.get("platform") or [])
    return rows


def _count_by_role(component_candidate_matrix: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    ready_server_rows = component_candidate_matrix.get(READY_SERVER_MATRIX_KEY)
    if isinstance(ready_server_rows, list) and ready_server_rows:
        result["ready_server"] = len(ready_server_rows)
    for role, matrix_key in SERVER_MATRIX_KEYS.items():
        rows = component_candidate_matrix.get(matrix_key)
        if isinstance(rows, list) and rows:
            result[role] = len(rows)
    return result


def _evaluation_sort_key(row: Mapping[str, Any]) -> tuple[int, int]:
    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    return (
        FIT_TIER_RANK.get(str(row.get("fit_tier") or ""), 99),
        confidence_rank.get(str(row.get("confidence") or ""), 2),
    )


def _coverage_diagnostics(
    *,
    considered_count_by_role: Mapping[str, int],
    matrix_count_by_role: Mapping[str, int],
) -> dict[str, Any]:
    coverage_percent_by_role: dict[str, float] = {}
    for role, matrix_count in matrix_count_by_role.items():
        considered = int(considered_count_by_role.get(role, 0) or 0)
        total = int(matrix_count or 0)
        coverage_percent_by_role[role] = round(
            (considered / total * 100) if total else 100.0,
            2,
        )
    skipped = {
        role: max(0, int(matrix_count or 0) - int(considered_count_by_role.get(role, 0) or 0))
        for role, matrix_count in matrix_count_by_role.items()
        if int(matrix_count or 0) > int(considered_count_by_role.get(role, 0) or 0)
    }
    return {
        "considered_count_by_role": {
            str(role): int(count or 0)
            for role, count in considered_count_by_role.items()
        },
        "matrix_count_by_role": {
            str(role): int(count or 0) for role, count in matrix_count_by_role.items()
        },
        "coverage_percent_by_role": coverage_percent_by_role,
        "skipped_candidate_count_by_role": skipped,
        "coverage_incomplete": bool(skipped),
        "incomplete_roles": list(skipped.keys()),
    }


def _mapping_rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _safe_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _compact_scalar_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        if key_text in {"raw", "raw_json", "payload", "debug", "diagnostics"}:
            continue
        if isinstance(item, Mapping | list | tuple):
            continue
        if item in (None, ""):
            continue
        result[key_text] = item
        if len(result) >= 24:
            break
    return result


def _compact_content_properties(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = _short_text(item.get("name"), limit=80)
        property_value = _short_text(item.get("value"), limit=120)
        if not name or not property_value:
            continue
        row = {
            "name": name,
            "value": property_value,
            "unit": _short_text(item.get("unit"), limit=24),
        }
        rows.append({key: val for key, val in row.items() if val})
        if len(rows) >= 20:
            break
    return rows


def _compact_catalog_path(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            text = _short_text(item.get("name") or item.get("category_name"), limit=80)
        else:
            text = _short_text(item, limit=80)
        if text:
            result.append(text)
        if len(result) >= 6:
            break
    return result


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [_short_text(item, limit=160) for item in value if _short_text(item, limit=160)]


def _short_text(value: Any, *, limit: int = 180) -> str:
    text = " ".join(str(value or "").replace("\x00", "").split())
    return text[:limit]


def _dedupe_ids(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(Decimal(text))
    except (InvalidOperation, ValueError):
        return None


def _float_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value or "").strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(Decimal(text))
    except (InvalidOperation, ValueError):
        return None


def _decimal_value(value: Any) -> Decimal | None:
    if isinstance(value, Decimal):
        return value
    text = str(value or "").strip().replace(",", ".")
    if not text:
        return None
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def _stable_text(value: Any) -> str:
    return str(value or "").strip().casefold()
