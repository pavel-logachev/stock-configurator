from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.api.routes.match import router as match_router
from app.llm.simple_stock_composer import (
    SIMPLE_STOCK_COMPOSER_PROMPT_VERSION,
    SIMPLE_STOCK_QUOTE_PIPELINE_VERSION,
)
from app.matching.simple_stock_matrix import SIMPLE_STOCK_MATRIX_SCHEMA_VERSION
from app.matching.simple_stock_quote_service import SIMPLE_STOCK_ROUTE_PROMPT_VERSION
from app.matching.simple_stock_reconciler import QUOTE_INTEGRITY_RECONCILER_VERSION
from app.matching.v3_full_category_quote_service import QUOTE_DRAFT_REVIEW_REQUIRED

BASELINE_PATH = Path(__file__).resolve().parents[1] / "config" / "production_pipeline_baseline.json"


def _load_baseline() -> dict[str, Any]:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def test_public_pipeline_baseline_matches_active_pipeline() -> None:
    baseline = _load_baseline()
    stages = baseline["stages"]
    route_paths = {route.path for route in match_router.routes}

    assert baseline["schema_version"] == "stock-configurator-production-pipeline-baseline-v1"
    assert baseline["entrypoint"] in route_paths
    assert baseline["pipeline_version"] == SIMPLE_STOCK_QUOTE_PIPELINE_VERSION
    assert baseline["result_state"] == QUOTE_DRAFT_REVIEW_REQUIRED
    assert stages == {
        "route_prompt_version": SIMPLE_STOCK_ROUTE_PROMPT_VERSION,
        "matrix_schema_version": SIMPLE_STOCK_MATRIX_SCHEMA_VERSION,
        "composer_prompt_version": SIMPLE_STOCK_COMPOSER_PROMPT_VERSION,
        "reconciler_version": QUOTE_INTEGRITY_RECONCILER_VERSION,
    }


def test_public_pipeline_baseline_is_synthetic_and_requires_guarded_changes() -> None:
    baseline = _load_baseline()

    assert baseline["environment"] == "public-synthetic"
    assert baseline["captured_at"] is None
    assert baseline["production_commit"] == "public-clean-room-snapshot"
    assert baseline["llm"] == {
        "provider": "openai-compatible",
        "model": "example/model",
        "timeout_seconds": 900,
        "read_timeout_seconds": 1800,
        "max_package_chars": 5_000_000,
        "max_output_tokens": 65_536,
        "thinking_enabled": False,
    }
    assert baseline["production_evidence"] == {
        "source": "synthetic",
        "match_run_id": None,
        "stock_row_count": None,
        "validation_error_count": None,
    }
    assert baseline["runtime_flags"] == {
        "high_quality_full_matrix_by_default": True,
        "refresh_categories_before_llm": True,
    }
    assert baseline["change_policy"] == {
        "preserve_default_behavior": True,
        "require_shadow_evaluation_for_semantic_changes": True,
        "require_single_change_axis_per_release": True,
        "require_rollback_plan": True,
    }
