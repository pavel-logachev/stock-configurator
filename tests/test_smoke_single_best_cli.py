from __future__ import annotations

import json
from io import BytesIO
from typing import Any

import httpx
import pytest
from openpyxl import Workbook

import app.cli.smoke_single_best as smoke_cli


def test_smoke_single_best_success_detects_required_fields(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "router-secret-for-smoke"
    monkeypatch.setenv("LLM_API_KEY", secret)
    client = _client(
        {
            "match_run_id": 62,
            "output_mode": "single_best_cost_valid",
            "primary_recommendation_status": "valid",
            "commercial_summary": {"copy_paste_text": f"КП готово без печати {secret}"},
            "report_xlsx_url": "/api/v1/match/62/report.xlsx",
        }
    )

    exit_code = smoke_cli.run(["--text", "secret prompt"], client=client)
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert summary["ok"] is True
    assert summary["match_run_id"] == 62
    assert summary["checks"]["single_best_output_mode"] is True
    assert summary["checks"]["primary_recommendation_valid"] is True
    assert summary["checks"]["commercial_summary_present"] is True
    assert summary["checks"]["excel_two_sheets"] is True
    assert summary["excel_sheet_names"] == ["AI-рекомендации", "Матрица компонентов"]
    assert secret not in captured.out
    assert "secret prompt" not in captured.out


def test_smoke_single_best_detects_wrong_single_best_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _client(
        {
            "match_run_id": 63,
            "output_mode": "grouped_presales",
            "primary_recommendation_status": "no_recommendation",
            "commercial_summary": {},
        }
    )

    exit_code = smoke_cli.run(["--match-run-id", "63"], client=client)
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 1
    assert summary["ok"] is False
    assert summary["checks"]["single_best_output_mode"] is False
    assert summary["checks"]["primary_recommendation_valid"] is False
    assert summary["checks"]["commercial_summary_present"] is False


def test_smoke_single_best_detects_forbidden_internal_fragments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _client(
        {
            "match_run_id": 64,
            "output_mode": "single_best_cost_valid",
            "primary_recommendation_status": "valid",
            "commercial_summary": {"copy_paste_text": "Коммерческий блок llm_rec_64"},
        },
        workbook_note="Отчет без служебных фрагментов",
    )

    exit_code = smoke_cli.run(["--match-run-id", "64"], client=client)
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 1
    assert summary["checks"]["commercial_summary_user_surface_clean"] is False
    assert "llm_rec_64" not in captured.out


def test_smoke_single_best_detects_forbidden_excel_fragments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _client(
        {
            "match_run_id": 65,
            "output_mode": "single_best_cost_valid",
            "primary_recommendation_status": "valid",
            "commercial_summary": {"copy_paste_text": "Коммерческий блок чистый"},
        },
        workbook_note="debug raw JSON",
    )

    exit_code = smoke_cli.run(["--match-run-id", "65"], client=client)
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 1
    assert summary["checks"]["excel_user_surfaces_clean"] is False
    assert "raw JSON" not in captured.out


def test_smoke_single_best_unit_test_uses_mock_client_not_live_llm(
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[tuple[str, str]] = []
    client = _client(
        {
            "match_run_id": 66,
            "output_mode": "single_best_cost_valid",
            "primary_recommendation_status": "valid",
            "commercial_summary": {"copy_paste_text": "Коммерческий блок чистый"},
        },
        requests=requests,
    )

    exit_code = smoke_cli.run(["--match-run-id", "66"], client=client)
    capsys.readouterr()

    assert exit_code == 0
    assert requests == [
        ("GET", "/api/v1/match/66"),
        ("GET", "/api/v1/match/66/report.xlsx"),
    ]


def _client(
    match_payload: dict[str, Any],
    *,
    workbook_note: str = "Коммерческий блок чистый",
    requests: list[tuple[str, str]] | None = None,
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append((request.method, request.url.path))
        if request.url.path == "/api/v1/match" and request.method == "POST":
            return httpx.Response(201, json=match_payload)
        if request.url.path == f"/api/v1/match/{match_payload['match_run_id']}":
            return httpx.Response(200, json=match_payload)
        if request.url.path == f"/api/v1/match/{match_payload['match_run_id']}/report.xlsx":
            return httpx.Response(
                200,
                content=_workbook_bytes(workbook_note),
                headers={
                    "content-type": (
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                },
            )
        return httpx.Response(404)

    return httpx.Client(
        base_url="http://stock-api.test",
        transport=httpx.MockTransport(handler),
    )


def _workbook_bytes(note: str) -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "AI-рекомендации"
    sheet["A1"] = note
    matrix = workbook.create_sheet("Матрица компонентов")
    matrix["A1"] = "Охват матрицы"

    output = BytesIO()
    workbook.save(output)
    workbook.close()
    return output.getvalue()
