# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

from metis.engine import llm_triage_service
from metis.engine.llm_triage_service import LlmTriageService


def test_llm_triage_filters_p5_and_sorts_priorities(monkeypatch, tmp_path):
    source = tmp_path / "driver.c"
    source.write_text(
        """
int clean_bug(char *p) {
    return p[64];
}

int noisy_warning(void) {
    return 0;
}
""".strip(),
        encoding="utf-8",
    )

    calls = []

    def _fake_invoke(_provider, _usage_runtime, **kwargs):
        calls.append(kwargs)
        batch = json.loads(kwargs["variables"]["batch_json"])
        assert len(batch) == 2
        assert "Nearby lines" in batch[0]["code_context"]
        assert any(
            match["query"] == "clean_bug" for match in batch[0]["repo_search_evidence"]
        )
        return json.dumps(
            {
                "decisions": [
                    {
                        "id": "F001",
                        "priority": "p0",
                        "keep": True,
                        "duplicate_of": None,
                        "reason": "Direct attacker-controlled out-of-bounds read.",
                        "exploitability": "Attacker controls p and can trigger the read.",
                    },
                    {
                        "id": "F002",
                        "priority": "p5",
                        "keep": False,
                        "duplicate_of": None,
                        "reason": "No security impact.",
                        "exploitability": "None.",
                    },
                ]
            }
        )

    monkeypatch.setattr(llm_triage_service, "invoke_reachability_prompt", _fake_invoke)

    results = {
        "reviews": [
            {
                "file": "driver.c",
                "file_path": str(source),
                "reviews": [
                    {
                        "issue": "Out-of-bounds read",
                        "line_number": 2,
                        "primary_file": "driver.c",
                        "primary_function": "clean_bug",
                        "severity": "High",
                        "confidence": 0.95,
                        "reasoning": "p is read past the end.",
                    },
                    {
                        "issue": "Style warning",
                        "line_number": 6,
                        "primary_file": "driver.c",
                        "primary_function": "noisy_warning",
                        "severity": "Low",
                        "confidence": 0.4,
                        "reasoning": "Code quality only.",
                    },
                ],
            }
        ]
    }
    service = LlmTriageService(
        codebase_path=tmp_path,
        llm_provider=object(),
        usage_runtime=SimpleNamespace(),
    )

    payload = service.triage_review_results(
        results,
        model="gpt-test",
        reasoning_effort="high",
        batch_size=10,
    )

    assert calls[0]["model"] == "gpt-test"
    assert calls[0]["reasoning_effort"] == "high"
    assert payload["summary"]["total_input_findings"] == 2
    assert payload["summary"]["kept_findings"] == 1
    assert payload["summary"]["filtered_findings"] == 1
    assert payload["issues"][0]["id"] == "F001"
    assert payload["issues"][0]["priority"] == "p0"
    assert payload["issues"][0]["llm_triage_reason"].startswith("Direct")


def test_llm_triage_keeps_omitted_decisions_conservatively(monkeypatch, tmp_path):
    source = tmp_path / "driver.c"
    source.write_text("int maybe_bug(void) { return 1; }\n", encoding="utf-8")

    monkeypatch.setattr(
        llm_triage_service,
        "invoke_reachability_prompt",
        lambda *_args, **_kwargs: '{"decisions": []}',
    )

    service = LlmTriageService(
        codebase_path=tmp_path,
        llm_provider=object(),
        usage_runtime=SimpleNamespace(),
    )
    payload = service.triage_review_results(
        {
            "reviews": [
                {
                    "file": "driver.c",
                    "file_path": str(source),
                    "reviews": [{"issue": "Maybe bug", "line_number": 1}],
                }
            ]
        }
    )

    assert payload["issues"][0]["priority"] == "p4"
    assert "omitted" in payload["issues"][0]["llm_triage_reason"]


def test_llm_triage_accepts_additional_findings(monkeypatch, tmp_path):
    source = tmp_path / "driver.c"
    source.write_text(
        """
int reported(void) { return 0; }
void missed(char *dst, char *src) { strcpy(dst, src); }
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        llm_triage_service,
        "invoke_reachability_prompt",
        lambda *_args, **_kwargs: json.dumps(
            {
                "decisions": [
                    {
                        "id": "F001",
                        "priority": "p4",
                        "keep": True,
                        "duplicate_of": None,
                        "reason": "Still plausible.",
                        "exploitability": "Requires context.",
                    }
                ],
                "additional_findings": [
                    {
                        "priority": "p1",
                        "file": "driver.c",
                        "line_number": 2,
                        "issue": "Unbounded strcpy in missed",
                        "severity": "High",
                        "confidence": 0.9,
                        "cwe": "CWE-120",
                        "reasoning": "Provided code shows strcpy from attacker data.",
                        "mitigation": "Use a bounded copy and validate lengths.",
                    }
                ],
            }
        ),
    )

    service = LlmTriageService(
        codebase_path=tmp_path,
        llm_provider=object(),
        usage_runtime=SimpleNamespace(),
    )
    payload = service.triage_review_results(
        {
            "reviews": [
                {
                    "file": "driver.c",
                    "file_path": str(source),
                    "reviews": [{"issue": "Reported issue", "line_number": 1}],
                }
            ]
        }
    )

    assert payload["summary"]["additional_findings"] == 1
    assert [issue["id"] for issue in payload["issues"]] == ["A001", "F001"]
    assert payload["issues"][0]["priority"] == "p1"
    assert payload["issues"][0]["llm_triage_source"] == "additional_finding"
