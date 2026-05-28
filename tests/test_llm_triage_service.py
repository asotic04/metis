# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

from metis.engine import llm_triage_service
from metis.engine.llm_triage_service import LlmTriageService


def test_llm_triage_prompt_uses_metis_priority_rubric():
    prompt = llm_triage_service._TRIAGE_SYSTEM_PROMPT

    assert "Metis default priority rubric" in prompt
    assert "p0: emergency response" in prompt
    assert "p2: normal security priority" in prompt
    assert "This is the default for kept real security" in prompt
    assert "p5: Metis-only filtered state" in prompt


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
        assert "system_prompt" not in kwargs
        assert "user_prompt" not in kwargs
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

    monkeypatch.setattr(llm_triage_service, "_invoke_triage_prompt", _fake_invoke)

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
    assert payload["summary"]["kept_input_findings"] == 1
    assert payload["summary"]["filtered_findings"] == 1
    assert payload["issues"][0]["id"] == "F001"
    assert payload["issues"][0]["priority"] == "p0"
    assert payload["issues"][0]["llm_triage_reason"].startswith("Direct")


def test_llm_triage_keeps_omitted_decisions_conservatively(monkeypatch, tmp_path):
    source = tmp_path / "driver.c"
    source.write_text("int maybe_bug(void) { return 1; }\n", encoding="utf-8")

    monkeypatch.setattr(
        llm_triage_service,
        "_invoke_triage_prompt",
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


def test_llm_triage_keeps_representative_when_duplicate_cluster_all_filtered(
    monkeypatch, tmp_path
):
    source = tmp_path / "config_loader.cpp"
    source.write_text(
        """
std::string LoadTemplate(const std::string& template_name) {
  const std::string path = config_root + "/" + template_name + ".cfg";
  std::ifstream input(path);
}
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        llm_triage_service,
        "_invoke_triage_prompt",
        lambda *_args, **_kwargs: json.dumps(
            {
                "decisions": [
                    {
                        "id": "F001",
                        "priority": "p5",
                        "keep": False,
                        "duplicate_of": None,
                        "reason": "Duplicate/no context.",
                        "exploitability": "Not assessed.",
                    },
                    {
                        "id": "F002",
                        "priority": "p5",
                        "keep": False,
                        "duplicate_of": "F001",
                        "reason": "Duplicate.",
                        "exploitability": "Not assessed.",
                    },
                ]
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
                    "file": "config_loader.cpp",
                    "file_path": str(source),
                    "reviews": [
                        {
                            "issue": "Path traversal in LoadTemplate",
                            "line_number": 3,
                            "primary_file": "config_loader.cpp",
                            "primary_function": "LoadTemplate",
                            "severity": "High",
                            "confidence": 0.95,
                            "cwe": "CWE-22",
                            "reasoning": (
                                "template_name is concatenated into a path and opened."
                            ),
                        },
                        {
                            "issue": "Untrusted template path is opened",
                            "line_number": 3,
                            "primary_file": "config_loader.cpp",
                            "primary_function": "LoadTemplate",
                            "severity": "High",
                            "confidence": 0.95,
                            "cwe": "CWE-22",
                            "reasoning": (
                                "caller-controlled template_name can escape the root."
                            ),
                        },
                    ],
                }
            ]
        }
    )

    assert payload["summary"]["total_input_findings"] == 2
    assert payload["summary"]["kept_findings"] == 1
    assert payload["summary"]["kept_input_findings"] == 1
    assert payload["summary"]["filtered_findings"] == 1
    assert payload["issues"][0]["id"] == "F001"
    assert payload["issues"][0]["priority"] == "p3"
    assert "high-confidence same-location duplicate cluster" in payload["issues"][0][
        "llm_triage_reason"
    ]


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
        "_invoke_triage_prompt",
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
    assert payload["summary"]["kept_input_findings"] == 1
    assert payload["summary"]["filtered_findings"] == 0
    assert [issue["id"] for issue in payload["issues"]] == ["A001", "F001"]
    assert payload["issues"][0]["priority"] == "p1"
    assert payload["issues"][0]["llm_triage_source"] == "additional_finding"


def test_llm_triage_collapses_duplicate_root_causes_kept_by_model(
    monkeypatch, tmp_path
):
    source = tmp_path / "proto.c"
    source.write_text(
        """
int proto_parse(message_t *msg) {
    free(msg->data);
    return -1;
}
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        llm_triage_service,
        "_invoke_triage_prompt",
        lambda *_args, **_kwargs: json.dumps(
            {
                "decisions": [
                    {
                        "id": "F001",
                        "priority": "p0",
                        "keep": True,
                        "duplicate_of": None,
                        "reason": "Direct double free.",
                        "exploitability": "Attacker sends an invalid command.",
                    },
                    {
                        "id": "F002",
                        "priority": "p2",
                        "keep": True,
                        "duplicate_of": None,
                        "reason": "Same invalid-command cleanup bug.",
                        "exploitability": "Same attacker input.",
                    },
                ]
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
                    "file": "proto.c",
                    "file_path": str(source),
                    "reviews": [
                        {
                            "issue": "Invalid command double free",
                            "line_number": 2,
                            "primary_file": "proto.c",
                            "primary_function": "proto_parse",
                            "severity": "High",
                            "confidence": 0.95,
                            "cwe": "CWE-415",
                            "reasoning": (
                                "Canonical key: proto.c:proto_parse:"
                                "double_free:invalid_command_frees_msg_data"
                            ),
                        },
                        {
                            "issue": "Invalid command leaves stale msg data",
                            "line_number": 3,
                            "primary_file": "proto.c",
                            "primary_function": "proto_parse",
                            "severity": "High",
                            "confidence": 0.95,
                            "cwe": "CWE-404",
                            "reasoning": (
                                "Canonical key: proto.c:proto_parse:"
                                "partial_cleanup:invalid_command_frees_msg_data"
                            ),
                        },
                    ],
                }
            ]
        }
    )

    assert payload["summary"]["total_input_findings"] == 2
    assert payload["summary"]["kept_findings"] == 1
    assert payload["summary"]["kept_input_findings"] == 1
    assert payload["summary"]["filtered_findings"] == 1
    assert payload["issues"][0]["id"] == "F001"


def test_llm_triage_collapses_cross_file_sink_and_callsite_duplicates(
    monkeypatch, tmp_path
):
    util_source = tmp_path / "util.c"
    util_source.write_text(
        'void util_log(char *msg) { vprintf(msg, ap); }\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        llm_triage_service,
        "_invoke_triage_prompt",
        lambda *_args, **_kwargs: json.dumps(
            {
                "decisions": [
                    {
                        "id": "F001",
                        "priority": "p3",
                        "keep": True,
                        "duplicate_of": None,
                        "reason": "Generic format-string sink.",
                        "exploitability": "Needs attacker-controlled caller.",
                    },
                ],
                "additional_findings": [
                    {
                        "priority": "p2",
                        "file": "src/dispatch.c",
                        "line_number": 74,
                        "issue": (
                            "User-controlled task title is used as a printf-style "
                            "log format string"
                        ),
                        "severity": "High",
                        "confidence": 0.9,
                        "cwe": "CWE-134",
                        "reasoning": (
                            "handle_task_get calls util_log(t->title), and util_log "
                            "passes msg to vprintf as the format string."
                        ),
                        "mitigation": "Use util_log(\"%s\", t->title).",
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
                    "file": "src/util.c",
                    "file_path": str(util_source),
                    "reviews": [
                        {
                            "issue": (
                                "util_log passes msg directly as the vprintf "
                                "format string"
                            ),
                            "line_number": 1,
                            "primary_file": "src/util.c",
                            "primary_function": "util_log",
                            "severity": "High",
                            "confidence": 0.95,
                            "cwe": "CWE-134",
                            "reasoning": (
                                "util_log calls vprintf(msg, ap). Repo evidence "
                                "shows util_log(t->title)."
                            ),
                        }
                    ],
                }
            ]
        }
    )

    assert payload["summary"]["kept_findings"] == 1
    assert payload["summary"]["kept_input_findings"] == 0
    assert payload["summary"]["filtered_findings"] == 1
    assert payload["summary"]["additional_findings"] == 1
    assert payload["issues"][0]["id"] == "A001"
