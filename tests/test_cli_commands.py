# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

from metis.cli import commands
from metis.cli.command_runtime import CommandRuntime
from metis.engine.options import ReviewOptions, TriageOptions


def test_run_review_code_uses_review_domain_surface(monkeypatch):
    calls = []

    class _ReviewDomain:
        def get_code_files(self, options=None):
            assert isinstance(options, ReviewOptions)
            calls.append(("get_code_files", options.use_retrieval_context))
            return ["a.py"]

        def review_code(self, options=None):
            assert isinstance(options, ReviewOptions)
            calls.append(("review_code", options.use_retrieval_context))
            yield {"file": "a.py", "reviews": []}

    engine = SimpleNamespace(review=_ReviewDomain())
    args = SimpleNamespace(
        verbose=True,
        quiet=True,
        triage=False,
        output_file=None,
    )
    runtime = CommandRuntime(
        command="review_code",
        command_args=[],
        use_retrieval_context=True,
    )

    monkeypatch.setattr(commands, "print_console", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        commands, "iterate_with_progress", lambda _total, iterable: list(iterable)
    )
    monkeypatch.setattr(
        commands, "_finalize_review_output", lambda *_args, **_kwargs: None
    )

    commands.run_review_code(engine, args, runtime)

    assert calls == [("get_code_files", True), ("review_code", True)]


def test_run_update_uses_indexing_domain_surface(monkeypatch, tmp_path):
    patch_file = tmp_path / "change.diff"
    patch_file.write_text("diff --git a/a.py b/a.py", encoding="utf-8")

    captured: list[str] = []

    class _IndexingDomain:
        def update_index(self, patch_text):
            captured.append(patch_text)

    engine = SimpleNamespace(indexing=_IndexingDomain())
    args = SimpleNamespace(quiet=True)

    monkeypatch.setattr(commands, "print_console", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        commands,
        "with_spinner",
        lambda _message, func, *func_args, **_func_kwargs: func(*func_args),
    )

    commands.run_update(
        engine,
        str(patch_file),
        args,
        CommandRuntime(
            command="update",
            command_args=[str(patch_file)],
            use_retrieval_context=True,
        ),
    )

    assert captured == ["diff --git a/a.py b/a.py"]


def test_run_index_verbose_uses_indexing_domain_surface(monkeypatch):
    calls: list[str] = []

    class _IndexingDomain:
        def count_index_items(self):
            calls.append("count")
            return 2

        def index_prepare_nodes_iter(self):
            calls.append("prepare")
            yield None
            yield None

        def index_finalize_embeddings(self):
            calls.append("finalize")

    engine = SimpleNamespace(indexing=_IndexingDomain())

    monkeypatch.setattr(commands, "print_console", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        commands, "iterate_with_progress", lambda _total, iterable: list(iterable)
    )
    monkeypatch.setattr(
        commands, "with_timer", lambda _message, func, **_kwargs: func()
    )

    commands.run_index(engine, verbose=True, quiet=True)

    assert calls == ["count", "prepare", "finalize"]


def test_run_triage_propagates_no_index_mode_and_warning(tmp_path, monkeypatch):
    sarif_path = tmp_path / "input.sarif"
    sarif_path.write_text('{"version":"2.1.0","runs":[]}', encoding="utf-8")
    captured = []

    class _DummyEngine:
        def triage_sarif_file(self, input_path, output_path=None, **kwargs):
            assert input_path == str(sarif_path)
            assert isinstance(kwargs["options"], TriageOptions)
            assert kwargs["options"].use_retrieval_context is False
            return output_path or input_path

    args = SimpleNamespace(
        quiet=False,
        output_file=None,
        include_triaged=False,
    )
    runtime = CommandRuntime(
        command="triage",
        command_args=[str(sarif_path)],
        use_retrieval_context=False,
    )
    monkeypatch.setattr(
        commands,
        "print_console",
        lambda message, *_args, **_kwargs: captured.append(str(message)),
    )

    commands.run_triage(_DummyEngine(), str(sarif_path), args, runtime)

    assert any("Running without index" in message for message in captured)


def test_run_review_patch_propagates_no_index_mode(monkeypatch, tmp_path):
    patch_file = tmp_path / "change.diff"
    patch_file.write_text("diff --git a/a.py b/a.py", encoding="utf-8")
    captured = []

    class _ReviewDomain:
        def review_patch(self, patch_file=None, options=None):
            assert isinstance(options, ReviewOptions)
            assert options.use_retrieval_context is False
            return {"reviews": [], "overall_changes": ""}

    engine = SimpleNamespace(review=_ReviewDomain())
    args = SimpleNamespace(
        quiet=False,
        triage=False,
        output_file=None,
    )
    runtime = CommandRuntime(
        command="review_patch",
        command_args=[str(patch_file)],
        use_retrieval_context=False,
    )

    monkeypatch.setattr(
        commands, "_finalize_review_output", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        commands,
        "print_console",
        lambda message, *_args, **_kwargs: captured.append(str(message)),
    )
    monkeypatch.setattr(
        commands,
        "with_spinner",
        lambda _message, func, *func_args, **func_kwargs: func(
            *func_args,
            **{k: v for k, v in func_kwargs.items() if k != "quiet"},
        ),
    )

    commands.run_review(engine, str(patch_file), args, runtime)

    assert any("Running without index" in message for message in captured)


def test_run_review_code_triggers_triage_when_global_flag_enabled(monkeypatch):
    calls = []

    class _ReviewDomain:
        def get_code_files(self, options=None):
            return ["a.py"]

        def review_code(self, options=None):
            yield {"file": "a.py", "reviews": []}

    class _Engine:
        def __init__(self):
            self.review = _ReviewDomain()

        def triage_sarif_payload(self, payload, **kwargs):
            calls.append(kwargs["options"].include_triaged)
            payload["runs"] = []
            return payload

    engine = _Engine()
    args = SimpleNamespace(
        verbose=False,
        quiet=True,
        triage=True,
        include_triaged=False,
        output_file=None,
    )
    runtime = CommandRuntime(
        command="review_code",
        command_args=[],
        use_retrieval_context=True,
    )

    monkeypatch.setattr(commands, "print_console", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        commands,
        "with_spinner",
        lambda _message, func, *func_args, **func_kwargs: func(
            *func_args,
            **{k: v for k, v in func_kwargs.items() if k != "quiet"},
        ),
    )
    monkeypatch.setattr(
        commands, "pretty_print_reviews", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(commands, "save_output", lambda *_args, **_kwargs: None)

    commands.run_review_code(engine, args, runtime)

    assert calls == [False]


def test_finalize_review_output_runs_llm_triage_and_writes_json(monkeypatch, tmp_path):
    triage_path = tmp_path / "triaged.json"
    calls = []

    class _Engine:
        def llm_triage_reviews(self, results, **kwargs):
            calls.append((results, kwargs))
            return {
                "summary": {
                    "total_input_findings": 1,
                    "kept_findings": 1,
                    "filtered_findings": 0,
                },
                "issues": [{"id": "F001", "priority": "p1"}],
            }

    args = SimpleNamespace(
        verbose=False,
        quiet=True,
        triage=False,
        output_file=[str(tmp_path / "review.json")],
        llm_triage=True,
        llm_triage_model="gpt-test",
        llm_triage_reasoning_effort="high",
        llm_triage_batch_size=10,
        llm_triage_output_file=str(triage_path),
    )
    runtime = CommandRuntime(
        command="review_file",
        command_args=["a.c"],
        use_retrieval_context=True,
    )
    results = {"reviews": [{"file": "a.c", "reviews": [{"issue": "bug"}]}]}

    monkeypatch.setattr(commands, "pretty_print_reviews", lambda *_args: None)
    monkeypatch.setattr(commands, "save_output", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        commands,
        "with_spinner",
        lambda _message, func, *func_args, **func_kwargs: func(
            *func_args,
            **{k: v for k, v in func_kwargs.items() if k != "quiet"},
        ),
    )

    commands._finalize_review_output(_Engine(), results, args, runtime)

    assert calls[0][0] == results
    assert calls[0][1]["model"] == "gpt-test"
    assert calls[0][1]["reasoning_effort"] == "high"
    assert calls[0][1]["batch_size"] == 10
    assert json.loads(triage_path.read_text(encoding="utf-8"))["issues"] == [
        {"id": "F001", "priority": "p1"}
    ]


def test_finalize_review_output_saves_llm_triaged_review_report(
    monkeypatch, tmp_path
):
    triage_path = tmp_path / "triaged.json"
    review_path = tmp_path / "review.json"
    captured = {}

    class _Engine:
        def llm_triage_reviews(self, results, **kwargs):
            return {
                "summary": {
                    "total_input_findings": 2,
                    "kept_findings": 1,
                    "kept_input_findings": 1,
                    "filtered_findings": 1,
                    "additional_findings": 0,
                    "errors": [],
                },
                "issues": [
                    {
                        "id": "F001",
                        "priority": "p1",
                        "file": "a.c",
                        "file_path": "/repo/a.c",
                        "line_number": 7,
                        "issue": "Kept issue",
                        "severity": "High",
                    }
                ],
                "filtered_issues": [
                    {
                        "id": "F002",
                        "priority": "p5",
                        "file": "a.c",
                        "file_path": "/repo/a.c",
                        "line_number": 9,
                        "issue": "Filtered issue",
                        "llm_triage_reason": "Duplicate.",
                    }
                ],
            }

    args = SimpleNamespace(
        verbose=False,
        quiet=True,
        triage=False,
        output_file=[str(review_path)],
        llm_triage=True,
        llm_triage_model="gpt-test",
        llm_triage_reasoning_effort="high",
        llm_triage_batch_size=10,
        llm_triage_output_file=str(triage_path),
    )
    runtime = CommandRuntime(
        command="review_file",
        command_args=["a.c"],
        use_retrieval_context=True,
    )
    results = {
        "reviews": [
            {
                "file": "a.c",
                "file_path": "/repo/a.c",
                "reviews": [
                    {"issue": "Kept issue", "line_number": 7},
                    {"issue": "Filtered issue", "line_number": 9},
                ],
            }
        ]
    }

    monkeypatch.setattr(
        commands,
        "pretty_print_reviews",
        lambda data, *_args: captured.setdefault("printed", data),
    )
    monkeypatch.setattr(
        commands,
        "save_output",
        lambda _files, data, *_args, **_kwargs: captured.setdefault("saved", data),
    )
    monkeypatch.setattr(
        commands,
        "with_spinner",
        lambda _message, func, *func_args, **func_kwargs: func(
            *func_args,
            **{k: v for k, v in func_kwargs.items() if k != "quiet"},
        ),
    )

    commands._finalize_review_output(_Engine(), results, args, runtime)

    saved = captured["saved"]
    assert saved == captured["printed"]
    assert saved["llm_triage_summary"]["filtered_findings"] == 1
    assert saved["reviews"] == [
        {
            "file": "a.c",
            "file_path": "/repo/a.c",
            "reviews": [
                {
                    "id": "F001",
                    "priority": "p1",
                    "file": "a.c",
                    "file_path": "/repo/a.c",
                    "line_number": 7,
                    "issue": "Kept issue",
                    "severity": "High",
                }
            ],
            "llm_triage_filtered_reviews": [
                {
                    "id": "F002",
                    "priority": "p5",
                    "file": "a.c",
                    "file_path": "/repo/a.c",
                    "line_number": 9,
                    "issue": "Filtered issue",
                    "llm_triage_reason": "Duplicate.",
                }
            ],
        }
    ]
    assert saved["llm_triage_filtered_issues"] == [
        {
            "id": "F002",
            "priority": "p5",
            "file": "a.c",
            "file_path": "/repo/a.c",
            "line_number": 9,
            "issue": "Filtered issue",
            "llm_triage_reason": "Duplicate.",
        }
    ]
    assert json.loads(triage_path.read_text(encoding="utf-8"))["summary"][
        "filtered_findings"
    ] == 1
