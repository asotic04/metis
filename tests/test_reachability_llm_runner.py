# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from metis.engine.reachability.llm_runner import _chat_model_kwargs


def test_structured_reachability_omits_reasoning_effort():
    usage_runtime = SimpleNamespace(
        hooks=SimpleNamespace(chat_model_kwargs=lambda: {"callbacks": []})
    )

    kwargs = _chat_model_kwargs(
        usage_runtime, reasoning_effort="high", structured_output=True
    )

    assert kwargs == {"callbacks": []}


def test_plain_reachability_keeps_reasoning_effort():
    usage_runtime = SimpleNamespace(
        hooks=SimpleNamespace(chat_model_kwargs=lambda: {"callbacks": []})
    )

    kwargs = _chat_model_kwargs(
        usage_runtime, reasoning_effort="high", structured_output=False
    )

    assert kwargs == {"callbacks": [], "reasoning_effort": "high"}
