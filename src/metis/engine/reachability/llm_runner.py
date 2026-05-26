# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from metis.utils import parse_json_output


def _chat_model_kwargs(
    usage_runtime, *, reasoning_effort=None, structured_output=False
):
    hooks = getattr(usage_runtime, "hooks", None)
    kwargs = hooks.chat_model_kwargs() if hooks is not None else {}
    if structured_output:
        # Chat Completions rejects reasoning_effort when structured output uses
        # function tools; keep these calls compatible with that API path.
        kwargs.pop("reasoning_effort", None)
        return kwargs
    if reasoning_effort and str(reasoning_effort).lower() not in {
        "none",
        "off",
        "false",
        "default",
    }:
        kwargs["reasoning_effort"] = reasoning_effort
    return kwargs


def invoke_reachability_prompt(
    llm_provider,
    usage_runtime,
    *,
    model,
    max_tokens,
    system_prompt,
    user_prompt,
    variables,
    response_model,
    reasoning_effort=None,
    temperature=0.1,
):
    kwargs = _chat_model_kwargs(
        usage_runtime,
        reasoning_effort=reasoning_effort,
        structured_output=response_model is not None,
    )
    chat = llm_provider.get_chat_model(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", user_prompt)]
    )
    structured_model = chat.with_structured_output(
        response_model, method="function_calling"
    )
    return (prompt | structured_model).invoke(variables)


def reachability_response_payload(raw):
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = parse_json_output(raw)
        return parsed if isinstance(parsed, dict) else None
    return None
