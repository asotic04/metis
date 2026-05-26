# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from metis.utils import parse_json_output

from .reachability.source_context import _read_line_context, _read_named_function_body

DEFAULT_LLM_TRIAGE_MODEL = "gpt-5.5"
DEFAULT_LLM_TRIAGE_REASONING_EFFORT = "high"
DEFAULT_LLM_TRIAGE_BATCH_SIZE = 10
DEFAULT_LLM_TRIAGE_MAX_TOKENS = 12000
DEFAULT_LLM_TRIAGE_MAX_SEARCH_MATCHES = 24

_PRIORITY_RANK = {"p0": 0, "p1": 1, "p2": 2, "p3": 3, "p4": 4, "p5": 5}
_SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}
_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}|CWE-\d+", re.IGNORECASE)
_SEARCH_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".py",
    ".go",
    ".rs",
    ".js",
    ".ts",
}
_SEARCH_SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
    "__pycache__",
    "build",
    "chromadb",
    "dist",
    "node_modules",
    "results",
}
_STOPWORDS = {
    "and",
    "are",
    "can",
    "code",
    "could",
    "file",
    "for",
    "from",
    "has",
    "into",
    "issue",
    "line",
    "may",
    "not",
    "that",
    "the",
    "this",
    "with",
}

_TRIAGE_SYSTEM_PROMPT = """\
You are a senior security triage reviewer for C/C++ and systems code.

Triage the supplied security review findings as exploitable vulnerability candidates.
You are not the original reporter. Your job is to reduce noise and keep only the
strongest independent security bugs. Use the code context, path context, evidence,
severity, and reasoning. Be strict.

Metis default priority rubric:
- p0: emergency response. Reserve this for an issue that can take down the service,
      disable a required security boundary, or break a must-have workflow for nearly
      all affected deployments with no practical mitigation.
- p1: urgent fix. Use this when exploitation is credible and the bug can cause major
      compromise, broad operational disruption, or block important dependent work;
      mitigations, if present, are incomplete or difficult to apply.
- p2: normal security priority. This is the default for kept, real security issues.
      Use it for important vulnerabilities with credible impact, issues that would be
      p0/p1 after removing a practical mitigation, deployment/setup blockers, or bugs
      that materially slow dependent teams.
- p3: backlog security work. Use this for real issues with limited blast radius,
      meaningful prerequisites, straightforward mitigations, or impact that is
      important but not currently blocking users or dependent teams.
- p4: low-priority follow-up. Use this for real but low-impact security hardening,
      niche edge cases, defense-in-depth work, or issues whose practical impact is
      currently small or indirect.
- p5: Metis-only filtered state, not an issue-tracker priority. Use p5 for false
      positives, duplicates, wrong code interpretation, non-exploitable code-quality
      issues, missing prerequisites, or findings that are not security vulnerabilities.

Assign p0-p4 by expected exploitability, security impact, affected deployment scope,
available mitigations, and urgency. Do not rank by CWE class alone. Default to p2 for
a kept real security issue unless the evidence justifies higher urgency or lower
practical impact.

Classify reliability-only crashes, generic missing validation in internal helpers, unchecked
allocation failures, development-only configuration issues, and theoretical resource exhaustion
as p5 unless the batch evidence shows a realistic attacker-controlled path and security impact.
Do not keep multiple findings for the same root cause; keep the clearest representative and mark
the rest p5 with duplicate_of. Do not keep every finding in a batch unless all are independent,
directly evidenced vulnerabilities. Return JSON only.
"""

_TRIAGE_USER_PROMPT = """\
Review this batch of related findings. The findings were grouped by similarity so duplicates
and near-duplicates should appear together.

For each finding id, return one decision object with:
- id: the finding id from the input
- priority: one of p0, p1, p2, p3, p4, p5
- keep: true only for p0-p4 real security findings
- duplicate_of: another finding id when this is a duplicate, otherwise null
- reason: concise justification for the priority or p5 filtering decision
- exploitability: concise statement of required attacker control and prerequisites

Output schema:
{{
  "decisions": [
    {{
      "id": "F001",
      "priority": "p1",
      "keep": true,
      "duplicate_of": null,
      "reason": "...",
      "exploitability": "..."
    }}
  ],
  "additional_findings": [
    {{
      "priority": "p2",
      "file": "src/example.c",
      "line_number": 42,
      "issue": "Newly identified security issue title",
      "severity": "High",
      "confidence": 0.85,
      "cwe": "CWE-120",
      "reasoning": "Why this is a real issue, tied to provided code/search evidence.",
      "mitigation": "Concrete fix."
    }}
  ]
}}

Only add additional_findings when the provided code context or repo search evidence directly
supports a real security issue not already represented by an input finding. Do not speculate
from names alone.

Batch:
{batch_json}
"""


@dataclass
class _FindingRecord:
    id: str
    file: str
    file_path: str
    line_number: int
    issue: dict[str, Any]
    code_context: str
    search_evidence: list[dict[str, Any]] = field(default_factory=list)
    tokens: set[str] = field(default_factory=set)


class LlmTriageService:
    def __init__(self, *, codebase_path, llm_provider, usage_runtime):
        self._codebase_path = str(codebase_path)
        self._llm_provider = llm_provider
        self._usage_runtime = usage_runtime

    def triage_review_results(
        self,
        results: dict[str, Any],
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        batch_size: int = DEFAULT_LLM_TRIAGE_BATCH_SIZE,
        max_tokens: int = DEFAULT_LLM_TRIAGE_MAX_TOKENS,
        progress_callback=None,
    ) -> dict[str, Any]:
        findings = self._flatten_findings(results)
        model = model or DEFAULT_LLM_TRIAGE_MODEL
        reasoning_effort = reasoning_effort or DEFAULT_LLM_TRIAGE_REASONING_EFFORT
        batch_size = max(1, int(batch_size or DEFAULT_LLM_TRIAGE_BATCH_SIZE))

        if not findings:
            return self._build_payload(
                [],
                {},
                [],
                model=model,
                reasoning_effort=reasoning_effort,
                batch_size=batch_size,
                errors=[],
            )

        batches = _similarity_batches(findings, batch_size)
        decisions_by_id: dict[str, dict[str, Any]] = {}
        additional_findings: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        if progress_callback:
            progress_callback(
                {
                    "event": "llm_triage_start",
                    "findings": len(findings),
                    "batches": len(batches),
                    "batch_size": batch_size,
                }
            )

        for batch_index, batch in enumerate(batches, start=1):
            batch_ids = [finding.id for finding in batch]
            if progress_callback:
                progress_callback(
                    {
                        "event": "llm_triage_batch_start",
                        "batch": batch_index,
                        "batches": len(batches),
                        "findings": len(batch),
                    }
                )
            try:
                raw = _invoke_triage_prompt(
                    self._llm_provider,
                    self._usage_runtime,
                    model=model,
                    max_tokens=max_tokens,
                    variables={
                        "batch_json": json.dumps(_batch_payload(batch), indent=2)
                    },
                    reasoning_effort=reasoning_effort,
                    temperature=0.0,
                )
                decisions, additions = _parse_triage_response(raw, batch)
                additional_findings.extend(additions)
            except Exception as exc:  # pragma: no cover
                errors.append(
                    {
                        "batch": batch_index,
                        "ids": batch_ids,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                decisions = {
                    finding.id: _fallback_decision(
                        finding.id,
                        "LLM triage failed; kept conservatively as p4.",
                    )
                    for finding in batch
                }

            for finding in batch:
                decisions_by_id[finding.id] = decisions.get(
                    finding.id,
                    _fallback_decision(
                        finding.id,
                        "LLM triage omitted this finding; kept conservatively as p4.",
                    ),
                )

            if progress_callback:
                kept = sum(
                    1
                    for decision in decisions.values()
                    if _normalize_priority(decision.get("priority")) != "p5"
                    and bool(decision.get("keep", True))
                )
                progress_callback(
                    {
                        "event": "llm_triage_batch_done",
                        "batch": batch_index,
                        "batches": len(batches),
                        "kept": kept,
                    }
                )

        payload = self._build_payload(
            findings,
            decisions_by_id,
            additional_findings,
            model=model,
            reasoning_effort=reasoning_effort,
            batch_size=batch_size,
            errors=errors,
        )
        if progress_callback:
            summary = payload["summary"]
            progress_callback(
                {
                    "event": "llm_triage_done",
                    "findings": summary["total_input_findings"],
                    "kept": summary["kept_findings"],
                    "filtered": summary["filtered_findings"],
                    "additional": summary["additional_findings"],
                }
            )
        return payload

    def _flatten_findings(self, results: dict[str, Any]) -> list[_FindingRecord]:
        records: list[_FindingRecord] = []
        reviews = results.get("reviews") if isinstance(results, dict) else None
        if not isinstance(reviews, list):
            return records

        next_id = 1
        for file_entry in reviews:
            if not isinstance(file_entry, dict):
                continue
            issues = file_entry.get("reviews")
            if not isinstance(issues, list):
                continue
            file_name = str(file_entry.get("file") or "")
            file_path = str(file_entry.get("file_path") or "")
            for issue in issues:
                if not isinstance(issue, dict):
                    continue
                issue_copy = copy.deepcopy(issue)
                primary_file = _primary_file(
                    issue_copy,
                    file_name,
                    file_path,
                    self._codebase_path,
                )
                primary_path = _absolute_path(
                    self._codebase_path, primary_file, file_path
                )
                line_number = _line_number(issue_copy)
                code_context = self._code_context(issue_copy, primary_file, line_number)
                record = _FindingRecord(
                    id=f"F{next_id:03d}",
                    file=primary_file,
                    file_path=primary_path,
                    line_number=line_number,
                    issue=issue_copy,
                    code_context=code_context,
                    search_evidence=self._repo_search_evidence(issue_copy),
                )
                record.tokens = _tokens_for_record(record)
                records.append(record)
                next_id += 1
        return records

    def _code_context(
        self, issue: dict[str, Any], rel_file: str, line_number: int
    ) -> str:
        parts: list[str] = []
        snippet = str(issue.get("code_snippet") or "").strip()
        if snippet:
            parts.append(f"Reported snippet:\n{snippet[:2000]}")

        if rel_file:
            line_context = _read_line_context(
                self._codebase_path,
                rel_file,
                line_number,
                context=5,
                max_chars=2500,
            )
            if line_context:
                parts.append(f"Nearby lines:\n{line_context}")

            primary_function = _function_name(issue.get("primary_function"))
            if primary_function:
                body = _read_named_function_body(
                    self._codebase_path,
                    rel_file,
                    primary_function,
                    near_line=line_number,
                    max_chars=5000,
                )
                if body:
                    parts.append(f"Primary function body:\n{body}")

        for path_context in self._path_context(issue, rel_file):
            parts.append(path_context)

        context = "\n\n".join(parts)
        return context[:10000]

    def _path_context(self, issue: dict[str, Any], rel_file: str) -> list[str]:
        path = issue.get("path")
        if not isinstance(path, list):
            return []
        contexts: list[str] = []
        seen: set[tuple[str, str]] = set()
        for entry in path[:6]:
            path_file, function = _split_path_function(str(entry or ""))
            if not path_file or not function:
                continue
            key = (path_file, function)
            if key in seen or path_file == rel_file:
                continue
            seen.add(key)
            body = _read_named_function_body(
                self._codebase_path,
                path_file,
                function,
                near_line=1,
                max_chars=3000,
            )
            if body:
                contexts.append(f"Related path function {entry}:\n{body}")
        return contexts[:3]

    def _repo_search_evidence(self, issue: dict[str, Any]) -> list[dict[str, Any]]:
        queries = _search_queries_for_issue(issue)
        if not queries:
            return []
        return _search_codebase(
            self._codebase_path,
            queries,
            max_matches=DEFAULT_LLM_TRIAGE_MAX_SEARCH_MATCHES,
        )

    def _build_payload(
        self,
        findings: list[_FindingRecord],
        decisions_by_id: dict[str, dict[str, Any]],
        additional_findings: list[dict[str, Any]],
        *,
        model: str,
        reasoning_effort: str,
        batch_size: int,
        errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        kept = []
        dropped = 0
        seen_duplicate_keys: set[tuple[Any, ...]] = set()

        for finding in findings:
            decision = decisions_by_id.get(
                finding.id, _fallback_decision(finding.id, "")
            )
            priority = _normalize_priority(decision.get("priority"))
            keep = bool(decision.get("keep", priority != "p5"))
            duplicate_of = _clean_optional_text(decision.get("duplicate_of"))
            if priority == "p5" or not keep or duplicate_of:
                dropped += 1
                continue
            duplicate_key = _dedupe_key(finding)
            if duplicate_key in seen_duplicate_keys:
                dropped += 1
                continue
            seen_duplicate_keys.add(duplicate_key)
            kept.append(_triaged_issue(finding, decision, priority))

        next_additional_id = 1
        for additional in additional_findings:
            priority = _normalize_priority(additional.get("priority"))
            if priority == "p5":
                continue
            issue = _additional_issue(additional, next_additional_id, priority)
            if issue is None:
                continue
            next_additional_id += 1
            duplicate_key = _dedupe_key_for_values(
                issue.get("file"),
                issue.get("line_number"),
                issue.get("issue"),
            )
            if duplicate_key in seen_duplicate_keys:
                continue
            seen_duplicate_keys.add(duplicate_key)
            kept.append(issue)

        kept, collapsed_duplicates = _consolidate_triaged_issues(kept)
        dropped += collapsed_duplicates
        kept.sort(key=_triaged_issue_sort_key)
        kept_input_findings = sum(
            1 for issue in kept if not _is_additional_issue(issue)
        )
        kept_additional_findings = sum(1 for issue in kept if _is_additional_issue(issue))
        filtered_findings = max(0, len(findings) - kept_input_findings)
        return {
            "summary": {
                "schema_version": 1,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "batch_size": batch_size,
                "total_input_findings": len(findings),
                "kept_findings": len(kept),
                "kept_input_findings": kept_input_findings,
                "filtered_findings": filtered_findings,
                "additional_findings": kept_additional_findings,
                "errors": errors,
            },
            "issues": kept,
        }


def _batch_payload(batch: list[_FindingRecord]) -> list[dict[str, Any]]:
    payload = []
    for finding in batch:
        issue = finding.issue
        payload.append(
            {
                "id": finding.id,
                "file": finding.file,
                "line_number": finding.line_number,
                "issue": issue.get("issue") or issue.get("title") or "",
                "severity": issue.get("severity"),
                "confidence": issue.get("confidence"),
                "cwe": issue.get("cwe"),
                "primary_function": issue.get("primary_function"),
                "analysis_type": issue.get("analysis_type"),
                "path": issue.get("path") or [],
                "reasoning": issue.get("reasoning") or "",
                "mitigation": issue.get("mitigation") or "",
                "code_context": finding.code_context,
                "repo_search_evidence": finding.search_evidence,
            }
        )
    return payload


def _invoke_triage_prompt(
    llm_provider,
    usage_runtime,
    *,
    model: str,
    max_tokens: int,
    variables: dict[str, Any],
    reasoning_effort: str | None = None,
    temperature: float = 0.0,
) -> str:
    kwargs = _triage_chat_model_kwargs(
        usage_runtime, reasoning_effort=reasoning_effort
    )
    chat = llm_provider.get_chat_model(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", _TRIAGE_SYSTEM_PROMPT), ("user", _TRIAGE_USER_PROMPT)]
    )
    return (prompt | chat | StrOutputParser()).invoke(variables).strip()


def _triage_chat_model_kwargs(usage_runtime, *, reasoning_effort=None) -> dict[str, Any]:
    hooks = getattr(usage_runtime, "hooks", None)
    kwargs = hooks.chat_model_kwargs() if hooks is not None else {}
    if reasoning_effort and str(reasoning_effort).lower() not in {
        "none",
        "off",
        "false",
        "default",
    }:
        kwargs["reasoning_effort"] = reasoning_effort
    return kwargs


def _parse_triage_response(
    raw: Any, batch: list[_FindingRecord]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if hasattr(raw, "model_dump"):
        parsed = raw.model_dump()
    elif isinstance(raw, dict):
        parsed = raw
    else:
        parsed = parse_json_output(raw)
    if not isinstance(parsed, dict):
        return {}, []
    decisions = parsed.get("decisions")
    if not isinstance(decisions, list):
        decisions = []
    valid_ids = {finding.id for finding in batch}
    normalized: dict[str, dict[str, Any]] = {}
    for item in decisions:
        if not isinstance(item, dict):
            continue
        finding_id = str(item.get("id") or item.get("finding_id") or "").strip()
        if finding_id not in valid_ids:
            continue
        priority = _normalize_priority(item.get("priority"))
        keep = item.get("keep")
        if not isinstance(keep, bool):
            keep = priority != "p5"
        normalized[finding_id] = {
            "id": finding_id,
            "priority": priority,
            "keep": keep,
            "duplicate_of": _clean_optional_text(item.get("duplicate_of")),
            "reason": str(item.get("reason") or "").strip(),
            "exploitability": str(item.get("exploitability") or "").strip(),
        }
    additions = _parse_additional_findings(parsed.get("additional_findings"))
    return normalized, additions


def _parse_additional_findings(raw_items) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []
    additions = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        priority = _normalize_priority(item.get("priority"))
        if priority == "p5":
            continue
        file_name = str(item.get("file") or item.get("primary_file") or "").strip()
        issue = str(item.get("issue") or item.get("title") or "").strip()
        if not file_name or not issue:
            continue
        additions.append(
            {
                "priority": priority,
                "file": file_name.replace("\\", "/").lstrip("./"),
                "line_number": _safe_positive_int(item.get("line_number"), 1),
                "issue": issue,
                "severity": str(item.get("severity") or "Medium").strip(),
                "confidence": _confidence_value(item.get("confidence") or 0.75),
                "cwe": item.get("cwe"),
                "reasoning": str(item.get("reasoning") or "").strip(),
                "mitigation": str(item.get("mitigation") or "").strip(),
            }
        )
    return additions


def _fallback_decision(finding_id: str, reason: str) -> dict[str, Any]:
    return {
        "id": finding_id,
        "priority": "p4",
        "keep": True,
        "duplicate_of": None,
        "reason": reason,
        "exploitability": "Not assessed by LLM triage.",
    }


def _similarity_batches(
    findings: list[_FindingRecord],
    batch_size: int,
) -> list[list[_FindingRecord]]:
    remaining = sorted(findings, key=_finding_seed_sort_key)
    batches: list[list[_FindingRecord]] = []
    while remaining:
        seed = remaining.pop(0)
        ranked = sorted(
            remaining,
            key=lambda finding: (
                -_jaccard(seed.tokens, finding.tokens),
                _finding_seed_sort_key(finding),
            ),
        )
        batch = [seed] + ranked[: batch_size - 1]
        selected_ids = {finding.id for finding in batch}
        remaining = [finding for finding in remaining if finding.id not in selected_ids]
        batches.append(batch)
    return batches


def _tokens_for_record(finding: _FindingRecord) -> set[str]:
    issue = finding.issue
    text = " ".join(
        str(part or "")
        for part in (
            finding.file,
            issue.get("issue"),
            issue.get("primary_function"),
            issue.get("analysis_type"),
            issue.get("cwe"),
            issue.get("severity"),
            issue.get("reasoning"),
            issue.get("mitigation"),
            " ".join(str(p) for p in issue.get("path") or []),
        )
    )
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if token.lower() not in _STOPWORDS
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _finding_seed_sort_key(finding: _FindingRecord):
    issue = finding.issue
    severity = _SEVERITY_RANK.get(str(issue.get("severity") or "").lower(), 9)
    confidence = _confidence_value(issue.get("confidence"))
    return (severity, -confidence, finding.file, finding.line_number, finding.id)


def _triaged_issue_sort_key(issue: dict[str, Any]):
    return (
        _PRIORITY_RANK.get(str(issue.get("priority") or "p5").lower(), 5),
        _SEVERITY_RANK.get(str(issue.get("severity") or "").lower(), 9),
        -_confidence_value(issue.get("confidence")),
        str(issue.get("file") or ""),
        int(issue.get("line_number") or 0),
        str(issue.get("issue") or ""),
    )


def _triaged_issue(
    finding: _FindingRecord,
    decision: dict[str, Any],
    priority: str,
) -> dict[str, Any]:
    item = copy.deepcopy(finding.issue)
    item["id"] = finding.id
    item["priority"] = priority
    item["file"] = finding.file
    item["file_path"] = finding.file_path
    item["line_number"] = finding.line_number
    item["llm_triage_reason"] = str(decision.get("reason") or "").strip()
    item["llm_triage_exploitability"] = str(
        decision.get("exploitability") or ""
    ).strip()
    item["llm_triage_duplicate_of"] = _clean_optional_text(decision.get("duplicate_of"))
    return item


def _additional_issue(
    additional: dict[str, Any],
    index: int,
    priority: str,
) -> dict[str, Any] | None:
    file_name = str(additional.get("file") or "").strip().replace("\\", "/")
    issue = str(additional.get("issue") or "").strip()
    if not file_name or not issue:
        return None
    return {
        "id": f"A{index:03d}",
        "priority": priority,
        "file": file_name.lstrip("./"),
        "file_path": file_name.lstrip("./"),
        "line_number": _safe_positive_int(additional.get("line_number"), 1),
        "issue": issue,
        "severity": str(additional.get("severity") or "Medium").strip() or "Medium",
        "confidence": _confidence_value(additional.get("confidence")),
        "cwe": additional.get("cwe"),
        "reasoning": str(additional.get("reasoning") or "").strip(),
        "mitigation": str(additional.get("mitigation") or "").strip(),
        "llm_triage_reason": "Additional finding identified during LLM triage.",
        "llm_triage_exploitability": str(additional.get("reasoning") or "").strip(),
        "llm_triage_duplicate_of": None,
        "llm_triage_source": "additional_finding",
    }


def _primary_file(
    issue: dict[str, Any],
    file_name: str,
    file_path: str,
    codebase_path: str,
) -> str:
    candidate = str(
        issue.get("primary_file")
        or issue.get("file")
        or issue.get("sink_file")
        or issue.get("source_file")
        or file_name
        or ""
    )
    if candidate:
        return _relative_path(candidate, codebase_path)
    if file_path:
        return _relative_path(file_path, codebase_path)
    return ""


def _absolute_path(codebase_path: str, rel_file: str, fallback_path: str) -> str:
    if fallback_path and os.path.isabs(str(fallback_path)):
        return str(fallback_path)
    if rel_file:
        return os.path.join(codebase_path, rel_file)
    return str(fallback_path or "")


def _relative_path(path: str, codebase_path: str) -> str:
    path = str(path or "").replace("\\", "/")
    if not path:
        return ""
    if codebase_path and os.path.isabs(path):
        try:
            return os.path.relpath(path, codebase_path).replace("\\", "/")
        except ValueError:
            return path
    return path.lstrip("./")


def _line_number(issue: dict[str, Any]) -> int:
    for line_field in ("line_number", "primary_line", "sink_line", "source_line"):
        value = _safe_positive_int(issue.get(line_field), 0)
        if value > 0:
            return value
    return 1


def _safe_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _function_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split("::")[-1]


def _split_path_function(value: str) -> tuple[str, str]:
    if "::" not in value:
        return "", ""
    file_name, function = value.rsplit("::", 1)
    return file_name.replace("\\", "/").lstrip("./"), _function_name(function)


def _normalize_priority(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in _PRIORITY_RANK:
        return text
    return "p5"


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "false"}:
        return None
    return text


def _confidence_value(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value or "").strip().lower()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return {"high": 0.95, "medium": 0.75, "low": 0.5}.get(text, 0.0)


def _dedupe_key(finding: _FindingRecord) -> tuple[Any, ...]:
    return _dedupe_key_for_values(
        finding.file,
        finding.line_number,
        finding.issue.get("issue"),
    )


def _dedupe_key_for_values(
    file_name: Any, line_number: Any, issue: Any
) -> tuple[Any, ...]:
    issue_text = re.sub(
        r"\s+",
        " ",
        str(issue or "").strip().lower(),
    )
    return (
        str(file_name or "").replace("\\", "/").lstrip("./"),
        _safe_positive_int(line_number, 0),
        issue_text,
    )


def _consolidate_triaged_issues(
    issues: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    removed_input_findings = 0
    for issue in sorted(issues, key=_triaged_issue_sort_key):
        duplicate_of = _find_duplicate_issue(issue, selected)
        if duplicate_of is None:
            selected.append(issue)
            continue
        if not str(issue.get("id") or "").startswith("A"):
            removed_input_findings += 1
    return selected, removed_input_findings


def _find_duplicate_issue(
    issue: dict[str, Any], selected: list[dict[str, Any]]
) -> dict[str, Any] | None:
    for existing in selected:
        if _issues_are_near_duplicates(issue, existing):
            return existing
    return None


def _issues_are_near_duplicates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    file_a = _issue_file(a)
    file_b = _issue_file(b)
    if not file_a or not file_b:
        return False

    family_a = _issue_family(a)
    family_b = _issue_family(b)
    same_family = bool(family_a and family_a == family_b)
    if file_a != file_b:
        return _issues_are_cross_file_duplicates(a, b, family_a, family_b)

    canonical_a = _canonical_key(a)
    canonical_b = _canonical_key(b)
    if canonical_a and canonical_a == canonical_b:
        return True

    line_a = _safe_positive_int(a.get("line_number"), 0)
    line_b = _safe_positive_int(b.get("line_number"), 0)
    function_a = _issue_function(a)
    function_b = _issue_function(b)
    same_function = bool(function_a and function_a == function_b)
    line_distance = abs(line_a - line_b) if line_a and line_b else 9999

    # Same primary statement: keep the strongest representative. This catches
    # common model/report variants such as null-deref plus overflow on one call.
    if line_a and line_a == line_b and (same_function or not function_a or not function_b):
        return True

    similarity = _issue_similarity(a, b)

    if same_function and line_distance <= 2 and similarity >= 0.30:
        return True
    if same_function and line_distance <= 5 and same_family and similarity >= 0.24:
        return True
    if same_family and similarity >= 0.50:
        return True

    canonical_similarity = _canonical_similarity(canonical_a, canonical_b)
    return bool(same_family and canonical_similarity >= 0.40)


_CROSS_FILE_DUPLICATE_FAMILIES = frozenset(
    {
        "authorization",
        "command_injection",
        "format_string",
        "path_traversal",
        "sql_injection",
        "unsafe_deserialization",
    }
)


def _issues_are_cross_file_duplicates(
    a: dict[str, Any],
    b: dict[str, Any],
    family_a: str,
    family_b: str,
) -> bool:
    if not family_a or family_a != family_b:
        return False
    if family_a not in _CROSS_FILE_DUPLICATE_FAMILIES:
        return False

    tokens_a = _duplicate_tokens(a)
    tokens_b = _duplicate_tokens(b)
    shared = tokens_a & tokens_b
    if family_a == "format_string" and {"util_log", "vprintf"} & shared:
        return True
    return _jaccard(tokens_a, tokens_b) >= 0.50


def _issue_file(issue: dict[str, Any]) -> str:
    return (
        str(issue.get("file") or issue.get("primary_file") or "")
        .replace("\\", "/")
        .lstrip("./")
    )


def _issue_function(issue: dict[str, Any]) -> str:
    return _function_name(issue.get("primary_function") or issue.get("sink_function"))


_CANONICAL_KEY_RE = re.compile(r"Canonical key:\s*([^\n\r]+)")


def _canonical_key(issue: dict[str, Any]) -> str:
    explicit = str(issue.get("canonical_key") or "").strip()
    if explicit:
        return explicit.lower()
    reasoning = str(issue.get("reasoning") or "")
    match = _CANONICAL_KEY_RE.search(reasoning)
    return match.group(1).strip().lower() if match else ""


def _issue_family(issue: dict[str, Any]) -> str:
    canonical = _canonical_key(issue)
    if canonical:
        parts = canonical.split(":")
        if len(parts) >= 3 and parts[2]:
            return _normalise_family(parts[2])

    cwe = str(issue.get("cwe") or "").upper()
    cwe_family = {
        "CWE-22": "path_traversal",
        "CWE-78": "command_injection",
        "CWE-89": "sql_injection",
        "CWE-120": "memory_bounds",
        "CWE-125": "memory_bounds",
        "CWE-134": "format_string",
        "CWE-190": "integer_overflow",
        "CWE-191": "integer_overflow",
        "CWE-200": "information_disclosure",
        "CWE-252": "unchecked_error",
        "CWE-256": "credential_storage",
        "CWE-285": "authorization",
        "CWE-287": "authentication",
        "CWE-404": "lifetime",
        "CWE-415": "lifetime",
        "CWE-416": "lifetime",
        "CWE-476": "null_deref",
        "CWE-502": "unsafe_deserialization",
        "CWE-639": "authorization",
        "CWE-664": "lifetime",
        "CWE-696": "state_order",
        "CWE-787": "memory_bounds",
        "CWE-798": "hardcoded_secret",
        "CWE-862": "authorization",
        "CWE-863": "authorization",
        "CWE-911": "lifetime",
        "CWE-1188": "unsafe_deployment",
    }.get(cwe)
    if cwe_family:
        return cwe_family

    text = _issue_text(issue).lower()
    keyword_families = (
        ("sql", "sql_injection"),
        ("command injection", "command_injection"),
        ("path traversal", "path_traversal"),
        ("pickle", "unsafe_deserialization"),
        ("deserialize", "unsafe_deserialization"),
        ("format string", "format_string"),
        ("double free", "lifetime"),
        ("use-after-free", "lifetime"),
        ("use after free", "lifetime"),
        ("refcount", "lifetime"),
        ("out-of-bounds", "memory_bounds"),
        ("out of bounds", "memory_bounds"),
        ("buffer", "memory_bounds"),
        ("integer overflow", "integer_overflow"),
        ("auth", "authorization"),
        ("permission", "authorization"),
        ("hardcoded", "hardcoded_secret"),
    )
    for keyword, family in keyword_families:
        if keyword in text:
            return family
    return ""


def _normalise_family(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "double_free": "lifetime",
        "partial_cleanup": "lifetime",
        "stale_metadata": "lifetime",
        "refcount_mismatch": "lifetime",
        "use_after_free": "lifetime",
        "memory_bounds": "memory_bounds",
        "out_of_bounds": "memory_bounds",
        "missing_validation": "memory_bounds",
        "missing_auth": "authorization",
        "permission_mismatch": "authorization",
        "auth_logic_error": "authorization",
        "state_ordering": "state_order",
    }
    return aliases.get(text, text)


def _issue_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    return _jaccard(_duplicate_tokens(a), _duplicate_tokens(b))


def _canonical_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return _jaccard(_token_set(a), _token_set(b))


def _duplicate_tokens(issue: dict[str, Any]) -> set[str]:
    return _token_set(_issue_text(issue))


def _issue_text(issue: dict[str, Any]) -> str:
    return " ".join(
        str(issue.get(field) or "")
        for field in (
            "issue",
            "reasoning",
            "mitigation",
            "code_snippet",
            "primary_function",
            "analysis_type",
            "cwe",
            "llm_triage_reason",
            "llm_triage_exploitability",
        )
    )


_DUPLICATE_STOPWORDS = _STOPWORDS | {
    "analysis",
    "candidate",
    "canonical",
    "connected",
    "confidence",
    "context",
    "evidence",
    "finding",
    "function",
    "functions",
    "issue",
    "line",
    "location",
    "mitigation",
    "path",
    "primary",
    "reason",
    "root",
    "severity",
}


def _token_set(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(str(text or ""))
        if token.lower() not in _DUPLICATE_STOPWORDS
    }


def _is_additional_issue(issue: dict[str, Any]) -> bool:
    return str(issue.get("id") or "").startswith("A")


def _search_queries_for_issue(issue: dict[str, Any]) -> list[str]:
    candidates: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if not text:
            return
        text = text.split("::")[-1]
        if len(text) < 4:
            return
        if text not in candidates:
            candidates.append(text)

    add(issue.get("primary_function"))
    for path_entry in issue.get("path") or []:
        add(path_entry)

    text = " ".join(
        str(issue.get(field) or "")
        for field in ("issue", "reasoning", "mitigation", "analysis_type")
    )
    for token in _TOKEN_RE.findall(text):
        lowered = token.lower()
        if lowered in _STOPWORDS:
            continue
        if "_" not in token and len(token) < 8:
            continue
        add(token)
        if len(candidates) >= 10:
            break

    return candidates[:10]


def _search_codebase(
    codebase_path: str,
    queries: list[str],
    *,
    max_matches: int,
) -> list[dict[str, Any]]:
    normalized_queries = []
    for query in queries:
        text = str(query or "").strip()
        if text and text.lower() not in {q.lower() for q in normalized_queries}:
            normalized_queries.append(text)
    if not normalized_queries:
        return []

    matches: list[dict[str, Any]] = []
    per_query_counts = {query.lower(): 0 for query in normalized_queries}
    for rel_file, abs_file in _iter_search_files(codebase_path):
        try:
            with open(abs_file, "r", encoding="utf-8", errors="ignore") as handle:
                for line_number, line in enumerate(handle, start=1):
                    lowered = line.lower()
                    for query in normalized_queries:
                        query_key = query.lower()
                        if per_query_counts[query_key] >= 4:
                            continue
                        if query_key not in lowered:
                            continue
                        per_query_counts[query_key] += 1
                        matches.append(
                            {
                                "query": query,
                                "file": rel_file,
                                "line_number": line_number,
                                "line": line.strip()[:240],
                            }
                        )
                        if len(matches) >= max_matches:
                            return matches
                    if len(matches) >= max_matches:
                        return matches
        except OSError:
            continue
    return matches


def _iter_search_files(codebase_path: str):
    for root, dirs, files in os.walk(codebase_path):
        dirs[:] = [
            dirname
            for dirname in dirs
            if dirname not in _SEARCH_SKIP_DIRS and not dirname.startswith(".uv-")
        ]
        for file_name in files:
            suffix = os.path.splitext(file_name)[1].lower()
            if suffix not in _SEARCH_EXTENSIONS:
                continue
            abs_file = os.path.join(root, file_name)
            rel_file = os.path.relpath(abs_file, codebase_path).replace("\\", "/")
            yield rel_file, abs_file
