# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger("metis")

_THREAT_MODEL_TOKEN_RE = re.compile(
    r"CWE-\d+|[A-Za-z_][A-Za-z0-9_]*(?:->[A-Za-z_][A-Za-z0-9_]*)?",
    re.IGNORECASE,
)

_THREAT_MODEL_KEYWORD_STOPWORDS = {
    "ability",
    "able",
    "about",
    "across",
    "actual",
    "add",
    "against",
    "all",
    "also",
    "and",
    "any",
    "api",
    "are",
    "around",
    "attacker",
    "based",
    "because",
    "before",
    "being",
    "bug",
    "bugs",
    "caller",
    "calls",
    "can",
    "capabilities",
    "capability",
    "case",
    "check",
    "code",
    "concrete",
    "context",
    "control",
    "could",
    "data",
    "direct",
    "directly",
    "does",
    "during",
    "each",
    "evidence",
    "exact",
    "file",
    "finding",
    "findings",
    "flow",
    "flows",
    "focus",
    "for",
    "from",
    "generic",
    "have",
    "impact",
    "include",
    "including",
    "input",
    "into",
    "issue",
    "issues",
    "keep",
    "likely",
    "local",
    "may",
    "missing",
    "model",
    "must",
    "named",
    "not",
    "only",
    "path",
    "paths",
    "pay",
    "project",
    "provided",
    "real",
    "report",
    "reported",
    "require",
    "scope",
    "security",
    "should",
    "shown",
    "special",
    "state",
    "still",
    "supported",
    "that",
    "the",
    "their",
    "these",
    "this",
    "threat",
    "through",
    "treat",
    "under",
    "use",
    "used",
    "user",
    "when",
    "where",
    "with",
    "within",
    "would",
}


def summarize_changes(llm_provider, file_path, issues, summary_prompt, callbacks=None):
    try:
        kwargs = {}
        if callbacks is not None:
            kwargs["callbacks"] = callbacks
        chat = llm_provider.get_chat_model(**kwargs)
        prompt_tmpl = ChatPromptTemplate.from_messages(
            [("system", "{system}"), ("user", "{input}")]
        )
        chain = prompt_tmpl | chat | StrOutputParser()
        return chain.invoke(
            {"system": summary_prompt or "", "input": issues or ""}
        ).strip()
    except Exception as e:
        logger.error(f"Error summarizing changes for {file_path}: {e}")
        return ""


def prepare_nodes_iter(
    code_docs,
    doc_docs,
    get_plugin_for_extension,
    get_splitter_cached,
    doc_splitter,
):
    """
    Generator that prepares nodes for code and docs
    """
    nodes_code = []
    nodes_docs = []

    for d in code_docs:
        ext = os.path.splitext(d.id_)[1].lower()
        plugin = get_plugin_for_extension(ext)
        if plugin:
            try:
                splitter = get_splitter_cached(plugin)
                parsed_nodes = splitter.get_nodes_from_documents([d])
                nodes_code.extend(parsed_nodes)
            except Exception as e:
                name = plugin.get_name() if hasattr(plugin, "get_name") else "unknown"
                logger.warning(
                    f"Could not parse code with language {name} for file {d.id_} (ext {ext}): {e}"
                )
        # yield regardless of success
        yield None

    for d in doc_docs:
        try:
            nodes_docs.extend(doc_splitter.get_nodes_from_documents([d]))
        except Exception as e:
            logger.warning(f"Could not parse docs for file {d.id_}: {e}")
        finally:
            yield None

    return nodes_code, nodes_docs


def apply_custom_guidance(base_prompt, custom_guidance, precedence_note):
    """Prepend precedence note and custom guidance to a base prompt.

    If custom_guidance is not set, returns base_prompt unchanged. The format is:
    [precedence_note]\n\nCustom Guidance:\n{custom_guidance}\n\n{base_prompt}
    """
    if not custom_guidance:
        return base_prompt
    guidance_block = f"Custom Guidance:\n{custom_guidance.strip()}"
    if precedence_note:
        return f"{precedence_note.strip()}\n\n{guidance_block}\n\n{base_prompt}"
    return f"{guidance_block}\n\n{base_prompt}"


def extract_threat_model_keywords(
    threat_model_text,
    explicit_keywords=None,
    *,
    limit=120,
):
    """Return ordered domain keywords derived from threat-model text and config.

    These keywords are used only as candidate-selection and prompt-focus hints. They
    are intentionally broad enough to capture function names, struct fields, CWE IDs,
    and domain terms from inline benchmark threat models, while dropping prose words
    that would make every function look relevant.
    """
    ordered: list[str] = []
    seen: set[str] = set()

    def add(value):
        text = str(value or "").strip().strip("`'\".,;:()[]{}")
        if not text:
            return
        key = text.lower()
        if key in seen or key in _THREAT_MODEL_KEYWORD_STOPWORDS:
            return
        if len(key) < 4 and not key.startswith("cwe-"):
            return
        ordered.append(text)
        seen.add(key)

    for keyword in explicit_keywords or ():
        add(keyword)

    for token in _THREAT_MODEL_TOKEN_RE.findall(str(threat_model_text or "")):
        lowered = token.lower()
        if (
            "_" not in token
            and "->" not in token
            and not lowered.startswith("cwe-")
            and len(token) < 7
        ):
            continue
        add(token)
        if len(ordered) >= limit:
            break

    return ordered[:limit]


def format_threat_model_guidance(threat_model_text):
    text = str(threat_model_text or "").strip()
    if not text:
        return ""
    return (
        "Project Threat Model:\n"
        f"{text}\n\n"
        "Threat-model handling rules:\n"
        "- Treat this threat model as authoritative project-specific security scope.\n"
        "- Actively specialize the review toward issue classes, data flows, APIs, "
        "state machines, resource lifetimes, and attacker capabilities named in "
        "the threat model.\n"
        "- Findings inside this threat model are in scope even when they would "
        "otherwise look like generic caller misuse, path handling, reliability, "
        "or low-priority hardening.\n"
        "- If the direct reported issue is not quite right but the shown code or "
        "path evidence exposes a different concrete root cause inside the threat "
        "model, report the exact root cause rather than discarding the signal.\n"
        "- Use threat-model terms as focus lenses for neighboring functions, "
        "publish/teardown ordering, lock/refcount/accounting symmetry, and shared "
        "state transitions when those mechanisms are named.\n"
        "- Still require concrete code evidence and do not invent findings not "
        "supported by the provided code."
    )


def apply_threat_model_guidance(base_prompt, threat_model_text):
    guidance = format_threat_model_guidance(threat_model_text)
    if not guidance:
        return base_prompt
    return f"{guidance}\n\n{base_prompt}"
