# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from metis.engine.core import MetisEngine
from metis.engine.helpers import extract_threat_model_keywords


def test_extract_threat_model_keywords_prefers_code_terms():
    keywords = extract_threat_model_keywords(
        """
        Focus Mali kbase queue teardown races, kbase_csf_ctx_term(),
        get_user_pages_fast pin leaks, dma_fence_put double release, and CWE-416.
        """,
        explicit_keywords=["explicit_lock"],
    )

    assert keywords[0] == "explicit_lock"
    assert "kbase_csf_ctx_term" in keywords
    assert "get_user_pages_fast" in keywords
    assert "dma_fence_put" in keywords
    assert "CWE-416" in keywords
    assert "Focus" not in keywords


def test_engine_derives_reachability_keywords_from_threat_model():
    engine = MetisEngine.__new__(MetisEngine)
    engine.threat_model_text = (
        "Mali kbase_mem_pool_grow page-accounting bugs and "
        "kbase_csf_queue_group_suspend_prepare GUP pin leaks are in scope."
    )
    engine.threat_model_keywords = ["explicit_queue_term"]
    engine.reachability_settings = {"domain_hints": []}

    engine._add_threat_model_to_reachability_settings()

    hints = engine.reachability_settings["domain_hints"]
    keywords = hints[0]["keywords"]
    assert "explicit_queue_term" in keywords
    assert "kbase_mem_pool_grow" in keywords
    assert "kbase_csf_queue_group_suspend_prepare" in keywords
    assert "page-accounting" not in keywords
