# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from metis.configuration import load_metis_config
from metis.configuration import load_runtime_config

import pytest


def test_load_metis_config_uses_yml_when_yaml_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "metis.yml").write_text("selected: yml\n", encoding="utf-8")

    config = load_metis_config()

    assert config == {"selected": "yml"}


def test_load_metis_config_prefers_yaml_over_yml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "metis.yaml").write_text("selected: yaml\n", encoding="utf-8")
    (tmp_path / "metis.yml").write_text("selected: yml\n", encoding="utf-8")

    config = load_metis_config()

    assert config == {"selected": "yaml"}


def test_load_runtime_config_reads_query_reasoning_effort(tmp_path, monkeypatch):
    config_path = tmp_path / "metis.yaml"
    config_path.write_text(
        """
llm_provider:
  name: openai
  model: gpt-test
  code_embedding_model: text-embedding-3-large
  docs_embedding_model: text-embedding-3-large
metis_engine:
  max_workers: 2
query:
  reasoning_effort: high
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    runtime = load_runtime_config(config_path)

    assert runtime["llama_query_reasoning_effort"] == "high"


def test_load_runtime_config_reads_llm_triage_settings(tmp_path, monkeypatch):
    config_path = tmp_path / "metis.yaml"
    config_path.write_text(
        """
llm_provider:
  name: openai
  model: gpt-test
  code_embedding_model: text-embedding-3-large
  docs_embedding_model: text-embedding-3-large
llm_triage:
  enabled: true
  model: gpt-5.5
  reasoning_effort: high
  batch_size: 7
  output_file: results/triage.json
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    runtime = load_runtime_config(config_path)

    assert runtime["llm_triage_enabled"] is True
    assert runtime["llm_triage_model"] == "gpt-5.5"
    assert runtime["llm_triage_reasoning_effort"] == "high"
    assert runtime["llm_triage_batch_size"] == 7
    assert runtime["llm_triage_output_file"] == "results/triage.json"


def test_load_runtime_config_reads_inline_threat_model(tmp_path, monkeypatch):
    config_path = tmp_path / "metis.yaml"
    config_path.write_text(
        """
llm_provider:
  name: openai
  model: gpt-test
  code_embedding_model: text-embedding-3-large
  docs_embedding_model: text-embedding-3-large
threat_model:
  text: |
    Public API output pointers and filenames are attacker-influenced.
  keywords:
    - png_image_write_to_memory
    - CWE-908
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    runtime = load_runtime_config(config_path)

    assert "output pointers" in runtime["threat_model_text"]
    assert runtime["threat_model_path"] == ""
    assert runtime["threat_model_keywords"] == [
        "png_image_write_to_memory",
        "CWE-908",
    ]


def test_load_runtime_config_reads_threat_model_file(tmp_path, monkeypatch):
    threat_model_path = tmp_path / "threat-model.md"
    threat_model_path.write_text(
        "Libpng simplified API caller-controlled buffers are in scope.",
        encoding="utf-8",
    )
    config_path = tmp_path / "metis.yaml"
    threat_model_yaml_path = str(threat_model_path).replace("\\", "/")
    config_path.write_text(
        f"""
llm_provider:
  name: openai
  model: gpt-test
  code_embedding_model: text-embedding-3-large
  docs_embedding_model: text-embedding-3-large
threat_model:
  path: '{threat_model_yaml_path}'
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    runtime = load_runtime_config(config_path)

    assert "simplified API" in runtime["threat_model_text"]
    assert runtime["threat_model_path"] == threat_model_yaml_path


def test_load_runtime_config_accepts_query_reasoning_level_alias(tmp_path, monkeypatch):
    config_path = tmp_path / "metis.yaml"
    config_path.write_text(
        """
llm_provider:
  name: openai
  model: gpt-test
  code_embedding_model: text-embedding-3-large
  docs_embedding_model: text-embedding-3-large
query:
  reasoning_level: low
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    runtime = load_runtime_config(config_path)

    assert runtime["llama_query_reasoning_effort"] == "low"


def test_load_runtime_config_reads_provider_reasoning_effort(tmp_path, monkeypatch):
    config_path = tmp_path / "metis.yaml"
    config_path.write_text(
        """
llm_provider:
  name: openai
  model: gpt-test
  reasoning_effort: xhigh
  code_embedding_model: text-embedding-3-large
  docs_embedding_model: text-embedding-3-large
query:
  max_tokens: 1000
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    runtime = load_runtime_config(config_path)

    assert runtime["llama_query_reasoning_effort"] == "xhigh"


def test_load_runtime_config_query_reasoning_overrides_provider(tmp_path, monkeypatch):
    config_path = tmp_path / "metis.yaml"
    config_path.write_text(
        """
llm_provider:
  name: openai
  model: gpt-test
  reasoning_effort: low
  code_embedding_model: text-embedding-3-large
  docs_embedding_model: text-embedding-3-large
query:
  reasoning_effort: high
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    runtime = load_runtime_config(config_path)

    assert runtime["llama_query_reasoning_effort"] == "high"


def test_load_runtime_config_reports_missing_openai_provider_keys(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "metis.yaml"
    config_path.write_text(
        """
llm_provider:
  name: openai
  model: ""
  code_embedding_model: text-embedding-3-large
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(ValueError) as exc_info:
        load_runtime_config(config_path)

    message = str(exc_info.value)
    assert "OpenAI provider requires additional metis.yaml configuration" in message
    assert "Missing: llm_provider.model" in message
    assert "llm_provider.docs_embedding_model" in message
    assert "Required keys:" in message


def test_load_runtime_config_reports_missing_openai_api_key(tmp_path, monkeypatch):
    config_path = tmp_path / "metis.yaml"
    config_path.write_text(
        """
llm_provider:
  name: openai
  model: gpt-test
  code_embedding_model: text-embedding-3-large
  docs_embedding_model: text-embedding-3-large
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        load_runtime_config(config_path)

    assert "OPENAI_API_KEY environment variable is required for OpenAI provider" in str(
        exc_info.value
    )


def test_load_runtime_config_reports_missing_azure_openai_provider_keys(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "metis.yaml"
    config_path.write_text(
        """
llm_provider:
  name: azure_openai
  azure_endpoint: https://example.openai.azure.com/
  azure_api_version: ""
  code_embedding_model: text-embedding-3-large
  docs_embedding_model: text-embedding-3-large
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

    with pytest.raises(ValueError) as exc_info:
        load_runtime_config(config_path)

    message = str(exc_info.value)
    assert (
        "Azure OpenAI provider requires additional metis.yaml configuration" in message
    )
    assert "Missing: llm_provider.azure_api_version" in message
    assert "llm_provider.engine" in message
    assert "llm_provider.chat_deployment_model" in message
    assert "llm_provider.code_embedding_deployment" in message
    assert "llm_provider.docs_embedding_deployment" in message
    assert "Required keys:" in message


def test_load_runtime_config_reports_missing_azure_openai_api_key(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "metis.yaml"
    config_path.write_text(
        """
llm_provider:
  name: azure_openai
  azure_endpoint: https://example.openai.azure.com/
  azure_api_version: "2024-02-01"
  engine: chat-deployment
  chat_deployment_model: gpt-4o-mini
  code_embedding_model: text-embedding-3-large
  docs_embedding_model: text-embedding-3-large
  code_embedding_deployment: code-embedding-deployment
  docs_embedding_deployment: docs-embedding-deployment
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        load_runtime_config(config_path)

    assert (
        "AZURE_OPENAI_API_KEY environment variable is required for Azure OpenAI provider"
        in str(exc_info.value)
    )


def test_load_runtime_config_reports_missing_vllm_provider_keys(tmp_path):
    config_path = tmp_path / "metis.yaml"
    config_path.write_text(
        """
llm_provider:
  name: vllm
  model: test-model
  docs_embedding_model: text-embedding-3-large
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_runtime_config(config_path)

    message = str(exc_info.value)
    assert "vLLM provider requires additional metis.yaml configuration" in message
    assert "Missing: llm_provider.base_url" in message
    assert "llm_provider.code_embedding_model" in message
    assert "Required keys:" in message


def test_load_runtime_config_resolves_vllm_api_key_from_configured_env(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "metis.yaml"
    config_path.write_text(
        """
llm_provider:
  name: vllm
  base_url: http://localhost:8000/v1
  api_key_env: CUSTOM_VLLM_KEY
  model: test-model
  code_embedding_model: text-embedding-3-large
  docs_embedding_model: text-embedding-3-large
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CUSTOM_VLLM_KEY", "vllm-key")
    monkeypatch.delenv("VLLM_API_KEY", raising=False)

    runtime = load_runtime_config(config_path)

    assert runtime["llm_api_key"] == "vllm-key"


def test_load_runtime_config_reports_missing_ollama_provider_keys(tmp_path):
    config_path = tmp_path / "metis.yaml"
    config_path.write_text(
        """
llm_provider:
  name: ollama
  model: llama3
  code_embedding_model: ""
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_runtime_config(config_path)

    message = str(exc_info.value)
    assert "Ollama provider requires additional metis.yaml configuration" in message
    assert "Missing: llm_provider.code_embedding_model" in message
    assert "llm_provider.docs_embedding_model" in message
    assert "Required keys:" in message


def test_load_runtime_config_keeps_ollama_api_key_optional(tmp_path):
    config_path = tmp_path / "metis.yaml"
    config_path.write_text(
        """
llm_provider:
  name: ollama
  model: llama3
  code_embedding_model: all-minilm
  docs_embedding_model: all-minilm
""",
        encoding="utf-8",
    )

    runtime = load_runtime_config(config_path)

    assert runtime["llm_api_key"] == ""


def test_load_runtime_config_accepts_complete_azure_provider_config(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "metis.yaml"
    config_path.write_text(
        """
llm_provider:
  name: azure_openai
  azure_endpoint: https://example.openai.azure.com/
  azure_api_version: "2024-02-01"
  engine: chat-deployment
  chat_deployment_model: gpt-4o-mini
  code_embedding_model: text-embedding-3-large
  docs_embedding_model: text-embedding-3-large
  code_embedding_deployment: code-embedding-deployment
  docs_embedding_deployment: docs-embedding-deployment
query:
  max_tokens: 1000
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

    runtime = load_runtime_config(config_path)

    assert runtime["azure_endpoint"] == "https://example.openai.azure.com/"
    assert runtime["azure_api_version"] == "2024-02-01"
    assert runtime["engine"] == "chat-deployment"
    assert runtime["chat_deployment_model"] == "gpt-4o-mini"
    assert runtime["code_embedding_deployment"] == "code-embedding-deployment"
    assert runtime["docs_embedding_deployment"] == "docs-embedding-deployment"
