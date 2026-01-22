"""Tests for configuration loading."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import (
    Config,
    DataManifest,
    load_config,
    load_data_manifest,
)


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary."""
    return {
        "llm": {
            "hypothesis_model": {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key_env": "OPENAI_API_KEY",
                "temperature": 0.7,
                "max_tokens": 4096,
            },
            "coding_model": {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key_env": "OPENAI_API_KEY",
                "temperature": 0.2,
                "max_tokens": 4096,
            },
            "summary_model": {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key_env": "OPENAI_API_KEY",
                "temperature": 0.3,
                "max_tokens": 2048,
            },
        },
        "execution": {
            "type": "jupyter",
            "timeout_seconds": 300,
            "max_retries": 3,
        },
        "pipeline": {
            "max_iterations": 10,
            "log_dir": "logs",
            "output_dir": "outputs",
        },
    }


@pytest.fixture
def sample_manifest_dict():
    """Sample data manifest dictionary."""
    return {
        "finding": "REST motif predicts ATF6 binding",
        "context": "Investigating transcription factor interactions",
        "data": {
            "chipseq": {
                "atf6": "/path/to/atf6.bed",
                "rest": "/path/to/rest.bed",
            },
        },
        "tools": ["bedtools", "samtools"],
    }


def test_config_validation(sample_config_dict):
    """Test that valid config passes validation."""
    config = Config.model_validate(sample_config_dict)

    assert config.llm.hypothesis_model.provider == "openai"
    assert config.llm.hypothesis_model.model == "gpt-4o"
    assert config.execution.timeout_seconds == 300
    assert config.pipeline.max_iterations == 10


def test_config_invalid_provider(sample_config_dict):
    """Test that invalid provider raises error."""
    sample_config_dict["llm"]["hypothesis_model"]["provider"] = "invalid"

    with pytest.raises(ValueError, match="Provider must be one of"):
        Config.model_validate(sample_config_dict)


def test_manifest_validation(sample_manifest_dict):
    """Test that valid manifest passes validation."""
    manifest = DataManifest.model_validate(sample_manifest_dict)

    assert manifest.finding == "REST motif predicts ATF6 binding"
    assert "chipseq" in manifest.data
    assert "bedtools" in manifest.tools


def test_load_config_from_file(sample_config_dict):
    """Test loading config from YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_config_dict, f)
        f.flush()

        try:
            config = load_config(f.name)
            assert config.llm.hypothesis_model.model == "gpt-4o"
        finally:
            os.unlink(f.name)


def test_load_manifest_from_file(sample_manifest_dict):
    """Test loading manifest from YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_manifest_dict, f)
        f.flush()

        try:
            manifest = load_data_manifest(f.name)
            assert manifest.finding == "REST motif predicts ATF6 binding"
        finally:
            os.unlink(f.name)


def test_config_file_not_found():
    """Test that missing config file raises error."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def test_get_api_key(sample_config_dict, monkeypatch):
    """Test API key retrieval from environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")

    config = Config.model_validate(sample_config_dict)
    key = config.llm.hypothesis_model.get_api_key()

    assert key == "test-key-123"


def test_get_api_key_missing(sample_config_dict):
    """Test that missing API key raises error."""
    # Ensure the env var is not set
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

    config = Config.model_validate(sample_config_dict)

    with pytest.raises(ValueError, match="API key not found"):
        config.llm.hypothesis_model.get_api_key()
