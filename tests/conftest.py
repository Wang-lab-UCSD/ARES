"""Pytest configuration and shared fixtures."""

import os
import pytest


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    """Reset environment variables for each test."""
    # Store original values
    original_keys = dict(os.environ)

    yield

    # Restore original values (handled by monkeypatch automatically)


@pytest.fixture
def mock_openai_key(monkeypatch):
    """Set a mock OpenAI API key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-mock-key-12345")


@pytest.fixture
def sample_finding():
    """Sample scientific finding for testing."""
    return "REST motif is the most important feature for predicting ATF6 binding"


@pytest.fixture
def sample_hypothesis():
    """Sample hypothesis for testing."""
    return {
        "name": "Nested Motif Hypothesis",
        "rationale": "The REST motif contains ATF6 binding core within it",
        "prediction": "REST motifs that predict ATF6 will contain CCACG substring",
        "verification_plan": [
            "Extract REST motif sequences",
            "Search for CCACG substring",
            "Calculate percentage containing the core",
        ],
        "priority": 1,
        "required_data": ["chipseq", "pwm"],
    }


@pytest.fixture
def sample_data_manifest():
    """Sample data manifest for testing."""
    return {
        "finding": "REST motif predicts ATF6 binding",
        "context": "Investigating TF interactions",
        "data": {
            "chipseq": {
                "atf6": "/path/to/atf6.bed",
                "rest": "/path/to/rest.bed",
            },
            "pwm": {
                "atf6": "/path/to/atf6.pwm",
                "rest": "/path/to/rest.pwm",
            },
        },
        "tools": ["bedtools", "samtools"],
    }
