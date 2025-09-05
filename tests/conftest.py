"""Test configuration and shared fixtures for RecursionHub tests."""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_pr_data():
    """Sample PR data for testing GitHub scripts."""
    return {
        "number": 123,
        "title": "Test PR",
        "body": "Test PR description",
        "state": "open",
        "merged": False,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T01:00:00Z",
    }


@pytest.fixture
def github_env():
    """Mock GitHub environment variables."""
    original_env = {}
    test_env = {
        "GITHUB_TOKEN": "test-token",
        "GITHUB_REPOSITORY": "recursionlab/RecursionHub",
        "GITHUB_API_URL": "https://api.github.com",
    }

    # Store original values
    for key in test_env:
        original_env[key] = os.environ.get(key)
        os.environ[key] = test_env[key]

    yield test_env

    # Restore original values
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
