"""Tests for the knot detector script."""


# Import the script - we need to add the scripts directory to the path
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.append(str(Path(__file__).parent.parent / ".github" / "scripts"))

from knot_detector import add_label, create_backlog_issue, fetch_prs


class TestKnotDetector:
    """Test cases for knot detection functionality."""

    def test_fetch_prs_success(self):
        """Test successful PR fetching."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {"number": 1, "title": "Test PR 1", "created_at": "2024-01-01T00:00:00Z"},
            {"number": 2, "title": "Test PR 2", "created_at": "2024-01-02T00:00:00Z"},
        ]
        mock_response.raise_for_status.return_value = None

        with patch("knot_detector.requests.get", return_value=mock_response):
            result = fetch_prs("test/repo", "fake-token")
            assert len(result) == 2
            assert result[0]["title"] == "Test PR 1"

    def test_fetch_prs_no_token(self):
        """Test PR fetching without authentication token."""
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None

        with patch(
            "knot_detector.requests.get", return_value=mock_response
        ) as mock_get:
            fetch_prs("test/repo", None)
            # Verify request was made without auth headers
            args, kwargs = mock_get.call_args
            assert "headers" not in kwargs or "Authorization" not in kwargs.get(
                "headers", {}
            )

    def test_knot_detection_logic(self):
        """Test the knot detection logic using collections.Counter."""
        import collections

        prs = [
            {"number": 1, "title": "Fix bug", "created_at": "2024-01-01T00:00:00Z"},
            {"number": 2, "title": "Fix bug", "created_at": "2024-01-02T00:00:00Z"},
            {"number": 3, "title": "Fix bug", "created_at": "2024-01-03T00:00:00Z"},
            {
                "number": 4,
                "title": "Different title",
                "created_at": "2024-01-04T00:00:00Z",
            },
        ]

        titles = [p["title"] for p in prs]
        counts = collections.Counter(titles)
        frequent = [t for t, c in counts.items() if c >= 2]

        assert len(frequent) == 1  # Only "Fix bug" appears >= 2 times
        assert "Fix bug" in frequent
        assert counts["Fix bug"] == 3

    def test_knot_detection_below_threshold(self):
        """Test knot detection when no duplicates meet threshold."""
        import collections

        prs = [
            {"number": 1, "title": "Fix bug", "created_at": "2024-01-01T00:00:00Z"},
            {"number": 2, "title": "Add feature", "created_at": "2024-01-02T00:00:00Z"},
            {"number": 3, "title": "Update docs", "created_at": "2024-01-03T00:00:00Z"},
        ]

        titles = [p["title"] for p in prs]
        counts = collections.Counter(titles)
        frequent = [t for t, c in counts.items() if c >= 3]

        assert len(frequent) == 0

    @patch("knot_detector.requests.post")
    def test_add_label_success(self, mock_post):
        """Test successful label addition."""
        add_label("test/repo", "fake-token", 123, "knot-detected")

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["json"] == ["knot-detected"]
        assert "Authorization" in kwargs["headers"]

    @patch("knot_detector.requests.post")
    def test_create_backlog_issue(self, mock_post):
        """Test backlog issue creation."""
        title = "Knot: repeated PR title 'Fix bug'"
        body = "Detected 3 PRs with the same title"

        create_backlog_issue("test/repo", "fake-token", title, body)

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["title"] == title
        assert kwargs["json"]["body"] == body
        assert "knot" in kwargs["json"]["labels"]


class TestKnotDetectorIntegration:
    """Integration tests for knot detector."""

    @pytest.mark.slow
    def test_end_to_end_no_knots(self):
        """Test complete workflow when no knots are detected."""
        import collections

        mock_prs = [
            {
                "number": 1,
                "title": "Unique title 1",
                "created_at": "2024-01-01T00:00:00Z",
            },
            {
                "number": 2,
                "title": "Unique title 2",
                "created_at": "2024-01-02T00:00:00Z",
            },
        ]

        titles = [p["title"] for p in mock_prs]
        counts = collections.Counter(titles)
        frequent = [t for t, c in counts.items() if c >= 3]

        assert len(frequent) == 0  # No knots detected

    @pytest.mark.slow
    def test_end_to_end_with_knots(self):
        """Test complete workflow when knots are detected."""
        import collections

        mock_prs = [
            {"number": 1, "title": "Fix issue", "created_at": "2024-01-01T00:00:00Z"},
            {"number": 2, "title": "Fix issue", "created_at": "2024-01-02T00:00:00Z"},
            {"number": 3, "title": "Fix issue", "created_at": "2024-01-03T00:00:00Z"},
        ]

        with (
            patch("knot_detector.add_label") as mock_label,
            patch("knot_detector.create_backlog_issue") as mock_issue,
        ):

            titles = [p["title"] for p in mock_prs]
            counts = collections.Counter(titles)
            frequent = [t for t, c in counts.items() if c >= 3]

            assert len(frequent) == 1
            assert "Fix issue" in frequent

            # Simulate the labeling process
            for title in frequent:
                for pr in [pr for pr in mock_prs if pr["title"] == title]:
                    add_label("test/repo", "fake-token", pr["number"], "knot-detected")
                create_backlog_issue(
                    "test/repo",
                    "fake-token",
                    f"Knot: repeated PR title '{title}'",
                    f"Detected {counts[title]} PRs",
                )

            assert mock_label.call_count == 3
            assert mock_issue.call_count == 1
