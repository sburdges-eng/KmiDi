import sys
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

from kitchen_core.claude_terminal import send_prompt


class FakeResponse:
    def __init__(self, json_data):
        self._json = json_data

    def raise_for_status(self):
        return None

    def json(self):
        return self._json


class ClaudeTerminalTests(unittest.TestCase):
    @patch("kitchen_core.claude_terminal.requests.post")
    def test_send_prompt_returns_completion(self, mock_post):
        mock_post.return_value = FakeResponse({"completion": "Hello from Claude"})
        result = send_prompt(
            "Say hi", api_key="fake-key", endpoint="https://example.com"
        )
        self.assertIn("Hello from Claude", result)

    @patch("kitchen_core.claude_terminal.requests.post")
    def test_send_prompt_handles_alternate_key(self, mock_post):
        mock_post.return_value = FakeResponse({"output": "Other format"})
        result = send_prompt("Test", api_key="fake-key", endpoint="https://example.com")
        self.assertIn("Other format", result)

    @patch("kitchen_core.claude_terminal.requests.post")
    def test_send_prompt_handles_list_response(self, mock_post):
        mock_post.return_value = FakeResponse(
            {"completion": ["First string", {"text": "Nested"}]}
        )
        result = send_prompt(
            "Test list", api_key="fake-key", endpoint="https://example.com"
        )
        self.assertIn("First string", result)

    @patch("kitchen_core.claude_terminal.requests.post")
    def test_send_prompt_handles_nested_structure(self, mock_post):
        mock_post.return_value = FakeResponse(
            {"result": {"content": {"text": "Deep text"}}}
        )
        result = send_prompt(
            "Test nested", api_key="fake-key", endpoint="https://example.com"
        )
        self.assertIn("Deep text", result)


if __name__ == "__main__":
    unittest.main()
