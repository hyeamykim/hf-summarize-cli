import unittest
from unittest.mock import patch

import hf_summarize_cli


class TestHFSummarizeCli(unittest.TestCase):
    def test_fetch_webpage_text_extracts_visible_content(self):
        html = """
        <html><head><style>.x{color:red}</style></head>
        <body><h1>Hallo Welt</h1><script>alert(1)</script><p>Dies ist ein Test.</p></body></html>
        """

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return html.encode("utf-8")

        with patch("hf_summarize_cli.urlopen", return_value=_FakeResponse()):
            result = hf_summarize_cli.fetch_webpage_text("https://example.com")

        self.assertIn("Hallo Welt", result)
        self.assertIn("Dies ist ein Test.", result)
        self.assertNotIn("alert(1)", result)
        self.assertNotIn("color:red", result)

    @patch("hf_summarize_cli._hf_request")
    @patch("hf_summarize_cli.fetch_webpage_text")
    def test_summarize_url_translates_then_summarizes(self, mock_fetch_text, mock_hf_request):
        mock_fetch_text.return_value = "Das ist ein deutscher Artikel"
        mock_hf_request.side_effect = [
            [{"translation_text": "This is an English article"}],
            [{"summary_text": "An English summary paragraph."}],
        ]

        summary = hf_summarize_cli.summarize_url("https://example.com", token="token")

        self.assertEqual(summary, "An English summary paragraph.")
        self.assertEqual(mock_hf_request.call_count, 2)
        first_call = mock_hf_request.call_args_list[0].args
        second_call = mock_hf_request.call_args_list[1].args
        self.assertEqual(first_call[0], hf_summarize_cli.TRANSLATION_MODEL)
        self.assertEqual(second_call[0], hf_summarize_cli.SUMMARIZATION_MODEL)


if __name__ == "__main__":
    unittest.main()
