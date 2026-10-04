"""
Minimal unit tests for the scrape tool. Kept to three, same reasoning
as test_tools.py - real HTTP calls cost time and shouldn't be spammed.

Run with: python -m unittest tests.test_scrape
"""

import unittest

from tools.scrape import scrape


class TestScrapeTool(unittest.TestCase):
    def test_returns_clean_text_no_html(self):
        """A real page should come back as plain text, not raw HTML tags."""
        result = scrape("https://en.wikipedia.org/wiki/Cellular_respiration")

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0, "Expected non-empty text back")
        self.assertNotIn("<html", result.lower())
        self.assertNotIn("<div", result.lower())

    def test_respects_max_chars_cap(self):
        """Output should never exceed max_chars, plus the truncation marker."""
        cap = 200
        result = scrape("https://en.wikipedia.org/wiki/Cellular_respiration", max_chars=cap)

        # allow a little headroom for the "... [truncated]" marker itself
        self.assertLessEqual(len(result), cap + len("... [truncated]"))

    def test_bad_url_returns_empty_string_not_a_crash(self):
        """A broken/fake URL should fail gracefully, not raise an exception."""
        result = scrape("https://this-domain-does-not-exist-12345.fake/page")

        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()