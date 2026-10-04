"""
Minimal unit tests - three only, as requested, to avoid burning API quota
on a bloated test suite. Each one hits a real API once.

Run with: python -m unittest test_tools.py
"""

import os
import unittest
from dotenv import load_dotenv

from tools.web_access import web_access as raw_web_access

load_dotenv()


class TestWebAccessTool(unittest.TestCase):
    def test_urls_can_be_fetched_given_a_prompt(self):
        """web_access should return at least one usable {title, url, content} result."""
        results = raw_web_access("What is the news today?")

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0, "Expected at least one search result")

        first = results[0]
        self.assertIn("url", first)
        self.assertTrue(first["url"].startswith("http"), "url should be a real link")


class TestLLMToolUse(unittest.TestCase):
    def setUp(self):
        from langchain_deepseek import ChatDeepSeek
        from langchain_core.tools import tool
        from langchain.agents import create_agent

        self.call_count = 0

        @tool
        def web_access(query: str) -> list[dict]:
            """Search trusted sources for information relevant to a question."""
            self.call_count += 1
            return raw_web_access(query)

        self.web_access = web_access
        llm = ChatDeepSeek(model="deepseek-v4-flash", api_key=os.getenv("DEEPSEEK_API_KEY"))
        self.agent = create_agent(model=llm, tools=[web_access])

    def test_llm_finds_and_uses_the_tool(self):
        """DeepSeek should choose to call web_access for a question needing current info."""
        self.agent.invoke({
            "messages": [
                {"role": "system", "content": "Use the web_access tool for factual questions you're unsure about."},
                {"role": "user", "content": "What is the news today?"},
            ]
        })

        self.assertGreaterEqual(self.call_count, 1, "Expected the LLM to call web_access at least once")


class TestFullLangChainFlow(unittest.TestCase):
    def test_full_process_works_with_langchain(self):
        """End to end: agent.invoke should return a non-empty final answer with no errors."""
        from langchain_deepseek import ChatDeepSeek
        from langchain.agents import create_agent

        llm = ChatDeepSeek(model="deepseek-v4-flash", api_key=os.getenv("DEEPSEEK_API_KEY"))
        agent = create_agent(model=llm, tools=[raw_web_access_as_tool()])

        result = agent.invoke({
            "messages": [
                {"role": "system", "content": "Use tools when you need current information."},
                {"role": "user", "content": "What is the news today?"},
            ]
        })

        final_answer = result["messages"][-1].content
        self.assertIsInstance(final_answer, str)
        self.assertGreater(len(final_answer.strip()), 0, "Final answer should not be empty")


def raw_web_access_as_tool():
    from langchain_core.tools import tool

    @tool
    def web_access(query: str) -> list[dict]:
        """Search trusted sources for information relevant to a question."""
        return raw_web_access(query)

    return web_access


if __name__ == "__main__":
    unittest.main()