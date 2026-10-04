"""
Standalone test for the web access tool.
Not connected to the orchestrator/agent yet - just run this file directly
to confirm your Tavily API key and domain filtering work.

Setup before running:
    pip install tavily-python python-dotenv --break-system-packages
    Create a .env file in the same folder with:
        TAVILY_API_KEY=tvly-your-actual-key-here
"""

import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def web_access(query: str, max_results: int = 3) -> list[dict]:
    """
    Search trusted sources for information relevant to a question.

    Args:
        query: the search query (usually the question itself).
        max_results: how many sources to return.

    Returns:
        A list of dicts, each with 'title', 'url', and 'content'.
        Returns an empty list if the search fails for any reason.
    """
    try:
        response = client.search(
            query=query,
            # include_domains=TRUSTED_DOMAINS,
            max_results=max_results,
        )
    except Exception as e:
        print(f"[web_access] search failed: {e}")
        return []

    return [
        {"title": r["title"], "url": r["url"], "content": r["content"]}
        for r in response.get("results", [])
    ]


if __name__ == "__main__":
    test_query = "What is the process of cellular respiration?"
    results = web_access(test_query)

    print(f"Query: {test_query}")
    print(f"Got {len(results)} result(s):\n")

    for i, r in enumerate(results, start=1):
        print(f"{i}. {r['title']}")
        print(f"   {r['url']}")
        print(f"   {r['content'][:150]}...\n")