"""
Standalone scrape tool. Not connected to LangChain or any LLM yet -
just plain Python so you can test it on its own first.

Setup before running:
    pip install trafilatura --break-system-packages
    (or: uv pip install trafilatura)
"""

import trafilatura


def scrape(url: str, max_chars: int = 5000) -> str:
    """
    Download a page and pull out just the clean article text.

    Args:
        url: the page to scrape.
        max_chars: caps how much text comes back. Long pages (a full
            Wikipedia article can run 10,000+ words) get cut off at this
            length instead of flooding the LLM's context with the whole
            thing.

    Returns:
        The extracted article text (truncated to max_chars), or an
        empty string if the page couldn't be downloaded or nothing
        readable could be pulled out.
    """
    downloaded = trafilatura.fetch_url(url)

    if downloaded is None:
        print(f"[scrape] failed to download: {url}")
        return ""

    text = trafilatura.extract(
        downloaded,
        output_format="markdown",
        include_tables=False,
    )

    if text is None:
        print(f"[scrape] nothing extractable from: {url}")
        return ""

    if len(text) > max_chars:
        text = text[:max_chars] + "... [truncated]"

    return text


if __name__ == "__main__":
    test_url = "https://en.wikipedia.org/wiki/LangChain"

    result = scrape(test_url)

    print(f"URL: {test_url}")
    print(f"Got {len(result)} characters back.\n")
    print(result)
    # print("..." if len(result) > 500 else "")