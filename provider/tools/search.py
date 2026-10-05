"""
Thin web search wrapper.

Defaults to a small local fixture (no network, no API key needed) so the
project is runnable and testable offline. If SEARCH_API_KEY is set in the
environment, it will use a real search API instead. Swapping the backend
does not change the tool's input/output contract.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Offline fixture backend
# --------------------------------------------------------------------------
_FIXTURE_RESULTS = {
    "mcp": [
        {
            "title": "Model Context Protocol -- Introduction",
            "url": "https://modelcontextprotocol.io/introduction",
            "snippet": "MCP is an open protocol that standardizes how applications provide context to LLMs.",
        },
        {
            "title": "MCP Python SDK",
            "url": "https://github.com/modelcontextprotocol/python-sdk",
            "snippet": "Official Python implementation of the Model Context Protocol.",
        },
    ],
    "default": [
        {
            "title": "No fixture entry for this query",
            "url": "https://example.com/search",
            "snippet": "This is stubbed search data; no network call was made.",
        },
    ],
}


def _search_fixture(query: str, max_results: int) -> list[dict]:
    key = query.strip().lower()
    for fixture_key, results in _FIXTURE_RESULTS.items():
        if fixture_key != "default" and fixture_key in key:
            return results[:max_results]
    return _FIXTURE_RESULTS["default"][:max_results]


# --------------------------------------------------------------------------
# Real backend (only used if SEARCH_API_KEY is set)
# --------------------------------------------------------------------------
def _search_real_api(query: str, max_results: int) -> list[dict]:
    import requests  # local import: only required if this path is used

    api_key = os.environ["SEARCH_API_KEY"]
    resp = requests.get(
        "https://api.example-search-provider.com/v1/search",
        params={"q": query, "limit": max_results},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=5,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r["title"], "url": r["url"], "snippet": r.get("snippet", "")}
        for r in data.get("results", [])[:max_results]
    ]


def web(query: str, max_results: int = 5) -> list[dict]:
    if os.environ.get("SEARCH_API_KEY"):
        return _search_real_api(query, max_results)
    return _search_fixture(query, max_results)
