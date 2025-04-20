import os
from duckduckgo_search import DDGS
from ..tool_pattern.tool import tool



@tool
def duckduckgo_search(query: str, num_results: int = 5) -> str:
    """Performs a web search using the DuckDuckGo Search API."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(keywords=query, max_results=num_results)
            if results:
                results_str = f"DuckDuckGo search results for '{query}':\n"
                for i, result in enumerate(results):
                    title = result.get('title', 'No Title')
                    snippet = result.get('body', 'No Snippet')
                    url = result.get('href', 'No URL')
                    results_str += f"{i+1}. {title}\n   URL: {url}\n   Snippet: {snippet}\n\n"
                return results_str.strip()
            else:
                return f"No DuckDuckGo results found for '{query}'."
    except Exception as e:
        return f"Error during DuckDuckGo web search for '{query}': {e}"