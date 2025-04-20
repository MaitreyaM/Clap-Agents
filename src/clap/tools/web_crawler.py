
import asyncio
import json
import os
from dotenv import load_dotenv

from clap.tool_pattern.tool import tool

try:
    from crawl4ai import AsyncWebCrawler
except ImportError:
    raise ImportError("crawl4ai library not found. Please install it using: pip install crawl4ai")

load_dotenv()

@tool
async def scrape_url(url: str) -> str:
    """
    Scrape a webpage and return its raw markdown content.

    Args:
        url: The URL of the webpage to scrape.

    Returns:
        The webpage content in markdown format or an error message.
    """
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            return result.markdown.raw_markdown if result.markdown else "No content found"
    except Exception as e:
        return f"Error scraping URL '{url}': {str(e)}"

@tool
async def extract_text_by_query(url: str, query: str, context_size: int = 300) -> str:
    """
    Extract relevant text snippets containing a query from a webpage's markdown content.

    Args:
        url: The URL of the webpage to search.
        query: The search query (case-insensitive) to look for.
        context_size: Number of characters around the match to include.

    Returns:
        Relevant text snippets containing the query or a message indicating no matches/content.
    """
    try:
        markdown_content = await scrape_url.run(url=url)

        if not markdown_content or markdown_content == "No content found" or markdown_content.startswith("Error"):
            # Pass through the error message from scrape_url if it failed
            return markdown_content if markdown_content.startswith("Error") else f"Could not retrieve content from URL: {url}"

        lower_query = query.lower()
        lower_content = markdown_content.lower()
        matches = []
        start_index = 0

        while len(matches) < 5: # Limit matches
            pos = lower_content.find(lower_query, start_index)
            if pos == -1:
                break

            start = max(0, pos - context_size)
            end = min(len(markdown_content), pos + len(lower_query) + context_size)
            context_snippet = markdown_content[start:end]
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(markdown_content) else ""
            matches.append(f"{prefix}{context_snippet}{suffix}")

            start_index = pos + len(lower_query)

        if matches:
            result_text = "\n\n---\n\n".join([f"Match {i+1}:\n{match}" for i, match in enumerate(matches)])
            return f"Found {len(matches)} matches for '{query}' on the page:\n\n{result_text}"
        else:
            return f"No matches found for '{query}' on the page."

    except Exception as e:
        # Catch potential errors during the find/string manipulation logic
        return f"Error processing content from '{url}' for query '{query}': {str(e)}"

