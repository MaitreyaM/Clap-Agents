
from .web_search import duckduckgo_search
from .web_crawler import scrape_url, extract_text_by_query 
from .email_tools import send_email, fetch_recent_emails

__all__ = [
    "duckduckgo_search",
    "scrape_url",
    "extract_text_by_query",
    "send_email",
    "fetch_recent_emails",
]

