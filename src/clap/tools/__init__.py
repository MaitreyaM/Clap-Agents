from .web_search import duckduckgo_search
from .web_crawler import scrape_url, extract_text_by_query 
from .email_tools import send_email, fetch_recent_emails

__all__ = [
    "duckduckgo_search",
    "scrape_url",
    "extract_text_by_query",
    "send_email",
    "fetch_recent_emails",
    "get_wallet_balance",
    "send_eth",
    "interact_with_contract",
    'get_erc20_balance'
    'swap_tokens_for_tokens',
    'wrap_eth',
    'get_token_price'
]

try:
    from .web3_tools import get_wallet_balance, send_eth, interact_with_contract, get_erc20_balance, swap_tokens_for_tokens, wrap_eth, get_token_price
    __all__.extend([
        "get_wallet_balance", "send_eth", "interact_with_contract", 
        "get_erc20_balance", "swap_tokens_for_tokens", "wrap_eth", "get_token_price"
    ])
except ImportError:
    pass