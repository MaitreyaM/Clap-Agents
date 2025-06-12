import os
from web3 import Web3
from clap.tool_pattern.tool import tool
import json 

# Environment variables are loaded by the main application script.

WEB3_PROVIDER_URL = os.getenv("WEB3_PROVIDER_URL")
AGENT_PRIVATE_KEY = os.getenv("AGENT_PRIVATE_KEY")

# --- Official Testnet Addresses ---
WETH_CONTRACT_ADDRESS = "0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14"
UNISWAP_ROUTER_ADDRESS = "0x3bFA4769FB09eefC5a399D6D47036A5d3fA67B54"
CHAINLINK_ETH_USD_PRICE_FEED_ADDRESS = "0x694AA1769357215DE4FAC081bf1f309aDC325306"


# --- Correct, Centralized ABIs ---
CHAINLINK_PRICE_FEED_ABI = """
[
  {
    "inputs": [],
    "name": "latestRoundData",
    "outputs": [
      { "internalType": "uint80", "name": "roundId", "type": "uint80" },
      { "internalType": "int256", "name": "answer", "type": "int256" },
      { "internalType": "uint256", "name": "startedAt", "type": "uint256" },
      { "internalType": "uint256", "name": "updatedAt", "type": "uint256" },
      { "internalType": "uint80", "name": "answeredInRound", "type": "uint80" }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "decimals",
    "outputs": [{ "internalType": "uint8", "name": "", "type": "uint8" }],
    "stateMutability": "view",
    "type": "function"
  }
]
"""
ERC20_STANDARD_ABI = """
[
  {"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"type":"function"},
  {"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},
  {"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"type":"function"}
]
"""
WETH_ABI = """
[
  {"constant":false,"inputs":[],"name":"deposit","outputs":[],"payable":true,"stateMutability":"payable","type":"function"},
  {"constant":true,"inputs":[{"name":"","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"}
]
"""
UNISWAP_ABI = """
[{"inputs":[{"components":[{"type":"address","name":"tokenIn"},{"type":"address","name":"tokenOut"},{"type":"uint24","name":"fee"},{"type":"address","name":"recipient"},{"type":"uint256","name":"amountIn"},{"type":"uint256","name":"amountOutMinimum"},{"type":"uint160","name":"sqrtPriceLimitX96"}],"type":"tuple","name":"params"}],"name":"exactInputSingle","outputs":[{"type":"uint256","name":"amountOut"}],"stateMutability":"payable","type":"function"}]
"""


w3 = None
agent_account = None

def _initialize_web3():
    """Initializes Web3 instance and account if they don't exist."""
    global w3, agent_account, WEB3_PROVIDER_URL, AGENT_PRIVATE_KEY
    if w3 is None:
        WEB3_PROVIDER_URL = os.getenv("WEB3_PROVIDER_URL")
        AGENT_PRIVATE_KEY = os.getenv("AGENT_PRIVATE_KEY")
        
        if not WEB3_PROVIDER_URL or not AGENT_PRIVATE_KEY:
            raise ConnectionError("Web3 provider URL or Agent private key not found.")
        w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URL))
        agent_account = w3.eth.account.from_key(AGENT_PRIVATE_KEY)

@tool
def get_wallet_balance(address: str) -> str:
    """
    Gets the native token balance (Sepolia ETH) of a given wallet address.
    """
    try:
        _initialize_web3()
        balance_wei = w3.eth.get_balance(address)
        balance_eth = w3.from_wei(balance_wei, 'ether')
        return f"The balance of address {address} is {balance_eth} ETH."
    except Exception as e:
        return f"Error getting balance for address {address}: {e}"
    
@tool
def get_token_price(token_pair: str) -> str:
    """
    Gets the latest price of a token pair (e.g., 'ETH/USD') from a Chainlink Price Feed.

    Args:
        token_pair: The token pair to get the price for. Currently supports 'ETH/USD'.

    Returns:
        A string indicating the latest price of the token pair.
    """
    try:
        _initialize_web3()
        if token_pair.upper() != "ETH/USD":
            return "Error: This tool currently only supports the 'ETH/USD' token pair."

        price_feed_contract = w3.eth.contract(
            address=CHAINLINK_ETH_USD_PRICE_FEED_ADDRESS,
            abi=CHAINLINK_PRICE_FEED_ABI
        )
        
        # The latestRoundData function returns a tuple of values. The price is the second element.
        latest_data = price_feed_contract.functions.latestRoundData().call()
        price_raw = latest_data[1]
        
        # The price feed also has a 'decimals' function to tell us where to put the decimal point.
        price_decimals = price_feed_contract.functions.decimals().call()
        
        # Convert the raw price to a human-readable format
        price = price_raw / (10 ** price_decimals)
        
        return f"The latest price for {token_pair} is ${price:.2f}"
        
    except Exception as e:
        return f"Error getting token price: {type(e).__name__} - {e}"

@tool
def send_eth(to_address: str, amount_eth: float) -> str:
    """
    Creates, signs, and sends a transaction to transfer Sepolia ETH.
    """
    try:
        _initialize_web3()
        nonce = w3.eth.get_transaction_count(agent_account.address)

        tx = {
            'from': agent_account.address,
            'to': to_address,
            'value': w3.to_wei(amount_eth, 'ether'),
            'gas': 21000,
            'gasPrice': w3.eth.gas_price,
            'nonce': nonce,
        }

        signed_tx = w3.eth.account.sign_transaction(tx,AGENT_PRIVATE_KEY)
        
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)

        return f"Transaction sent successfully. Transaction hash: {w3.to_hex(tx_hash)}"
    except Exception as e:
        return f"Error sending transaction: {type(e).__name__} - {e}"
    


@tool
def interact_with_contract(
    contract_address: str,
    abi: str,
    function_name: str,
    function_args: list,
    is_write_transaction: bool = False
) -> str:
    """
    Interacts with a smart contract by calling one of its functions.
    Can perform read-only calls (e.g., getting data) or write transactions (e.g., sending tokens).

    Args:
        contract_address: The address of the smart contract (e.g., "0x...").
        abi: The contract's Application Binary Interface (ABI) as a JSON string.
        function_name: The exact name of the function to call.
        function_args: A list of arguments to pass to the function, in order.
        is_write_transaction: Set to True if this is a state-changing transaction that requires gas. Defaults to False (read-only call).

    Returns:
        A string containing the result of the call or a transaction hash if it's a write transaction.
    """
    try:
        _initialize_web3()

        # 1. Validate inputs
        if not w3.is_address(contract_address):
            return "Error: Invalid 'contract_address'."
        try:
            abi_json = json.loads(abi)
        except json.JSONDecodeError:
            return "Error: The provided 'abi' is not a valid JSON string."

        # 2. Create the contract object
        contract = w3.eth.contract(address=contract_address, abi=abi_json)

        # 3. Get the function object from the contract
        func_to_call = getattr(contract.functions, function_name)
        if not func_to_call:
            return f"Error: Function '{function_name}' not found in the contract's ABI."

        # 4. Prepare the function call with its arguments
        prepared_func = func_to_call(*function_args)

        # 5. Execute as either a read-only 'call' or a write 'transaction'
        if is_write_transaction:
            # This is a state-changing transaction that costs gas
            nonce = w3.eth.get_transaction_count(agent_account.address)
            tx = prepared_func.build_transaction({
                'from': agent_account.address,
                'nonce': nonce,
                'gasPrice': w3.eth.gas_price
            })
            signed_tx = w3.eth.account.sign_transaction(tx, private_key=AGENT_PRIVATE_KEY)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            return f"Write transaction sent successfully. Transaction hash: {w3.to_hex(tx_hash)}"
        else:
            # This is a read-only call that is free and instant
            result = prepared_func.call()
            # Convert the result to a JSON string to ensure it's readable by the LLM
            return json.dumps(result)

    except Exception as e:
        return f"Error interacting with contract: {type(e).__name__} - {e}"
    
@tool
def get_erc20_balance(token_address: str, wallet_address: str) -> str:
    """Gets the balance of a specific ERC-20 token for a given wallet."""
    try:
        _initialize_web3()
        chk_token_address = Web3.to_checksum_address(token_address)
        chk_wallet_address = Web3.to_checksum_address(wallet_address)
        
        token_contract = w3.eth.contract(address=chk_token_address, abi=ERC20_STANDARD_ABI)
        
        # This will now work correctly with a real contract address
        decimals = token_contract.functions.decimals().call()
        raw_balance = token_contract.functions.balanceOf(chk_wallet_address).call()
            
        balance = raw_balance / (10 ** decimals)
        return str(balance)
        
    except Exception as e:
        # A simple, honest error handler for genuine problems.
        return f"Error in get_erc20_balance for token {token_address}: {e}"

@tool
def wrap_eth(amount_eth: float) -> str:
    """Converts native ETH into WETH (Wrapped ETH) by depositing it into the WETH contract."""
    try:
        _initialize_web3()
        weth_contract = w3.eth.contract(address=WETH_CONTRACT_ADDRESS, abi=WETH_ABI)
        tx = weth_contract.functions.deposit().build_transaction({
            'from': agent_account.address,
            'value': w3.to_wei(amount_eth, 'ether'),
            'nonce': w3.eth.get_transaction_count(agent_account.address),
            'gasPrice': w3.eth.gas_price
        })
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=AGENT_PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        print(f"Waiting for wrap transaction {w3.to_hex(tx_hash)} to be confirmed...")
        w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return f"ETH wrapping successful and confirmed. Hash: {w3.to_hex(tx_hash)}"
    except Exception as e:
        return f"Error wrapping ETH: {e}"

@tool
def swap_tokens_for_tokens(token_in_address: str, token_out_address: str, amount_in: float, fee: int = 3000) -> str:
    """Swaps an exact amount of an input token for another on Uniswap V3, waiting for confirmations."""
    try:
        _initialize_web3()
        chk_token_in = Web3.to_checksum_address(token_in_address)
        chk_token_out = Web3.to_checksum_address(token_out_address)
        chk_router_address = Web3.to_checksum_address(UNISWAP_ROUTER_ADDRESS)
        
        token_in_contract = w3.eth.contract(address=chk_token_in, abi=ERC20_STANDARD_ABI)
        decimals = token_in_contract.functions.decimals().call()
        amount_in_wei = int(amount_in * (10**decimals))

        current_nonce = w3.eth.get_transaction_count(agent_account.address)

        # Step 1: Approve
        approve_tx = token_in_contract.functions.approve(chk_router_address, amount_in_wei).build_transaction({
            'from': agent_account.address, 'nonce': current_nonce
        })
        signed_approve_tx = w3.eth.account.sign_transaction(approve_tx, private_key=os.getenv("AGENT_PRIVATE_KEY"))
        approve_tx_hash = w3.eth.send_raw_transaction(signed_approve_tx.raw_transaction)
        
        # --- THE FIX ---
        # Wait for the approval transaction to be confirmed.
        print(f"Waiting for approval transaction {w3.to_hex(approve_tx_hash)} to be confirmed...")
        w3.eth.wait_for_transaction_receipt(approve_tx_hash)

        # Step 2: Swap
        uniswap_router = w3.eth.contract(address=chk_router_address, abi=UNISWAP_ABI)
        swap_params = (chk_token_in, chk_token_out, fee, agent_account.address, amount_in_wei, 0, 0)
        
        swap_tx = uniswap_router.functions.exactInputSingle(swap_params).build_transaction({
            'from': agent_account.address, 'nonce': current_nonce + 1
        })
        signed_swap_tx = w3.eth.account.sign_transaction(swap_tx, private_key=os.getenv("AGENT_PRIVATE_KEY"))
        swap_tx_hash = w3.eth.send_raw_transaction(signed_swap_tx.raw_transaction)

        print(f"Waiting for swap transaction {w3.to_hex(swap_tx_hash)} to be confirmed...")
        w3.eth.wait_for_transaction_receipt(swap_tx_hash)
        
        return f"Swap successful and confirmed. Swap hash: {w3.to_hex(swap_tx_hash)}"
    except Exception as e:
        return f"Error during swap: {e}"