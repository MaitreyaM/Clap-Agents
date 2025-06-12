# examples/verify_weth_balance.py
import os
from dotenv import load_dotenv
from web3 import Web3

# --- Configuration ---
load_dotenv()
WEB3_PROVIDER_URL = os.getenv("WEB3_PROVIDER_URL")
if not WEB3_PROVIDER_URL:
    raise ValueError("WEB3_PROVIDER_URL not found in .env file")

WETH_ADDRESS = "0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14"
ADDRESS_WITH_WETH = "0x785239105435919318a804391305417B62657e05" # An address known to hold WETH
AGENT_ADDRESS = Web3().eth.account.from_key(os.getenv("AGENT_PRIVATE_KEY")).address


# Standard ERC20 ABI to check balance and decimals
ERC20_ABI = """
[
  {"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"type":"function"},
  {"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"}
]
"""

def main():
    print("--- Verifying WETH Balance Tool Behavior ---")
    try:
        w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URL))
        
        # Checksum the addresses
        chk_weth_address = Web3.to_checksum_address(WETH_ADDRESS)
        chk_target_address = Web3.to_checksum_address(AGENT_ADDRESS)
        
        # Load the contract
        weth_contract = w3.eth.contract(address=chk_weth_address, abi=ERC20_ABI)
        
        print(f"Attempting to get WETH balance for address: {chk_target_address}...")
        
        # Call the balanceOf function directly
        raw_balance = weth_contract.functions.balanceOf(chk_target_address).call()
        decimals = weth_contract.functions.decimals().call()
        balance = raw_balance / (10 ** decimals)
        
        print("\n--- TEST SUCCEEDED ---")
        print(f"Successfully retrieved balance without errors: {balance} WETH")
        print("This proves the ABI is correct and the error only occurs for zero-balance addresses.")

    except Exception as e:
        print("\n--- TEST FAILED ---")
        print(f"An unexpected error occurred: {type(e).__name__} - {e}")
        print("This would indicate a deeper issue with the contract or ABI.")

if __name__ == "__main__":
    main()