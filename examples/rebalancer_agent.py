# examples/portfolio_rebalancer_test.py
import os
import asyncio
from dotenv import load_dotenv
from clap import ReactAgent, GroqService
from clap.tools import wrap_eth, get_erc20_balance, swap_tokens_for_tokens

load_dotenv()

WETH_ADDRESS = "0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14"
USDC_ADDRESS = "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7a98"

from web3 import Web3
AGENT_ADDRESS = Web3().eth.account.from_key(os.getenv("AGENT_PRIVATE_KEY")).address

async def main():
    agent = ReactAgent(
        llm_service=GroqService(),
        tools=[wrap_eth, get_erc20_balance, swap_tokens_for_tokens],
        model="llama-3.3-70b-versatile",
        system_prompt= "You are a DeFi trading agent. You execute tasks logically, step-by-step, and wait for confirmation before proceeding. You MUST use the addresses provided in the user query."
    )

    # A clearer, sequential query to guide the agent.
    user_query = f"""
    My wallet address is {AGENT_ADDRESS}.
    Here is the plan:
    1. First, wrap 0.02 ETH into WETH. The WETH contract address is {WETH_ADDRESS}.
    2. After the wrap is successful, swap exactly 0.01 of the WETH for USDC. The USDC contract address is {USDC_ADDRESS}.
    Execute this plan step-by-step.
    """

    print("\n--- Running Final Portfolio Rebalancer Agent ---")
    response = await agent.run(user_msg=user_query, max_rounds=7)
    
    print("\n--- Agent Final Response ---")
    print(response)
    print("--------------------------")

if __name__ == "__main__":
    asyncio.run(main())