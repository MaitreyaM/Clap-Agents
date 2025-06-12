# examples/contract_interaction_test.py

import os
import asyncio
from dotenv import load_dotenv
from clap import ReactAgent, GroqService
from clap.tools import interact_with_contract

load_dotenv()

# We will test our agent on the official Wrapped Ether (WETH) contract on the Sepolia testnet.
WETH_CONTRACT_ADDRESS = "0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14"

# This is the instruction manual (ABI) for the WETH contract. We only need the parts we care about.
# An agent could get this from a block explorer API in a more advanced use case.
WETH_ABI = """
[
  {
    "constant": true,
    "inputs": [],
    "name": "name",
    "outputs": [{ "name": "", "type": "string" }],
    "payable": false,
    "stateMutability": "view",
    "type": "function"
  },
  {
    "constant": true,
    "inputs": [],
    "name": "symbol",
    "outputs": [{ "name": "", "type": "string" }],
    "payable": false,
    "stateMutability": "view",
    "type": "function"
  },
  {
    "constant": true,
    "inputs": [],
    "name": "totalSupply",
    "outputs": [{ "name": "", "type": "uint256" }],
    "payable": false,
    "stateMutability": "view",
    "type": "function"
  }
]
"""

async def main():
    # We use a ReactAgent because it can reason and call the tool multiple times.
    agent = ReactAgent(
        llm_service=GroqService(),
        tools=[interact_with_contract],
        model="llama-3.3-70b-versatile",
        system_prompt="You are a blockchain analysis assistant. You use the interact_with_contract tool to read data from smart contracts."
    )

    # This query requires the agent to reason that it needs to call the tool twice.
    user_query = f"Please tell me the official token symbol and the total supply for the contract at address {WETH_CONTRACT_ADDRESS}."

    print("\n--- Running Contract Interaction Agent ---")
    response = await agent.run(user_msg=user_query)
    
    print("\n--- Agent Final Response ---")
    print(response)
    print("--------------------------")


if __name__ == "__main__":
    asyncio.run(main())