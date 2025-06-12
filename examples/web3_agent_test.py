import os
import asyncio
from dotenv import load_dotenv 
from clap import ToolAgent, GroqService , Agent
from clap.tools import get_wallet_balance, send_eth
from web3 import Web3

load_dotenv()

AGENT_PRIVATE_KEY = os.getenv("AGENT_PRIVATE_KEY")
if not AGENT_PRIVATE_KEY:
    raise ValueError("AGENT_PRIVATE_KEY not found in the root .env file.")

w3 = Web3()
AGENT_ACCOUNT = w3.eth.account.from_key(AGENT_PRIVATE_KEY)
AGENT_ADDRESS = AGENT_ACCOUNT.address

async def main():
    agent = ToolAgent(
        llm_service=GroqService(),
        tools=[get_wallet_balance, send_eth],
        model="llama-3.3-70b-versatile"
    )
    
    recipient_address = "0x000000000000000000000000000000000000dEaD"

    user_query = f"What is the current ETH balance of our agent's wallet, which is address {AGENT_ADDRESS}?"
    print("\n--- Running Web3 Test Agent ---")
    response = await agent.run(user_msg=user_query)
    print("\n--- Agent Final Response ---")
    print(response)
    print("--------------------------")

if __name__ == "__main__":
    asyncio.run(main())