import os
import asyncio
from dotenv import load_dotenv
from clap import Team, Agent, GroqService , GoogleOpenAICompatService
from clap.tools import get_token_price, get_erc20_balance, wrap_eth, swap_tokens_for_tokens

load_dotenv()

WETH_ADDRESS = "0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14"
USDC_ADDRESS = "0x8267cf9254734c6eb452a7bb9aaf97b392258b21"
CHECK_INTERVAL_SECONDS = 60 

from web3 import Web3
AGENT_ADDRESS = Web3().eth.account.from_key(os.getenv("AGENT_PRIVATE_KEY")).address

async def main():
    print("\n--- Starting Autonomous Multi-Agent Rebalancer Service ---")
    print(f"Agent Wallet Address: {AGENT_ADDRESS}")
    print("Service will check the portfolio every 1 minutes. Press Ctrl+C to stop.")
    
    llm_service = GoogleOpenAICompatService()

    while True:
        try:
            with Team() as team:
                analyst = Agent(
                    name="Portfolio_Analyst",
                    backstory="I am a financial analyst. I observe market data and wallet balances to determine if a portfolio is unbalanced. I create a simple, actionable plan if rebalancing is needed.",
                    task_description=f"""
            Your goal is to check the portfolio at address {AGENT_ADDRESS} and create a single-line command for a trader.
            
            1.  First, gather all necessary information: the ETH/USD price, the WETH balance ({WETH_ADDRESS}), and the USDC balance ({USDC_ADDRESS}).
            2.  Analyze this information to determine the required action based on a 50/50 portfolio target.
            
            3.  **Your final output MUST be one of the following three exact strings:**
                *   If the portfolio is balanced, your output must be the string: `No action needed.`
                *   If the WETH balance is too low, your output must be the string: `ACTION: wrap_eth(amount_eth=0.02)`
                *   If a rebalance is needed, your output must be a string in this exact format: `ACTION: swap_exact_tokens_for_tokens(token_in_address='...', token_out_address='...', amount_in=...)`
            
            Do not add any other text, explanation, or tool calls. Your entire job is to produce one of these three strings.
            """,
                    llm_service=llm_service,
                    model="gemini-2.5-flash-preview-05-20",
                    tools=[get_token_price, get_erc20_balance],
                    parallel_tool_calls=False,
                    
                )

                # Agent 2: The Trader (write-only)
                trader = Agent(
                    name="OnChain_Trader",
                    backstory="I am an execution agent. I receive a clear, single-step plan from the Analyst and I execute it precisely using my tools.",
                    task_description="Execute the plan provided in the context from the Portfolio_Analyst. Do not do any analysis yourself. Simply execute the function call as instructed. If the plan is 'No action needed', then your final answer should be the same.",
                    llm_service=llm_service,
                    model="gemini-2.5-flash-preview-05-20",
                    tools=[wrap_eth, swap_tokens_for_tokens],
                    parallel_tool_calls=False,
                )

                # Define the dependency
                analyst >> trader

                # Run the team workflow for one cycle
                print("\n--- [Service] Starting new portfolio check cycle... ---")
                await team.run()

                print("\n--- [Service] Team cycle complete. Result from Trader: ---")
                print(team.results.get("OnChain_Trader", "No result from trader."))

        except Exception as e:
            print(f"An error occurred during the team run: {e}")
        
        print(f"\n--- [Service] Sleeping for {CHECK_INTERVAL_SECONDS} seconds... ---")
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nService stopped by user.")