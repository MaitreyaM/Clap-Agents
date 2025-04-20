import os
import operator
from dotenv import load_dotenv
from clap.react_pattern.react_agent import ReactAgent
from clap.tool_pattern.tool import tool
from clap import GroqService
import a
load_dotenv()

@tool
def add(a: int, b: int) -> int:
    """Calculates the sum of two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of a and b.
    """
    print(f"[User Tool Executing] add({a}, {b})")
    return operator.add(a, b)

@tool
def multiply(a: int, b: int) -> int:
    """Calculates the product of two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The product of a and b.
    """
    print(f"[User Tool Executing] multiply({a}, {b})")
    return operator.mul(a, b)

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("Error: Ensure GROQ_API_KEY is in .env")
        exit(1)

    user_tools = [add, multiply]
    groq_service=GroqService()
    agent = ReactAgent(
        llm_service=groq_service,
        tools=user_tools,
        model="llama-3.3-70b-versatile"
    )

    user_query = "Calculate (5 + 9) * 2."

    print(f"--- Running ReAct Agent with User-Defined Tools ---")
    print(f"Query: {user_query}")
    response = agent.run(user_msg=user_query)
    print(f"\n--- Final Response ---")
    print(response)
    print("---------------------------------------------------")