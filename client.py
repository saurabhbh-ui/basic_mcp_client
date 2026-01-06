"""
LangChain MCP Client - Modern Implementation
Uses LangChain 1.x API with create_agent
"""

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
import asyncio
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI Configuration
AZURE_OAI_ENDPOINT = os.getenv("AZURE_OAI_ENDPOINT", "https://oai-bispoc-dev-002.openai.azure.com")
AZURE_OAI_API_KEY = os.getenv("AZURE_OAI_API_KEY", "4e5edc403cef43c0aab6287f89567d6d")
AZURE_OAI_DEPLOYMENT = os.getenv("AZURE_OAI_DEPLOYMENT", "gpt41")
AZURE_OAI_API_VERSION = os.getenv("AZURE_OAI_API_VERSION", "2024-12-01-preview")

# Initialize Azure OpenAI model
model = AzureChatOpenAI(
    azure_endpoint=AZURE_OAI_ENDPOINT,
    azure_deployment=AZURE_OAI_DEPLOYMENT,
    api_key=AZURE_OAI_API_KEY,
    api_version=AZURE_OAI_API_VERSION,
    temperature=0.7,
    max_tokens=4000,
    top_p=1.0,
    max_retries=0
)


async def main():
    """Main function to run the MCP client agent"""
    
    # Configure MCP servers
    client = MultiServerMCPClient(
        {
            "calculator": {
                "url": "https://pretty-emerald-mule.fastmcp.app/mcp",
                "transport": "streamable_http",
            },
            "expense": {
                "url": "https://old-cyan-beetle.fastmcp.app/mcp",
                "transport": "streamable_http",
            }
        }
    )

    # Get tools from MCP servers
    print("Loading tools from MCP servers...")
    tools = await client.get_tools()
    print(f"Loaded {len(tools)} tools")
    
    # Create agent using LangChain 1.x API
    agent = create_agent(
        model,
        tools=tools,
        system_prompt="You are a helpful assistant. Use the available tools to answer questions accurately."
    )

    # Run the agent with a question
    question = "what's (3 + 5) x 12?"
    print(f"\nQuestion: {question}")
    print("\nProcessing...\n")
    
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    # Extract and display the answer
    answer = response['messages'][-1].content
    print(f"Answer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
