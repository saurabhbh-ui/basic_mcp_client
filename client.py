import os
import sys
import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI

load_dotenv()


def _build_model() -> AzureChatOpenAI:
    """Create an Azure OpenAI chat model using environment configuration."""

    azure_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OAI_API_KEY")
    azure_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")
    azure_api_version = os.getenv("AZURE_OAI_API_VERSION", "2024-12-01-preview")

    missing = [
        name
        for name, value in {
            "AZURE_OAI_ENDPOINT": azure_endpoint,
            "AZURE_OAI_API_KEY": azure_api_key,
            "AZURE_OAI_DEPLOYMENT": azure_deployment,
        }.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Missing required Azure OpenAI configuration: " + ", ".join(missing)
        )

    return AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        api_key=azure_api_key,
        api_version=azure_api_version,
        temperature=0.7,
        max_tokens=4000,
        top_p=1.0,
        max_retries=0,
    )


async def run_calculation(question: str) -> str:
    """Run a calculation prompt through the MCP client and return the answer."""

    model = _build_model()
    client = MultiServerMCPClient(
        {
            "calculator": {
                "url": "https://pretty-emerald-mule.fastmcp.app/mcp",
                "transport": "streamable_http",
            },
            "expense": {
                "url": "https://old-cyan-beetle.fastmcp.app/mcp",
                "transport": "streamable_http",
            },
        }
    )

    tools = await client.get_tools()
    agent = create_agent(model, tools)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})

    messages = result.get("messages", [])
    if messages:
        return messages[-1].content
    return ""


async def main() -> None:
    try:
        response = await run_calculation("what's (3 + 5) x 12?")
    except RuntimeError as exc:
        # Surface configuration problems without an unhandled exception trace.
        print(f"Configuration error: {exc}")
        sys.exit(1)

    print("Response:", response)


if __name__ == "__main__":
    asyncio.run(main())
