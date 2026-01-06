from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

import asyncio

from dotenv import load_dotenv
load_dotenv()
import asyncio

AZURE_OAI_ENDPOINT="https://oai-bispoc-dev-002.openai.azure.com"
AZURE_OAI_API_KEY="4e5edc403cef43c0aab6287f89567d6d"
AZURE_OAI_DEPLOYMENT="gpt41"
AZURE_OAI_API_VERSION="2024-12-01-preview"

model = AzureChatOpenAI(
        azure_endpoint=AZURE_OAI_ENDPOINT,
        azure_deployment=AZURE_OAI_DEPLOYMENT,
        api_key=AZURE_OAI_API_KEY,
        api_version=AZURE_OAI_API_VERSION,
        temperature=0.7,
        max_tokens=4000,
        top_p=1.0,
        max_retries=0)

async def main():
    client=MultiServerMCPClient(
        {
            "calculator":{
                "url": "https://pretty-emerald-mule.fastmcp.app/mcp",
                "transport": "streamable_http",
            
            },
            "expense": {
                "url": "https://old-cyan-beetle.fastmcp.app/mcp",
                "transport": "streamable_http",
            }

        }
    )

    tools=await client.get_tools()
    agent=create_react_agent(
        model,tools
    )

    _response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    )

    print("Response:", _response['messages'][-1].content)

asyncio.run(main())