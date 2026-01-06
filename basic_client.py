import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import ToolMessage
import json
import os

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

SERVERS = {
    "calculator": {
        "url": "https://pretty-emerald-mule.fastmcp.app/mcp",
        "transport": "streamable_http",
    },
    "expense": {
        "url": "https://old-cyan-beetle.fastmcp.app/mcp",
        "transport": "streamable_http",
    }
}


async def main():
    
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()


    named_tools = {}
    for tool in tools:
        named_tools[tool.name] = tool

    print("Available tools:", named_tools.keys())

    llm = model
    llm_with_tools = llm.bind_tools(tools)

    prompt = "Please add 30 euros spend on bus on 29 december, 2025."
    response = await llm_with_tools.ainvoke(prompt)

    if not getattr(response, "tool_calls", None):
        print("\nLLM Reply:", response.content)
        return

    tool_messages = []
    for tc in response.tool_calls:
        selected_tool = tc["name"]
        selected_tool_args = tc.get("args") or {}
        selected_tool_id = tc["id"]

        result = await named_tools[selected_tool].ainvoke(selected_tool_args)
        tool_messages.append(ToolMessage(tool_call_id=selected_tool_id, content=json.dumps(result)))
        

    final_response = await llm_with_tools.ainvoke([prompt, response, *tool_messages])
    print(f"Final response: {final_response.content}")


if __name__ == '__main__':
    asyncio.run(main())