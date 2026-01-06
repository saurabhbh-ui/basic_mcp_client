"""
LangChain MCP Client - Interactive Mode
Allows multiple questions in a loop
Shows intermediate steps and tool calls
"""

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import asyncio
from dotenv import load_dotenv
import os
import json

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

def print_separator():
    """Print a visual separator"""
    print("\n" + "="*80 + "\n")

def display_tool_call(tool_call, index):
    """Display a tool call in a formatted way"""
    print(f"🔧 Tool Call #{index + 1}")
    print(f"   Tool: {tool_call.get('name', 'Unknown')}")
    print(f"   Arguments: {json.dumps(tool_call.get('args', {}), indent=6)}")

def display_tool_result(message, index):
    """Display a tool result in a formatted way"""
    print(f"✅ Tool Result #{index + 1}")
    print(f"   Content: {message.content}")

def display_thinking(message):
    """Display AI thinking/reasoning"""
    if message.content and not hasattr(message, 'tool_calls'):
        print(f"💭 AI Thinking: {message.content}")

async def process_question(agent, question):
    """Process a single question and show intermediate steps"""
    
    print_separator()
    print("🤖 Agent Processing (showing all intermediate steps)...")
    print_separator()
    
    # Stream the agent execution to see intermediate steps
    step_number = 0
    async for event in agent.astream(
        {"messages": [{"role": "user", "content": question}]}
    ):
        for node_name, node_data in event.items():
            if node_name == "agent" or node_name == "model":
                # This is the agent/model thinking
                messages = node_data.get("messages", [])
                for msg in messages:
                    if isinstance(msg, AIMessage):
                        step_number += 1
                        print(f"📍 Step {step_number}:")
                        
                        # Show AI thinking/reasoning
                        if msg.content:
                            display_thinking(msg)
                        
                        # Show tool calls if any
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for i, tool_call in enumerate(msg.tool_calls):
                                display_tool_call(tool_call, i)
                        
                        print()
            
            elif node_name == "tools":
                # This is tool execution results
                messages = node_data.get("messages", [])
                for i, msg in enumerate(messages):
                    if isinstance(msg, ToolMessage):
                        display_tool_result(msg, i)
                        print()
    
    print_separator()
    
    # Get final response
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    
    # Extract and display the final answer
    answer = response['messages'][-1].content
    print(f"🎯 Final Answer: {answer}")
    print_separator()

async def main():
    """Main function to run the MCP client agent in interactive mode"""
    
    print("="*80)
    print("🤖 LangChain MCP Client - Interactive Mode")
    print("="*80)
    
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
    print("\n🔄 Loading tools from MCP servers...")
    tools = await client.get_tools()
    print(f"✅ Loaded {len(tools)} tools")
    
    # Display available tools
    print("\n📋 Available Tools:")
    for i, tool in enumerate(tools, 1):
        print(f"   {i}. {tool.name}: {tool.description}")
    
    print_separator()
    
    # Create agent using LangChain 1.x API
    agent = create_agent(
        model,
        tools=tools,
        system_prompt="You are a helpful assistant. Use the available tools to answer questions accurately. Think step by step."
    )

    print("💬 Interactive Chat Mode")
    print("   Type your questions and press Enter")
    print("   Type 'quit', 'exit', or 'q' to stop")
    print_separator()

    # Interactive loop
    while True:
        try:
            # Get question from user input
            question = input("❓ Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                print("⚠️  Please enter a question.")
                continue
            
            # Process the question
            await process_question(agent, question)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.\n")

if __name__ == "__main__":
    asyncio.run(main())