import os
import json
import asyncio
import sys

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Windows-specific fix: ensure selector event loop policy to reduce asyncio issues
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# Azure OpenAI Configuration (do not hardcode secrets; use your .env or environment)
AZURE_OAI_ENDPOINT = os.getenv("AZURE_OAI_ENDPOINT")
AZURE_OAI_API_KEY = os.getenv("AZURE_OAI_API_KEY")
AZURE_OAI_DEPLOYMENT = os.getenv("AZURE_OAI_DEPLOYMENT")
AZURE_OAI_API_VERSION = os.getenv("AZURE_OAI_API_VERSION", "2024-12-01-preview")

# Validate required environment variables
if not all([AZURE_OAI_ENDPOINT, AZURE_OAI_API_KEY, AZURE_OAI_DEPLOYMENT]):
    raise RuntimeError(
        "Missing required Azure OpenAI environment variables: "
        "AZURE_OAI_ENDPOINT, AZURE_OAI_API_KEY, AZURE_OAI_DEPLOYMENT"
    )

# Initialize Azure OpenAI model (synchronous usage to avoid event loop issues)
model = AzureChatOpenAI(
    azure_endpoint=AZURE_OAI_ENDPOINT,
    azure_deployment=AZURE_OAI_DEPLOYMENT,
    api_key=AZURE_OAI_API_KEY,
    api_version=AZURE_OAI_API_VERSION,
    temperature=0.7,
    max_tokens=4000,
    top_p=1.0,
    max_retries=0,
)

# MCP server configurations
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


# System prompt for the assistant
SYSTEM_PROMPT = (
    "You have access to tools. When you choose to call a tool, do not narrate status updates. "
    "After tools run, return only a concise final answer."
)

# Streamlit page configuration
st.set_page_config(page_title="MCP Chat", page_icon="🧰", layout="wide")
st.title("🧰 MCP Chat with Intermediate Steps")

# Helper to safely parse JSON args
def _safe_parse_args(args):
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return args
    return args or {}

# Helper to display tool calls
def display_tool_calls(tool_calls):
    """Display tool calls in an expandable section"""
    if not tool_calls:
        return
    
    with st.expander("🔧 Tool Calls", expanded=True):
        for i, tc in enumerate(tool_calls, 1):
            st.markdown(f"**Tool #{i}: {tc.get('name', 'Unknown')}**")
            args = _safe_parse_args(tc.get('args'))
            st.json(args)

# Helper to display tool results
def display_tool_results(tool_msgs):
    """Display tool results in an expandable section"""
    if not tool_msgs:
        return
    
    with st.expander("✅ Tool Results", expanded=True):
        for i, msg in enumerate(tool_msgs, 1):
            st.markdown(f"**Result #{i}:**")
            try:
                # Try to parse and display as JSON if possible
                content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                st.json(content)
            except:
                st.code(msg.content)

# One-time initialization
if "initialized" not in st.session_state:
    # 1) LLM
    st.session_state.llm = model

    # 2) MCP tools (async one-time fetch)
    st.session_state.client = MultiServerMCPClient(SERVERS)
    try:
        tools = asyncio.run(st.session_state.client.get_tools())
    except Exception as e:
        st.error(f"Failed to load MCP tools: {e}")
        tools = []
    st.session_state.tools = tools
    st.session_state.tool_by_name = {t.name: t for t in tools}

    # 3) Bind tools to the LLM (produces a Runnable; we'll use synchronous invoke)
    st.session_state.llm_with_tools = st.session_state.llm.bind_tools(tools)

    # 4) Conversation state
    st.session_state.history = [SystemMessage(content=SYSTEM_PROMPT)]
    
    # 5) Display state for intermediate steps
    st.session_state.messages_display = []  # For rendering
    
    st.session_state.initialized = True

# Sidebar with available tools
with st.sidebar:
    st.header("🔧 Available Tools")
    if st.session_state.tools:
        for i, tool in enumerate(st.session_state.tools, 1):
            with st.expander(f"{i}. {tool.name}"):
                st.write(tool.description)
    else:
        st.info("No tools loaded")
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = [SystemMessage(content=SYSTEM_PROMPT)]
        st.session_state.messages_display = []
        st.rerun()

# Render chat history with intermediate steps
for msg_data in st.session_state.messages_display:
    msg_type = msg_data["type"]
    
    if msg_type == "user":
        with st.chat_message("user"):
            st.markdown(msg_data["content"])
    
    elif msg_type == "assistant_thinking":
        with st.chat_message("assistant"):
            st.info(f"💭 **Thinking:** {msg_data['content']}")
    
    elif msg_type == "tool_calls":
        with st.chat_message("assistant"):
            display_tool_calls(msg_data["tool_calls"])
    
    elif msg_type == "tool_results":
        with st.chat_message("assistant"):
            display_tool_results(msg_data["tool_msgs"])
    
    elif msg_type == "assistant_final":
        with st.chat_message("assistant"):
            st.markdown(msg_data["content"])

# Chat input
user_text = st.chat_input("Type a message…")
if user_text:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_text)
    
    # Store in display history
    st.session_state.messages_display.append({
        "type": "user",
        "content": user_text
    })
    
    # Add to conversation history
    st.session_state.history.append(HumanMessage(content=user_text))

    # Step counter for this interaction
    step_num = 0

    # First pass: decide whether to call tools (synchronous invoke)
    try:
        with st.chat_message("assistant"):
            with st.spinner("🤔 Processing..."):
                first = st.session_state.llm_with_tools.invoke(st.session_state.history)
    except Exception as e:
        st.error(f"Model error: {e}")
        first = None

    if not first:
        st.stop()

    tool_calls = getattr(first, "tool_calls", None)

    if not tool_calls:
        # No tools → show & store assistant reply as final answer
        with st.chat_message("assistant"):
            st.markdown(first.content or "")
        
        st.session_state.messages_display.append({
            "type": "assistant_final",
            "content": first.content or ""
        })
        st.session_state.history.append(first)
    else:
        # Tools are being called - show intermediate steps
        step_num += 1
        
        # Show AI thinking (if there's any content before tool calls)
        if first.content:
            with st.chat_message("assistant"):
                st.info(f"💭 **Step {step_num} - Thinking:** {first.content}")
            
            st.session_state.messages_display.append({
                "type": "assistant_thinking",
                "content": first.content
            })
        
        # Show tool calls
        with st.chat_message("assistant"):
            st.markdown(f"**📍 Step {step_num} - Calling Tools**")
            display_tool_calls(tool_calls)
        
        st.session_state.messages_display.append({
            "type": "tool_calls",
            "tool_calls": tool_calls
        })
        
        # ── IMPORTANT ORDER ──
        # 1) Append assistant message WITH tool_calls
        st.session_state.history.append(first)

        # 2) Execute requested tools and append ToolMessages
        tool_msgs = []
        with st.chat_message("assistant"):
            with st.spinner("⚙️ Executing tools..."):
                for tc in tool_calls:
                    name = tc.get("name")
                    args = _safe_parse_args(tc.get("args"))
                    tool = st.session_state.tool_by_name.get(name)

                    if tool is None:
                        # Tool not found; record an error ToolMessage
                        tool_msgs.append(
                            ToolMessage(
                                tool_call_id=tc.get("id", ""),
                                content=json.dumps({"error": f"Tool '{name}' not found"})
                            )
                        )
                        continue

                    try:
                        # Use synchronous invoke to avoid event loop issues
                        res = tool.invoke(args)
                        tool_msgs.append(
                            ToolMessage(
                                tool_call_id=tc.get("id", ""),
                                content=json.dumps(res)
                            )
                        )
                    except Exception as e:
                        tool_msgs.append(
                            ToolMessage(
                                tool_call_id=tc.get("id", ""),
                                content=json.dumps({"error": str(e)})
                            )
                        )
        
        # Show tool results
        with st.chat_message("assistant"):
            display_tool_results(tool_msgs)
        
        st.session_state.messages_display.append({
            "type": "tool_results",
            "tool_msgs": tool_msgs
        })
        
        st.session_state.history.extend(tool_msgs)

        # 3) Final assistant reply using tool outputs
        try:
            with st.chat_message("assistant"):
                with st.spinner("💬 Generating final response..."):
                    final = st.session_state.llm.invoke(st.session_state.history)
        except Exception as e:
            st.error(f"Model error during final response: {e}")
            final = AIMessage(content="Sorry, I encountered an error generating the response.")

        # Show final answer
        with st.chat_message("assistant"):
            st.success(f"🎯 **Final Answer:**")
            st.markdown(final.content or "")
        
        st.session_state.messages_display.append({
            "type": "assistant_final",
            "content": final.content or ""
        })
        
        st.session_state.history.append(AIMessage(content=final.content or ""))
