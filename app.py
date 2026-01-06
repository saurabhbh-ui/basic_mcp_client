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
    },
}

# System prompt for the assistant
SYSTEM_PROMPT = (
    "You have access to tools. When you choose to call a tool, do not narrate status updates. "
    "After tools run, return only a concise final answer."
)

# Streamlit page configuration
st.set_page_config(page_title="MCP Chat", page_icon="🧰", layout="centered")
st.title("🧰 MCP Chat")

# Helper to safely parse JSON args
def _safe_parse_args(args):
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return args
    return args or {}

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
    st.session_state.initialized = True

# Render chat history (skip system + tool messages; hide intermediate AI with tool_calls)
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        if getattr(msg, "tool_calls", None):
            # Skip intermediate assistant messages that contain tool_calls
            continue
        with st.chat_message("assistant"):
            st.markdown(msg.content)
    # ToolMessage and SystemMessage are not rendered as bubbles

# Chat input
user_text = st.chat_input("Type a message…")
if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)
    st.session_state.history.append(HumanMessage(content=user_text))

    # First pass: decide whether to call tools (synchronous invoke)
    try:
        first = st.session_state.llm_with_tools.invoke(st.session_state.history)
    except Exception as e:
        st.error(f"Model error: {e}")
        first = None

    if not first:
        st.stop()

    tool_calls = getattr(first, "tool_calls", None)

    if not tool_calls:
        # No tools → show & store assistant reply
        with st.chat_message("assistant"):
            st.markdown(first.content or "")
        st.session_state.history.append(first)
    else:
        # ── IMPORTANT ORDER ──
        # 1) Append assistant message WITH tool_calls (do NOT render)
        st.session_state.history.append(first)

        # 2) Execute requested tools and append ToolMessages (do NOT render)
        tool_msgs = []
        for tc in tool_calls:
            name = tc.get("name")
            args = _safe_parse_args(tc.get("args"))
            tool = st.session_state.tool_by_name.get(name)

            if tool is None:
                # Tool not found; record an error ToolMessage to give the LLM context
                tool_msgs.append(ToolMessage(tool_call_id=tc.get("id", ""), content=json.dumps({"error": f"Tool '{name}' not found"})))
                continue

            try:
                # Use synchronous invoke to avoid event loop issues
                res = tool.invoke(args)
                tool_msgs.append(ToolMessage(tool_call_id=tc.get("id", ""), content=json.dumps(res)))
            except Exception as e:
                tool_msgs.append(ToolMessage(tool_call_id=tc.get("id", ""), content=json.dumps({"error": str(e)})))

        st.session_state.history.extend(tool_msgs)

        # 3) Final assistant reply using tool outputs → render & store (synchronous invoke)
        try:
            final = st.session_state.llm.invoke(st.session_state.history)
        except Exception as e:
            st.error(f"Model error during final response: {e}")
            final = AIMessage(content="Sorry, I encountered an error generating the response.")

        with st.chat_message("assistant"):
            st.markdown(final.content or "")
        st.session_state.history.append(AIMessage(content=final.content or ""))