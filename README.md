# LangChain MCP Client

A simple MCP (Model Context Protocol) client using LangChain 1.x and Azure OpenAI.

## Features

- Uses modern LangChain 1.x API (`create_agent`)
- Connects to multiple MCP servers
- Azure OpenAI integration
- Environment variable configuration

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

## Configuration

The application uses environment variables for configuration. These are already set in the `.env` file:

- `AZURE_OAI_ENDPOINT`: Your Azure OpenAI endpoint
- `AZURE_OAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OAI_DEPLOYMENT`: Your deployment name
- `AZURE_OAI_API_VERSION`: API version

**To use your own credentials:** Edit the `.env` file with your values.

## Usage

Run the client:
```bash
python client.py
```

The client will:
1. Connect to the configured MCP servers (calculator and expense tracker)
2. Load available tools
3. Process a sample question using the agent
4. Display the answer

## MCP Servers

The client connects to two MCP servers:
- **Calculator**: `https://pretty-emerald-mule.fastmcp.app/mcp`
- **Expense Tracker**: `https://old-cyan-beetle.fastmcp.app/mcp`

## Project Structure

```
.
├── client.py           # Main application file
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (configuration)
└── README.md          # This file
```

## Requirements

- Python 3.8+
- Network access to MCP servers
- Valid Azure OpenAI credentials

## Dependencies

- **langchain** (1.2.0): Framework for building LLM applications
- **langchain-openai** (0.2.14): Azure OpenAI integration
- **langchain-mcp-adapters** (0.1.3): MCP server adapters
- **python-dotenv** (1.0.1): Environment variable management

See `requirements.txt` for complete list with versions.

## Troubleshooting

**Network Error (403 Forbidden):**
- Check your network connection
- Verify you can access the MCP server URLs
- Check if you're behind a proxy

**Azure OpenAI Errors:**
- Verify your API key is correct in `.env`
- Check your Azure OpenAI endpoint URL
- Ensure your deployment name matches

**Import Errors:**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Verify you're using Python 3.8 or higher

## License

This project uses the Apache 2.0 License (see original LICENSE file).
