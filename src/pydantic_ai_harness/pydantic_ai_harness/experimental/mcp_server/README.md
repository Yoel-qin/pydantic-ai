# MCP Server (experimental)

Serve existing Pydantic AI toolsets as an MCP server, so MCP clients such as Claude Code can call them
without the tools being rewritten as FastMCP tools. This is the reverse of
[`MCPToolset`](https://ai.pydantic.dev/mcp/) (Pydantic AI docs), which consumes MCP servers as toolsets.

Prototype for [pydantic-ai#9191](https://github.com/pydantic/pydantic-ai/issues/9191); experimental: the
API may change or be removed in any release, without a deprecation period, and importing it emits a
`HarnessExperimentalWarning`.

## Installation

uv:

```bash
uv add "pydantic-ai-harness[mcp-server]"
```

pip:

```bash
pip install "pydantic-ai-harness[mcp-server]"
```

## Usage

```python
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.experimental.mcp_server import run_mcp_toolset_server

toolset = FunctionToolset()


@toolset.tool
def double(ctx: RunContext[dict[str, str]], x: int) -> str:
    return ctx.deps['prefix'] + str(x * 2)


run_mcp_toolset_server(toolsets=[toolset], name='math', deps={'prefix': 'result: '})
```

Run the script and point an MCP client at its stdio transport; the client sees one tool, `double`,
with the JSON schema derived from the type hints, and calls return the tool's result.

`mcp_server_from_toolsets` builds the [`FastMCP`](https://gofastmcp.com) server without running it, for
custom transports (`stdio`, `http` or `sse`) or embedding.

## RunContext contract

Each tool call executes against one synthetic RunContext built at server construction:

| Field | Value |
| --- | --- |
| `deps` | the `deps` argument passed to the server |
| `model` | the `model` argument, or a placeholder whose metadata is benign and whose requests fail clearly |
| `usage` | a fresh `RunUsage` shared by the server |
| everything else | the default; there is no agent run, so `messages`, `retry` and friends carry no run state |

## Out of scope for the prototype

- Tools that require approval, external execution or deferred loading are skipped rather than served.
- Dynamic toolsets resolve once at construction, not per session or per call.
- Per-client session deps need the session-contract work tracked in the issue.

Source: [`pydantic_ai_harness/experimental/mcp_server/`](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/experimental/mcp_server/).
