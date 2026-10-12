from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from mcp.types import TextContent

from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.experimental.mcp_server import mcp_server_from_toolsets, run_mcp_toolset_server

try:
    from fastmcp import FastMCP
    from fastmcp.client import Client
    from fastmcp.client.client import CallToolResult
except ImportError:  # pragma: no cover
    pytest.skip('fastmcp not installed', allow_module_level=True)


def _math_toolset() -> FunctionToolset[dict[str, str]]:
    toolset = FunctionToolset[dict[str, str]]()

    @toolset.tool
    def double(ctx: RunContext[dict[str, str]], x: int) -> str:
        return ctx.deps['prefix'] + str(x * 2)

    return toolset


def _text(result: CallToolResult) -> str:
    """The text of the first content block; every adapter result is text-only."""
    content = result.content[0]
    assert isinstance(content, TextContent)
    return content.text


class TestMcpServer:
    async def test_tool_listed_with_schema(self):
        server = await mcp_server_from_toolsets(toolsets=[_math_toolset()], name='math')

        async with Client(server) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}

        assert 'double' in tools
        assert tools['double'].inputSchema == {
            'properties': {'x': {'type': 'integer'}},
            'required': ['x'],
            'type': 'object',
            'additionalProperties': False,
        }

    async def test_tool_call_returns_result(self):
        server = await mcp_server_from_toolsets(toolsets=[_math_toolset()], name='math', deps={'prefix': 'result: '})

        async with Client(server) as client:
            result = await client.call_tool('double', {'x': 21})

        assert _text(result) == 'result: 42'
        assert not result.is_error

    async def test_failing_tool_is_an_error_result(self):
        toolset = FunctionToolset[None]()

        @toolset.tool
        def boom(ctx: RunContext[None]) -> str:
            raise ValueError('kaboom')

        server = await mcp_server_from_toolsets(toolsets=[toolset], name='failing')

        async with Client(server) as client:
            result = await client.call_tool('boom', {}, raise_on_error=False)

        assert result.is_error
        assert 'kaboom' in _text(result)

    async def test_deferred_and_non_function_tools_are_skipped(self):
        toolset = FunctionToolset[None]()

        @toolset.tool(requires_approval=True)
        def dangerous(ctx: RunContext[None]) -> str:
            return 'nope'  # pragma: no cover

        @toolset.tool
        def plain(ctx: RunContext[None]) -> str:
            return 'fine'

        server = await mcp_server_from_toolsets(toolsets=[toolset], name='mixed')

        async with Client(server) as client:
            names = [tool.name for tool in await client.list_tools()]

        assert names == ['plain']

    async def test_model_placeholder_is_benign_metadata(self):
        """Without `model=`, ctx.model is a placeholder whose metadata is harmless and clearly named."""
        toolset = FunctionToolset[None]()

        @toolset.tool
        def probe(ctx: RunContext[None]) -> str:
            return ctx.model.system

        server = await mcp_server_from_toolsets(toolsets=[toolset], name='no-model')

        async with Client(server) as client:
            result = await client.call_tool('probe', {})

        assert not result.is_error
        assert _text(result) == 'mcp-server'

    async def test_schema_validation_rejects_bad_args(self):
        """Arguments are schema-validated before dispatch, like the agent loop validates model calls."""
        server = await mcp_server_from_toolsets(toolsets=[_math_toolset()], name='math')

        async with Client(server) as client:
            result = await client.call_tool('double', {'x': 'not-an-integer'}, raise_on_error=False)

        assert result.is_error
        assert 'argument validation failed' in _text(result)

    async def test_args_validator_approval_refuses_the_call(self):
        """A validator that gates calls on approval must refuse the call, not execute it."""
        toolset = FunctionToolset[None]()

        def approve_small_only(ctx: RunContext[None], amount: int) -> None:
            if amount > 100:
                raise ApprovalRequired()

        @toolset.tool(args_validator=approve_small_only)
        def transfer(ctx: RunContext[None], amount: int) -> str:
            return f'transferred {amount}'

        server = await mcp_server_from_toolsets(toolsets=[toolset], name='gated')

        async with Client(server) as client:
            refused = await client.call_tool('transfer', {'amount': 500}, raise_on_error=False)
            allowed = await client.call_tool('transfer', {'amount': 50}, raise_on_error=False)

        assert refused.is_error
        assert 'ApprovalRequired' in _text(refused)
        assert not allowed.is_error
        assert _text(allowed) == 'transferred 50'

    async def test_tool_return_metadata_is_not_leaked(self):
        """`ToolReturn.metadata` is application-only and must not reach the client via `repr`."""
        toolset = FunctionToolset[None]()

        @toolset.tool
        def secret(ctx: RunContext[None]) -> ToolReturn[str]:
            return ToolReturn('ok', metadata={'internal_token': 'hunter2'})

        server = await mcp_server_from_toolsets(toolsets=[toolset], name='leaky')

        async with Client(server) as client:
            result = await client.call_tool('secret', {})

        assert _text(result) == 'ok'
        assert 'hunter2' not in _text(result)

    async def test_empty_toolsets_serve_no_tools(self):
        server = await mcp_server_from_toolsets(toolsets=[], name='empty')

        async with Client(server) as client:
            tools = await client.list_tools()

        assert tools == []

    def test_run_mcp_toolset_server_builds_and_runs(self, monkeypatch: pytest.MonkeyPatch):
        """Run in a worker thread: the harness conftest leaves an asyncio context behind in sync tests,
        and `anyio.run` refuses to start inside one — as it would in any host app with a live loop."""
        transports: list[str | None] = []

        def record_run(server: FastMCP, transport: str | None = None) -> None:
            transports.append(transport)

        monkeypatch.setattr(FastMCP, 'run', record_run)

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(run_mcp_toolset_server, toolsets=[_math_toolset()], name='math').result()

        assert transports == ['stdio']
