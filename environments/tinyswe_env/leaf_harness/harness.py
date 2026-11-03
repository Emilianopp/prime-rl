"""leaf-harness: the Leaf coding-agent harness (github.com/microsoft/FrogNano, MIT) for env-driven
exchanges: `env.player.harness.id = "leaf-harness"`.

Leaf gives the model five function-calling tools (Read, Write, Edit, Glob, Bash) and runs every call
it makes, several per reply allowed, until a reply makes none. Here the model call is the
inline-chat harness's (one streamed chat completion through the rollout's interception endpoint,
from the env server's own process) with Leaf's tool schemas attached, and the ENV runs the tools: it
reads the assistant's `tool_calls` off each `Segment`, executes them in its sandbox (tinyswe's
command server implements them, cmdserver.py `leaf` op) and answers with one `ToolMessage` per call
through `interaction.turn(...)`. The renderer turns the schemas and calls into the model's own
format (Qwen3.5: `<tool_call><function=...>`), which is what FrogNano was trained on.
"""

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.errors import MODEL_TRANSPORT_ERROR_EXIT_CODE
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.utils.compaction import is_context_overflow
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

LEAF_SYSTEM_PROMPT = (
    "You are a software-engineering agent working in a checked-out repository.\n"
    "Use the available tools to inspect files, make focused edits, and run commands.\n"
    "Solve the task, validate when practical, then give a brief final answer."
)


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


# Leaf's schemas verbatim (frognano/harness/leaf/tools.py).
LEAF_TOOLS = [
    _fn(
        "Read",
        "Read a text file. Returns numbered lines.",
        {
            "file_path": {"type": "string", "description": "Path to the file to read."},
            "offset": {"type": "integer", "description": "Optional 1-based starting line."},
            "limit": {"type": "integer", "description": "Optional maximum number of lines."},
        },
        ["file_path"],
    ),
    _fn(
        "Write",
        "Write a complete file, creating parent directories when needed.",
        {
            "file_path": {"type": "string", "description": "Path to write."},
            "content": {"type": "string", "description": "Full file contents."},
        },
        ["file_path", "content"],
    ),
    _fn(
        "Edit",
        "Replace an exact string in a text file.",
        {
            "file_path": {"type": "string", "description": "Path to edit."},
            "old_string": {"type": "string", "description": "Exact text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence instead of requiring one match.",
            },
        },
        ["file_path", "old_string", "new_string"],
    ),
    _fn(
        "Glob",
        "Find files by glob pattern.",
        {
            "pattern": {"type": "string", "description": "Glob pattern, e.g. **/*.py."},
            "path": {"type": "string", "description": "Optional directory to search from."},
        },
        ["pattern"],
    ),
    _fn(
        "Bash",
        "Run a shell command in the repository workdir.",
        {
            "command": {"type": "string", "description": "Command to run."},
            "description": {"type": "string", "description": "Short description of the command."},
            "timeout": {"type": "number", "description": "Optional timeout in seconds."},
        },
        ["command"],
    ),
]

_HTTP: httpx.AsyncClient | None = None


def _http() -> httpx.AsyncClient:
    """One connection pool per env-server process, shared by every rollout."""
    global _HTTP
    if _HTTP is None:
        _HTTP = httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=5.0),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=256),
        )
    return _HTTP


class LeafHarnessConfig(HarnessConfig):
    parallel_tool_calls: bool = True
    """Let a reply make several tool calls, as Leaf does."""


class LeafHarness(Harness[LeafHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_RESUME = True
    EXECUTES_CODE = False
    NEEDS_CONTAINER = False

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ProgramResult:
        if mcp_urls:
            raise ValueError("leaf-harness serves its own five tools; MCP servers are not supported")
        system, prompt = self.resolve_prompt(data)
        messages = [{"role": "system", "content": system}] if system else []
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif prompt is not None:
            messages.extend(message_to_wire(message) for message in prompt)
        client = AsyncOpenAI(base_url=endpoint, api_key=secret, http_client=_http(), max_retries=2)
        try:
            stream = await client.chat.completions.create(
                model=ctx.model,
                messages=messages,
                tools=LEAF_TOOLS,
                tool_choice="auto",
                parallel_tool_calls=self.config.parallel_tool_calls,
                stream=True,
                stream_options={"include_usage": True},
            )
            finished = False
            async for chunk in stream:
                finished = finished or any(choice.finish_reason is not None for choice in chunk.choices)
            if not finished:
                return ProgramResult(
                    MODEL_TRANSPORT_ERROR_EXIT_CODE, "", "model stream ended before a completion finished"
                )
        except APIStatusError as error:
            if is_context_overflow(error):  # out of context ends the run, as in Leaf (max_context_len)
                return ProgramResult(0, "", "")
            return ProgramResult(1, "", f"{error.status_code}: {error}")
        except (APIConnectionError, httpx.TransportError) as error:
            return ProgramResult(MODEL_TRANSPORT_ERROR_EXIT_CODE, "", repr(error))
        return ProgramResult(0, "", "")
