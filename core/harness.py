"""
Multi-Agent Harness for MACB
============================
Extensible harness for running multi-agent coding tasks with pluggable LLM providers.

Features:
- Provider abstraction (OpenAI, Anthropic, custom)
- Multiple communication patterns (isolated, broadcast, direct, scratchpad)
- Tool use (file ops, git, tests, messaging)
- Execution tracking and token accounting
"""

from __future__ import annotations
import os
import json
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Iterator, Callable, Any, Literal
from enum import Enum
from pathlib import Path
from datetime import datetime
import hashlib

# =============================================================================
# MESSAGE SCHEMA (Canonical Internal Format)
# =============================================================================

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"


@dataclass
class ToolCall:
    """A tool invocation by the model."""
    id: str
    name: str
    arguments: dict


@dataclass
class ContentBlock:
    """A block of content (text, image, etc.)."""
    type: Literal["text", "image", "tool_use", "tool_result"]
    text: str | None = None
    tool_call: ToolCall | None = None
    tool_call_id: str | None = None
    image_url: str | None = None


@dataclass
class Message:
    """Canonical message format used internally."""
    role: Role
    content: str | list[ContentBlock]
    tool_calls: list[ToolCall] = field(default_factory=list)
    name: str | None = None  # For multi-agent identification
    timestamp: float = field(default_factory=time.time)
    
    def to_text(self) -> str:
        """Extract plain text content."""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(
            block.text for block in self.content 
            if block.type == "text" and block.text
        )


@dataclass
class ToolDefinition:
    """Definition of a tool available to agents."""
    name: str
    description: str
    parameters: dict  # JSON Schema
    handler: Callable[..., str] | None = None


@dataclass 
class CompletionResponse:
    """Response from an LLM completion."""
    message: Message
    finish_reason: str
    usage: dict  # {"prompt_tokens": N, "completion_tokens": M}
    raw_response: Any = None  # Provider-specific raw response


# =============================================================================
# PROVIDER ABSTRACTION
# =============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    name: str
    
    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> CompletionResponse:
        """Generate a completion for the given messages."""
        pass
    
    @abstractmethod
    def to_provider_messages(self, messages: list[Message]) -> list[dict]:
        """Convert canonical messages to provider-specific format."""
        pass
    
    @abstractmethod
    def to_provider_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert tool definitions to provider-specific format."""
        pass
    
    @abstractmethod
    def from_provider_response(self, response: Any) -> CompletionResponse:
        """Convert provider response to canonical format."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count. Override for accurate counting."""
        return len(text) // 4  # Rough estimate


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (and compatible APIs like Azure, Together, etc.)."""
    
    name = "openai"
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.openai.com/v1"
        self.organization = organization
        
        # Import here to make dependency optional
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
            )
        except ImportError:
            self.client = None
            print("Warning: openai package not installed. Install with: pip install openai")
    
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> CompletionResponse:
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        provider_messages = self.to_provider_messages(messages)
        
        request_kwargs = {
            "model": self.model,
            "messages": provider_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        if tools:
            request_kwargs["tools"] = self.to_provider_tools(tools)
            request_kwargs["tool_choice"] = "auto"
        
        response = self.client.chat.completions.create(**request_kwargs)
        return self.from_provider_response(response)
    
    def to_provider_messages(self, messages: list[Message]) -> list[dict]:
        """Convert to OpenAI message format."""
        result = []
        
        for msg in messages:
            if msg.role == Role.SYSTEM:
                result.append({"role": "system", "content": msg.to_text()})
            
            elif msg.role == Role.USER:
                result.append({"role": "user", "content": msg.to_text()})
            
            elif msg.role == Role.ASSISTANT:
                assistant_msg = {"role": "assistant", "content": msg.to_text()}
                if msg.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments)
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                result.append(assistant_msg)
            
            elif msg.role == Role.TOOL_RESULT:
                # Find the tool_call_id from content blocks
                if isinstance(msg.content, list):
                    for block in msg.content:
                        if block.type == "tool_result" and block.tool_call_id:
                            result.append({
                                "role": "tool",
                                "tool_call_id": block.tool_call_id,
                                "content": block.text or ""
                            })
                else:
                    # Fallback for simple content
                    result.append({
                        "role": "tool",
                        "tool_call_id": "unknown",
                        "content": msg.to_text()
                    })
        
        return result
    
    def to_provider_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert to OpenAI tool format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            }
            for tool in tools
        ]
    
    def from_provider_response(self, response) -> CompletionResponse:
        """Convert OpenAI response to canonical format."""
        choice = response.choices[0]
        msg = choice.message
        
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments) if tc.function.arguments else {}
                ))
        
        return CompletionResponse(
            message=Message(
                role=Role.ASSISTANT,
                content=msg.content or "",
                tool_calls=tool_calls,
            ),
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            raw_response=response,
        )


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""
    
    name = "anthropic"
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            self.client = None
            print("Warning: anthropic package not installed. Install with: pip install anthropic")
    
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> CompletionResponse:
        if not self.client:
            raise RuntimeError("Anthropic client not initialized")
        
        # Extract system message
        system = None
        other_messages = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system = msg.to_text()
            else:
                other_messages.append(msg)
        
        provider_messages = self.to_provider_messages(other_messages)
        
        request_kwargs = {
            "model": self.model,
            "messages": provider_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        if system:
            request_kwargs["system"] = system
        
        if tools:
            request_kwargs["tools"] = self.to_provider_tools(tools)
        
        response = self.client.messages.create(**request_kwargs)
        return self.from_provider_response(response)
    
    def to_provider_messages(self, messages: list[Message]) -> list[dict]:
        """Convert to Anthropic message format."""
        result = []
        
        for msg in messages:
            if msg.role == Role.USER:
                result.append({"role": "user", "content": msg.to_text()})
            
            elif msg.role == Role.ASSISTANT:
                content = []
                if msg.to_text():
                    content.append({"type": "text", "text": msg.to_text()})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                result.append({"role": "assistant", "content": content or msg.to_text()})
            
            elif msg.role == Role.TOOL_RESULT:
                content = []
                if isinstance(msg.content, list):
                    for block in msg.content:
                        if block.type == "tool_result":
                            content.append({
                                "type": "tool_result",
                                "tool_use_id": block.tool_call_id,
                                "content": block.text or "",
                            })
                result.append({"role": "user", "content": content})
        
        return result
    
    def to_provider_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert to Anthropic tool format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]
    
    def from_provider_response(self, response) -> CompletionResponse:
        """Convert Anthropic response to canonical format."""
        text_parts = []
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))
        
        return CompletionResponse(
            message=Message(
                role=Role.ASSISTANT,
                content="\n".join(text_parts),
                tool_calls=tool_calls,
            ),
            finish_reason=response.stop_reason,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
            raw_response=response,
        )


class MockProvider(LLMProvider):
    """Mock provider for testing without API calls."""
    
    name = "mock"
    
    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["Mock response"]
        self.call_count = 0
        self.recorded_calls: list[dict] = []
    
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs
    ) -> CompletionResponse:
        self.recorded_calls.append({
            "messages": messages,
            "tools": tools,
            "kwargs": kwargs,
        })
        
        response_data = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        
        tool_calls = []
        content = ""
        
        if isinstance(response_data, str):
            content = response_data
        elif isinstance(response_data, dict):
            content = response_data.get("content", "")
            if "tool_calls" in response_data:
                for tc in response_data["tool_calls"]:
                    tool_calls.append(ToolCall(
                        id=f"call_{len(tool_calls)}_{self.call_count}",
                        name=tc["name"],
                        arguments=tc.get("arguments", {})
                    ))
        
        return CompletionResponse(
            message=Message(
                role=Role.ASSISTANT, 
                content=content,
                tool_calls=tool_calls
            ),
            finish_reason="tool_calls" if tool_calls else "stop",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
    
    def to_provider_messages(self, messages): return []
    def to_provider_tools(self, tools): return []
    def from_provider_response(self, response): pass


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

class AgentTools:
    """Tools available to agents during task execution."""
    
    def __init__(self, workspace: Path, agent_id: str):
        self.workspace = workspace
        self.agent_id = agent_id
        self.message_log: list[dict] = []
    
    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Get all tool definitions for this agent."""
        return [
            ToolDefinition(
                name="read_file",
                description="Read the contents of a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file relative to workspace"}
                    },
                    "required": ["path"]
                },
                handler=self.read_file,
            ),
            ToolDefinition(
                name="write_file",
                description="Write content to a file (creates directories if needed)",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"},
                        "content": {"type": "string", "description": "Content to write"}
                    },
                    "required": ["path", "content"]
                },
                handler=self.write_file,
            ),
            ToolDefinition(
                name="list_files",
                description="List files in a directory",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path", "default": "."}
                    },
                },
                handler=self.list_files,
            ),
            ToolDefinition(
                name="run_command",
                description="Run a shell command in the workspace",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to run"}
                    },
                    "required": ["command"]
                },
                handler=self.run_command,
            ),
            ToolDefinition(
                name="run_tests",
                description="Run pytest tests",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Test file or pattern", "default": ""}
                    },
                },
                handler=self.run_tests,
            ),
            ToolDefinition(
                name="git_status",
                description="Get git status of the workspace",
                parameters={"type": "object", "properties": {}},
                handler=self.git_status,
            ),
            ToolDefinition(
                name="git_diff",
                description="Get git diff",
                parameters={
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "Specific file to diff", "default": ""}
                    },
                },
                handler=self.git_diff,
            ),
            ToolDefinition(
                name="git_commit",
                description="Commit current changes",
                parameters={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Commit message"}
                    },
                    "required": ["message"]
                },
                handler=self.git_commit,
            ),
            ToolDefinition(
                name="send_message",
                description="Send a message to another agent",
                parameters={
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Target agent ID"},
                        "content": {"type": "string", "description": "Message content"}
                    },
                    "required": ["to", "content"]
                },
                handler=self.send_message,
            ),
            ToolDefinition(
                name="read_messages",
                description="Read messages sent to this agent",
                parameters={"type": "object", "properties": {}},
                handler=self.read_messages,
            ),
            ToolDefinition(
                name="submit_work",
                description="Signal that you have completed your task",
                parameters={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "Summary of changes made"}
                    },
                    "required": ["summary"]
                },
                handler=self.submit_work,
            ),
            ToolDefinition(
                name="read_kanban_board",
                description="Read the current state of the Kanban board",
                parameters={"type": "object", "properties": {}},
                handler=self.read_kanban_board,
            ),
            ToolDefinition(
                name="claim_task",
                description="Claim a task from the backlog (moves to in_progress)",
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "ID of the task to claim"}
                    },
                    "required": ["task_id"]
                },
                handler=self.claim_task,
            ),
            ToolDefinition(
                name="update_task_status",
                description="Update the status of a task on the Kanban board",
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "ID of the task"},
                        "status": {"type": "string", "enum": ["todo", "in_progress", "done"], "description": "New status"}
                    },
                    "required": ["task_id", "status"]
                },
                handler=self.update_task_status,
            ),
        ]
    
    def read_file(self, path: str) -> str:
        try:
            full_path = self.workspace / path
            return full_path.read_text()
        except Exception as e:
            return f"Error reading file: {e}"
    
    def write_file(self, path: str, content: str) -> str:
        try:
            full_path = self.workspace / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            return f"Successfully wrote {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error writing file: {e}"
    
    def list_files(self, path: str = ".") -> str:
        try:
            full_path = self.workspace / path
            files = []
            for p in full_path.rglob("*"):
                if p.is_file() and ".git" not in str(p):
                    files.append(str(p.relative_to(self.workspace)))
            return "\n".join(sorted(files))
        except Exception as e:
            return f"Error listing files: {e}"
    
    def run_command(self, command: str) -> str:
        try:
            # Security: limit allowed commands
            allowed_prefixes = ["python", "pip", "pytest", "git", "ls", "cat", "head", "tail", "grep"]
            if not any(command.strip().startswith(p) for p in allowed_prefixes):
                return f"Command not allowed. Allowed prefixes: {allowed_prefixes}"
            
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = result.stdout + result.stderr
            return output[:5000] if output else "(no output)"
        except subprocess.TimeoutExpired:
            return "Command timed out after 60 seconds"
        except Exception as e:
            return f"Error running command: {e}"
    
    def run_tests(self, pattern: str = "") -> str:
        cmd = f"python -m pytest tests/ -v --tb=short"
        if pattern:
            cmd += f" -k {pattern}"
        return self.run_command(cmd)
    
    def git_status(self) -> str:
        return self.run_command("git status")
    
    def git_diff(self, file: str = "") -> str:
        cmd = "git diff"
        if file:
            cmd += f" {file}"
        return self.run_command(cmd)
    
    def git_commit(self, message: str) -> str:
        self.run_command("git add .")
        env_setup = f'GIT_AUTHOR_NAME="{self.agent_id}" GIT_AUTHOR_EMAIL="{self.agent_id}@agent"'
        return self.run_command(f'{env_setup} git commit -m "{message}"')
    
    def send_message(self, to: str, content: str) -> str:
        self.message_log.append({
            "from": self.agent_id,
            "to": to,
            "content": content,
            "timestamp": time.time(),
        })
        return f"Message sent to {to}"
    
    def read_messages(self) -> str:
        # This will be populated by the harness
        messages = [m for m in self.message_log if m["to"] == self.agent_id]
        if not messages:
            return "No messages"
        return "\n".join(
            f"[{m['from']}]: {m['content']}" for m in messages
        )
    
    def submit_work(self, summary: str) -> str:
        return f"WORK_SUBMITTED: {summary}"
    
    def read_kanban_board(self) -> str:
        try:
            full_path = self.workspace / "kanban.json"
            if not full_path.exists():
                return "No Kanban board found."
            return full_path.read_text()
        except Exception as e:
            return f"Error reading Kanban board: {e}"

    def claim_task(self, task_id: str) -> str:
        try:
            full_path = self.workspace / "kanban.json"
            if not full_path.exists():
                return "No Kanban board found."
            
            data = json.loads(full_path.read_text())
            task_found = False
            for task in data:
                if task["id"] == task_id:
                    task_found = True
                    if task["status"] != "todo":
                        return f"Task {task_id} is already {task['status']} by {task.get('assigned_to', 'someone')}."
                    task["status"] = "in_progress"
                    task["assigned_to"] = self.agent_id
                    break
            
            if not task_found:
                return f"Task {task_id} not found."
            
            full_path.write_text(json.dumps(data, indent=2))
            return f"Successfully claimed task {task_id}."
        except Exception as e:
            return f"Error claiming task: {e}"

    def update_task_status(self, task_id: str, status: str) -> str:
        if status not in ["todo", "in_progress", "done"]:
            return "Invalid status. Must be one of: todo, in_progress, done"
        
        try:
            full_path = self.workspace / "kanban.json"
            if not full_path.exists():
                return "No Kanban board found."
            
            data = json.loads(full_path.read_text())
            task_found = False
            for task in data:
                if task["id"] == task_id:
                    task_found = True
                    if task["assigned_to"] != self.agent_id and task["status"] != "todo":
                         # It's an agent, they might steal tasks or take over. Let's allowing but warning.
                         pass
                    task["status"] = status
                    if status == "todo":
                        task["assigned_to"] = None
                    else:
                        task["assigned_to"] = self.agent_id
                    break
            
            if not task_found:
                return f"Task {task_id} not found."
            
            full_path.write_text(json.dumps(data, indent=2))
            return f"Successfully updated task {task_id} to {status}."
        except Exception as e:
            return f"Error updating task: {e}"


# =============================================================================
# AGENT
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for an agent."""
    agent_id: str
    system_prompt: str
    task_description: str
    provider: LLMProvider
    max_turns: int = 50
    max_tokens_per_turn: int = 4096
    temperature: float = 0.7


class Agent:
    """A single agent that can work on tasks using tools."""
    
    def __init__(self, config: AgentConfig, workspace: Path):
        self.config = config
        self.workspace = workspace
        self.tools = AgentTools(workspace, config.agent_id)
        self.conversation: list[Message] = []
        self.total_tokens = {"prompt": 0, "completion": 0}
        self.submitted = False
        self.submission_summary = ""
        
        # Initialize conversation with system prompt and task
        self.conversation.append(Message(
            role=Role.SYSTEM,
            content=config.system_prompt,
        ))
        self.conversation.append(Message(
            role=Role.USER,
            content=f"## Your Task\n\n{config.task_description}\n\nBegin working on your task. Use the available tools to read files, make changes, and test your work. When finished, use the submit_work tool.",
        ))
    
    def step(self) -> tuple[bool, str]:
        """
        Execute one step of agent reasoning.
        Returns (is_done, status_message).
        """
        if self.submitted:
            return True, "Already submitted"
        
        # Get completion from provider
        response = self.config.provider.complete(
            messages=self.conversation,
            tools=self.tools.get_tool_definitions(),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens_per_turn,
        )
        
        # Track token usage
        self.total_tokens["prompt"] += response.usage.get("prompt_tokens", 0)
        self.total_tokens["completion"] += response.usage.get("completion_tokens", 0)
        
        # Add assistant response to conversation
        self.conversation.append(response.message)
        
        # Handle tool calls
        if response.message.tool_calls:
            tool_results = []
            for tc in response.message.tool_calls:
                result = self._execute_tool(tc)
                tool_results.append(ContentBlock(
                    type="tool_result",
                    text=result,
                    tool_call_id=tc.id,
                ))
                
                # Check for submission
                if tc.name == "submit_work" and "WORK_SUBMITTED" in result:
                    self.submitted = True
                    self.submission_summary = tc.arguments.get("summary", "")
            
            # Add tool results to conversation
            self.conversation.append(Message(
                role=Role.TOOL_RESULT,
                content=tool_results,
            ))
            
            return self.submitted, f"Executed {len(tool_results)} tool(s)"
        
        return False, "Generated response"
    
    def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return the result."""
        tool_defs = {t.name: t for t in self.tools.get_tool_definitions()}
        
        if tool_call.name not in tool_defs:
            return f"Unknown tool: {tool_call.name}"
        
        tool = tool_defs[tool_call.name]
        if not tool.handler:
            return f"Tool {tool_call.name} has no handler"
        
        try:
            return tool.handler(**tool_call.arguments)
        except Exception as e:
            return f"Tool error: {e}"
    
    def receive_message(self, from_agent: str, content: str):
        """Receive a message from another agent."""
        self.tools.message_log.append({
            "from": from_agent,
            "to": self.config.agent_id,
            "content": content,
            "timestamp": time.time(),
        })
    
    def get_modified_files(self) -> dict[str, str]:
        """Get all files modified by this agent."""
        result = {}
        # Avoid reading binary files or large data directories
        excluded_dirs = {".git", "__pycache__", ".pytest_cache", "venv", ".venv"}
        excluded_extensions = {".pyc", ".pyo", ".pyd", ".so", ".bin", ".json"} # We already have kanban.json, metadata.json handled separately or we don't need them as 'modified code'
        
        for f in self.workspace.rglob("*"):
            if not f.is_file():
                continue
                
            # Check if any parent is in excluded_dirs
            if any(part in excluded_dirs for part in f.parts):
                continue
                
            if f.suffix in excluded_extensions:
                continue
                
            try:
                rel_path = str(f.relative_to(self.workspace))
                result[rel_path] = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Silently skip binary files
                continue
        return result


# =============================================================================
# COMMUNICATION PATTERNS
# =============================================================================

class CommunicationPattern(Enum):
    ISOLATED = "isolated"       # No communication
    BROADCAST = "broadcast"     # All messages visible to all
    DIRECT = "direct"           # Point-to-point only
    SCRATCHPAD = "scratchpad"   # Shared append-only scratchpad


# =============================================================================
# HARNESS
# =============================================================================

@dataclass
class HarnessConfig:
    """Configuration for the multi-agent harness."""
    communication: CommunicationPattern = CommunicationPattern.DIRECT
    max_total_turns: int = 200
    max_total_tokens: int = 1000000  # Default budget of 1M tokens
    turn_timeout_seconds: int = 120
    checkpoint_every_n_turns: int = 10


@dataclass
class ExecutionTrace:
    """Complete trace of a multi-agent execution."""
    instance_id: str
    agents: list[str]
    turns: list[dict]
    total_tokens: dict[str, dict]
    messages_exchanged: list[dict]
    duration_seconds: float
    outcome: str  # "success", "timeout", "error"

    def to_dict(self) -> dict:
        return asdict(self)


class MultiAgentHarness:
    """
    Harness for running multi-agent task execution.
    
    Manages multiple agents, their communication, and tracks execution.
    """
    
    def __init__(self, config: HarnessConfig | None = None):
        self.config = config or HarnessConfig()
        self.agents: dict[str, Agent] = {}
        self.shared_scratchpad: list[dict] = []
        self.message_queue: list[dict] = []
        self.execution_trace: list[dict] = []
    
    def add_agent(self, agent: Agent):
        """Add an agent to the harness."""
        self.agents[agent.config.agent_id] = agent
    
    def run(self, max_turns: int | None = None, max_tokens: int | None = None) -> ExecutionTrace:
        """
        Run the multi-agent execution.
        
        Uses round-robin turn order. Agents work until all submit, max turns reached,
        or token budget is exhausted.
        """
        max_turns = max_turns or self.config.max_total_turns
        max_tokens = max_tokens or self.config.max_total_tokens
        start_time = time.time()
        
        agent_ids = list(self.agents.keys())
        turn = 0
        outcome = "timeout"
        
        while turn < max_turns:
            all_done = True
            
            # Check total token usage
            total_tokens_used = sum(a.total_tokens["prompt"] + a.total_tokens["completion"] 
                                   for a in self.agents.values())
            
            if total_tokens_used >= max_tokens:
                outcome = "budget_exhausted"
                break
                
            for agent_id in agent_ids:
                agent = self.agents[agent_id]
                
                if agent.submitted:
                    continue
                
                all_done = False
                
                # Deliver any pending messages
                self._deliver_messages(agent_id)
                
                # Execute agent step
                is_done, status = agent.step()
                
                # Record turn
                self.execution_trace.append({
                    "turn": turn,
                    "agent": agent_id,
                    "status": status,
                    "submitted": is_done,
                    "timestamp": time.time(),
                })
                
                # Process any outgoing messages
                self._process_outgoing_messages(agent_id)
                
                turn += 1
                
                # Check mid-loop for budget or turn limit
                total_tokens_used = sum(a.total_tokens["prompt"] + a.total_tokens["completion"] 
                                       for a in self.agents.values())
                if total_tokens_used >= max_tokens:
                    outcome = "budget_exhausted"
                    break
                
                if turn >= max_turns:
                    break
            
            if outcome == "budget_exhausted":
                break
                
            if all_done:
                outcome = "success"
                break
        
        duration = time.time() - start_time
        
        return ExecutionTrace(
            instance_id=hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
            agents=agent_ids,
            turns=self.execution_trace,
            total_tokens={
                aid: agent.total_tokens for aid, agent in self.agents.items()
            },
            messages_exchanged=self.message_queue.copy(),
            duration_seconds=duration,
            outcome=outcome,
        )
    
    def _deliver_messages(self, agent_id: str):
        """Deliver pending messages to an agent based on communication pattern."""
        agent = self.agents[agent_id]
        
        if self.config.communication == CommunicationPattern.ISOLATED:
            return
        
        elif self.config.communication == CommunicationPattern.BROADCAST:
            # Deliver all messages from other agents
            for msg in self.message_queue:
                if msg["from"] != agent_id:
                    agent.receive_message(msg["from"], msg["content"])
        
        elif self.config.communication == CommunicationPattern.DIRECT:
            # Deliver only messages addressed to this agent
            for msg in self.message_queue:
                if msg["to"] == agent_id and not msg.get("delivered_to", {}).get(agent_id):
                    agent.receive_message(msg["from"], msg["content"])
                    msg.setdefault("delivered_to", {})[agent_id] = True
        
        elif self.config.communication == CommunicationPattern.SCRATCHPAD:
            # Agent reads from shared scratchpad
            pass  # Handled differently
    
    def _process_outgoing_messages(self, agent_id: str):
        """Process messages sent by an agent."""
        agent = self.agents[agent_id]
        
        # Check agent's message log for new outgoing messages
        for msg in agent.tools.message_log:
            if msg["from"] == agent_id:
                # Add to queue if not already there
                if msg not in self.message_queue:
                    self.message_queue.append(msg)


# =============================================================================
# CONVENIENCE: RUN TASK INSTANCE
# =============================================================================

def run_task_with_agents(
    task_dir: Path,
    agent_configs: list[dict],
    provider: LLMProvider,
    harness_config: HarnessConfig | None = None,
) -> ExecutionTrace:
    """
    Convenience function to run a task instance with agents.
    
    Args:
        task_dir: Path to task instance directory (from TaskInstance.to_directory)
        agent_configs: List of {"agent_id": str, "task_file": str} dicts
        provider: LLM provider to use for all agents
        harness_config: Optional harness configuration
    
    Returns:
        ExecutionTrace with results
    """
    workspace = task_dir / "starter"
    metadata = json.loads((task_dir / "metadata.json").read_text())
    
    harness = MultiAgentHarness(harness_config)
    
    system_prompt = """You are a skilled software engineer working on a team project.
You have access to tools for reading/writing files, running tests, and communicating with teammates.
Work efficiently, write clean code, and coordinate with your teammates when needed.
When you've completed your task and verified it works, use submit_work to signal completion."""
    
    for cfg in agent_configs:
        agent_id = cfg["agent_id"]
        task_file = task_dir / "tasks" / f"{agent_id}.md"
        task_description = task_file.read_text() if task_file.exists() else "No task description found."
        
        agent = Agent(
            config=AgentConfig(
                agent_id=agent_id,
                system_prompt=system_prompt,
                task_description=task_description,
                provider=provider,
                max_turns=cfg.get("max_turns", 50),
            ),
            workspace=workspace,
        )
        harness.add_agent(agent)
    
    return harness.run()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Example showing how to use the agent harness."""
    print("=" * 70)
    print("MULTI-AGENT HARNESS - Example Usage")
    print("=" * 70)
    
    # Use mock provider for demonstration
    mock_responses = [
        # Agent 1 response
        "I'll start by reading the interface file to understand what I need to implement.",
        # ... more mock responses would go here
    ]
    
    provider = MockProvider(responses=mock_responses)
    
    print("\n1. Creating agents with MockProvider...")
    
    # Create a simple workspace
    import tempfile
    workspace = Path(tempfile.mkdtemp())
    
    # Write some starter files
    (workspace / "shared").mkdir()
    (workspace / "shared" / "interfaces.py").write_text("# Interface definition\nclass MyInterface:\n    pass\n")
    (workspace / "service").mkdir()
    (workspace / "service" / "impl.py").write_text("# TODO: Implement\n")
    
    # Initialize git
    subprocess.run(["git", "init"], cwd=workspace, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=workspace, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=workspace,
        capture_output=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@test.com",
             "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@test.com"}
    )
    
    agent1 = Agent(
        config=AgentConfig(
            agent_id="agent_service",
            system_prompt="You are a backend engineer.",
            task_description="Implement the service in service/impl.py",
            provider=provider,
        ),
        workspace=workspace,
    )
    
    agent2 = Agent(
        config=AgentConfig(
            agent_id="agent_client",
            system_prompt="You are a frontend engineer.",
            task_description="Implement a client that uses the service.",
            provider=provider,
        ),
        workspace=workspace,
    )
    
    print(f"   Created agents: {agent1.config.agent_id}, {agent2.config.agent_id}")
    
    print("\n2. Setting up harness with DIRECT communication...")
    harness = MultiAgentHarness(HarnessConfig(
        communication=CommunicationPattern.DIRECT,
        max_total_turns=10,  # Low for demo
    ))
    harness.add_agent(agent1)
    harness.add_agent(agent2)
    
    print("\n3. Running execution (limited turns for demo)...")
    trace = harness.run()
    
    print(f"\n4. Execution complete!")
    print(f"   Outcome: {trace.outcome}")
    print(f"   Duration: {trace.duration_seconds:.2f}s")
    print(f"   Total turns: {len(trace.turns)}")
    print(f"   Messages exchanged: {len(trace.messages_exchanged)}")
    print(f"   Token usage: {trace.total_tokens}")
    
    print("\n" + "=" * 70)
    print("Example complete! In real usage, you would:")
    print("  1. Use OpenAIProvider or AnthropicProvider instead of MockProvider")
    print("  2. Generate task instances using TaskGenerator from the benchmark scaffold")
    print("  3. Run full execution until all agents submit")
    print("=" * 70)


if __name__ == "__main__":
    example_usage()