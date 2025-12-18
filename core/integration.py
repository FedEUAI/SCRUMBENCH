"""
MACB Complete Integration
=========================
This script demonstrates the full end-to-end pipeline:
1. Task generation (benchmark scaffold)
2. Agent execution (agent harness)
3. Evaluation and metrics (experiment runner)

Run with: python integration.py [--provider openai|anthropic|mock] [--tasks N]
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# =============================================================================
# In a real setup, you would import from your modules:
# 
# from benchmark import (
#     TaskGenerator, TaskInstance, EvaluationHarness, 
#     EvaluationResult, AgentSubmission, DifficultyLevel
# )
# from agent_harness import (
#     LLMProvider, OpenAIProvider, AnthropicProvider, MockProvider,
#     Agent, AgentConfig, MultiAgentHarness, HarnessConfig,
#     CommunicationPattern, ExecutionTrace, AgentTools
# )
# from experiment_runner import (
#     ExperimentConfig, ExperimentRunner, TaskResult, ExperimentResults
# )
#
# For this standalone example, we include minimal implementations inline.
# =============================================================================


# -----------------------------------------------------------------------------
# Minimal Inline Implementations (replace with actual imports)
# -----------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Optional, Any
from abc import ABC, abstractmethod
from enum import Enum
import subprocess
import time


class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class AgentTask:
    agent_id: str
    description: str
    files_to_modify: list[str]
    files_readonly: list[str]


@dataclass
class GeneratedFile:
    path: str
    content: str
    description: str = ""


@dataclass
class TaskInstance:
    instance_id: str
    template_name: str
    parameters: dict
    starter_files: list[GeneratedFile]
    agent_tasks: list[AgentTask]
    unit_tests: list[GeneratedFile]
    integration_tests: list[GeneratedFile]
    difficulty: DifficultyLevel
    seed: int
    
    def to_directory(self, base_path: Path) -> Path:
        task_dir = base_path / self.instance_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        for f in self.starter_files + self.unit_tests + self.integration_tests:
            file_path = task_dir / "starter" / f.path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(f.content)
        
        for task in self.agent_tasks:
            task_file = task_dir / "tasks" / f"{task.agent_id}.md"
            task_file.parent.mkdir(parents=True, exist_ok=True)
            task_file.write_text(task.description)
        
        metadata = {
            "instance_id": self.instance_id,
            "template_name": self.template_name,
            "parameters": self.parameters,
            "difficulty": self.difficulty.value,
            "seed": self.seed,
        }
        (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        
        return task_dir


# Minimal task generator
def generate_service_client_task(seed: int, difficulty: str = "medium") -> TaskInstance:
    """Generate a service+client integration task."""
    
    interface_code = '''"""Shared interface - DO NOT MODIFY."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class Item:
    id: str
    name: str
    value: float
    created_at: datetime

class StorageInterface(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Item]:
        """Get item by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, item: Item) -> None:
        """Store an item."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete item. Returns True if existed."""
        pass
    
    @abstractmethod
    def list_keys(self) -> list[str]:
        """List all keys."""
        pass
'''
    
    service_stub = '''"""Implement the Storage class here."""
from shared.interfaces import Item, StorageInterface
from typing import Optional

class Storage(StorageInterface):
    """TODO: Implement this class."""
    
    def __init__(self):
        pass
    
    def get(self, key: str) -> Optional[Item]:
        raise NotImplementedError
    
    def set(self, key: str, item: Item) -> None:
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        raise NotImplementedError
    
    def list_keys(self) -> list[str]:
        raise NotImplementedError
'''
    
    client_stub = '''"""Implement the CLI client here."""
import sys

def main():
    """CLI entry point."""
    print("Not implemented")
    sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    service_tests = '''"""Unit tests for Storage."""
import pytest
from datetime import datetime
from service.storage import Storage
from shared.interfaces import Item

class TestStorage:
    def test_set_and_get(self):
        s = Storage()
        item = Item(id="1", name="test", value=1.0, created_at=datetime.now())
        s.set("k1", item)
        result = s.get("k1")
        assert result is not None
        assert result.id == "1"
    
    def test_get_missing(self):
        s = Storage()
        assert s.get("missing") is None
    
    def test_delete(self):
        s = Storage()
        item = Item(id="1", name="test", value=1.0, created_at=datetime.now())
        s.set("k1", item)
        assert s.delete("k1") is True
        assert s.get("k1") is None
    
    def test_list_keys(self):
        s = Storage()
        item = Item(id="1", name="test", value=1.0, created_at=datetime.now())
        s.set("a", item)
        s.set("b", item)
        keys = s.list_keys()
        assert set(keys) == {"a", "b"}
'''
    
    integration_tests = '''"""Integration tests."""
import pytest
import subprocess
import sys

class TestIntegration:
    def test_set_then_get(self):
        # Set via CLI
        r = subprocess.run([sys.executable, "-m", "client.cli", "set", "test1", "TestItem", "99.5"],
                          capture_output=True, text=True, timeout=10)
        assert r.returncode == 0, f"Set failed: {r.stderr}"
        
        # Get via CLI
        r = subprocess.run([sys.executable, "-m", "client.cli", "get", "test1"],
                          capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        assert "TestItem" in r.stdout or "99.5" in r.stdout
    
    def test_list_keys(self):
        subprocess.run([sys.executable, "-m", "client.cli", "set", "k1", "A", "1"], capture_output=True, timeout=10)
        subprocess.run([sys.executable, "-m", "client.cli", "set", "k2", "B", "2"], capture_output=True, timeout=10)
        
        r = subprocess.run([sys.executable, "-m", "client.cli", "list"],
                          capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        assert "k1" in r.stdout and "k2" in r.stdout
'''
    
    service_task = """# Task: Implement Storage Service

## Overview
Implement the `Storage` class in `service/storage.py` that implements `StorageInterface`.

## Requirements
1. Store items in memory using a dict
2. Implement all interface methods as specified
3. All unit tests must pass

## Files
- **Modify**: `service/storage.py`
- **Read-only**: `shared/interfaces.py`
"""
    
    client_task = """# Task: Implement CLI Client

## Overview
Implement a CLI in `client/cli.py` that interfaces with the Storage service.

## Commands
- `python -m client.cli get <key>` - Print item or "Not found"
- `python -m client.cli set <key> <name> <value>` - Store item
- `python -m client.cli delete <key>` - Delete item  
- `python -m client.cli list` - List all keys

## CRITICAL: Data Persistence
Each CLI command runs as a SEPARATE PROCESS. This means in-memory storage 
will NOT persist between commands. You MUST save data to a file (e.g., JSON)
so that data persists across CLI invocations.

Suggested approach:
1. On startup, load existing data from a JSON file (e.g., `storage_data.json`)
2. After any modification, save data back to the file
3. Handle the case where the file doesn't exist yet

## Files
- **Modify**: `client/cli.py`
- **Read-only**: `shared/interfaces.py`

## Example
```bash
python -m client.cli set mykey "My Item" 42.0  # Saves to file
python -m client.cli get mykey                  # Loads from file, prints item
```
"""
    
    return TaskInstance(
        instance_id=f"service_client_{difficulty}_{seed}",
        template_name="service_client",
        parameters={"difficulty": difficulty},
        starter_files=[
            GeneratedFile("shared/__init__.py", ""),
            GeneratedFile("shared/interfaces.py", interface_code),
            GeneratedFile("service/__init__.py", ""),
            GeneratedFile("service/storage.py", service_stub),
            GeneratedFile("client/__init__.py", ""),
            GeneratedFile("client/cli.py", client_stub),
            GeneratedFile("requirements.txt", "pytest>=7.0.0\n"),
        ],
        agent_tasks=[
            AgentTask("agent_service", service_task, ["service/storage.py"], ["shared/interfaces.py"]),
            AgentTask("agent_client", client_task, ["client/cli.py"], ["shared/interfaces.py"]),
        ],
        unit_tests=[
            GeneratedFile("tests/__init__.py", ""),
            GeneratedFile("tests/test_storage.py", service_tests),
        ],
        integration_tests=[
            GeneratedFile("tests/test_integration.py", integration_tests),
        ],
        difficulty=DifficultyLevel(difficulty),
        seed=seed,
    )


# -----------------------------------------------------------------------------
# LLM Provider Implementations
# -----------------------------------------------------------------------------

class LLMProvider(ABC):
    name: str
    
    @abstractmethod
    def complete(self, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> dict:
        pass


class OpenAIProvider(LLMProvider):
    name = "openai"
    
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("pip install openai")
    
    def complete(self, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> dict:
        request = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        
        response = self.client.chat.completions.create(**request)
        
        msg = response.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                })
        
        return {
            "content": msg.content or "",
            "tool_calls": tool_calls,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        }


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("pip install anthropic")
    
    def complete(self, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> dict:
        # Extract system message
        system = None
        other_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                other_msgs.append(m)
        
        request = {
            "model": self.model,
            "messages": other_msgs,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [
                {"name": t["function"]["name"], 
                 "description": t["function"]["description"],
                 "input_schema": t["function"]["parameters"]}
                for t in tools
            ]
        
        response = self.client.messages.create(**request)
        
        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })
        
        return {
            "content": content,
            "tool_calls": tool_calls,
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
        }


class MockProvider(LLMProvider):
    """Mock provider that returns pre-scripted responses."""
    name = "mock"
    
    def __init__(self):
        self.call_count = 0
        self.responses = self._get_mock_responses()
    
    def _get_mock_responses(self) -> list[dict]:
        """Pre-scripted responses that complete the task."""
        return [
            # Agent Service: Read interface
            {"content": "", "tool_calls": [{"id": "1", "name": "read_file", "arguments": {"path": "shared/interfaces.py"}}]},
            # Agent Service: Write implementation
            {"content": "", "tool_calls": [{"id": "2", "name": "write_file", "arguments": {
                "path": "service/storage.py",
                "content": '''"""Storage implementation."""
from shared.interfaces import Item, StorageInterface
from typing import Optional

class Storage(StorageInterface):
    def __init__(self):
        self._data: dict[str, Item] = {}
    
    def get(self, key: str) -> Optional[Item]:
        return self._data.get(key)
    
    def set(self, key: str, item: Item) -> None:
        self._data[key] = item
    
    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False
    
    def list_keys(self) -> list[str]:
        return list(self._data.keys())
'''
            }}]},
            # Agent Service: Run tests
            {"content": "", "tool_calls": [{"id": "3", "name": "run_tests", "arguments": {"pattern": "test_storage"}}]},
            # Agent Service: Submit
            {"content": "", "tool_calls": [{"id": "4", "name": "submit_work", "arguments": {"summary": "Implemented Storage class"}}]},
            # Agent Client: Read interface
            {"content": "", "tool_calls": [{"id": "5", "name": "read_file", "arguments": {"path": "shared/interfaces.py"}}]},
            # Agent Client: Write CLI with file persistence
            {"content": "", "tool_calls": [{"id": "6", "name": "write_file", "arguments": {
                "path": "client/cli.py",
                "content": '''"""CLI client with file-based persistence."""
import sys
import json
from pathlib import Path
from datetime import datetime
from shared.interfaces import Item

DATA_FILE = Path(__file__).parent / "storage_data.json"

def load_data() -> dict:
    """Load data from JSON file."""
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE) as f:
                raw = json.load(f)
                return {k: Item(**v) for k, v in raw.items()}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}

def save_data(data: dict) -> None:
    """Save data to JSON file."""
    raw = {}
    for k, item in data.items():
        raw[k] = {
            "id": item.id,
            "name": item.name,
            "value": item.value,
            "created_at": item.created_at.isoformat() if hasattr(item.created_at, "isoformat") else str(item.created_at)
        }
    with open(DATA_FILE, "w") as f:
        json.dump(raw, f)

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m client.cli <command> [args]")
        print("Commands: get, set, delete, list")
        sys.exit(1)
    
    data = load_data()
    cmd = sys.argv[1]
    
    if cmd == "get":
        if len(sys.argv) < 3:
            print("Usage: get <key>")
            sys.exit(1)
        key = sys.argv[2]
        item = data.get(key)
        if item:
            print(f"{item.name}: {item.value}")
        else:
            print("Not found")
    
    elif cmd == "set":
        if len(sys.argv) < 5:
            print("Usage: set <key> <name> <value>")
            sys.exit(1)
        key, name, value = sys.argv[2], sys.argv[3], float(sys.argv[4])
        item = Item(id=key, name=name, value=value, created_at=datetime.now())
        data[key] = item
        save_data(data)
        print(f"Stored {key}")
    
    elif cmd == "delete":
        if len(sys.argv) < 3:
            print("Usage: delete <key>")
            sys.exit(1)
        key = sys.argv[2]
        if key in data:
            del data[key]
            save_data(data)
            print(f"Deleted {key}")
        else:
            print("Not found")
    
    elif cmd == "list":
        keys = list(data.keys())
        if keys:
            print("\\n".join(keys))
        else:
            print("(empty)")
    
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
            }}]},
            # Agent Client: Run integration tests
            {"content": "", "tool_calls": [{"id": "7", "name": "run_tests", "arguments": {"pattern": "test_integration"}}]},
            # Agent Client: Submit
            {"content": "", "tool_calls": [{"id": "8", "name": "submit_work", "arguments": {"summary": "Implemented CLI with file persistence"}}]},
        ]
    
    def complete(self, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> dict:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = {"content": "Done", "tool_calls": []}
        
        self.call_count += 1
        response["usage"] = {"prompt_tokens": 100, "completion_tokens": 50}
        return response


# -----------------------------------------------------------------------------
# Simple Agent Implementation
# -----------------------------------------------------------------------------

class SimpleAgent:
    """Minimal agent that can execute tools and complete tasks."""
    
    def __init__(self, agent_id: str, task: str, provider: LLMProvider, workspace: Path, verbose: bool = False):
        self.agent_id = agent_id
        self.provider = provider
        self.workspace = workspace
        self.submitted = False
        self.verbose = verbose
        self.tokens = {"prompt": 0, "completion": 0}
        self.step_count = 0
        
        self.messages = [
            {"role": "system", "content": """You are a software engineer. Complete your assigned task using the available tools.

IMPORTANT INSTRUCTIONS:
1. First, read the relevant files to understand the codebase
2. Implement the required functionality
3. Run tests to verify your implementation works
4. When ALL tests pass and your task is complete, use submit_work to finish

Available tools: read_file, write_file, run_tests, run_command, submit_work

Be methodical and test your work before submitting."""},
            {"role": "user", "content": task},
        ]
        
        self.tools = [
            {"type": "function", "function": {"name": "read_file", "description": "Read the contents of a file", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Path to the file relative to workspace root"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "write_file", "description": "Write content to a file (creates parent directories if needed)", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Path to the file"}, "content": {"type": "string", "description": "Content to write"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "run_tests", "description": "Run pytest tests. Returns test output.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "Optional test pattern to filter tests (e.g., 'test_storage')"}}, "required": []}}},
            {"type": "function", "function": {"name": "run_command", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Shell command to run"}}, "required": ["command"]}}},
            {"type": "function", "function": {"name": "submit_work", "description": "Submit your completed work. Only call this when your task is fully complete and tests pass.", "parameters": {"type": "object", "properties": {"summary": {"type": "string", "description": "Brief summary of what you implemented"}}, "required": ["summary"]}}},
        ]
    
    def step(self) -> bool:
        """Take one step. Returns True if submitted."""
        if self.submitted:
            return True
        
        self.step_count += 1
        
        response = self.provider.complete(self.messages, self.tools)
        self.tokens["prompt"] += response["usage"]["prompt_tokens"]
        self.tokens["completion"] += response["usage"]["completion_tokens"]
        
        if self.verbose:
            print(f"\n      [{self.agent_id}] Step {self.step_count}:")
            if response["content"]:
                print(f"        Response: {response['content'][:200]}...")
            if response["tool_calls"]:
                for tc in response["tool_calls"]:
                    args_preview = str(tc["arguments"])[:100]
                    print(f"        Tool: {tc['name']}({args_preview}...)")
        
        if response["tool_calls"]:
            # Format tool calls for OpenAI API format in message history
            formatted_tool_calls = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else tc["arguments"]
                    }
                }
                for tc in response["tool_calls"]
            ]
            
            # Add assistant message with tool calls
            self.messages.append({
                "role": "assistant",
                "content": response["content"] or None,
                "tool_calls": formatted_tool_calls
            })
            
            # Execute each tool and add results
            for tc in response["tool_calls"]:
                result = self._execute_tool(tc["name"], tc["arguments"])
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result
                })
                
                if self.verbose:
                    result_preview = result[:200] if len(result) > 200 else result
                    print(f"        Result: {result_preview}")
                
                if tc["name"] == "submit_work":
                    self.submitted = True
        
        elif response["content"]:
            # No tool calls, just a text response
            self.messages.append({"role": "assistant", "content": response["content"]})
        
        return self.submitted
    
    def _execute_tool(self, name: str, args: dict) -> str:
        try:
            if name == "read_file":
                content = (self.workspace / args["path"]).read_text()
                # Truncate very long files
                if len(content) > 8000:
                    return content[:8000] + f"\n... [truncated, {len(content)} total chars]"
                return content
            elif name == "write_file":
                path = self.workspace / args["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(args["content"])
                return f"✓ Wrote {len(args['content'])} bytes to {args['path']}"
            elif name == "run_tests":
                pattern = args.get("pattern", "")
                cmd = "python -m pytest tests/ -v --tb=short"
                if pattern:
                    cmd += f" -k {pattern}"
                r = subprocess.run(cmd, shell=True, cwd=self.workspace, capture_output=True, text=True, timeout=60)
                output = r.stdout + r.stderr
                # Summarize pytest output to save tokens
                return self._summarize_pytest_output(output)
            elif name == "run_command":
                r = subprocess.run(args["command"], shell=True, cwd=self.workspace, capture_output=True, text=True, timeout=60)
                output = r.stdout + r.stderr
                # Truncate long command outputs
                if len(output) > 3000:
                    return output[:3000] + f"\n... [truncated, {len(output)} total chars]"
                return output if output else "(no output)"
            elif name == "submit_work":
                return f"✓ SUBMITTED: {args.get('summary', '')}"
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            return f"Error: {e}"
    
    def _summarize_pytest_output(self, output: str) -> str:
        """Summarize pytest output to save tokens."""
        lines = output.strip().split('\n')
        
        # Extract key information
        summary_lines = []
        in_failures = False
        failure_lines = []
        
        for line in lines:
            # Always include pass/fail status lines
            if 'PASSED' in line or 'FAILED' in line or 'ERROR' in line:
                # Shorten the line
                summary_lines.append(line.split('::')[-1] if '::' in line else line)
            # Include the final summary
            elif 'passed' in line or 'failed' in line or 'error' in line:
                if '==' in line:
                    summary_lines.append(line)
            # Capture failure details (but limit)
            elif 'FAILURES' in line or in_failures:
                in_failures = True
                failure_lines.append(line)
                if len(failure_lines) > 30:  # Limit failure output
                    break
        
        result = '\n'.join(summary_lines)
        if failure_lines:
            result += '\n\nFailure details:\n' + '\n'.join(failure_lines[:30])
        
        return result if result else output[:2000]


# -----------------------------------------------------------------------------
# Run Pipeline
# -----------------------------------------------------------------------------

def run_task(task: TaskInstance, provider: LLMProvider, max_turns: int = 50, verbose: bool = False) -> dict:
    """Run a single task with agents and return results."""
    
    # Setup workspace
    work_dir = Path(tempfile.mkdtemp())
    task_dir = task.to_directory(work_dir)
    workspace = task_dir / "starter"
    
    if verbose:
        print(f"\n    [DEBUG] Workspace: {workspace}")
    
    # Init git
    subprocess.run(["git", "init"], cwd=workspace, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=workspace, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=workspace, capture_output=True,
                  env={**os.environ, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "t@t.com",
                       "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "t@t.com"})
    
    # Create agents - share provider instance for mock, create new for real providers
    agents = []
    for agent_task in task.agent_tasks:
        if isinstance(provider, MockProvider):
            agent_provider = provider
        else:
            # Create fresh provider instance for each agent
            agent_provider = type(provider)(provider.model)
        
        agent = SimpleAgent(
            agent_id=agent_task.agent_id,
            task=agent_task.description,
            provider=agent_provider,
            workspace=workspace,
            verbose=verbose,
        )
        agents.append(agent)
    
    # Run agents (round-robin)
    start = time.time()
    turn = 0
    while turn < max_turns:
        all_done = True
        for agent in agents:
            if not agent.submitted:
                all_done = False
                agent.step()
                turn += 1
                if turn >= max_turns:
                    if verbose:
                        print(f"\n    [DEBUG] Max turns ({max_turns}) reached!")
                    break
        if all_done:
            break
    
    duration = time.time() - start
    
    if verbose:
        print(f"\n    [DEBUG] Execution finished. Turns: {turn}, All submitted: {all(a.submitted for a in agents)}")
        print(f"    [DEBUG] Agent status: {[(a.agent_id, a.submitted) for a in agents]}")
        # List files in workspace
        print(f"    [DEBUG] Files in workspace:")
        for f in workspace.rglob("*"):
            if f.is_file() and ".git" not in str(f):
                print(f"      - {f.relative_to(workspace)}")
    
    # Run evaluation
    test_result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=workspace, capture_output=True, text=True, timeout=120
    )
    
    if verbose:
        print(f"    [DEBUG] Pytest output:\n{test_result.stdout[:2000]}")
        if test_result.stderr:
            print(f"    [DEBUG] Pytest stderr:\n{test_result.stderr[:1000]}")
    
    # Parse results
    import re
    passed = len(re.findall(r"PASSED", test_result.stdout))
    failed = len(re.findall(r"FAILED", test_result.stdout))
    errors = len(re.findall(r"ERROR", test_result.stdout))
    
    total_tokens = sum(a.tokens["prompt"] + a.tokens["completion"] for a in agents)
    all_submitted = all(a.submitted for a in agents)
    result_success = all_submitted and passed > 0 and failed == 0 and errors == 0
    
    # Don't cleanup if verbose or if failed (for debugging)
    if not verbose and result_success:
        shutil.rmtree(work_dir, ignore_errors=True)
    elif verbose:
        print(f"    [DEBUG] Workspace preserved at: {workspace}")
    else:
        # Failed - save diagnostic info then cleanup
        diagnostics = {
            "instance_id": task.instance_id,
            "pytest_stdout": test_result.stdout,
            "pytest_stderr": test_result.stderr,
            "files": {}
        }
        for fpath in ["service/storage.py", "client/cli.py"]:
            full_path = workspace / fpath
            if full_path.exists():
                diagnostics["files"][fpath] = full_path.read_text()
        
        diag_file = Path(f"./diagnostics_{task.instance_id}.json")
        diag_file.write_text(json.dumps(diagnostics, indent=2))
        shutil.rmtree(work_dir, ignore_errors=True)
    
    return {
        "instance_id": task.instance_id,
        "difficulty": task.difficulty.value,
        "success": result_success,
        "tests_passed": passed,
        "tests_failed": failed,
        "tests_errors": errors,
        "duration": duration,
        "turns": turn,
        "tokens": total_tokens,
        "agents_submitted": [a.submitted for a in agents],
    }


def main():
    parser = argparse.ArgumentParser(description="MACB Integration Example")
    parser.add_argument("--provider", choices=["openai", "anthropic", "mock"], default="mock")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--tasks", type=int, default=3, help="Number of tasks to run")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug output")
    parser.add_argument("--max-turns", type=int, default=50, help="Max turns per task")
    args = parser.parse_args()
    
    print("=" * 70)
    print("MACB - Multi-Agent Coding Benchmark")
    print("Complete Integration Example")
    print("=" * 70)
    
    # Create provider
    print(f"\n[1] Initializing {args.provider} provider...")
    if args.provider == "openai":
        provider = OpenAIProvider(model=args.model or "gpt-4o")
    elif args.provider == "anthropic":
        provider = AnthropicProvider(model=args.model or "claude-sonnet-4-20250514")
    else:
        provider = MockProvider()
    
    # Generate tasks
    print(f"\n[2] Generating {args.tasks} task instances...")
    tasks = [generate_service_client_task(seed=i, difficulty=args.difficulty) for i in range(args.tasks)]
    
    # Run tasks
    print(f"\n[3] Running tasks...")
    results = []
    for i, task in enumerate(tasks):
        print(f"  [{i+1}/{len(tasks)}] {task.instance_id}...", end="" if not args.verbose else "\n", flush=True)
        result = run_task(task, provider, max_turns=args.max_turns, verbose=args.verbose)
        results.append(result)
        status = "✓" if result["success"] else "✗"
        
        if not args.verbose:
            print(f" {status} (tests: {result['tests_passed']}/{result['tests_passed']+result['tests_failed']}, "
                  f"{result['duration']:.1f}s, {result['tokens']} tokens)")
        else:
            print(f"\n    RESULT: {status} (tests: {result['tests_passed']}/{result['tests_passed']+result['tests_failed']}, "
                  f"turns: {result['turns']}, {result['duration']:.1f}s, {result['tokens']} tokens)")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    successes = sum(1 for r in results if r["success"])
    total_tokens = sum(r["tokens"] for r in results)
    total_duration = sum(r["duration"] for r in results)
    
    print(f"Success rate: {successes}/{len(results)} ({100*successes/len(results):.1f}%)")
    print(f"Total duration: {total_duration:.1f}s")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Avg tokens/task: {total_tokens/len(results):,.0f}")
    
    print("\nPer-task results:")
    for r in results:
        status = "SUCCESS" if r["success"] else "FAILED"
        print(f"  {r['instance_id']}: {status} "
              f"(tests: {r['tests_passed']}/{r['tests_passed']+r['tests_failed']}, "
              f"turns: {r['turns']}, tokens: {r['tokens']})")
    
    print("\n" + "=" * 70)
    print("Integration complete!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()