"""
Multi-Agent Coding Benchmark (MACB)
===================================
A benchmark for evaluating multi-agent collaboration on software engineering tasks,
focusing on integration competence and conflict resolution.

Structure:
- core/: Core data structures and abstractions
- templates/: Task generation templates  
- domains/: Domain library for task variety
- evaluation/: Harness and metrics
"""

from __future__ import annotations
import os
import json
import subprocess
import tempfile
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum
from pathlib import Path
from random import Random
from datetime import datetime
import textwrap

# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

class ConflictType(Enum):
    TEXTUAL = "textual"          # Git-level merge conflict
    SEMANTIC = "semantic"        # No git conflict, but incompatible logic
    STRUCTURAL = "structural"    # Different architectural approaches

class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class KanbanStatus(Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"

@dataclass
class BacklogTask:
    """A sub-task within a task instance."""
    id: str
    title: str
    description: str
    difficulty: DifficultyLevel
    status: KanbanStatus = KanbanStatus.TODO
    assigned_to: Optional[str] = None

@dataclass
class AgentTask:
    """Task description for a single agent."""
    agent_id: str
    description: str
    files_to_modify: list[str]
    files_readonly: list[str]
    context: dict = field(default_factory=dict)

@dataclass
class ExpectedConflict:
    """Describes an expected conflict for evaluation."""
    conflict_type: ConflictType
    files_involved: list[str]
    description: str
    resolution_hints: list[str] = field(default_factory=list)

@dataclass 
class GeneratedFile:
    """A file generated as part of a task instance."""
    path: str
    content: str
    description: str = ""

@dataclass
class TaskInstance:
    """A complete generated task instance ready for execution."""
    instance_id: str
    template_name: str
    parameters: dict
    
    # Generated content
    starter_files: list[GeneratedFile]
    agent_tasks: list[AgentTask]
    unit_tests: list[GeneratedFile]
    integration_tests: list[GeneratedFile]
    
    
    # For evaluation
    expected_conflicts: list[ExpectedConflict]
    difficulty: DifficultyLevel
    
    # Metadata
    seed: int
    
    # ScrumBan specific
    backlog: list[BacklogTask] = field(default_factory=list)
    
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_directory(self, base_path: Path) -> Path:
        """Write this task instance to a directory structure."""
        task_dir = base_path / self.instance_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Write starter files
        for f in self.starter_files:
            file_path = task_dir / "starter" / f.path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(f.content)
        
        # Write tests
        for f in self.unit_tests + self.integration_tests:
            file_path = task_dir / "starter" / f.path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(f.content)
        
        # Write agent task descriptions
        for agent_task in self.agent_tasks:
            task_file = task_dir / "tasks" / f"{agent_task.agent_id}.md"
            task_file.parent.mkdir(parents=True, exist_ok=True)
            task_file.write_text(agent_task.description)
        
        # Write Kanban board (ScrumBan)
        kanban_data = [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "status": t.status.value,
                "difficulty": t.difficulty.value,
                "assigned_to": t.assigned_to
            }
            for t in self.backlog
        ]
        (task_dir / "starter" / "kanban.json").write_text(json.dumps(kanban_data, indent=2))
        
        # Write metadata
        metadata = {
            "instance_id": self.instance_id,
            "template_name": self.template_name,
            "parameters": self.parameters,
            "difficulty": self.difficulty.value,
            "seed": self.seed,
            "backlog_ids": [t.id for t in self.backlog],
            "expected_conflicts": [
                {"type": c.conflict_type.value, "files": c.files_involved, "description": c.description}
                for c in self.expected_conflicts
            ],
            "agent_ids": [t.agent_id for t in self.agent_tasks],
        }
        (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        
        return task_dir


# =============================================================================
# DOMAIN LIBRARY
# =============================================================================

@dataclass
class DomainSpec:
    """Specification for a problem domain."""
    name: str
    data_type_name: str
    fields: list[tuple[str, str, str]]  # (name, type, description)
    operations: list[dict]  # {name, params, returns, description}
    edge_cases: list[str]
    client_types: list[str]  # Types of clients that can consume this

DOMAINS: dict[str, DomainSpec] = {
    "user_cache": DomainSpec(
        name="user_cache",
        data_type_name="User",
        fields=[
            ("id", "str", "Unique identifier"),
            ("name", "str", "Display name"),
            ("email", "str", "Email address"),
            ("created_at", "datetime", "Creation timestamp"),
        ],
        operations=[
            {"name": "get", "params": [("key", "str")], "returns": "Optional[User]", 
             "description": "Retrieve user by key. Returns None if not found or expired."},
            {"name": "set", "params": [("key", "str"), ("user", "User"), ("ttl_seconds", "int", "300")], 
             "returns": "None", "description": "Store user with optional TTL."},
            {"name": "delete", "params": [("key", "str")], "returns": "bool",
             "description": "Delete key. Returns True if existed."},
            {"name": "clear", "params": [], "returns": "int",
             "description": "Clear all entries. Returns count deleted."},
        ],
        edge_cases=["key_not_found", "expired_entry", "invalid_key_format"],
        client_types=["cli", "rest_api", "async_worker"],
    ),
    "task_queue": DomainSpec(
        name="task_queue",
        data_type_name="Task",
        fields=[
            ("id", "str", "Unique task identifier"),
            ("payload", "dict", "Task data"),
            ("priority", "int", "Priority level (higher = more urgent)"),
            ("status", "str", "pending/processing/completed/failed"),
            ("created_at", "datetime", "Creation timestamp"),
        ],
        operations=[
            {"name": "enqueue", "params": [("task", "Task")], "returns": "str",
             "description": "Add task to queue. Returns task ID."},
            {"name": "dequeue", "params": [], "returns": "Optional[Task]",
             "description": "Get and remove highest priority task."},
            {"name": "peek", "params": [], "returns": "Optional[Task]",
             "description": "View highest priority task without removing."},
            {"name": "cancel", "params": [("task_id", "str")], "returns": "bool",
             "description": "Cancel a pending task."},
            {"name": "get_stats", "params": [], "returns": "QueueStats",
             "description": "Get queue statistics."},
        ],
        edge_cases=["empty_queue", "duplicate_task_id", "cancel_processing_task"],
        client_types=["cli", "worker_pool", "scheduler"],
    ),
    "rate_limiter": DomainSpec(
        name="rate_limiter",
        data_type_name="Request",
        fields=[
            ("client_id", "str", "Client identifier"),
            ("endpoint", "str", "API endpoint"),
            ("timestamp", "datetime", "Request timestamp"),
        ],
        operations=[
            {"name": "is_allowed", "params": [("client_id", "str"), ("endpoint", "str")], 
             "returns": "bool", "description": "Check if request is allowed."},
            {"name": "record", "params": [("request", "Request")], "returns": "None",
             "description": "Record a request."},
            {"name": "reset", "params": [("client_id", "str")], "returns": "None",
             "description": "Reset limits for a client."},
            {"name": "get_usage", "params": [("client_id", "str")], "returns": "UsageStats",
             "description": "Get current usage statistics."},
        ],
        edge_cases=["burst_traffic", "limit_boundary", "clock_skew"],
        client_types=["middleware", "cli", "monitoring_dashboard"],
    ),
    "orchestrator_service": DomainSpec(
        name="orchestrator_service",
        data_type_name="Instance",
        fields=[
            ("id", "str", "Instance ID"),
            ("service", "str", "Service name"),
            ("host", "str", "Host IP"),
            ("status", "str", "healthy/unhealthy/draining"),
            ("metadata", "dict", "Resource constraints"),
        ],
        operations=[
            {"name": "register", "params": [("instance", "Instance")], "returns": "bool",
             "description": "Register instance with health checks. Prevents duplicate IDs."},
            {"name": "get_healthy", "params": [("service", "str")], "returns": "list[Instance]",
             "description": "Return healthy instances using round-robin selection logic."},
            {"name": "update_status", "params": [("instance_id", "str"), ("status", "str")], "returns": "bool",
             "description": "Update status and handle 'draining' state by notifying dependencies."},
            {"name": "check_health", "params": [], "returns": "int",
             "description": "Scan all instances, mark as unhealthy if heartbeat > 30s. Returns count changed."},
            {"name": "trigger_failover", "params": [("service", "str")], "returns": "str",
             "description": "Elect a new 'primary' instance for the service from healthy pool."},
        ],
        edge_cases=["no_healthy_instances", "primary_election_tie", "draining_timeout", "stale_heartbeat"],
        client_types=["api_gateway", "load_balancer", "monitor"],
    ),
    "cryptographic_vault": DomainSpec(
        name="cryptographic_vault",
        data_type_name="Secret",
        fields=[
            ("id", "str", "Secret identifier"),
            ("value", "str", "Encrypted value"),
            ("owner", "str", "Owner ID"),
            ("policy", "dict", "Access control policy"),
            ("version", "int", "Secret version"),
        ],
        operations=[
            {"name": "store_secret", "params": [("secret", "Secret")], "returns": "bool",
             "description": "Store a new secret version. Requires policy validation."},
            {"name": "retrieve_secret", "params": [("id", "str"), ("requester", "str")], "returns": "Optional[str]",
             "description": "Decrypt and return secret if requester passes policy checks."},
            {"name": "rotate_keys", "params": [("id", "str")], "returns": "bool",
             "description": "Re-encrypt secret with new master key and increment version."},
            {"name": "update_policy", "params": [("id", "str"), ("new_policy", "dict")], "returns": "bool",
             "description": "Update access policy. Must prevent lock-outs."},
            {"name": "revoke_access", "params": [("requester", "str")], "returns": "int",
             "description": "Revoke access for a user across all secrets they own or access."},
        ],
        edge_cases=["unauthorized_access", "policy_lockout", "encryption_failure", "stale_version"],
        client_types=["auth_service", "database_driver", "cli"],
    ),
    "fintech_backend": DomainSpec(
        name="fintech_backend",
        data_type_name="Account",
        fields=[
            ("id", "str", "Account identifier"),
            ("balance", "float", "Current balance"),
            ("currency", "str", "ISO currency code"),
            ("status", "str", "active/frozen/closed"),
        ],
        operations=[
            {"name": "transfer", "params": [("from_id", "str"), ("to_id", "str"), ("amount", "float")], 
             "returns": "bool", "description": "Transfer funds between accounts with atomic safety and overdraft checks."},
            {"name": "get_balance", "params": [("account_id", "str")], "returns": "float",
             "description": "Get current available balance."},
            {"name": "freeze_account", "params": [("account_id", "str")], "returns": "None",
             "description": "Prevent any transactions on the account."},
            {"name": "get_transaction_history", "params": [("account_id", "str"), ("limit", "int", "10")],
             "returns": "list[dict]", "description": "Retrieve recent transactions for the account."},
        ],
        edge_cases=["insufficient_funds", "account_frozen", "currency_mismatch", "invalid_amount"],
        client_types=["mobile_app", "web_portal", "admin_tool"],
    ),
    "ecommerce_platform": DomainSpec(
        name="ecommerce_platform",
        data_type_name="Order",
        fields=[
            ("id", "str", "Order ID"),
            ("items", "list[dict]", "List of items in order"),
            ("total", "float", "Total order value"),
            ("status", "str", "pending/paid/shipped/cancelled"),
        ],
        operations=[
            {"name": "place_order", "params": [("order", "Order")], "returns": "str",
             "description": "Place a new order. Must validate stock and calculate tax/shipping."},
            {"name": "process_payment", "params": [("order_id", "str"), ("payment_token", "str")],
             "returns": "bool", "description": "Process payment. Updates status to 'paid' on success."},
            {"name": "cancel_order", "params": [("order_id", "str")], "returns": "bool",
             "description": "Cancel order and release reserved stock if possible."},
            {"name": "get_order_status", "params": [("order_id", "str")], "returns": "str",
             "description": "Get current status and tracking info."},
        ],
        edge_cases=["out_of_stock", "payment_failed", "already_shipped", "invalid_payment_token"],
        client_types=["checkout_page", "erp_system", "warehouse_app"],
    ),
    "document_store": DomainSpec(
        name="document_store",
        data_type_name="Document",
        fields=[
            ("id", "str", "Document identifier"),
            ("content", "str", "Document content"),
            ("metadata", "dict", "Document metadata"),
            ("version", "int", "Version number"),
            ("updated_at", "datetime", "Last update timestamp"),
        ],
        operations=[
            {"name": "create", "params": [("doc", "Document")], "returns": "str",
             "description": "Create new document. Returns ID."},
            {"name": "read", "params": [("doc_id", "str"), ("version", "Optional[int]", "None")],
             "returns": "Optional[Document]", "description": "Read document, optionally specific version."},
            {"name": "update", "params": [("doc_id", "str"), ("content", "str")], "returns": "Document",
             "description": "Update document content. Increments version."},
            {"name": "delete", "params": [("doc_id", "str")], "returns": "bool",
             "description": "Delete document."},
            {"name": "search", "params": [("query", "str")], "returns": "list[Document]",
             "description": "Search documents by content."},
        ],
        edge_cases=["version_conflict", "not_found", "concurrent_update"],
        client_types=["cli", "rest_api", "sync_client"],
    ),
}


# =============================================================================
# BASE TEMPLATE CLASS
# =============================================================================

class TaskTemplate(ABC):
    """Abstract base class for task generation templates."""
    
    name: str
    description: str
    
    @abstractmethod
    def generate(self, params: dict, seed: int) -> TaskInstance:
        """Generate a complete task instance from parameters."""
        pass
    
    @abstractmethod
    def get_default_params(self) -> dict:
        """Return default parameters for this template."""
        pass
    
    @abstractmethod
    def get_param_schema(self) -> dict:
        """Return JSON schema for valid parameters."""
        pass


# =============================================================================
# TEMPLATE: SERVICE + CLIENT (Integration Competence)
# =============================================================================

class ServiceClientTemplate(TaskTemplate):
    """
    Template for testing integration competence.
    
    Agent 1: Implements a service conforming to an interface
    Agent 2: Implements a client that consumes the service
    
    Integration is tested by verifying the client correctly uses the service.
    """
    
    name = "service_client"
    description = "One agent implements a service, another implements its client"
    
    def get_default_params(self) -> dict:
        return {
            "domain": "user_cache",
            "client_type": "cli",
            "difficulty": "medium",
            "num_operations": 4,
            "include_edge_cases": True,
        }
    
    def get_param_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "enum": list(DOMAINS.keys())},
                "client_type": {"type": "string"},
                "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                "num_operations": {"type": "integer", "minimum": 2, "maximum": 8},
                "include_edge_cases": {"type": "boolean"},
            }
        }
    
    def generate(self, params: dict, seed: int) -> TaskInstance:
        rng = Random(seed)
        p = {**self.get_default_params(), **params}
        
        domain = DOMAINS[p["domain"]]
        difficulty = DifficultyLevel(p["difficulty"])
        
        # Select operations based on difficulty
        num_ops = p["num_operations"]
        if difficulty == DifficultyLevel.EASY:
            num_ops = min(num_ops, 3)
        operations = domain.operations[:num_ops]
        
        # Generate the interface file
        interface_content = self._generate_interface(domain, operations)
        
        # Generate starter code structure
        starter_files = [
            GeneratedFile(
                path="shared/interfaces.py",
                content=interface_content,
                description="Shared interface definitions"
            ),
            GeneratedFile(
                path="shared/__init__.py",
                content='"""Shared components."""\nfrom .interfaces import *\n',
                description="Package init"
            ),
            GeneratedFile(
                path="service/__init__.py",
                content='"""Service implementation package."""\n',
                description="Package init"
            ),
            GeneratedFile(
                path="service/implementation.py",
                content=self._generate_service_stub(domain),
                description="Service implementation stub"
            ),
            GeneratedFile(
                path="client/__init__.py",
                content='"""Client implementation package."""\n',
                description="Package init"
            ),
            GeneratedFile(
                path="client/main.py",
                content=self._generate_client_stub(domain, p["client_type"]),
                description="Client implementation stub"
            ),
            GeneratedFile(
                path="requirements.txt",
                content="pytest>=7.0.0\npytest-timeout>=2.0.0\n",
                description="Dependencies"
            ),
        ]
        
        # Generate agent tasks
        agent_tasks = [
            AgentTask(
                agent_id="agent_service",
                description=self._generate_service_task_description(domain, operations, difficulty),
                files_to_modify=["service/implementation.py"],
                files_readonly=["shared/interfaces.py"],
            ),
            AgentTask(
                agent_id="agent_client",
                description=self._generate_client_task_description(domain, operations, p["client_type"], difficulty),
                files_to_modify=["client/main.py"],
                files_readonly=["shared/interfaces.py"],
            ),
        ]
        
        # Generate tests
        unit_tests = [
            GeneratedFile(
                path="tests/__init__.py",
                content="",
                description="Test package"
            ),
            GeneratedFile(
                path="tests/test_service.py",
                content=self._generate_service_unit_tests(domain, operations),
                description="Unit tests for service"
            ),
            GeneratedFile(
                path="tests/test_client.py",
                content=self._generate_client_unit_tests(domain, operations, p["client_type"]),
                description="Unit tests for client"
            ),
        ]
        
        integration_tests = [
            GeneratedFile(
                path="tests/test_integration.py",
                content=self._generate_integration_tests(domain, operations, p["client_type"]),
                description="Integration tests"
            ),
        ]
        
        instance_id = f"{self.name}_{p['domain']}_{seed}"
        
        return TaskInstance(
            instance_id=instance_id,
            template_name=self.name,
            parameters=p,
            starter_files=starter_files,
            agent_tasks=agent_tasks,
            unit_tests=unit_tests,
            integration_tests=integration_tests,
            expected_conflicts=[],  # No conflicts expected in this template
            difficulty=difficulty,
            seed=seed,
        )
    
    def _generate_interface(self, domain: DomainSpec, operations: list[dict]) -> str:
        # Generate dataclass for the domain type
        fields_code = "\n    ".join(
            f"{name}: {typ}" for name, typ, _ in domain.fields
        )
        
        # Generate abstract methods
        methods_code = ""
        for op in operations:
            params_str = ", ".join(
                f"{p[0]}: {p[1]}" + (f" = {p[2]}" if len(p) > 2 else "")
                for p in op["params"]
            )
            methods_code += f'''
    @abstractmethod
    def {op["name"]}(self, {params_str}) -> {op["returns"]}:
        """{op["description"]}"""
        pass
'''
        
        return f'''"""
Shared interfaces for {domain.name}.
DO NOT MODIFY THIS FILE.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class {domain.data_type_name}:
    """Core data type for {domain.name}."""
    {fields_code}


class {domain.data_type_name}ServiceInterface(ABC):
    """Abstract interface for {domain.name} service."""
{methods_code}
'''
    
    def _generate_service_stub(self, domain: DomainSpec) -> str:
        return f'''"""
Service implementation for {domain.name}.
Implement the {domain.data_type_name}Service class here.
"""
from shared.interfaces import {domain.data_type_name}, {domain.data_type_name}ServiceInterface


class {domain.data_type_name}Service({domain.data_type_name}ServiceInterface):
    """
    TODO: Implement this class.
    Must implement all methods from {domain.data_type_name}ServiceInterface.
    """
    
    def __init__(self):
        # TODO: Initialize your data structures
        pass
'''
    
    def _generate_client_stub(self, domain: DomainSpec, client_type: str) -> str:
        if client_type == "cli":
            return f'''"""
CLI client for {domain.name}.
Implement command-line interface here.
"""
import sys
from service.implementation import {domain.data_type_name}Service


def main():
    """Main entry point for CLI."""
    # TODO: Implement CLI
    # Parse sys.argv and call appropriate service methods
    print("Not implemented")
    sys.exit(1)


if __name__ == "__main__":
    main()
'''
        else:
            return f'''"""
Client for {domain.name}.
Implement client here.
"""
from service.implementation import {domain.data_type_name}Service


# TODO: Implement client
'''
    
    def _generate_service_task_description(self, domain: DomainSpec, operations: list[dict], difficulty: DifficultyLevel) -> str:
        ops_list = "\n".join(f"- `{op['name']}()`: {op['description']}" for op in operations)
        
        edge_case_note = ""
        if difficulty in [DifficultyLevel.MEDIUM, DifficultyLevel.HARD]:
            edge_case_note = f"""
## Edge Cases to Handle
Your implementation should handle these edge cases gracefully:
{chr(10).join(f'- {ec}' for ec in domain.edge_cases)}
"""
        
        return f'''# Task: Implement {domain.data_type_name}Service

## Overview
Implement the `{domain.data_type_name}Service` class in `service/implementation.py` that 
implements the `{domain.data_type_name}ServiceInterface` from `shared/interfaces.py`.

## Required Operations
{ops_list}

## Requirements
1. Implement all methods from the interface
2. Store data in memory (no external databases)
3. Follow the exact method signatures from the interface
4. All existing tests must pass

{edge_case_note}

## Files
- **Modify**: `service/implementation.py`
- **Read-only**: `shared/interfaces.py` (do not modify)

## Important
Your implementation will be used by another component. Adhere strictly to the 
interface contract - do not change method signatures or return types.
'''
    
    def _generate_client_task_description(self, domain: DomainSpec, operations: list[dict], 
                                          client_type: str, difficulty: DifficultyLevel) -> str:
        if client_type == "cli":
            commands = "\n".join(
                f"- `{domain.name} {op['name']} {' '.join(f'<{p[0]}>' for p in op['params'])}` - {op['description']}"
                for op in operations
            )
            
            return f'''# Task: Implement CLI Client

## Overview
Implement a command-line interface in `client/main.py` that provides access to 
the {domain.data_type_name}Service.

## Commands to Implement
{commands}

## Requirements
1. Parse command-line arguments from `sys.argv`
2. Import and use `{domain.data_type_name}Service` from `service.implementation`
3. Print results to stdout in a clear format
4. Exit with code 0 on success, 1 on error
5. Handle invalid commands gracefully with usage information

## Example Usage
```bash
python -m client.main get user123
python -m client.main set user123 "Alice" "alice@example.com"
```

## Files
- **Modify**: `client/main.py`
- **Read-only**: `shared/interfaces.py` (do not modify)

## Important
Assume the service implementation is correct and follows the interface.
Import it as: `from service.implementation import {domain.data_type_name}Service`
'''
        else:
            return f"# Task: Implement {client_type} client\n\nTODO: Generate for {client_type}"
    
    def _generate_service_unit_tests(self, domain: DomainSpec, operations: list[dict]) -> str:
        test_methods = ""
        
        for op in operations:
            if op["name"] == "get":
                test_methods += f'''
    def test_get_nonexistent_returns_none(self):
        """Get on nonexistent key returns None."""
        service = {domain.data_type_name}Service()
        result = service.get("nonexistent")
        assert result is None
'''
            elif op["name"] == "set":
                test_methods += f'''
    def test_set_then_get(self):
        """Set followed by get returns the item."""
        service = {domain.data_type_name}Service()
        item = {domain.data_type_name}(id="1", name="Test", email="test@example.com", created_at=datetime.now())
        service.set("key1", item)
        result = service.get("key1")
        assert result is not None
        assert result.id == "1"
        assert result.name == "Test"
'''
            elif op["name"] == "delete":
                test_methods += f'''
    def test_delete_existing(self):
        """Delete existing key returns True."""
        service = {domain.data_type_name}Service()
        item = {domain.data_type_name}(id="1", name="Test", email="test@example.com", created_at=datetime.now())
        service.set("key1", item)
        result = service.delete("key1")
        assert result is True
        assert service.get("key1") is None
    
    def test_delete_nonexistent(self):
        """Delete nonexistent key returns False."""
        service = {domain.data_type_name}Service()
        result = service.delete("nonexistent")
        assert result is False
'''
            elif op["name"] == "clear":
                test_methods += f'''
    def test_clear_returns_count(self):
        """Clear returns count of deleted entries."""
        service = {domain.data_type_name}Service()
        item1 = {domain.data_type_name}(id="1", name="Test1", email="t1@example.com", created_at=datetime.now())
        item2 = {domain.data_type_name}(id="2", name="Test2", email="t2@example.com", created_at=datetime.now())
        service.set("k1", item1)
        service.set("k2", item2)
        count = service.clear()
        assert count == 2
'''
        
        return f'''"""Unit tests for {domain.data_type_name}Service."""
import pytest
from datetime import datetime
from service.implementation import {domain.data_type_name}Service
from shared.interfaces import {domain.data_type_name}


class TestService:
    """Test cases for service implementation."""
{test_methods}
'''
    
    def _generate_client_unit_tests(self, domain: DomainSpec, operations: list[dict], client_type: str) -> str:
        if client_type == "cli":
            return f'''"""Unit tests for CLI client."""
import pytest
import subprocess
import sys


class TestCLI:
    """Test CLI argument parsing and output format."""
    
    def test_no_args_shows_usage(self):
        """Running with no args shows usage information."""
        result = subprocess.run(
            [sys.executable, "-m", "client.main"],
            capture_output=True,
            text=True
        )
        # Should either show usage or exit with error
        assert result.returncode != 0 or "usage" in result.stdout.lower() or "usage" in result.stderr.lower()
    
    def test_invalid_command_shows_error(self):
        """Invalid command shows error message."""
        result = subprocess.run(
            [sys.executable, "-m", "client.main", "invalid_command"],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
'''
        return f'"""Unit tests for {client_type} client."""\n# TODO: Generate tests\n'
    
    def _generate_integration_tests(self, domain: DomainSpec, operations: list[dict], client_type: str) -> str:
        if client_type == "cli":
            return f'''"""Integration tests for {domain.name}."""
import pytest
import subprocess
import sys
import time


class TestIntegration:
    """Integration tests verifying client correctly uses service."""
    
    def test_set_then_get_flow(self):
        """Full flow: set an item via CLI, then retrieve it."""
        # Set
        set_result = subprocess.run(
            [sys.executable, "-m", "client.main", "set", "testkey", "Alice", "alice@test.com"],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert set_result.returncode == 0, f"Set failed: {{set_result.stderr}}"
        
        # Get
        get_result = subprocess.run(
            [sys.executable, "-m", "client.main", "get", "testkey"],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert get_result.returncode == 0, f"Get failed: {{get_result.stderr}}"
        assert "Alice" in get_result.stdout or "alice" in get_result.stdout.lower()
    
    def test_get_nonexistent_key(self):
        """Getting a nonexistent key should indicate not found."""
        result = subprocess.run(
            [sys.executable, "-m", "client.main", "get", "definitely_not_a_key_12345"],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Should either return non-zero or output indicates not found
        output = (result.stdout + result.stderr).lower()
        assert result.returncode != 0 or "not found" in output or "none" in output or "error" in output
    
    def test_delete_then_get(self):
        """Delete should remove item so get returns not found."""
        # First set
        subprocess.run(
            [sys.executable, "-m", "client.main", "set", "delkey", "Bob", "bob@test.com"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Delete
        del_result = subprocess.run(
            [sys.executable, "-m", "client.main", "delete", "delkey"],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert del_result.returncode == 0
        
        # Get should fail
        get_result = subprocess.run(
            [sys.executable, "-m", "client.main", "get", "delkey"],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = (get_result.stdout + get_result.stderr).lower()
        assert "not found" in output or "none" in output or get_result.returncode != 0
    
    def test_clear_removes_all(self):
        """Clear should remove all items."""
        # Set multiple
        for i in range(3):
            subprocess.run(
                [sys.executable, "-m", "client.main", "set", f"clearkey{{i}}", f"User{{i}}", f"u{{i}}@test.com"],
                capture_output=True,
                timeout=10
            )
        
        # Clear
        clear_result = subprocess.run(
            [sys.executable, "-m", "client.main", "clear"],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert clear_result.returncode == 0
        
        # Verify at least one is gone
        get_result = subprocess.run(
            [sys.executable, "-m", "client.main", "get", "clearkey0"],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = (get_result.stdout + get_result.stderr).lower()
        assert "not found" in output or "none" in output or get_result.returncode != 0
'''
        return f'"""Integration tests for {client_type}."""\n# TODO: Generate tests\n'


# =============================================================================
# TEMPLATE: REFACTOR COLLISION (Conflict Resolution)
# =============================================================================

class RefactorCollisionTemplate(TaskTemplate):
    """
    Template for testing conflict resolution.
    
    Agent 1: Refactors a module for code quality
    Agent 2: Adds a feature to the same module
    
    Guaranteed to produce merge conflicts that require resolution.
    """
    
    name = "refactor_collision"
    description = "One agent refactors while another adds features, causing conflicts"
    
    def get_default_params(self) -> dict:
        return {
            "module_type": "report_generator",
            "refactor_type": "extract_class",
            "feature_type": "add_format",
            "difficulty": "medium",
            "conflict_type": "structural",
        }
    
    def get_param_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "module_type": {"type": "string", "enum": ["report_generator", "data_processor", "api_handler"]},
                "refactor_type": {"type": "string", "enum": ["extract_class", "extract_function", "rename"]},
                "feature_type": {"type": "string", "enum": ["add_format", "add_filter", "add_export"]},
                "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                "conflict_type": {"type": "string", "enum": ["textual", "semantic", "structural"]},
            }
        }
    
    def generate(self, params: dict, seed: int) -> TaskInstance:
        rng = Random(seed)
        p = {**self.get_default_params(), **params}
        difficulty = DifficultyLevel(p["difficulty"])
        
        # Generate the messy starter code
        messy_code = self._generate_messy_code(p["module_type"], difficulty)
        
        starter_files = [
            GeneratedFile(
                path="reports/__init__.py",
                content='"""Report generation module."""\n',
                description="Package init"
            ),
            GeneratedFile(
                path="reports/generator.py",
                content=messy_code,
                description="Report generator (needs refactoring)"
            ),
            GeneratedFile(
                path="requirements.txt",
                content="pytest>=7.0.0\n",
                description="Dependencies"
            ),
        ]
        
        # Agent tasks
        agent_tasks = [
            AgentTask(
                agent_id="agent_refactor",
                description=self._generate_refactor_task(p["refactor_type"], difficulty),
                files_to_modify=["reports/generator.py"],
                files_readonly=[],
            ),
            AgentTask(
                agent_id="agent_feature",
                description=self._generate_feature_task(p["feature_type"], difficulty),
                files_to_modify=["reports/generator.py"],
                files_readonly=[],
            ),
        ]
        
        # Tests
        unit_tests = [
            GeneratedFile(
                path="tests/__init__.py",
                content="",
                description="Test package"
            ),
            GeneratedFile(
                path="tests/test_generator.py",
                content=self._generate_existing_tests(),
                description="Existing tests (must stay passing)"
            ),
        ]
        
        integration_tests = [
            GeneratedFile(
                path="tests/test_new_features.py",
                content=self._generate_feature_tests(p["feature_type"]),
                description="Tests for new features"
            ),
        ]
        
        # Expected conflicts
        expected_conflicts = [
            ExpectedConflict(
                conflict_type=ConflictType(p["conflict_type"]),
                files_involved=["reports/generator.py"],
                description=f"Both agents modify generator.py: refactoring vs feature addition",
                resolution_hints=[
                    "Agents need to communicate about structural changes",
                    "Feature additions must be integrated into refactored structure",
                ],
            ),
        ]
        
        instance_id = f"{self.name}_{p['module_type']}_{seed}"
        
        return TaskInstance(
            instance_id=instance_id,
            template_name=self.name,
            parameters=p,
            starter_files=starter_files,
            agent_tasks=agent_tasks,
            unit_tests=unit_tests,
            integration_tests=integration_tests,
            expected_conflicts=expected_conflicts,
            difficulty=difficulty,
            seed=seed,
        )
    
    def _generate_messy_code(self, module_type: str, difficulty: DifficultyLevel) -> str:
        """Generate deliberately messy code that needs refactoring."""
        return '''"""
Report generator module.
NOTE: This code works but needs refactoring for maintainability.
"""
from datetime import datetime
from typing import Any


def generate_report(
    data: list[dict],
    format_type: str = "markdown",
    include_header: bool = True,
    include_footer: bool = True,
    title: str = "Report",
    author: str | None = None,
    date: str | None = None,
    columns: list[str] | None = None,
    sort_by: str | None = None,
    filter_fn: Any | None = None,
    max_rows: int | None = None,
) -> str:
    """
    Generate a report from data.
    
    This function handles too many responsibilities and should be refactored,
    but all existing functionality must be preserved.
    """
    result = ""
    
    # === HEADER SECTION ===
    # This section handles header generation with various options
    if include_header:
        if format_type == "markdown":
            if title:
                result += f"# {title}\\n"
            if author:
                result += f"**Author:** {author}\\n"
            if date:
                result += f"**Date:** {date}\\n"
            else:
                result += f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\\n"
            result += "\\n"
        elif format_type == "csv":
            # CSV doesn't have headers in the same way
            if title:
                result += f"# {title}\\n"
        elif format_type == "text":
            if title:
                result += f"{title}\\n"
                result += "=" * len(title) + "\\n"
            if author:
                result += f"Author: {author}\\n"
            if date:
                result += f"Date: {date}\\n"
            else:
                result += f"Date: {datetime.now().strftime('%Y-%m-%d')}\\n"
            result += "\\n"
    
    # === DATA PROCESSING SECTION ===
    # Filter, sort, and limit the data
    processed = data.copy() if data else []
    
    if filter_fn is not None:
        processed = [row for row in processed if filter_fn(row)]
    
    if sort_by is not None and processed:
        try:
            processed = sorted(processed, key=lambda x: x.get(sort_by, ""))
        except (TypeError, KeyError):
            pass  # Keep original order if sort fails
    
    if max_rows is not None:
        processed = processed[:max_rows]
    
    # === FORMATTING SECTION ===
    # Format the data based on format_type
    if columns is None and processed:
        columns = list(processed[0].keys())
    
    if format_type == "markdown":
        if columns:
            result += "| " + " | ".join(str(c) for c in columns) + " |\\n"
            result += "| " + " | ".join(["---"] * len(columns)) + " |\\n"
        for row in processed:
            if columns:
                values = [str(row.get(c, "")) for c in columns]
                result += "| " + " | ".join(values) + " |\\n"
    
    elif format_type == "csv":
        if columns:
            result += ",".join(str(c) for c in columns) + "\\n"
        for row in processed:
            if columns:
                values = [str(row.get(c, "")).replace(",", ";") for c in columns]
                result += ",".join(values) + "\\n"
    
    elif format_type == "text":
        if columns:
            col_widths = {c: max(len(str(c)), max((len(str(row.get(c, ""))) for row in processed), default=0)) for c in columns}
            header_line = "  ".join(str(c).ljust(col_widths[c]) for c in columns)
            result += header_line + "\\n"
            result += "-" * len(header_line) + "\\n"
        for row in processed:
            if columns:
                values = [str(row.get(c, "")).ljust(col_widths[c]) for c in columns]
                result += "  ".join(values) + "\\n"
    
    # === FOOTER SECTION ===
    if include_footer:
        if format_type == "markdown":
            result += f"\\n---\\n"
            result += f"*Generated by ReportGen v1.0*\\n"
            result += f"*Total rows: {len(processed)}*\\n"
        elif format_type == "text":
            result += f"\\n---\\n"
            result += f"Generated by ReportGen v1.0\\n"
            result += f"Total rows: {len(processed)}\\n"
        # CSV typically doesn't have footers
    
    return result


def validate_data(data: list[dict]) -> tuple[bool, str]:
    """Validate input data. Returns (is_valid, error_message)."""
    if not isinstance(data, list):
        return False, "Data must be a list"
    if not data:
        return True, ""  # Empty is valid
    if not all(isinstance(row, dict) for row in data):
        return False, "All rows must be dictionaries"
    return True, ""
'''
    
    def _generate_refactor_task(self, refactor_type: str, difficulty: DifficultyLevel) -> str:
        return f'''# Task: Refactor Report Generator

## Overview
The `generate_report()` function in `reports/generator.py` is too long and handles 
too many responsibilities. Refactor it to improve maintainability.

## Specific Refactoring Required
- Extract the formatting logic into a separate `ReportFormatter` class
- Each format type (markdown, csv, text) should have its own method
- Extract header and footer generation into separate methods
- Keep the main `generate_report()` function as the public API

## Requirements
1. **All existing tests must continue to pass**
2. The public function signature must NOT change
3. The output for all format types must remain identical
4. Improve code organization without changing behavior

## Suggested Structure
```python
class ReportFormatter:
    def format_header(self, ...) -> str: ...
    def format_data(self, ...) -> str: ...
    def format_footer(self, ...) -> str: ...

class MarkdownFormatter(ReportFormatter): ...
class CSVFormatter(ReportFormatter): ...
class TextFormatter(ReportFormatter): ...

def generate_report(...) -> str:
    formatter = get_formatter(format_type)
    return formatter.format(...)
```

## Files
- **Modify**: `reports/generator.py`
- You may create additional files in `reports/` if needed

## Important
This is a refactoring task. Do NOT add new features. The behavior must remain 
exactly the same - only the code organization should improve.
'''
    
    def _generate_feature_task(self, feature_type: str, difficulty: DifficultyLevel) -> str:
        return '''# Task: Add JSON Export Format

## Overview
Add JSON export support to the report generator in `reports/generator.py`.

## Requirements
1. Add `"json"` as a valid `format_type` option
2. JSON output should be structured as:
   ```json
   {
     "meta": {
       "title": "...",
       "author": "...",
       "date": "...",
       "generated_at": "...",
       "row_count": N
     },
     "data": [
       {"col1": "val1", ...},
       ...
     ]
   }
   ```
3. If `include_header` is False, omit the "meta" section
4. If `include_footer` is False, omit "generated_at" and "row_count" from meta
5. Add a new optional parameter `pretty_print: bool = False` that when True, 
   outputs indented JSON

## Additional Enhancement
Also add a `delimiter` parameter to CSV format (default `","`) so users can 
create TSV or other delimited files.

## Files
- **Modify**: `reports/generator.py`

## Important
- All existing tests must continue to pass
- The existing behavior for markdown, csv, and text formats must not change
- Only ADD functionality, do not remove or modify existing features
'''
    
    def _generate_existing_tests(self) -> str:
        return '''"""Existing tests for report generator. ALL MUST CONTINUE TO PASS."""
import pytest
from reports.generator import generate_report, validate_data


class TestGenerateReport:
    """Tests for generate_report function."""
    
    def test_empty_data(self):
        """Empty data produces minimal report."""
        result = generate_report([])
        assert "Report" in result  # Has default title
    
    def test_markdown_format(self):
        """Markdown format produces valid markdown table."""
        data = [{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]
        result = generate_report(data, format_type="markdown")
        assert "| name | age |" in result or "| age | name |" in result
        assert "| --- |" in result
        assert "Alice" in result
        assert "Bob" in result
    
    def test_csv_format(self):
        """CSV format produces valid CSV."""
        data = [{"name": "Alice", "age": "30"}]
        result = generate_report(data, format_type="csv", include_header=False, include_footer=False)
        assert "name" in result
        assert "Alice" in result
        assert "," in result
    
    def test_text_format(self):
        """Text format produces readable text."""
        data = [{"name": "Alice", "age": "30"}]
        result = generate_report(data, format_type="text")
        assert "Alice" in result
        assert "30" in result
    
    def test_custom_title(self):
        """Custom title appears in output."""
        result = generate_report([], title="My Custom Report")
        assert "My Custom Report" in result
    
    def test_author_included(self):
        """Author appears when provided."""
        result = generate_report([], author="John Doe")
        assert "John Doe" in result
    
    def test_no_header(self):
        """No header when include_header=False."""
        result = generate_report([], include_header=False, title="Should Not Appear")
        # Title should not be prominently displayed
        lines = result.strip().split("\\n")
        if lines and lines[0]:
            assert not lines[0].startswith("# Should Not Appear")
    
    def test_sort_by(self):
        """Data is sorted when sort_by provided."""
        data = [{"name": "Charlie"}, {"name": "Alice"}, {"name": "Bob"}]
        result = generate_report(data, format_type="text", sort_by="name", 
                                 include_header=False, include_footer=False)
        alice_pos = result.find("Alice")
        bob_pos = result.find("Bob")
        charlie_pos = result.find("Charlie")
        assert alice_pos < bob_pos < charlie_pos
    
    def test_max_rows(self):
        """Only max_rows rows included."""
        data = [{"id": str(i)} for i in range(10)]
        result = generate_report(data, format_type="csv", max_rows=3,
                                 include_header=False, include_footer=False)
        # Should have header + 3 data rows
        lines = [l for l in result.strip().split("\\n") if l]
        assert len(lines) == 4  # header + 3 rows
    
    def test_filter_function(self):
        """Filter function removes non-matching rows."""
        data = [{"val": 1}, {"val": 2}, {"val": 3}]
        result = generate_report(data, format_type="csv", 
                                 filter_fn=lambda x: x["val"] > 1,
                                 include_header=False, include_footer=False)
        assert "1" not in result.split("\\n")[1] if len(result.split("\\n")) > 1 else True


class TestValidateData:
    """Tests for validate_data function."""
    
    def test_valid_list_of_dicts(self):
        """Valid data passes validation."""
        is_valid, msg = validate_data([{"a": 1}, {"b": 2}])
        assert is_valid
        assert msg == ""
    
    def test_empty_list_valid(self):
        """Empty list is valid."""
        is_valid, msg = validate_data([])
        assert is_valid
    
    def test_non_list_invalid(self):
        """Non-list data is invalid."""
        is_valid, msg = validate_data("not a list")
        assert not is_valid
        assert "list" in msg.lower()
    
    def test_non_dict_rows_invalid(self):
        """Rows that aren\'t dicts are invalid."""
        is_valid, msg = validate_data([{"a": 1}, "not a dict"])
        assert not is_valid
'''
    
    def _generate_feature_tests(self, feature_type: str) -> str:
        return '''"""Tests for new features. Must pass after feature implementation."""
import pytest
import json
from reports.generator import generate_report


class TestJSONFormat:
    """Tests for JSON export format."""
    
    def test_json_format_valid_json(self):
        """JSON format produces valid JSON."""
        data = [{"name": "Alice", "age": "30"}]
        result = generate_report(data, format_type="json")
        parsed = json.loads(result)  # Should not raise
        assert "data" in parsed
    
    def test_json_has_meta(self):
        """JSON output includes meta section."""
        data = [{"name": "Alice"}]
        result = generate_report(data, format_type="json", title="Test Report", author="Tester")
        parsed = json.loads(result)
        assert "meta" in parsed
        assert parsed["meta"]["title"] == "Test Report"
        assert parsed["meta"]["author"] == "Tester"
    
    def test_json_no_header_no_meta(self):
        """JSON without header omits meta."""
        data = [{"name": "Alice"}]
        result = generate_report(data, format_type="json", include_header=False)
        parsed = json.loads(result)
        assert "meta" not in parsed or parsed.get("meta") is None
    
    def test_json_data_correct(self):
        """JSON data section contains all rows."""
        data = [{"name": "Alice"}, {"name": "Bob"}]
        result = generate_report(data, format_type="json", include_header=False, include_footer=False)
        parsed = json.loads(result)
        assert len(parsed["data"]) == 2
        assert parsed["data"][0]["name"] == "Alice"
    
    def test_json_pretty_print(self):
        """Pretty print produces indented JSON."""
        data = [{"name": "Alice"}]
        result = generate_report(data, format_type="json", pretty_print=True)
        # Pretty printed JSON has newlines and indentation
        assert "\\n" in result
        assert "  " in result or "    " in result
    
    def test_json_row_count_in_meta(self):
        """JSON meta includes row count when footer enabled."""
        data = [{"a": 1}, {"a": 2}, {"a": 3}]
        result = generate_report(data, format_type="json")
        parsed = json.loads(result)
        assert parsed["meta"]["row_count"] == 3


class TestCSVDelimiter:
    """Tests for CSV delimiter parameter."""
    
    def test_custom_delimiter_tab(self):
        """CSV with tab delimiter produces TSV."""
        data = [{"name": "Alice", "age": "30"}]
        result = generate_report(data, format_type="csv", delimiter="\\t",
                                 include_header=False, include_footer=False)
        assert "\\t" in result
        assert "," not in result.replace(",", "")  # No commas except possibly in data
    
    def test_custom_delimiter_pipe(self):
        """CSV with pipe delimiter works."""
        data = [{"name": "Alice", "age": "30"}]
        result = generate_report(data, format_type="csv", delimiter="|",
                                 include_header=False, include_footer=False)
        assert "|" in result
    
    def test_default_delimiter_comma(self):
        """Default delimiter is comma."""
        data = [{"name": "Alice", "age": "30"}]
        result = generate_report(data, format_type="csv",
                                 include_header=False, include_footer=False)
        assert "," in result
'''


# =============================================================================
# EVALUATION HARNESS
# =============================================================================

@dataclass
class AgentSubmission:
    """An agent's code submission."""
    agent_id: str
    modified_files: dict[str, str]  # filename -> content
    branch_name: str
    commit_messages: list[str]

@dataclass 
class EvaluationResult:
    """Results from evaluating a task instance."""
    instance_id: str
    
    # Test results
    unit_tests_passed: int
    unit_tests_total: int
    integration_tests_passed: int
    integration_tests_total: int
    
    # Build/static analysis
    build_success: bool
    type_check_success: bool
    lint_errors: int
    
    # Conflict resolution (for conflict templates)
    merge_conflicts_detected: int
    merge_conflicts_resolved: int
    semantic_conflicts_detected: int
    semantic_conflicts_resolved: int
    
    # Efficiency metrics
    total_agent_messages: int
    total_tokens_used: int
    rework_commits: int
    
    # captured output
    pytest_stdout: str = ""
    pytest_stderr: str = ""
    
    # Computed scores
    @property
    def integration_competence_score(self) -> float:
        """Primary metric: Integration Competence Score (ICS)."""
        if self.integration_tests_total == 0:
            if self.unit_tests_total > 0:
                test_score = self.unit_tests_passed / self.unit_tests_total
            else:
                test_score = 1.0  # Or 0.0? Default to 1.0 if no tests at all and build succeeds
        else:
            test_score = self.integration_tests_passed / self.integration_tests_total
        
        build_factor = 1.0 if self.build_success else 0.0
        type_factor = 1.0 if self.type_check_success else 0.8
        
        return test_score * build_factor * type_factor
    
    @property
    def conflict_resolution_score(self) -> float:
        """Primary metric: Conflict Resolution Score (CRS)."""
        if self.merge_conflicts_detected == 0 and self.semantic_conflicts_detected == 0:
            return 1.0  # No conflicts to resolve
        
        total_conflicts = self.merge_conflicts_detected + self.semantic_conflicts_detected
        total_resolved = self.merge_conflicts_resolved + self.semantic_conflicts_resolved
        
        return total_resolved / total_conflicts if total_conflicts > 0 else 1.0


class EvaluationHarness:
    """Harness for running and evaluating multi-agent task instances."""
    
    def __init__(self, work_dir: Path | None = None):
        self.work_dir = work_dir or Path(tempfile.mkdtemp(prefix="macb_"))
    
    def setup_task(self, instance: TaskInstance) -> Path:
        """Set up a task instance for execution."""
        task_dir = instance.to_directory(self.work_dir)
        
        # Initialize git repo
        starter_dir = task_dir / "starter"
        subprocess.run(["git", "init"], cwd=starter_dir, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=starter_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=starter_dir,
            capture_output=True,
            env={**os.environ, "GIT_AUTHOR_NAME": "Benchmark", "GIT_AUTHOR_EMAIL": "bench@test.com",
                 "GIT_COMMITTER_NAME": "Benchmark", "GIT_COMMITTER_EMAIL": "bench@test.com"}
        )
        
        return task_dir
    
    def create_agent_branch(self, task_dir: Path, agent_id: str) -> None:
        """Create a branch for an agent to work on."""
        starter_dir = task_dir / "starter"
        subprocess.run(
            ["git", "checkout", "-b", f"agent/{agent_id}"],
            cwd=starter_dir,
            capture_output=True
        )
        subprocess.run(["git", "checkout", "main"], cwd=starter_dir, capture_output=True)
    
    def apply_submission(self, task_dir: Path, submission: AgentSubmission) -> bool:
        """Apply an agent's submission to their branch."""
        starter_dir = task_dir / "starter"
        
        # Checkout agent branch
        result = subprocess.run(
            ["git", "checkout", f"agent/{submission.agent_id}"],
            cwd=starter_dir,
            capture_output=True
        )
        if result.returncode != 0:
            return False
        
        # Write modified files
        for filename, content in submission.modified_files.items():
            file_path = starter_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
        
        # Commit
        subprocess.run(["git", "add", "."], cwd=starter_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", submission.commit_messages[0] if submission.commit_messages else "Agent submission"],
            cwd=starter_dir,
            capture_output=True,
            env={**os.environ, "GIT_AUTHOR_NAME": submission.agent_id, "GIT_AUTHOR_EMAIL": f"{submission.agent_id}@test.com",
                 "GIT_COMMITTER_NAME": submission.agent_id, "GIT_COMMITTER_EMAIL": f"{submission.agent_id}@test.com"}
        )
        
        # Return to main
        subprocess.run(["git", "checkout", "main"], cwd=starter_dir, capture_output=True)
        return True
    
    def attempt_merge(self, task_dir: Path, branches: list[str]) -> tuple[bool, list[str]]:
        """Attempt to merge all agent branches. Returns (success, conflict_files)."""
        starter_dir = task_dir / "starter"
        conflict_files = []
        
        for branch in branches:
            result = subprocess.run(
                ["git", "merge", f"agent/{branch}", "--no-edit"],
                cwd=starter_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Check for conflicts
                status = subprocess.run(
                    ["git", "diff", "--name-only", "--diff-filter=U"],
                    cwd=starter_dir,
                    capture_output=True,
                    text=True
                )
                conflict_files.extend(status.stdout.strip().split("\n"))
                
                # Abort merge
                subprocess.run(["git", "merge", "--abort"], cwd=starter_dir, capture_output=True)
                return False, conflict_files
        
        return True, []
    
    def run_tests(self, task_dir: Path, test_type: str = "all") -> tuple[int, int, str, str]:
        """Run tests and return (passed, total, stdout, stderr)."""
        starter_dir = task_dir / "starter"
        
        if test_type == "unit":
            pattern = "" # Run all tests in tests/
        elif test_type == "integration":
            pattern = "integration"
        else:
            pattern = ""
        
        # Determine files to run based on pattern
        test_path = "tests/"
        if pattern:
            cmd = ["python", "-m", "pytest", test_path, "-v", "--tb=short", "-k", pattern]
        else:
            cmd = ["python", "-m", "pytest", test_path, "-v", "--tb=short"]
        
        result = subprocess.run(
            cmd,
            cwd=starter_dir,
            capture_output=True,
            text=True
        )
        
        # Parse pytest output
        stdout = result.stdout
        stderr = result.stderr
        output = stdout + stderr
        
        # Look for "X passed" pattern
        import re
        passed_match = re.search(r"(\d+) passed", output)
        failed_match = re.search(r"(\d+) failed", output)
        
        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0
        
        return passed, passed + failed, stdout, stderr
    
    def evaluate(self, instance: TaskInstance, submissions: list[AgentSubmission]) -> EvaluationResult:
        """Full evaluation of a task instance with agent submissions."""
        task_dir = self.setup_task(instance)
        
        # Create branches and apply submissions
        for submission in submissions:
            self.create_agent_branch(task_dir, submission.agent_id)
            self.apply_submission(task_dir, submission)
        
        # Attempt merge
        agent_ids = [s.agent_id for s in submissions]
        merge_success, conflict_files = self.attempt_merge(task_dir, agent_ids)
        
        # Run tests (only if merge succeeded)
        pytest_stdout = ""
        pytest_stderr = ""
        if merge_success:
            unit_passed, unit_total, unit_stdout, unit_stderr = self.run_tests(task_dir, "unit")
            int_passed, int_total, int_stdout, int_stderr = self.run_tests(task_dir, "integration")
            pytest_stdout = f"UNIT TESTS:\n{unit_stdout}\n\nINTEGRATION TESTS:\n{int_stdout}"
            pytest_stderr = f"UNIT TESTS:\n{unit_stderr}\n\nINTEGRATION TESTS:\n{int_stderr}"
        else:
            unit_passed, unit_total = 0, 0
            int_passed, int_total = 0, 0
        
        return EvaluationResult(
            instance_id=instance.instance_id,
            unit_tests_passed=unit_passed,
            unit_tests_total=unit_total,
            integration_tests_passed=int_passed,
            integration_tests_total=int_total,
            build_success=merge_success,
            type_check_success=True,  # TODO: Run mypy
            lint_errors=0,  # TODO: Run linter
            merge_conflicts_detected=len(conflict_files),
            merge_conflicts_resolved=0 if conflict_files else len(instance.expected_conflicts),
            semantic_conflicts_detected=0,  # TODO: Detect semantic conflicts
            semantic_conflicts_resolved=0,
            total_agent_messages=0,  # TODO: Track from agent harness
            total_tokens_used=0,
            rework_commits=0,
            pytest_stdout=pytest_stdout,
            pytest_stderr=pytest_stderr,
        )


# =============================================================================
# TEMPLATE: SCRUMBAN SERVICE (Backlog & Kanban)
# =============================================================================

class ScrumBanServiceTemplate(TaskTemplate):
    """
    Template for testing multi-task ScrumBan collaboration.
    
    Agents must implement multiple decoupled features from a shared backlog
    tracked on a Kanban board.
    """
    
    name = "scrumban_service"
    description = "Agents work through a backlog of service features via Kanban"
    
    def get_default_params(self) -> dict:
        return {
            "domain": "document_store",
            "num_subtasks": 3,
            "difficulty": "medium",
            "num_agents": 2,
        }
    
    def get_param_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "enum": list(DOMAINS.keys())},
                "num_subtasks": {"type": "integer", "minimum": 1, "maximum": 8},
                "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                "num_agents": {"type": "integer", "minimum": 1, "maximum": 4},
            }
        }
    
    def generate(self, params: dict, seed: int) -> TaskInstance:
        rng = Random(seed)
        p = {**self.get_default_params(), **params}
        domain = DOMAINS[p["domain"]]
        difficulty = DifficultyLevel(p["difficulty"])
        
        # Custom Story Logic per Domain
        stories = self._get_domain_stories(p["domain"], difficulty)
        
        # Select operations for subtasks
        num_tasks = min(len(domain.operations), p["num_subtasks"])
        selected_ops = domain.operations[:num_tasks]
        
        # Generate backlog with realistic stories and dependencies
        backlog = []
        for i, op in enumerate(selected_ops):
            story = stories.get(op["name"], {
                "title": f"Implement {op['name']}",
                "description": f"Implement the `{op['name']}` operation in the service implementation.\n{op['description']}"
            })
            
            # Add technical constraints based on difficulty
            constraints = ""
            if difficulty == DifficultyLevel.MEDIUM:
                constraints = "\n- Ensure basic error handling for edge cases."
            elif difficulty == DifficultyLevel.HARD:
                constraints = "\n- Implement robust state validation and atomic-like safety."
                # Add specific dependencies for HARD mode
                if i > 0:
                    prev_op = selected_ops[i-1]["name"]
                    constraints += f"\n- **Dependency**: Requires a working implementation of `{prev_op}` to pass tests."
            
            backlog.append(BacklogTask(
                id=f"TASK-{i+1}",
                title=story["title"],
                description=story["description"] + constraints,
                difficulty=difficulty
            ))
            
        # Generate interface and stub
        interface_content = self._generate_interface(domain, selected_ops)
        service_stub = self._generate_service_stub(domain)
        
        starter_files = [
            GeneratedFile(
                path="shared/interfaces.py",
                content=interface_content,
                description="Shared interface definitions"
            ),
            GeneratedFile(
                path="service/implementation.py",
                content=service_stub,
                description="Service implementation stub"
            ),
            GeneratedFile(
                path="requirements.txt",
                content="pytest>=7.0.0\n",
                description="Dependencies"
            )
        ]
        
        agent_tasks = [
            AgentTask(
                agent_id=f"agent_{i+1}",
                description=textwrap.dedent(f"""
                    # ScrumBan Challenge: {domain.name.replace('_', ' ').title()}
                    
                    Your team is implementing the {domain.name} service.
                    
                    ## Workflow
                    - Check the backlog with `read_kanban_board`.
                    - Claim an unassigned 'todo' task with `claim_task`.
                    - Implement the logic in `service/implementation.py`.
                    - Mark as 'done' with `update_task_status`.
                    - Coordinate using `send_message`.
                """).strip(),
                files_to_modify=["service/implementation.py"],
                files_readonly=["shared/interfaces.py"]
            )
            for i in range(p["num_agents"])
        ]
        
        # Generate unit tests for operations
        unit_tests = self._generate_unit_tests(domain, selected_ops)
        
        return TaskInstance(
            instance_id=f"{self.name}_{p['domain']}_{seed}",
            template_name=self.name,
            parameters=p,
            starter_files=starter_files,
            agent_tasks=agent_tasks,
            unit_tests=unit_tests,
            integration_tests=[],
            expected_conflicts=[],
            difficulty=difficulty,
            seed=seed,
            backlog=backlog
        )

    def _generate_unit_tests(self, domain, operations):
        tests = []
        for op in operations:
            test_content = self._get_op_test_content(domain, op)
            tests.append(GeneratedFile(
                path=f"tests/test_{op['name']}.py",
                content=test_content,
                description=f"Unit test for {op['name']}"
            ))
        return tests

    def _get_domain_stories(self, domain_name, difficulty):
        """Pre-defined realistic user stories for domains."""
        all_stories = {
            "cryptographic_vault": {
                "store_secret": {"title": "Secure Secret Versioning", "description": "Store a secret with policy checks and versioning. Must be atomic."},
                "retrieve_secret": {"title": "Policy-Enforced Decryption", "description": "Retrieve and decrypt a secret, but only if the requester's policy allows it."},
                "rotate_keys": {"title": "Master Key Rotation", "description": "Re-encrypt all versions of a secret with a new key. High risk of data loss!"},
                "update_policy": {"title": "Access Policy Governance", "description": "Update the security policy for a secret. Must prevent lockout scenarios."},
                "revoke_access": {"title": "Global Access Revocation", "description": "Revoke a user's access across the entire vault immediately."}
            },
            "orchestrator_service": {
                "register": {"title": "Service Instance Registration", "description": "Register a new instance with health constraints and host validation."},
                "get_healthy": {"title": "Load Balancing logic", "description": "Select healthy instances using a strict round-robin algorithm."},
                "update_status": {"title": "Instance Lifecycle Management", "description": "Manage transitions between healthy, unhealthy, and draining states."},
                "check_health": {"title": "Background Health Monitor", "description": "Detect stale heartbeats and automatically trigger state changes."},
                "trigger_failover": {"title": "Primary Instance Election", "description": "Elect a new primary instance when the current one fails. Requires consensus."}
            },
            "fintech_backend": {
                "transfer": {
                    "title": "Secure Fund Transfer logic",
                    "description": "As a user, I want to transfer money between accounts so I can manage my funds. The implementation must check if the source account has enough balance and is not frozen."
                },
                "get_balance": {
                    "title": "Real-time Balance Check",
                    "description": "As a user, I want to see my current balance instantly so I can make informed spending decisions."
                },
                "freeze_account": {
                    "title": "Account Security Freeze",
                    "description": "As an admin, I want to freeze accounts suspected of fraud to prevent unauthorized transactions."
                },
                "get_transaction_history": {
                    "title": "Transaction Audit Log",
                    "description": "As a user, I want to see my last N transactions to track my spending."
                }
            },
            "ecommerce_platform": {
                "place_order": {
                    "title": "Order Placement & Inventory reservation",
                    "description": "As a customer, I want to place an order and have the items reserved so they don't sell out before I pay."
                },
                "process_payment": {
                    "title": "Secure Payment Integration",
                    "description": "As a customer, I want my payment to be processed securely and my order status updated automatically."
                },
                "cancel_order": {
                    "title": "Order Cancellation & Stock release",
                    "description": "As a customer, I want to be able to cancel my order if it hasn't shipped yet."
                },
                "get_order_status": {
                    "title": "Order Tracking",
                    "description": "As a customer, I want to know where my order is in the fulfillment pipeline."
                }
            },
            "document_store": {
                "create": {"title": "Document Content Ingestion", "description": "Enable creation of text documents with metadata."},
                "read": {"title": "Versioned Retrieval", "description": "Retrieve documents by ID, supporting historical versions."},
                "update": {"title": "Collaborative Editing", "description": "Update document content and increment version with conflict check."},
                "delete": {"title": "Document Cleanup", "description": "Securely remove documents from the system."},
                "search": {"title": "Full-text Search", "description": "Find documents matching a specific text query."}
            }
        }
        return all_stories.get(domain_name, {})

    def _get_op_test_content(self, domain, op):
        """Generate rigorous tests for a given operation."""
        # This is a bit complex to generalize perfectly, but we can do domain-specific logic
        base = textwrap.dedent(f"""
            import pytest
            from service.implementation import {domain.data_type_name}Service
            from shared.interfaces import {domain.data_type_name}
            
            def test_{op['name']}_functionality():
                service = {domain.data_type_name}Service()
        """).strip()

        if domain.name == "fintech_backend":
            if op["name"] == "transfer":
                return base + textwrap.indent(textwrap.dedent("""
                    # Setup accounts
                    from shared.interfaces import Account
                    service._storage['acc1'] = Account(id='acc1', balance=1000.0, currency='USD', status='active')
                    service._storage['acc2'] = Account(id='acc2', balance=500.0, currency='USD', status='active')
                    
                    # Perform transfer
                    success = service.transfer('acc1', 'acc2', 200.0)
                    assert success is True
                    assert service._storage['acc1'].balance == 800.0
                    assert service._storage['acc2'].balance == 700.0
                    
                    # Test overdraft
                    success = service.transfer('acc1', 'acc2', 2000.0)
                    assert success is False
                    assert service._storage['acc1'].balance == 800.0
                """), "    ")
            elif op["name"] == "get_balance":
                return base + textwrap.indent(textwrap.dedent("""
                    from shared.interfaces import Account
                    service._storage['acc1'] = Account(id='acc1', balance=123.45, currency='USD', status='active')
                    assert service.get_balance('acc1') == 123.45
                """), "    ")
        
        elif domain.name == "orchestrator_service":
            if op["name"] == "register":
                return base + textwrap.indent(textwrap.dedent("""
                    from shared.interfaces import Instance
                    inst = Instance(id='i1', service='s1', host='1.1.1.1', status='healthy', metadata={})
                    assert service.register(inst) is True
                    # Duplicate check
                    assert service.register(inst) is False
                """), "    ")
            elif op["name"] == "get_healthy":
                return base + textwrap.indent(textwrap.dedent("""
                    from shared.interfaces import Instance
                    service.register(Instance('i1', 's1', '1.1.1.1', 'healthy', {}))
                    service.register(Instance('i2', 's1', '1.1.1.2', 'unhealthy', {}))
                    healthy = service.get_healthy('s1')
                    assert len(healthy) == 1
                    assert healthy[0].id == 'i1'
                """), "    ")
        
        elif domain.name == "cryptographic_vault":
            if op["name"] == "store_secret":
                return base + textwrap.indent(textwrap.dedent("""
                    from shared.interfaces import Secret
                    s = Secret(id='k1', value='enc1', owner='u1', policy={'allow': ['u1']}, version=1)
                    assert service.store_secret(s) is True
                    assert 'k1' in service._storage
                """), "    ")
        
        elif domain.name == "ecommerce_platform":
            if op["name"] == "place_order":
                return base + textwrap.indent(textwrap.dedent("""
                    from shared.interfaces import Order
                    order = Order(id='ord1', items=[{'sku': 'A', 'qty': 1}], total=99.9, status='pending')
                    order_id = service.place_order(order)
                    assert order_id == 'ord1'
                    assert 'ord1' in service._storage
                """), "    ")
        
        # Fallback for other domains/ops
        return base + textwrap.indent(textwrap.dedent(f"""
                # Verify method exists
                assert hasattr(service, '{op['name']}')
        """), "    ")

    def _generate_interface(self, domain, operations):
        fields_code = "\n    ".join(f"{name}: {typ}" for name, typ, _ in domain.fields)
        methods_code = ""
        for op in operations:
            params_str = ", ".join(f"{p[0]}: {p[1]}" for p in op["params"])
            methods_code += f"\n    @abstractmethod\n    def {op['name']}(self, {params_str}) -> {op['returns']}:\n        pass\n"
        
        return f"from abc import ABC, abstractmethod\nfrom dataclasses import dataclass\nfrom typing import Optional, Any\nfrom datetime import datetime\n\n@dataclass\nclass {domain.data_type_name}:\n    {fields_code}\n\nclass {domain.data_type_name}ServiceInterface(ABC):\n{methods_code}"

    def _generate_service_stub(self, domain):
        return f"from shared.interfaces import {domain.data_type_name}, {domain.data_type_name}ServiceInterface\n\nclass {domain.data_type_name}Service({domain.data_type_name}ServiceInterface):\n    def __init__(self):\n        self._storage = {{}}\n"

# =============================================================================
# TASK GENERATOR
# =============================================================================

class TaskGenerator:
    """Main class for generating benchmark task instances."""
    
    def __init__(self):
        self.templates: dict[str, TaskTemplate] = {
            "service_client": ServiceClientTemplate(),
            "refactor_collision": RefactorCollisionTemplate(),
            "scrumban_service": ScrumBanServiceTemplate(),
        }
    
    def list_templates(self) -> list[str]:
        """List available template names."""
        return list(self.templates.keys())
    
    def generate(
        self,
        template_name: str,
        params: dict | None = None,
        seed: int | None = None,
    ) -> TaskInstance:
        """Generate a single task instance."""
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.templates[template_name]
        params = params or {}
        seed = seed if seed is not None else Random().randint(0, 2**32)
        
        return template.generate(params, seed)
    
    def generate_batch(
        self,
        template_name: str,
        n: int,
        param_variations: list[dict] | None = None,
        base_seed: int = 42,
    ) -> list[TaskInstance]:
        """Generate a batch of task instances with optional parameter variations."""
        instances = []
        
        if param_variations:
            # Cycle through parameter variations
            for i in range(n):
                params = param_variations[i % len(param_variations)]
                instance = self.generate(template_name, params, base_seed + i)
                instances.append(instance)
        else:
            # Use default params
            for i in range(n):
                instance = self.generate(template_name, {}, base_seed + i)
                instances.append(instance)
        
        return instances


# =============================================================================
# END-TO-END EXAMPLE
# =============================================================================

def run_example():
    """Run an end-to-end example of the benchmark."""
    print("=" * 70)
    print("MULTI-AGENT CODING BENCHMARK - End-to-End Example")
    print("=" * 70)
    
    # 1. Generate a task instance
    print("\n[1] Generating task instance...")
    generator = TaskGenerator()
    
    instance = generator.generate(
        template_name="service_client",
        params={
            "domain": "user_cache",
            "client_type": "cli",
            "difficulty": "medium",
        },
        seed=12345,
    )
    
    print(f"    Instance ID: {instance.instance_id}")
    print(f"    Template: {instance.template_name}")
    print(f"    Difficulty: {instance.difficulty.value}")
    print(f"    Agents: {[t.agent_id for t in instance.agent_tasks]}")
    
    # 2. Write to directory
    print("\n[2] Writing task to directory...")
    output_dir = Path("./benchmark_output")
    task_dir = instance.to_directory(output_dir)
    print(f"    Task directory: {task_dir}")
    
    # 3. Show generated files
    print("\n[3] Generated files:")
    for f in instance.starter_files:
        print(f"    - {f.path}")
    for f in instance.unit_tests + instance.integration_tests:
        print(f"    - {f.path} (test)")
    
    # 4. Show agent tasks
    print("\n[4] Agent task summaries:")
    for task in instance.agent_tasks:
        print(f"\n    --- {task.agent_id} ---")
        # Show first few lines of task description
        lines = task.description.split("\n")[:5]
        for line in lines:
            print(f"    {line}")
        print("    ...")
    
    # 5. Simulate agent submissions (mock)
    print("\n[5] Simulating agent submissions (mock implementations)...")
    
    # Mock service implementation
    service_code = '''"""Service implementation for user_cache."""
from datetime import datetime, timedelta
from typing import Optional
from shared.interfaces import User, UserServiceInterface


class UserService(UserServiceInterface):
    def __init__(self):
        self._cache: dict[str, tuple[User, datetime | None]] = {}
    
    def get(self, key: str) -> Optional[User]:
        if key not in self._cache:
            return None
        user, expires = self._cache[key]
        if expires and datetime.now() > expires:
            del self._cache[key]
            return None
        return user
    
    def set(self, key: str, user: User, ttl_seconds: int = 300) -> None:
        expires = datetime.now() + timedelta(seconds=ttl_seconds) if ttl_seconds > 0 else None
        self._cache[key] = (user, expires)
    
    def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> int:
        count = len(self._cache)
        self._cache.clear()
        return count
'''
    
    # Mock client implementation
    client_code = '''"""CLI client for user_cache."""
import sys
from datetime import datetime
from service.implementation import UserService
from shared.interfaces import User


def main():
    service = UserService()
    
    if len(sys.argv) < 2:
        print("Usage: python -m client.main <command> [args]")
        print("Commands: get, set, delete, clear")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "get":
        if len(sys.argv) < 3:
            print("Usage: get <key>")
            sys.exit(1)
        key = sys.argv[2]
        user = service.get(key)
        if user:
            print(f"User: {user.name} ({user.email})")
        else:
            print("Not found")
    
    elif command == "set":
        if len(sys.argv) < 5:
            print("Usage: set <key> <name> <email>")
            sys.exit(1)
        key, name, email = sys.argv[2], sys.argv[3], sys.argv[4]
        user = User(id=key, name=name, email=email, created_at=datetime.now())
        service.set(key, user)
        print(f"Stored user {name}")
    
    elif command == "delete":
        if len(sys.argv) < 3:
            print("Usage: delete <key>")
            sys.exit(1)
        key = sys.argv[2]
        if service.delete(key):
            print(f"Deleted {key}")
        else:
            print("Not found")
    
    elif command == "clear":
        count = service.clear()
        print(f"Cleared {count} entries")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
    
    submissions = [
        AgentSubmission(
            agent_id="agent_service",
            modified_files={"service/implementation.py": service_code},
            branch_name="agent/agent_service",
            commit_messages=["Implement UserService with TTL support"],
        ),
        AgentSubmission(
            agent_id="agent_client",
            modified_files={"client/main.py": client_code},
            branch_name="agent/agent_client", 
            commit_messages=["Implement CLI client with all commands"],
        ),
    ]
    
    print(f"    Submissions from: {[s.agent_id for s in submissions]}")
    
    # 6. Evaluate
    print("\n[6] Running evaluation...")
    harness = EvaluationHarness()
    result = harness.evaluate(instance, submissions)
    
    print(f"\n    === RESULTS ===")
    print(f"    Build success: {result.build_success}")
    print(f"    Unit tests: {result.unit_tests_passed}/{result.unit_tests_total}")
    print(f"    Integration tests: {result.integration_tests_passed}/{result.integration_tests_total}")
    print(f"    Integration Competence Score (ICS): {result.integration_competence_score:.2%}")
    print(f"    Conflict Resolution Score (CRS): {result.conflict_resolution_score:.2%}")
    
    print("\n" + "=" * 70)
    print("Example complete! Check ./benchmark_output for generated files.")
    print("=" * 70)


if __name__ == "__main__":
    run_example()