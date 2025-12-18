"""
Example: Creating a Custom Template for SCRUMBENCH
===================================================

This example shows how to create a new task template (Code Review Challenge).
Templates define how tasks are generated and what agents need to do.

Usage:
    1. Copy this file to your project
    2. Modify the template logic
    3. Register in TaskGenerator
    4. Use in experiments
"""

from core.benchmark import (
    TaskTemplate, TaskInstance, AgentTask, GeneratedFile,
    DifficultyLevel, DOMAINS
)
from random import Random
from pathlib import Path


class CodeReviewTemplate(TaskTemplate):
    """
    Template for testing code review capabilities.
    
    Agent 1: Writes code with intentional bugs
    Agent 2: Reviews code and identifies issues
    
    Success requires Agent 2 to find all bugs and suggest fixes.
    """
    
    name = "code_review"
    description = "Agents review code and identify bugs"
    
    def get_default_params(self) -> dict:
        return {
            "domain": "user_cache",
            "num_bugs": 3,
            "bug_types": ["logic", "edge_case", "performance"],
            "difficulty": "medium"
        }
    
    def get_param_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "enum": list(DOMAINS.keys())
                },
                "num_bugs": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10
                },
                "bug_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["logic", "edge_case", "performance", "security"]
                    }
                },
                "difficulty": {
                    "type": "string",
                    "enum": ["easy", "medium", "hard"]
                }
            }
        }
    
    def generate(self, params: dict, seed: int) -> TaskInstance:
        """Generate a code review task instance."""
        rng = Random(seed)
        p = {**self.get_default_params(), **params}
        
        domain = DOMAINS[p["domain"]]
        difficulty = DifficultyLevel(p["difficulty"])
        
        # Generate buggy implementation
        buggy_code = self._generate_buggy_code(domain, p["num_bugs"], p["bug_types"], rng)
        
        # Create starter files
        starter_files = [
            GeneratedFile(
                path="shared/interfaces.py",
                content=self._generate_interface(domain),
                description="Shared interface (correct)"
            ),
            GeneratedFile(
                path="implementation/service.py",
                content=buggy_code,
                description="Implementation with bugs"
            ),
            GeneratedFile(
                path="tests/test_service.py",
                content=self._generate_tests(domain),
                description="Test suite"
            ),
        ]
        
        # Define agent tasks
        agent_tasks = [
            AgentTask(
                agent_id="agent_reviewer",
                description=f"""
                Review the code in implementation/service.py and identify all bugs.
                
                The implementation should conform to the interface in shared/interfaces.py.
                There are approximately {p['num_bugs']} bugs of types: {', '.join(p['bug_types'])}.
                
                For each bug:
                1. Identify the location (file, line number)
                2. Describe the issue
                3. Suggest a fix
                
                Create a file review_report.md with your findings.
                """.strip(),
                files_to_modify=["review_report.md"],
                files_readonly=["shared/interfaces.py", "implementation/service.py", "tests/test_service.py"]
            ),
        ]
        
        # Generate expected bug list for evaluation
        expected_bugs = self._generate_bug_list(p["num_bugs"], p["bug_types"])
        
        # Create unit tests that will fail due to bugs
        unit_tests = [
            GeneratedFile(
                path="tests/test_service.py",
                content=self._generate_tests(domain),
                description="Tests that expose bugs"
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
            integration_tests=[],
            expected_conflicts=[],
            difficulty=difficulty,
            seed=seed,
        )
    
    def _generate_buggy_code(self, domain, num_bugs, bug_types, rng):
        """Generate implementation with intentional bugs."""
        # This is a simplified example
        # In practice, you'd generate more sophisticated buggy code
        
        bugs = []
        if "logic" in bug_types:
            bugs.append("# BUG: Using wrong comparison operator")
        if "edge_case" in bug_types:
            bugs.append("# BUG: Not handling None/empty input")
        if "performance" in bug_types:
            bugs.append("# BUG: O(n^2) when O(n) is possible")
        
        return f'''"""
Implementation with bugs for code review.
"""
from shared.interfaces import {domain.data_type_name}, {domain.data_type_name}ServiceInterface

class {domain.data_type_name}Service({domain.data_type_name}ServiceInterface):
    def __init__(self):
        self.storage = {{}}
    
    def get(self, key: str):
        {bugs[0] if bugs else ""}
        return self.storage.get(key)  # Missing validation
    
    def set(self, key: str, value):
        {bugs[1] if len(bugs) > 1 else ""}
        self.storage[key] = value  # No error handling
    
    def delete(self, key: str):
        {bugs[2] if len(bugs) > 2 else ""}
        if key in self.storage:  # Inefficient check
            del self.storage[key]
            return True
        return False
'''
    
    def _generate_interface(self, domain):
        """Generate the correct interface."""
        return f'''"""Interface for {domain.name}."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class {domain.data_type_name}:
    id: str
    value: str

class {domain.data_type_name}ServiceInterface(ABC):
    @abstractmethod
    def get(self, key: str):
        pass
    
    @abstractmethod
    def set(self, key: str, value):
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        pass
'''
    
    def _generate_tests(self, domain):
        """Generate tests that will expose bugs."""
        return f'''"""Tests for {domain.name} service."""
import pytest
from implementation.service import {domain.data_type_name}Service

def test_get_nonexistent():
    service = {domain.data_type_name}Service()
    result = service.get("missing")
    assert result is None

def test_set_and_get():
    service = {domain.data_type_name}Service()
    service.set("key1", "value1")
    assert service.get("key1") == "value1"

def test_delete():
    service = {domain.data_type_name}Service()
    service.set("key1", "value1")
    assert service.delete("key1") == True
    assert service.get("key1") is None
'''
    
    def _generate_bug_list(self, num_bugs, bug_types):
        """Generate list of expected bugs for evaluation."""
        return [
            {"type": bug_type, "severity": "medium"}
            for bug_type in bug_types[:num_bugs]
        ]


# Register the template
if __name__ == "__main__":
    from core.benchmark import TaskGenerator
    
    # Add to task generator
    generator = TaskGenerator()
    generator.templates["code_review"] = CodeReviewTemplate()
    
    # Generate a sample task
    template = CodeReviewTemplate()
    task = template.generate({"domain": "user_cache", "num_bugs": 3}, seed=42)
    
    print(f"Generated task: {task.instance_id}")
    print(f"Starter files: {[f.path for f in task.starter_files]}")
    print(f"Agent tasks: {[t.agent_id for t in task.agent_tasks]}")
    
    # Write to directory for inspection
    output_dir = Path("./example_task_output")
    task.to_directory(output_dir)
    print(f"\nTask written to: {output_dir}")
