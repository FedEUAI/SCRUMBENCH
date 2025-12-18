"""
MACB Experiment Runner
======================
Integration script and experiment runner for the Multi-Agent Coding Benchmark.

Features:
- End-to-end task generation → agent execution → evaluation pipeline
- Experiment configuration via YAML/dict
- Batch execution with parallelization
- Result aggregation and statistical analysis
- Ablation study support
- Progress tracking and checkpointing
"""

from __future__ import annotations
import os
import json
import time
import hashlib
import shutil
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Literal
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import statistics
import subprocess

from .benchmark import TaskGenerator, TaskInstance, EvaluationHarness, EvaluationResult, AgentSubmission
from .harness import (LLMProvider, OpenAIProvider, AnthropicProvider, MockProvider,
                            Agent, AgentConfig, MultiAgentHarness, HarnessConfig, 
                            CommunicationPattern, ExecutionTrace)


# =============================================================================
# CONFIGURATION SCHEMA
# =============================================================================

@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    type: Literal["openai", "anthropic", "mock"]
    model: str = ""
    api_key: str | None = None  # If None, uses environment variable
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    extra_params: dict = field(default_factory=dict)


@dataclass
class AgentRoleConfig:
    """Configuration for an agent role in experiments."""
    role_id: str  # e.g., "service_implementer", "client_implementer"
    provider_config: ProviderConfig
    max_turns: int = 50
    system_prompt_override: str | None = None


@dataclass  
class TaskConfig:
    """Configuration for task generation."""
    template: str
    params: dict = field(default_factory=dict)
    num_instances: int = 10
    base_seed: int = 42
    difficulty_levels: list[str] = field(default_factory=lambda: ["easy", "medium", "hard"])


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    description: str = ""
    
    # Task generation
    tasks: list[TaskConfig] = field(default_factory=list)
    
    # Agent configuration
    agent_roles: list[AgentRoleConfig] = field(default_factory=list)
    communication_pattern: str = "direct"  # isolated, broadcast, direct, scratchpad
    
    # Execution settings
    max_total_turns: int = 200
    sprint_token_budget: int = 1000000  # Default 1M tokens per sprint
    timeout_seconds: int = 1800  # 30 min per task
    parallel_workers: int = 1
    
    # Output settings
    output_dir: str = "./experiments"
    save_traces: bool = True
    save_agent_outputs: bool = True
    checkpoint_interval: int = 10  # Save progress every N tasks
    
    # Reproducibility
    random_seed: int = 42
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        # Handle nested dataclasses
        if "tasks" in d:
            d["tasks"] = [TaskConfig(**t) if isinstance(t, dict) else t for t in d["tasks"]]
        if "agent_roles" in d:
            roles = []
            for r in d["agent_roles"]:
                if isinstance(r, dict):
                    if "provider_config" in r and isinstance(r["provider_config"], dict):
                        r["provider_config"] = ProviderConfig(**r["provider_config"])
                    roles.append(AgentRoleConfig(**r))
                else:
                    roles.append(r)
            d["agent_roles"] = roles
        return cls(**d)
    
    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        import yaml
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))
    
    def to_yaml(self, path: str):
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# =============================================================================
# RESULT DATA STRUCTURES
# =============================================================================

@dataclass
class TaskResult:
    """Result from running a single task instance."""
    instance_id: str
    template: str
    parameters: dict
    difficulty: str
    seed: int
    
    # Execution metrics
    success: bool
    outcome: str  # "success", "timeout", "error", "merge_failed"
    duration_seconds: float
    total_turns: int
    
    # Agent metrics
    agent_tokens: dict[str, dict]  # {agent_id: {prompt: N, completion: M}}
    agent_submissions: dict[str, bool]  # {agent_id: submitted?}
    messages_exchanged: int
    
    # Evaluation metrics (from EvaluationHarness)
    build_success: bool = False
    unit_tests_passed: int = 0
    unit_tests_total: int = 0
    integration_tests_passed: int = 0
    integration_tests_total: int = 0
    merge_conflicts_detected: int = 0
    merge_conflicts_resolved: int = 0
    
    # Computed scores
    integration_competence_score: float = 0.0
    conflict_resolution_score: float = 0.0
    
    # Error info if failed
    error_message: str | None = None
    error_traceback: str | None = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentResults:
    """Aggregated results from an experiment."""
    experiment_name: str
    config: ExperimentConfig
    
    # Timing
    start_time: str
    end_time: str
    total_duration_seconds: float
    
    # Task results
    task_results: list[TaskResult] = field(default_factory=list)
    
    # Aggregated metrics
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    timeout_tasks: int = 0
    budget_exhausted_tasks: int = 0
    
    # Summary statistics (computed)
    mean_ics: float = 0.0  # Integration Competence Score
    std_ics: float = 0.0
    mean_crs: float = 0.0  # Conflict Resolution Score
    std_crs: float = 0.0
    mean_duration: float = 0.0
    mean_turns: float = 0.0
    mean_tokens: float = 0.0
    
    def compute_statistics(self):
        """Compute summary statistics from task results."""
        if not self.task_results:
            return
        
        self.total_tasks = len(self.task_results)
        self.successful_tasks = sum(1 for r in self.task_results if r.success)
        self.failed_tasks = sum(1 for r in self.task_results if r.outcome == "error")
        self.timeout_tasks = sum(1 for r in self.task_results if r.outcome == "timeout")
        self.budget_exhausted_tasks = sum(1 for r in self.task_results if r.outcome == "budget_exhausted")
        
        ics_scores = [r.integration_competence_score for r in self.task_results]
        crs_scores = [r.conflict_resolution_score for r in self.task_results]
        durations = [r.duration_seconds for r in self.task_results]
        turns = [r.total_turns for r in self.task_results]
        
        total_tokens = []
        for r in self.task_results:
            task_tokens = sum(
                t.get("prompt", 0) + t.get("completion", 0) 
                for t in r.agent_tokens.values()
            )
            total_tokens.append(task_tokens)
        
        self.mean_ics = statistics.mean(ics_scores) if ics_scores else 0
        self.std_ics = statistics.stdev(ics_scores) if len(ics_scores) > 1 else 0
        self.mean_crs = statistics.mean(crs_scores) if crs_scores else 0
        self.std_crs = statistics.stdev(crs_scores) if len(crs_scores) > 1 else 0
        self.mean_duration = statistics.mean(durations) if durations else 0
        self.mean_turns = statistics.mean(turns) if turns else 0
        self.mean_tokens = statistics.mean(total_tokens) if total_tokens else 0
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["config"] = self.config.to_dict()
        return d
    
    def save(self, path: str):
        """Save results to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def summary_table(self) -> str:
        """Generate a summary table for display."""
        lines = [
            "=" * 70,
            f"EXPERIMENT: {self.experiment_name}",
            "=" * 70,
            f"Duration: {self.total_duration_seconds:.1f}s",
            f"Tasks: {self.total_tasks} total, {self.successful_tasks} successful, "
            f"{self.failed_tasks} failed, {self.timeout_tasks} timeout, "
            f"{self.budget_exhausted_tasks} budget exhausted",
            "",
            "METRICS (mean ± std):",
            f"  Integration Competence Score: {self.mean_ics:.3f} ± {self.std_ics:.3f}",
            f"  Conflict Resolution Score:    {self.mean_crs:.3f} ± {self.std_crs:.3f}",
            f"  Duration per task:            {self.mean_duration:.1f}s",
            f"  Turns per task:               {self.mean_turns:.1f}",
            f"  Tokens per task:              {self.mean_tokens:.0f}",
            "=" * 70,
        ]
        return "\n".join(lines)
    
    def breakdown_by_difficulty(self) -> dict:
        """Get metrics broken down by difficulty level."""
        by_difficulty = {}
        for r in self.task_results:
            d = r.difficulty
            if d not in by_difficulty:
                by_difficulty[d] = {"ics": [], "crs": [], "success": [], "duration": []}
            by_difficulty[d]["ics"].append(r.integration_competence_score)
            by_difficulty[d]["crs"].append(r.conflict_resolution_score)
            by_difficulty[d]["success"].append(1 if r.success else 0)
            by_difficulty[d]["duration"].append(r.duration_seconds)
        
        summary = {}
        for d, metrics in by_difficulty.items():
            summary[d] = {
                "n": len(metrics["ics"]),
                "mean_ics": statistics.mean(metrics["ics"]) if metrics["ics"] else 0,
                "mean_crs": statistics.mean(metrics["crs"]) if metrics["crs"] else 0,
                "success_rate": statistics.mean(metrics["success"]) if metrics["success"] else 0,
                "mean_duration": statistics.mean(metrics["duration"]) if metrics["duration"] else 0,
            }
        return summary
    
    def breakdown_by_template(self) -> dict:
        """Get metrics broken down by task template."""
        by_template = {}
        for r in self.task_results:
            t = r.template
            if t not in by_template:
                by_template[t] = {"ics": [], "crs": [], "success": []}
            by_template[t]["ics"].append(r.integration_competence_score)
            by_template[t]["crs"].append(r.conflict_resolution_score)
            by_template[t]["success"].append(1 if r.success else 0)
        
        summary = {}
        for t, metrics in by_template.items():
            summary[t] = {
                "n": len(metrics["ics"]),
                "mean_ics": statistics.mean(metrics["ics"]) if metrics["ics"] else 0,
                "mean_crs": statistics.mean(metrics["crs"]) if metrics["crs"] else 0,
                "success_rate": statistics.mean(metrics["success"]) if metrics["success"] else 0,
            }
        return summary


# =============================================================================
# PROVIDER FACTORY
# =============================================================================

def create_provider(config: ProviderConfig):
    """Factory function to create LLM providers from config."""
    # In real implementation, import from agent_harness
    # from agent_harness import OpenAIProvider, AnthropicProvider, MockProvider
    
    if config.type == "openai":
        return OpenAIProvider(
            model=config.model or "gpt-4o",
            api_key=config.api_key,
            base_url=config.base_url,
        )
    
    elif config.type == "anthropic":
        return AnthropicProvider(
            model=config.model or "claude-sonnet-4-20250514",
            api_key=config.api_key,
        )
    
    elif config.type == "mock":
        return MockProvider(
            responses=config.extra_params.get("responses", ["Mock response"])
        )
    
    else:
        raise ValueError(f"Unknown provider type: {config.type}")




# =============================================================================
# SINGLE TASK RUNNER
# =============================================================================

class TaskRunner:
    """
    Runs a single task instance with agents and returns results.
    
    This is the core execution unit that:
    1. Sets up the workspace from a TaskInstance
    2. Initializes agents with their tasks
    3. Runs the multi-agent harness
    4. Evaluates the results
    5. Returns a TaskResult
    """
    
    def __init__(
        self,
        task_instance,  # TaskInstance from benchmark
        agent_roles: list[AgentRoleConfig],
        communication_pattern: str = "direct",
        max_total_turns: int = 200,
        sprint_token_budget: int = 1000000,
        timeout_seconds: int = 1800,
        work_dir: Path | None = None,
        save_traces: bool = True,
    ):
        self.task_instance = task_instance
        self.agent_roles = agent_roles
        self.communication_pattern = communication_pattern
        self.max_total_turns = max_total_turns
        self.sprint_token_budget = sprint_token_budget
        self.timeout_seconds = timeout_seconds
        self.work_dir = work_dir or Path(f"./work/{task_instance.instance_id}")
        self.save_traces = save_traces
    
    def run(self) -> TaskResult:
        """Execute the task and return results."""
        start_time = time.time()
        
        try:
            # 1. Set up workspace
            task_dir = self._setup_workspace()
            
            # 2. Create agents
            agents = self._create_agents(task_dir)
            
            # 3. Run multi-agent execution
            trace = self._run_agents(agents)
            
            # 4. Collect agent outputs
            submissions = self._collect_submissions(agents, task_dir)
            
            # 5. Evaluate results
            eval_result = self._evaluate(task_dir, submissions)
            
            duration = time.time() - start_time
            
            # 6. Save diagnostics and traces
            if self.save_traces:
                trace_file = task_dir / "trace.json"
                trace_file.write_text(json.dumps(trace.to_dict(), indent=2))
                
                diag_file = task_dir / "diagnostics.json"
                diag_data = {
                    "instance_id": self.task_instance.instance_id,
                    "pytest_stdout": eval_result.pytest_stdout,
                    "pytest_stderr": eval_result.pytest_stderr,
                    "build_success": eval_result.build_success,
                    "unit_tests_passed": eval_result.unit_tests_passed,
                    "unit_tests_total": eval_result.unit_tests_total,
                    "unit_tests_failed": eval_result.unit_tests_total - eval_result.unit_tests_passed,
                    "integration_tests_passed": eval_result.integration_tests_passed,
                    "integration_tests_total": eval_result.integration_tests_total,
                    "integration_tests_failed": eval_result.integration_tests_total - eval_result.integration_tests_passed,
                    "integration_competence_score": eval_result.integration_competence_score,
                    "outcome": trace.outcome,
                }
                diag_file.write_text(json.dumps(diag_data, indent=2))
            
            return TaskResult(
                instance_id=self.task_instance.instance_id,
                template=self.task_instance.template_name,
                parameters=self.task_instance.parameters,
                difficulty=self.task_instance.difficulty.value,
                seed=self.task_instance.seed,
                success=eval_result.build_success and (
                    (eval_result.unit_tests_passed > 0 or eval_result.integration_tests_passed > 0) and
                    (eval_result.unit_tests_total - eval_result.unit_tests_passed == 0) and
                    (eval_result.integration_tests_total - eval_result.integration_tests_passed == 0)
                ),
                outcome=trace.outcome,
                duration_seconds=duration,
                total_turns=len(trace.turns),
                agent_tokens=trace.total_tokens,
                agent_submissions={a.config.agent_id: a.submitted for a in agents},
                messages_exchanged=len(trace.messages_exchanged),
                build_success=eval_result.build_success,
                unit_tests_passed=eval_result.unit_tests_passed,
                unit_tests_total=eval_result.unit_tests_total,
                integration_tests_passed=eval_result.integration_tests_passed,
                integration_tests_total=eval_result.integration_tests_total,
                merge_conflicts_detected=eval_result.merge_conflicts_detected,
                merge_conflicts_resolved=eval_result.merge_conflicts_resolved,
                integration_competence_score=eval_result.integration_competence_score,
                conflict_resolution_score=eval_result.conflict_resolution_score,
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TaskResult(
                instance_id=self.task_instance.instance_id,
                template=self.task_instance.template_name,
                parameters=self.task_instance.parameters,
                difficulty=self.task_instance.difficulty.value,
                seed=self.task_instance.seed,
                success=False,
                outcome="error",
                duration_seconds=duration,
                total_turns=0,
                agent_tokens={},
                agent_submissions={},
                messages_exchanged=0,
                error_message=str(e),
                error_traceback=traceback.format_exc(),
            )
    
    def _setup_workspace(self) -> Path:
        """Set up the task workspace."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        task_dir = self.task_instance.to_directory(self.work_dir)
        
        # Initialize git in starter directory
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
    
    def _create_agents(self, task_dir: Path) -> list:
        """Create agent instances."""
        # In real implementation:
        # from agent_harness import Agent, AgentConfig
        
        agents = []
        workspace = task_dir / "starter"
        
        # Map task instance agents to role configs
        task_agents = {t.agent_id: t for t in self.task_instance.agent_tasks}
        
        for role in self.agent_roles:
            # Find matching task
            matching_task = None
            for task_agent_id, task in task_agents.items():
                if role.role_id in task_agent_id or task_agent_id in role.role_id:
                    matching_task = task
                    break
            
            if not matching_task:
                # Use first available unassigned task
                for task_agent_id, task in task_agents.items():
                    matching_task = task
                    break
            
            if not matching_task:
                continue
            
            provider = create_provider(role.provider_config)
            
            # Create agent
            agent = Agent(
                config=AgentConfig(
                    agent_id=matching_task.agent_id,
                    system_prompt=role.system_prompt_override or "You are a helpful coding agent.",
                    task_description=matching_task.description,
                    provider=provider,
                    max_turns=role.max_turns,
                ),
                workspace=workspace,
            )
            agents.append(agent)
        
        return agents
    
    def _run_agents(self, agents: list):
        """Run the multi-agent execution."""
        harness_config = HarnessConfig(
            communication=CommunicationPattern(self.communication_pattern),
            max_total_turns=self.max_total_turns,
            max_total_tokens=self.sprint_token_budget,
        )
        harness = MultiAgentHarness(harness_config)
        for agent in agents:
            harness.add_agent(agent)
        
        return harness.run()
    
    def _collect_submissions(self, agents: list[Agent], task_dir: Path) -> list[AgentSubmission]:
        """Collect agent submissions."""
        submissions = []
        for agent in agents:
            submissions.append(AgentSubmission(
                agent_id=agent.config.agent_id,
                modified_files=agent.get_modified_files(),
                branch_name=f"agent-{agent.config.agent_id}",
                commit_messages=["Final submission"]
            ))
        return submissions
    
    def _evaluate(self, task_dir: Path, submissions: list[AgentSubmission]) -> EvaluationResult:
        """Evaluate the task results."""
        harness = EvaluationHarness()
        return harness.evaluate(self.task_instance, submissions)




# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """
    Main experiment runner that orchestrates:
    - Task generation
    - Parallel execution
    - Result collection
    - Checkpointing
    - Statistical analysis
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.task_instances = []
        self.results = []
        self.start_time = None
        self.checkpoint_file = self.output_dir / "checkpoint.json"
    
    def generate_tasks(self) -> list:
        """Generate all task instances for the experiment."""
        all_instances = []
        generator = TaskGenerator()
        
        for task_config in self.config.tasks:
            for difficulty in task_config.difficulty_levels:
                params = {**task_config.params, "difficulty": difficulty}
                
                for i in range(task_config.num_instances):
                    seed = task_config.base_seed + i + hash(difficulty) % 1000
                    
                    instance = generator.generate(
                        template_name=task_config.template,
                        params=params,
                        seed=seed,
                    )
                    
                    all_instances.append(instance)
        
        self.task_instances = all_instances
        print(f"Generated {len(all_instances)} task instances")
        return all_instances
    
    def run(self, resume: bool = True) -> ExperimentResults:
        """
        Run the complete experiment.
        
        Args:
            resume: If True, resume from checkpoint if available
        """
        self.start_time = datetime.now()
        
        # Load checkpoint if resuming
        completed_ids = set()
        if resume and self.checkpoint_file.exists():
            checkpoint = json.loads(self.checkpoint_file.read_text())
            self.results = [TaskResult(**r) for r in checkpoint.get("results", [])]
            completed_ids = {r.instance_id for r in self.results}
            print(f"Resumed from checkpoint: {len(completed_ids)} tasks already completed")
        
        # Generate tasks if not already done
        if not self.task_instances:
            self.generate_tasks()
        
        # Filter to remaining tasks
        remaining = [t for t in self.task_instances if t.instance_id not in completed_ids]
        print(f"Running {len(remaining)} remaining tasks...")
        
        # Save config
        config_file = self.output_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save task manifest for dashboard
        manifest_file = self.output_dir / "task_manifest.json"
        manifest = [{
            "instance_id": t.instance_id,
            "template": t.template_name,
            "difficulty": t.difficulty.value if hasattr(t.difficulty, 'value') else str(t.difficulty),
            "parameters": t.parameters,
        } for t in self.task_instances]
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Run tasks
        if self.config.parallel_workers > 1:
            self._run_parallel(remaining)
        else:
            self._run_sequential(remaining)
        
        end_time = datetime.now()
        
        experiment_results = self._update_results_files()
        
        print(experiment_results.summary_table())
        
        return experiment_results
    
    def _run_sequential(self, tasks: list):
        """Run tasks sequentially."""
        for i, task in enumerate(tasks):
            print(f"[{i+1}/{len(tasks)}] Running {task.instance_id}...")
            
            result = self._run_single_task(task)
            self.results.append(result)
            
            print(f"  -> {result.outcome}, ICS={result.integration_competence_score:.3f}, "
                  f"duration={result.duration_seconds:.1f}s")
            
            # Update results files after each task
            self._update_results_files()
            
            # Checkpoint
            if (i + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint()
    
    def _run_parallel(self, tasks: list):
        """Run tasks in parallel."""
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = {
                executor.submit(self._run_single_task, task): task 
                for task in tasks
            }
            
            completed = 0
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    self.results.append(result)
                    completed += 1
                    
                    print(f"[{completed}/{len(tasks)}] {task.instance_id}: "
                          f"{result.outcome}, ICS={result.integration_competence_score:.3f}")
                    
                    # Update results files after each task
                    self._update_results_files()
                    
                    if completed % self.config.checkpoint_interval == 0:
                        self._save_checkpoint()
                        
                except Exception as e:
                    print(f"[ERROR] {task.instance_id}: {e}")
                    self.results.append(TaskResult(
                        instance_id=task.instance_id,
                        template=task.template_name,
                        parameters=task.parameters,
                        difficulty=task.difficulty,
                        seed=task.seed,
                        success=False,
                        outcome="error",
                        duration_seconds=0,
                        total_turns=0,
                        agent_tokens={},
                        agent_submissions={},
                        messages_exchanged=0,
                        error_message=str(e),
                    ))
    
    def _run_single_task(self, task_instance) -> TaskResult:
        """Run a single task instance."""
        runner = TaskRunner(
            task_instance=task_instance,
            agent_roles=self.config.agent_roles,
            communication_pattern=self.config.communication_pattern,
            max_total_turns=self.config.max_total_turns,
            sprint_token_budget=self.config.sprint_token_budget,
            timeout_seconds=self.config.timeout_seconds,
            work_dir=self.output_dir / "workspaces",
            save_traces=self.config.save_traces,
        )
        return runner.run()
    
    def _save_checkpoint(self):
        """Save checkpoint for resumability."""
        checkpoint = {
            "results": [r.to_dict() for r in self.results],
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f)
        print(f"  [Checkpoint saved: {len(self.results)} results]")
    
    def _update_results_files(self) -> ExperimentResults:
        """Update results.json and summary.txt with current progress."""
        current_time = datetime.now()
        start_time = self.start_time or current_time
        
        experiment_results = ExperimentResults(
            experiment_name=self.config.name,
            config=self.config,
            start_time=start_time.isoformat(),
            end_time=current_time.isoformat(),
            total_duration_seconds=(current_time - start_time).total_seconds(),
            task_results=self.results,
        )
        experiment_results.compute_statistics()
        
        # Save results
        results_file = self.output_dir / "results.json"
        experiment_results.save(str(results_file))
        
        # Save summary
        summary_file = self.output_dir / "summary.txt"
        summary_file.write_text(experiment_results.summary_table())
        
        return experiment_results


# Mock for standalone testing
class MockTaskInstance:
    def __init__(self, instance_id, template_name, parameters, difficulty, seed):
        self.instance_id = instance_id
        self.template_name = template_name
        self.parameters = parameters
        self.difficulty = type('obj', (object,), {'value': difficulty})()
        self.seed = seed
        self.agent_tasks = [
            type('obj', (object,), {'agent_id': 'agent_service', 'description': 'Implement service'})(),
            type('obj', (object,), {'agent_id': 'agent_client', 'description': 'Implement client'})(),
        ]
    
    def to_directory(self, base_path):
        task_dir = Path(base_path) / self.instance_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "starter").mkdir(exist_ok=True)
        (task_dir / "tasks").mkdir(exist_ok=True)
        return task_dir


# =============================================================================
# ABLATION STUDY SUPPORT
# =============================================================================

class AblationRunner:
    """
    Runner for ablation studies that varies experimental conditions.
    
    Supports:
    - Communication pattern ablations
    - Model ablations
    - Agent count ablations
    - Parameter sweeps
    """
    
    def __init__(self, base_config: ExperimentConfig):
        self.base_config = base_config
        self.ablation_results: dict[str, ExperimentResults] = {}
    
    def run_communication_ablation(self) -> dict[str, ExperimentResults]:
        """Run ablation over communication patterns."""
        patterns = ["isolated", "broadcast", "direct", "scratchpad"]
        
        for pattern in patterns:
            config = ExperimentConfig.from_dict(self.base_config.to_dict())
            config.name = f"{self.base_config.name}_comm_{pattern}"
            config.communication_pattern = pattern
            
            print(f"\n{'='*70}")
            print(f"ABLATION: Communication Pattern = {pattern}")
            print(f"{'='*70}")
            
            runner = ExperimentRunner(config)
            results = runner.run()
            self.ablation_results[f"comm_{pattern}"] = results
        
        return self.ablation_results
    
    def run_model_ablation(self, models: list[dict]) -> dict[str, ExperimentResults]:
        """
        Run ablation over different models.
        
        Args:
            models: List of {"name": str, "provider_config": ProviderConfig}
        """
        for model_spec in models:
            config = ExperimentConfig.from_dict(self.base_config.to_dict())
            config.name = f"{self.base_config.name}_model_{model_spec['name']}"
            
            # Update all agent roles to use this model
            for role in config.agent_roles:
                role.provider_config = model_spec["provider_config"]
            
            print(f"\n{'='*70}")
            print(f"ABLATION: Model = {model_spec['name']}")
            print(f"{'='*70}")
            
            runner = ExperimentRunner(config)
            results = runner.run()
            self.ablation_results[f"model_{model_spec['name']}"] = results
        
        return self.ablation_results
    
    def compare_results(self) -> str:
        """Generate comparison table across ablations."""
        if not self.ablation_results:
            return "No ablation results to compare."
        
        lines = [
            "=" * 90,
            "ABLATION COMPARISON",
            "=" * 90,
            f"{'Condition':<30} {'N':>5} {'ICS':>10} {'CRS':>10} {'Success%':>10} {'Tokens':>12}",
            "-" * 90,
        ]
        
        for name, results in self.ablation_results.items():
            success_rate = results.successful_tasks / results.total_tasks * 100 if results.total_tasks > 0 else 0
            lines.append(
                f"{name:<30} {results.total_tasks:>5} "
                f"{results.mean_ics:>10.3f} {results.mean_crs:>10.3f} "
                f"{success_rate:>9.1f}% {results.mean_tokens:>12.0f}"
            )
        
        lines.append("=" * 90)
        return "\n".join(lines)


# =============================================================================
# BASELINE CONFIGURATIONS
# =============================================================================

def get_baseline_configs() -> dict[str, ExperimentConfig]:
    """Get standard baseline experiment configurations."""
    
    # Base provider config (override with actual API keys)
    openai_config = ProviderConfig(type="openai", model="gpt-4o")
    anthropic_config = ProviderConfig(type="anthropic", model="claude-sonnet-4-20250514")
    
    baselines = {}
    
    # Baseline 1: Single agent, sequential
    baselines["single_agent_sequential"] = ExperimentConfig(
        name="baseline_single_agent",
        description="Single agent handles all tasks sequentially",
        tasks=[
            TaskConfig(template="service_client", num_instances=10),
        ],
        agent_roles=[
            AgentRoleConfig(role_id="solo_agent", provider_config=openai_config, max_turns=100),
        ],
        communication_pattern="isolated",
    )
    
    # Baseline 2: Multi-agent, no communication
    baselines["multi_agent_isolated"] = ExperimentConfig(
        name="baseline_multi_agent_isolated",
        description="Multiple agents with no communication",
        tasks=[
            TaskConfig(template="service_client", num_instances=10),
        ],
        agent_roles=[
            AgentRoleConfig(role_id="agent_service", provider_config=openai_config),
            AgentRoleConfig(role_id="agent_client", provider_config=openai_config),
        ],
        communication_pattern="isolated",
    )
    
    # Baseline 3: Multi-agent, direct communication
    baselines["multi_agent_direct"] = ExperimentConfig(
        name="baseline_multi_agent_direct",
        description="Multiple agents with direct messaging",
        tasks=[
            TaskConfig(template="service_client", num_instances=10),
        ],
        agent_roles=[
            AgentRoleConfig(role_id="agent_service", provider_config=openai_config),
            AgentRoleConfig(role_id="agent_client", provider_config=openai_config),
        ],
        communication_pattern="direct",
    )
    
    # Baseline 4: Conflict resolution template
    baselines["conflict_resolution"] = ExperimentConfig(
        name="baseline_conflict_resolution",
        description="Testing conflict resolution with refactor collision template",
        tasks=[
            TaskConfig(template="refactor_collision", num_instances=10),
        ],
        agent_roles=[
            AgentRoleConfig(role_id="agent_refactor", provider_config=openai_config),
            AgentRoleConfig(role_id="agent_feature", provider_config=openai_config),
        ],
        communication_pattern="direct",
    )
    
    # Baseline 5: ScrumBan (Shared Backlog)
    baselines["scrumban_shared"] = ExperimentConfig(
        name="baseline_scrumban_shared",
        description="Multiple agents working from a shared ScrumBan backlog (Extreme Orchestration)",
        tasks=[
            TaskConfig(
                template="scrumban_service",
                params={"domain": "orchestrator_service", "num_subtasks": 5},
                num_instances=5,
                difficulty_levels=["hard"]
            ),
        ],
        agent_roles=[
            AgentRoleConfig(role_id="agent_1", provider_config=openai_config),
            AgentRoleConfig(role_id="agent_2", provider_config=openai_config),
        ],
        communication_pattern="direct",
        sprint_token_budget=2000000,
    )
    
    return baselines


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main entry point for running experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MACB Experiment Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run experiment
    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument("--config", type=str, help="Path to config YAML file")
    run_parser.add_argument("--baseline", type=str, help="Run a baseline config by name")
    run_parser.add_argument("--output", type=str, default="./experiments", help="Output directory")
    run_parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    run_parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    
    # List baselines
    list_parser = subparsers.add_parser("list-baselines", help="List available baseline configs")
    
    # Run ablation
    ablation_parser = subparsers.add_parser("ablation", help="Run ablation study")
    ablation_parser.add_argument("--type", choices=["communication", "model"], required=True)
    ablation_parser.add_argument("--config", type=str, help="Base config YAML")
    ablation_parser.add_argument("--baseline", type=str, help="Base baseline config name")
    
    # Example
    example_parser = subparsers.add_parser("example", help="Run example experiment")
    
    args = parser.parse_args()
    
    if args.command == "list-baselines":
        baselines = get_baseline_configs()
        print("\nAvailable baseline configurations:")
        print("-" * 50)
        for name, config in baselines.items():
            print(f"  {name}")
            print(f"    {config.description}")
        print()
    
    elif args.command == "run":
        if args.config:
            config = ExperimentConfig.from_yaml(args.config)
        elif args.baseline:
            baselines = get_baseline_configs()
            if args.baseline not in baselines:
                print(f"Unknown baseline: {args.baseline}")
                print(f"Available: {list(baselines.keys())}")
                return
            config = baselines[args.baseline]
        else:
            print("Must specify --config or --baseline")
            return
        
        config.output_dir = args.output
        config.parallel_workers = args.parallel
        
        runner = ExperimentRunner(config)
        results = runner.run(resume=not args.no_resume)
        
        print(f"\nResults saved to: {runner.output_dir}")
    
    elif args.command == "ablation":
        if args.config:
            base_config = ExperimentConfig.from_yaml(args.config)
        elif args.baseline:
            baselines = get_baseline_configs()
            base_config = baselines[args.baseline]
        else:
            print("Must specify --config or --baseline for ablation")
            return
        
        ablation = AblationRunner(base_config)
        
        if args.type == "communication":
            ablation.run_communication_ablation()
        elif args.type == "model":
            # Would need model specs
            print("Model ablation requires model specifications")
            return
        
        print(ablation.compare_results())
    
    elif args.command == "example":
        run_example()
    
    else:
        parser.print_help()


def run_example():
    """Run example experiment demonstrating the system."""
    print("=" * 70)
    print("MACB EXPERIMENT RUNNER - Example")
    print("=" * 70)
    
    # Create a minimal test config
    config = ExperimentConfig(
        name="example_experiment",
        description="Example experiment demonstrating the runner",
        tasks=[
            TaskConfig(
                template="service_client",
                num_instances=3,
                difficulty_levels=["easy", "medium"],
                base_seed=42,
            ),
        ],
        agent_roles=[
            AgentRoleConfig(
                role_id="agent_service",
                provider_config=ProviderConfig(type="mock"),
            ),
            AgentRoleConfig(
                role_id="agent_client",
                provider_config=ProviderConfig(type="mock"),
            ),
        ],
        communication_pattern="direct",
        max_total_turns=20,
        parallel_workers=1,
        output_dir="./example_output",
        checkpoint_interval=2,
    )
    
    print("\n1. Configuration:")
    print(f"   Name: {config.name}")
    print(f"   Tasks: {len(config.tasks)} task configs")
    print(f"   Agents: {len(config.agent_roles)} agent roles")
    print(f"   Communication: {config.communication_pattern}")
    
    print("\n2. Running experiment...")
    runner = ExperimentRunner(config)
    results = runner.run()
    
    print("\n3. Results by difficulty:")
    difficulty_breakdown = results.breakdown_by_difficulty()
    for diff, metrics in difficulty_breakdown.items():
        print(f"   {diff}: n={metrics['n']}, ICS={metrics['mean_ics']:.3f}, "
              f"success={metrics['success_rate']*100:.1f}%")
    
    print("\n4. Results by template:")
    template_breakdown = results.breakdown_by_template()
    for template, metrics in template_breakdown.items():
        print(f"   {template}: n={metrics['n']}, ICS={metrics['mean_ics']:.3f}")
    
    print(f"\n5. Full results saved to: {runner.output_dir}")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()