# Template Guide

SCRUMBENCH includes several task templates that test different aspects of multi-agent collaboration. This guide explains each template in detail.

## Overview

| Template | Complexity | Agents | Focus | Conflicts |
|----------|-----------|--------|-------|-----------|
| Service-Client | Basic | 2 | Integration | None |
| Refactor Collision | Intermediate | 2 | Conflict Resolution | Yes |
| ScrumBan Service | Advanced | 2-4 | Coordination | Optional |

## Service-Client Template

### Purpose

Tests **basic integration competence** - can agents implement complementary components that work together?

### Scenario

- **Agent 1 (Service)**: Implements a backend service conforming to an interface
- **Agent 2 (Client)**: Implements a client that consumes the service

Both agents see the shared interface but work independently. Success requires their implementations to integrate correctly.

### Example Task

**Domain**: User Cache Service

**Agent 1 Task**:
```
Implement the UserCacheService class in service/implementation.py.
Must implement all methods from UserCacheServiceInterface:
- get(key: str) -> Optional[User]
- set(key: str, user: User) -> None
- delete(key: str) -> bool
- list_keys() -> list[str]
```

**Agent 2 Task**:
```
Implement a CLI client in client/cli.py that uses UserCacheService.
Support commands: get, set, delete, list
Example: python client/cli.py set user1 "John Doe" john@example.com
```

### Evaluation

**Unit Tests**: Each component tested independently
- Service methods work correctly
- Client parsing and logic work

**Integration Tests**: Components tested together
- Client can call service methods
- Data flows correctly between components
- Edge cases handled properly

**ICS Calculation**:
```python
ICS = (unit_pass_rate + integration_pass_rate + build_success) / 3
```

### Difficulty Levels

**Easy**:
- 3-4 simple operations (get, set, delete, list)
- Basic data types
- Minimal edge cases

**Medium**:
- 5-6 operations with some complexity
- Nested data structures
- Error handling required

**Hard**:
- 7+ operations with dependencies
- Complex state management
- Atomic operations and consistency

### Domains

Available domains for this template:

1. **user_cache** - User data storage
2. **document_store** - Document management
3. **task_queue** - Task scheduling
4. **session_manager** - Session handling
5. **config_service** - Configuration management

See [Domains Guide](domains.md) for full specifications.

### Configuration

```python
TaskConfig(
    template="service_client",
    params={
        "domain": "user_cache",
        "client_type": "cli",  # or "api", "sdk"
        "difficulty": "medium",
        "num_operations": 5,
        "include_edge_cases": True
    },
    num_instances=10,
    difficulty_levels=["easy", "medium", "hard"]
)
```

---

## Refactor Collision Template

### Purpose

Tests **conflict resolution** - can agents detect and resolve merge conflicts when their work overlaps?

### Scenario

- **Agent 1 (Refactor)**: Refactors existing code (rename functions, restructure)
- **Agent 2 (Feature)**: Adds new features to the same code

Both agents modify the same files, creating merge conflicts that must be resolved.

### Example Task

**Starting Code**: Legacy payment processor

**Agent 1 Task**:
```
Refactor the payment processing code:
1. Rename process_payment() to execute_transaction()
2. Extract validation logic into validate_payment_data()
3. Improve error handling
```

**Agent 2 Task**:
```
Add new payment features:
1. Support for recurring payments
2. Add refund capability
3. Implement payment history tracking
```

### Conflict Types

**Textual Conflicts** (Git-level):
- Both agents modify the same lines
- Git merge fails
- Requires manual resolution

**Semantic Conflicts** (Logic-level):
- No Git conflict, but code doesn't work together
- Example: Agent 1 renames a function, Agent 2 calls old name
- Harder to detect, requires testing

**Structural Conflicts** (Architecture-level):
- Different approaches to solving the same problem
- Example: Agent 1 uses classes, Agent 2 uses functions
- Requires design decisions

### Evaluation

**Conflict Detection**:
- Are conflicts identified correctly?
- Both textual and semantic conflicts counted

**Conflict Resolution**:
- Are conflicts resolved successfully?
- Does code build after resolution?
- Do all tests pass?

**CRS Calculation**:
```python
CRS = conflicts_resolved / max(conflicts_detected, 1)
```

### Configuration

```python
TaskConfig(
    template="refactor_collision",
    params={
        "domain": "payment_processor",
        "refactor_type": "rename",  # or "extract", "restructure"
        "feature_complexity": "medium",
        "expected_conflicts": 3
    },
    num_instances=5
)
```

---

## ScrumBan Service Template

### Purpose

Tests **advanced coordination** - can agents work from a shared backlog using Kanban-style task management?

### Scenario

- **Multiple Agents** (2-4): Work collaboratively on a shared backlog
- **Kanban Board**: Tasks move through Todo → In Progress → Done
- **Tool-Based Coordination**: Agents use tools to claim tasks and communicate

### Example Task

**Backlog** (5 subtasks):
1. Implement user authentication
2. Add password hashing
3. Create session management
4. Implement logout functionality
5. Add token refresh

**Agent Workflow**:
1. Call `read_kanban_board()` to see available tasks
2. Call `claim_task(task_id)` to take ownership
3. Implement the feature
4. Call `update_task_status(task_id, "done")` when complete
5. Optionally `send_message(recipient, message)` to coordinate

### Coordination Patterns

**Isolated** (Baseline):
- Agents cannot communicate
- Tests pure task selection and completion

**Direct Messaging**:
- Agents can send messages to each other
- Tests communication and coordination

**Shared Backlog** (Full ScrumBan):
- Agents see real-time board updates
- Can coordinate on task dependencies
- Tests advanced collaboration

### Evaluation

**Task Completion**:
- How many backlog items completed?
- Were tasks completed in logical order?

**Coordination Quality**:
- Did agents avoid duplicate work?
- Were dependencies respected?
- Was communication effective?

**Integration**:
- Do all completed features work together?
- Standard ICS calculation applies

### Domains

Special domains for ScrumBan:

1. **orchestrator_service** - Service orchestration (Extreme complexity)
2. **cryptographic_vault** - Secure key management (Extreme complexity)
3. **fintech_backend** - Financial transaction system
4. **ecommerce_platform** - E-commerce features

### Configuration

```python
TaskConfig(
    template="scrumban_service",
    params={
        "domain": "orchestrator_service",
        "num_subtasks": 5,
        "difficulty": "hard",
        "num_agents": 2
    },
    num_instances=3,
    difficulty_levels=["hard"]
)
```

---

## Choosing the Right Template

### For Basic Integration Testing
→ **Service-Client Template**
- Simple to understand
- Clear success criteria
- Good for baseline evaluation

### For Conflict Resolution Research
→ **Refactor Collision Template**
- Tests merge conflict handling
- Evaluates semantic understanding
- Good for studying agent robustness

### For Coordination Research
→ **ScrumBan Service Template**
- Tests multi-agent coordination
- Evaluates task planning
- Good for studying collaboration strategies

---

## Creating Custom Templates

See [Custom Experiments](custom-experiments.md) for how to create your own templates.

### Template Interface

```python
class MyCustomTemplate(TaskTemplate):
    name = "my_custom"
    description = "Brief description"
    
    def get_default_params(self) -> dict:
        return {"param1": "default"}
    
    def get_param_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string"}
            }
        }
    
    def generate(self, params: dict, seed: int) -> TaskInstance:
        # Generate task instance
        pass
```

---

## Next Steps

- **[Scoring Methodology](scoring.md)** - Understand ICS and CRS calculations
- **[Dashboard Guide](dashboard.md)** - Visualize experiments in real-time
- **[Custom Experiments](custom-experiments.md)** - Create your own configurations

---

**Previous**: [Getting Started](getting-started.md) | **Next**: [Scoring Methodology](scoring.md) →
