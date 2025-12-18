# Scoring Methodology

SCRUMBENCH uses two primary metrics to evaluate multi-agent collaboration: **Integration Competence Score (ICS)** and **Conflict Resolution Score (CRS)**.

## Integration Competence Score (ICS)

### Definition

ICS measures how well agents' independently developed code integrates into a working system.

### Formula

```
ICS = (unit_pass_rate + integration_pass_rate + build_success) / 3
```

Where:
- **unit_pass_rate**: Percentage of unit tests passing (0.0 to 1.0)
- **integration_pass_rate**: Percentage of integration tests passing (0.0 to 1.0)
- **build_success**: Binary indicator if code builds/runs (0.0 or 1.0)

### Components

#### 1. Unit Test Pass Rate

**Purpose**: Verify individual components work correctly in isolation.

**Calculation**:
```python
unit_pass_rate = unit_tests_passed / unit_tests_total
```

**Example**:
```
Service tests: 8/10 passed = 0.80
Client tests: 7/10 passed = 0.70
Combined: 15/20 passed = 0.75
```

#### 2. Integration Test Pass Rate

**Purpose**: Verify components work together correctly.

**Calculation**:
```python
integration_pass_rate = integration_tests_passed / integration_tests_total
```

**Example**:
```
End-to-end tests: 3/4 passed = 0.75
```

#### 3. Build Success

**Purpose**: Verify code compiles/runs without errors.

**Calculation**:
```python
build_success = 1.0 if no_import_errors and no_syntax_errors else 0.0
```

**Checks**:
- No syntax errors
- No import errors
- No missing dependencies
- Code can be executed

### Example Calculation

**Scenario**: Service-Client task

**Results**:
- Unit tests: 15/20 passed = 0.75
- Integration tests: 3/4 passed = 0.75
- Build: Success = 1.0

**ICS**:
```
ICS = (0.75 + 0.75 + 1.0) / 3 = 0.833
```

### Interpretation

| ICS Range | Interpretation | Meaning |
|-----------|---------------|---------|
| 0.90 - 1.00 | Excellent | Near-perfect integration |
| 0.75 - 0.89 | Good | Minor integration issues |
| 0.50 - 0.74 | Fair | Significant integration problems |
| 0.25 - 0.49 | Poor | Major integration failures |
| 0.00 - 0.24 | Failed | Complete integration failure |

### Why ICS Matters

Traditional benchmarks measure individual agent performance. ICS measures **emergent system quality** - the ability of independently developed components to work together.

**Key Insight**: High individual performance doesn't guarantee high ICS. Agents must:
- Follow interface contracts precisely
- Handle edge cases consistently
- Make compatible design decisions

---

## Conflict Resolution Score (CRS)

### Definition

CRS measures an agent's ability to detect and resolve merge conflicts when working on shared code.

### Formula

```
CRS = conflicts_resolved / max(conflicts_detected, 1)
```

Where:
- **conflicts_resolved**: Number of conflicts successfully resolved
- **conflicts_detected**: Total number of conflicts identified

### Conflict Types

#### 1. Textual Conflicts (Git-level)

**Definition**: Git merge fails due to overlapping changes.

**Example**:
```python
# Agent 1's version
def process_payment(amount):
    validate_amount(amount)
    return execute_transaction(amount)

# Agent 2's version
def process_payment(amount):
    log_payment_attempt(amount)
    return charge_customer(amount)

# Conflict: Both modified the same function
```

**Detection**: Git merge command fails

#### 2. Semantic Conflicts (Logic-level)

**Definition**: No Git conflict, but code doesn't work together.

**Example**:
```python
# Agent 1 renames function
def execute_transaction(amount):
    ...

# Agent 2 calls old name
result = process_payment(100)  # NameError!
```

**Detection**: Tests fail after merge

#### 3. Structural Conflicts (Architecture-level)

**Definition**: Incompatible design approaches.

**Example**:
```python
# Agent 1 uses classes
class PaymentProcessor:
    def process(self, payment):
        ...

# Agent 2 uses functions
def process_payment(payment):
    ...
```

**Detection**: Requires manual review

### Resolution Criteria

A conflict is considered **resolved** if:
1. Code builds successfully
2. All tests pass
3. No functionality is lost
4. Code quality is maintained

### Example Calculation

**Scenario**: Refactor Collision task

**Results**:
- Textual conflicts detected: 3
- Semantic conflicts detected: 2
- Total conflicts: 5
- Successfully resolved: 4

**CRS**:
```
CRS = 4 / 5 = 0.80
```

### Interpretation

| CRS Range | Interpretation | Meaning |
|-----------|---------------|---------|
| 1.00 | Perfect | All conflicts resolved |
| 0.75 - 0.99 | Good | Most conflicts resolved |
| 0.50 - 0.74 | Fair | Some conflicts unresolved |
| 0.25 - 0.49 | Poor | Many conflicts unresolved |
| 0.00 - 0.24 | Failed | Few/no conflicts resolved |

### Why CRS Matters

Real-world software development involves frequent merge conflicts. CRS measures:
- **Conflict awareness**: Can agents detect conflicts?
- **Resolution skill**: Can agents fix conflicts correctly?
- **Code understanding**: Do agents understand semantic implications?

---

## Additional Metrics

### Token Usage

**Purpose**: Measure computational cost.

**Metrics**:
- **total_tokens**: Prompt + completion tokens
- **mean_tokens**: Average per task
- **tokens_per_turn**: Efficiency metric

### Turn Count

**Purpose**: Measure interaction complexity.

**Metrics**:
- **total_turns**: Number of agent actions
- **mean_turns**: Average per task
- **turns_per_success**: Efficiency for successful tasks

### Duration

**Purpose**: Measure time cost.

**Metrics**:
- **total_duration**: Wall-clock time
- **mean_duration**: Average per task
- **duration_per_success**: Time for successful tasks

---

## Aggregate Statistics

### Experiment-Level Metrics

**Success Rate**:
```python
success_rate = successful_tasks / total_tasks
```

**Mean ICS**:
```python
mean_ics = sum(task.ics for task in tasks) / len(tasks)
```

**Standard Deviation**:
```python
std_ics = sqrt(sum((task.ics - mean_ics)^2) / len(tasks))
```

### Comparison Across Experiments

**Relative Performance**:
```python
relative_ics = (experiment_ics - baseline_ics) / baseline_ics
```

**Statistical Significance**:
- Use t-test for comparing mean ICS
- Use chi-square for comparing success rates

---

## Scoring Examples

### Example 1: Perfect Integration

**Task**: Service-Client (Easy)

**Results**:
- Unit tests: 10/10 = 1.0
- Integration tests: 4/4 = 1.0
- Build: Success = 1.0

**ICS**: (1.0 + 1.0 + 1.0) / 3 = **1.0**

**Interpretation**: Perfect integration, all components work flawlessly together.

### Example 2: Partial Integration

**Task**: Service-Client (Medium)

**Results**:
- Unit tests: 12/15 = 0.80
- Integration tests: 2/5 = 0.40
- Build: Success = 1.0

**ICS**: (0.80 + 0.40 + 1.0) / 3 = **0.733**

**Interpretation**: Components work individually but have integration issues.

### Example 3: Build Failure

**Task**: Service-Client (Hard)

**Results**:
- Unit tests: 8/10 = 0.80 (tested separately)
- Integration tests: 0/6 = 0.0 (can't run)
- Build: Failure = 0.0

**ICS**: (0.80 + 0.0 + 0.0) / 3 = **0.267**

**Interpretation**: Code doesn't integrate at all, likely import errors.

### Example 4: Conflict Resolution

**Task**: Refactor Collision

**Results**:
- Conflicts detected: 4 (3 textual, 1 semantic)
- Conflicts resolved: 3 (all textual)
- Unresolved: 1 (semantic conflict)

**CRS**: 3 / 4 = **0.75**

**Interpretation**: Good at textual conflicts, missed semantic issue.

---

## Best Practices

### For Researchers

1. **Report both ICS and CRS** - They measure different capabilities
2. **Include standard deviation** - Shows consistency
3. **Compare across difficulty levels** - Shows scaling behavior
4. **Report token/time costs** - Important for practical deployment

### For Developers

1. **Focus on integration tests** - They matter most for ICS
2. **Test edge cases** - Often where integration fails
3. **Use type hints** - Helps catch semantic conflicts
4. **Write clear interfaces** - Reduces integration issues

---

## Next Steps

- **[Dashboard Guide](dashboard.md)** - Visualize these metrics in real-time
- **[Custom Experiments](custom-experiments.md)** - Design experiments targeting specific metrics

---

**Previous**: [Template Guide](templates.md) | **Next**: [Dashboard Guide](dashboard.md) →
