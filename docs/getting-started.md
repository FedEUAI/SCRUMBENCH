# Getting Started with SCRUMBENCH

This guide will walk you through installing SCRUMBENCH, running your first experiment, and understanding the results.

## Prerequisites

- **Python 3.11+** (Python 3.12 recommended)
- **API Keys**: OpenAI or Anthropic API key
- **System**: Linux, macOS, or Windows with WSL

## Installation

### Option 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/FedEUAI/SCRUMBENCH.git
cd scrumbench

# Install dependencies
uv sync
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/FedEUAI/SCRUMBENCH.git
cd scrumbench

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Configuration

### Set Up API Keys

```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your favorite editor
nano .env
```

Add your API key:

```bash
# For OpenAI
OPENAI_API_KEY=sk-your-key-here

# Or for Anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Verify Installation

```bash
# Check that the CLI works
uv run python -m core.runner list-baselines

# You should see a list of available baseline experiments
```

## Running Your First Experiment

### Option 1: Using the CLI

Run a simple single-agent baseline:

```bash
uv run python -m core.runner run \
  --baseline single_agent \
  --output ./outputs
```

This will:
1. Generate 10 task instances
2. Run a single agent on each task
3. Evaluate integration competence
4. Save results to `./outputs/baseline_single_agent/`

### Option 2: Using the Dashboard

The dashboard provides real-time visualization of experiments:

```bash
# Start the dashboard server
uv run python api.py

# Open your browser to http://localhost:8000
```

From the dashboard:
1. Click **"Run Experiment"** in the sidebar
2. Select a baseline (e.g., "Baseline Single Agent")
3. Watch the experiment run in real-time!

## Understanding the Results

### Output Structure

After running an experiment, you'll find:

```
outputs/
└── baseline_single_agent/
    ├── config.json              # Experiment configuration
    ├── results.json             # Detailed results
    ├── summary.txt              # Human-readable summary
    ├── task_manifest.json       # List of all tasks
    └── workspaces/              # Individual task workspaces
        └── service_client_user_cache_42/
            ├── trace.json       # Agent interaction trace
            ├── diagnostics.json # Test results and metrics
            ├── metadata.json    # Task metadata
            └── starter/         # Initial code
                └── ...
```

### Reading Results

**`results.json`** contains:
- Overall experiment statistics
- Per-task results with ICS and CRS scores
- Token usage and timing information

**`summary.txt`** provides:
- Quick overview of success rates
- Mean ICS and CRS scores
- Performance statistics

Example summary:

```
=== Experiment Results ===
Experiment: baseline_single_agent
Total Tasks: 10
Successful: 8 (80.0%)
Failed: 2 (20.0%)

Mean ICS: 0.75
Mean CRS: 1.00
Mean Duration: 45.3s
Mean Turns: 12.4
```

### Key Metrics

**Integration Competence Score (ICS)**:
- Measures how well agents' code integrates
- Range: 0.0 (complete failure) to 1.0 (perfect integration)
- Based on unit tests, integration tests, and build success

**Conflict Resolution Score (CRS)**:
- Measures ability to resolve merge conflicts
- Range: 0.0 (no conflicts resolved) to 1.0 (all resolved)
- Only applicable to templates with expected conflicts

See [Scoring Methodology](scoring.md) for detailed calculation.

## Exploring Different Templates

### Service-Client Template

Tests basic integration without conflicts:

```bash
uv run python -m core.runner run \
  --baseline single_agent \
  --output ./outputs
```

### Multi-Agent Collaboration

Test agents working in isolation:

```bash
uv run python -m core.runner run \
  --baseline multi_agent_isolated \
  --output ./outputs
```

### ScrumBan Coordination

Advanced template with shared backlog:

```bash
uv run python -m core.runner run \
  --baseline scrumban_shared \
  --output ./outputs
```

## Next Steps

- **[Template Guide](templates.md)** - Deep dive into each template
- **[Scoring Methodology](scoring.md)** - Understand ICS and CRS calculations
- **[Dashboard Guide](dashboard.md)** - Master the real-time visualization
- **[Custom Experiments](custom-experiments.md)** - Create your own configurations

## Troubleshooting

### API Key Issues

**Error**: `openai.error.AuthenticationError`

**Solution**: Verify your API key in `.env`:
```bash
cat .env | grep OPENAI_API_KEY
```

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'core'`

**Solution**: Install in development mode:
```bash
pip install -e .
```

### Dashboard Not Loading

**Error**: Dashboard shows blank page

**Solution**: Check console for errors and ensure API is running:
```bash
# Check if API is running
curl http://localhost:8000/experiments

# Restart if needed
uv run python api.py
```

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/FedEUAI/SCRUMBENCH/issues)
- **Discussions**: [GitHub Discussions](https://github.com/FedEUAI/SCRUMBENCH/discussions)
- **Email**: maintainers@scrumbench.org

---

**Next**: [Template Guide](templates.md) →
