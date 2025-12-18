# SCRUMBENCH: Multi-Agent Collaboration Benchmark

**SCRUMBENCH** is a comprehensive benchmark for evaluating AI agents on realistic software engineering tasks that require multi-agent coordination, conflict resolution, and integration competence.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## 🎯 Overview

Modern software development increasingly involves multiple agents working together on complex tasks. SCRUMBENCH evaluates how well AI agents can:

- **Collaborate** on shared codebases without explicit coordination
- **Resolve conflicts** when changes overlap or interfere
- **Integrate work** from multiple contributors into a cohesive solution
- **Communicate** effectively to coordinate complex workflows

### Key Features

- **📊 Novel Metrics**: Integration Competence Score (ICS) and Conflict Resolution Score (CRS)
- **🎭 Multiple Templates**: Service-client, refactor collision, ScrumBan orchestration
- **🤝 Communication Patterns**: Isolated, direct messaging, shared backlog
- **📈 Real-time Dashboard**: Live visualization of experiment progress
- **🔧 Extensible**: Easy to add custom domains, templates, and evaluation criteria

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/FedEUAI/SCRUMBENCH.git
cd scrumbench

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Set Up API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
```

### Run Your First Experiment

```bash
# Run a baseline experiment
uv run python -m core.runner run --baseline single_agent --output ./outputs

# Or start the dashboard
uv run python api.py
# Open http://localhost:8000 in your browser
```

## 📚 Documentation

- **[Getting Started Guide](docs/getting-started.md)** - Detailed installation and first steps
- **[Template Guide](docs/templates.md)** - Understanding each benchmark template
- **[Scoring Methodology](docs/scoring.md)** - How ICS and CRS are calculated

## 🎨 Templates

SCRUMBENCH includes several task templates of varying complexity:

### 1. Service-Client Template
**Complexity**: Basic  
**Agents**: 2 (service, client)  
**Focus**: Basic integration without conflicts

Agents implement complementary components (service backend + client) that must integrate correctly.

### 2. Refactor Collision Template
**Complexity**: Intermediate  
**Agents**: 2 (refactor, feature)  
**Focus**: Conflict detection and resolution

One agent refactors code while another adds features, creating merge conflicts that must be resolved.

### 3. ScrumBan Service Template
**Complexity**: Advanced  
**Agents**: 2-4  
**Focus**: Coordination via shared backlog

Multiple agents work from a shared Kanban board, claiming tasks and coordinating through tool calls.

## 📊 Evaluation Metrics

### Integration Competence Score (ICS)
Measures how well agents' work integrates:
- **Unit tests**: Individual component correctness
- **Integration tests**: Cross-component functionality
- **Build success**: Code compiles/runs without errors

**Formula**: `ICS = (unit_pass_rate + integration_pass_rate + build_success) / 3`

### Conflict Resolution Score (CRS)
Measures conflict handling ability:
- **Detection**: Identifying merge conflicts
- **Resolution**: Successfully resolving conflicts
- **Semantic correctness**: Maintaining functionality after resolution

**Formula**: `CRS = conflicts_resolved / max(conflicts_detected, 1)`

## 🖥️ Dashboard

SCRUMBENCH includes a real-time dashboard for monitoring experiments:

- **Live Progress**: Watch tasks complete in real-time
- **Kanban Visualization**: See agent coordination on shared backlogs
- **Trace Inspection**: Step through agent actions turn-by-turn
- **Metrics**: ICS, CRS, token usage, and more

![Dashboard Screenshot](docs/images/dashboard.png)

## 🔧 Custom Experiments

Create custom experiment configurations:

```python
from core.runner import ExperimentConfig, TaskConfig, AgentRoleConfig

config = ExperimentConfig(
    name="my_experiment",
    description="Custom multi-agent experiment",
    tasks=[
        TaskConfig(
            template="service_client",
            params={"domain": "user_cache"},
            num_instances=10,
            difficulty_levels=["easy", "medium", "hard"]
        )
    ],
    agent_roles=[
        AgentRoleConfig(role_id="agent_1", provider_config=...),
        AgentRoleConfig(role_id="agent_2", provider_config=...)
    ],
    communication_pattern="direct"
)
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Adding New Templates

1. Extend `TaskTemplate` in `core/benchmark.py`
2. Implement `generate()` method
3. Add to template registry
4. Write tests

### Adding New Domains

1. Define domain spec in `DOMAINS` dict
2. Specify data types, operations, and constraints
3. Add domain-specific test generators

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use SCRUMBENCH in your research, please cite:

```bibtex
@misc{scrumbench2024,
  title={SCRUMBENCH: A Multi-Agent Collaboration Benchmark for Software Engineering},
  author={Your Name},
  year={2024},
  url={https://github.com/FedEUAI/SCRUMBENCH}
}
```

## 🙏 Acknowledgments

- Inspired by SWE-bench and other software engineering benchmarks
- Built with FastAPI, React, and modern AI frameworks

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/FedEUAI/SCRUMBENCH/issues)
- **Discussions**: [GitHub Discussions](https://github.com/FedEUAI/SCRUMBENCH/discussions)
- **Email**: your.email@example.com

---

**Status**: 🚧 Active Development | **Version**: 0.1.0 | **Last Updated**: December 2024
