"""
Example: Using Anthropic Claude with SCRUMBENCH
================================================

This example shows how to configure experiments to use Anthropic's Claude
instead of OpenAI's GPT models.

Usage:
    1. Set ANTHROPIC_API_KEY in your .env file
    2. Run this script to test Claude on the benchmark
"""

from core.runner import (
    ExperimentConfig,
    TaskConfig,
    AgentRoleConfig,
    ProviderConfig,
    ExperimentRunner
)


def create_anthropic_config():
    """Create experiment config using Claude."""
    
    # Configure Anthropic provider
    anthropic_config = ProviderConfig(
        type="anthropic",
        model="claude-3-5-sonnet-20241022",  # or claude-3-opus-20240229
        temperature=0.7,
        max_tokens=4096,
    )
    
    # Create experiment with Anthropic agents
    config = ExperimentConfig(
        name="anthropic_baseline",
        description="Single agent using Claude 3.5 Sonnet",
        tasks=[
            TaskConfig(
                template="service_client",
                params={
                    "domain": "user_cache",
                    "client_type": "cli",
                    "difficulty": "medium",
                },
                num_instances=5,
                difficulty_levels=["easy", "medium"]
            )
        ],
        agent_roles=[
            AgentRoleConfig(
                role_id="agent_service",
                provider_config=anthropic_config
            ),
            AgentRoleConfig(
                role_id="agent_client",
                provider_config=anthropic_config
            ),
        ],
        communication_pattern="isolated",
        output_dir="./outputs",
    )
    
    return config


def create_mixed_config():
    """Create experiment with mixed providers (GPT-4 + Claude)."""
    
    openai_config = ProviderConfig(
        type="openai",
        model="gpt-4o",
        temperature=0.7,
    )
    
    anthropic_config = ProviderConfig(
        type="anthropic",
        model="claude-3-5-sonnet-20241022",
        temperature=0.7,
    )
    
    # GPT-4 for service, Claude for client
    config = ExperimentConfig(
        name="mixed_providers",
        description="GPT-4o (service) vs Claude 3.5 (client)",
        tasks=[
            TaskConfig(
                template="service_client",
                params={"domain": "user_cache"},
                num_instances=5,
            )
        ],
        agent_roles=[
            AgentRoleConfig(
                role_id="agent_service",
                provider_config=openai_config
            ),
            AgentRoleConfig(
                role_id="agent_client",
                provider_config=anthropic_config
            ),
        ],
        communication_pattern="isolated",
    )
    
    return config


def compare_providers():
    """Run experiments comparing OpenAI and Anthropic."""
    
    # OpenAI baseline
    openai_config = ExperimentConfig(
        name="openai_baseline",
        description="GPT-4o baseline",
        tasks=[
            TaskConfig(
                template="service_client",
                params={"domain": "user_cache"},
                num_instances=10,
            )
        ],
        agent_roles=[
            AgentRoleConfig(
                role_id="agent_service",
                provider_config=ProviderConfig(type="openai", model="gpt-4o")
            ),
            AgentRoleConfig(
                role_id="agent_client",
                provider_config=ProviderConfig(type="openai", model="gpt-4o")
            ),
        ],
        communication_pattern="isolated",
    )
    
    # Anthropic baseline
    anthropic_config = create_anthropic_config()
    anthropic_config.name = "anthropic_baseline"
    anthropic_config.tasks[0].num_instances = 10
    
    # Run both
    print("Running OpenAI baseline...")
    openai_runner = ExperimentRunner(openai_config)
    openai_results = openai_runner.run()
    
    print("\nRunning Anthropic baseline...")
    anthropic_runner = ExperimentRunner(anthropic_config)
    anthropic_results = anthropic_runner.run()
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"\nOpenAI (GPT-4o):")
    print(f"  Mean ICS: {openai_results.mean_ics:.3f}")
    print(f"  Success Rate: {openai_results.successful_tasks}/{openai_results.total_tasks}")
    print(f"  Mean Tokens: {openai_results.mean_tokens:.0f}")
    print(f"  Mean Duration: {openai_results.mean_duration:.1f}s")
    
    print(f"\nAnthropic (Claude 3.5 Sonnet):")
    print(f"  Mean ICS: {anthropic_results.mean_ics:.3f}")
    print(f"  Success Rate: {anthropic_results.successful_tasks}/{anthropic_results.total_tasks}")
    print(f"  Mean Tokens: {anthropic_results.mean_tokens:.0f}")
    print(f"  Mean Duration: {anthropic_results.mean_duration:.1f}s")
    
    # Calculate relative performance
    ics_diff = ((anthropic_results.mean_ics - openai_results.mean_ics) / 
                openai_results.mean_ics * 100)
    print(f"\nRelative ICS: {ics_diff:+.1f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SCRUMBENCH with Anthropic")
    parser.add_argument(
        "--mode",
        choices=["anthropic", "mixed", "compare"],
        default="anthropic",
        help="Experiment mode"
    )
    
    args = parser.parse_args()
    
    if args.mode == "anthropic":
        print("Running experiment with Claude...")
        config = create_anthropic_config()
        runner = ExperimentRunner(config)
        results = runner.run()
        print(f"\nMean ICS: {results.mean_ics:.3f}")
        
    elif args.mode == "mixed":
        print("Running experiment with mixed providers...")
        config = create_mixed_config()
        runner = ExperimentRunner(config)
        results = runner.run()
        print(f"\nMean ICS: {results.mean_ics:.3f}")
        
    elif args.mode == "compare":
        print("Comparing OpenAI vs Anthropic...")
        compare_providers()
