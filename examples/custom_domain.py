"""
Example: Adding a Custom Domain to SCRUMBENCH
==============================================

This example shows how to add a new domain (Blog Platform) to the benchmark.
Domains define the problem space for tasks - the data types, operations, and constraints.

Usage:
    1. Copy this file to your project
    2. Modify the domain specification
    3. Import and use in your experiments
"""

from core.benchmark import DomainSpec, DOMAINS

# Define the domain specification
blog_platform_domain = DomainSpec(
    name="blog_platform",
    data_type_name="BlogPost",
    
    # Define the core data structure
    fields=[
        ("id", "str", "Unique post identifier"),
        ("title", "str", "Post title"),
        ("content", "str", "Post content (markdown)"),
        ("author_id", "str", "Author identifier"),
        ("tags", "list[str]", "Post tags"),
        ("published_at", "Optional[datetime]", "Publication timestamp"),
        ("is_draft", "bool", "Draft status"),
    ],
    
    # Define operations that agents must implement
    operations=[
        {
            "name": "create_post",
            "params": [
                ("title", "str"),
                ("content", "str"),
                ("author_id", "str"),
                ("tags", "list[str]", "[]"),
            ],
            "returns": "BlogPost",
            "description": "Create a new blog post in draft status"
        },
        {
            "name": "publish_post",
            "params": [("post_id", "str")],
            "returns": "bool",
            "description": "Publish a draft post (sets published_at, is_draft=False)"
        },
        {
            "name": "get_post",
            "params": [("post_id", "str")],
            "returns": "Optional[BlogPost]",
            "description": "Retrieve a post by ID"
        },
        {
            "name": "update_post",
            "params": [
                ("post_id", "str"),
                ("title", "Optional[str]", "None"),
                ("content", "Optional[str]", "None"),
                ("tags", "Optional[list[str]]", "None"),
            ],
            "returns": "bool",
            "description": "Update post fields (only if draft)"
        },
        {
            "name": "delete_post",
            "params": [("post_id", "str")],
            "returns": "bool",
            "description": "Delete a post"
        },
        {
            "name": "list_posts",
            "params": [
                ("author_id", "Optional[str]", "None"),
                ("tag", "Optional[str]", "None"),
                ("include_drafts", "bool", "False"),
            ],
            "returns": "list[BlogPost]",
            "description": "List posts with optional filters"
        },
        {
            "name": "search_posts",
            "params": [("query", "str")],
            "returns": "list[BlogPost]",
            "description": "Search posts by title/content"
        },
    ],
    
    # Define edge cases to test
    edge_cases=[
        "Publishing an already published post",
        "Updating a published post (should fail)",
        "Deleting a post that doesn't exist",
        "Searching with empty query",
        "Creating post with empty title",
        "Listing posts with no results",
    ],
    
    # Define client types
    client_types=["cli", "api", "sdk"]
)

# Register the domain
DOMAINS["blog_platform"] = blog_platform_domain

# Now you can use it in experiments!
if __name__ == "__main__":
    from core.runner import ExperimentConfig, TaskConfig, AgentRoleConfig, ProviderConfig
    
    # Create an experiment using the new domain
    config = ExperimentConfig(
        name="blog_platform_test",
        description="Test agents on blog platform implementation",
        tasks=[
            TaskConfig(
                template="service_client",
                params={
                    "domain": "blog_platform",
                    "client_type": "cli",
                    "difficulty": "medium",
                    "num_operations": 5,
                },
                num_instances=5,
                difficulty_levels=["easy", "medium"]
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
    
    # Run the experiment
    from core.runner import ExperimentRunner
    runner = ExperimentRunner(config)
    results = runner.run()
    
    print(f"Completed {len(results.task_results)} tasks")
    print(f"Mean ICS: {results.mean_ics:.3f}")
