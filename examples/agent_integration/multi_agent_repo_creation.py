"""
Multi-Agent Repository Creation Example

Demonstrates how multiple agents can coordinate to create repositories in parallel
using Modal for compute scaling.
"""

import modal
from typing import List, Dict, Any
import json
import os

app = modal.App("multi-agent-repo-creation")

# Shared volume for coordinating between agents
shared_volume = modal.Volume.from_name("agent-coordination", create_if_missing=True)

base_image = modal.Image.debian_slim().pip_install(
    "pydantic",
    "httpx"
)


@app.function(
    image=base_image,
    volumes={"/shared": shared_volume}
)
def create_repository_parallel(
    repo_config: Dict[str, Any],
    agent_id: str
) -> Dict[str, Any]:
    """
    Create a repository configuration in parallel.
    
    Multiple agents can call this simultaneously to create different repos.
    
    Args:
        repo_config: Dictionary with repo configuration (name, description, files, etc.)
        agent_id: ID of the agent creating this repo
    
    Returns:
        Dictionary with creation result
    """
    try:
        repo_name = repo_config.get("name", "unnamed-repo")
        
        # Create repo structure
        repo_dir = f"/shared/repos/{repo_name}"
        os.makedirs(repo_dir, exist_ok=True)
        
        # Write repo config
        config_path = f"{repo_dir}/config.json"
        with open(config_path, "w") as f:
            json.dump(repo_config, f, indent=2)
        
        # Create README if provided
        if "readme" in repo_config:
            readme_path = f"{repo_dir}/README.md"
            with open(readme_path, "w") as f:
                f.write(repo_config["readme"])
        
        # Create files if provided
        if "files" in repo_config:
            for file_path, file_content in repo_config["files"].items():
                full_path = f"{repo_dir}/{file_path}"
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(file_content)
        
        # Update coordination log
        log_path = "/shared/repo_creation_log.json"
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                log = json.load(f)
        else:
            log = []
        
        log.append({
            "agent_id": agent_id,
            "repo_name": repo_name,
            "status": "created",
            "timestamp": "2024-01-01T00:00:00Z"
        })
        
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        
        # Commit volume changes
        shared_volume.commit()
        
        return {
            "status": "success",
            "agent_id": agent_id,
            "repo_name": repo_name,
            "message": f"Repository {repo_name} created successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "agent_id": agent_id,
            "error": str(e)
        }


@app.function(
    image=base_image,
    volumes={"/shared": shared_volume}
)
def get_creation_status() -> Dict[str, Any]:
    """Get status of all repository creations"""
    log_path = "/shared/repo_creation_log.json"
    
    if not os.path.exists(log_path):
        return {
            "status": "success",
            "repos": [],
            "count": 0
        }
    
    with open(log_path, "r") as f:
        log = json.load(f)
    
    return {
        "status": "success",
        "repos": log,
        "count": len(log)
    }


@app.function(image=base_image)
def coordinate_repo_creation(
    repo_configs: List[Dict[str, Any]],
    agent_ids: List[str]
) -> Dict[str, Any]:
    """
    Coordinate parallel repository creation across multiple agents.
    
    Args:
        repo_configs: List of repository configurations
        agent_ids: List of agent IDs (one per repo config)
    
    Returns:
        Dictionary with results from all creations
    """
    if len(repo_configs) != len(agent_ids):
        return {
            "status": "error",
            "message": "repo_configs and agent_ids must have same length"
        }
    
    # Execute all repo creations in parallel
    results = list(create_repository_parallel.map(
        repo_configs,
        agent_ids
    ))
    
    # Aggregate results
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "error"]
    
    return {
        "status": "completed",
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "results": results
    }


@app.local_entrypoint()
def example_multi_agent_repo_creation():
    """Example of multi-agent repository creation"""
    
    # Example: Three agents creating three different repositories in parallel
    repo_configs = [
        {
            "name": "frontend-app",
            "description": "Frontend application",
            "readme": "# Frontend App\n\nA modern frontend application.",
            "files": {
                "package.json": '{"name": "frontend-app", "version": "1.0.0"}',
                "src/index.js": "console.log('Hello from frontend');"
            }
        },
        {
            "name": "backend-api",
            "description": "Backend API service",
            "readme": "# Backend API\n\nRESTful API service.",
            "files": {
                "requirements.txt": "fastapi==0.100.0\nuvicorn==0.23.0",
                "main.py": "from fastapi import FastAPI\napp = FastAPI()"
            }
        },
        {
            "name": "data-pipeline",
            "description": "Data processing pipeline",
            "readme": "# Data Pipeline\n\nETL pipeline for data processing.",
            "files": {
                "requirements.txt": "pandas==2.0.0\nnumpy==1.24.0",
                "pipeline.py": "import pandas as pd\nprint('Pipeline ready')"
            }
        }
    ]
    
    agent_ids = ["agent-1", "agent-2", "agent-3"]
    
    print("Creating repositories in parallel with multiple agents...")
    result = coordinate_repo_creation.remote(repo_configs, agent_ids)
    
    print(f"\nResults:")
    print(f"  Total: {result['total']}")
    print(f"  Successful: {result['successful']}")
    print(f"  Failed: {result['failed']}")
    
    # Check status
    print("\nChecking creation status...")
    status = get_creation_status.remote()
    print(f"  Repositories created: {status['count']}")
    
    for repo in status["repos"]:
        print(f"  - {repo['repo_name']} by {repo['agent_id']}")
