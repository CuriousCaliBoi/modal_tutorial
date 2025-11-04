"""
Multi-Agent Coordinator Pattern

Demonstrates how a master agent can distribute work to worker agents via Modal,
coordinating parallel analysis and task distribution.
"""

import modal
from typing import List, Dict, Any, Optional, Callable
import json
import os

app = modal.App("multi-agent-coordinator")

# Shared volume for agent coordination
shared_volume = modal.Volume.from_name("agent-coordinator", create_if_missing=True)

base_image = modal.Image.debian_slim().pip_install(
    "pydantic",
    "httpx"
)


@app.function(
    image=base_image,
    volumes={"/shared": shared_volume}
)
def worker_agent_task(
    task_payload: Dict[str, Any],
    worker_id: str
) -> Dict[str, Any]:
    """
    Execute a task assigned to a worker agent.
    
    Worker agents can process different parts of a larger task in parallel.
    
    Args:
        task_payload: Dictionary with task details (type, data, parameters)
        worker_id: ID of the worker agent
    
    Returns:
        Dictionary with task result
    """
    try:
        task_type = task_payload.get("type", "unknown")
        task_data = task_payload.get("data", {})
        task_params = task_payload.get("parameters", {})
        
        # Process task based on type
        if task_type == "analysis":
            # Analyze a codebase section
            section = task_data.get("section", "")
            result = {
                "lines_of_code": len(section.split("\n")),
                "functions": section.count("def "),
                "classes": section.count("class "),
                "complexity": "medium"  # Simplified
            }
        
        elif task_type == "transformation":
            # Transform code
            code = task_data.get("code", "")
            transformation = task_params.get("transformation", "none")
            
            if transformation == "format":
                result = {"transformed_code": code.strip()}
            else:
                result = {"transformed_code": code}
        
        elif task_type == "validation":
            # Validate code
            code = task_data.get("code", "")
            result = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
        
        else:
            result = {"message": f"Unknown task type: {task_type}"}
        
        # Save result to shared volume
        result_dir = f"/shared/results/{worker_id}"
        os.makedirs(result_dir, exist_ok=True)
        
        result_path = f"{result_dir}/result.json"
        result_data = {
            "worker_id": worker_id,
            "task_type": task_type,
            "result": result,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        with open(result_path, "w") as f:
            json.dump(result_data, f, indent=2)
        
        # Update task log
        log_path = "/shared/task_log.json"
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                log = json.load(f)
        else:
            log = []
        
        log.append({
            "worker_id": worker_id,
            "task_type": task_type,
            "status": "completed"
        })
        
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        
        # Commit volume changes
        shared_volume.commit()
        
        return {
            "status": "success",
            "worker_id": worker_id,
            "task_type": task_type,
            "result": result
        }
    except Exception as e:
        return {
            "status": "error",
            "worker_id": worker_id,
            "error": str(e)
        }


@app.function(
    image=base_image,
    volumes={"/shared": shared_volume}
)
def aggregate_worker_results(
    worker_ids: List[str]
) -> Dict[str, Any]:
    """
    Aggregate results from multiple worker agents.
    
    Args:
        worker_ids: List of worker agent IDs
    
    Returns:
        Dictionary with aggregated results
    """
    aggregated_results = []
    
    for worker_id in worker_ids:
        result_path = f"/shared/results/{worker_id}/result.json"
        
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result_data = json.load(f)
            aggregated_results.append(result_data)
    
    # Save aggregated results
    aggregated_path = "/shared/results/aggregated.json"
    with open(aggregated_path, "w") as f:
        json.dump(aggregated_results, f, indent=2)
    
    shared_volume.commit()
    
    return {
        "status": "success",
        "worker_count": len(aggregated_results),
        "results": aggregated_results,
        "aggregated_path": aggregated_path
    }


@app.function(
    image=base_image,
    volumes={"/shared": shared_volume}
)
def get_coordination_status() -> Dict[str, Any]:
    """Get status of all worker tasks"""
    log_path = "/shared/task_log.json"
    
    if not os.path.exists(log_path):
        return {
            "status": "success",
            "tasks": [],
            "count": 0
        }
    
    with open(log_path, "r") as f:
        log = json.load(f)
    
    # Count by status
    completed = len([t for t in log if t.get("status") == "completed"])
    
    return {
        "status": "success",
        "tasks": log,
        "total": len(log),
        "completed": completed
    }


@app.function(image=base_image)
def master_agent_coordinator(
    tasks: List[Dict[str, Any]],
    worker_ids: List[str],
    aggregate: bool = True
) -> Dict[str, Any]:
    """
    Master agent coordinates work distribution to worker agents.
    
    Args:
        tasks: List of task payloads to distribute
        worker_ids: List of worker agent IDs
        aggregate: Whether to aggregate results after completion
    
    Returns:
        Dictionary with coordination results
    """
    # Ensure we have enough workers
    if len(tasks) > len(worker_ids):
        # Reuse workers if needed (round-robin)
        worker_ids = [worker_ids[i % len(worker_ids)] for i in range(len(tasks))]
    
    # Distribute tasks to workers in parallel
    results = list(worker_agent_task.map(
        tasks,
        worker_ids[:len(tasks)]
    ))
    
    # Aggregate results if requested
    if aggregate:
        successful_workers = [
            r.get("worker_id")
            for r in results
            if r.get("status") == "success"
        ]
        
        if successful_workers:
            aggregated = aggregate_worker_results.remote(successful_workers)
        else:
            aggregated = {"status": "skipped", "message": "No successful tasks to aggregate"}
    else:
        aggregated = None
    
    # Analyze results
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "error"]
    
    return {
        "status": "completed",
        "total_tasks": len(tasks),
        "successful": len(successful),
        "failed": len(failed),
        "results": results,
        "aggregated": aggregated
    }


@app.function(image=base_image)
def parallel_analysis_coordinator(
    codebase_sections: List[str],
    analysis_type: str = "analysis"
) -> Dict[str, Any]:
    """
    Coordinate parallel analysis of codebase sections.
    
    Args:
        codebase_sections: List of code sections to analyze
        analysis_type: Type of analysis to perform
    
    Returns:
        Dictionary with analysis results
    """
    # Create tasks for each section
    tasks = [
        {
            "type": analysis_type,
            "data": {"section": section},
            "parameters": {}
        }
        for section in codebase_sections
    ]
    
    # Create worker IDs
    worker_ids = [f"analyst-{i+1}" for i in range(len(tasks))]
    
    # Coordinate analysis
    return master_agent_coordinator.remote(tasks, worker_ids, aggregate=True)


@app.local_entrypoint()
def example_multi_agent_coordination():
    """Example of multi-agent coordination"""
    
    # Example 1: Parallel analysis
    print("Example 1: Parallel codebase analysis")
    codebase_sections = [
        "def function1():\n    pass\n\nclass Class1:\n    pass",
        "def function2():\n    return True\n\nclass Class2:\n    def method(self):\n        pass",
        "def function3():\n    x = 1\n    y = 2\n    return x + y"
    ]
    
    analysis_result = parallel_analysis_coordinator.remote(codebase_sections, "analysis")
    print(f"  Total sections: {analysis_result['total_tasks']}")
    print(f"  Successful: {analysis_result['successful']}")
    print(f"  Failed: {analysis_result['failed']}")
    
    # Example 2: Task distribution
    print("\nExample 2: Master-worker task distribution")
    tasks = [
        {"type": "validation", "data": {"code": "def test():\n    pass"}, "parameters": {}},
        {"type": "transformation", "data": {"code": "  def test():\n      pass"}, "parameters": {"transformation": "format"}},
        {"type": "analysis", "data": {"section": "def func():\n    return 42"}, "parameters": {}}
    ]
    
    worker_ids = ["worker-1", "worker-2", "worker-3"]
    coordination_result = master_agent_coordinator.remote(tasks, worker_ids)
    
    print(f"  Total tasks: {coordination_result['total_tasks']}")
    print(f"  Successful: {coordination_result['successful']}")
    print(f"  Failed: {coordination_result['failed']}")
    
    # Check coordination status
    print("\nChecking coordination status...")
    status = get_coordination_status.remote()
    print(f"  Total tasks: {status['total']}")
    print(f"  Completed: {status['completed']}")
