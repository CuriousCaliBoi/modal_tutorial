# Modal-Agent Integration Guide

This guide provides examples and utilities for integrating Cursor cloud agents with Modal's compute platform, enabling agents to scale their compute, spawn parallel tasks, and coordinate multi-agent workflows.

## Overview

The Modal-Agent Integration Framework enables:
- **Dynamic Function Deployment**: Agents can deploy Modal functions on-the-fly
- **Parallel Processing**: Agents can spawn multiple containers to process tasks in parallel
- **Resource Management**: Agents can request GPUs, volumes, and secrets
- **Multi-Agent Coordination**: Multiple agents can coordinate using Modal Volumes for shared state

## Quick Start

### 1. Using the Agent Helper Library

```python
from agent_modal_client import ModalAgentClient
import asyncio

async def main():
    async with ModalAgentClient(api_url="https://your-modal-api.com") as client:
        # Deploy a function
        result = await client.deploy_function(
            function_code='''
def process_item(item):
    return item * 2
''',
            function_name="process_item"
        )
        
        # Run in parallel
        parallel_result = await client.run_parallel(
            function_name="process_item",
            inputs=[1, 2, 3, 4, 5],
            wait=True
        )
        print(parallel_result)

asyncio.run(main())
```

### 2. Using Modal Functions Directly

```python
from agent_tools import deploy_modal_function, spawn_parallel_tasks

# Deploy a function
deploy_result = deploy_modal_function.remote(
    function_code="def hello(): return 'world'",
    function_name="hello"
)

# Run in parallel
results = spawn_parallel_tasks.remote(
    function_ref="process_item",
    inputs=[1, 2, 3, 4, 5],
    wait=True
)
```

## Components

### 1. Modal Agent API Service (`modal_agent_api.py`)

FastAPI service deployed on Modal that provides HTTP endpoints for agents:
- `POST /deploy-function` - Deploy a Modal function dynamically
- `POST /run-parallel` - Execute function in parallel across containers
- `POST /spawn-task` - Spawn a single task
- `GET /status/{task_id}` - Check task status
- `POST /webhook` - Register webhook for task completion
- `GET /functions` - List registered functions
- `GET /tasks` - List all tasks

**Deploy:**
```bash
modal deploy examples/agent_integration/modal_agent_api.py
```

### 2. Agent Tools (`agent_tools.py`)

Core tool functions that agents can call:
- `deploy_modal_function()` - Deploy function from code string
- `spawn_parallel_tasks()` - Execute tasks in parallel
- `request_gpu_task()` - Execute task with GPU
- `get_task_result()` - Retrieve task results
- `coordinate_state()` - Share state across agents via Modal Volumes

### 3. Parallel Executor (`parallel_executor.py`)

Utilities for parallel execution:
- `process_item()` - Generic item processor
- `batch_processor()` - Process items in batches
- `parallel_map_executor()` - Execute function in parallel using .map()
- `parallel_starmap_executor()` - Execute with multiple arguments using .starmap()
- `parallel_filter_executor()` - Filter items in parallel
- `parallel_reduce_executor()` - Perform parallel reduce operations

### 4. Resource Manager (`resource_manager.py`)

Utilities for managing compute resources:
- `request_gpu_resources()` - Request GPU allocation
- `create_volume()` - Create or get Modal Volume
- `mount_secrets()` - Mount Modal Secrets
- `request_compute_resources()` - Request CPU/memory resources
- `get_resource_status()` - Get current resource availability

### 5. Multi-Agent Coordination Examples

#### Parallel Repository Creation (`multi_agent_repo_creation.py`)

Example showing multiple agents creating repositories in parallel:

```python
from multi_agent_repo_creation import coordinate_repo_creation

repo_configs = [
    {"name": "frontend", "files": {...}},
    {"name": "backend", "files": {...}},
    {"name": "data", "files": {...}}
]

agent_ids = ["agent-1", "agent-2", "agent-3"]
result = coordinate_repo_creation.remote(repo_configs, agent_ids)
```

#### Parallel Code Generation (`multi_agent_codegen.py`)

Example showing agents generating code in parallel, then aggregating:

```python
from multi_agent_codegen import coordinate_code_generation

tasks = [
    {"component": "processor", "type": "class", "requirements": "..."},
    {"component": "client", "type": "class", "requirements": "..."}
]

agent_ids = ["codegen-1", "codegen-2"]
result = coordinate_code_generation.remote(tasks, agent_ids)
```

#### Agent Coordinator (`multi_agent_coordinator.py`)

Example showing master agent distributing work to worker agents:

```python
from multi_agent_coordinator import master_agent_coordinator

tasks = [
    {"type": "analysis", "data": {...}},
    {"type": "validation", "data": {...}}
]

worker_ids = ["worker-1", "worker-2"]
result = master_agent_coordinator.remote(tasks, worker_ids)
```

### 6. Agent Helper Library (`agent_modal_client.py`)

Python library for agents to easily interact with Modal:

```python
from agent_modal_client import ModalAgentClient, StatusMonitor

async with ModalAgentClient() as client:
    # Deploy and run functions
    result = await client.deploy_function(...)
    task = await client.spawn_task(...)
    
    # Monitor status
    monitor = StatusMonitor(client)
    statuses = await monitor.monitor_tasks([task["task_id"]])
```

## Best Practices

### 1. Error Handling

Always wrap Modal calls in try-except blocks:

```python
try:
    result = await client.run_parallel(...)
except Exception as e:
    print(f"Error: {e}")
    # Handle error appropriately
```

### 2. Resource Management

Request resources appropriately:

```python
# For CPU-intensive tasks
result = request_compute_resources.remote(cpu_count=8, memory_gb=16)

# For GPU tasks
result = request_gpu_resources.remote("T4", count=1)
```

### 3. State Coordination

Use Modal Volumes for shared state:

```python
result = coordinate_state.remote(
    volume_name="agent-storage",
    operation="write",
    data={"key": "value"},
    key="shared-state"
)
```

### 4. Parallel Execution

Use `.map()` for parallel processing:

```python
# Process 100 items in parallel
results = process_item.map(range(100))
```

### 5. Webhooks for Async Tasks

Register webhooks for long-running tasks:

```python
task = await client.spawn_task(..., wait=False)
await client.register_webhook(task["task_id"], "https://your-webhook.com")
```

## Performance Tips

1. **Batch Processing**: Process items in batches to avoid overwhelming the system
2. **Resource Pooling**: Reuse volumes and secrets across tasks
3. **Async Execution**: Use `wait=False` for long-running tasks and poll for status
4. **Parallel Limits**: Be mindful of Modal's concurrency limits
5. **Cost Optimization**: Use appropriate GPU types and resource sizes

## Troubleshooting

### Task Not Starting
- Check function deployment status
- Verify resource availability
- Check Modal app logs

### Parallel Execution Issues
- Ensure inputs are properly formatted
- Check for function errors in Modal logs
- Verify resource limits

### State Coordination Problems
- Ensure volumes are properly mounted
- Check volume commit operations
- Verify file paths are correct

## Next Steps

- See `docs/AGENT_INTEGRATION.md` for detailed documentation
- Explore the example files in this directory
- Deploy the API service and test with your agents

## Support

For issues or questions:
- Check Modal documentation: https://modal.com/docs
- Review example files in this directory
- See detailed docs in `docs/AGENT_INTEGRATION.md`
