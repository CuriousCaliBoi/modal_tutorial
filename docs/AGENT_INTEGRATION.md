# Modal-Agent Integration Documentation

Comprehensive documentation for integrating Cursor cloud agents with Modal's compute platform.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Getting Started](#getting-started)
3. [API Reference](#api-reference)
4. [Multi-Agent Patterns](#multi-agent-patterns)
5. [Resource Management](#resource-management)
6. [Best Practices](#best-practices)
7. [Advanced Topics](#advanced-topics)

## Architecture Overview

The Modal-Agent Integration Framework consists of several components:

### Components

1. **Modal Agent API Service**: FastAPI service providing HTTP endpoints for agent operations
2. **Agent Tools**: Modal functions that agents can call as tools
3. **Parallel Executor**: Utilities for parallel task execution
4. **Resource Manager**: Utilities for managing compute resources
5. **Agent Helper Library**: Python client library for easy Modal interaction
6. **Multi-Agent Coordination Examples**: Patterns for coordinating multiple agents

### Architecture Diagram

```
┌─────────────┐         ┌──────────────┐         ┌──────────┐
│   Agent 1   │────────▶│  Modal Agent  │◀────────│  Modal    │
│             │         │  API Service │         │  Compute │
└─────────────┘         └──────────────┘         └──────────┘
         │                       │
         │                       │
┌─────────────┐         ┌──────────────┐
│   Agent 2   │────────▶│  Agent Tools │
│             │         │  & Executors │
└─────────────┘         └──────────────┘
         │                       │
         │                       │
┌─────────────┐         ┌──────────────┐
│   Agent 3   │────────▶│  Shared      │
│             │         │  Volumes     │
└─────────────┘         └──────────────┘
```

## Getting Started

### Prerequisites

- Modal account and API token
- Python 3.8+
- `modal` package installed: `pip install modal`

### Installation

1. Install Modal:
```bash
pip install modal
```

2. Set up Modal authentication:
```bash
modal token new
```

3. Deploy the API service:
```bash
modal deploy examples/agent_integration/modal_agent_api.py
```

### Basic Usage

#### Using the Agent Client

```python
from agent_modal_client import ModalAgentClient
import asyncio

async def main():
    async with ModalAgentClient(api_url="https://your-api-url.modal.com") as client:
        # Deploy a function
        deploy_result = await client.deploy_function(
            function_code='''
def process_data(data):
    return data * 2
''',
            function_name="process_data",
            gpu="T4"
        )
        
        # Run in parallel
        result = await client.run_parallel(
            function_name="process_data",
            inputs=[1, 2, 3, 4, 5],
            wait=True
        )
        print(result)

asyncio.run(main())
```

#### Using Modal Functions Directly

```python
import modal
from agent_tools import deploy_modal_function, spawn_parallel_tasks

app = modal.App("my-agent-app")

# Deploy function
result = deploy_modal_function.remote(
    function_code="def hello(): return 'world'",
    function_name="hello"
)

# Execute in parallel
results = spawn_parallel_tasks.remote(
    function_ref="hello",
    inputs=[1, 2, 3],
    wait=True
)
```

## API Reference

### Modal Agent API Endpoints

#### POST /deploy-function

Deploy a Modal function dynamically from code string.

**Request:**
```json
{
  "function_code": "def my_function(x): return x * 2",
  "function_name": "my_function",
  "image_config": {"packages": ["numpy"]},
  "gpu": "T4",
  "timeout": 3600,
  "secrets": ["api-keys"],
  "volumes": ["my-volume"]
}
```

**Response:**
```json
{
  "function_id": "uuid",
  "function_name": "my_function",
  "status": "registered"
}
```

#### POST /run-parallel

Execute a function in parallel across multiple containers.

**Request:**
```json
{
  "function_name": "my_function",
  "inputs": [1, 2, 3, 4, 5],
  "wait": true
}
```

**Response:**
```json
{
  "task_id": "uuid",
  "function_name": "my_function",
  "num_tasks": 5,
  "status": "completed"
}
```

#### POST /spawn-task

Spawn a single task execution.

**Request:**
```json
{
  "function_name": "my_function",
  "args": [42],
  "kwargs": {"param": "value"},
  "wait": false
}
```

**Response:**
```json
{
  "task_id": "uuid",
  "function_name": "my_function",
  "status": "pending"
}
```

#### GET /status/{task_id}

Get the status of a task.

**Response:**
```json
{
  "task_id": "uuid",
  "function_name": "my_function",
  "status": "completed",
  "result": "result_value"
}
```

#### POST /webhook

Register a webhook URL for task completion.

**Request:**
```json
{
  "task_id": "uuid",
  "webhook_url": "https://your-webhook.com/callback"
}
```

### Agent Tools Functions

#### deploy_modal_function()

Deploy a Modal function from code string.

**Parameters:**
- `function_code` (str)`: Python code string
- `function_name` (str)`: Name for the function
- `image_packages` (List[str], optional)`: Pip packages to install
- `gpu` (str, optional)`: GPU specification
- `timeout` (int, optional)`: Function timeout
- `secrets` (List[str], optional)`: Modal secret names
- `volumes` (List[str], optional)`: Modal volume names

**Returns:** Dictionary with deployment information

#### spawn_parallel_tasks()

Execute tasks in parallel using `.map()`.

**Parameters:**
- `function_ref` (str)`: Function reference
- `inputs` (List[Any])`: List of inputs
- `wait` (bool)`: Whether to wait for completion

**Returns:** Dictionary with task IDs and results

#### request_gpu_task()

Execute a task with GPU resources.

**Parameters:**
- `function_ref` (str)`: GPU-enabled function reference
- `args` (List[Any], optional)`: Positional arguments
- `kwargs` (Dict[str, Any], optional)`: Keyword arguments

**Returns:** Dictionary with task result

#### coordinate_state()

Coordinate state across agents via Modal Volumes.

**Parameters:**
- `volume_name` (str)`: Volume name
- `operation` (str)`: "read", "write", or "list"
- `data` (Dict[str, Any], optional)`: Data to write
- `key` (str, optional)`: Key/path for data

**Returns:** Dictionary with operation result

## Multi-Agent Patterns

### Pattern 1: Parallel Repository Creation

Multiple agents create repositories in parallel, coordinating via shared volume.

```python
from multi_agent_repo_creation import coordinate_repo_creation

repo_configs = [
    {"name": "repo1", "files": {...}},
    {"name": "repo2", "files": {...}}
]

agent_ids = ["agent-1", "agent-2"]
result = coordinate_repo_creation.remote(repo_configs, agent_ids)
```

### Pattern 2: Parallel Code Generation

Agents generate code components in parallel, then aggregate results.

```python
from multi_agent_codegen import coordinate_code_generation

tasks = [
    {"component": "processor", "type": "class"},
    {"component": "client", "type": "class"}
]

agent_ids = ["codegen-1", "codegen-2"]
result = coordinate_code_generation.remote(tasks, agent_ids)
```

### Pattern 3: Master-Worker Coordination

Master agent distributes work to worker agents via Modal.

```python
from multi_agent_coordinator import master_agent_coordinator

tasks = [
    {"type": "analysis", "data": {...}},
    {"type": "validation", "data": {...}}
]

worker_ids = ["worker-1", "worker-2"]
result = master_agent_coordinator.remote(tasks, worker_ids)
```

## Resource Management

### GPU Resources

Request GPU resources for compute-intensive tasks:

```python
from resource_manager import request_gpu_resources

result = request_gpu_resources.remote(
    gpu_type="T4",
    count=1,
    function_name="gpu_function"
)
```

Available GPU types:
- `T4`: General purpose, cost-effective
- `A10G`: High performance
- `A100`: Very high performance
- `H100`: Latest generation

### Volumes

Create and use persistent volumes:

```python
from resource_manager import create_volume

result = create_volume.remote(
    volume_name="agent-storage",
    create_if_missing=True
)
```

### Secrets

Mount Modal secrets:

```python
from resource_manager import mount_secrets

result = mount_secrets.remote(
    secret_names=["api-keys", "db-credentials"],
    function_name="my_function"
)
```

### Compute Resources

Request CPU and memory:

```python
from resource_manager import request_compute_resources

result = request_compute_resources.remote(
    cpu_count=8,
    memory_gb=16,
    timeout=3600
)
```

## Best Practices

### 1. Error Handling

Always implement proper error handling:

```python
async def safe_operation(client):
    try:
        result = await client.run_parallel(...)
        return result
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}")
        # Retry logic
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Fallback logic
```

### 2. Resource Efficiency

- Request appropriate resource sizes
- Reuse volumes and secrets across tasks
- Use appropriate GPU types for workloads
- Clean up unused resources

### 3. Parallel Execution

- Use `.map()` for independent tasks
- Batch items to avoid overwhelming the system
- Monitor task status for long-running operations

### 4. State Coordination

- Use Modal Volumes for shared state
- Commit volume changes explicitly
- Use atomic operations when possible

### 5. Monitoring

- Register webhooks for async tasks
- Use StatusMonitor for tracking multiple tasks
- Log important operations

## Advanced Topics

### Dynamic Function Compilation

In a production system, you would need to dynamically compile and register functions:

```python
def compile_and_register_function(code: str, name: str, app: modal.App):
    # Compile code
    compiled = compile(code, f"<{name}>", "exec")
    
    # Execute in namespace
    namespace = {}
    exec(compiled, namespace)
    
    # Extract function
    func = namespace[name]
    
    # Register with Modal app
    @app.function()
    def registered_function(*args, **kwargs):
        return func(*args, **kwargs)
    
    return registered_function
```

### Custom Image Configuration

Create custom images for specific workloads:

```python
image = modal.Image.debian_slim().pip_install(
    "numpy",
    "pandas",
    "torch"
).apt_install("ffmpeg")

app = modal.App("custom-app", image=image)
```

### Webhook Integration

Set up webhooks for async task completion:

```python
# Register webhook
await client.register_webhook(
    task_id="task-123",
    webhook_url="https://your-app.com/webhook"
)

# Handle webhook in your application
@app.post("/webhook")
async def handle_webhook(payload: dict):
    task_id = payload["task_id"]
    status = payload["status"]
    # Process completed task
```

### Volume-Based State Management

Coordinate state across agents using volumes:

```python
# Write state
result = coordinate_state.remote(
    volume_name="shared-state",
    operation="write",
    data={"key": "value"},
    key="state"
)

# Read state
result = coordinate_state.remote(
    volume_name="shared-state",
    operation="read",
    key="state"
)
```

## Troubleshooting

### Common Issues

1. **Task Not Starting**
   - Check function deployment status
   - Verify resource availability
   - Check Modal app logs

2. **Parallel Execution Fails**
   - Ensure inputs are properly formatted
   - Check for function errors
   - Verify resource limits

3. **State Coordination Issues**
   - Ensure volumes are mounted
   - Check volume commit operations
   - Verify file paths

4. **Resource Allocation Errors**
   - Check resource availability
   - Verify GPU types
   - Check quota limits

### Debugging Tips

1. Use Modal's logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. Check task status regularly:
```python
status = await client.get_task_status(task_id)
print(status)
```

3. Monitor resource usage:
```python
status = get_resource_status.remote()
print(status)
```

## Performance Optimization

1. **Batch Processing**: Process items in batches
2. **Resource Pooling**: Reuse volumes and secrets
3. **Async Execution**: Use `wait=False` for long tasks
4. **Parallel Limits**: Be mindful of concurrency limits
5. **Cost Optimization**: Use appropriate resource sizes

## Security Considerations

1. **API Authentication**: Secure your API endpoints
2. **Secret Management**: Use Modal secrets for sensitive data
3. **Input Validation**: Validate all inputs
4. **Error Messages**: Don't expose sensitive information
5. **Webhook Security**: Verify webhook signatures

## Further Reading

- Modal Documentation: https://modal.com/docs
- FastAPI Documentation: https://fastapi.tiangolo.com
- Async Python Patterns: https://docs.python.org/3/library/asyncio.html

## Support

For issues or questions:
- Check example files in `examples/agent_integration/`
- Review Modal documentation
- See `examples/agent_integration/README.md` for quick start
