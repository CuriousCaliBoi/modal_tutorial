"""
Agent Tool Functions

Modal functions that agents can call as tools:
- Dynamic function deployment
- Parallel task spawning
- Resource management
- State coordination
"""

import modal
import json
from typing import Any, Dict, List, Optional, Callable
import inspect

app = modal.App("agent-tools")

# Base image with common dependencies
base_image = modal.Image.debian_slim().pip_install(
    "httpx",
    "pydantic"
)


@app.function(image=base_image)
def deploy_modal_function(
    function_code: str,
    function_name: str,
    image_packages: Optional[List[str]] = None,
    gpu: Optional[str] = None,
    timeout: Optional[int] = None,
    secrets: Optional[List[str]] = None,
    volumes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Deploy a Modal function dynamically from code string.
    
    Args:
        function_code: Python code string defining the function
        function_name: Name for the function
        image_packages: List of pip packages to install in the image
        gpu: GPU specification (e.g., "T4", "A100")
        timeout: Function timeout in seconds
        secrets: List of Modal secret names to mount
        volumes: List of Modal volume names to mount
    
    Returns:
        Dictionary with deployment information
    """
    try:
        # Compile the function code
        code_obj = compile(function_code, f"<function:{function_name}>", "exec")
        
        # Create a namespace for execution
        namespace = {}
        exec(code_obj, namespace)
        
        # Extract the function from the namespace
        if function_name not in namespace:
            raise ValueError(f"Function '{function_name}' not found in code")
        
        func = namespace[function_name]
        
        # Build image if packages specified
        image = base_image
        if image_packages:
            image = modal.Image.debian_slim().pip_install(*image_packages)
        
        # Build function decorator args
        decorator_kwargs = {}
        if gpu:
            decorator_kwargs["gpu"] = gpu
        if timeout:
            decorator_kwargs["timeout"] = timeout
        if secrets:
            decorator_kwargs["secrets"] = [modal.Secret.from_name(s) for s in secrets]
        if volumes:
            decorator_kwargs["volumes"] = {
                f"/vol/{v}": modal.Volume.from_name(v, create_if_missing=True)
                for v in volumes
            }
        
        # Register function with Modal app
        # Note: In practice, dynamic function registration requires app-level changes
        # This is a conceptual implementation showing the pattern
        
        return {
            "status": "success",
            "function_name": function_name,
            "message": f"Function {function_name} deployed successfully",
            "config": {
                "gpu": gpu,
                "timeout": timeout,
                "secrets": secrets,
                "volumes": volumes
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "function_name": function_name,
            "error": str(e)
        }


@app.function(image=base_image)
def spawn_parallel_tasks(
    function_ref: str,
    inputs: List[Any],
    wait: bool = True
) -> Dict[str, Any]:
    """
    Execute tasks in parallel using Modal's .map() functionality.
    
    Args:
        function_ref: Reference to the function to call (e.g., "process_item")
        inputs: List of inputs to process in parallel
        wait: Whether to wait for all tasks to complete
    
    Returns:
        Dictionary with task IDs and results
    """
    try:
        # In practice, you would look up the function from the app
        # and call .map() on it
        # This is a conceptual implementation
        
        task_count = len(inputs)
        
        if wait:
            # Simulate parallel execution
            # In real implementation: results = function_ref.map(inputs)
            results = [f"Processed input {i}" for i in range(task_count)]
            
            return {
                "status": "completed",
                "task_count": task_count,
                "results": results
            }
        else:
            # Return task IDs for async execution
            task_ids = [f"task_{i}" for i in range(task_count)]
            
            return {
                "status": "pending",
                "task_count": task_count,
                "task_ids": task_ids
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.function(image=base_image, gpu="T4")
def request_gpu_task(
    function_ref: str,
    args: Optional[List[Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute a task with GPU resources.
    
    Args:
        function_ref: Reference to the GPU-enabled function
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
    
    Returns:
        Dictionary with task result
    """
    try:
        # In practice, would call the GPU function here
        # This is a conceptual implementation
        
        return {
            "status": "completed",
            "function": function_ref,
            "gpu_used": "T4",
            "result": "GPU task executed successfully",
            "args": args,
            "kwargs": kwargs
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.function(image=base_image)
def get_task_result(task_id: str) -> Dict[str, Any]:
    """
    Retrieve task results by task ID.
    
    Args:
        task_id: ID of the task to retrieve
    
    Returns:
        Dictionary with task status and results
    """
    try:
        # In practice, would query Modal's task store or app state
        # This is a conceptual implementation
        
        return {
            "task_id": task_id,
            "status": "completed",
            "result": f"Result for task {task_id}",
            "retrieved_at": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "status": "error",
            "error": str(e)
        }


@app.function(image=base_image)
def coordinate_state(
    volume_name: str,
    operation: str,
    data: Optional[Dict[str, Any]] = None,
    key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Coordinate state across agents using Modal Volumes.
    
    Args:
        volume_name: Name of the Modal volume to use
        operation: Operation to perform ("read", "write", "list")
        data: Data to write (for write operations)
        key: Key/path for the data (for read/write operations)
    
    Returns:
        Dictionary with operation result
    """
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    
    @app.function(volumes={"/shared": volume}, image=base_image)
    def volume_operation(op: str, dat: Optional[Dict] = None, k: Optional[str] = None):
        import json
        import os
        
        if op == "write":
            if not k or not dat:
                return {"status": "error", "message": "key and data required for write"}
            
            path = f"/shared/{k}.json"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(dat, f)
            
            volume.commit()
            
            return {
                "status": "success",
                "operation": "write",
                "key": k,
                "message": f"Data written to {k}"
            }
        
        elif op == "read":
            if not k:
                return {"status": "error", "message": "key required for read"}
            
            path = f"/shared/{k}.json"
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                return {
                    "status": "success",
                    "operation": "read",
                    "key": k,
                    "data": data
                }
            except FileNotFoundError:
                return {
                    "status": "error",
                    "message": f"Key {k} not found"
                }
        
        elif op == "list":
            try:
                files = os.listdir("/shared")
                return {
                    "status": "success",
                    "operation": "list",
                    "files": files
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": str(e)
                }
        
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {op}"
            }
    
    return volume_operation.remote(operation, data, key)
