"""
Agent Modal Client

Python library that agents can import to easily interact with Modal.
Provides simple wrapper functions for common Modal operations with
error handling and retries.
"""

from typing import Any, Dict, List, Optional, Callable
import time
import asyncio
import httpx
from functools import wraps


class ModalAgentClient:
    """
    Client for agents to interact with Modal Agent API.
    
    This client provides a simple interface for agents to:
    - Deploy functions dynamically
    - Run parallel tasks
    - Query task status
    - Manage resources
    """
    
    def __init__(self, api_url: Optional[str] = None):
        """
        Initialize the Modal Agent Client.
        
        Args:
            api_url: URL of the Modal Agent API service. If None, uses default.
        """
        self.api_url = api_url or "https://modal-agent-api.modal.com"
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def deploy_function(
        self,
        function_code: str,
        function_name: str,
        image_config: Optional[Dict[str, Any]] = None,
        gpu: Optional[str] = None,
        timeout: Optional[int] = None,
        secrets: Optional[List[str]] = None,
        volumes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Deploy a Modal function dynamically.
        
        Args:
            function_code: Python code string defining the function
            function_name: Name for the function
            image_config: Optional image configuration
            gpu: GPU specification (e.g., "T4", "A100")
            timeout: Function timeout in seconds
            secrets: List of Modal secret names to mount
            volumes: List of Modal volume names to mount
        
        Returns:
            Dictionary with deployment information
        """
        payload = {
            "function_code": function_code,
            "function_name": function_name,
            "image_config": image_config,
            "gpu": gpu,
            "timeout": timeout,
            "secrets": secrets,
            "volumes": volumes
        }
        
        response = await self.client.post(
            f"{self.api_url}/deploy-function",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def run_parallel(
        self,
        function_name: str,
        inputs: List[Any],
        wait: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a function in parallel across multiple containers.
        
        Args:
            function_name: Name of the function to execute
            inputs: List of inputs to process in parallel
            wait: Whether to wait for completion
        
        Returns:
            Dictionary with task IDs and results
        """
        payload = {
            "function_name": function_name,
            "inputs": inputs,
            "wait": wait
        }
        
        response = await self.client.post(
            f"{self.api_url}/run-parallel",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def spawn_task(
        self,
        function_name: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        wait: bool = False
    ) -> Dict[str, Any]:
        """
        Spawn a single task execution.
        
        Args:
            function_name: Name of the function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            wait: Whether to wait for completion
        
        Returns:
            Dictionary with task ID and status
        """
        payload = {
            "function_name": function_name,
            "args": args or [],
            "kwargs": kwargs or {},
            "wait": wait
        }
        
        response = await self.client.post(
            f"{self.api_url}/spawn-task",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.
        
        Args:
            task_id: ID of the task to check
        
        Returns:
            Dictionary with task status and results
        """
        response = await self.client.get(
            f"{self.api_url}/status/{task_id}"
        )
        response.raise_for_status()
        return response.json()
    
    async def wait_for_task(
        self,
        task_id: str,
        poll_interval: float = 1.0,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Wait for a task to complete, polling for status.
        
        Args:
            task_id: ID of the task to wait for
            poll_interval: Seconds between status checks
            timeout: Maximum time to wait (None for no timeout)
        
        Returns:
            Dictionary with final task status and results
        """
        start_time = time.time()
        
        while True:
            status = await self.get_task_status(task_id)
            
            if status["status"] in ["completed", "failed"]:
                return status
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            
            await asyncio.sleep(poll_interval)
    
    async def register_webhook(
        self,
        task_id: str,
        webhook_url: str
    ) -> Dict[str, Any]:
        """
        Register a webhook URL for task completion notification.
        
        Args:
            task_id: ID of the task
            webhook_url: URL to call when task completes
        
        Returns:
            Dictionary with webhook registration status
        """
        payload = {
            "task_id": task_id,
            "webhook_url": webhook_url
        }
        
        response = await self.client.post(
            f"{self.api_url}/webhook",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def list_functions(self) -> Dict[str, Any]:
        """List all registered functions."""
        response = await self.client.get(f"{self.api_url}/functions")
        response.raise_for_status()
        return response.json()
    
    async def list_tasks(self) -> Dict[str, Any]:
        """List all tasks."""
        response = await self.client.get(f"{self.api_url}/tasks")
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator for retrying function calls on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (attempt + 1))
                    else:
                        raise
            
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


class StatusMonitor:
    """
    Utility for monitoring task status and progress.
    """
    
    def __init__(self, client: ModalAgentClient):
        """
        Initialize status monitor.
        
        Args:
            client: ModalAgentClient instance
        """
        self.client = client
    
    async def monitor_tasks(
        self,
        task_ids: List[str],
        poll_interval: float = 2.0,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Monitor multiple tasks and return their final statuses.
        
        Args:
            task_ids: List of task IDs to monitor
            poll_interval: Seconds between status checks
            callback: Optional callback function called with each status update
        
        Returns:
            Dictionary mapping task_id to final status
        """
        import asyncio
        
        results = {}
        remaining = set(task_ids)
        
        while remaining:
            for task_id in list(remaining):
                try:
                    status = await self.client.get_task_status(task_id)
                    results[task_id] = status
                    
                    if callback:
                        callback(status)
                    
                    if status["status"] in ["completed", "failed"]:
                        remaining.remove(task_id)
                except Exception as e:
                    print(f"Error checking task {task_id}: {e}")
            
            if remaining:
                await asyncio.sleep(poll_interval)
        
        return results


# Example usage
async def example_usage():
    """Example of using the ModalAgentClient"""
    import asyncio
    
    async with ModalAgentClient() as client:
        # Deploy a function
        deploy_result = await client.deploy_function(
            function_code='''
def process_data(data):
    return data * 2
''',
            function_name="process_data"
        )
        print(f"Deployed: {deploy_result}")
        
        # Run in parallel
        parallel_result = await client.run_parallel(
            function_name="process_data",
            inputs=[1, 2, 3, 4, 5],
            wait=True
        )
        print(f"Parallel result: {parallel_result}")
        
        # Spawn a task
        task_result = await client.spawn_task(
            function_name="process_data",
            args=[42],
            wait=False
        )
        print(f"Task spawned: {task_result}")
        
        # Wait for task
        if not task_result.get("wait"):
            final_status = await client.wait_for_task(task_result["task_id"])
            print(f"Task completed: {final_status}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
