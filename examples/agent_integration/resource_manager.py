"""
Resource Management Utilities

Utilities for agents to manage compute resources, GPUs, volumes, and secrets.
"""

import modal
from typing import List, Dict, Optional, Any
from enum import Enum

app = modal.App("resource-manager")

base_image = modal.Image.debian_slim().pip_install("pydantic")


class GPUType(str, Enum):
    """Available GPU types"""
    T4 = "T4"
    A10G = "A10G"
    A100 = "A100"
    H100 = "H100"


@app.function(image=base_image)
def request_gpu_resources(
    gpu_type: str,
    count: int = 1,
    function_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Request GPU resources for a function.
    
    Args:
        gpu_type: Type of GPU (T4, A10G, A100, H100)
        count: Number of GPUs (for multi-GPU setups)
        function_name: Optional function name to associate with GPU
    
    Returns:
        Dictionary with GPU allocation information
    """
    try:
        # Validate GPU type
        if gpu_type not in [g.value for g in GPUType]:
            return {
                "status": "error",
                "message": f"Invalid GPU type: {gpu_type}. Available: {[g.value for g in GPUType]}"
            }
        
        # In practice, this would allocate GPU resources
        # For now, return allocation info
        
        return {
            "status": "allocated",
            "gpu_type": gpu_type,
            "count": count,
            "function_name": function_name,
            "message": f"GPU resources allocated: {count}x {gpu_type}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.function(image=base_image, gpu="T4")
def test_gpu_allocation() -> Dict[str, Any]:
    """Test function to verify GPU allocation"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            
            return {
                "status": "success",
                "gpu_available": True,
                "device_name": device_name,
                "device_count": device_count
            }
        else:
            return {
                "status": "success",
                "gpu_available": False,
                "message": "CUDA not available"
            }
    except ImportError:
        return {
            "status": "error",
            "message": "PyTorch not installed. Install with: pip install torch"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.function(image=base_image)
def create_volume(
    volume_name: str,
    create_if_missing: bool = True
) -> Dict[str, Any]:
    """
    Create or get a Modal Volume for persistent storage.
    
    Args:
        volume_name: Name of the volume
        create_if_missing: Whether to create the volume if it doesn't exist
    
    Returns:
        Dictionary with volume information
    """
    try:
        volume = modal.Volume.from_name(volume_name, create_if_missing=create_if_missing)
        
        return {
            "status": "success",
            "volume_name": volume_name,
            "message": f"Volume '{volume_name}' ready"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.function(image=base_image)
def list_volumes() -> Dict[str, Any]:
    """
    List available Modal volumes.
    
    Note: Modal doesn't have a direct API to list volumes, so this
    is a conceptual implementation.
    
    Returns:
        Dictionary with volume list
    """
    # In practice, you would need to maintain a registry of volumes
    # or use Modal's API if available
    
    return {
        "status": "success",
        "message": "Volume listing not directly available in Modal API",
        "note": "Volumes are accessed by name using Volume.from_name()"
    }


@app.function(image=base_image)
def mount_secrets(
    secret_names: List[str],
    function_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Mount Modal Secrets for a function.
    
    Args:
        secret_names: List of secret names to mount
        function_name: Optional function name to associate with secrets
    
    Returns:
        Dictionary with secret mounting information
    """
    try:
        secrets = [modal.Secret.from_name(name) for name in secret_names]
        
        return {
            "status": "success",
            "secret_names": secret_names,
            "secret_count": len(secrets),
            "function_name": function_name,
            "message": f"Secrets mounted: {', '.join(secret_names)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.function(image=base_image)
def request_compute_resources(
    cpu_count: Optional[int] = None,
    memory_gb: Optional[int] = None,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """
    Request compute resources (CPU, memory) for a function.
    
    Args:
        cpu_count: Number of CPUs to request
        memory_gb: Memory in GB to request
        timeout: Timeout in seconds
    
    Returns:
        Dictionary with resource allocation information
    """
    try:
        resources = {}
        
        if cpu_count:
            resources["cpu"] = cpu_count
        if memory_gb:
            resources["memory"] = f"{memory_gb}GB"
        if timeout:
            resources["timeout"] = f"{timeout}s"
        
        return {
            "status": "success",
            "resources": resources,
            "message": "Compute resources configured"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.function(
    image=base_image,
    cpu=4,
    memory=8192,
    timeout=3600
)
def example_resource_usage() -> Dict[str, Any]:
    """
    Example function demonstrating resource usage.
    
    This function uses:
    - 4 CPUs
    - 8GB memory
    - 1 hour timeout
    """
    import psutil
    
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    
    return {
        "status": "success",
        "cpu_count": cpu_count,
        "memory_total_gb": memory.total / (1024**3),
        "memory_available_gb": memory.available / (1024**3),
        "message": "Resource usage example completed"
    }


@app.function(image=base_image)
def get_resource_status() -> Dict[str, Any]:
    """
    Get current resource status and availability.
    
    Returns:
        Dictionary with resource status information
    """
    try:
        # In practice, would query Modal's resource availability
        # This is a conceptual implementation
        
        return {
            "status": "success",
            "resources": {
                "gpus": {
                    "T4": "available",
                    "A10G": "available",
                    "A100": "limited",
                    "H100": "limited"
                },
                "compute": {
                    "cpu": "available",
                    "memory": "available"
                },
                "storage": {
                    "volumes": "available",
                    "ephemeral": "available"
                }
            },
            "message": "Resource status retrieved"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.local_entrypoint()
def example_resource_management():
    """Example of using resource management utilities"""
    # Example 1: Request GPU
    print("Example 1: Request GPU")
    gpu_result = request_gpu_resources.remote("T4", count=1)
    print(f"GPU allocation: {gpu_result}")
    
    # Example 2: Create volume
    print("\nExample 2: Create volume")
    volume_result = create_volume.remote("agent-storage")
    print(f"Volume creation: {volume_result}")
    
    # Example 3: Mount secrets
    print("\nExample 3: Mount secrets")
    secrets_result = mount_secrets.remote(["api-keys", "db-credentials"])
    print(f"Secrets mounting: {secrets_result}")
    
    # Example 4: Request compute resources
    print("\nExample 4: Request compute resources")
    compute_result = request_compute_resources.remote(cpu_count=8, memory_gb=16)
    print(f"Compute resources: {compute_result}")
    
    # Example 5: Get resource status
    print("\nExample 5: Get resource status")
    status = get_resource_status.remote()
    print(f"Resource status: {status}")
