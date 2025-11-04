"""
Parallel Execution Utilities

Utilities for agents to execute tasks in parallel across Modal containers.
"""

import modal
from typing import List, Any, Callable, Dict, Optional, Tuple

app = modal.App("parallel-executor")

base_image = modal.Image.debian_slim().pip_install("tqdm")


@app.function(image=base_image)
def process_item(item: Any, processor_config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Generic item processor that can be used with .map() for parallel execution.
    
    Args:
        item: The item to process
        processor_config: Optional configuration dict for processing
    
    Returns:
        Processed item result
    """
    # Default processing: just return the item squared (as an example)
    if processor_config is None:
        processor_config = {}
    
    operation = processor_config.get("operation", "square")
    
    if operation == "square":
        return item ** 2
    elif operation == "double":
        return item * 2
    elif operation == "identity":
        return item
    else:
        return item


@app.function(image=base_image)
def batch_processor(
    items: List[Any],
    batch_size: int = 10,
    processor_config: Optional[Dict[str, Any]] = None
) -> List[Any]:
    """
    Process items in batches, with each batch running in parallel.
    
    Args:
        items: List of items to process
        batch_size: Number of items per batch
        processor_config: Configuration for processing
    
    Returns:
        List of processed results
    """
    results = []
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Process batch in parallel
        batch_results = list(process_item.map(batch, processor_config))
        results.extend(batch_results)
    
    return results


@app.function(image=base_image)
def parallel_map_executor(
    function_name: str,
    inputs: List[Any],
    function_kwargs: Optional[Dict[str, Any]] = None
) -> List[Any]:
    """
    Execute a function in parallel across multiple inputs using .map().
    
    This is a generic executor that agents can use to run any function
    in parallel across Modal containers.
    
    Args:
        function_name: Name of the function to execute
        inputs: List of inputs to process
        function_kwargs: Optional kwargs to pass to each function call
    
    Returns:
        List of results from parallel execution
    """
    # In practice, this would dynamically look up and call the function
    # For now, we use process_item as an example
    
    if function_kwargs is None:
        function_kwargs = {}
    
    # Execute in parallel using .map()
    results = list(process_item.map(inputs, function_kwargs))
    
    return results


@app.function(image=base_image)
def parallel_starmap_executor(
    function_name: str,
    inputs: List[tuple],
    function_kwargs: Optional[Dict[str, Any]] = None
) -> List[Any]:
    """
    Execute a function in parallel with multiple arguments using .starmap().
    
    Args:
        function_name: Name of the function to execute
        inputs: List of tuples, where each tuple contains arguments for one call
        function_kwargs: Optional kwargs to pass to each function call
    
    Returns:
        List of results from parallel execution
    """
    @app.function(image=base_image)
    def process_with_args(*args, **kwargs):
        """Process function with multiple arguments"""
        # In practice, would call the actual function here
        return {"args": args, "kwargs": kwargs}
    
    # Execute in parallel using .starmap()
    results = list(process_with_args.starmap(inputs, function_kwargs or {}))
    
    return results


@app.function(image=base_image)
def parallel_filter_executor(
    inputs: List[Any],
    filter_function: str,
    filter_config: Optional[Dict[str, Any]] = None
) -> List[Any]:
    """
    Filter items in parallel using a filter function.
    
    Args:
        inputs: List of items to filter
        filter_function: Name of the filter function logic
        filter_config: Configuration for filtering
    
    Returns:
        List of filtered items
    """
    @app.function(image=base_image)
    def apply_filter(item: Any, config: Optional[Dict[str, Any]] = None) -> Tuple[Any, bool]:
        """Apply filter to an item, returning (item, should_keep)"""
        if config is None:
            config = {}
        
        filter_type = config.get("type", "greater_than")
        threshold = config.get("threshold", 0)
        
        if filter_type == "greater_than":
            return (item, item > threshold)
        elif filter_type == "less_than":
            return (item, item < threshold)
        elif filter_type == "equals":
            return (item, item == threshold)
        else:
            return (item, True)
    
    # Process all items in parallel
    filtered_results = list(apply_filter.map(inputs, filter_config))
    
    # Filter out items where should_keep is False
    results = [item for item, should_keep in filtered_results if should_keep]
    
    return results


@app.function(image=base_image)
def parallel_reduce_executor(
    inputs: List[Any],
    reduce_function: str,
    initial_value: Any = None
) -> Any:
    """
    Perform a parallel reduce operation.
    
    First processes items in parallel, then reduces the results.
    
    Args:
        inputs: List of items to process and reduce
        reduce_function: Name of the reduce operation ("sum", "product", "max", "min")
        initial_value: Initial value for reduction
    
    Returns:
        Reduced result
    """
    # First, process items in parallel
    processed = list(process_item.map(inputs))
    
    # Then reduce the results
    if reduce_function == "sum":
        return sum(processed) + (initial_value or 0)
    elif reduce_function == "product":
        result = initial_value if initial_value is not None else 1
        for item in processed:
            result *= item
        return result
    elif reduce_function == "max":
        return max(processed) if processed else initial_value
    elif reduce_function == "min":
        return min(processed) if processed else initial_value
    else:
        raise ValueError(f"Unknown reduce function: {reduce_function}")


@app.local_entrypoint()
def example_parallel_execution():
    """Example of using parallel execution utilities"""
    # Example 1: Simple parallel map
    print("Example 1: Parallel map")
    inputs = list(range(10))
    results = parallel_map_executor.remote("process_item", inputs)
    print(f"Results: {results}")
    
    # Example 2: Parallel filter
    print("\nExample 2: Parallel filter")
    filtered = parallel_filter_executor.remote(
        inputs,
        "greater_than",
        {"type": "greater_than", "threshold": 5}
    )
    print(f"Filtered results: {filtered}")
    
    # Example 3: Parallel reduce
    print("\nExample 3: Parallel reduce")
    reduced = parallel_reduce_executor.remote(inputs, "sum")
    print(f"Sum: {reduced}")
