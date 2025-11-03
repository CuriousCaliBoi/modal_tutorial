import modal

app = modal.App("example-parallel-map")


@app.function()
def process_item(item):
    """Process a single item - this runs in parallel on multiple containers"""
    import time
    
    # Simulate some work
    time.sleep(1)
    result = item ** 2
    print(f"Processed {item} -> {result}")
    return result


@app.function()
def fibonacci(n):
    """Calculate Fibonacci number - useful for demonstrating parallel computation"""
    if n <= 1:
        return n
    return fibonacci.local(n - 1) + fibonacci.local(n - 2)


@app.local_entrypoint()
def main():
    # Process items in parallel using map
    print("Processing items in parallel...")
    items = range(10)
    
    # This will run on multiple containers simultaneously
    results = list(process_item.map(items))
    print(f"Results: {results}")
    
    # Calculate Fibonacci numbers in parallel
    print("\nCalculating Fibonacci numbers in parallel...")
    fib_inputs = [10, 15, 20, 25]
    fib_results = list(fibonacci.map(fib_inputs))
    
    for n, result in zip(fib_inputs, fib_results):
        print(f"Fibonacci({n}) = {result}")
    
    # You can also use starmap for multiple arguments
    print("\nDemonstrating parallel computation...")
    total = sum(results)
    print(f"Sum of all results: {total}")


