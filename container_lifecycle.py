import modal

app = modal.App("example-lifecycle")


@app.function()
def simple_function(x):
    """
    Container lifecycle:
    1. Container is created (cold start)
    2. Function executes
    3. Container may be kept alive for reuse (warm)
    4. After idle timeout, container shuts down
    """
    result = x * 2
    print(f"Processing {x} = {result}")
    return result


@app.cls()
class StatefulContainer:
    """
    Demonstrates container lifecycle with state.
    The state persists across invocations while the container is alive.
    """
    
    @modal.enter()
    def startup(self):
        """Called once when container starts (cold start)"""
        print("🚀 Container starting up...")
        self.call_count = 0
        self.state = {"initialized": True}
    
    @modal.method()
    def process(self, data):
        """Called for each invocation"""
        self.call_count += 1
        print(f"📊 Processing call #{self.call_count}")
        return {"call_number": self.call_count, "data": data}
    
    @modal.exit()
    def shutdown(self):
        """Called when container is shutting down"""
        print(f"👋 Container shutting down after {self.call_count} calls")


@app.function(
    # Keep container alive for 5 minutes after last use
    keep_warm=1,  # Keep 1 container always warm
    container_idle_timeout=300,  # 5 minutes
)
def optimized_function(x):
    """
    This function has optimized settings:
    - keep_warm: Keeps containers ready for instant execution
    - container_idle_timeout: How long to keep container alive
    """
    import time
    time.sleep(0.1)  # Simulate some work
    return x ** 2


@app.local_entrypoint()
def main():
    print("=== Testing simple function ===")
    simple_function.remote(5)
    
    print("\n=== Testing stateful container ===")
    container = StatefulContainer()
    
    # Multiple calls will reuse the same container
    for i in range(3):
        result = container.process.remote(f"data_{i}")
        print(f"Result: {result}")
    
    print("\n=== Testing optimized function ===")
    results = list(optimized_function.map(range(5)))
    print(f"Results: {results}")


