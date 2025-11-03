# Modal Tutorial Examples

This directory contains comprehensive examples covering all major Modal features and use cases.

## 📚 Table of Contents

### Getting Started
- **`get_started.py`** - Basic Modal function (from official tutorial Step 1)
- **`hello_world.py`** - Local, remote, and parallel execution
- **`custom_container.py`** - Custom container with dependencies (from official tutorial Step 2)

### Core Features
- **`parallel_map.py`** - Parallel processing with `.map()`
- **`class_example.py`** - Class-based functions with persistent state
- **`container_lifecycle.py`** - Understanding container lifecycle and optimization

### Advanced Features
- **`gpu_example.py`** - Using GPUs for computation
- **`web_endpoint.py`** - Creating HTTP endpoints
- **`fastapi_app.py`** - Full FastAPI application with ASGI
- **`scheduled_function.py`** - Cron jobs and periodic tasks
- **`volume_example.py`** - Persistent storage across runs
- **`secrets_example.py`** - Managing API keys and secrets
- **`mount_files.py`** - Mounting local files and directories

## 🚀 Quick Start

### Installation
```bash
pip install modal
modal setup
```

### Running Examples

**Run a script:**
```bash
modal run get_started.py
modal run hello_world.py
modal run gpu_example.py
```

**Deploy a web service:**
```bash
modal deploy web_endpoint.py
modal deploy fastapi_app.py
```

**Deploy scheduled functions:**
```bash
modal deploy scheduled_function.py
```

## 📖 Example Details

### Basic Examples

#### get_started.py
The simplest Modal function that squares a number.
```bash
modal run get_started.py
```

#### hello_world.py
Demonstrates three execution modes:
- `.local()` - Run in current Python process
- `.remote()` - Run on Modal
- `.map()` - Run in parallel on Modal

#### custom_container.py
Shows how to add Python packages (NumPy) to your container environment.

### GPU Computing

#### gpu_example.py
Request GPUs for your functions:
```python
@app.function(gpu="T4")  # or "A100", "A10G", etc.
def gpu_function():
    import torch
    # Use GPU for computation
```

### Web Services

#### web_endpoint.py
Simple HTTP endpoints:
```python
@app.function()
@modal.web_endpoint()
def hello():
    return {"message": "Hello!"}
```

#### fastapi_app.py
Full-featured FastAPI application with:
- GET/POST endpoints
- Path parameters
- Query parameters
- Request body handling

Deploy with: `modal deploy fastapi_app.py`

### Scheduled Jobs

#### scheduled_function.py
Run functions on a schedule:
```python
@app.function(schedule=modal.Cron("0 * * * *"))  # Every hour
def hourly_job():
    print("Running!")

@app.function(schedule=modal.Period(minutes=5))  # Every 5 minutes
def frequent_job():
    print("Running!")
```

### Persistent Storage

#### volume_example.py
Store data that persists across runs:
```python
volume = modal.Volume.from_name("my-volume", create_if_missing=True)

@app.function(volumes={"/data": volume})
def write_data():
    with open("/data/file.txt", "w") as f:
        f.write("Persistent data!")
```

### Secrets Management

#### secrets_example.py
Securely use API keys and credentials:
```bash
# Create a secret
modal secret create my-secret API_KEY=your_key

# Use it in your function
@app.function(secrets=[modal.Secret.from_name("my-secret")])
def use_secret():
    import os
    api_key = os.environ["API_KEY"]
```

### Parallel Processing

#### parallel_map.py
Process items in parallel:
```python
@app.function()
def process(item):
    return item ** 2

results = list(process.map(range(100)))
```

### Class-Based Functions

#### class_example.py
Use classes to maintain state:
```python
@app.cls()
class Model:
    @modal.enter()
    def load_model(self):
        # Load once when container starts
        self.model = load_large_model()
    
    @modal.method()
    def predict(self, x):
        # Reuse loaded model
        return self.model.predict(x)
```

### Container Lifecycle

#### container_lifecycle.py
Understand and optimize container behavior:
- `@modal.enter()` - Runs on container startup
- `@modal.exit()` - Runs on container shutdown
- `keep_warm` - Keep containers ready
- `container_idle_timeout` - Control container lifetime

### File Mounting

#### mount_files.py
Mount local files into containers:
```python
@app.function(
    mounts=[modal.Mount.from_local_file("data.txt", "/data/data.txt")]
)
def process_file():
    with open("/data/data.txt") as f:
        return f.read()
```

## 🎯 Common Use Cases

### Machine Learning
- Use `gpu_example.py` for GPU-accelerated training
- Use `class_example.py` for model inference with warm loading
- Use `volume_example.py` to store model checkpoints

### Web Applications
- Use `fastapi_app.py` for REST APIs
- Use `web_endpoint.py` for simple webhooks
- Use `secrets_example.py` for API keys

### Data Processing
- Use `parallel_map.py` for ETL pipelines
- Use `volume_example.py` for data persistence
- Use `scheduled_function.py` for periodic data updates

### Background Jobs
- Use `scheduled_function.py` for cron jobs
- Use `container_lifecycle.py` for optimized job execution
- Use `volume_example.py` for job state management

## 🔧 Tips & Best Practices

1. **Container Images**: Define dependencies once with `modal.Image`
2. **Parallelization**: Use `.map()` for processing multiple items
3. **State Management**: Use classes with `@modal.enter()` for expensive setup
4. **Cold Starts**: Use `keep_warm` for latency-sensitive applications
5. **Secrets**: Never hardcode credentials, always use `modal.Secret`
6. **Volumes**: Use volumes for data larger than a few hundred MB
7. **GPUs**: Specify the exact GPU type you need to optimize cost

## 📝 Next Steps

1. Start with `get_started.py` to understand basics
2. Progress through `hello_world.py` and `custom_container.py`
3. Explore feature-specific examples based on your use case
4. Check [Modal documentation](https://modal.com/docs) for more details

## 🆘 Troubleshooting

**Authentication Issues:**
```bash
modal setup
```

**Container Not Found:**
Make sure you've installed all required packages in your image

**Function Not Found:**
Ensure you're using `modal run` (not `python`) to execute scripts

**Secrets Not Working:**
Create secrets first: `modal secret create secret-name KEY=value`

## 📚 Resources

- [Modal Documentation](https://modal.com/docs)
- [Modal Examples](https://modal.com/docs/examples)
- [Modal Discord Community](https://discord.gg/modal)


