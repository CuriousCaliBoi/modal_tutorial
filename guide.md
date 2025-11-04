# Modal Examples Guide

This guide provides a comprehensive map of all Python examples in this repository, explaining what each example accomplishes and when to use it.

## Table of Contents

1. [Getting Started Examples](#getting-started-examples)
2. [Core Features](#core-features)
3. [Web Services](#web-services)
4. [GPU & ML Examples](#gpu--ml-examples)
5. [Advanced Features](#advanced-features)
6. [Agent Integration Examples](#agent-integration-examples)

---

## Getting Started Examples

### `get_started.py`
**Purpose:** The simplest Modal function example - your first step with Modal.

**What it does:**
- Creates a basic Modal app
- Defines a simple function that squares a number
- Demonstrates the fundamental `@app.function()` decorator

**When to use:**
- First time using Modal
- Learning the basics of Modal's function decorator
- Understanding how to run code in the cloud

**Run with:** `modal run get_started.py`

---

### `hello_world.py`
**Purpose:** Demonstrates multiple execution modes: local, remote, and parallel.

**What it does:**
- Shows `.local()` - runs function in current Python process
- Shows `.remote()` - runs function on Modal's cloud infrastructure
- Shows `.map()` - runs function in parallel across many containers

**When to use:**
- Understanding the difference between local and remote execution
- Learning how to parallelize work with `.map()`
- Seeing real-time logging and error handling

**Run with:** `modal run hello_world.py`

---

### `custom_container.py`
**Purpose:** Shows how to add Python packages and dependencies to your container.

**What it does:**
- Creates a custom container image with NumPy installed
- Demonstrates `modal.Image` for dependency management
- Shows how to install packages with `pip_install()`

**When to use:**
- Adding Python packages to your Modal functions
- Understanding Modal's container image system
- Setting up custom environments

**Run with:** `modal run custom_container.py`

---

## Core Features

### `parallel_map.py`
**Purpose:** Demonstrates parallel processing using `.map()` to process multiple items simultaneously.

**What it does:**
- Processes multiple items in parallel across many containers
- Shows how to use `.map()` for batch processing
- Demonstrates efficient parallel computation patterns

**When to use:**
- Processing large datasets in parallel
- ETL pipelines that need to scale
- Any task that can be parallelized

**Run with:** `modal run parallel_map.py`

---

### `class_example.py`
**Purpose:** Shows how to use class-based functions with persistent state.

**What it does:**
- Uses `@app.cls()` to create a class-based function
- Demonstrates `@modal.enter()` for one-time setup (e.g., loading models)
- Shows `@modal.method()` for methods that can be called multiple times
- Uses `@modal.exit()` for cleanup

**When to use:**
- Loading expensive resources (like ML models) once and reusing them
- Maintaining state across multiple function calls
- Optimizing cold start times by preloading dependencies

**Run with:** `modal run class_example.py`

---

### `container_lifecycle.py`
**Purpose:** Explains container lifecycle and optimization strategies.

**What it does:**
- Demonstrates `@modal.enter()` - runs on container startup
- Demonstrates `@modal.exit()` - runs on container shutdown
- Shows `keep_warm` parameter to keep containers ready
- Explains `container_idle_timeout` for controlling container lifetime

**When to use:**
- Understanding when containers start and stop
- Optimizing for latency-sensitive applications
- Managing container lifecycle for cost optimization

**Run with:** `modal run container_lifecycle.py`

---

### `gpu_example.py`
**Purpose:** Basic example of using GPUs for computation.

**What it does:**
- Requests a GPU (`T4`) for the function
- Checks GPU availability
- Performs matrix multiplication on GPU using PyTorch
- Demonstrates GPU tensor operations

**When to use:**
- First time using GPUs on Modal
- Understanding GPU availability and device management
- Performing GPU-accelerated computations

**Run with:** `modal run gpu_example.py`

---

## Web Services

### `web_endpoint.py`
**Purpose:** Creates simple HTTP endpoints using Modal's web decorators.

**What it does:**
- Creates GET endpoints with `@modal.web_endpoint()`
- Creates POST endpoints for handling data
- Shows path parameters and query parameters
- Returns JSON responses

**When to use:**
- Creating simple webhooks
- Building REST API endpoints
- Exposing Modal functions as HTTP services

**Deploy with:** `modal deploy web_endpoint.py`

---

### `fastapi_app.py`
**Purpose:** Full-featured FastAPI application with ASGI support.

**What it does:**
- Creates a complete FastAPI application
- Shows GET/POST endpoints with path and query parameters
- Demonstrates request body handling
- Uses `@modal.asgi_app()` for ASGI compatibility

**When to use:**
- Building production REST APIs
- Creating complex web applications
- Integrating with existing FastAPI patterns

**Deploy with:** `modal deploy fastapi_app.py`

---

### `scheduled_function.py`
**Purpose:** Runs functions on a schedule (cron jobs and periodic tasks).

**What it does:**
- Uses `modal.Cron()` for scheduled execution
- Uses `modal.Period()` for interval-based execution
- Demonstrates cron syntax and periodic scheduling

**When to use:**
- Running periodic data processing jobs
- Scheduled data updates
- Automated tasks and cron jobs

**Deploy with:** `modal deploy scheduled_function.py`

---

## GPU & ML Examples

### `amazon_embeddings.py`
**Purpose:** High-performance example embedding 30 million Amazon reviews at scale using Qwen2-7B.

**What it does:**
- Creates embeddings for large text datasets (30M+ reviews)
- Uses Text Embeddings Inference (TEI) framework
- Demonstrates massive parallel processing on L40S GPUs
- Shows how to use Modal's `spawn` system for job queuing
- Processes at 575k tokens per second

**When to use:**
- Creating embeddings for semantic search
- Processing massive text datasets
- Building RAG (Retrieval Augmented Generation) systems
- Understanding large-scale parallel processing

**Key features:**
- Uses Modal Volumes for model caching
- Dynamic batching and auto-scaling
- Handles millions of inputs efficiently

**Run with:** `modal run --detach amazon_embeddings.py --dataset-subset raw_review_Books`

---

### `batched_whisper.py`
**Purpose:** Fast batch transcription of audio files using OpenAI's Whisper model with dynamic batching.

**What it does:**
- Transcribes audio files to text using Whisper Large V3
- Uses dynamic batching for 2.8x throughput improvement
- Processes multiple audio samples in parallel
- Demonstrates efficient GPU utilization with batching

**When to use:**
- Transcribing large audio datasets
- Processing podcasts, videos, or any audio content
- Building speech-to-text services
- Understanding dynamic batching optimization

**Key features:**
- Dynamic batching with `@modal.batched()` decorator
- Processes up to 64 samples per batch
- Uses Modal Volumes for model caching

**Run with:** `modal run batched_whisper.py`

---

### `vllm_inference.py`
**Purpose:** Deploy an OpenAI-compatible LLM service using vLLM with Qwen3-8B.

**What it does:**
- Serves large language models via OpenAI-compatible API
- Uses vLLM for high-performance inference
- Supports streaming responses
- Drop-in replacement for OpenAI API

**When to use:**
- Deploying your own LLM service
- Replacing OpenAI API with self-hosted models
- Building custom LLM applications
- Understanding LLM serving infrastructure

**Key features:**
- OpenAI-compatible API endpoints
- Supports streaming and non-streaming requests
- Fast boot mode for cold starts
- Uses H100 GPUs for optimal performance

**Deploy with:** `modal deploy vllm_inference.py`

---

### `llama_cpp.py`
**Purpose:** Run large and small language models (DeepSeek-R1, Phi-4) using llama.cpp.

**What it does:**
- Runs DeepSeek-R1 (671B parameters) on 4x L40S GPUs
- Runs Phi-4 (14B parameters) on CPU
- Compiles llama.cpp with CUDA support
- Demonstrates model quantization (1.58 bit for DeepSeek)

**When to use:**
- Running quantized LLMs efficiently
- Using llama.cpp for inference
- Deploying models that require specific quantization
- Understanding model size vs. hardware requirements

**Key features:**
- Supports both GPU and CPU inference
- Handles very large models (671B parameters)
- Uses Modal Volumes for model storage
- Can generate code (e.g., Flappy Bird game)

**Run with:** 
- `modal run llama_cpp.py --model="deepseek-r1"` (for DeepSeek-R1)
- `modal run llama_cpp.py --model="phi-4"` (for Phi-4)

---

### `flux.py`
**Purpose:** Fast image generation using Flux.1-schnell diffusion model with torch.compile optimization.

**What it does:**
- Generates images from text prompts using Flux.1-schnell
- Uses H100 GPUs for optimal performance
- Optional `torch.compile` for 2x+ speedup
- Returns images in under 1 second (with compilation) or ~1.2 seconds (without)

**When to use:**
- Building image generation services
- Understanding diffusion model inference
- Optimizing model performance with torch.compile
- Creating AI art applications

**Key features:**
- Uses torch.compile for maximum performance
- Caches compilation artifacts in Modal Volumes
- Supports both "schnell" (fast) and "dev" (quality) variants
- Requires Hugging Face token for gated model access

**Run with:** `modal run flux.py --compile` (for compiled version)

---

### `boltz_predict.py`
**Purpose:** Predict protein structures and binding affinities using Boltz-2 model.

**What it does:**
- Predicts 3D protein structures from sequences
- Calculates binding affinities between proteins and ligands
- Uses H100 GPUs for computational biology tasks
- Outputs Crystallographic Information Files (CIF) for visualization

**When to use:**
- Computational biology research
- Drug discovery and protein engineering
- Understanding protein-ligand interactions
- Scientific research requiring protein structure prediction

**Key features:**
- Uses Modal Volumes for model storage
- Auto-generates Multiple Sequence Alignments (MSA)
- Outputs can be visualized with Molstar Viewer
- Handles complex molecular structures

**Run with:** `modal run boltz_predict.py`

---

### `webrtc_yolo.py`
**Purpose:** Real-time object detection on webcam footage using WebRTC and YOLO.

**What it does:**
- Streams video from browser to Modal using WebRTC
- Performs real-time object detection with YOLO
- Streams annotated video back to browser
- Achieves 2-4ms inference per frame on A100 GPUs

**When to use:**
- Building real-time video processing applications
- Creating interactive video analysis tools
- Understanding WebRTC on Modal
- Building low-latency video streaming services

**Key features:**
- Uses WebRTC for peer-to-peer streaming
- Real-time object detection with YOLO
- Sub-30ms round-trip times
- Uses Modal's WebSocket support for signaling

**Note:** This example requires helper files (`modal_webrtc.py` and `yolo.py`) from the [Modal examples repository](https://github.com/modal-labs/modal-examples/tree/main/07_web_endpoints/webrtc). The file uses relative imports, so ensure all dependencies are in the same directory.

**Deploy with:** `modal serve webrtc_yolo.py`

---

## Advanced Features

### `volume_example.py`
**Purpose:** Demonstrates persistent storage across runs using Modal Volumes.

**What it does:**
- Creates and uses Modal Volumes for persistent data
- Writes data to volumes that persists across function runs
- Shows how to read and write files to volumes

**When to use:**
- Storing model weights and large files
- Caching data between runs
- Sharing data across multiple functions
- Managing datasets that are too large for container images

**Run with:** `modal run volume_example.py`

---

### `secrets_example.py`
**Purpose:** Securely manage API keys and credentials using Modal Secrets.

**What it does:**
- Shows how to create and use Modal Secrets
- Accesses secrets as environment variables
- Demonstrates secure credential management

**When to use:**
- Storing API keys and credentials
- Managing database passwords
- Accessing external services securely
- Following security best practices

**Setup:**
1. Create secret: `modal secret create my-secret API_KEY=your_key`
2. Run: `modal run secrets_example.py`

---

### `mount_files.py`
**Purpose:** Mount local files and directories into Modal containers.

**What it does:**
- Mounts local files into container filesystem
- Shows how to access local files in Modal functions
- Demonstrates file mounting patterns

**When to use:**
- Including local data files in your functions
- Sharing configuration files
- Accessing local datasets
- Including code or assets from your local machine

**Run with:** `modal run mount_files.py`

---

### `cursor_modal_bridge.py`
**Purpose:** Bridge between Cursor IDE and Modal for asynchronous task execution with status tracking.

**What it does:**
- Creates a FastAPI service that accepts tasks from Cursor
- Runs tasks asynchronously in Modal containers
- Provides status tracking via Modal Dict for task state
- Supports webhook callbacks for task completion
- Tracks progress, logs, and results in real-time

**When to use:**
- Integrating Cursor IDE with Modal compute
- Building async task execution systems
- Creating webhook-based task handlers
- Needing status tracking for long-running operations

**Key features:**
- Uses Modal Dict for persistent task state
- Fire-and-forget task spawning with `.spawn()`
- Optional callback URLs for task completion
- Real-time progress tracking and logging

**Deploy with:** `modal deploy examples/advanced/cursor_modal_bridge.py`

**Usage:**
- POST to `/cursor/tasks` with `{operation, params, callback_url?}`
- GET `/cursor/tasks/{run_id}` to check status
- Supports operations: `echo`, `sleep`, `square` (extendable)

---

## Agent Integration Examples

This section covers examples for integrating Cursor cloud agents and other AI agents with Modal's compute platform. These examples enable agents to dynamically deploy functions, spawn parallel tasks, manage resources, and coordinate multi-agent workflows.

**See also:** Detailed documentation in [`docs/AGENT_INTEGRATION.md`](docs/AGENT_INTEGRATION.md) and [`examples/agent_integration/README.md`](examples/agent_integration/README.md).

### `modal_agent_api.py`
**Purpose:** FastAPI service deployed on Modal that provides HTTP endpoints for agents to interact with Modal.

**What it does:**
- Exposes REST API endpoints for agent operations
- Handles dynamic function deployment from code strings
- Manages parallel task execution
- Provides task status tracking and webhook support
- Lists registered functions and tasks

**When to use:**
- Building agent-to-Modal integration systems
- Creating HTTP APIs for AI agents to use Modal
- Enabling agents to deploy and run functions dynamically
- Managing task queues and status tracking

**Key features:**
- `POST /deploy-function` - Deploy Modal functions dynamically
- `POST /run-parallel` - Execute functions in parallel
- `POST /spawn-task` - Spawn single tasks
- `GET /status/{task_id}` - Query task status
- `POST /webhook` - Register webhooks for async tasks

**Deploy with:** `modal deploy examples/agent_integration/modal_agent_api.py`

---

### `agent_tools.py`
**Purpose:** Core Modal functions that agents can call as tools for deploying and executing functions.

**What it does:**
- `deploy_modal_function()` - Deploys functions from code strings
- `spawn_parallel_tasks()` - Executes tasks in parallel using `.map()`
- `request_gpu_task()` - Executes tasks with GPU resources
- `get_task_result()` - Retrieves task results
- `coordinate_state()` - Shares state across agents via Modal Volumes

**When to use:**
- Agents need to deploy functions on-the-fly
- Parallel execution of agent tasks
- GPU-accelerated agent computations
- Multi-agent state coordination

**Key features:**
- Dynamic function compilation and registration
- Support for custom images, GPUs, secrets, and volumes
- Parallel task execution with `.map()`
- Volume-based state sharing

**Import and use:** Functions are Modal functions that can be called remotely

---

### `agent_modal_client.py`
**Purpose:** Python client library for agents to easily interact with Modal Agent API.

**What it does:**
- Provides `ModalAgentClient` class for async API interactions
- Handles HTTP requests with retries and error handling
- Includes `StatusMonitor` for tracking multiple tasks
- Simplifies common agent operations (deploy, run, query)

**When to use:**
- Building Python-based agent integrations
- Need async/await patterns for Modal operations
- Monitoring multiple concurrent tasks
- Implementing agent workflows with Modal

**Key features:**
- Async/await support
- Automatic retries with exponential backoff
- Task status monitoring utilities
- Webhook registration support

**Usage:**
```python
from agent_modal_client import ModalAgentClient
async with ModalAgentClient(api_url="...") as client:
    result = await client.deploy_function(...)
```

---

### `parallel_executor.py`
**Purpose:** Utilities for parallel execution patterns that agents commonly need.

**What it does:**
- `process_item()` - Generic item processor
- `batch_processor()` - Process items in batches
- `parallel_map_executor()` - Execute function in parallel using `.map()`
- `parallel_starmap_executor()` - Execute with multiple arguments using `.starmap()`
- `parallel_filter_executor()` - Filter items in parallel
- `parallel_reduce_executor()` - Perform parallel reduce operations

**When to use:**
- Agents need to process large datasets in parallel
- Implementing batch processing workflows
- Filtering or reducing data across containers
- Understanding parallel execution patterns

**Key features:**
- Multiple parallel execution patterns
- Batch processing utilities
- Generic, reusable components

---

### `resource_manager.py`
**Purpose:** Utilities for managing compute resources that agents can request.

**What it does:**
- `request_gpu_resources()` - Request GPU allocation (T4, A10G, A100, H100)
- `create_volume()` - Create or get Modal Volumes
- `mount_secrets()` - Mount Modal Secrets
- `request_compute_resources()` - Request CPU/memory resources
- `get_resource_status()` - Get current resource availability

**When to use:**
- Agents need to request specific compute resources
- Managing GPU allocations for ML tasks
- Creating persistent storage for agent workflows
- Mounting secrets for API access

**Key features:**
- GPU type selection and allocation
- Volume creation and management
- Secret mounting utilities
- Resource status monitoring

---

### `multi_agent_repo_creation.py`
**Purpose:** Example showing multiple agents creating repositories in parallel, coordinating via shared volume.

**What it does:**
- Coordinates multiple agents to create repositories simultaneously
- Uses Modal Volumes for shared state and coordination
- Demonstrates parallel repository creation patterns
- Aggregates results from multiple agents

**When to use:**
- Multiple agents need to create repositories in parallel
- Understanding multi-agent coordination patterns
- Building systems where agents work on separate tasks simultaneously
- Learning volume-based state sharing

**Key features:**
- Parallel agent execution
- Volume-based coordination
- Result aggregation

---

### `multi_agent_codegen.py`
**Purpose:** Example showing agents generating code components in parallel, then aggregating results.

**What it does:**
- Coordinates multiple agents to generate code in parallel
- Each agent generates different components (classes, functions, etc.)
- Aggregates generated code into a complete solution
- Uses Modal for parallel execution and coordination

**When to use:**
- Agents need to generate code in parallel
- Building multi-component systems where agents specialize
- Understanding code generation workflows
- Learning result aggregation patterns

**Key features:**
- Parallel code generation
- Component specialization per agent
- Result aggregation and integration

---

### `multi_agent_coordinator.py`
**Purpose:** Example showing master agent distributing work to worker agents via Modal.

**What it does:**
- Implements master-worker coordination pattern
- Master agent distributes tasks to worker agents
- Workers execute tasks in parallel on Modal
- Results are collected and aggregated by master

**When to use:**
- Building hierarchical agent systems
- Implementing task distribution patterns
- Coordinating multiple specialized agents
- Understanding master-worker architectures

**Key features:**
- Master-worker pattern
- Task distribution
- Result collection and aggregation

---

## Quick Reference by Use Case

### Machine Learning & AI
- **Model Inference:** `class_example.py`, `vllm_inference.py`, `llama_cpp.py`
- **Image Generation:** `flux.py`
- **Audio Processing:** `batched_whisper.py`
- **Computer Vision:** `webrtc_yolo.py`, `gpu_example.py`
- **Embeddings:** `amazon_embeddings.py`
- **Scientific Computing:** `boltz_predict.py`

### Web Development
- **Simple Endpoints:** `web_endpoint.py`
- **Full API:** `fastapi_app.py`
- **Real-time Streaming:** `webrtc_yolo.py`

### Data Processing
- **Parallel Processing:** `parallel_map.py`, `amazon_embeddings.py`
- **Batch Jobs:** `batched_whisper.py`, `scheduled_function.py`
- **Persistent Storage:** `volume_example.py`

### Getting Started
- **First Steps:** `get_started.py`, `hello_world.py`
- **Container Setup:** `custom_container.py`, `container_lifecycle.py`
- **State Management:** `class_example.py`

### Agent Integration
- **API Service:** `modal_agent_api.py`
- **Agent Tools:** `agent_tools.py`, `agent_modal_client.py`
- **Parallel Execution:** `parallel_executor.py`
- **Resource Management:** `resource_manager.py`
- **Multi-Agent Patterns:** `multi_agent_coordinator.py`, `multi_agent_codegen.py`, `multi_agent_repo_creation.py`
- **IDE Integration:** `cursor_modal_bridge.py`

---

## Performance Tips

1. **Cold Starts:** Use `@modal.enter()` in classes to preload models
2. **Parallelization:** Use `.map()` for processing multiple items
3. **Batching:** Use `@modal.batched()` for GPU-intensive tasks
4. **Caching:** Use Modal Volumes for model weights and large datasets
5. **Keep Warm:** Use `keep_warm` parameter for latency-sensitive apps
6. **GPU Selection:** Choose the right GPU for your workload (T4 for basic, H100 for heavy)

---

## Next Steps

1. Start with `get_started.py` to understand basics
2. Progress through `hello_world.py` and `custom_container.py`
3. Explore feature-specific examples based on your use case
4. Check [Modal Documentation](https://modal.com/docs) for more details
5. Join the [Modal Discord Community](https://discord.gg/modal) for help

---

## Resources

- [Modal Documentation](https://modal.com/docs)
- [Modal Examples Repository](https://github.com/modal-labs/modal-examples)
- [Modal Blog](https://modal.com/blog)
- [Modal Discord](https://discord.gg/modal)
- [Agent Integration Documentation](docs/AGENT_INTEGRATION.md)
- [Agent Integration Examples README](examples/agent_integration/README.md)

