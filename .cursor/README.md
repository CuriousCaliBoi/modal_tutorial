# Cursor Background Agent Configuration

This directory contains configuration for running Cursor background agents with this Modal tutorial repository.

## Setup

The `.cursor/environment.json` file configures the environment that background agents will use:

- **setup_commands**: Install Modal and dependencies
- **start_commands**: Optional commands to start services
- **environment_variables**: Custom environment variables

## Usage

### Running a Background Agent in Cursor IDE

1. **Open Background Agent Sidebar**
   - Click the background agent icon in Cursor's sidebar
   - Or press `Ctrl+E` (Windows/Linux) or `Cmd+E` (Mac)

2. **Submit Tasks**
   - Type your task in the agent prompt
   - Example: "Review and improve all Modal examples for best practices"
   - The agent will clone this repo and work on the task

3. **Monitor Progress**
   - Watch the agent's progress in the Background Agent Sidebar
   - Agent will create branches and push changes as it works

### Agent Capabilities

The Modal Tutorial Specialist agent can:
- Review and improve Modal example code
- Optimize container configurations
- Enhance documentation
- Explain Modal concepts
- Detect and fix anti-patterns
- Suggest cost optimizations

### Example Tasks

```
"Add a new example showing Modal with TensorFlow and GPUs"
"Optimize the container lifecycle examples for better performance"
"Review all examples for Modal best practices and suggest improvements"
"Create documentation explaining when to use volumes vs mounts"
"Add error handling to all web endpoint examples"
```

## Configuration Details

- Modal is installed via `requirements.txt`
- Setup runs `modal setup || true` (won't fail if already configured)
- Environment is Python 3 with Modal SDK

## GitHub Integration

Ensure Cursor's GitHub app has read-write access to your repositories so background agents can:
- Clone the repository
- Create branches
- Push changes
- Create pull requests

## Resources

- [Cursor Background Agents Docs](https://docs.cursor.com/en/background-agents)
- [Modal Documentation](https://modal.com/docs)

