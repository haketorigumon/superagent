# Unified AI Agent System

A flexible and modular AI agent system designed for prompt-driven behavior and easy extensibility.

## Core Features

- **Modular Architecture**: The system is broken down into distinct components for entities, memory, plugins, and the core engine, making it easy to maintain and extend.
- **Prompt-Driven Behavior**: All agent behavior is guided by prompts, allowing for dynamic and adaptable responses.
- **Persistent Memory**: The system includes a persistent memory system that allows agents to retain information across sessions.
- **Plugin System**: Extend the system's capabilities with custom plugins.

## Architecture Overview

The system is composed of the following core components:
- `entities.py`: Defines the `UniversalEntity` class, which represents all objects in the system (agents, tasks, etc.).
- `engine.py`: Contains the `PromptEngine`, which manages and generates prompts for the AI.
- `memory.py`: Implements the `PersistentMemorySystem` for storing and retrieving agent memories.
- `plugins.py`: Manages the discovery and execution of plugins.
- `system.py`: The central `UnifiedSystem` class that orchestrates all the components.

## Getting Started

### Prerequisites

- Python 3.8+
- An LLM provider (e.g., Ollama, OpenAI)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-repo/novel-AI-agent.git
    cd novel-AI-agent
    ```

2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Configure your LLM provider in `unified_config.yaml`.

### Running the System

You can start the system using the main entry point:

```bash
python unified_main.py start
```

This will launch the system in interactive mode, where you can issue commands and interact with the AI agents.

### Running the Demo

To see a demonstration of the system's capabilities, run:

```bash
python demo_unified.py
```

### Running Tests

To run the test suite, use `pytest`:

```bash
pytest
```

## Basic Usage

The system can be controlled via the interactive command-line interface. Here are some of the available commands:

- `status`: Display the current system status.
- `create agent [name]`: Create a new AI agent.
- `list agents`: List all available agents.
- `list tasks`: List all tasks in the system.
- `[any text]`: Send a request to the system to be processed as a task.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.