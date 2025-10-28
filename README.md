# Unified AI Agent System

A flexible, modular, and prompt-driven AI agent system designed for infinite capabilities and self-evolution.

## 🚀 Core Features

- **Modular Architecture**: The system is broken down into distinct components for entities, memory, plugins, and the core engine, making it easy to maintain and extend.
- **Prompt-Driven Behavior**: All agent behavior is guided by prompts, allowing for dynamic and adaptable responses.
- **Persistent Memory**: The system includes a persistent memory system that allows agents to retain information across sessions, ensuring no loss of context.
- **Dynamic Plugin System**: Extend the system's capabilities with custom plugins. The system can automatically discover, load, and even generate core plugins if they are missing.
- **Self-Evolution**: The system is designed to evolve and optimize itself over time by analyzing its performance and generating new configurations.

## 🏛️ Architecture Overview

The system is composed of the following core components:

-   **`main.py`**: The main entry point for the application, providing a command-line interface for starting and managing the system.
-   **`src/core/entities.py`**: Defines the `UniversalEntity` class, which is the fundamental data structure for all objects in the system (agents, tasks, memories, etc.).
-   **`src/core/engine.py`**: Contains the `PromptEngine`, which dynamically generates and manages prompts for the AI.
-   **`src/core/memory.py`**: Implements the `PersistentMemorySystem` for storing, retrieving, and indexing agent memories.
-   **`src/core/plugins.py`**: Manages the discovery, loading, and execution of plugins.
-   **`src/core/system.py`**: The central `UnifiedSystem` class that orchestrates all the components and manages the main event loop.
-   **`src/utils/unified_config.py`**: A flexible configuration system that supports loading from files, environment variables, and dynamic generation.

## 🔧 Getting Started

### Prerequisites

-   Python 3.8+
-   Pip

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/unified-ai-agent-system.git
    cd unified-ai-agent-system
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

The system is configured via a `config.yaml` file. If this file does not exist, a default configuration will be created when you first run the system. You can customize this file to change the behavior of the system, such as the LLM provider, API keys, and other settings.

### Usage

To start the system, run the following command:

```bash
python main.py start
```

This will start the system in interactive mode, where you can issue commands to create agents, list entities, and process requests.

#### Available Commands

-   **`help`**: Show a list of available commands.
-   **`status`**: Display the current status of the system.
-   **`create agent [name]`**: Create a new agent.
-   **`list agents`**: List all agents.
-   **`list tasks`**: List all tasks.
-   **`list entities`**: List all entities.
-   **`exit`**: Exit the system.

Any other text you enter will be treated as a general request to be processed by the system.

## 📦 Plugins

The system can be extended with plugins located in the `plugins` directory. Each plugin is a Python file that contains a `PLUGIN_INFO` dictionary and a `create_plugin` function.

### Core Plugins

The system includes a set of core plugins that are automatically generated if they are not found in the `plugins` directory:

-   **`task_executor`**: Executes tasks using the LLM.
-   **`capability_manager`**: Manages and generates new capabilities.
-   **`workflow_engine`**: Manages complex workflows.
-   **`resource_manager`**: Manages system resources.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
