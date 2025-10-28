# AI Agent System

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
