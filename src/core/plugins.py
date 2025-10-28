import importlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import aiofiles

logger = logging.getLogger(__name__)


class PluginSystem:
    """
    A dynamic plugin system for infinite extensibility.

    This class is responsible for discovering, loading, and managing plugins
    within the AI system. It allows for the dynamic extension of the system's
    capabilities without modifying the core codebase.

    Attributes:
        system: A reference to the main UnifiedSystem instance.
        plugins_dir: The directory where plugins are located.
        loaded_plugins: A dictionary of loaded plugin instances.
        plugin_registry: A dictionary of discovered plugins and their metadata.
        plugin_dependencies: A dictionary tracking plugin dependencies.
    """

    def __init__(self, system, plugins_dir: str = "plugins"):
        """
        Initializes the PluginSystem.

        Args:
            system: A reference to the main UnifiedSystem instance.
            plugins_dir: The directory where plugins are located.
        """
        self.system = system
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_plugins: Dict[str, Any] = {}
        self.plugin_registry: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        """
        Initializes the plugin system.

        This method discovers available plugins and loads a set of core plugins
        that are essential for the system's operation.
        """
        await self._discover_plugins()
        await self._load_core_plugins()

    async def _discover_plugins(self):
        """Discovers available plugins in the plugins directory."""
        for plugin_file in self.plugins_dir.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue

            plugin_name = plugin_file.stem
            try:
                spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Check for plugin metadata
                if hasattr(module, "PLUGIN_INFO"):
                    self.plugin_registry[plugin_name] = module.PLUGIN_INFO
                    self.plugin_registry[plugin_name]["module"] = module
                    self.plugin_registry[plugin_name]["file"] = plugin_file

            except Exception as e:
                logger.error(f"Error discovering plugin {plugin_name}: {e}")

    async def _load_core_plugins(self):
        """
        Loads the core system plugins.

        If a core plugin is not found, it is generated from a template.
        """
        core_plugins = [
            "task_executor",
            "capability_manager",
            "workflow_engine",
            "resource_manager",
        ]

        for plugin_name in core_plugins:
            if plugin_name in self.plugin_registry:
                await self.load_plugin(plugin_name)
            else:
                await self._generate_core_plugin(plugin_name)

    async def _generate_core_plugin(self, plugin_name: str):
        """
        Generates a core plugin if it doesn't exist.

        This method uses a set of predefined templates to create the Python
        source code for core plugins. This ensures that the system has a
        baseline set of capabilities even if no custom plugins are installed.

        Args:
            plugin_name: The name of the core plugin to generate.
        """
        try:
            template_file = (
                Path(__file__).parent
                / "plugin_templates"
                / f"{plugin_name}.py.template"
            )
            if template_file.exists():
                async with aiofiles.open(template_file, "r", encoding="utf-8") as f:
                    template_content = await f.read()

                plugin_file = self.plugins_dir / f"{plugin_name}.py"
                async with aiofiles.open(plugin_file, "w", encoding="utf-8") as f:
                    await f.write(template_content)

                # Reload plugin registry
                await self._discover_plugins()
                await self.load_plugin(plugin_name)
            else:
                logger.warning(f"Core plugin template not found: {plugin_name}")
        except Exception as e:
            logger.error(f"Error generating core plugin {plugin_name}: {e}")

    async def load_plugin(self, plugin_name: str) -> bool:
        """
        Loads a specific plugin.

        This method loads the plugin's module, checks its dependencies, and
        creates an instance of the plugin.

        Args:
            plugin_name: The name of the plugin to load.

        Returns:
            True if the plugin was loaded successfully, False otherwise.
        """
        if plugin_name not in self.plugin_registry:
            return False

        try:
            plugin_info = self.plugin_registry[plugin_name]
            module = plugin_info["module"]

            # Check dependencies
            for dep in plugin_info.get("dependencies", []):
                if dep not in self.loaded_plugins:
                    await self.load_plugin(dep)

            # Create plugin instance
            if hasattr(module, "create_plugin"):
                plugin_instance = module.create_plugin(self.system)
                self.loaded_plugins[plugin_name] = plugin_instance
                logger.info(f"Loaded plugin: {plugin_name}")
                return True

        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")

        return False

    async def get_plugin(self, plugin_name: str) -> Optional[Any]:
        """
        Gets a loaded plugin by its name.

        Args:
            plugin_name: The name of the plugin to get.

        Returns:
            The plugin instance, or None if the plugin is not loaded.
        """
        return self.loaded_plugins.get(plugin_name)

    async def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """
        Lists all available plugins.

        Returns:
            A dictionary of available plugins, including their metadata and
            whether they are currently loaded.
        """
        return {
            name: {**info, "loaded": name in self.loaded_plugins}
            for name, info in self.plugin_registry.items()
        }
