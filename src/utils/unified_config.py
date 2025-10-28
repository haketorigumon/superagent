"""
Configuration System - Completely flexible and prompt-driven configuration
Eliminates hardcoding through dynamic configuration generation and adaptation
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PromptConfig:
    """
    Configuration for prompt-driven behavior.

    Attributes:
        templates_dir: The directory for prompt templates.
        system_prompts_dir: The directory for system prompts.
        auto_generate: Whether to automatically generate new prompts.
        optimization_enabled: Whether to enable prompt optimization.
        fallback_enabled: Whether to use fallback prompts.
        cache_prompts: Whether to cache generated prompts.
    """
    templates_dir: str = "prompts"
    system_prompts_dir: str = "system"
    auto_generate: bool = True
    optimization_enabled: bool = True
    fallback_enabled: bool = True
    cache_prompts: bool = True


@dataclass
class MemoryConfig:
    """
    Configuration for the persistent memory system.

    Attributes:
        storage_dir: The directory for persistent memory storage.
        auto_consolidation: Whether to automatically consolidate memories.
        max_working_memory: The maximum size of the working memory.
        importance_threshold: The threshold for memory importance.
        retention_days: The number of days to retain memories.
        backup_enabled: Whether to enable memory backups.
        compression_enabled: Whether to enable memory compression.
    """
    storage_dir: str = "persistent_memory"
    auto_consolidation: bool = True
    max_working_memory: int = 1000
    importance_threshold: float = 0.3
    retention_days: int = 365
    backup_enabled: bool = True
    compression_enabled: bool = True


@dataclass
class PluginConfig:
    """
    Configuration for the plugin system.

    Attributes:
        plugins_dir: The directory for plugins.
        auto_discovery: Whether to automatically discover plugins.
        auto_generation: Whether to automatically generate core plugins.
        dependency_resolution: Whether to resolve plugin dependencies.
        hot_reload: Whether to enable hot reloading of plugins.
        sandbox_enabled: Whether to run plugins in a sandbox.
    """
    plugins_dir: str = "plugins"
    auto_discovery: bool = True
    auto_generation: bool = True
    dependency_resolution: bool = True
    hot_reload: bool = True
    sandbox_enabled: bool = True


@dataclass
class TaskConfig:
    """
    Configuration for task management.

    Attributes:
        max_concurrent_tasks: The maximum number of concurrent tasks.
        task_timeout: The timeout for tasks in seconds.
        auto_retry: Whether to automatically retry failed tasks.
        max_retries: The maximum number of retries for a task.
        priority_scheduling: Whether to use priority scheduling.
        load_balancing: Whether to enable load balancing for tasks.
    """
    max_concurrent_tasks: int = 10
    task_timeout: int = 300
    auto_retry: bool = True
    max_retries: int = 3
    priority_scheduling: bool = True
    load_balancing: bool = True


@dataclass
class LLMConfig:
    """
    Configuration for LLM integration.

    Attributes:
        provider: The name of the LLM provider.
        model: The name of the LLM model.
        base_url: The base URL for the LLM API.
        api_key: The API key for the LLM provider.
        temperature: The temperature setting for the LLM.
        max_tokens: The maximum number of tokens to generate.
        timeout: The timeout for LLM requests in seconds.
        retry_attempts: The number of retry attempts for LLM requests.
        fallback_providers: A list of fallback LLM providers.
    """
    provider: str = "ollama"
    model: str = "llama3"
    base_url: Optional[str] = "http://localhost:11434"
    api_key: Optional[str] = None
    temperature: float = 0.8
    max_tokens: int = 4096
    timeout: int = 60
    retry_attempts: int = 3
    fallback_providers: List[str] = field(default_factory=list)


@dataclass
class SystemConfig:
    """
    Configuration for system behavior.

    Attributes:
        auto_evolution: Whether to enable automatic system evolution.
        self_optimization: Whether to enable self-optimization of the system.
        continuous_learning: Whether to enable continuous learning.
        adaptive_behavior: Whether to enable adaptive behavior.
        error_recovery: Whether to enable automatic error recovery.
        performance_monitoring: Whether to enable performance monitoring.
    """
    auto_evolution: bool = True
    self_optimization: bool = True
    continuous_learning: bool = True
    adaptive_behavior: bool = True
    error_recovery: bool = True
    performance_monitoring: bool = True


@dataclass
class SecurityConfig:
    """
    Configuration for security settings.

    Attributes:
        sandbox_execution: Whether to execute code in a sandbox.
        code_validation: Whether to validate code before execution.
        resource_limits: Whether to enforce resource limits.
        access_control: Whether to enable access control.
        audit_logging: Whether to enable audit logging.
    """
    sandbox_execution: bool = True
    code_validation: bool = True
    resource_limits: bool = True
    access_control: bool = True
    audit_logging: bool = True


@dataclass
class WebConfig:
    """
    Configuration for the web interface.

    Attributes:
        host: The host for the web interface.
        port: The port for the web interface.
        enable_cors: Whether to enable CORS.
        real_time_updates: Whether to enable real-time updates.
        authentication: Whether to enable authentication.
        ssl_enabled: Whether to enable SSL.
    """
    host: str = "0.0.0.0"
    port: int = 12000
    enable_cors: bool = True
    real_time_updates: bool = True
    authentication: bool = False
    ssl_enabled: bool = False


class UnifiedConfig:
    """
    A unified configuration system with infinite flexibility.

    This class manages the configuration of the entire system. It can load
    configuration from a file, override it with environment variables, and
    even generate new configurations dynamically using an LLM.

    Attributes:
        config_file: The path to the configuration file.
        config_data: The raw configuration data loaded from the file.
        environment_overrides: A dictionary of configuration overrides from
                               environment variables.
        runtime_overrides: A dictionary of configuration overrides set at
                           runtime.
        prompt: The configuration for the prompt engine.
        memory: The configuration for the memory system.
        plugin: The configuration for the plugin system.
        task: The configuration for task management.
        llm: The configuration for the LLM client.
        system: The configuration for general system behavior.
        security: The configuration for security settings.
        web: The configuration for the web interface.
        dynamic_configs: A dictionary for storing dynamically generated
                         configurations.
        config_history: A list of dictionaries tracking configuration changes.
    """
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Initializes the UnifiedConfig.

        Args:
            config_file: The path to the configuration file.
        """
        self.config_file = Path(config_file)
        self.config_data: Dict[str, Any] = {}
        self.environment_overrides: Dict[str, Any] = {}
        self.runtime_overrides: Dict[str, Any] = {}
        
        # Core configuration sections
        self.prompt = PromptConfig()
        self.memory = MemoryConfig()
        self.plugin = PluginConfig()
        self.task = TaskConfig()
        self.llm = LLMConfig()
        self.system = SystemConfig()
        self.security = SecurityConfig()
        self.web = WebConfig()
        
        # Dynamic configuration
        self.dynamic_configs: Dict[str, Any] = {}
        self.config_history: List[Dict[str, Any]] = []
        
    async def initialize(self):
        """Initializes the configuration system."""
        await self._load_config()
        await self._load_environment_overrides()
        await self._apply_configurations()
        await self._validate_configuration()
        
    async def _load_config(self):
        """Loads the configuration from a file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                self.config_data = {}
        else:
            # Create default configuration
            await self._create_default_config()
    
    async def _create_default_config(self):
        """Creates a default configuration file."""
        default_config = {
            "prompt": asdict(self.prompt),
            "memory": asdict(self.memory),
            "plugin": asdict(self.plugin),
            "task": asdict(self.task),
            "llm": asdict(self.llm),
            "system": asdict(self.system),
            "security": asdict(self.security),
            "web": asdict(self.web),
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "description": "Unified AI Agent System Configuration"
            }
        }
        
        self.config_data = default_config
        await self.save_config()
    
    async def _load_environment_overrides(self):
        """Loads configuration overrides from environment variables."""
        env_prefix = "UNIFIED_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower()
                
                # Parse nested keys (e.g., UNIFIED_LLM_PROVIDER -> llm.provider)
                key_parts = config_key.split('_')
                
                # Try to parse as JSON first, then as string
                try:
                    parsed_value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    parsed_value = value
                
                # Set nested configuration
                current = self.environment_overrides
                for part in key_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[key_parts[-1]] = parsed_value
    
    async def _apply_configurations(self):
        """Applies configurations to the dataclass instances."""
        # Apply base configuration
        self._apply_config_section("prompt", self.prompt)
        self._apply_config_section("memory", self.memory)
        self._apply_config_section("plugin", self.plugin)
        self._apply_config_section("task", self.task)
        self._apply_config_section("llm", self.llm)
        self._apply_config_section("system", self.system)
        self._apply_config_section("security", self.security)
        self._apply_config_section("web", self.web)
        
        # Apply environment overrides
        self._apply_overrides(self.environment_overrides)
        
        # Apply runtime overrides
        self._apply_overrides(self.runtime_overrides)
    
    def _apply_config_section(self, section_name: str, config_obj):
        """
        Applies configuration to a specific section.

        Args:
            section_name: The name of the configuration section.
            config_obj: The dataclass instance for the section.
        """
        if section_name in self.config_data:
            section_data = self.config_data[section_name]
            for key, value in section_data.items():
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
    
    def _apply_overrides(self, overrides: Dict[str, Any]):
        """
        Applies configuration overrides.

        Args:
            overrides: A dictionary of overrides to apply.
        """
        for section, values in overrides.items():
            if hasattr(self, section):
                config_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    async def _validate_configuration(self):
        """Validates the configuration settings."""
        validation_errors = []
        
        # Validate LLM configuration
        if not self.llm.provider:
            validation_errors.append("LLM provider is required")
        
        if not self.llm.model:
            validation_errors.append("LLM model is required")
        
        # Validate directories
        for dir_attr in ['templates_dir', 'storage_dir', 'plugins_dir']:
            for config_obj in [self.prompt, self.memory, self.plugin]:
                if hasattr(config_obj, dir_attr):
                    dir_path = Path(getattr(config_obj, dir_attr))
                    try:
                        dir_path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        validation_errors.append(f"Cannot create directory {dir_path}: {e}")
        
        # Validate numeric ranges
        if self.task.max_concurrent_tasks <= 0:
            validation_errors.append("max_concurrent_tasks must be positive")
        
        if self.memory.importance_threshold < 0 or self.memory.importance_threshold > 1:
            validation_errors.append("importance_threshold must be between 0 and 1")
        
        if validation_errors:
            logger.warning(f"Configuration validation warnings: {validation_errors}")
    
    async def save_config(self):
        """Saves the current configuration to a file."""
        config_to_save = {
            "prompt": asdict(self.prompt),
            "memory": asdict(self.memory),
            "plugin": asdict(self.plugin),
            "task": asdict(self.task),
            "llm": asdict(self.llm),
            "system": asdict(self.system),
            "security": asdict(self.security),
            "web": asdict(self.web),
            "dynamic": self.dynamic_configs,
            "metadata": {
                "updated_at": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        }
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_to_save, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def set_runtime_override(self, section: str, key: str, value: Any):
        """
        Sets a runtime configuration override.

        Args:
            section: The configuration section.
            key: The configuration key.
            value: The configuration value.
        """
        if section not in self.runtime_overrides:
            self.runtime_overrides[section] = {}
        
        self.runtime_overrides[section][key] = value
        
        # Apply immediately
        if hasattr(self, section):
            config_obj = getattr(self, section)
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
    
    def get_config_value(self, section: str, key: str, default: Any = None) -> Any:
        """
        Gets a configuration value with a fallback.

        This method checks for the value in the following order:
        1. Runtime overrides
        2. Environment overrides
        3. Main configuration
        4. Dynamic configurations

        Args:
            section: The configuration section.
            key: The configuration key.
            default: The default value to return if the key is not found.

        Returns:
            The configuration value.
        """
        # Check runtime overrides first
        if (section in self.runtime_overrides and 
            key in self.runtime_overrides[section]):
            return self.runtime_overrides[section][key]
        
        # Check environment overrides
        if (section in self.environment_overrides and 
            key in self.environment_overrides[section]):
            return self.environment_overrides[section][key]
        
        # Check main configuration
        if hasattr(self, section):
            config_obj = getattr(self, section)
            if hasattr(config_obj, key):
                return getattr(config_obj, key)
        
        # Check dynamic configurations
        if section in self.dynamic_configs and key in self.dynamic_configs[section]:
            return self.dynamic_configs[section][key]
        
        return default
    
    def set_dynamic_config(self, section: str, key: str, value: Any):
        """
        Sets a dynamic configuration value.

        Args:
            section: The configuration section.
            key: The configuration key.
            value: The configuration value.
        """
        if section not in self.dynamic_configs:
            self.dynamic_configs[section] = {}
        
        self.dynamic_configs[section][key] = value
    
    def get_api_key_for_provider(self, provider: str) -> Optional[str]:
        """
        Gets the API key for a specific provider.

        This method checks for the API key in the LLM configuration,
        environment variables, and dynamic configuration.

        Args:
            provider: The name of the LLM provider.

        Returns:
            The API key, or None if not found.
        """
        # Check LLM config first
        if self.llm.api_key:
            return self.llm.api_key
        
        # Check environment variables
        env_key = f"{provider.upper()}_API_KEY"
        if env_key in os.environ:
            return os.environ[env_key]
        
        # Check dynamic config
        api_key = self.get_config_value("api_keys", provider.lower())
        if api_key:
            return api_key
        
        return None
    
    async def generate_config_for_purpose(self, purpose: str, context: Dict[str, Any], 
                                        llm_client=None) -> Dict[str, Any]:
        """
        Generates configuration dynamically for a specific purpose.

        This method uses an LLM to generate configuration adjustments based on
        a given purpose and context.

        Args:
            purpose: The purpose for which to generate the configuration.
            context: The context for the configuration generation.
            llm_client: An optional LLM client.

        Returns:
            A dictionary of configuration adjustments.
        """
        if not llm_client:
            return {}
        
        try:
            prompt = f"""You are a configuration generator. Create optimal configuration settings for the following purpose:

Purpose: {purpose}
Context: {json.dumps(context, indent=2)}
Current Config: {json.dumps(self.to_dict(), indent=2)}

Generate configuration adjustments that would optimize the system for this purpose.
Return only the configuration changes as a JSON object.

Configuration adjustments:"""
            
            response = await llm_client.generate(prompt, "You are a helpful assistant.")
            
            if response:
                try:
                    config_adjustments = json.loads(response)
                    
                    # Apply adjustments to dynamic config
                    for section, values in config_adjustments.items():
                        for key, value in values.items():
                            self.set_dynamic_config(section, key, value)
                    
                    return config_adjustments
                except json.JSONDecodeError:
                    logger.error("Failed to parse generated configuration")
            
        except Exception as e:
            logger.error(f"Error generating configuration: {e}")
        
        return {}
    
    async def optimize_config_based_on_metrics(self, metrics: Dict[str, Any], 
                                             llm_client=None) -> bool:
        """
        Optimizes the configuration based on performance metrics.

        This method uses an LLM to suggest configuration improvements based on
        the provided performance metrics.

        Args:
            metrics: A dictionary of performance metrics.
            llm_client: An optional LLM client.

        Returns:
            True if the configuration was optimized, False otherwise.
        """
        if not llm_client:
            return False
        
        try:
            prompt = f"""You are a configuration optimizer. Analyze the following performance metrics and suggest configuration improvements:

Current Metrics: {json.dumps(metrics, indent=2)}
Current Config: {json.dumps(self.to_dict(), indent=2)}

Identify performance bottlenecks and suggest specific configuration changes to improve:
1. Task execution speed
2. Memory efficiency
3. Resource utilization
4. Error rates
5. Overall system performance

Return configuration changes as a JSON object.

Optimization suggestions:"""
            
            response = await llm_client.generate(prompt, "You are a helpful assistant.")
            
            if response:
                try:
                    optimizations = json.loads(response)
                    
                    # Record optimization in history
                    self.config_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "type": "optimization",
                        "metrics": metrics,
                        "changes": optimizations
                    })
                    
                    # Apply optimizations
                    for section, values in optimizations.items():
                        for key, value in values.items():
                            self.set_runtime_override(section, key, value)
                    
                    await self.save_config()
                    return True
                    
                except json.JSONDecodeError:
                    logger.error("Failed to parse optimization suggestions")
            
        except Exception as e:
            logger.error(f"Error optimizing configuration: {e}")
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the configuration to a dictionary.

        Returns:
            A dictionary representation of the configuration.
        """
        return {
            "prompt": asdict(self.prompt),
            "memory": asdict(self.memory),
            "plugin": asdict(self.plugin),
            "task": asdict(self.task),
            "llm": asdict(self.llm),
            "system": asdict(self.system),
            "security": asdict(self.security),
            "web": asdict(self.web),
            "dynamic": self.dynamic_configs,
            "environment_overrides": self.environment_overrides,
            "runtime_overrides": self.runtime_overrides
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Gets system configuration information.

        Returns:
            A dictionary of system configuration information.
        """
        return {
            "config_file": str(self.config_file),
            "sections": list(self.to_dict().keys()),
            "dynamic_configs": len(self.dynamic_configs),
            "environment_overrides": len(self.environment_overrides),
            "runtime_overrides": len(self.runtime_overrides),
            "config_history": len(self.config_history),
            "last_updated": datetime.now().isoformat()
        }
    
    async def reset_to_defaults(self):
        """Resets the configuration to its default values."""
        self.prompt = PromptConfig()
        self.memory = MemoryConfig()
        self.plugin = PluginConfig()
        self.task = TaskConfig()
        self.llm = LLMConfig()
        self.system = SystemConfig()
        self.security = SecurityConfig()
        self.web = WebConfig()
        
        self.dynamic_configs.clear()
        self.runtime_overrides.clear()
        
        await self.save_config()
        logger.info("Configuration reset to defaults")
    
    async def backup_config(self, backup_name: str = None) -> str:
        """
        Creates a backup of the current configuration.

        Args:
            backup_name: An optional name for the backup file.

        Returns:
            The path to the backup file, or an empty string if the backup
            failed.
        """
        if not backup_name:
            backup_name = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_file = self.config_file.parent / f"{backup_name}.yaml"
        
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"Configuration backed up to {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"Error creating configuration backup: {e}")
            return ""
    
    async def restore_config(self, backup_file: str) -> bool:
        """
        Restores the configuration from a backup file.

        Args:
            backup_file: The path to the backup file.

        Returns:
            True if the configuration was restored successfully, False otherwise.
        """
        backup_path = Path(backup_file)
        
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False
        
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = yaml.safe_load(f)
            
            # Apply backup data
            self.config_data = backup_data
            await self._apply_configurations()
            await self.save_config()
            
            logger.info(f"Configuration restored from {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring configuration: {e}")
            return False


# Global configuration instance
unified_config = UnifiedConfig()


async def get_config() -> UnifiedConfig:
    """
    Gets the global configuration instance.

    Returns:
        The global UnifiedConfig instance.
    """
    if not unified_config.config_data:
        await unified_config.initialize()
    return unified_config


async def load_config(config_file: str = "config.yaml") -> UnifiedConfig:
    """
    Loads the configuration from a specific file.

    Args:
        config_file: The path to the configuration file.

    Returns:
        A new UnifiedConfig instance.
    """
    config = UnifiedConfig(config_file)
    await config.initialize()
    return config
