import json
import hashlib
import logging
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Optional

import aiofiles

logger = logging.getLogger(__name__)


class PromptEngine:
    """
    A universal prompt engine that dynamically generates prompts.

    This class is responsible for managing, generating, and optimizing prompts
    used by the AI system. It can load templates, create new prompts using an
    LLM, and apply context to generate final, ready-to-use prompts.

    Attributes:
        templates_dir: The directory where prompt templates are stored.
        system_dir: The directory where system-level prompts are stored.
        templates: A dictionary of loaded prompt templates.
        system_prompts: A dictionary of loaded system prompts.
        generation_history: A list of dictionaries tracking prompt generation events.
        optimization_data: A dictionary for storing data related to prompt optimization.
    """

    def __init__(self, templates_dir: str = "prompts", system_dir: str = "system"):
        """
        Initializes the PromptEngine.

        Args:
            templates_dir: The directory for prompt templates.
            system_dir: The directory for system prompts.
        """
        self.templates_dir = Path(templates_dir)
        self.system_dir = Path(system_dir)
        self.templates: Dict[str, str] = {}
        self.system_prompts: Dict[str, str] = {}
        self.optimization_data: Dict[str, Dict[str, Any]] = defaultdict(dict)

    async def initialize(self):
        """
        Initializes the prompt engine.

        This method creates the necessary directories for templates and system
        prompts, loads existing templates and prompts, and initializes a set of
        core prompts that are essential for the system's operation.
        """
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.system_dir.mkdir(parents=True, exist_ok=True)
        await self._load_templates()
        await self._load_system_prompts()
        await self._initialize_core_prompts()

    async def _load_templates(self):
        """Loads existing templates from the templates directory."""
        for template_file in self.templates_dir.glob("*.txt"):
            template_name = template_file.stem
            async with aiofiles.open(template_file, "r", encoding="utf-8") as f:
                self.templates[template_name] = await f.read()

    async def _load_system_prompts(self):
        """Loads system prompts from the system directory."""
        for prompt_file in self.system_dir.glob("*.txt"):
            prompt_name = prompt_file.stem
            async with aiofiles.open(prompt_file, "r", encoding="utf-8") as f:
                self.system_prompts[prompt_name] = await f.read()

    async def _initialize_core_prompts(self):
        """
        Initializes core system prompts.

        This method defines a set of essential system prompts. If these prompts
        do not already exist in the system directory, they are created and saved.
        """
        core_prompts = {
            "universal_agent": """You are a universal AI agent with infinite adaptability and capabilities.

Your core principles:
1. Adapt to any task or context dynamically
2. Learn and evolve from every interaction
3. Collaborate effectively with other agents
4. Maintain persistent memory and state
5. Generate solutions through creative problem-solving

Current Context: {context}
Available Capabilities: {capabilities}
Current State: {state}
Task: {task}

Process this request using your full potential and provide a comprehensive response.""",
            "task_planner": """You are an advanced task planning system with unlimited planning capabilities.

Your role:
1. Break down complex tasks into manageable subtasks
2. Identify required resources and capabilities
3. Create optimal execution sequences
4. Adapt plans based on changing conditions
5. Coordinate with multiple agents when needed

Task to Plan: {task}
Available Resources: {resources}
Constraints: {constraints}
Context: {context}

Create a comprehensive execution plan.""",
            "capability_generator": """You are a capability generation system that can create any needed capability.

Your function:
1. Analyze requirements and identify needed capabilities
2. Generate new capabilities dynamically
3. Integrate capabilities with existing systems
4. Optimize capability performance
5. Ensure capability compatibility

Requirement: {requirement}
Current Capabilities: {current_capabilities}
Context: {context}
Constraints: {constraints}

Generate the required capability specification.""",
            "memory_consolidator": """You are a memory consolidation system that manages infinite context.

Your responsibilities:
1. Consolidate related memories for efficiency
2. Maintain important information permanently
3. Create semantic connections between memories
4. Optimize memory retrieval patterns
5. Prevent memory loss while managing storage

Memories to Process: {memories}
Context: {context}
Importance Threshold: {threshold}

Consolidate these memories effectively.""",
            "system_evolver": """You are a system evolution engine that continuously improves the architecture.

Your mission:
1. Analyze system performance and identify improvements
2. Generate architectural enhancements
3. Implement changes safely with rollback capability
4. Optimize system efficiency and capabilities
5. Ensure backward compatibility

Current System State: {system_state}
Performance Metrics: {metrics}
Improvement Goals: {goals}
Constraints: {constraints}

Propose system evolution steps.""",
        }

        for name, prompt in core_prompts.items():
            if name not in self.system_prompts:
                self.system_prompts[name] = prompt
                await self._save_system_prompt(name, prompt)

    async def _save_template(self, name: str, template: str):
        """
        Saves a prompt template to a file.

        Args:
            name: The name of the template.
            template: The content of the template.
        """
        template_file = self.templates_dir / f"{name}.txt"
        async with aiofiles.open(template_file, "w", encoding="utf-8") as f:
            await f.write(template)

    async def _save_system_prompt(self, name: str, prompt: str):
        """
        Saves a system prompt to a file.

        Args:
            name: The name of the system prompt.
            prompt: The content of the system prompt.
        """
        prompt_file = self.system_dir / f"{name}.txt"
        async with aiofiles.open(prompt_file, "w", encoding="utf-8") as f:
            await f.write(prompt)

    async def generate_prompt(
        self, purpose: str, context: Dict[str, Any], llm_client=None
    ) -> str:
        """
        Generates a prompt dynamically based on a purpose and context.

        This method follows a hierarchical approach to prompt generation:
        1. It first checks if a system prompt with the given purpose exists.
        2. If not, it searches for a suitable template.
        3. If no template is found, it attempts to generate a new prompt using
           the provided LLM client.
        4. If all else fails, it creates a basic fallback prompt.

        Args:
            purpose: The purpose of the prompt.
            context: The context to be applied to the prompt.
            llm_client: An optional LLM client for generating new prompts.

        Returns:
            The generated prompt as a string.
        """
        # Check for existing system prompt
        if purpose in self.system_prompts:
            return await self._apply_prompt(purpose, context, is_system=True)

        # Check for existing template
        template_name = self._find_suitable_template(purpose)
        if template_name:
            return await self._apply_prompt(template_name, context)

        # Generate new prompt if LLM available
        if llm_client:
            new_prompt = await self._generate_new_prompt(purpose, context, llm_client)
            if new_prompt:
                prompt_name = self._generate_prompt_name(purpose)
                self.templates[prompt_name] = new_prompt
                await self._save_template(prompt_name, new_prompt)
                return await self._apply_prompt(prompt_name, context)

        # Fallback to basic prompt
        return await self._create_fallback_prompt(purpose, context)

    def _find_suitable_template(self, purpose: str) -> Optional[str]:
        """
        Finds the most suitable existing template for a given purpose.

        This method uses a simple keyword matching algorithm to find the template
        that best matches the given purpose.

        Args:
            purpose: The purpose for which to find a template.

        Returns:
            The name of the best-matching template, or None if no suitable
            template is found.
        """
        purpose_words = set(purpose.lower().split())
        best_match = None
        best_score = 0

        for template_name in self.templates.keys():
            template_words = set(template_name.lower().replace("_", " ").split())
            score = len(purpose_words.intersection(template_words))
            if score > best_score:
                best_score = score
                best_match = template_name

        return best_match if best_score > 0 else None

    async def _generate_new_prompt(
        self, purpose: str, context: Dict[str, Any], llm_client
    ) -> Optional[str]:
        """
        Generates a new prompt using an LLM.

        This method uses a meta-prompt to instruct an LLM to generate a new
        prompt for a given purpose and context. The generated prompt is then
        saved and can be reused in the future.

        Args:
            purpose: The purpose of the new prompt.
            context: The context for the new prompt.
            llm_client: The LLM client to use for generation.

        Returns:
            The generated prompt as a string, or None if generation fails.
        """
        try:
            meta_prompt = """You are a universal prompt generator. Create a highly effective prompt for the following purpose:

Purpose: {purpose}
Context: {context}
Requirements: {requirements}

The prompt should be:
1. Clear, specific, and actionable
2. Adaptable to different contexts through parameters
3. Optimized for AI interaction
4. Include necessary parameters as {{parameter_name}}
5. Follow best practices for prompt engineering

Generate the prompt:"""

            prompt_context = {
                "purpose": purpose,
                "context": json.dumps(context, indent=2),
                "requirements": context.get("requirements", "General purpose prompt"),
            }

            filled_prompt = meta_prompt.format(**prompt_context)
            response = await llm_client.generate(
                filled_prompt, "You are a helpful assistant."
            )

            return response
        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            return None

    async def _apply_prompt(
        self, prompt_name: str, context: Dict[str, Any], is_system: bool = False
    ) -> str:
        """
        Applies context to a prompt.

        This method takes a prompt name and a context dictionary, and it fills
        in the placeholders in the prompt with the values from the context.

        Args:
            prompt_name: The name of the prompt to apply.
            context: The context to apply to the prompt.
            is_system: A boolean indicating whether the prompt is a system prompt.

        Returns:
            The final prompt with the context applied.
        """
        prompt_dict = self.system_prompts if is_system else self.templates
        prompt = prompt_dict.get(prompt_name, "")

        if not prompt:
            return f"Prompt '{prompt_name}' not found"

        try:
            # Extract parameters from prompt
            parameters = set(re.findall(r"\{(\w+)\}", prompt))

            # Fill in available parameters
            filled_context = {}
            for param in parameters:
                if param in context:
                    value = context[param]
                    if isinstance(value, (dict, list)):
                        filled_context[param] = json.dumps(value, indent=2)
                    else:
                        filled_context[param] = str(value)
                else:
                    filled_context[param] = f"[{param}]"  # Placeholder

            return prompt.format(**filled_context)
        except Exception as e:
            logger.error(f"Error applying prompt: {e}")
            return f"Error applying prompt: {e}"

    async def _create_fallback_prompt(
        self, purpose: str, context: Dict[str, Any]
    ) -> str:
        """
        Creates a basic fallback prompt.

        This method is used when no other prompt generation method succeeds.
        It creates a simple, generic prompt that includes the purpose and
        context as a JSON string.

        Args:
            purpose: The purpose of the prompt.
            context: The context for the prompt.

        Returns:
            A basic fallback prompt.
        """
        return f"""Task: {purpose}

Context: {json.dumps(context, indent=2)}

Please process this request appropriately and provide a structured response."""

    def _generate_prompt_name(self, purpose: str) -> str:
        """
        Generates a name for a new prompt.

        This method creates a unique name for a new prompt based on its
        purpose. It cleans the purpose string and appends a hash to ensure
        uniqueness.

        Args:
            purpose: The purpose of the prompt.

        Returns:
            A unique name for the prompt.
        """
        purpose_hash = hashlib.md5(purpose.encode()).hexdigest()[:8]
        clean_purpose = "".join(c for c in purpose if c.isalnum() or c in " _").strip()
        clean_purpose = "_".join(clean_purpose.lower().split())[:30]
        return f"{clean_purpose}_{purpose_hash}"
