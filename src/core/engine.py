import json
import hashlib
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional

import aiofiles

logger = logging.getLogger(__name__)


class PromptEngine:
    """Universal prompt engine that generates any prompt dynamically"""

    def __init__(self, templates_dir: str = "prompts", system_dir: str = "system"):
        self.templates_dir = Path(templates_dir)
        self.system_dir = Path(system_dir)
        self.templates: Dict[str, str] = {}
        self.system_prompts: Dict[str, str] = {}
        self.generation_history: List[Dict[str, Any]] = []
        self.optimization_data: Dict[str, Dict[str, Any]] = defaultdict(dict)

    async def initialize(self):
        """Initialize the prompt engine"""
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.system_dir.mkdir(parents=True, exist_ok=True)
        await self._load_templates()
        await self._load_system_prompts()
        await self._initialize_core_prompts()

    async def _load_templates(self):
        """Load existing templates"""
        for template_file in self.templates_dir.glob("*.txt"):
            template_name = template_file.stem
            async with aiofiles.open(template_file, 'r', encoding='utf-8') as f:
                self.templates[template_name] = await f.read()

    async def _load_system_prompts(self):
        """Load system prompts"""
        for prompt_file in self.system_dir.glob("*.txt"):
            prompt_name = prompt_file.stem
            async with aiofiles.open(prompt_file, 'r', encoding='utf-8') as f:
                self.system_prompts[prompt_name] = await f.read()

    async def _initialize_core_prompts(self):
        """Initialize core system prompts"""
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

Propose system evolution steps."""
        }

        for name, prompt in core_prompts.items():
            if name not in self.system_prompts:
                self.system_prompts[name] = prompt
                await self._save_system_prompt(name, prompt)

    async def _save_template(self, name: str, template: str):
        """Save template to file"""
        template_file = self.templates_dir / f"{name}.txt"
        async with aiofiles.open(template_file, 'w', encoding='utf-8') as f:
            await f.write(template)

    async def _save_system_prompt(self, name: str, prompt: str):
        """Save system prompt to file"""
        prompt_file = self.system_dir / f"{name}.txt"
        async with aiofiles.open(prompt_file, 'w', encoding='utf-8') as f:
            await f.write(prompt)

    async def generate_prompt(self, purpose: str, context: Dict[str, Any],
                            llm_client=None) -> str:
        """Generate a prompt dynamically"""
        # Check for existing system prompt
        if purpose in self.system_prompts:
            return await self._apply_prompt(purpose, context, is_system=True)

        # Check for existing template
        template_name = self._find_suitable_template(purpose, context)
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

    def _find_suitable_template(self, purpose: str, context: Dict[str, Any]) -> Optional[str]:
        """Find the most suitable existing template"""
        purpose_words = set(purpose.lower().split())
        best_match = None
        best_score = 0

        for template_name in self.templates.keys():
            template_words = set(template_name.lower().replace('_', ' ').split())
            score = len(purpose_words.intersection(template_words))
            if score > best_score:
                best_score = score
                best_match = template_name

        return best_match if best_score > 0 else None

    async def _generate_new_prompt(self, purpose: str, context: Dict[str, Any],
                                 llm_client) -> Optional[str]:
        """Generate a new prompt using LLM"""
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
                "requirements": context.get("requirements", "General purpose prompt")
            }

            filled_prompt = meta_prompt.format(**prompt_context)
            response = await llm_client.generate(filled_prompt, "You are a helpful assistant.")

            self.generation_history.append({
                "purpose": purpose,
                "context": context,
                "generated_at": datetime.now().isoformat(),
                "success": bool(response)
            })

            return response
        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            return None

    async def _apply_prompt(self, prompt_name: str, context: Dict[str, Any],
                          is_system: bool = False) -> str:
        """Apply prompt with context"""
        prompt_dict = self.system_prompts if is_system else self.templates
        prompt = prompt_dict.get(prompt_name, "")

        if not prompt:
            return f"Prompt '{prompt_name}' not found"

        try:
            # Extract parameters from prompt
            import re
            parameters = set(re.findall(r'\{(\w+)\}', prompt))

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

    async def _create_fallback_prompt(self, purpose: str, context: Dict[str, Any]) -> str:
        """Create a basic fallback prompt"""
        return f"""Task: {purpose}

Context: {json.dumps(context, indent=2)}

Please process this request appropriately and provide a structured response."""

    def _generate_prompt_name(self, purpose: str) -> str:
        """Generate a name for a new prompt"""
        purpose_hash = hashlib.md5(purpose.encode()).hexdigest()[:8]
        clean_purpose = "".join(c for c in purpose if c.isalnum() or c in " _").strip()
        clean_purpose = "_".join(clean_purpose.lower().split())[:30]
        return f"{clean_purpose}_{purpose_hash}"