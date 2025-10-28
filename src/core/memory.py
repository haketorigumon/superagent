import asyncio
import pickle
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Set

from src.core.entities import UniversalEntity, MemoryType

logger = logging.getLogger(__name__)


class PersistentMemorySystem:
    """
    A persistent memory system with infinite context and no memory loss.

    This class manages the storage, retrieval, and indexing of memories for
    the AI system. It uses a multi-layered approach to memory, including
    working, episodic, and persistent storage. It also includes a background
    worker for memory consolidation and maintenance.

    Attributes:
        storage_dir: The directory where persistent memory is stored.
        memory_layers: A dictionary of memory layers, each containing a
                       dictionary of memories.
        content_index: An index for fast content-based memory retrieval.
        temporal_index: An index for retrieving memories by date.
        importance_index: An index for retrieving memories by importance.
        tag_index: An index for retrieving memories by tags.
        persistent_storage: A dictionary for storing arbitrary persistent data.
        storage_file: The file where the persistent state is saved.
        consolidation_queue: A queue for memories that need to be consolidated.
        consolidation_task: The background task for memory consolidation.
    """

    def __init__(self, storage_dir: str = "persistent_memory"):
        """
        Initializes the PersistentMemorySystem.

        Args:
            storage_dir: The directory for storing persistent memory.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Multi-layer memory storage
        self.memory_layers: Dict[MemoryType, Dict[str, UniversalEntity]] = {
            layer: {} for layer in MemoryType
        }

        # Memory indices for fast retrieval
        self.content_index: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)
        self.importance_index: Dict[float, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)

        # Persistent storage
        self.persistent_storage: Dict[str, Any] = {}
        self.storage_file = self.storage_dir / "persistent_state.pkl"

        # Memory consolidation
        self.consolidation_queue: asyncio.Queue = asyncio.Queue()
        self.consolidation_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """
        Initializes the memory system.

        This method loads the persistent state from storage and starts the
        background consolidation worker.
        """
        await self._load_persistent_state()
        self.consolidation_task = asyncio.create_task(self._consolidation_worker())

    async def _load_persistent_state(self):
        """Loads the persistent state from storage."""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'rb') as f:
                    data = pickle.load(f)
                    self.persistent_storage = data.get('storage', {})

                    # Restore memory layers
                    for layer_name, memories in data.get('memory_layers', {}).items():
                        layer = MemoryType(layer_name)
                        for memory_id, memory_data in memories.items():
                            entity = UniversalEntity.from_dict(memory_data)
                            self.memory_layers[layer][memory_id] = entity

                    # Rebuild indices
                    await self._rebuild_indices()

                logger.info(f"Loaded persistent state with {len(self.persistent_storage)} items")
            except Exception as e:
                logger.error(f"Error loading persistent state: {e}")

    async def _save_persistent_state(self):
        """Saves the persistent state to storage."""
        try:
            data = {
                'storage': self.persistent_storage,
                'memory_layers': {
                    layer.value: {
                        memory_id: memory.to_dict()
                        for memory_id, memory in memories.items()
                    }
                    for layer, memories in self.memory_layers.items()
                },
                'saved_at': datetime.now().isoformat()
            }

            with open(self.storage_file, 'wb') as f:
                pickle.dump(data, f)

        except Exception as e:
            logger.error(f"Error saving persistent state: {e}")

    async def _rebuild_indices(self):
        """Rebuilds all memory indices."""
        self.content_index.clear()
        self.temporal_index.clear()
        self.importance_index.clear()
        self.tag_index.clear()

        for layer, memories in self.memory_layers.items():
            for memory_id, memory in memories.items():
                await self._index_memory(memory_id, memory)

    async def _index_memory(self, memory_id: str, memory: UniversalEntity):
        """
        Indexes a memory for fast retrieval.

        This method adds the memory to the content, temporal, importance, and
        tag indices.

        Args:
            memory_id: The ID of the memory to index.
            memory: The memory entity to index.
        """
        # Content indexing
        if memory.content:
            content_words = str(memory.content).lower().split()
            for word in content_words:
                self.content_index[word].add(memory_id)

        # Temporal indexing
        date_key = memory.created_at.strftime('%Y-%m-%d')
        self.temporal_index[date_key].append(memory_id)

        # Importance indexing
        importance_bucket = round(memory.importance, 1)
        self.importance_index[importance_bucket].add(memory_id)

        # Tag indexing
        for tag in memory.tags:
            self.tag_index[tag].add(memory_id)

    async def store_memory(self, entity: UniversalEntity,
                          memory_type: MemoryType = MemoryType.WORKING) -> str:
        """
        Stores a memory in the specified layer.

        Args:
            entity: The memory entity to store.
            memory_type: The layer in which to store the memory.

        Returns:
            The ID of the stored memory.
        """
        memory_id = entity.id
        self.memory_layers[memory_type][memory_id] = entity
        await self._index_memory(memory_id, entity)

        # Queue for consolidation if not working memory
        if memory_type != MemoryType.WORKING:
            await self.consolidation_queue.put((memory_type, memory_id))

        # Auto-save for persistent memories
        if memory_type == MemoryType.PERSISTENT:
            await self._save_persistent_state()

        return memory_id

    async def retrieve_memory(self, memory_id: str) -> Optional[UniversalEntity]:
        """
        Retrieves a specific memory by its ID.

        Args:
            memory_id: The ID of the memory to retrieve.

        Returns:
            The memory entity, or None if not found.
        """
        for layer, memories in self.memory_layers.items():
            if memory_id in memories:
                memory = memories[memory_id]
                memory.access_count += 1
                memory.accessed_at = datetime.now()
                return memory
        return None

    async def search_memories(self, query: str, context: Dict[str, Any] = None,
                            limit: int = 10) -> List[UniversalEntity]:
        """
        Searches for memories using various criteria.

        This method performs a content-based and tag-based search, scores the
        results based on relevance, and returns the top N memories.

        Args:
            query: The search query.
            context: An optional context dictionary, which can include tags.
            limit: The maximum number of memories to return.

        Returns:
            A list of the most relevant memory entities.
        """
        results = []
        query_words = set(query.lower().split())

        # Content-based search
        matching_ids = set()
        for word in query_words:
            if word in self.content_index:
                matching_ids.update(self.content_index[word])

        # Tag-based search
        if context and 'tags' in context:
            for tag in context['tags']:
                if tag in self.tag_index:
                    matching_ids.update(self.tag_index[tag])

        # Retrieve and score matches
        scored_memories = []
        for memory_id in matching_ids:
            memory = await self.retrieve_memory(memory_id)
            if memory:
                score = self._calculate_relevance_score(memory, query, context)
                scored_memories.append((score, memory))

        # Sort by relevance and return top results
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]

    def _calculate_relevance_score(self, memory: UniversalEntity, query: str,
                                 context: Dict[str, Any] = None) -> float:
        """
        Calculates a relevance score for a memory.

        The score is based on content relevance, importance, recency, and
        access frequency.

        Args:
            memory: The memory entity to score.
            query: The search query.
            context: An optional context dictionary.

        Returns:
            The relevance score as a float.
        """
        score = 0.0
        query_words = set(query.lower().split())

        # Content relevance
        if memory.content:
            content_words = set(str(memory.content).lower().split())
            content_overlap = len(query_words.intersection(content_words))
            score += content_overlap * 2.0

        # Importance factor
        score += memory.importance * 1.5

        # Recency factor
        days_old = (datetime.now() - memory.updated_at).days
        recency_factor = max(0.1, 1.0 - (days_old / 365))
        score *= recency_factor

        # Access frequency factor
        access_factor = min(2.0, 1.0 + (memory.access_count / 100))
        score *= access_factor

        return score

    async def _consolidation_worker(self):
        """A background worker for memory consolidation."""
        while True:
            try:
                memory_type, memory_id = await asyncio.wait_for(
                    self.consolidation_queue.get(), timeout=60.0
                )
                await self._consolidate_memory(memory_type, memory_id)
            except asyncio.TimeoutError:
                # Periodic maintenance
                await self._periodic_maintenance()
            except Exception as e:
                logger.error(f"Error in consolidation worker: {e}")

    async def _consolidate_memory(self, memory_type: MemoryType, memory_id: str):
        """
        Consolidates a specific memory.

        This method increases the importance of frequently accessed memories and
        moves important memories to persistent storage.

        Args:
            memory_type: The current memory type of the memory.
            memory_id: The ID of the memory to consolidate.
        """
        memory = self.memory_layers[memory_type].get(memory_id)
        if not memory:
            return

        # Increase importance for frequently accessed memories
        if memory.access_count > 10:
            memory.importance = min(1.0, memory.importance + 0.1)

        # Move important memories to persistent storage
        if memory.importance > 0.8 and memory_type != MemoryType.PERSISTENT:
            await self.store_memory(memory, MemoryType.PERSISTENT)

    async def _periodic_maintenance(self):
        """Performs periodic maintenance tasks."""
        # Clean up expired memories
        current_time = datetime.now()
        for layer, memories in self.memory_layers.items():
            expired_ids = [
                memory_id for memory_id, memory in memories.items()
                if memory.expires_at and memory.expires_at < current_time
            ]
            for memory_id in expired_ids:
                del memories[memory_id]

        # Save persistent state
        await self._save_persistent_state()

    async def cleanup(self):
        """Cleans up the memory system."""
        if self.consolidation_task:
            self.consolidation_task.cancel()
        await self._save_persistent_state()