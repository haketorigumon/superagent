import asyncio
import pytest
import pytest_asyncio
from src.core.system import UnifiedSystem
from src.core.entities import EntityType, Priority, MemoryType
from src.utils.unified_config import load_config
from src.utils.llm.llm_client import LLMClient


@pytest_asyncio.fixture
async def unified_system():
    """Provides a fully initialized UnifiedSystem instance for testing."""
    config = await load_config("config.yaml")
    llm_client = LLMClient(
        provider=config.llm.provider,
        model=config.llm.model,
        api_key=config.get_api_key_for_provider(config.llm.provider),
    )
    system = UnifiedSystem(config, llm_client)
    await system.initialize()
    yield system
    await system.shutdown()


@pytest.mark.asyncio
async def test_system_initialization(unified_system: UnifiedSystem):
    """Tests if the UnifiedSystem initializes correctly."""
    assert unified_system.is_initialized
    assert unified_system.is_running
    # Check if core components are created
    assert unified_system.prompt_engine is not None
    assert unified_system.memory_system is not None
    assert unified_system.plugin_system is not None


@pytest.mark.asyncio
async def test_create_agent(unified_system: UnifiedSystem):
    """Tests the creation of a new agent."""
    agent_id = await unified_system.create_agent(
        name="TestAgent",
        description="An agent for testing purposes.",
        capabilities={"testing", "assertion"},
    )
    agent = await unified_system.get_entity(agent_id)
    assert agent is not None
    assert agent.type == EntityType.AGENT
    assert agent.name == "TestAgent"
    assert "testing" in agent.capabilities


@pytest.mark.asyncio
async def test_create_and_process_task(unified_system: UnifiedSystem):
    """Tests creating a task and verifying its completion."""
    task_content = "This is a test task."
    task_id = await unified_system.create_task(
        content=task_content,
        priority=Priority.HIGH,
    )

    # Give the system time to process the task
    await asyncio.sleep(1)  # Adjust if tasks take longer

    task_status = await unified_system.get_task_status(task_id)
    assert task_status is not None
    assert task_status["status"] == "completed"
    assert task_status["content"] == task_content
    assert "simulated response" in task_status["result"]


@pytest.mark.asyncio
async def test_memory_storage_and_retrieval(unified_system: UnifiedSystem):
    """Tests the persistent memory system."""
    memory_content = "This is an important memory."
    memory_id = await unified_system.create_entity(
        EntityType.MEMORY,
        content=memory_content,
        importance=0.9,
        tags={"test", "important"},
    )

    # Store it in persistent memory
    memory_entity = await unified_system.get_entity(memory_id)
    await unified_system.memory_system.store_memory(
        memory_entity, MemoryType.PERSISTENT
    )

    # Retrieve the memory
    retrieved_memory = await unified_system.memory_system.retrieve_memory(memory_id)
    assert retrieved_memory is not None
    assert retrieved_memory.content == memory_content
    assert "important" in retrieved_memory.tags


@pytest.mark.asyncio
async def test_search_memories(unified_system: UnifiedSystem):
    """Tests searching for memories."""
    # First, store a memory to search for
    await unified_system.create_entity(
        EntityType.MEMORY,
        content="Searchable test memory content.",
        tags={"search_test"},
        importance=0.8,
    )

    # Give it a moment to be indexed
    await asyncio.sleep(0.1)

    search_results = await unified_system.memory_system.search_memories("searchable")
    assert len(search_results) > 0
    assert "searchable" in search_results[0].content.lower()

    # Test searching by tag
    search_results_by_tag = await unified_system.memory_system.search_memories(
        "any", context={"tags": ["search_test"]}
    )
    assert len(search_results_by_tag) > 0
    assert "search_test" in search_results_by_tag[0].tags


@pytest.mark.asyncio
async def test_search_entity_with_none_attributes_does_not_crash(
    unified_system: UnifiedSystem,
):
    """
    Tests that searching entities does not crash when an entity has None for name or description.
    """
    # This entity should not match the search, but has description=None which could cause a crash
    await unified_system.create_entity(
        EntityType.RESOURCE,
        name="A completely different thing",
        description=None,
        content="Nothing to see here.",
    )

    # This entity should match the search
    await unified_system.create_entity(
        EntityType.RESOURCE,
        name="Another thing",
        description="This contains the magic word.",
        content="Some other content.",
    )

    try:
        # This search should trigger the check on the entity with description=None
        results = await unified_system.search_entities(query="magic")
        # We should find one result
        assert len(results) == 1
        assert results[0].name == "Another thing"
    except AttributeError as e:
        pytest.fail(f"Searching with None attribute raised an exception: {e}")


@pytest.mark.asyncio
async def test_complex_task_execution(unified_system: UnifiedSystem):
    """Tests that a complex task is broken down into a plan and executed."""
    task_content = "First, do step 1, and then do step 2."
    task_id = await unified_system.create_task(
        content=task_content,
        priority=Priority.HIGH,
    )

    # Give the system time to process the task
    await asyncio.sleep(2)

    task_status = await unified_system.get_task_status(task_id)
    assert task_status is not None
    assert task_status["status"] == "completed"
    assert "plan" in task_status["result"]


@pytest.mark.asyncio
async def test_memory_search_with_metadata(unified_system: UnifiedSystem):
    """Tests that memory search is influenced by metadata."""
    # Create two memories with similar content but different metadata
    await unified_system.create_entity(
        EntityType.MEMORY,
        name="low_priority_memory",
        content="This is a test memory.",
        tags={"generic"},
        priority=Priority.LOW,
        importance=0.5,
    )
    await unified_system.create_entity(
        EntityType.MEMORY,
        name="high_priority_memory",
        content="This is also a test memory.",
        tags={"specific_test"},
        priority=Priority.HIGH,
        importance=0.9,
    )

    # Give them a moment to be indexed
    await asyncio.sleep(0.1)

    # Search for a term present in both, but also in the tags of the second memory
    search_results = await unified_system.memory_system.search_memories(
        "test specific_test"
    )

    # The second memory should be ranked higher due to the tag and priority
    assert len(search_results) > 0
    assert search_results[0].name == "high_priority_memory"
