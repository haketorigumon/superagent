import pytest
import pytest_asyncio
import httpx
from src.api import app
from src.core.system import UnifiedSystem
from src.utils.unified_config import load_config
from src.utils.llm.llm_client import LLMClient


@pytest_asyncio.fixture
async def client():
    config = await load_config("config.yaml")
    llm_client = LLMClient(
        provider=config.llm.provider,
        model=config.llm.model,
        api_key=config.get_api_key_for_provider(config.llm.provider),
    )
    system = UnifiedSystem(config, llm_client)
    await system.initialize()
    app.state.system = system

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client

    await system.shutdown()


@pytest.mark.asyncio
async def test_get_status(client: httpx.AsyncClient):
    response = await client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["system"]["initialized"] is True


@pytest.mark.asyncio
async def test_create_agent(client: httpx.AsyncClient):
    response = await client.post("/agents", json={"name": "api_agent"})
    assert response.status_code == 200
    data = response.json()
    assert "agent_id" in data


@pytest.mark.asyncio
async def test_create_task(client: httpx.AsyncClient):
    response = await client.post("/tasks", json={"content": "api task"})
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data


@pytest.mark.asyncio
async def test_get_task_status(client: httpx.AsyncClient):
    # First create a task
    response = await client.post("/tasks", json={"content": "api status task"})
    task_id = response.json()["task_id"]

    # Then get its status
    response = await client.get(f"/tasks/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == task_id
