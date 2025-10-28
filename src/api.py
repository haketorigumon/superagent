from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Set

from src.core.system import UnifiedSystem
from src.core.entities import Priority

app = FastAPI(
    title="Unified AI Agent System",
    description="The ultimate flexible AI architecture",
    version="1.0.0",
)


class CreateAgentRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = "Universal AI agent"
    capabilities: Optional[Set[str]] = None


class CreateTaskRequest(BaseModel):
    content: str
    priority: Priority = Priority.NORMAL


@app.get("/status")
async def get_status(request: Request):
    system: UnifiedSystem = request.app.state.system
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    return system.get_system_status()


@app.post("/agents")
async def create_agent(agent_request: CreateAgentRequest, request: Request):
    system: UnifiedSystem = request.app.state.system
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")

    agent_id = await system.create_agent(
        name=agent_request.name,
        description=agent_request.description,
        capabilities=agent_request.capabilities,
    )
    return {"agent_id": agent_id}


@app.post("/tasks")
async def create_task(task_request: CreateTaskRequest, request: Request):
    system: UnifiedSystem = request.app.state.system
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")

    task_id = await system.create_task(
        content=task_request.content,
        priority=task_request.priority,
    )
    return {"task_id": task_id}


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str, request: Request):
    system: UnifiedSystem = request.app.state.system
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")

    status = await system.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status
