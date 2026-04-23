"""A2A Agent - HTTP server exposing A2A protocol endpoints for multi-agent orchestration."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from synthetic_data_flywheel.config import get_settings
from synthetic_data_flywheel.engine import create_engine


app = FastAPI(title="Synthetic Data Flywheel A2A Agent")


# A2A Protocol Models
class AgentCapability(BaseModel):
    """A2A agent capability."""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AgentCapabilities(BaseModel):
    """A2A agent capabilities response."""
    agent_name: str = "synthetic_data_flywheel"
    version: str = "0.1.0"
    capabilities: List[AgentCapability]


class TaskInput(BaseModel):
    """A2A task input."""
    type: str
    content: str


class TaskRequest(BaseModel):
    """A2A task send request."""
    task_id: Optional[str] = None
    capability: str
    inputs: List[TaskInput]
    parameters: Dict[str, Any] = Field(default_factory=dict)


class TaskStatus(BaseModel):
    """A2A task status."""
    state: str  # pending, running, completed, failed
    message: Optional[str] = None


class TaskResult(BaseModel):
    """A2A task result."""
    type: str
    content: Any


class TaskResponse(BaseModel):
    """A2A task response."""
    task_id: str
    status: TaskStatus
    result: Optional[TaskResult] = None
    created_at: str
    updated_at: str


class TaskGetRequest(BaseModel):
    """A2A task get request."""
    task_id: str


class TaskCancelRequest(BaseModel):
    """A2A task cancel request."""
    task_id: str


# In-memory task storage (would use Redis/DB in production)
tasks: Dict[str, TaskResponse] = {}


@app.get("/a2a/capabilities", response_model=AgentCapabilities)
async def get_capabilities():
    """Get agent capabilities."""
    return AgentCapabilities(
        capabilities=[
            AgentCapability(
                name="generate_synthetic_data",
                description="Generate synthetic training data using the flywheel",
                parameters={
                    "seeds": {"type": "array", "items": "string"},
                    "max_cycles": {"type": "integer"},
                },
            ),
            AgentCapability(
                name="get_status",
                description="Get current flywheel status",
                parameters={},
            ),
            AgentCapability(
                name="generate_report",
                description="Generate HTML report from checkpoints",
                parameters={
                    "output_dir": {"type": "string"},
                },
            ),
        ],
    )


@app.post("/a2a/tasks/send", response_model=TaskResponse)
async def send_task(request: TaskRequest):
    """Send a task to the agent."""
    task_id = request.task_id or str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    # Create initial task response
    task = TaskResponse(
        task_id=task_id,
        status=TaskStatus(state="pending"),
        created_at=now,
        updated_at=now,
    )
    tasks[task_id] = task
    
    # Process task based on capability
    try:
        if request.capability == "generate_synthetic_data":
            result = await _handle_generate_synthetic_data(request)
        elif request.capability == "get_status":
            result = await _handle_get_status(request)
        elif request.capability == "generate_report":
            result = await _handle_generate_report(request)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown capability: {request.capability}")
        
        task.status.state = "completed"
        task.result = result
        
    except Exception as e:
        task.status.state = "failed"
        task.status.message = str(e)
    
    task.updated_at = datetime.utcnow().isoformat()
    return task


@app.post("/a2a/tasks/get", response_model=TaskResponse)
async def get_task(request: TaskGetRequest):
    """Get task status and result."""
    if request.task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[request.task_id]


@app.post("/a2a/tasks/cancel", response_model=TaskResponse)
async def cancel_task(request: TaskCancelRequest):
    """Cancel a running task."""
    if request.task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[request.task_id]
    if task.status.state == "running":
        task.status.state = "cancelled"
        task.updated_at = datetime.utcnow().isoformat()
    
    return task


async def _handle_generate_synthetic_data(request: TaskRequest) -> TaskResult:
    """Handle synthetic data generation task."""
    # Extract parameters
    seeds = []
    for inp in request.inputs:
        if inp.type == "seeds":
            seeds = json.loads(inp.content)
    
    max_cycles = request.parameters.get("max_cycles", 3)
    
    # Create engine and run
    engine = create_engine(seeds=seeds, max_cycles=max_cycles)
    
    # Update task status
    task_id = request.task_id or str(uuid.uuid4())
    if task_id in tasks:
        tasks[task_id].status.state = "running"
    
    # Run flywheel
    import asyncio
    await engine.run_full_loop()
    
    summary = engine.get_summary()
    
    return TaskResult(
        type="synthetic_data_result",
        content={
            "total_cycles": summary["total_cycles"],
            "total_passed_pairs": summary["total_passed_pairs"],
            "avg_pass_rate": summary["avg_pass_rate"],
            "cycles": summary["cycles"],
        },
    )


async def _handle_get_status(request: TaskRequest) -> TaskResult:
    """Handle status check task."""
    settings = get_settings()
    cp_dir = Path(settings.checkpoint_dir)
    
    checkpoints = sorted(cp_dir.glob("checkpoint_*.json")) if cp_dir.exists() else []
    
    return TaskResult(
        type="status_result",
        content={
            "checkpoints_found": len(checkpoints),
            "checkpoint_dir": str(cp_dir),
        },
    )


async def _handle_generate_report(request: TaskRequest) -> TaskResult:
    """Handle report generation task."""
    from synthetic_data_flywheel.report_generator import create_report_generator
    
    settings = get_settings()
    output_dir = request.parameters.get("output_dir", settings.report_output_dir)
    
    # Load engine state
    engine = create_engine(seeds=[])
    cp_dir = Path(settings.checkpoint_dir)
    checkpoints = sorted(cp_dir.glob("checkpoint_*.json")) if cp_dir.exists() else []
    
    if checkpoints:
        engine.load_checkpoint(str(checkpoints[-1]))
        
        generator = create_report_generator(output_dir=output_dir)
        report_path = generator.generate_report(engine.cycles)
        
        return TaskResult(
            type="report_result",
            content={
                "report_path": str(report_path),
                "cycles_included": len(engine.cycles),
            },
        )
    else:
        return TaskResult(
            type="report_result",
            content={"error": "No checkpoints found"},
        )


def create_a2a_app() -> FastAPI:
    """Factory function to create A2A app."""
    return app


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.a2a_host,
        port=settings.a2a_port,
    )
