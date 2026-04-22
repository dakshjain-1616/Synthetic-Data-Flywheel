"""A2A Agent - HTTP server exposing A2A protocol endpoints for multi-agent orchestration."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import structlog
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from synthetic_data_flywheel.config import get_settings
from synthetic_data_flywheel.engine import FlywheelEngine
from synthetic_data_flywheel.models import CycleState

logger = structlog.get_logger()

# A2A Protocol Models


class AgentCapability(BaseModel):
    """A2A agent capability."""
    name: str
    description: str
    parameters: Optional[Dict[str, Any]] = None


class AgentStatus(BaseModel):
    """A2A agent status."""
    state: str = "idle"  # idle, busy, error
    current_task: Optional[str] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class TaskRequest(BaseModel):
    """A2A task request."""
    task_id: Optional[str] = None
    capability: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    callback_url: Optional[str] = None


class TaskResponse(BaseModel):
    """A2A task response."""
    task_id: str
    status: str  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class A2AAgent:
    """A2A Protocol compliant agent for multi-agent orchestration."""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_name: str = "synthetic-data-flywheel",
        version: str = "0.1.0",
    ):
        """Initialize A2A agent.
        
        Args:
            agent_id: Unique agent ID
            agent_name: Agent name
            version: Agent version
        """
        self.agent_id = agent_id or str(uuid4())
        self.agent_name = agent_name
        self.version = version
        
        self.capabilities = [
            AgentCapability(
                name="generate_synthetic_data",
                description="Generate synthetic training data from seeds",
                parameters={
                    "seeds": {"type": "array", "items": "string"},
                    "max_cycles": {"type": "integer", "default": 1},
                    "template_type": {"type": "string", "enum": ["qa", "instruction", "reasoning", "creative"]},
                },
            ),
            AgentCapability(
                name="judge_quality",
                description="Judge quality of synthetic data pairs",
                parameters={
                    "pairs": {"type": "array"},
                    "threshold": {"type": "number", "default": 7.0},
                },
            ),
            AgentCapability(
                name="get_status",
                description="Get current agent status and cycle information",
                parameters={},
            ),
            AgentCapability(
                name="run_full_flywheel",
                description="Run complete flywheel with feedback loop",
                parameters={
                    "initial_seeds": {"type": "array", "items": "string"},
                    "max_cycles": {"type": "integer", "default": 5},
                    "min_pass_rate": {"type": "number", "default": 0.5},
                },
            ),
        ]
        
        self.status = AgentStatus()
        self.tasks: Dict[str, TaskResponse] = {}
        self.engine: Optional[FlywheelEngine] = None
        
        logger.info(
            "a2a_agent_initialized",
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            capabilities=len(self.capabilities),
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities (A2A /a2a/capabilities endpoint)."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "version": self.version,
            "capabilities": [
                {
                    "name": c.name,
                    "description": c.description,
                    "parameters": c.parameters,
                }
                for c in self.capabilities
            ],
            "status": {
                "state": self.status.state,
                "current_task": self.status.current_task,
                "last_updated": self.status.last_updated.isoformat(),
            },
        }
    
    async def submit_task(self, request: TaskRequest) -> TaskResponse:
        """Submit a task (A2A /a2a/tasks/send endpoint)."""
        task_id = request.task_id or str(uuid4())
        
        task = TaskResponse(
            task_id=task_id,
            status="pending",
        )
        self.tasks[task_id] = task
        
        logger.info(
            "task_submitted",
            task_id=task_id,
            capability=request.capability,
        )
        
        # Execute task
        try:
            task.status = "running"
            self.status.state = "busy"
            self.status.current_task = task_id
            
            result = await self._execute_capability(request.capability, request.parameters)
            
            task.status = "completed"
            task.result = result
            task.completed_at = datetime.utcnow()
            
            logger.info("task_completed", task_id=task_id)
            
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            
            logger.error("task_failed", task_id=task_id, error=str(e))
        
        finally:
            self.status.state = "idle"
            self.status.current_task = None
            self.status.last_updated = datetime.utcnow()
        
        return task
    
    async def _execute_capability(
        self,
        capability: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a capability."""
        if capability == "generate_synthetic_data":
            return await self._generate_synthetic_data(parameters)
        
        elif capability == "judge_quality":
            return await self._judge_quality(parameters)
        
        elif capability == "get_status":
            return self._get_status()
        
        elif capability == "run_full_flywheel":
            return await self._run_full_flywheel(parameters)
        
        else:
            raise ValueError(f"Unknown capability: {capability}")
    
    async def _generate_synthetic_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate synthetic data capability."""
        from synthetic_data_flywheel.generator import OpenRouterClient, PromptTemplate
        
        seeds = parameters.get("seeds", [])
        if not seeds:
            raise ValueError("No seeds provided")
        
        template_type = parameters.get("template_type", PromptTemplate.INSTRUCTION)
        max_concurrent = parameters.get("max_concurrent", 5)
        
        generator = OpenRouterClient()
        
        pairs = await generator.generate_batch(
            seeds=seeds,
            template_type=template_type,
            max_concurrent=max_concurrent,
        )
        
        await generator.close()
        
        return {
            "generated_count": len(pairs),
            "pairs": [
                {
                    "id": str(p.id),
                    "instruction": p.instruction,
                    "output": p.output,
                    "category": p.category,
                }
                for p in pairs
            ],
        }
    
    async def _judge_quality(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Judge quality capability."""
        from synthetic_data_flywheel.judge import QualityJudge
        from synthetic_data_flywheel.models import SyntheticPair
        
        pairs_data = parameters.get("pairs", [])
        threshold = parameters.get("threshold", 7.0)
        
        pairs = []
        for data in pairs_data:
            pairs.append(SyntheticPair(
                instruction=data.get("instruction", ""),
                output=data.get("output", ""),
            ))
        
        judge = QualityJudge(min_overall_score=threshold)
        judgments = judge.judge_batch(pairs)
        
        passed = sum(1 for j in judgments if j.passed)
        
        return {
            "total": len(judgments),
            "passed": passed,
            "failed": len(judgments) - passed,
            "pass_rate": passed / len(judgments) if judgments else 0,
            "judgments": [
                {
                    "pair_id": str(j.pair_id),
                    "passed": j.passed,
                    "scores": {
                        "coherence": j.scores.coherence,
                        "accuracy": j.scores.accuracy,
                        "helpfulness": j.scores.helpfulness,
                        "overall": j.scores.overall,
                    },
                }
                for j in judgments
            ],
        }
    
    def _get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "version": self.version,
            "status": {
                "state": self.status.state,
                "current_task": self.status.current_task,
                "last_updated": self.status.last_updated.isoformat(),
            },
            "tasks": {
                "total": len(self.tasks),
                "pending": sum(1 for t in self.tasks.values() if t.status == "pending"),
                "running": sum(1 for t in self.tasks.values() if t.status == "running"),
                "completed": sum(1 for t in self.tasks.values() if t.status == "completed"),
                "failed": sum(1 for t in self.tasks.values() if t.status == "failed"),
            },
        }
    
    async def _run_full_flywheel(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run full flywheel capability."""
        initial_seeds = parameters.get("initial_seeds", [])
        max_cycles = parameters.get("max_cycles", 5)
        min_pass_rate = parameters.get("min_pass_rate", 0.5)
        
        if not initial_seeds:
            raise ValueError("No initial seeds provided")
        
        self.engine = FlywheelEngine()
        
        cycles = await self.engine.run_full_loop(
            initial_seeds=initial_seeds,
            max_cycles=max_cycles,
            min_pass_rate=min_pass_rate,
        )
        
        summary = self.engine.get_summary()
        
        return {
            "cycles_completed": len(cycles),
            "total_generated": summary.get("total_generated", 0),
            "total_passed": summary.get("total_passed", 0),
            "overall_pass_rate": summary.get("overall_pass_rate", 0),
            "cycle_summary": [
                {
                    "cycle_id": c.cycle_id,
                    "generated": len(c.generated_pairs),
                    "passed": len(c.passed_pairs),
                    "pass_rate": c.pass_rate,
                }
                for c in cycles
            ],
        }
    
    def get_task(self, task_id: str) -> Optional[TaskResponse]:
        """Get task status (A2A /a2a/tasks/get endpoint)."""
        return self.tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task (A2A /a2a/tasks/cancel endpoint)."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status in ["pending", "running"]:
                task.status = "cancelled"
                task.completed_at = datetime.utcnow()
                logger.info("task_cancelled", task_id=task_id)
                return True
        return False
    
    def list_tasks(self) -> List[TaskResponse]:
        """List all tasks."""
        return list(self.tasks.values())


# FastAPI Application
app = FastAPI(
    title="Synthetic Data Flywheel A2A Agent",
    description="A2A Protocol compliant agent for synthetic data generation",
    version="0.1.0",
)

# Global agent instance
agent: Optional[A2AAgent] = None


@app.on_event("startup")
async def startup():
    """Initialize agent on startup."""
    global agent
    agent = A2AAgent()
    logger.info("a2a_server_started")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("a2a_server_shutdown")


@app.get("/a2a/capabilities")
async def get_capabilities():
    """A2A capabilities endpoint."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return agent.get_capabilities()


@app.post("/a2a/tasks/send", response_model=TaskResponse)
async def send_task(request: TaskRequest):
    """A2A task submission endpoint."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    result = await agent.submit_task(request)
    return result


@app.get("/a2a/tasks/{task_id}")
async def get_task(task_id: str):
    """A2A task status endpoint."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    task = agent.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task


@app.post("/a2a/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """A2A task cancellation endpoint."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    success = agent.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=400, detail="Task cannot be cancelled")
    
    return {"status": "cancelled", "task_id": task_id}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent_id": agent.agent_id if agent else None,
        "agent_status": agent.status.state if agent else "unknown",
    }


def create_app() -> FastAPI:
    """Factory function to create FastAPI app."""
    return app
