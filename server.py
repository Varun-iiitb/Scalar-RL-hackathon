"""
server.py
---------
FastAPI server exposing the OpenEnv API endpoints:
  POST /reset   — reset environment with an episode_id
  POST /step    — take an action and return (obs, reward, done, info)
  GET  /state   — return full internal state
  GET  /        — health check (returns 200)
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from env.environment import ISSEnvironment
from env.objects import Action, Observation, Reward

app = FastAPI(title="ISS Safety Operations — OpenEnv API", version="1.0.0")

# Single global environment instance
env = ISSEnvironment()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    episode_id: str


class ResetResponse(BaseModel):
    observation: Observation


class StepRequest(BaseModel):
    action_type: str
    target_object_id: Optional[str] = None
    target_module: Optional[str] = None
    reasoning: str


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict


class StateResponse(BaseModel):
    state: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    """Health check — returns 200."""
    return {"status": "ok", "environment": "iss-safety-operations", "version": "1.0.0"}


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    """Reset the environment with the given episode_id."""
    valid_episodes = ["audit_001", "emergency_001", "investigation_001"]
    if req.episode_id not in valid_episodes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid episode_id '{req.episode_id}'. Must be one of {valid_episodes}",
        )
    try:
        obs = env.reset(req.episode_id)
        return ResetResponse(observation=obs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    """Take one action in the environment."""
    try:
        action = Action(
            action_type=req.action_type,
            target_object_id=req.target_object_id,
            target_module=req.target_module,
            reasoning=req.reasoning,
        )
        obs, reward, done, info = env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=StateResponse)
def state():
    """Return the full internal environment state."""
    try:
        return StateResponse(state=env.state())
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Run with: python server.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
