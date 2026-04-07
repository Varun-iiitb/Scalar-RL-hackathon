"""
server/app.py
OpenEnv-compatible server entry point.
"""

import sys
import os
import uvicorn

# Add project root to path so imports work when run as module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Re-export the FastAPI app from the root server.py module
# We import it directly to avoid circular package issues
from env.environment import ISSEnvironment
from env.objects import Action, Observation, Reward
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional


app = FastAPI(title="ISS Safety Operations - OpenEnv API", version="1.0.0")

env = ISSEnvironment()


class ResetRequest(BaseModel):
    episode_id: str = "audit_001"


class ResetResponse(BaseModel):
    observation: Observation


class StepRequest(BaseModel):
    action_type: str
    target_object_id: Optional[str] = None
    target_module: Optional[str] = None
    reasoning: str = "automated"


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict


class StateResponse(BaseModel):
    state: dict


@app.get("/")
def health():
    return {"status": "ok", "environment": "iss-safety-operations", "version": "1.0.0"}


def _do_reset(episode_id: str) -> ResetResponse:
    valid_episodes = ["audit_001", "emergency_001", "investigation_001"]
    if episode_id not in valid_episodes:
        raise HTTPException(status_code=400, detail=f"Invalid episode_id '{episode_id}'.")
    obs = env.reset(episode_id)
    return ResetResponse(observation=obs)


@app.post("/reset", response_model=ResetResponse)
def reset_post(req: ResetRequest = None):
    episode_id = req.episode_id if req else "audit_001"
    return _do_reset(episode_id)


@app.get("/reset", response_model=ResetResponse)
def reset_get(episode_id: str = "audit_001"):
    return _do_reset(episode_id)


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    try:
        action = Action(
            action_type=req.action_type,
            target_object_id=req.target_object_id,
            target_module=req.target_module,
            reasoning=req.reasoning,
        )
        obs, reward, done, info = env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=StateResponse)
def state():
    try:
        return StateResponse(state=env.state())
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


def main(host: str = "0.0.0.0", port: int = 7860):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
