"""
env/objects.py
All Pydantic models for the ISS Safety Operations environment.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------

class SafetyObject(BaseModel):
    object_id: str
    object_type: Literal[
        "OxygenTank",
        "NitrogenTank",
        "FirstAidBox",
        "FireAlarm",
        "SafetySwitchPanel",
        "EmergencyPhone",
        "FireExtinguisher",
    ]
    module: Literal[
        "Lab", "Node1", "Node2", "Airlock", "CrewQuarters", "ServiceModule"
    ]
    status: Literal["operational", "degraded", "failed", "depleted"]
    pressure_level: Optional[float] = None   # 0.0-1.0, tanks & extinguishers
    last_inspection_days_ago: int
    expiry_days_remaining: int               # negative = already expired
    inspection_tag_valid: bool
    dependency_object_id: Optional[str] = None  # e.g. FireAlarm → SwitchPanel


class LogEntry(BaseModel):
    timestamp: str                           # e.g. "14:32"
    object_id: str
    event: str
    visible_to_agent: bool                   # False = hidden until pulled


class Alert(BaseModel):
    alert_id: str
    alert_type: Literal["fire", "pressure_drop", "medical", "comms_loss"]
    module: str
    severity: Literal["low", "medium", "high", "critical"]
    triggered_at_timestep: int


# ---------------------------------------------------------------------------
# Observation / Action / Reward models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    episode_id: str
    episode_type: Literal["audit", "emergency", "investigation"]
    mission_context: str
    timestep: int
    objects: list[SafetyObject]
    active_alerts: list[Alert]
    crew_locations: dict[str, str]           # {"crew_1": "Lab", ...}
    evidence_log: list[LogEntry]             # empty in audit/emergency start
    actions_taken: list[str]                 # grows each turn
    turns_remaining: int
    ground_control_available: bool


class Action(BaseModel):
    action_type: Literal[
        "inspect_object",
        "flag_non_compliant",
        "clear_for_mission",
        "deploy_resource",
        "evacuate_module",
        "trigger_switch",
        "contact_ground",
        "pull_sensor_log",
        "cross_reference",
        "identify_root_cause",
        "escalate",
        "submit_report",
    ]
    target_object_id: Optional[str] = None
    target_module: Optional[str] = None
    reasoning: str                           # mandatory every action


class Reward(BaseModel):
    score: float                             # final 0.0 to 1.0
    outcome_score: float
    efficiency_bonus: float
    danger_penalty: float
    escalation_penalty: float
    reasoning_bonus: float
    breakdown: dict


# ---------------------------------------------------------------------------
# Internal state model
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    episode_id: str
    episode_type: str
    current_observation: Observation
    object_registry: dict[str, SafetyObject]
    fire_spread_schedule: Optional[dict] = None
    hidden_logs: Optional[list[LogEntry]] = None
    ground_truth: dict
    done: bool
    total_reward: float
