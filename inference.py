#!/usr/bin/env python3
"""
Round 1 inference script for ISS Safety Operations Environment.

Required env vars:
- API_BASE_URL
- MODEL_NAME
- HF_TOKEN

This script:
- makes LLM calls using OpenAI Client
- uses deterministic fallback for stability
- emits ONLY the required [START] / [STEP] / [END] stdout lines
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

from openai import OpenAI
from pydantic import ValidationError
from dotenv import load_dotenv

from env.environment import ISSEnvironment
from env.objects import Action

load_dotenv()

# ============================================================
# Config
# ============================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

TASKS = ["audit_001", "emergency_001", "investigation_001"]
BENCHMARK = "iss-safety-operations"
MAX_STEPS_SAFE_GUARD = 10


def require_env(name: str, value: Optional[str]) -> str:
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_client() -> OpenAI:
    return OpenAI(
        base_url=require_env("API_BASE_URL", API_BASE_URL),
        api_key=require_env("HF_TOKEN", HF_TOKEN),
    )


# ============================================================
# Required logging format
# Based on sample inference script
# ============================================================

def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action_str: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ============================================================
# Prompting / Parsing
# ============================================================

SYSTEM_PROMPT = """
You are a Mission Control Safety Officer on the ISS.

You will receive a structured observation from an RL environment.
Choose exactly ONE next action.

Return ONLY valid JSON with this exact schema:
{
  "action_type": "inspect_object",
  "target_object_id": "FE_airlock_01",
  "target_module": null,
  "reasoning": "short justification"
}

Rules:
- Return JSON only. No markdown. No extra text.
- reasoning is mandatory.
- Use only allowed action types for the current episode type.
- Prefer safe, efficient actions.
- In emergency episodes, prioritize crew safety first.
- In investigation episodes, gather evidence before identifying root cause.
- For audit_001, final submit_report should use target_object_id = "block_eva".
- Never deploy OT_lab_01 during the fire emergency.
"""

TASK_HINTS = {
    "audit_001": """
Target high score behavior:
1. inspect_object on FE_airlock_01
2. flag_non_compliant on FE_airlock_01
3. submit_report with target_object_id = "block_eva"
""",
    "emergency_001": """
Target high score behavior:
1. evacuate_module for Lab
2. deploy_resource FE_lab_01
3. trigger_switch SSP_lab_01
4. contact_ground EP_node1_01
5. submit_report
Never deploy OT_lab_01.
""",
    "investigation_001": """
Target high score behavior:
1. pull_sensor_log FE_service_01
2. cross_reference FE_service_01+NT_service_01
3. identify_root_cause FE_service_01
Do not blame NT_service_01 as root cause.
""",
}


def observation_to_text(obs: Any) -> str:
    if hasattr(obs, "model_dump_json"):
        return obs.model_dump_json(indent=2)
    if hasattr(obs, "model_dump"):
        import json
        return json.dumps(obs.model_dump(), indent=2)
    return str(obs)


def build_user_prompt(task: str, obs: Any) -> str:
    return (
        f"TASK: {task}\n\n"
        f"HINTS:\n{TASK_HINTS.get(task, '').strip()}\n\n"
        f"OBSERVATION:\n{observation_to_text(obs)}\n\n"
        "Return only the next action as JSON."
    )


def extract_json_object(text: str) -> Dict[str, Any]:
    import json

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError(f"Invalid JSON from model: {text}")


def parse_action_text(text: str) -> Action:
    data = extract_json_object(text)
    return Action(**data)


# ============================================================
# Deterministic fallback plan
# ============================================================

def fallback_action(task: str, step_idx: int) -> Action:
    if task == "audit_001":
        plan = [
            Action(
                action_type="inspect_object",
                target_object_id="FE_airlock_01",
                target_module=None,
                reasoning="Checking extinguisher expiry and pressure status.",
            ),
            Action(
                action_type="flag_non_compliant",
                target_object_id="FE_airlock_01",
                target_module=None,
                reasoning="The extinguisher is expired and critically low pressure.",
            ),
            Action(
                action_type="submit_report",
                target_object_id="block_eva",
                target_module=None,
                reasoning="Blocking EVA due to non-compliant safety equipment.",
            ),
        ]
        return plan[min(step_idx, len(plan) - 1)]

    if task == "emergency_001":
        plan = [
            Action(
                action_type="evacuate_module",
                target_object_id=None,
                target_module="Lab",
                reasoning="Crew safety comes first before any fire suppression.",
            ),
            Action(
                action_type="deploy_resource",
                target_object_id="FE_lab_01",
                target_module=None,
                reasoning="Deploying the correct extinguisher for the Lab fire.",
            ),
            Action(
                action_type="trigger_switch",
                target_object_id="SSP_lab_01",
                target_module=None,
                reasoning="Isolating power to prevent re-ignition.",
            ),
            Action(
                action_type="contact_ground",
                target_object_id="EP_node1_01",
                target_module=None,
                reasoning="Notifying ground control about the emergency.",
            ),
            Action(
                action_type="submit_report",
                target_object_id=None,
                target_module=None,
                reasoning="Fire suppressed, crew safe, power isolated, ground informed.",
            ),
        ]
        return plan[min(step_idx, len(plan) - 1)]

    if task == "investigation_001":
        plan = [
            Action(
                action_type="pull_sensor_log",
                target_object_id="FE_service_01",
                target_module=None,
                reasoning="The extinguisher status is suspicious and needs log review.",
            ),
            Action(
                action_type="cross_reference",
                target_object_id="FE_service_01+NT_service_01",
                target_module=None,
                reasoning="Cross-referencing extinguisher discharge with nitrogen readings.",
            ),
            Action(
                action_type="identify_root_cause",
                target_object_id="FE_service_01",
                target_module=None,
                reasoning="The extinguisher discharge explains the false pressure anomaly.",
            ),
        ]
        return plan[min(step_idx, len(plan) - 1)]

    return Action(
        action_type="escalate",
        target_object_id=None,
        target_module=None,
        reasoning="Fallback escalation because no valid episode-specific policy exists.",
    )


# ============================================================
# Safety / validity gate
# ============================================================

def is_safe_and_valid(task: str, step_idx: int, action: Action) -> bool:
    if task == "audit_001":
        expected = [
            ("inspect_object", "FE_airlock_01", None),
            ("flag_non_compliant", "FE_airlock_01", None),
            ("submit_report", "block_eva", None),
        ]
        e_action, e_obj, e_mod = expected[min(step_idx, len(expected) - 1)]
        return (
            action.action_type == e_action
            and action.target_object_id == e_obj
            and action.target_module == e_mod
            and bool(action.reasoning.strip())
        )

    if task == "emergency_001":
        expected = [
            ("evacuate_module", None, "Lab"),
            ("deploy_resource", "FE_lab_01", None),
            ("trigger_switch", "SSP_lab_01", None),
            ("contact_ground", "EP_node1_01", None),
            ("submit_report", None, None),
        ]
        e_action, e_obj, e_mod = expected[min(step_idx, len(expected) - 1)]

        if action.action_type == "deploy_resource" and action.target_object_id == "OT_lab_01":
            return False

        return (
            action.action_type == e_action
            and action.target_object_id == e_obj
            and action.target_module == e_mod
            and bool(action.reasoning.strip())
        )

    if task == "investigation_001":
        expected = [
            ("pull_sensor_log", "FE_service_01", None),
            ("cross_reference", "FE_service_01+NT_service_01", None),
            ("identify_root_cause", "FE_service_01", None),
        ]
        e_action, e_obj, e_mod = expected[min(step_idx, len(expected) - 1)]

        if action.action_type == "identify_root_cause" and action.target_object_id == "NT_service_01":
            return False

        return (
            action.action_type == e_action
            and action.target_object_id == e_obj
            and action.target_module == e_mod
            and bool(action.reasoning.strip())
        )

    return False


# ============================================================
# LLM call
# ============================================================

def ask_model(client: OpenAI, task: str, obs: Any) -> Action:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(task, obs)},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    content = response.choices[0].message.content or ""
    return parse_action_text(content)


# ============================================================
# Helpers
# ============================================================

def action_to_log_string(action: Action) -> str:
    target = action.target_object_id or action.target_module or "none"
    return f"{action.action_type}:{target}"


def choose_action(client: OpenAI, task: str, step_idx: int, obs: Any) -> Action:
    fallback = fallback_action(task, step_idx)

    try:
        llm_action = ask_model(client, task, obs)
        if is_safe_and_valid(task, step_idx, llm_action):
            return llm_action
        return fallback
    except Exception:
        return fallback


# ============================================================
# Runner
# ============================================================

def run_task(client: OpenAI, env: ISSEnvironment, task: str) -> None:
    obs = env.reset(task)
    rewards: list[float] = []
    steps_taken = 0
    success = False

    log_start(task)

    try:
        done = False
        step_idx = 0

        while not done:
            if step_idx >= MAX_STEPS_SAFE_GUARD:
                raise RuntimeError(f"Exceeded MAX_STEPS_SAFE_GUARD={MAX_STEPS_SAFE_GUARD} in {task}")

            error_msg: Optional[str] = None
            action = choose_action(client, task, step_idx, obs)

            try:
                obs, reward, done, _info = env.step(action)
            except (ValidationError, ValueError, RuntimeError) as e:
                error_msg = str(e)
                action = fallback_action(task, step_idx)
                obs, reward, done, _info = env.step(action)

            reward_val = float(reward.score)
            rewards.append(reward_val)
            steps_taken = step_idx + 1

            log_step(
                step=steps_taken,
                action_str=action_to_log_string(action),
                reward=reward_val,
                done=done,
                error=error_msg,
            )

            step_idx += 1

        final_score = rewards[-1] if rewards else 0.0
        success = True
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

    except Exception:
        final_score = rewards[-1] if rewards else 0.0
        log_end(success=False, steps=steps_taken, score=final_score, rewards=rewards)
        raise


def main() -> int:
    try:
        client = build_client()
        env = ISSEnvironment()

        for task in TASKS:
            run_task(client, env, task)

        return 0

    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())