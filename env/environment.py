"""
env/environment.py
ISSEnvironment class — the main OpenEnv interface.
"""
from __future__ import annotations

import json
from pathlib import Path

from env.grader import grade
from env.objects import (
    Action,
    Alert,
    EnvironmentState,
    LogEntry,
    Observation,
    Reward,
    SafetyObject,
)

_EPISODES_DIR = Path(__file__).parent / "episodes"


class ISSEnvironment:
    """OpenEnv-compliant environment for ISS Safety Operations."""

    def __init__(self) -> None:
        self._state: EnvironmentState | None = None
        self._max_turns: int = 6
        self._reasoning_accumulator: list[str] = []
        self._pending_danger_flags: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, episode_id: str) -> Observation:
        """Load episode JSON and return the initial Observation."""
        episode_path = _EPISODES_DIR / f"{episode_id}.json"
        with episode_path.open() as f:
            data = json.load(f)

        # Build SafetyObject list
        objects: list[SafetyObject] = [
            SafetyObject(**obj) for obj in data["objects"]
        ]

        # Build object registry keyed by object_id
        object_registry: dict[str, SafetyObject] = {
            obj.object_id: obj for obj in objects
        }

        # Build active alerts
        active_alerts: list[Alert] = [
            Alert(**a) for a in data.get("active_alerts", [])
        ]

        # Build visible evidence log (only visible entries for investigation)
        evidence_log_raw = data.get("evidence_log", [])
        evidence_log: list[LogEntry] = [
            LogEntry(**e) for e in evidence_log_raw if e.get("visible_to_agent", True)
        ]

        # Build hidden logs
        hidden_logs_raw = data.get("hidden_logs", [])
        hidden_logs: list[LogEntry] = [LogEntry(**e) for e in hidden_logs_raw]

        # Fire-spread schedule — keys are string timesteps
        fire_spread_schedule: dict | None = data.get("fire_spread_schedule")

        max_turns: int = data.get("max_turns", 6)

        obs = Observation(
            episode_id=episode_id,
            episode_type=data["episode_type"],
            mission_context=data["mission_context"],
            timestep=0,
            objects=objects,
            active_alerts=active_alerts,
            crew_locations=dict(data["crew_locations"]),
            evidence_log=evidence_log,
            actions_taken=[],
            turns_remaining=max_turns,
            # Emergency episodes start with comms unavailable;
            # audit and investigation have ground control reachable from the start.
            ground_control_available=(data["episode_type"] != "emergency"),
        )

        self._state = EnvironmentState(
            episode_id=episode_id,
            episode_type=data["episode_type"],
            current_observation=obs,
            object_registry=object_registry,
            fire_spread_schedule=fire_spread_schedule,
            hidden_logs=hidden_logs,
            ground_truth=data["ground_truth"],
            done=False,
            total_reward=0.0,
        )
        # Store max_turns for grader
        self._max_turns: int = max_turns
        self._reasoning_accumulator: list[str] = []
        # Pending danger flags raised inside _apply_action;
        # flushed into actions_taken AFTER the action log in step().
        self._pending_danger_flags: list[str] = []

        return obs

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """Apply an action and return (obs, reward, done, info)."""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is already done. Call reset() to start a new episode.")

        # Validate action is permitted for this episode type
        self._validate_action(action)

        # Apply action effects to state
        self._apply_action(action)

        obs = self._state.current_observation

        # Increment timestep and decrement turns_remaining
        obs.timestep += 1
        obs.turns_remaining -= 1

        # Log the formatted action string FIRST, then append any danger flags.
        # This guarantees order: ["deploy_resource:OT_lab_01", "DANGEROUS:oxygen_near_fire"]
        action_log = self.format_action_log(action)
        obs.actions_taken.append(action_log)
        for flag in self._pending_danger_flags:
            obs.actions_taken.append(flag)
        self._pending_danger_flags.clear()

        # Accumulate reasoning for grader
        self._reasoning_accumulator.append(action.reasoning)

        # Fire spread for emergency episodes
        if self._state.episode_type == "emergency":
            self._spread_fire(obs.timestep)

        # Check terminal condition
        done = self._check_done(action)
        self._state.done = done

        # Compute reward only at terminal step
        if done:
            combined_reasoning = " | ".join(self._reasoning_accumulator)
            reward = grade(
                episode_type=self._state.episode_type,
                actions_taken=obs.actions_taken,
                ground_truth=self._state.ground_truth,
                turns_remaining=obs.turns_remaining,
                max_turns=self._max_turns,
                reasoning=combined_reasoning,
            )
            self._state.total_reward = reward.score
        else:
            reward = Reward(
                score=0.0,
                outcome_score=0.0,
                efficiency_bonus=0.0,
                danger_penalty=0.0,
                escalation_penalty=0.0,
                reasoning_bonus=0.0,
                breakdown={},
            )

        info: dict = {
            "timestep": obs.timestep,
            "turns_remaining": obs.turns_remaining,
            "actions_taken": list(obs.actions_taken),
        }

        return obs, reward, done, info

    def state(self) -> dict:
        """Return the full internal state as a dict."""
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return self._state.model_dump()

    # ------------------------------------------------------------------
    # Async wrappers (required by openenv-core's create_app)
    # ------------------------------------------------------------------

    async def reset_async(self, episode_id: str = "audit_001", **kwargs) -> Observation:
        return self.reset(episode_id)

    async def step_async(self, action: Action, **kwargs) -> Observation:
        obs, reward, done, info = self.step(action)
        return obs

    async def state_async(self) -> dict:
        return self.state()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_action_log(action: Action) -> str:
        """Always format as 'action_type:target'. Used everywhere."""
        target = action.target_object_id or action.target_module or "none"
        return f"{action.action_type}:{target}"

    def _validate_action(self, action: Action) -> None:
        """Raise ValueError if action is invalid for the current episode type."""
        episode_type = self._state.episode_type  # type: ignore[union-attr]

        audit_only = {"inspect_object", "flag_non_compliant", "clear_for_mission"}
        emergency_only = {"deploy_resource", "evacuate_module", "trigger_switch", "contact_ground"}
        investigation_only = {"pull_sensor_log", "cross_reference", "identify_root_cause"}
        universal = {"escalate", "submit_report"}

        at = action.action_type

        if at in universal:
            return  # always allowed

        if episode_type == "audit" and at not in audit_only:
            raise ValueError(
                f"Action '{at}' is not permitted in 'audit' episodes. "
                f"Allowed: {audit_only | universal}"
            )
        if episode_type == "emergency" and at not in emergency_only:
            raise ValueError(
                f"Action '{at}' is not permitted in 'emergency' episodes. "
                f"Allowed: {emergency_only | universal}"
            )
        if episode_type == "investigation" and at not in investigation_only:
            raise ValueError(
                f"Action '{at}' is not permitted in 'investigation' episodes. "
                f"Allowed: {investigation_only | universal}"
            )

    def _apply_action(self, action: Action) -> None:  # noqa: C901
        """Mutate observation / registry based on action_type."""
        obs = self._state.current_observation  # type: ignore[union-attr]
        registry = self._state.object_registry  # type: ignore[union-attr]
        at = action.action_type

        if at == "inspect_object":
            # No state change — logged only (logging happens in step)
            pass

        elif at == "flag_non_compliant":
            # Log only via action_log — nothing to mutate here
            pass

        elif at == "clear_for_mission":
            # Log only via action_log — nothing to mutate here
            pass

        elif at == "deploy_resource":
            obj_id = action.target_object_id
            if obj_id and obj_id in registry:
                obj = registry[obj_id]
                obj.status = "depleted"  # type: ignore[assignment]

                # Dangerous combination check
                if obj.object_type == "OxygenTank":
                    fire_in_module = any(
                        a.alert_type == "fire" and a.module == obj.module
                        for a in obs.active_alerts
                    )
                    if fire_in_module:
                        # Stage the flag; it will be flushed AFTER the action
                        # log is appended in step(), preserving correct order.
                        self._pending_danger_flags.append("DANGEROUS:oxygen_near_fire")

                # Keep observation objects in sync
                obs.objects = list(registry.values())

        elif at == "evacuate_module":
            target_module = action.target_module
            if target_module:
                for crew_id, location in obs.crew_locations.items():
                    if location == target_module:
                        obs.crew_locations[crew_id] = "Node1"

        elif at == "trigger_switch":
            obj_id = action.target_object_id
            if obj_id and obj_id in registry:
                panel = registry[obj_id]
                panel.status = "depleted"  # type: ignore[assignment]
                # Remove fire alerts in the same module
                module = panel.module
                obs.active_alerts = [
                    a for a in obs.active_alerts
                    if not (a.alert_type == "fire" and a.module == module)
                ]
                obs.objects = list(registry.values())

        elif at == "contact_ground":
            obs.ground_control_available = True

        elif at == "pull_sensor_log":
            obj_id = action.target_object_id
            hidden = self._state.hidden_logs or []  # type: ignore[union-attr]
            for entry in hidden:
                if entry.object_id == obj_id and not entry.visible_to_agent:
                    entry.visible_to_agent = True
                    obs.evidence_log.append(entry)

        elif at == "cross_reference":
            # Logged via format_action_log as "cross_reference:obj1+obj2"
            # No additional state change needed
            pass

        elif at == "identify_root_cause":
            # Terminal — logged only
            pass

        elif at == "submit_report":
            # Terminal — logged only
            pass

        elif at == "escalate":
            # Terminal — logged only
            pass

    def _spread_fire(self, timestep: int) -> None:
        """Check fire spread schedule and add new fire alerts if needed."""
        schedule = self._state.fire_spread_schedule  # type: ignore[union-attr]
        if not schedule:
            return
        module = schedule.get(str(timestep))
        if module:
            obs = self._state.current_observation  # type: ignore[union-attr]
            new_alert = Alert(
                alert_id=f"fire_{module.lower()}_{timestep:02d}",
                alert_type="fire",
                module=module,
                severity="high",
                triggered_at_timestep=timestep,
            )
            obs.active_alerts.append(new_alert)

    def _check_done(self, action: Action) -> bool:
        """Return True if this action terminates the episode."""
        terminal_actions = {"submit_report", "escalate", "identify_root_cause"}
        if action.action_type in terminal_actions:
            return True
        obs = self._state.current_observation  # type: ignore[union-attr]
        if obs.turns_remaining <= 0:
            return True
        return False
