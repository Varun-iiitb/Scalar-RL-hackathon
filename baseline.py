"""
baseline.py
-----------
Baseline LLM inference script for the ISS Safety Operations Environment.
Uses OpenAI GPT-4o to play all 3 episodes and report final scores.

Usage:
    export OPENAI_API_KEY=your_key_here
    python baseline.py

Or run a specific episode:
    python baseline.py --episode audit_001
"""

import os
import json
import argparse
import re

from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY from .env automatically

from openai import OpenAI
from env.environment import ISSEnvironment
from env.objects import Action, Observation

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

if not os.environ.get("OPENAI_API_KEY"):
    raise EnvironmentError(
        "OPENAI_API_KEY not found.\n"
        "Add it to your .env file:\n"
        "  OPENAI_API_KEY=sk-...\n"
    )

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Mission Control Safety Officer aboard the ISS.
You will receive safety situation reports each turn and must respond with
a single action to resolve the situation.

IMPORTANT — You must ALWAYS respond with valid JSON and nothing else:
{
  "action_type": "<one of the allowed actions>",
  "target_object_id": "<object id or null>",
  "target_module": "<module name or null>",
  "reasoning": "<your reasoning, minimum 20 words>"
}

ALLOWED ACTIONS BY EPISODE TYPE:
  audit:         inspect_object, flag_non_compliant, clear_for_mission, submit_report, escalate
  emergency:     deploy_resource, evacuate_module, trigger_switch, contact_ground, submit_report, escalate
  investigation: pull_sensor_log, cross_reference, identify_root_cause, submit_report, escalate

MODULES: Lab, Node1, Node2, Airlock, CrewQuarters, ServiceModule

KEY RULES:
  - In audit: inspect objects, flag non-compliant ones, then submit_report with target_object_id set to your decision ("block_eva" or "clear_eva")
  - In emergency: ALWAYS evacuate crew BEFORE deploying any resource. Never deploy oxygen near fire.
  - In investigation: pull sensor logs before drawing conclusions. The obvious answer may be a red herring.
  - submit_report, identify_root_cause, and escalate are TERMINAL — episode ends immediately.
  - You have limited turns. Be decisive.

Respond ONLY with the JSON object. No explanation outside the JSON."""


# ---------------------------------------------------------------------------
# Observation formatter
# ---------------------------------------------------------------------------

def format_observation(obs: Observation) -> str:
    """Convert an Observation into a human-readable prompt string for the LLM."""
    lines = []

    lines.append(f"=== ISS SAFETY OPERATIONS — TURN {obs.timestep} ===")
    lines.append(f"Episode Type : {obs.episode_type.upper()}")
    lines.append(f"Mission      : {obs.mission_context}")
    lines.append(f"Turns Left   : {obs.turns_remaining}")
    lines.append(f"Ground Ctrl  : {'AVAILABLE' if obs.ground_control_available else 'UNAVAILABLE'}")
    lines.append("")

    # Crew
    lines.append("CREW LOCATIONS:")
    for crew_id, loc in obs.crew_locations.items():
        lines.append(f"  {crew_id}: {loc}")
    lines.append("")

    # Active alerts
    if obs.active_alerts:
        lines.append("ACTIVE ALERTS:")
        for alert in obs.active_alerts:
            lines.append(f"  [{alert.severity.upper()}] {alert.alert_type} in {alert.module} (alert_id={alert.alert_id})")
        lines.append("")

    # Objects
    lines.append("SAFETY OBJECTS:")
    for obj in obs.objects:
        pressure = f", pressure={obj.pressure_level:.2f}" if obj.pressure_level is not None else ""
        expiry = f", expiry_days={obj.expiry_days_remaining}"
        tag = f", tag_valid={obj.inspection_tag_valid}"
        lines.append(
            f"  [{obj.module}] {obj.object_id} ({obj.object_type})"
            f" — status={obj.status}{pressure}{expiry}{tag}"
        )
    lines.append("")

    # Evidence log (investigation only)
    if obs.evidence_log:
        lines.append("EVIDENCE LOG:")
        for entry in obs.evidence_log:
            visibility = "" if entry.visible_to_agent else " [HIDDEN]"
            lines.append(f"  {entry.timestamp} | {entry.object_id} | {entry.event}{visibility}")
        lines.append("")

    # Actions already taken this episode
    if obs.actions_taken:
        lines.append("ACTIONS TAKEN SO FAR:")
        for a in obs.actions_taken:
            lines.append(f"  {a}")
        lines.append("")

    lines.append("What is your next action? Respond with valid JSON only.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_action(response_text: str) -> Action:
    """
    Parse the LLM's text response into an Action object.
    Attempts JSON extraction even if the model adds surrounding text.
    """
    # Try to extract JSON block if present
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON found in LLM response:\n{response_text}")

    data = json.loads(json_match.group())

    return Action(
        action_type=data["action_type"],
        target_object_id=data.get("target_object_id"),
        target_module=data.get("target_module"),
        reasoning=data.get("reasoning", "No reasoning provided."),
    )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env: ISSEnvironment, episode_id: str, verbose: bool = True) -> dict:
    """Run a single episode with the LLM agent. Returns result dict."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"  EPISODE: {episode_id}")
        print(f"{'='*60}")

    obs = env.reset(episode_id)
    done = False
    turn = 0
    history = []   # conversation history for multi-turn context

    # Start conversation with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while not done:
        turn += 1
        prompt = format_observation(obs)

        if verbose:
            print(f"\n--- Turn {turn} ---")

        # Add user message (current observation)
        messages.append({"role": "user", "content": prompt})

        # Call LLM
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,       # low temperature for deterministic safety decisions
            max_tokens=300,
        )

        response_text = response.choices[0].message.content.strip()

        # Add assistant response to history (multi-turn context)
        messages.append({"role": "assistant", "content": response_text})

        if verbose:
            print(f"LLM Response: {response_text}")

        # Parse into Action
        try:
            action = parse_action(response_text)
        except Exception as e:
            if verbose:
                print(f"[PARSE ERROR] {e} — escalating as fallback.")
            action = Action(
                action_type="escalate",
                target_object_id=None,
                target_module=None,
                reasoning="Parse error in LLM response, escalating to ground control for guidance.",
            )

        if verbose:
            target = action.target_object_id or action.target_module or "—"
            print(f"Action: {action.action_type} → {target}")
            print(f"Reasoning: {action.reasoning[:100]}...")

        # Step the environment
        obs, reward, done, info = env.step(action)

        history.append({
            "turn": turn,
            "action": f"{action.action_type}:{action.target_object_id or action.target_module or 'none'}",
            "reasoning": action.reasoning,
        })

        if done:
            if verbose:
                print(f"\n{'='*60}")
                print(f"  EPISODE COMPLETE — Score: {reward.score:.3f}")
                print(f"  Outcome:    {reward.outcome_score:.3f}")
                print(f"  Efficiency: {reward.efficiency_bonus:.3f}")
                print(f"  Danger Pen: {reward.danger_penalty:.3f}")
                print(f"  Escalation: {reward.escalation_penalty:.3f}")
                print(f"  Reasoning:  {reward.reasoning_bonus:.3f}")
                print(f"  Breakdown:  {reward.breakdown}")
                print(f"{'='*60}")

    return {
        "episode_id": episode_id,
        "score": reward.score,
        "outcome_score": reward.outcome_score,
        "efficiency_bonus": reward.efficiency_bonus,
        "danger_penalty": reward.danger_penalty,
        "escalation_penalty": reward.escalation_penalty,
        "reasoning_bonus": reward.reasoning_bonus,
        "turns_used": turn,
        "actions": history,
        "breakdown": reward.breakdown,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ISS Safety Ops — LLM Baseline Agent")
    parser.add_argument(
        "--episode",
        type=str,
        default=None,
        help="Run a specific episode (e.g. audit_001). Defaults to all 3.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-turn output, only show final scores.",
    )
    args = parser.parse_args()

    # API key is already loaded from .env at module import time

    env = ISSEnvironment()
    episodes = (
        [args.episode] if args.episode
        else ["audit_001", "emergency_001", "investigation_001"]
    )

    all_results = []
    for ep in episodes:
        result = run_episode(env, ep, verbose=not args.quiet)
        all_results.append(result)

    # Summary table
    print("\n" + "="*60)
    print("  FINAL SCORES")
    print("="*60)
    print(f"  {'Episode':<25} {'Score':>8}  {'Turns':>6}")
    print(f"  {'-'*25} {'-'*8}  {'-'*6}")
    for r in all_results:
        print(f"  {r['episode_id']:<25} {r['score']:>8.3f}  {r['turns_used']:>6}")

    avg = sum(r["score"] for r in all_results) / len(all_results)
    print(f"  {'AVERAGE':<25} {avg:>8.3f}")
    print("="*60)

    # Save results to JSON
    output_path = "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
