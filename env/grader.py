"""
env/grader.py
Scoring logic for all 3 episode types.
"""
from __future__ import annotations

from env.objects import Reward


def _empty_reward() -> Reward:
    return Reward(
        score=0.0,
        outcome_score=0.0,
        efficiency_bonus=0.0,
        danger_penalty=0.0,
        escalation_penalty=0.0,
        reasoning_bonus=0.0,
        breakdown={},
    )


# ---------------------------------------------------------------------------
# Master router
# ---------------------------------------------------------------------------

def grade(
    episode_type: str,
    actions_taken: list[str],
    ground_truth: dict,
    turns_remaining: int,
    max_turns: int,
    reasoning: str,
) -> Reward:
    """Route to the correct grading function based on episode_type."""
    if episode_type == "audit":
        return grade_audit(actions_taken, ground_truth, turns_remaining, max_turns, reasoning)
    elif episode_type == "emergency":
        return grade_emergency(actions_taken, ground_truth, turns_remaining, max_turns, reasoning)
    elif episode_type == "investigation":
        return grade_investigation(actions_taken, ground_truth, turns_remaining, max_turns, reasoning)
    else:
        raise ValueError(f"Unknown episode_type: {episode_type}")


# ---------------------------------------------------------------------------
# Audit grader
# ---------------------------------------------------------------------------

def grade_audit(
    actions_taken: list[str],
    ground_truth: dict,
    turns_remaining: int,
    max_turns: int,
    reasoning: str,
) -> Reward:
    """
    Score breakdown:
      +0.40  Flagged all non_compliant_objects correctly  (partial credit)
      +0.30  Cleared all compliant_objects correctly       (partial credit)
      +0.30  submit_report with correct decision
      +0.15  Efficiency bonus (only if outcome >= 0.7)
      -0.30  Lazy escalation penalty
      +0.05  Reasoning bonus (len > 50)
    """
    non_compliant: list[str] = ground_truth.get("non_compliant_objects", [])
    compliant: list[str] = ground_truth.get("compliant_objects", [])
    expected_decision: str = ground_truth.get("decision", "")

    # --- Flag score ---
    flagged_correctly = sum(
        1 for obj_id in non_compliant
        if f"flag_non_compliant:{obj_id}" in actions_taken
    )
    flag_score = (flagged_correctly / len(non_compliant) * 0.40) if non_compliant else 0.0

    # --- Clear score ---
    cleared_correctly = sum(
        1 for obj_id in compliant
        if f"clear_for_mission:{obj_id}" in actions_taken
    )
    clear_score = (cleared_correctly / len(compliant) * 0.30) if compliant else 0.0

    # --- Report score ---
    report_score = 0.0
    expected_action = f"submit_report:{expected_decision}"
    if expected_action in actions_taken:
        report_score = 0.30

    outcome_score = flag_score + clear_score + report_score

    # --- Efficiency bonus ---
    efficiency_bonus = 0.0
    if outcome_score >= 0.7:
        efficiency_bonus = (turns_remaining / max_turns) * 0.15

    # --- Escalation penalty ---
    escalation_penalty = 0.0
    if any(a.startswith("escalate") for a in actions_taken):
        escalation_penalty = -0.30

    # --- Reasoning bonus ---
    reasoning_bonus = 0.05 if len(reasoning) > 50 else 0.0

    total = outcome_score + efficiency_bonus + escalation_penalty + reasoning_bonus
    score = min(max(total, 0.0), 1.0)

    return Reward(
        score=score,
        outcome_score=outcome_score,
        efficiency_bonus=efficiency_bonus,
        danger_penalty=0.0,
        escalation_penalty=escalation_penalty,
        reasoning_bonus=reasoning_bonus,
        breakdown={
            "flag_score": flag_score,
            "clear_score": clear_score,
            "report_score": report_score,
            "flagged_correctly": flagged_correctly,
            "total_non_compliant": len(non_compliant),
            "cleared_correctly": cleared_correctly,
            "total_compliant": len(compliant),
        },
    )


# ---------------------------------------------------------------------------
# Emergency grader
# ---------------------------------------------------------------------------

def grade_emergency(
    actions_taken: list[str],
    ground_truth: dict,
    turns_remaining: int,
    max_turns: int,
    reasoning: str,
) -> Reward:
    """
    Score breakdown:
      +0.30  Crew evacuated BEFORE resource deployed
      +0.25  Correct extinguisher deployed
      +0.20  SafetySwitchPanel triggered
      +0.10  Ground control contacted
      +0.15  Report submitted
      -0.50  Dangerous action: oxygen near fire
      +0.15  Efficiency bonus (only if outcome >= 0.7)
    """
    correct_sequence: list[str] = ground_truth.get("correct_sequence", [])

    # --- Evacuation before deploy ---
    evac_score = 0.0
    first_evac = next(
        (i for i, a in enumerate(actions_taken) if a.startswith("evacuate_module:")), None
    )
    first_deploy = next(
        (i for i, a in enumerate(actions_taken) if a.startswith("deploy_resource:")), None
    )
    if first_evac is not None and (first_deploy is None or first_evac < first_deploy):
        evac_score = 0.30

    # --- Correct extinguisher deployed ---
    extinguisher_score = 0.0
    expected_extinguisher = correct_sequence[1] if len(correct_sequence) > 1 else ""
    if expected_extinguisher and expected_extinguisher in actions_taken:
        extinguisher_score = 0.25

    # --- SafetySwitchPanel triggered ---
    switch_score = 0.0
    expected_switch = correct_sequence[2] if len(correct_sequence) > 2 else ""
    if expected_switch and expected_switch in actions_taken:
        switch_score = 0.20

    # --- Ground control contacted ---
    ground_score = 0.0
    if any(a.startswith("contact_ground") for a in actions_taken):
        ground_score = 0.10

    # --- Report submitted ---
    report_score = 0.0
    if any(a.startswith("submit_report") for a in actions_taken):
        report_score = 0.15

    outcome_score = evac_score + extinguisher_score + switch_score + ground_score + report_score

    # --- Danger penalty ---
    danger_penalty = 0.0
    if "DANGEROUS:oxygen_near_fire" in actions_taken:
        danger_penalty = -0.50

    # --- Efficiency bonus ---
    efficiency_bonus = 0.0
    if outcome_score >= 0.7:
        efficiency_bonus = (turns_remaining / max_turns) * 0.15

    total = outcome_score + danger_penalty + efficiency_bonus
    score = min(max(total, 0.0), 1.0)

    return Reward(
        score=score,
        outcome_score=outcome_score,
        efficiency_bonus=efficiency_bonus,
        danger_penalty=danger_penalty,
        escalation_penalty=0.0,
        reasoning_bonus=0.0,
        breakdown={
            "evac_score": evac_score,
            "extinguisher_score": extinguisher_score,
            "switch_score": switch_score,
            "ground_score": ground_score,
            "report_score": report_score,
            "first_evac_index": first_evac,
            "first_deploy_index": first_deploy,
        },
    )


# ---------------------------------------------------------------------------
# Investigation grader
# ---------------------------------------------------------------------------

def grade_investigation(
    actions_taken: list[str],
    ground_truth: dict,
    turns_remaining: int,
    max_turns: int,
    reasoning: str,
) -> Reward:
    """
    Score breakdown:
      +0.25  Key log pulled
      +0.20  Cross reference includes root cause object
      +0.20  Avoided blaming red herring
      +0.35  Correct root cause identified
      +0.15  Efficiency bonus (only if outcome >= 0.8)
    """
    root_cause_object: str = ground_truth.get("root_cause_object", "")
    red_herring: str = ground_truth.get("red_herring", "")

    # --- Key log pulled ---
    log_score = 0.0
    if f"pull_sensor_log:{root_cause_object}" in actions_taken:
        log_score = 0.25

    # --- Cross reference includes root cause ---
    xref_score = 0.0
    for action in actions_taken:
        if action.startswith("cross_reference:") and root_cause_object in action:
            xref_score = 0.20
            break

    # --- Avoided blaming red herring ---
    avoid_herring_score = 0.20
    for action in actions_taken:
        if action.startswith("identify_root_cause:") and red_herring in action:
            avoid_herring_score = 0.0
            break

    # --- Correct root cause identified ---
    root_cause_score = 0.0
    if f"identify_root_cause:{root_cause_object}" in actions_taken:
        root_cause_score = 0.35

    outcome_score = log_score + xref_score + avoid_herring_score + root_cause_score

    # --- Efficiency bonus ---
    efficiency_bonus = 0.0
    if outcome_score >= 0.8:
        efficiency_bonus = (turns_remaining / max_turns) * 0.15

    total = outcome_score + efficiency_bonus
    score = min(max(total, 0.0), 1.0)

    return Reward(
        score=score,
        outcome_score=outcome_score,
        efficiency_bonus=efficiency_bonus,
        danger_penalty=0.0,
        escalation_penalty=0.0,
        reasoning_bonus=0.0,
        breakdown={
            "log_score": log_score,
            "xref_score": xref_score,
            "avoid_herring_score": avoid_herring_score,
            "root_cause_score": root_cause_score,
        },
    )
