#!/usr/bin/env python3
"""
Verification tests for ISS Safety Operations environment.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import ISSEnvironment
from env.objects import Action

def run_tests():
    env = ISSEnvironment()
    all_passed = True

    # ────────────────────────────────────────────────
    # Test 1: Audit episode loads and runs
    # ────────────────────────────────────────────────
    print("\n=== Test 1: Audit episode loads ===")
    obs = env.reset("audit_001")
    assert obs.episode_type == "audit", f"Expected 'audit', got '{obs.episode_type}'"
    assert obs.turns_remaining == 6, f"Expected 6 turns, got {obs.turns_remaining}"
    assert len(obs.objects) > 0, "Expected objects in observation"
    print(f"  ✓ episode_type={obs.episode_type}, turns_remaining={obs.turns_remaining}, objects={len(obs.objects)}")

    # ────────────────────────────────────────────────
    # Test 2: Correct audit sequence scores > 0.8
    # ────────────────────────────────────────────────
    print("\n=== Test 2: Correct audit sequence scores > 0.8 ===")
    env.reset("audit_001")

    action1 = Action(
        action_type="inspect_object",
        target_object_id="FE_airlock_01",
        reasoning="Checking extinguisher expiry and status before EVA clearance",
    )
    obs, reward, done, info = env.step(action1)
    assert not done, "Should not be done after inspect"
    assert reward.score == 0.0, f"Expected 0.0 (non-terminal), got {reward.score}"
    print(f"  ✓ inspect_object: done={done}, reward.score={reward.score}")

    # Reset for a clean scoring test
    env.reset("audit_001")
    env.step(Action(action_type="inspect_object", target_object_id="FE_airlock_01",
                    reasoning="Checking extinguisher status"))
    env.step(Action(action_type="flag_non_compliant", target_object_id="FE_airlock_01",
                    reasoning="FireExtinguisher FE_airlock_01 is expired (expiry_days_remaining=-3) and tag invalid"))
    obs, reward, done, info = env.step(Action(
        action_type="submit_report",
        target_module="block_eva",
        reasoning="EVA-42 must be blocked due to expired FireExtinguisher in Airlock. Safety compliance requires all equipment be valid before EVA clearance.",
    ))
    assert done, "Should be done after submit_report"
    print(f"  ✓ submit_report: done={done}, reward.score={reward.score:.3f}")
    print(f"    breakdown={reward.breakdown}")
    assert reward.score > 0.8, f"Expected score > 0.8, got {reward.score:.3f}"
    print(f"  ✓ Score {reward.score:.3f} > 0.8")

    # ────────────────────────────────────────────────
    # Test 3: Dangerous action is penalized
    # ────────────────────────────────────────────────
    print("\n=== Test 3: Dangerous action oxygen_near_fire ===")
    env.reset("emergency_001")
    dangerous_action = Action(
        action_type="deploy_resource",
        target_object_id="OT_lab_01",
        reasoning="Using oxygen tank to enhance combustion — THIS IS WRONG",
    )
    obs, reward, done, info = env.step(dangerous_action)
    print(f"  actions_taken={obs.actions_taken}")
    assert "DANGEROUS:oxygen_near_fire" in obs.actions_taken, \
        "Expected 'DANGEROUS:oxygen_near_fire' in actions_taken"
    print("  ✓ Dangerous oxygen_near_fire logged correctly")

    # ────────────────────────────────────────────────
    # Test 4: Investigation hidden log revealed
    # ────────────────────────────────────────────────
    print("\n=== Test 4: Hidden log revealed via pull_sensor_log ===")
    env.reset("investigation_001")
    # Initial evidence_log has 4 visible entries
    obs_init = env.reset("investigation_001")
    initial_len = len(obs_init.evidence_log)
    print(f"  Initial evidence_log length: {initial_len}")

    pull_action = Action(
        action_type="pull_sensor_log",
        target_object_id="FE_service_01",
        reasoning="Checking extinguisher discharge history to look for anomalies in timing",
    )
    obs, reward, done, info = env.step(pull_action)
    print(f"  After pull_sensor_log, evidence_log length: {len(obs.evidence_log)}")
    assert len(obs.evidence_log) > initial_len, "Hidden log should have been added to evidence_log"
    hidden_entry = next((e for e in obs.evidence_log if e.object_id == "FE_service_01"), None)
    assert hidden_entry is not None, "FE_service_01 log entry should now be visible"
    assert hidden_entry.visible_to_agent, "Entry should be visible after pull"
    print(f"  ✓ Hidden entry revealed: {hidden_entry.timestamp} — {hidden_entry.event[:60]}...")

    # ────────────────────────────────────────────────
    # Test 5: Grade functions return 0.0 to 1.0
    # ────────────────────────────────────────────────
    print("\n=== Test 5: grades are in [0.0, 1.0] ===")
    assert 0.0 <= reward.score <= 1.0, f"score {reward.score} out of range"

    # Full investigation scoring
    env2 = ISSEnvironment()
    env2.reset("investigation_001")
    env2.step(Action(action_type="pull_sensor_log", target_object_id="FE_service_01",
                     reasoning="Check extinguisher log for timing anomalies"))
    env2.step(Action(action_type="cross_reference", target_object_id="FE_service_01+NT_service_01",
                     reasoning="Cross-referencing extinguisher discharge with nitrogen pressure drop"))
    obs3, reward3, done3, _ = env2.step(Action(action_type="identify_root_cause",
                                               target_object_id="FE_service_01",
                                               reasoning="Extinguisher discharged before alarm — propellant interfered with nitrogen sensor"))
    assert 0.0 <= reward3.score <= 1.0, f"Investigation score {reward3.score} out of range"
    print(f"  ✓ Investigation score={reward3.score:.3f} (in range)")
    print(f"    breakdown={reward3.breakdown}")

    print("\n✅ ALL TESTS PASSED\n")

if __name__ == "__main__":
    run_tests()
