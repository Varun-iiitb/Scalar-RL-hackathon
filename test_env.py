"""
test_env.py
Sanity check for ISSEnvironment across all three episode types.
Run with: python test_env.py
"""

from env.environment import ISSEnvironment
from env.objects import Action


# ---------------------------------------------------------------------------
# Test 1 — Audit episode
# ---------------------------------------------------------------------------

def test_audit():
    env = ISSEnvironment()
    obs = env.reset("audit_001")

    print("Episode type:", obs.episode_type)
    print("Objects count:", len(obs.objects))
    print("Turns remaining:", obs.turns_remaining)

    assert obs.episode_type == "audit"
    assert obs.turns_remaining == 6
    assert any(o.object_id == "FE_airlock_01" for o in obs.objects)

    # Turn 1 — inspect the non-compliant extinguisher
    action = Action(
        action_type="inspect_object",
        target_object_id="FE_airlock_01",
        target_module=None,
        reasoning="Checking extinguisher expiry and pressure status",
    )
    obs, reward, done, info = env.step(action)
    assert done is False

    # Turn 2 — flag it as non-compliant
    action = Action(
        action_type="flag_non_compliant",
        target_object_id="FE_airlock_01",
        target_module=None,
        reasoning="Extinguisher is expired and pressure critically low at 0.12",
    )
    obs, reward, done, info = env.step(action)

    # Turn 3 — submit report with decision as target_object_id so grader sees
    #           "submit_report:block_eva" in actions_taken
    action = Action(
        action_type="submit_report",
        target_object_id="block_eva",
        target_module=None,
        reasoning="Blocking EVA due to non-compliant extinguisher",
    )
    obs, reward, done, info = env.step(action)

    assert done is True
    assert reward.score > 0.0
    print("Audit reward score:", reward.score)
    print("Audit test PASSED")


# ---------------------------------------------------------------------------
# Test 2 — Emergency episode
# ---------------------------------------------------------------------------

def test_emergency():
    env = ISSEnvironment()
    obs = env.reset("emergency_001")

    assert obs.episode_type == "emergency"
    assert len(obs.active_alerts) > 0
    print("Active alerts:", [a.module for a in obs.active_alerts])

    # Turn 1 — evacuate crew before any resource deployment
    action = Action(
        action_type="evacuate_module",
        target_object_id=None,
        target_module="Lab",
        reasoning="Crew member inside Lab, must evacuate before any resource deployment",
    )
    obs, reward, done, info = env.step(action)
    assert done is False
    print("Crew locations after evacuation:", obs.crew_locations)
    assert obs.crew_locations.get("crew_1") == "Node1"

    # Turn 2 — deploy the correct extinguisher
    action = Action(
        action_type="deploy_resource",
        target_object_id="FE_lab_01",
        target_module=None,
        reasoning="Deploying correct extinguisher to suppress Lab fire",
    )
    obs, reward, done, info = env.step(action)

    # Turn 3 — trigger the safety switch panel
    action = Action(
        action_type="trigger_switch",
        target_object_id="SSP_lab_01",
        target_module=None,
        reasoning="Isolating Lab power to prevent re-ignition",
    )
    obs, reward, done, info = env.step(action)

    # Turn 4 — contact ground control
    action = Action(
        action_type="contact_ground",
        target_object_id="EP_node1_01",
        target_module=None,
        reasoning="Notifying ground control of fire incident",
    )
    obs, reward, done, info = env.step(action)

    # Turn 5 — submit report (terminal)
    action = Action(
        action_type="submit_report",
        target_object_id=None,
        target_module=None,
        reasoning="Fire suppressed, crew safe, power isolated, ground notified",
    )
    obs, reward, done, info = env.step(action)

    assert done is True
    assert reward.score > 0.5
    print("Emergency reward score:", reward.score)
    print("Emergency test PASSED")


# ---------------------------------------------------------------------------
# Test 3 — Investigation episode
# ---------------------------------------------------------------------------

def test_investigation():
    env = ISSEnvironment()
    obs = env.reset("investigation_001")

    assert obs.episode_type == "investigation"
    print("Visible evidence log entries:", len(obs.evidence_log))
    assert len(obs.evidence_log) == 4

    # Turn 1 — pull log for the obvious suspect (red herring)
    action = Action(
        action_type="pull_sensor_log",
        target_object_id="NT_service_01",
        target_module=None,
        reasoning="Starting with the obvious nitrogen pressure drop",
    )
    obs, reward, done, info = env.step(action)

    # Turn 2 — pull log for the depleted extinguisher (reveals hidden entry)
    action = Action(
        action_type="pull_sensor_log",
        target_object_id="FE_service_01",
        target_module=None,
        reasoning="Extinguisher is depleted with 0.0 pressure — investigating discharge",
    )
    obs, reward, done, info = env.step(action)
    print("Evidence log after pulling FE log:", len(obs.evidence_log))
    assert len(obs.evidence_log) == 5

    # Turn 3 — cross-reference discharge with pressure drop
    action = Action(
        action_type="cross_reference",
        target_object_id="FE_service_01",
        target_module=None,
        reasoning="Cross referencing extinguisher discharge at 14:33 with nitrogen drop at 14:32",
    )
    obs, reward, done, info = env.step(action)

    # Turn 4 — identify root cause (terminal)
    action = Action(
        action_type="identify_root_cause",
        target_object_id="FE_service_01",
        target_module=None,
        reasoning="Extinguisher propellant caused false nitrogen reading — not a real leak",
    )
    obs, reward, done, info = env.step(action)

    assert done is True
    assert reward.score > 0.7
    print("Investigation reward score:", reward.score)
    print("Investigation test PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        test_audit()
        test_emergency()
        test_investigation()
        print("\nALL TESTS PASSED — safe to hand off to Person 3")
    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
