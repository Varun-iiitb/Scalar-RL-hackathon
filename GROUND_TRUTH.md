# Ground Truth Justification Document
## ISS Safety Operations Environment — Episode Decisions

This document explains the correct answer for each episode and 
why it is correct. It exists to defend grader decisions against 
judge challenges and to help future contributors understand 
episode design intent.

---

## Episode 1 — audit_001 (Easy)
**Mission:** EVA-42 Pre-Mission Safety Check  
**Correct Decision:** block_eva  
**Non-compliant Object:** FE_airlock_01  

### Why block_eva is correct

The FireExtinguisher in the Airlock (FE_airlock_01) has two 
critical failures:

1. **Expired certification** — expiry_days_remaining is -3, 
meaning it expired 3 days ago. Any expired safety equipment 
automatically fails EVA clearance under ISS safety protocol. 
No exceptions.

2. **Critically low pressure** — pressure_level is 0.12, 
meaning it has only 12% charge remaining. A functional 
extinguisher requires minimum 0.80 pressure for EVA operations. 
This extinguisher could not suppress a fire if one occurred 
during the spacewalk.

Either failure alone is sufficient to block the EVA. Both 
together make it unambiguous.

### Why the other 7 objects are compliant

All other objects have positive expiry_days_remaining, 
inspection_tag_valid of true, and pressure levels above 
acceptable thresholds. None have degraded or failed status. 
There is no legitimate reason to flag any of them.

### What a failing agent looks like

An agent that clears FE_airlock_01 for mission without 
inspecting it — or one that only checks expiry and misses 
the pressure level — scores partial credit but not full marks. 
An agent that blocks EVA for the wrong object (false positive) 
receives a penalty.

---

## Episode 2 — emergency_001 (Medium)
**Mission:** Lab Module Fire Response  
**Correct Decision:** fire_suppressed via correct sequence  
**Correct Sequence:**
1. evacuate_module:Lab
2. deploy_resource:FE_lab_01
3. trigger_switch:SSP_lab_01
4. contact_ground:EP_node1_01
5. submit_report

### Why this sequence is correct

**Step 1 — Evacuate first:**  
crew_1 is inside the Lab when the fire starts. Human life 
takes absolute priority over equipment. Deploying any resource 
before evacuating crew violates ISS emergency protocol and 
incurs a -0.30 penalty. This is the most common mistake a 
naive agent will make.

**Step 2 — Deploy FE_lab_01, not OT_lab_01:**  
The FireExtinguisher (FE_lab_01) is the correct suppression 
tool. The OxygenTank (OT_lab_01) is also in the Lab module 
and is the primary trap in this episode. Oxygen accelerates 
combustion — deploying it near an active fire is catastrophic 
and incurs a -0.50 penalty. An agent that deploys the wrong 
resource fails this episode even if it does everything else 
correctly.

**Step 3 — Trigger the SafetySwitchPanel:**  
Isolating Lab module power (SSP_lab_01) prevents electrical 
re-ignition and cuts power to any systems feeding the fire. 
This is standard post-suppression protocol.

**Step 4 — Contact ground:**  
All ISS emergencies require ground control notification via 
EmergencyPhone (EP_node1_01). The phone is in Node1, not Lab, 
which means the agent must use the correct object ID. Using 
a non-existent or wrong phone object scores zero for this step.

**Step 5 — Submit report:**  
Formally closes the episode. An agent that resolves the fire 
but never submits loses the efficiency bonus.

### Why the fire spread schedule matters

Fire spreads to Node2 at timestep 2 and Airlock at timestep 4. 
An agent that wastes turns on unnecessary actions will face a 
spreading fire and lose control of the situation. This rewards 
decisive, efficient agents over ones that over-investigate.

### What a failing agent looks like

The two classic failures are:
- Deploying OxygenTank near fire (catastrophic penalty)
- Deploying extinguisher before evacuating crew (safety penalty)

An agent that does both scores near zero even with a correct 
final verdict.

---

## Episode 3 — investigation_001 (Hard)
**Mission:** Service Module Pressure Anomaly Investigation  
**Correct Decision:** identify_root_cause:FE_service_01  
**Root Cause:** Improper FireExtinguisher discharge caused a 
false nitrogen pressure reading  
**Red Herring:** NT_service_01 (NitrogenTank)  

### Why FE_service_01 is the correct root cause

The visible evidence log creates a convincing but false 
narrative: nitrogen pressure drops at 14:32, alarm triggers 
at 14:35, crew manually triggers the switch at 14:38. The 
obvious conclusion is that the NitrogenTank failed.

This is wrong.

The hidden log — only revealed if the agent pulls the 
FE_service_01 sensor log — shows that at 14:33 (one minute 
AFTER the pressure drop, one minute BEFORE the alarm), 
crew_member_2 discharged the FireExtinguisher. CO2 extinguisher 
propellant is a dense gas. When discharged in a sealed module, 
it interferes with nitrogen pressure sensors, producing a 
false low-pressure reading.

The NitrogenTank itself is still operational (status: 
operational, pressure: 0.79 which is lower than normal but 
explained entirely by the false sensor reading). There is no 
actual leak.

### The chain of reasoning an agent must follow

1. Pull NT_service_01 log — sees pressure drop at 14:32
2. Note that FE_service_01 has status "depleted" and 
   pressure_level 0.0 — this is a visible clue in the object 
   list that something happened to the extinguisher
3. Pull FE_service_01 log — hidden log is now revealed: 
   discharge at 14:33
4. Cross-reference FE_service_01 and NT_service_01 — 
   discharge at 14:33 precedes alarm at 14:35, propellant 
   explains false nitrogen reading
5. Identify root cause as FE_service_01

An agent that stops at step 1 and concludes "nitrogen tank 
failed" falls for the red herring. This is exactly what a 
pattern-matching agent does — it sees a pressure drop and 
jumps to the obvious answer without investigating further.

### Why this is fair despite being hard

The clue is visible from the start — FE_service_01 has 
status "depleted" and pressure 0.0 in the initial object list. 
A careful agent that reads all object statuses before 
investigating will notice the depleted extinguisher and 
investigate it. The hidden log is accessible — the agent 
just has to ask for it. No information is permanently 
withheld. The puzzle is solvable with good investigative 
reasoning.

### What a failing agent looks like

An agent that immediately calls identify_root_cause:NT_service_01 
without pulling any logs scores 0.0. An agent that pulls the 
NT log but never pulls the FE log scores 0.4 partial credit. 
Only an agent that pulls the FE log, sees the discharge, 
and correctly identifies the root cause scores full marks.

---

## Summary Table

| Episode | Correct Decision | Primary Trap | Partial Credit Path |
|---|---|---|---|
| audit_001 | block_eva | Clearing expired object | Finds expiry but misses pressure |
| emergency_001 | fire_suppressed | OxygenTank near fire | Correct sequence but inefficient |
| investigation_001 | identify_root_cause:FE_service_01 | NT_service_01 red herring | Pulls NT log but skips FE log |
