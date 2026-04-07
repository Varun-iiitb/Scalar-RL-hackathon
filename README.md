---
title: RLhackathon
emoji: "\U0001F680"
colorFrom: indigo
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# ISS Safety Operations Environment (OpenEnv)

A real-world OpenEnv environment where an AI agent acts as a **Mission Control Safety Officer** aboard the International Space Station. The agent must make multi-turn safety decisions under time pressure across three distinct episode types.

## Episodes

| ID | Difficulty | Type | Description |
|----|-----------|------|-------------|
| `audit_001` | Easy | Audit | Pre-EVA safety audit. Inspect equipment, flag non-compliant items, decide whether to block the spacewalk. |
| `emergency_001` | Medium | Emergency | Live fire in Lab Module. Evacuate crew, deploy correct extinguisher, isolate power, contact ground. |
| `investigation_001` | Hard | Investigation | Post-anomaly root cause analysis. Pull hidden sensor logs, cross-reference evidence, avoid red herrings. |

## Observation Space

**Type:** `structured_json`

Each observation includes:
- `episode_id`, `episode_type`, `mission_context`
- `objects`: list of `SafetyObject` (7 types across 6 ISS modules)
- `active_alerts`: fire, pressure_drop, medical, comms_loss
- `crew_locations`: current crew positions
- `evidence_log`: visible log entries (investigation episodes reveal hidden logs via actions)
- `actions_taken`: history of actions this episode
- `turns_remaining`: countdown to forced termination
- `ground_control_available`: comms status

## Action Space

**Type:** `categorical_with_target`

```json
{
  "action_type": "<action_name>",
  "target_object_id": "<object_id or null>",
  "target_module": "<module_name or null>",
  "reasoning": "<mandatory short justification>"
}
```

**12 action types** (episode-restricted):
- **Audit:** `inspect_object`, `flag_non_compliant`, `clear_for_mission`
- **Emergency:** `evacuate_module`, `deploy_resource`, `trigger_switch`, `contact_ground`
- **Investigation:** `pull_sensor_log`, `cross_reference`, `identify_root_cause`
- **Universal:** `submit_report`, `escalate`

## Reward Function

Scores range **0.0 to 1.0** with partial credit:
- **Outcome score**: correct actions weighted by importance
- **Efficiency bonus**: completing quickly (+0.15)
- **Danger penalty**: unsafe actions like deploying oxygen near fire (-0.50)
- **Escalation penalty**: lazy escalation instead of solving (-0.30)
- **Reasoning bonus**: providing substantive justifications (+0.05)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check (returns 200) |
| `POST` | `/reset` | Reset with `{"episode_id": "audit_001"}`, returns initial observation |
| `POST` | `/step` | Send action, returns `{observation, reward, done, info}` |
| `GET` | `/state` | Full internal environment state |

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_hf_token_here"
```

### 3. Run the server

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

### 4. Run inference

```bash
python inference.py
```

### Docker

```bash
docker build -t iss-safety-ops .
docker run -p 7860:7860 iss-safety-ops
```

## Project Structure

```
.
├── openenv.yaml          # OpenEnv spec definition
├── server.py             # FastAPI server (reset/step/state endpoints)
├── inference.py          # Baseline inference script (required)
├── baseline.py           # GPT-4o multi-turn baseline
├── app.py                # Gradio web UI
├── Dockerfile            # HF Spaces deployment
├── requirements.txt      # Python dependencies
├── env/
│   ├── objects.py        # Pydantic models (Action, Observation, Reward, etc.)
│   ├── environment.py    # ISSEnvironment class (OpenEnv interface)
│   ├── grader.py         # Episode-specific scoring logic
│   └── episodes/         # JSON episode definitions
│       ├── audit_001.json
│       ├── emergency_001.json
│       └── investigation_001.json
```
