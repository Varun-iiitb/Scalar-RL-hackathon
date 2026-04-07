"""
app.py
------
Gradio web interface for the ISS Safety Operations Environment.
Deployed on Hugging Face Spaces.

This app lets users:
  1. Select an episode
  2. Watch the LLM agent play through it step by step
  3. See the final score and action breakdown

Run locally:
    pip install gradio
    export OPENAI_API_KEY=your_key_here
    python app.py
"""

import os
import json
import gradio as gr

from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY from .env automatically

from baseline import run_episode, format_observation, parse_action, SYSTEM_PROMPT, client
from env.environment import ISSEnvironment
from env.objects import Action

# ---------------------------------------------------------------------------
# Episode runner that yields step-by-step output for Gradio streaming
# ---------------------------------------------------------------------------

def run_episode_streaming(episode_id: str, api_key: str):
    """Run one episode and yield text chunks for Gradio streaming output."""
    if not api_key.strip():
        yield "❌ Please enter your OpenAI API key above.", "", ""
        return

    os.environ["OPENAI_API_KEY"] = api_key.strip()

    # Reinitialise client with new key
    import openai
    openai.api_key = api_key.strip()

    env = ISSEnvironment()

    try:
        obs = env.reset(episode_id)
    except Exception as e:
        yield f"❌ Failed to load episode: {e}", "", ""
        return

    done = False
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    log_lines = []
    turn = 0

    log_lines.append(f"🚀 Starting episode: **{episode_id}**\n")
    log_lines.append(f"📋 Mission: {obs.mission_context}\n")
    log_lines.append("---\n")
    yield "\n".join(log_lines), "", "⏳ Running..."

    while not done:
        turn += 1
        prompt = format_observation(obs)
        messages.append({"role": "user", "content": prompt})

        log_lines.append(f"### Turn {turn}")
        yield "\n".join(log_lines), "", "⏳ Calling LLM..."

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.2,
                max_tokens=300,
            )
            response_text = response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": response_text})
            action = parse_action(response_text)
        except Exception as e:
            log_lines.append(f"⚠️ LLM error: {e} — escalating.\n")
            action = Action(
                action_type="escalate",
                target_object_id=None,
                target_module=None,
                reasoning="LLM parse error, escalating.",
            )

        target = action.target_object_id or action.target_module or "—"
        log_lines.append(f"**Action:** `{action.action_type}` → `{target}`")
        log_lines.append(f"**Reasoning:** {action.reasoning}\n")
        yield "\n".join(log_lines), "", "⏳ Stepping environment..."

        obs, reward, done, info = env.step(action)

        if done:
            log_lines.append("---")
            log_lines.append(f"## ✅ Episode Complete")
            log_lines.append(f"**Final Score: {reward.score:.3f}**")
            log_lines.append(f"- Outcome Score: {reward.outcome_score:.3f}")
            log_lines.append(f"- Efficiency Bonus: {reward.efficiency_bonus:.3f}")
            log_lines.append(f"- Danger Penalty: {reward.danger_penalty:.3f}")
            log_lines.append(f"- Escalation Penalty: {reward.escalation_penalty:.3f}")
            log_lines.append(f"- Reasoning Bonus: {reward.reasoning_bonus:.3f}")
            log_lines.append(f"\n**Breakdown:** `{json.dumps(reward.breakdown, indent=2)}`")

            score_display = f"{reward.score:.3f} / 1.000"
            color = "🟢" if reward.score >= 0.7 else ("🟡" if reward.score >= 0.4 else "🔴")
            yield "\n".join(log_lines), score_display, f"{color} Done!"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EPISODE_DESCRIPTIONS = {
    "audit_001":        "🟢 Easy  — Pre-EVA safety audit. Find expired equipment before the spacewalk.",
    "emergency_001":    "🟡 Medium — Active lab fire. Protect crew, suppress fire, contact ground.",
    "investigation_001":"🔴 Hard  — Post-incident investigation. Navigate red herrings with incomplete logs.",
}

with gr.Blocks(
    title="ISS Safety Operations — LLM Agent",
    theme=gr.themes.Soft(primary_hue="indigo"),
) as demo:
    gr.Markdown("""
    # 🛸 ISS Safety Operations Environment
    ### LLM Agent Baseline Demo

    Watch a **GPT-4o agent** play through ISS safety scenarios in real time.
    The agent must make multi-turn decisions under time pressure to score 0.0–1.0.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            api_key_input = gr.Textbox(
                label="OpenAI API Key",
                placeholder="sk-...",
                type="password",
                info="Your key is used only for this session and never stored.",
            )
            episode_dropdown = gr.Dropdown(
                choices=list(EPISODE_DESCRIPTIONS.keys()),
                value="audit_001",
                label="Select Episode",
            )
            episode_info = gr.Markdown(EPISODE_DESCRIPTIONS["audit_001"])
            run_button = gr.Button("▶ Run Episode", variant="primary")

        with gr.Column(scale=2):
            score_display = gr.Textbox(label="Final Score", interactive=False)
            status_display = gr.Textbox(label="Status", interactive=False, value="Ready")
            log_display = gr.Markdown(label="Episode Log", value="*Select an episode and click Run.*")

    # Update episode description on dropdown change
    def update_description(ep):
        return EPISODE_DESCRIPTIONS.get(ep, "")

    episode_dropdown.change(update_description, inputs=episode_dropdown, outputs=episode_info)

    # Run button triggers streaming generator
    run_button.click(
        fn=run_episode_streaming,
        inputs=[episode_dropdown, api_key_input],
        outputs=[log_display, score_display, status_display],
    )

    gr.Markdown("""
    ---
    **Scoring Guide:**
    - 🟢 ≥ 0.70 — Excellent
    - 🟡 ≥ 0.40 — Partial
    - 🔴 < 0.40 — Poor

    Built for the OpenEnv Hackathon. [View source on GitHub](https://github.com/sahiti3636/Ghost-Hunter)
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
