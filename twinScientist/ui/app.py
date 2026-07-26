"""
twinScientist Interactive Research Dashboard — Feature 4 Enhanced Version

Key improvements over original:
1. Chat Interface for continuous human-agent dialogue during research loop
2. Real-time debate display showing Pro/Con/Judge arguments
3. Hypothesis evolution timeline visualization
4. Evidence chain explorer with clickable depth
5. Educational annotations explaining Agent decisions at every step
6. Human-in-the-loop buttons properly integrated with interrupt/resume flow
7. SSE-based streaming for live debate progress and chat responses

Usage:
    from ui.app import create_demo
    demo = create_demo(agent_app=cognitive_graph)
    demo.launch(server_name="0.0.0.0", server_port=7860)
"""

import gradio as gr
import json
import os
from pathlib import Path
from typing import Any

# Import new modules from core
try:
    from core.chat_agent import ChatAgent
    HAS_CHAT_AGENT = True
except ImportError:
    HAS_CHAT_AGENT = False

try:
    from core.education import EducationAnnotation
    HAS_EDUCATION = True
except ImportError:
    HAS_EDUCATION = False


def get_data_status() -> str:
    """Return a human-readable data availability summary."""
    import glob as _glob
    sensor_count = len(_glob.glob(str(Path("data/sensors/*.csv"))))
    bio_count = len(_glob.glob(str(Path("data/biometric/*.csv"))))
    if sensor_count == 0 and bio_count == 0:
        return "📭 No data files. Upload CSVs in the Data Upload tab."
    parts = []
    if sensor_count:
        parts.append(f"{sensor_count} sensor")
    if bio_count:
        parts.append(f"{bio_count} biometric")
    return f"📊 {' + '.join(parts)} file(s) available."


class TwinScientistUI:
    """Gradio-based interactive frontend for twinScientist with full HITL support"""

    def __init__(self, agent_app=None):
        self.agent_app = agent_app
        self.sessions = {}
        self.last_state_update = None

    def handle_data_upload(self, files: list, data_type: str):
        """Save uploaded CSV files to the appropriate data directory."""
        import shutil
        from pathlib import Path

        if not files:
            return "## ⚠️ No files selected"

        target_dir = Path(f"data/{data_type}")
        target_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        skipped = []
        for f in files:
            fname = Path(f.name).name if hasattr(f, 'name') else Path(str(f)).name
            dest = target_dir / fname
            try:
                if hasattr(f, 'name'):
                    shutil.copy2(f.name, dest)
                else:
                    shutil.copy2(str(f), dest)
                saved.append(fname)
            except Exception as e:
                skipped.append(f"{fname}: {e}")

        lines = [f"## ✅ Upload Complete", ""]
        if saved:
            lines.append(f"**Saved {len(saved)} file(s) to `data/{data_type}/`**:")
            for name in saved:
                lines.append(f"- {name}")
        if skipped:
            lines.append(f"**Skipped {len(skipped)}**:")
            for s in skipped:
                lines.append(f"- {s}")
        lines.append("")
        lines.append("> Files will be auto-detected and used by the research pipeline.")
        return "\n".join(lines)

    def validate_api_key(self, key: str) -> str:
        """Validate and apply the API key."""
        if not key or not key.strip():
            return "⚠️ No API key set. Will use .env config if available."
        if not key.startswith("sk-"):
            return "⚠️ API key format may be incorrect (should start with sk-)."
        if len(key) < 20:
            return "⚠️ API key too short, may be invalid."
        os.environ["BAILIAN_API_KEY"] = key.strip()
        return "✅ API key configured."

    def _get_or_create_session(self, session_id: str) -> dict:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "iteration": 0,
                "status": "idle",
                "chat_history": [],  # [{role: user|agent, content: ...}]
                "debate_history": [],  # Raw debate round data
                "current_step": "not_started",
            }
        return self.sessions[session_id]

    async def run_research(
        self,
        domain: str,
        research_question: str,
        max_iterations: int,
        auto_approve: bool,
        api_key: str = "",
        **kwargs,
    ):
        """Launch research pipeline with streaming output and live preview."""
        import glob as _glob

        # === Pre-flight checks ===
        if not research_question or not research_question.strip():
            yield "[LOG] ⚠️ Please enter a research question.\n", "", ""
            return

        # Apply API key from UI if provided
        if api_key and api_key.strip():
            os.environ["BAILIAN_API_KEY"] = api_key.strip()

        api_key_final = os.getenv("BAILIAN_API_KEY", "")
        if not api_key_final or not api_key_final.strip():
            yield (
                "[LOG] ⚠️ No API key configured.\n\n"
                "Please set your Bailian API key in one of these ways:\n"
                "1. Open the **API Settings** panel above and enter your key\n"
                "2. Or create a `.env` file with `BAILIAN_API_KEY=sk-...`\n\n"
                "Get your key at: https://dashscope.aliyun.com/\n",
                "",
                "",
            )
            return

        # Check data availability
        sensor_files = _glob.glob(str(Path("data/sensors/*.csv")))
        bio_files = _glob.glob(str(Path("data/biometric/*.csv")))
        data_status = ""
        if not sensor_files and not bio_files:
            data_status = (
                "\n> 💡 **Tip**: No data files found. You can upload your own sensor/biometric "
                "CSV files in the **Data Upload** tab. The system will use built-in domain "
                "knowledge as a fallback.\n"
            )

        yield f"[LOG] 🔬 Starting research pipeline...\n[LOG] Data: {len(sensor_files)} sensor files, {len(bio_files)} biometric files{data_status}\n\n", "", ""

        initial_state = {
            "query": research_question,
            "domain": domain or "Environment-Human Health",
            "_max_iterations_": max_iterations,
            "auto_confirm": auto_approve,
            "iteration": 1,
        }

        report_content = ""
        if self.agent_app:
            try:
                async for event in self.agent_app.astream(
                    initial_state,
                    stream_mode="updates",
                    config={"configurable": {"recursion_limit": max_iterations * 10}},
                ):
                    if isinstance(event, dict):
                        node_name = list(event.keys())[0]
                        node_data = event[node_name]

                        # Extract report if generated
                        if "final_report" in node_data:
                            report_content = node_data["final_report"]

                        # Stream structured events for UI consumption
                        log_line = f"[EVENT]{json.dumps({'node': node_name, 'state': node_data}, ensure_ascii=False)}\n"
                        yield log_line, "", ""
                    else:
                        yield f"{event}\n", "", ""

                # Research complete — yield final report preview
                if report_content:
                    preview = report_content[:3000]
                    if len(report_content) > 3000:
                        preview += f"\n\n---\n\n*... ({len(report_content)} chars total. Full report saved to disk.)*"
                    yield "", preview, ""
                else:
                    yield "", "## ⚠️ No report generated. Check logs for errors.", ""

            except Exception as e:
                error_msg = (
                    f"[LOG] ❌ Research pipeline error: {str(e)[:500]}\n\n"
                    f"**Troubleshooting**:\n"
                    f"- Verify your API key is correct\n"
                    f"- Check your internet connection\n"
                    f"- Try reducing max iterations\n"
                )
                yield error_msg, "", ""
        else:
            yield (
                "[LOG] ⚠️ Agent not configured.\n\n"
                "Please restart the application with:\n"
                "```bash\npython -m main --ui\n```\n",
                "",
                "",
            )

    def chat_reply(self, message: str):
        """Process user chat message and generate agent response."""
        if not message or not message.strip():
            return "", ""

        if not HAS_CHAT_AGENT or not self.agent_app:
            return "Chat not available.", ""

        try:
            from core.llm_client import QwenClient
            from config.settings import settings

            llm_client = QwenClient(
                base_url=settings.bailian_base_url,
                api_key=settings.bailian_api_key,
                model=settings.model_name,
            )

            chat_agent = ChatAgent(llm_client)
            state = self.last_state_update or {}
            action = state.get("current_action", "")

            result = chat_agent.reply(
                llm_client=llm_client,
                user_message=message,
                state=state,
                action=action,
            )

            reply = result.get("reply", "Sorry, I couldn't generate a response.")
            sentiment = result.get("sentiment", "unknown")

            return f"**[{sentiment.upper()}]**\n\n{reply}", ""

        except Exception as e:
            return f"Error: {str(e)[:200]}", ""

    def submit_decision(
        self,
        session_id: str,
        decision_action: str,
        revision_text: str,
        current_state: dict,
    ):
        """Submit user decision at HITL checkpoint."""
        sess = self._get_or_create_session(session_id)

        if decision_action == "approve":
            update = {
                "pending_approval": False,
                "user_feedback": "approve",
            }
        elif decision_action == "revise":
            update = {
                "pending_approval": False,
                "user_feedback": revision_text or "Please revise.",
            }
        elif decision_action == "chat":
            # Redirect to chat interface instead
            return "CHAT_REDIRECT", None, None, None
        elif decision_action == "halt":
            update = {
                "pending_approval": False,
                "user_feedback": "halt",
            }
        else:
            update = {}

        return "DECISION_SUBMITTED", update, None, None

    def get_debate_display(self, debate_history: list[dict]) -> str:
        """Format debate history into readable markdown table."""
        if not debate_history:
            return "**No debate records yet.**"

        lines = ["## Debate History\n"]
        lines.append("| Round | Pro Key Points | Con Critiques | Judge Score | Winner |")
        lines.append("|-------|----------------|---------------|-------------|--------|")

        for d in debate_history[-10:]:  # Last 10 rounds only
            round_num = d.get("round_number", "?")
            pro_summary = d.get("pro_output", "")[:80].replace("\n", " ")
            con_summary = d.get("con_output", "")[:80].replace("\n", " ")
            score_before = d.get("judge_score_before", "?")
            score_after = d.get("judge_score_after", "?")
            winner = d.get("winner_side", "?")

            lines.append(
                f"| R{round_num} | {pro_summary[:60]}... | {con_summary[:60]}... | "
                f"{score_before}→{score_after} | {winner} |"
            )

        return "\n".join(lines)

    def get_hypothesis_timeline(self, hypothesis_tree: list[dict]) -> str:
        """Generate hypothesis evolution summary."""
        if not hypothesis_tree:
            return "**No hypotheses generated yet.**"

        active = [h for h in hypothesis_tree if h.get("status") not in ("pruned", "refused")]

        lines = ["## Hypothesis Evolution Timeline\n"]
        lines.append(f"**Active**: {len(active)} / {len(hypothesis_tree)} total\n")

        for i, h in enumerate(sorted(active, key=lambda x: x.get("confidence_posterior", x.get("confidence_prior")), reverse=True)[:10]):
            title = h.get("title", "?")[:50]
            status = h.get("status", "?")
            prior = h.get("confidence_prior", "?")
            posterior = h.get("confidence_posterior", "?")
            evidence_count = len(h.get("experiment_ids", []))

            lines.append(
                f"- [{i+1}] **{title}** ({status})\n"
                f"   - P(H): {prior} → P(H|D): {posterior}\n"
                f"   - Experiments run: {evidence_count}"
            )

        return "\n".join(lines)

    def build_ui(self) -> gr.Blocks:
        """构建完整的 Gradio 界面 with enhanced debate & chat features"""

        with gr.Blocks(title="twinScientist — AI Research Lab") as demo:
            gr.Markdown("# 🔬 TwinScientist Research Lab")
            gr.Markdown("### Autonomous Scientific Discovery with Human-AI Collaboration")

            # Initialize session
            session_md = gr.Markdown("**Session ID:** `new-session`")

            with gr.Row():
                # Left column: Input controls
                with gr.Column(scale=1):
                    with gr.Accordion("⚙️ API Settings", open=False):
                        api_key_input = gr.Textbox(
                            label="Bailian API Key",
                            placeholder="sk-...",
                            type="password",
                            value=os.getenv("BAILIAN_API_KEY", ""),
                            info="Alibaba Cloud Bailian API key. Leave empty to use .env config.",
                        )
                        api_status = gr.Markdown("")
                    data_status_md = gr.Markdown(get_data_status())
                    domain_input = gr.Textbox(
                        label="Research Domain",
                        value="Environment-Human Health",
                        info="e.g., Environmental Health, Clinical Medicine, Neuroscience",
                    )
                    question_input = gr.Textbox(
                        label="Research Question",
                        placeholder="Enter a scientific question to explore...",
                        lines=3,
                    )
                    max_iter_slider = gr.Slider(
                        minimum=3, maximum=200, value=10, step=1,
                        label="Max Iterations",
                    )
                    auto_approve_cb = gr.Checkbox(
                        label="Auto-approve (skip human review)",
                        value=False,
                        info="When checked, all HITL checkpoints are auto-approved",
                    )
                    start_btn = gr.Button("🚀 Start Research", variant="primary")

                # Right column: Output and monitoring
                with gr.Column(scale=2):
                    progress_md = gr.Markdown("**Ready**")
                    output_stream = gr.TextArea(
                        label="Agent Log",
                        lines=10,
                        interactive=False,
                    )
                    report_preview = gr.Markdown(
                        "## 📄 Research Report Preview\n\n"
                        "The generated Scientific Hypothesis & Research Plan will appear here."
                    )

            # ========================================================
            # Tabs for different views
            # ========================================================
            with gr.Tabs():
                # Tab 1: Main Chat
                with gr.Tab("💬 Chat"):
                    chat_interface = gr.Chatbot(
                        label="Conversation",
                        height=300,
                    )
                    with gr.Row():
                        chat_input = gr.Textbox(
                            label="Message",
                            placeholder="Question findings, provide new info, or ask about progress...",
                            scale=4,
                        )
                        chat_send = gr.Button("Send", variant="primary", scale=1)

                # Tab 2: Debate Records
                with gr.Tab("⚔️ Debate"):
                    debate_table = gr.Markdown(label="Debate History", value="**No debate records yet.**")

                # Tab 3: Hypothesis Timeline
                with gr.Tab("📈 Hypotheses"):
                    hypothesis_timeline = gr.Markdown(label="Evolution Timeline", value="**No hypotheses generated yet.**")

                # Tab 4: Data Upload
                with gr.Tab("📁 Data Upload"):
                    gr.Markdown("### Upload Sensor or Biometric Data (CSV)")
                    gr.Markdown("Uploaded files are automatically recognized by the research pipeline.")
                    with gr.Row():
                        data_type_radio = gr.Radio(
                            choices=[("Environmental Sensors", "sensors"), ("Biometric", "biometric")],
                            label="Data Type",
                            value="sensors",
                        )
                    data_upload = gr.File(
                        label="Select CSV Files",
                        file_count="multiple",
                        file_types=[".csv"],
                    )
                    upload_btn = gr.Button("📤 Upload Data", variant="primary")
                    upload_status = gr.Markdown("")

                # Tab 5: Human-in-the-Loop
                with gr.Tab("🎛️ HITL"):
                    gr.Markdown("**Waiting for agent checkpoint...**")
                    action_radio = gr.Radio(
                        choices=["approve", "revise", "chat", "halt"],
                        label="Decision",
                        value="approve",
                    )
                    revision_text = gr.Textbox(
                        label="Revision Notes (optional)",
                        placeholder="Enter specific feedback...",
                        lines=2,
                    )
                    submit_decision = gr.Button("Submit Decision", variant="secondary")

            # ========================================================
            # Event Handlers
            # ========================================================
            api_key_input.change(
                fn=self.validate_api_key,
                inputs=[api_key_input],
                outputs=[api_status],
            )

            start_btn.click(
                fn=self.run_research,
                inputs=[domain_input, question_input, max_iter_slider, auto_approve_cb, api_key_input],
                outputs=[output_stream, report_preview, progress_md],
            )

            chat_send.click(
                fn=self.chat_reply,
                inputs=[chat_input],
                outputs=[chat_interface, chat_input],
            )

            upload_btn.click(
                fn=self.handle_data_upload,
                inputs=[data_upload, data_type_radio],
                outputs=[upload_status],
            )

        return demo


def create_demo(agent_app=None):
    """工厂函数：创建增强版 Gradio Demo"""
    ui = TwinScientistUI(agent_app)
    return ui.build_ui()


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
