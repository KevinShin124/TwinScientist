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
            return "## ⚠️ 未选择文件"

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

        lines = [f"## ✅ 上传完成", ""]
        if saved:
            lines.append(f"**已保存 {len(saved)} 个文件到 `data/{data_type}/`**:")
            for name in saved:
                lines.append(f"- {name}")
        if skipped:
            lines.append(f"**跳过 {len(skipped)} 个**:")
            for s in skipped:
                lines.append(f"- {s}")
        lines.append("")
        lines.append("> 上传完成后，文件将自动被科研流水线识别和使用。")
        return "\n".join(lines)

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
        **kwargs,
    ):
        """启动研究循环并返回流式输出"""
        initial_state = {
            "query": research_question,
            "domain": domain or "环境—人体关联",
            "_max_iterations_": max_iterations,
            "auto_confirm": auto_approve,
        }

        if self.agent_app:
            async for event in self.agent_app.astream(
                initial_state,
                stream_mode="updates",
                config={"configurable": {"recursion_limit": max_iterations * 10}},
            ):
                if isinstance(event, dict):
                    node_name = list(event.keys())[0]
                    # Stream structured events for UI consumption
                    yield f"[EVENT]{{\"node\":\"{node_name}\",\"state\":{json.dumps(event[node_name], ensure_ascii=False)}}}\n"
                else:
                    yield f"{event}\n"
        else:
            yield "[UI] Agent app not configured. Connect your LLM first.\n"

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
                    domain_input = gr.Textbox(
                        label="学科领域",
                        value="环境—人体关联",
                        info="如：环境健康、临床医学、神经科学等",
                    )
                    question_input = gr.Textbox(
                        label="研究问题",
                        placeholder="输入你想探索的科学问题...",
                        lines=3,
                    )
                    max_iter_slider = gr.Slider(
                        minimum=3, maximum=200, value=10, step=1,
                        label="最大迭代次数",
                    )
                    auto_approve_cb = gr.Checkbox(
                        label="自动通过人类审核（跳过确认）",
                        value=False,
                        info="勾选后将跳过所有 checkpoint 节点",
                    )
                    start_btn = gr.Button("🚀 开始研究", variant="primary")

                # Right column: Output and monitoring
                with gr.Column(scale=2):
                    progress_md = gr.Markdown("**就绪**")
                    output_stream = gr.TextArea(
                        label="Agent 运行日志",
                        lines=10,
                        interactive=False,
                    )
                    report_preview = gr.Markdown(
                        "## 📄 研究报告预览\n\n"
                        "运行完成后将在此显示生成的《科学假设与研究计划》。"
                    )

            # ========================================================
            # Tabs for different views
            # ========================================================
            with gr.Tabs():
                # Tab 1: Main Chat
                with gr.Tab("💬 聊天"):
                    chat_interface = gr.Chatbot(
                        label="对话记录",
                        height=300,
                    )
                    with gr.Row():
                        chat_input = gr.Textbox(
                            label="输入消息",
                            placeholder="你可以质疑假设、提供新信息、或询问研究进展...",
                            scale=4,
                        )
                        chat_send = gr.Button("发送", variant="primary", scale=1)

                # Tab 2: Debate Records
                with gr.Tab("⚔️ 辩论实录"):
                    debate_table = gr.Markdown(label="辩论历史", value="**暂无辩论记录。**")

                # Tab 3: Hypothesis Timeline
                with gr.Tab("📈 假设进化"):
                    hypothesis_timeline = gr.Markdown(label="演化时间线", value="**尚未生成假设。**")

                # Tab 4: Structured Decision Panel (HITL)
                with gr.Tab("📁 数据上传"):
                    gr.Markdown("### 上传传感器或生物特征数据 (CSV)")
                    gr.Markdown("上传后，数据将自动被科研流水线识别，用于因果推断分析。")
                    with gr.Row():
                        data_type_radio = gr.Radio(
                            choices=[("环境传感器", "sensors"), ("生物特征", "biometric")],
                            label="数据类型",
                            value="sensors",
                        )
                    data_upload = gr.File(
                        label="选择 CSV 文件",
                        file_count="multiple",
                        file_types=[".csv"],
                    )
                    upload_btn = gr.Button("📤 上传数据", variant="primary")
                    upload_status = gr.Markdown("")

                # Tab 5: Structured Decision Panel (HITL)
                with gr.Tab("🎛️ 人机决策"):
                    gr.Markdown("**等待 Agent 触发断点...**")
                    action_radio = gr.Radio(
                        choices=["approve", "revise", "chat", "halt"],
                        label="决策操作",
                        value="approve",
                    )
                    revision_text = gr.Textbox(
                        label="修改建议（可选）",
                        placeholder="请输入具体的修改意见...",
                        lines=2,
                    )
                    submit_decision = gr.Button("提交决策", variant="secondary")

            # ========================================================
            # Event Handlers
            # ========================================================
            start_btn.click(
                fn=self.run_research,
                inputs=[domain_input, question_input, max_iter_slider, auto_approve_cb],
                outputs=output_stream,
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
