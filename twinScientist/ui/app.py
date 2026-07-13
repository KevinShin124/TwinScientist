"""
twinScientist Gradio Frontend

Competition bonus feature: interactive web UI with structured decision panels,
real-time progress tracking, and report preview.
"""

import gradio as gr
import json
from pathlib import Path
from typing import Any


class TwinScientistUI:
    """Gradio-based interactive frontend for twinScientist"""

    def __init__(self, agent_app=None):
        self.agent_app = agent_app
        self.sessions = {}  # session_id -> state

    def _get_or_create_session(self, session_id: str) -> dict:
        if session_id not in self.sessions:
            self.sessions[session_id] = {"iteration": 0, "status": "idle"}
        return self.sessions[session_id]

    async def run_research(
        self,
        domain: str,
        research_question: str,
        max_iterations: int,
        auto_approve: bool,
    ):
        """启动研究循环并返回流式输出"""
        initial_state = {
            "query": research_question,
            "domain": domain or "环境—人体关联",
            "_max_iterations_": max_iterations,
            "auto_confirm": auto_approve,
        }

        if self.agent_app:
            async for event in self.agent_app.astream(initial_state, stream_mode="updates"):
                # Convert internal events to human-readable log lines
                if isinstance(event, dict):
                    yield f"[{list(event.keys())[0]}] {json.dumps(event[list(event.keys())[0]], ensure_ascii=False, indent=2)}\n"
                else:
                    yield f"{event}\n"
        else:
            yield "[UI] Agent app not configured. Connect your LLM first.\n"

    def build_ui(self) -> gr.Blocks:
        """构建完整的 Gradio 界面"""

        with gr.Blocks(title="twinScientist — AI Scientist") as demo:
            gr.Markdown("# 🔬 twinScientist")
            gr.Markdown("### 面向环境—人体关联研究的自主科研与实验迭代智能体")

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
                        minimum=3, maximum=30, value=10, step=1,
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
                        lines=20,
                        interactive=False,
                    )
                    report_preview = gr.Markdown(
                        "## 📄 研究报告预览\n\n"
                        "运行完成后将在此显示生成的《科学假设与研究计划》。"
                    )

            # Structured Decision Panel (visible during checkpoints)
            with gr.Accordion("🎛️ 结构化决策面板", open=False) as decision_panel:
                gr.Markdown("**等待 Agent 触发断点...**")
                action_radio = gr.Radio(
                    choices=["approve", "revise", "redirect", "halt"],
                    label="决策操作",
                    value="approve",
                )
                revision_text = gr.Textbox(
                    label="修改建议（可选）",
                    placeholder="请输入具体的修改意见...",
                    lines=2,
                )
                submit_decision = gr.Button("提交决策", variant="secondary")

            start_btn.click(
                fn=self.run_research,
                inputs=[domain_input, question_input, max_iter_slider, auto_approve_cb],
                outputs=output_stream,
            )

        return demo


def create_demo(agent_app=None):
    """工厂函数：创建 Gradio Demo"""
    ui = TwinScientistUI(agent_app)
    return ui.build_ui()


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
