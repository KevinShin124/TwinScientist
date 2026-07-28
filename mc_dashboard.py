"""
Monte Carlo RL 学习可视化与诊断工具

Usage:
    # 显示学习统计摘要
    python mc_dashboard.py

    # 显示 Top-N Q-values
    python mc_dashboard.py --top 10

    # 显示最近 episode 历史
    python mc_dashboard.py --episodes 20

    # 对给定状态查询策略推荐
    python mc_dashboard.py --query "iteration=3, evidence=0.5, convergence=0.7"

    # 重置所有学习数据
    python mc_dashboard.py --reset
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.mc_learning import (
    mc_policy,
    extract_state_key,
    VALID_ACTIONS,
    BINS,
)


def print_stats():
    """打印学习统计摘要"""
    stats = mc_policy.get_learning_stats()

    print("=" * 60)
    print("  [MC] 蒙特卡洛强化学习 — 学习统计摘要")
    print("=" * 60)

    if stats.get("total_episodes", 0) == 0:
        print("\n  [!] 尚无学习数据。请运行至少一次研究会话以积累经验。")
        print(f"  DB 路径: {mc_policy._db_path}")
        return

    print(f"\n  [Stats] 基本统计:")
    print(f"     总 Episode 数:    {stats['total_episodes']}")
    print(f"     平均回报:         {stats['avg_return']:.3f}")
    print(f"     平均步数:         {stats['avg_steps']:.1f}")
    print(f"     Q-表大小:         {stats['q_table_size']} 条记录")
    print(f"     总访问次数:       {stats['total_visits']}")

    print(f"\n  [Params] 超参数:")
    print(f"     折扣因子 γ:       {stats['gamma']}")
    print(f"     探索率 ε:         {stats['epsilon']}")
    print(f"     学习率 α:         {stats['alpha']}")


def print_top_qvalues(n: int = 10):
    """打印 Q-value 最高的 Top-N 动作"""
    top = mc_policy.get_top_actions(n)

    print("=" * 60)
    print(f"  [TOP] Top-{n} Q-values (最高价值状态-动作对)")
    print("=" * 60)

    if not top:
        print("\n  [!] 无数据。需要更多 episode 来学习。")
        return

    print(f"\n  {'#':>3}  {'Q值':>8}  {'访问':>5}  {'状态':>40}  {'动作'}")
    print(f"  {'─'*3}  {'─'*8}  {'─'*5}  {'─'*40}  {'─'*25}")

    for i, entry in enumerate(top, 1):
        state_short = entry["state_key"][:38] + ".." if len(entry["state_key"]) > 40 else entry["state_key"]
        print(f"  {i:>3}  {entry['q_value']:>8.4f}  {entry['visits']:>5}  {state_short:>40}  {entry['action']}")


def print_episode_history(limit: int = 20):
    """打印最近的 episode 历史"""
    episodes = mc_policy.get_episode_history(limit)

    print("=" * 60)
    print(f"  [History] 最近 {limit} 个 Episode 历史")
    print("=" * 60)

    if not episodes:
        print("\n  [!] 无 episode 数据。")
        return

    print(f"\n  {'ID':>5}  {'领域':>15}  {'时长':>8}  {'步数':>5}  {'总回报':>8}  {'终端奖励':>8}  {'状态'}")
    print(f"  {'─'*5}  {'─'*15}  {'─'*8}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*10}")

    for ep in episodes:
        domain_short = ep["domain"][:13] + ".." if len(ep["domain"]) > 15 else ep["domain"]
        duration = f"{ep['duration_sec']:.0f}s"
        print(
            f"  {ep['episode_id']:>5}  {domain_short:>15}  {duration:>8}  "
            f"{ep['num_steps']:>5}  {ep['total_return']:>8.3f}  "
            f"{ep['terminal_reward']:>8.3f}  {ep['status']}"
        )

    # Simple ASCII trend chart
    returns = [ep["total_return"] for ep in reversed(episodes)]
    if len(returns) >= 2:
        print(f"\n  📊 回报趋势 (最近 {len(returns)} episodes):")
        max_r = max(returns) if returns else 1
        min_r = min(returns) if returns else 0
        rng = max_r - min_r if max_r != min_r else 1

        for i, r in enumerate(returns):
            bar_len = int((r - min_r) / rng * 30) if rng > 0 else 15
            bar = "█" * max(1, bar_len)
            print(f"     Ep{i+1:>3}: {'▏' if bar_len == 0 else ''}{bar} {r:.3f}")


def query_state(state_desc: str):
    """查询给定状态下的策略推荐"""
    print("=" * 60)
    print("  [Query] 状态策略查询")
    print("=" * 60)

    # Parse simple key=value format
    state = {"_max_iterations_": 200}
    for part in state_desc.split(","):
        part = part.strip()
        if "=" in part:
            key, val = part.split("=", 1)
            key = key.strip()
            val = val.strip()
            # Map friendly names to state fields
            mapping = {
                "iteration": "iteration",
                "evidence": "avg_evidence",
                "convergence": "convergence_score",
                "review": "latest_review",
                "hyps": "num_hyps",
                "failures": "consecutive_failures",
                "prev_action": "current_action",
            }
            state_key = mapping.get(key, key)
            try:
                state[state_key] = float(val)
            except ValueError:
                state[state_key] = val

    # Build mock state for querying
    mock_state = {
        "iteration": int(state.get("iteration", 0)),
        "_max_iterations_": 200,
        "evidence_chains": [{"strength": state.get("avg_evidence", 0.5)}] if state.get("avg_evidence") else [],
        "convergence_score": state.get("convergence_score", 0.0),
        "review_records": [{"total_score": int(state.get("latest_review", 0))}] if state.get("latest_review") else [],
        "hypothesis_tree": [{"status": "proposed"}] * int(state.get("num_hyps", 3)),
        "consecutive_failures": int(state.get("consecutive_failures", 0)),
        "current_action": state.get("current_action", "none"),
    }

    state_key = extract_state_key(mock_state)
    print(f"\n  输入状态: {state_desc}")
    print(f"  离散化键: `{state_key}`")

    rec = mc_policy.recommend(mock_state)
    print(f"\n  推荐结果:")
    print(f"     方法:     {rec.get('method', 'N/A')}")
    print(f"     推荐动作: {rec.get('recommended_action', 'N/A')}")
    print(f"     最佳动作: {rec.get('best_action', 'N/A')}")
    print(f"     最佳Q值:  {rec.get('best_q_value', 0):.4f}")
    print(f"     置信度:   {rec.get('confidence', 0):.0%}")
    print(f"     探索中:   {rec.get('is_exploring', False)}")

    if rec.get("q_values"):
        print(f"\n  Q-values:")
        for action, q in sorted(rec["q_values"].items(), key=lambda x: x[1], reverse=True):
            marker = " ← 推荐" if action == rec.get("recommended_action") else ""
            print(f"     {action:<30} {q:.4f}{marker}")


def show_state_space():
    """显示状态空间离散化方案"""
    print("=" * 60)
    print("  [Map] 状态空间离散化方案")
    print("=" * 60)

    total_states = 1
    for name, bins in BINS.items():
        n_buckets = len(bins)
        total_states *= n_buckets
        print(f"\n  {name}: {n_buckets} 个桶")
        print(f"     边界: {bins}")

    print(f"\n  总状态空间 ≈ {total_states:,} 个离散状态")
    print(f"  动作空间: {len(VALID_ACTIONS)} 个动作")
    print(f"  总 (s,a) 对 ≈ {total_states * len(VALID_ACTIONS):,}")


def main():
    parser = argparse.ArgumentParser(description="蒙特卡洛 RL 学习可视化工具")
    parser.add_argument("--top", type=int, default=0, help="显示 Top-N Q-values")
    parser.add_argument("--episodes", type=int, default=0, help="显示最近 N 个 episode")
    parser.add_argument("--query", type=str, help="查询状态策略 (格式: 'iteration=3, evidence=0.5')")
    parser.add_argument("--state-space", action="store_true", help="显示状态空间离散化方案")
    parser.add_argument("--reset", action="store_true", help="重置所有学习数据")
    parser.add_argument("--all", action="store_true", help="显示所有信息")

    args = parser.parse_args()

    if args.reset:
        confirm = input("[!] 确认重置所有学习数据？(y/N): ")
        if confirm.lower() == "y":
            mc_policy.reset()
            print("[OK] 已重置所有学习数据。")
        else:
            print("取消。")
        return

    if args.state_space or args.all:
        show_state_space()
        print()

    # Default: show stats
    print_stats()

    if args.top > 0 or args.all:
        print()
        print_top_qvalues(args.top or 10)

    if args.episodes > 0 or args.all:
        print()
        print_episode_history(args.episodes or 10)

    if args.query:
        print()
        query_state(args.query)


if __name__ == "__main__":
    main()
