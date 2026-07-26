"""
Adaptive Iterations — Test-Time Compute Scaling (OpenAI-style).

Dynamically adjusts iteration budget based on query complexity,
matching OpenAI Deep Research's approach of allocating more compute
to harder problems.

Usage:
    from core.adaptive import compute_iteration_budget
    max_iter = compute_iteration_budget(query, domain)
"""

from __future__ import annotations


# Complexity signals and their weights
COMPLEXITY_SIGNALS = {
    # Multi-factor / interaction terms
    "interaction": [
        "交互", "interaction", "协同", "synerg", "中介", "mediation",
        "调节", "moderation", "复合", "combined effect",
    ],
    # Causal / mechanistic questions
    "causal": [
        "因果", "causal", "机制", "mechanism", "通路", "pathway",
        "方向", "direction", "影响", "effect of", "导致", "cause",
    ],
    # Temporal / dynamic
    "temporal": [
        "时间序列", "time series", "动态", "dynamic", "滞后", "lag",
        "延迟", "delay", "长期", "long-term", "短期", "short-term",
        "昼夜", "circadian", "节律", "rhythm",
    ],
    # Multi-variable / high-dimensional
    "multivariate": [
        "多变量", "multivariate", "多因素", "multi-factor", "多模态",
        "multi-modal", "多维", "high-dimensional", "综合", "comprehensive",
    ],
    # Individual differences / personalization
    "personalization": [
        "个体差异", "individual differ", "个性化", "personalized",
        "N-of-1", "n-of-1", "不同人群", "different population",
        "年龄", "age", "性别", "gender", "基础", "baseline",
    ],
    # Cross-domain / interdisciplinary
    "cross_domain": [
        "跨学科", "interdisciplinary", "交叉", "cross", "对比",
        "comparison", "不同环境", "different environment",
    ],
}


def compute_iteration_budget(query: str, domain: str = "") -> int:
    """
    Compute the optimal iteration budget based on query complexity.

    Simple descriptive questions: 2-3 iterations
    Moderate causal questions: 5-8 iterations
    Complex multi-factor mechanistic questions: 10-15 iterations

    Returns: int between 2 and 200
    """
    query_lower = query.lower()
    domain_lower = domain.lower() if domain else ""

    # Count complexity signals
    scores = {}
    for category, keywords in COMPLEXITY_SIGNALS.items():
        score = sum(1 for kw in keywords if kw.lower() in query_lower or kw.lower() in domain_lower)
        scores[category] = min(score, 3)  # Cap per category

    # Weighted complexity score
    weights = {
        "interaction": 2.0,     # Multi-factor interactions are hardest
        "causal": 1.5,          # Causal questions are moderately hard
        "temporal": 1.5,        # Time series analysis adds complexity
        "multivariate": 1.5,    # More variables = more hypotheses
        "personalization": 1.0, # Individual differences add nuance
        "cross_domain": 1.0,    # Cross-domain adds breadth
    }

    complexity = sum(scores[cat] * weights.get(cat, 1.0) for cat in scores)

    # Map complexity to iterations
    if complexity <= 2:
        budget = 3      # Simple question
    elif complexity <= 5:
        budget = 5      # Moderate
    elif complexity <= 8:
        budget = 8      # Moderately complex
    elif complexity <= 12:
        budget = 12     # Complex
    else:
        budget = 15     # Very complex

    # Clamp and add domain bonus
    domain_bonus = 2 if "环境" in domain_lower and "人体" in domain_lower else 0
    budget = min(budget + domain_bonus, 200)

    return max(budget, 2)


def explain_budget(query: str, budget: int) -> str:
    """Generate a human-readable explanation of the iteration budget."""
    query_lower = query.lower()
    signals_found = []
    for category, keywords in COMPLEXITY_SIGNALS.items():
        matched = [kw for kw in keywords if kw.lower() in query_lower]
        if matched:
            signals_found.append(f"{category}({', '.join(matched[:2])})")

    return (
        f"Query complexity: {budget} iterations allocated. "
        f"Signals detected: {', '.join(signals_found) if signals_found else 'basic'}."
    )