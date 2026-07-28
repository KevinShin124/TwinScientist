"""TwinScientist Reliability Benchmark Suite

Four-layer proof system for technical review:
  Layer 1: Causal Discovery Accuracy (SHD, Edge F1, Direction Accuracy)
  Layer 2: Effect Size Recovery (MAE against simulator ground truth)
  Layer 3: Baseline Comparison (vs. correlation, traditional causal, LLM)
  Layer 4: Robustness (noise, small sample, cross-domain stability)

Usage:
    python -m benchmark.runner          # Run all scenarios
    python -m benchmark.runner --quick  # 3 scenarios, fast proof-of-concept
"""
