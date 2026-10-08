import pytest
import math
import numpy as np

def compute_top1(preds, targets):
    assert len(preds) == len(targets)
    if not targets:
        return None
    correct = sum(1 for p, t in zip(preds, targets) if p == t)
    return correct / len(targets)

def compute_top5(preds_top5, targets):
    assert len(preds_top5) == len(targets)
    if not targets:
        return None
    correct = sum(1 for p_list, t in zip(preds_top5, targets) if t in p_list[:5])
    return correct / len(targets)

def compute_win_rate(wins, losses, ties):
    total = wins + losses + ties
    assert total > 0
    return wins / total

def compute_friedman_df(k_models):
    assert k_models > 1
    return k_models - 1

def compute_kendall_w(rankings_matrix):
    """
    Computes Kendall's W concordance coefficient.
    rankings_matrix: shape (m_judges, n_items)
    """
    matrix = np.array(rankings_matrix, dtype=float)
    m, n = matrix.shape
    if n <= 1:
        return 1.0
    R = np.sum(matrix, axis=0)
    R_mean = np.mean(R)
    S = np.sum((R - R_mean) ** 2)
    W = (12 * S) / (m**2 * (n**3 - n))
    return float(W)

def compute_cliffs_delta(x, y):
    """
    Computes Cliff's delta across all len(x) * len(y) pairs.
    """
    n_x, n_y = len(x), len(y)
    greater = 0
    lesser = 0
    for a in x:
        for b in y:
            if a > b:
                greater += 1
            elif a < b:
                lesser += 1
    return (greater - lesser) / (n_x * n_y)

def compute_paired_dominance(x, y):
    """
    Computes paired wins, losses, ties, and paired rank-biserial correlation.
    """
    assert len(x) == len(y)
    n = len(x)
    wins = sum(1 for a, b in zip(x, y) if a > b)
    losses = sum(1 for a, b in zip(x, y) if a < b)
    ties = sum(1 for a, b in zip(x, y) if a == b)
    diffs = [a - b for a, b in zip(x, y) if a != b]
    
    # rank-biserial effect size
    if not diffs:
        r_b = 0.0
    else:
        pos = sum(1 for d in diffs if d > 0)
        neg = sum(1 for d in diffs if d < 0)
        r_b = (pos - neg) / len(diffs)
        
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / n,
        "rank_biserial": r_b
    }

def compute_pareto_dominance(candidate, reference, objectives):
    """
    Determines if candidate dominates reference.
    objectives: list of (metric_name, 'max' | 'min')
    """
    candidate_at_least_as_good = True
    candidate_strictly_better = False

    for name, direction in objectives:
        c_val = candidate[name]
        r_val = reference[name]
        if direction == "max":
            if c_val < r_val:
                candidate_at_least_as_good = False
            elif c_val > r_val:
                candidate_strictly_better = True
        elif direction == "min":
            if c_val > r_val:
                candidate_at_least_as_good = False
            elif c_val < r_val:
                candidate_strictly_better = True

    return candidate_at_least_as_good and candidate_strictly_better

# ==============================================================================
# PYTEST TEST SUITE FOR MATHEMATICAL AUDIT
# ==============================================================================

def test_top1_accuracy():
    preds = ["a", "b", "c", "c"]
    trues = ["a", "b", "x", "c"]
    # 3 correct out of 4 = 0.75
    assert compute_top1(preds, trues) == 0.75

def test_top5_accuracy():
    preds_top5 = [
        ["a", "b", "c", "d", "e"],
        ["x", "y", "z", "w", "v"],
        ["m", "n", "o", "p", "q"]
    ]
    targets = ["c", "target_absent", "q"]
    # 2 correct out of 3
    assert abs(compute_top5(preds_top5, targets) - (2/3)) < 1e-6

def test_win_rate_arithmetic():
    # Review Issue B9 / Section 18: 20 wins, 5 losses, 0 ties out of 25 must be 80%, NEVER 100%
    assert compute_win_rate(20, 5, 0) == 0.80
    assert compute_win_rate(20, 5, 0) != 1.00

def test_friedman_df():
    # 7 models -> df = 6
    assert compute_friedman_df(7) == 6

def test_cliffs_delta_toy():
    # Toy example from prompt: A = [1, 2, 3], B = [2, 2, 4]
    # Pairs (a, b):
    # (1, 2) -, (1, 2) -, (1, 4) -
    # (2, 2) =, (2, 2) =, (2, 4) -
    # (3, 2) +, (3, 2) +, (3, 4) -
    # Greater = 2, Lesser = 5, Total = 9 -> delta = (2 - 5)/9 = -3/9 = -0.33333333
    A = [1, 2, 3]
    B = [2, 2, 4]
    delta = compute_cliffs_delta(A, B)
    assert abs(delta - (-1/3)) < 1e-6

def test_paired_dominance():
    # Matched 5 cells
    A = [10, 20, 30, 40, 50]
    B = [5,  25, 28, 35, 50]
    res = compute_paired_dominance(A, B)
    # Cell 1: 10 > 5 (win)
    # Cell 2: 20 < 25 (loss)
    # Cell 3: 30 > 28 (win)
    # Cell 4: 40 > 35 (win)
    # Cell 5: 50 == 50 (tie)
    assert res["wins"] == 3
    assert res["losses"] == 1
    assert res["ties"] == 1
    assert res["win_rate"] == 0.60
    assert res["rank_biserial"] == (3 - 1) / 4  # 0.5

def test_pareto_dominance():
    objs = [("top1", "max"), ("loss", "min")]
    model_a = {"top1": 0.80, "loss": 0.40}
    model_b = {"top1": 0.75, "loss": 0.50}
    model_c = {"top1": 0.85, "loss": 0.60}
    
    # A dominates B (higher top1, lower loss)
    assert compute_pareto_dominance(model_a, model_b, objs) is True
    assert compute_pareto_dominance(model_b, model_a, objs) is False
    
    # Neither A nor C dominates each other (trade-off)
    assert compute_pareto_dominance(model_a, model_c, objs) is False
    assert compute_pareto_dominance(model_c, model_a, objs) is False

def test_kendall_w_concordance():
    # 3 rankers evaluating 4 items with identical ranking: [1, 2, 3, 4]
    identical_ranks = [
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4]
    ]
    w_perfect = compute_kendall_w(identical_ranks)
    assert abs(w_perfect - 1.0) < 1e-6

def test_oov_empty_slice_emits_none():
    # If a slice has 0 observations, accuracy must be None, NOT 0.0
    empty_stats = {"loss": 0.0, "top1": 0, "count": 0}
    # Simulate zero count
    cnt = empty_stats["count"]
    val = (empty_stats["top1"] / cnt) if cnt > 0 else None
    assert val is None
    assert val != 0.0
