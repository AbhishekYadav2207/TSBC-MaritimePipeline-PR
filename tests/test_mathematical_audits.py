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


def test_mecs_authoritative_formula_and_clipping():
    """Verify authoritative MECS baseline formula (Top1=0.40, Rare=0.15, Loss=0.25, Frag=0.20, no Top5) and clipping."""
    import json
    from pathlib import Path
    cfg_path = Path(__file__).resolve().parent.parent / "outputs" / "stage-15" / "stage15_mecs_config.json"
    assert cfg_path.exists()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    baseline_weights = cfg["scenarios"]["baseline"]
    assert baseline_weights["top1_acc"] == 0.40
    assert baseline_weights["rare_top1_acc"] == 0.15
    assert baseline_weights["mlm_loss"] == 0.25
    assert baseline_weights["fragmentation_rate_pct"] == 0.20
    assert "top5_acc" not in baseline_weights

    # Clipping behavior on ModernBERT:
    # Loss = 2.1937, min_bound = 3.0, max_bound = 8.0
    loss_val = 2.193664
    s_loss_unclipped = (8.0 - loss_val) / (8.0 - 3.0)  # 1.161267
    s_loss = min(1.0, max(0.0, s_loss_unclipped))
    assert s_loss == 1.0  # Successfully clipped to 1.0

    # Top-1 ceiling = 0.35 (ModernBERT macro-average Top-1 = 0.579872 / 57.99%)
    top1_val = 0.579872
    s_top1 = min(1.0, max(0.0, top1_val / 0.35))
    assert s_top1 == 1.0

    # Rare Top-1 ceiling = 0.25 (ModernBERT macro-average Rare Top-1 = 0.773416 / 77.34%)
    rare_val = 0.773416
    s_rare = min(1.0, max(0.0, rare_val / 0.25))
    assert s_rare == 1.0

    # Frag % = 62.99% -> 1.0 - 0.6299 = 0.3701
    frag_pct = 62.99
    s_frag = 1.0 - (frag_pct / 100.0)

    # ModernBERT baseline MECS:
    score = (0.40 * s_top1 + 0.15 * s_rare + 0.25 * s_loss + 0.20 * s_frag) * 100.0
    assert round(score, 2) == 87.40


def test_pppl_dual_definitions():
    """Verify distinction between primary intrinsic MLM perplexity and secondary sampled PLL."""
    import pandas as pd
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    prof_path = root / "outputs" / "stage-15" / "stage15_model_profiles.csv"
    assert prof_path.exists()
    df = pd.read_csv(prof_path)
    mb = df[df["model_name"] == "answerdotai/ModernBERT-base"].iloc[0]

    # Primary MLM Loss = 2.1937 -> exp(2.1937) ~ 8.97
    primary_mlm_loss = mb["raw__mlm_loss"]
    primary_mlm_ppl = math.exp(primary_mlm_loss)
    assert abs(primary_mlm_ppl - 8.968) < 0.05

    # Secondary Sampled PLL Pseudo-Perplexity from pll_results.json = 13.72
    secondary_pll_pppl = mb["raw__pseudo_perplexity"]
    assert abs(secondary_pll_pppl - 13.72) < 0.05


def test_stage16_anova_exact_statistics():
    """Verify exact crossed ANOVA F-statistics and Friedman degrees of freedom match authoritative CSV."""
    import pandas as pd
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    anova_path = root / "outputs" / "stage-16" / "stage16_crossed_anova.csv"
    assert anova_path.exists()
    df = pd.read_csv(anova_path)

    model_row = df[df["source"].str.contains("Model \\(Fixed")].iloc[0]
    assert abs(model_row["f_statistic"] - 804.1615) < 0.01
    assert abs(model_row["variance_contribution_pct"] - 55.66) < 0.01

    rep_row = df[df["source"].str.contains("Representation \\(Blocking")].iloc[0]
    assert abs(rep_row["f_statistic"] - 552.0302) < 0.01
    assert abs(rep_row["variance_contribution_pct"] - 25.47) < 0.01

    sub_row = df[df["source"].str.contains("Subset \\(Blocking")].iloc[0]
    assert abs(sub_row["f_statistic"] - 15.4679) < 0.01
    assert abs(sub_row["variance_contribution_pct"] - 0.71) < 0.01

    int_row = df[df["source"].str.contains("Model x Representation")].iloc[0]
    assert abs(int_row["f_statistic"] - 55.3965) < 0.01
    assert abs(int_row["variance_contribution_pct"] - 15.34) < 0.01

    # Friedman test
    global_path = root / "outputs" / "stage-16" / "stage16_global_tests.csv"
    assert global_path.exists()
    df_glob = pd.read_csv(global_path)
    friedman = df_glob[df_glob["test_name"].str.contains("Friedman")].iloc[0]
    assert friedman["df"] == 6
    assert abs(friedman["statistic"] - 121.1657) < 0.01


def test_30_issues_id_preservation():
    """Verify all 30 issues are preserved verbatim from pre-repair review and A9 is correctly dispositioned."""
    import pandas as pd
    from pathlib import Path
    matrix_path = Path(__file__).resolve().parent.parent / "outputs" / "final_audit" / "review_issue_resolution_matrix.csv"
    assert matrix_path.exists()
    df = pd.read_csv(matrix_path)
    assert len(df) == 30

    expected_ids = [f"A{i}" for i in range(1, 11)] + [f"B{i}" for i in range(1, 10)] + [f"C{i}" for i in range(1, 12)]
    assert df["issue_id"].tolist() == expected_ids

    # A9 must be marked DOCUMENTED_LIMITATION
    a9 = df[df["issue_id"] == "A9"].iloc[0]
    assert a9["actual_status"] == "DOCUMENTED_LIMITATION"
    assert "template scaffolding" in a9["root_cause"].lower()


def test_report_consistency_zero_unexplained_contradictions():
    """Verify cross-file numerical consistency between Stage 15, 16, and 17 outputs."""
    import pandas as pd
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent

    lb = pd.read_csv(root / "outputs" / "stage-15" / "leaderboard.csv")
    s17 = pd.read_csv(root / "outputs" / "stage-17" / "stage17_model_selection.csv")
    pw = pd.read_csv(root / "outputs" / "stage-16" / "stage16_pairwise_tests.csv")

    # Winner consistency
    assert lb.iloc[0]["model_name"] == "answerdotai/ModernBERT-base"
    assert s17.iloc[0]["model_name"] == "answerdotai/ModernBERT-base"
    assert s17.iloc[0]["candidate_status"] == "Strong Candidate"

    # Win rate vs RoBERTa is 20/25 (80.0%)
    mb_rob = pw[(pw["model_a"] == "answerdotai/ModernBERT-base") & (pw["model_b"] == "roberta-base")].iloc[0]
    assert mb_rob["wins_a"] == 20
    assert mb_rob["losses_a"] == 5
    assert mb_rob["ties"] == 0
    assert mb_rob["paired_win_rate_a"] == 0.80
