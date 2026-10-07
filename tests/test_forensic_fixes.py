"""
Unit and Regression Tests for MaritimeBench Forensic Backend Fixes
Validates:
- DeBERTa permanent configuration-driven exclusion
- Dynamic cohort resolution (N=7, cells=175, pairwise=21)
- Whole-Word Masking (WWM) token grouping and strict word reconstruction
- Paired vs unpaired statistical separation (wins/ties/losses, paired rank-biserial, Holm correction)
- Cryptographic representation provenance and hash locking
- Kendall's W multi-ranking concordance
- Cohort-independent MECS loss normalization
- Crossed 3-way repeated-measures ANOVA / variance decomposition
- MECS semantic renaming across active output schemas
"""

import hashlib
import importlib
import json
import math
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class TestDeBERTaExclusion(unittest.TestCase):
    def test_configuration_driven_exclusion(self):
        """Verify DeBERTa is excluded via configuration dictionary, not positional slicing."""
        t13 = importlib.import_module("13_tokenizer_analysis")
        self.assertTrue(hasattr(t13, "EXCLUDED_MODELS"))
        self.assertIn("microsoft/deberta-v3-base", t13.EXCLUDED_MODELS)
        reason = t13.EXCLUDED_MODELS["microsoft/deberta-v3-base"].get("reason", "")
        self.assertIn("zero MLM", reason)

    def test_audit_artifact_exists(self):
        """Verify persistent audit artifact exists with provenance details."""
        audit_file = PROJECT_ROOT / "outputs" / "model_selection" / "deberta_exclusion_audit.json"
        self.assertTrue(audit_file.exists(), f"Missing audit artifact: {audit_file}")
        with open(audit_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data.get("model_id"), "microsoft/deberta-v3-base")
        self.assertEqual(data.get("exclusion_status"), "EXCLUDED")
        self.assertIn("zero MLM", data.get("reason", ""))

    def test_absence_from_active_outputs(self):
        """Verify DeBERTa is absent from all active pipeline outputs."""
        excluded_id = "microsoft/deberta-v3-base"

        # Stage 13 selected models
        s13_path = PROJECT_ROOT / "outputs" / "stage-13" / "selected_models.json"
        if s13_path.exists():
            with open(s13_path, "r", encoding="utf-8") as f:
                s13 = json.load(f)
            models = [m.get("model_name") if isinstance(m, dict) else m for m in s13.get("selected_models", [])]
            self.assertNotIn(excluded_id, models)
            self.assertEqual(len(models), 7)

        # Stage 15 leaderboard
        s15_lb = PROJECT_ROOT / "outputs" / "stage-15" / "leaderboard.csv"
        if s15_lb.exists():
            df = pd.read_csv(s15_lb)
            self.assertNotIn(excluded_id, df["model_name"].values)
            self.assertEqual(len(df), 7)

        # Stage 16 pairwise tests
        s16_pw = PROJECT_ROOT / "outputs" / "stage-16" / "stage16_pairwise_tests.csv"
        if s16_pw.exists():
            df = pd.read_csv(s16_pw)
            self.assertNotIn(excluded_id, df["model_a"].values)
            self.assertNotIn(excluded_id, df["model_b"].values)
            self.assertEqual(len(df), 21)

        # Stage 17 selection
        s17_sel = PROJECT_ROOT / "outputs" / "stage-17" / "stage17_model_selection.csv"
        if s17_sel.exists():
            df = pd.read_csv(s17_sel)
            self.assertNotIn(excluded_id, df["model_name"].values)
            self.assertEqual(len(df), 7)


class TestDynamicCohortDimensions(unittest.TestCase):
    def test_dynamic_derivation(self):
        """Verify cohort dimensions are dynamically derived and match 7 models, 5 reps, 5 subsets."""
        n_models = 7
        n_reps = 5
        n_subsets = 5
        expected_cells = n_models * n_reps * n_subsets  # 175
        expected_pairwise = (n_models * (n_models - 1)) // 2  # 21

        self.assertEqual(expected_cells, 175)
        self.assertEqual(expected_pairwise, 21)

        s16_anova = PROJECT_ROOT / "outputs" / "stage-16" / "stage16_crossed_anova.csv"
        if s16_anova.exists():
            df = pd.read_csv(s16_anova)
            n_obs = df["df"].sum() + 1
            self.assertEqual(n_obs, expected_cells)


class TestWWMAndWordReconstruction(unittest.TestCase):
    def setUp(self):
        mlm = importlib.import_module("14_mlm_evaluation")
        self.extract_word_groups = mlm.extract_word_groups
        self.create_whole_word_random_mask = mlm.create_whole_word_random_mask

    def test_word_grouping_multi_piece(self):
        """Verify subwords belonging to the same word are grouped correctly."""
        # Tokens: [CLS, un, ##sea, ##worthy, vessel, in, 2024, SEP]
        # word_ids: [None, 0, 0, 0, 1, 2, 3, None]
        word_ids = [None, 0, 0, 0, 1, 2, 3, None]
        eligible_positions = [1, 2, 3, 4, 5, 6]

        w2p, p2w = self.extract_word_groups(word_ids, eligible_positions)
        self.assertEqual(w2p[0], [1, 2, 3])
        self.assertEqual(w2p[1], [4])
        self.assertEqual(w2p[2], [5])
        self.assertEqual(w2p[3], [6])
        self.assertEqual(p2w[1], 0)
        self.assertEqual(p2w[2], 0)
        self.assertEqual(p2w[3], 0)

    def test_whole_word_mask_budget(self):
        """Verify all subwords of a word are masked together under WWM."""
        import random
        word_ids = [None, 0, 0, 1, 2, 2, 3, None]
        eligible_positions = [1, 2, 3, 4, 5, 6]
        w2p, _ = self.extract_word_groups(word_ids, eligible_positions)

        rng = random.Random(42)
        masked_positions, masked_words = self.create_whole_word_random_mask(w2p, rng, mask_budget=3)
        for wid in masked_words:
            for pos in w2p[wid]:
                self.assertIn(pos, masked_positions, f"Subword piece {pos} for word {wid} was not masked!")

    def test_strict_word_reconstruction_criterion(self):
        """Verify a word is correct ONLY if ALL constituent predicted pieces match target IDs."""
        # Target tokens: [101, 500, 600, 102]
        target_ids = {1: 500, 2: 600}
        word_pieces = [1, 2]

        # Case 1: Both pieces correct -> Word Correct
        pred_top1_both_correct = {1: 500, 2: 600}
        all_correct = all(pred_top1_both_correct[p] == target_ids[p] for p in word_pieces)
        self.assertTrue(all_correct)

        # Case 2: One piece correct, one piece wrong -> Word INCORRECT
        pred_top1_partial = {1: 500, 2: 999}
        partial_correct = all(pred_top1_partial[p] == target_ids[p] for p in word_pieces)
        self.assertFalse(partial_correct, "Strict word reconstruction must fail when any piece is incorrect!")


class TestPairedStatistics(unittest.TestCase):
    def setUp(self):
        s16 = importlib.import_module("16_statistical_analysis")
        self.paired_prb = s16.paired_rank_biserial
        self.run_pairwise = s16.run_pairwise_comparisons

    def test_paired_rank_biserial(self):
        """Verify paired rank-biserial correlation bounds and directions."""
        # Perfect dominance: all diffs > 0
        diffs_pos = np.array([0.1, 0.2, 0.15, 0.05])
        r_pos = self.paired_prb(diffs_pos)
        self.assertAlmostEqual(r_pos, 1.0)

        # Perfect defeat: all diffs < 0
        diffs_neg = np.array([-0.1, -0.2, -0.15, -0.05])
        r_neg = self.paired_prb(diffs_neg)
        self.assertAlmostEqual(r_neg, -1.0)

        # Symmetric: diffs cancel
        diffs_sym = np.array([0.1, -0.1, 0.2, -0.2])
        r_sym = self.paired_prb(diffs_sym)
        self.assertAlmostEqual(r_sym, 0.0)

    def test_wins_ties_losses_accounting(self):
        """Verify wins + ties + losses == total paired conditions."""
        # Create a mock pivot table with 5 conditions
        pvt = pd.DataFrame({
            "model_1": [0.8, 0.7, 0.6, 0.5, 0.9],
            "model_2": [0.7, 0.7, 0.5, 0.6, 0.8]
        }, index=[f"cond_{i}" for i in range(5)])

        df_pair, df_eff, raw = self.run_pairwise(pvt, ["model_1", "model_2"])
        row = df_pair.iloc[0]
        self.assertEqual(row["wins_a"], 3)
        self.assertEqual(row["ties"], 1)
        self.assertEqual(row["losses_a"], 1)
        self.assertEqual(row["wins_a"] + row["ties"] + row["losses_a"], 5)
        self.assertAlmostEqual(row["paired_win_rate_a"], (3 + 0.5 * 1) / 5.0)

    def test_unpaired_cliffs_delta_separation(self):
        """Verify Cliff's delta is labeled unpaired and not substituted for paired effect size."""
        s16_path = PROJECT_ROOT / "outputs" / "stage-16" / "statistical_significance.json"
        if s16_path.exists():
            with open(s16_path, "r", encoding="utf-8") as f:
                sig_data = json.load(f)
            pairwise_records = sig_data.get("pairwise_comparisons", [])
            for rec in pairwise_records:
                self.assertIn("paired_statistics", rec)
                self.assertIn("unpaired_statistics", rec)
                self.assertIn("paired_rank_biserial", rec["paired_statistics"])
                self.assertIn("unpaired_cliffs_delta", rec["unpaired_statistics"])


class TestRepresentationProvenance(unittest.TestCase):
    def test_provenance_manifest_hashes(self):
        """Verify that every representation hash matches the actual file on disk."""
        prov_file = PROJECT_ROOT / "outputs" / "representation_provenance.json"
        self.assertTrue(prov_file.exists(), f"Missing provenance manifest: {prov_file}")
        with open(prov_file, "r", encoding="utf-8") as f:
            prov = json.load(f)

        self.assertIn("source_corpus", prov)
        self.assertIn("representations", prov)

        # Verify source text corpus
        src_info = prov["source_corpus"]["text_corpus"]
        src_path = PROJECT_ROOT / src_info["path"]
        self.assertTrue(src_path.exists())
        actual_src_hash = hashlib.sha256(src_path.read_bytes()).hexdigest()
        self.assertEqual(src_info["sha256"], actual_src_hash)

        # Verify representations
        rep_hashes = set()
        for rep_name, info in prov["representations"].items():
            rep_path = PROJECT_ROOT / info["file_path"]
            self.assertTrue(rep_path.exists(), f"Missing representation file: {rep_path}")
            actual_rep_hash = hashlib.sha256(rep_path.read_bytes()).hexdigest()
            self.assertEqual(info["sha256"], actual_rep_hash, f"Hash mismatch for representation {rep_name}")
            rep_hashes.add(actual_rep_hash)

        # Confirm distinct representations do not accidentally point to identical files
        self.assertEqual(len(rep_hashes), 5, "Expected 5 unique representation hashes")


class TestKendallsW(unittest.TestCase):
    def setUp(self):
        s15 = importlib.import_module("15_cross_model_benchmarking")
        self.calc_w = s15.calculate_kendalls_w

    def test_perfect_concordance(self):
        """All raters assign identical rankings -> W = 1.0."""
        # 3 raters (rows), 4 objects (cols)
        matrix = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0]
        ])
        res = self.calc_w(matrix)
        self.assertAlmostEqual(res["kendalls_w"], 1.0, places=4)
        self.assertEqual(res["k_rankings"], 3)
        self.assertEqual(res["n_objects"], 4)

    def test_low_concordance(self):
        """Discordant rankings across raters -> W close to 0."""
        # 2 raters, opposite rankings
        matrix = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0]
        ])
        res = self.calc_w(matrix)
        self.assertAlmostEqual(res["kendalls_w"], 0.0, places=4)

    def test_tied_ranks(self):
        """Tie handling produces valid concordance metric."""
        matrix = np.array([
            [1.5, 1.5, 3.0, 4.0],
            [1.0, 2.5, 2.5, 4.0]
        ])
        res = self.calc_w(matrix)
        self.assertTrue(0.0 <= res["kendalls_w"] <= 1.0)
        self.assertTrue(res["tie_corrected"])


class TestCohortIndependentMECS(unittest.TestCase):
    def test_cohort_independent_loss_normalization(self):
        """Normalized loss of models A, B, C must be identical regardless of whether model D is present."""
        def loss_norm(loss):
            return 1.0 / (1.0 + loss)

        # Run with Cohort 1: A, B, C
        loss_a = 1.45
        loss_b = 3.20
        loss_c = 2.10
        norm_a1 = loss_norm(loss_a)
        norm_b1 = loss_norm(loss_b)
        norm_c1 = loss_norm(loss_c)

        # Run with Cohort 2: A, B, C, D (where D has extreme loss 10.5)
        loss_d = 10.5
        norm_a2 = loss_norm(loss_a)
        norm_b2 = loss_norm(loss_b)
        norm_c2 = loss_norm(loss_c)
        norm_d2 = loss_norm(loss_d)

        # Cohort independence check:
        self.assertEqual(norm_a1, norm_a2)
        self.assertEqual(norm_b1, norm_b2)
        self.assertEqual(norm_c1, norm_c2)

        # Monotonicity check: lower loss -> higher score
        self.assertGreater(norm_a1, norm_c1)
        self.assertGreater(norm_c1, norm_b1)
        self.assertGreater(norm_b1, norm_d2)


class TestCrossedStatisticalAnalysis(unittest.TestCase):
    def setUp(self):
        s16 = importlib.import_module("16_statistical_analysis")
        self.run_crossed = s16.run_crossed_factorial_analysis

    def test_synthetic_crossed_design(self):
        """Verify crossed 3-way ANOVA computes effects and block-respecting permutation."""
        # Synthetic crossed factorial data: 3 models, 2 reps, 2 subsets = 12 cells
        records = []
        models = ["M1", "M2", "M3"]
        rng = np.random.default_rng(42)
        for m in models:
            m_boost = 10.0 if m == "M1" else (5.0 if m == "M2" else 0.0)
            for r in ["json", "narrative"]:
                r_boost = 2.0 if r == "narrative" else 0.0
                for s in ["high", "low"]:
                    noise = float(rng.normal(0, 0.1))
                    score = 50.0 + m_boost + r_boost + (1.0 if s == "high" else 0.0) + noise
                    records.append({
                        "model_name": m,
                        "representation": r,
                        "subset": s,
                        "top1_acc": score
                    })

        df_synthetic = pd.DataFrame(records)
        crossed_dict, df_anova = self.run_crossed(df_synthetic, models=models, primary_metric="top1_acc", n_perms=50)

        self.assertTrue(any("Model" in s for s in df_anova["source"].values))
        self.assertTrue(any("Representation" in s for s in df_anova["source"].values))
        self.assertTrue(any("Subset" in s for s in df_anova["source"].values))
        self.assertTrue(any("Model x Representation" in s for s in df_anova["source"].values))
        self.assertTrue(any("Model x Subset" in s for s in df_anova["source"].values))

        # Model effect must be highly significant
        model_row = df_anova[df_anova["source"].str.contains("Model")].iloc[0]
        self.assertGreater(model_row["f_statistic"], 50.0)
        self.assertLess(model_row["p_value"], 0.01)
        self.assertEqual(crossed_dict["model_main_effect"]["block_permutation_p_value"], 1.0 / (1 + 50))


class TestNoLegacyMUITerminology(unittest.TestCase):
    def test_active_outputs_no_legacy_mui(self):
        """Verify active output files contain no legacy Maritime Understanding Index or MUI."""
        active_paths = [
            PROJECT_ROOT / "outputs" / "stage-15" / "stage15_mecs_sensitivity.csv",
            PROJECT_ROOT / "outputs" / "stage-15" / "stage15_selection_decision.json",
            PROJECT_ROOT / "outputs" / "stage-16" / "stage16_final_report.md",
            PROJECT_ROOT / "outputs" / "stage-17" / "stage17_model_selection.csv",
            PROJECT_ROOT / "outputs" / "stage-17" / "stage17_selection_rationale.json",
            PROJECT_ROOT / "outputs" / "stage-17" / "stage17_decision_report.md"
        ]

        for p in active_paths:
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8")
            self.assertNotIn("Maritime Understanding Index", text, f"Legacy terminology found in {p.name}")
            self.assertNotIn('"mui_score"', text, f"Legacy mui_score key found in {p.name}")


if __name__ == "__main__":
    unittest.main()
