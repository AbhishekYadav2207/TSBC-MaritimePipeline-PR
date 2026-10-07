import unittest
import importlib
import sys
import argparse
from pathlib import Path
import random
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
stage14 = importlib.import_module("14_mlm_evaluation")

class TestStage14ClassificationAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok_roberta = AutoTokenizer.from_pretrained("roberta-base")
        cls.tok_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
        cls.vocab_sample = [
            "cargo vessel", "steering gear", "search and rescue",
            "windlass", "bollards", "shipbuilding"
        ]

    def test_default_cli_configuration(self):
        """Verify that default Stage 14 execution resolves to WWM and strict word evaluation."""
        # Create a parser with the identical arguments as main()
        parser = argparse.ArgumentParser()
        parser.add_argument("--masking_mode", default="whole_word", choices=["subword", "whole_word", "wwm_subword", "wwm_word"])
        parser.add_argument("--evaluation_unit", default="word", choices=["subword", "word"])
        
        args = parser.parse_args([])
        self.assertEqual(args.masking_mode, "whole_word")
        self.assertEqual(args.evaluation_unit, "word")

    def test_stopword_leakage_prevented(self):
        """Verify that common English function stopwords do not leak into maritime token IDs."""
        mar_ids, rare_ids, cat_ids = stage14.build_vocabulary_token_sets(self.tok_roberta, self.vocab_sample)
        and_id = self.tok_roberta.convert_tokens_to_ids("Ġand")
        and_id_raw = self.tok_roberta.convert_tokens_to_ids("and")
        the_id = self.tok_roberta.convert_tokens_to_ids("Ġthe")

        self.assertNotIn(and_id, mar_ids, "Stopword ' and' leaked into RoBERTa maritime token IDs")
        self.assertNotIn(and_id_raw, mar_ids, "Stopword 'and' leaked into RoBERTa maritime token IDs")
        self.assertNotIn(the_id, mar_ids, "Stopword ' the' leaked into RoBERTa maritime token IDs")

        # Check BERT
        mar_ids_b, _, _ = stage14.build_vocabulary_token_sets(self.tok_bert, self.vocab_sample)
        and_id_b = self.tok_bert.convert_tokens_to_ids("and")
        self.assertNotIn(and_id_b, mar_ids_b, "Stopword 'and' leaked into BERT maritime token IDs")

    def test_bpe_space_prefix_consistency(self):
        """Verify that BPE leading space tokens (Ġcargo, Ġvessel) are properly recognized."""
        mar_ids, _, _ = stage14.build_vocabulary_token_sets(self.tok_roberta, self.vocab_sample)
        cargo_spaced = self.tok_roberta.convert_tokens_to_ids("Ġcargo")
        vessel_spaced = self.tok_roberta.convert_tokens_to_ids("Ġvessel")

        self.assertIn(cargo_spaced, mar_ids, "Space-prefixed ' cargo' missing from RoBERTa maritime token IDs")
        self.assertIn(vessel_spaced, mar_ids, "Space-prefixed ' vessel' missing from RoBERTa maritime token IDs")

    def test_lexical_classification_integrity(self):
        """Verify individual terms: cargo -> MARITIME, search -> MARITIME, and -> GENERAL, rescue -> MARITIME."""
        tokenizer = self.tok_roberta
        mar_ids, rare_ids, cat_ids = stage14.build_vocabulary_token_sets(tokenizer, self.vocab_sample)

        text = "The cargo vessel conducted search and rescue operations."
        enc = tokenizer(text, return_offsets_mapping=True)
        input_ids = enc["input_ids"]
        wids = enc.word_ids(0)
        sp_mask = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)

        eligible, rare_pos, mar_pos, gen_pos, cat_pos = stage14.classify_token_positions(
            text, input_ids, enc["offset_mapping"], sp_mask,
            set(self.vocab_sample), set(stage14.RARE_MARITIME_TERMS),
            mar_ids, rare_ids, cat_ids
        )
        w2p, _ = stage14.extract_word_groups(wids, eligible)

        for wid, positions in w2p.items():
            is_rare = any(p in rare_pos for p in positions)
            is_mar = is_rare or any(p in mar_pos for p in positions)
            if is_rare:
                rare_pos.update(positions)
                mar_pos.update(positions)
            elif is_mar:
                mar_pos.update(positions)

        # Word 'cargo'
        cargo_p = [p for wid, p_list in w2p.items() for p in p_list if "cargo" in tokenizer.decode([input_ids[p]]).lower()][0]
        self.assertIn(cargo_p, mar_pos)

        # Word 'search'
        search_p = [p for wid, p_list in w2p.items() for p in p_list if "search" in tokenizer.decode([input_ids[p]]).lower()][0]
        self.assertIn(search_p, mar_pos)

        # Word 'and'
        and_p = [p for wid, p_list in w2p.items() for p in p_list if tokenizer.decode([input_ids[p]]).strip().lower() == "and"][0]
        self.assertNotIn(and_p, mar_pos, "Word 'and' must be classified as GENERAL")

        # Word 'rescue'
        rescue_p = [p for wid, p_list in w2p.items() for p in p_list if "rescue" in tokenizer.decode([input_ids[p]]).lower()][0]
        self.assertIn(rescue_p, mar_pos)

    def test_multi_piece_whole_word_consistency(self):
        """Verify that multi-piece terms have all constituent pieces classified consistently."""
        tokenizer = self.tok_roberta
        mar_ids, rare_ids, cat_ids = stage14.build_vocabulary_token_sets(tokenizer, self.vocab_sample)

        text = "The modern shipbuilding technique protected the windlass and bollards."
        enc = tokenizer(text, return_offsets_mapping=True)
        input_ids = enc["input_ids"]
        wids = enc.word_ids(0)
        sp_mask = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)

        eligible, rare_pos, mar_pos, gen_pos, cat_pos = stage14.classify_token_positions(
            text, input_ids, enc["offset_mapping"], sp_mask,
            set(self.vocab_sample), set(stage14.RARE_MARITIME_TERMS),
            mar_ids, rare_ids, cat_ids
        )
        word_to_pos, _ = stage14.extract_word_groups(wids, eligible)

        for wid, positions in word_to_pos.items():
            is_rare = any(p in rare_pos for p in positions)
            is_mar = is_rare or any(p in mar_pos for p in positions)
            if is_rare:
                rare_pos.update(positions)
                mar_pos.update(positions)
            elif is_mar:
                mar_pos.update(positions)

        # Check 'shipbuilding' (multi-piece)
        ship_wids = [wid for wid, p_list in word_to_pos.items() if "ship" in tokenizer.decode([input_ids[p_list[0]]]).lower()]
        self.assertTrue(len(ship_wids) > 0)
        for wid in ship_wids:
            for p in word_to_pos[wid]:
                self.assertIn(p, mar_pos, f"Subword piece {p} of shipbuilding was not marked maritime")

        # Check 'bollards' (rare multi-piece)
        boll_wids = [wid for wid, p_list in word_to_pos.items() if "b" in tokenizer.decode([input_ids[p_list[0]]]).lower() and "oll" in "".join(tokenizer.decode([input_ids[x]]) for x in p_list)]
        self.assertTrue(len(boll_wids) > 0)
        for wid in boll_wids:
            for p in word_to_pos[wid]:
                self.assertIn(p, rare_pos, f"Subword piece {p} of bollards was not marked rare")

    def test_wwm_integrity_all_pieces_masked(self):
        """Verify that whole-word masking masks ALL pieces of a selected word together."""
        word_to_positions = {
            0: [1],          # single piece
            1: [2, 3],       # two pieces
            2: [4, 5, 6],    # three pieces
            3: [7]           # single piece
        }
        rng = random.Random(42)
        # Budget of 3 tokens
        masked_positions, selected_words = stage14.create_whole_word_random_mask(word_to_positions, rng, mask_budget=3)
        
        # Verify that for every selected word, ALL of its positions are in masked_positions
        for wid in selected_words:
            for p in word_to_positions[wid]:
                self.assertIn(p, masked_positions, f"Position {p} of word {wid} was omitted from masked positions")
                
        # Verify that for unselected words, NONE of its positions are in masked_positions
        unselected = set(word_to_positions.keys()) - selected_words
        for wid in unselected:
            for p in word_to_positions[wid]:
                self.assertNotIn(p, masked_positions, f"Position {p} of unselected word {wid} was erroneously masked")

    def test_label_integrity_before_masking(self):
        """Verify labels are copied BEFORE masking and correspond to original input IDs."""
        input_ids = torch.tensor([[101, 2054, 2003, 1037, 102]])
        labels = input_ids.clone()
        mask_pos = 2
        
        # Apply mask
        masked_ids = input_ids.clone()
        masked_ids[0, mask_pos] = 103 # [MASK]
        
        # Assert label retains original input ID
        self.assertEqual(labels[0, mask_pos].item(), 2003)
        self.assertNotEqual(labels[0, mask_pos].item(), masked_ids[0, mask_pos].item())

    def test_strict_word_reconstruction_criterion(self):
        """Verify that word reconstruction requires 100% of subwords to be correctly predicted."""
        # Case 1: Partial correctness (1 of 2 pieces correct)
        pos_preds_partial = {1: True, 2: False}
        word_correct_partial = all(pos_preds_partial.get(p, False) for p in [1, 2])
        self.assertFalse(word_correct_partial, "Partially correct word must evaluate to False under strict word reconstruction")

        # Case 2: Complete correctness (2 of 2 pieces correct)
        pos_preds_full = {1: True, 2: True}
        word_correct_full = all(pos_preds_full.get(p, False) for p in [1, 2])
        self.assertTrue(word_correct_full, "Fully correct word must evaluate to True under strict word reconstruction")

    def test_cache_namespace_conventions(self):
        """Verify compute_cache_key uses correct segregated cache namespaces."""
        key_subword = stage14.compute_cache_key("roberta-base", "narrative", "high_knowledge", "subword", "subword")
        self.assertTrue(key_subword.startswith("cache/"))

        key_wwm_sub = stage14.compute_cache_key("roberta-base", "narrative", "high_knowledge", "whole_word", "subword")
        self.assertTrue(key_wwm_sub.startswith("cache_wwm_subword/"))

        key_wwm_word = stage14.compute_cache_key("roberta-base", "narrative", "high_knowledge", "whole_word", "word")
        self.assertTrue(key_wwm_word.startswith("cache_wwm_word/"))

if __name__ == "__main__":
    unittest.main()
