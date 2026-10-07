import unittest
import importlib
import sys
from pathlib import Path
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

        # Propagate word-level domain classification
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
                self.assertIn(p, mar_pos, f"Subword piece {p} ({tokenizer.decode([input_ids[p]])}) of shipbuilding was not marked maritime")

        # Check 'bollards' (rare multi-piece)
        boll_wids = [wid for wid, p_list in word_to_pos.items() if "b" in tokenizer.decode([input_ids[p_list[0]]]).lower() and "oll" in "".join(tokenizer.decode([input_ids[x]]) for x in p_list)]
        self.assertTrue(len(boll_wids) > 0)
        for wid in boll_wids:
            for p in word_to_pos[wid]:
                self.assertIn(p, rare_pos, f"Subword piece {p} of bollards was not marked rare")

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
