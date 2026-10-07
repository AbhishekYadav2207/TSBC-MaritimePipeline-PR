import unittest
import json
import tempfile
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class TestNotebookOrchestration(unittest.TestCase):
    def setUp(self):
        nb_at_root = PROJECT_ROOT / "MaritimeBench_Full_Pipeline.ipynb"
        nb_in_dir = PROJECT_ROOT / "notebooks" / "MaritimeBench_Full_Pipeline.ipynb"
        self.nb_path = nb_at_root if nb_at_root.exists() else nb_in_dir
        with open(self.nb_path, "r", encoding="utf-8") as f:
            self.nb = json.load(f)

    def test_notebook_syntax_and_compilation(self):
        """1. Validate notebook syntax and verify all Python cells compile."""
        self.assertIn("cells", self.nb)
        code_cells = [c for c in self.nb["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 22, "Expected 22 code cells in notebook")
        for idx, cell in enumerate(code_cells):
            src = "".join(cell["source"])
            try:
                compile(src, f"<cell_{idx}>", "exec")
            except Exception as e:
                self.fail(f"Cell {idx} failed compilation: {e}")

    def test_no_deep_schema_validation_in_notebook(self):
        """Verify no deep schema assertions (required_keys, required_columns, schema definitions) remain in notebook code."""
        code_cells = [c for c in self.nb["cells"] if c["cell_type"] == "code"]
        for idx, cell in enumerate(code_cells):
            src = "".join(cell["source"])
            self.assertNotIn("required_keys", src, f"Found 'required_keys' in code cell {idx}")
            self.assertNotIn("required_columns", src, f"Found 'required_columns' in code cell {idx}")
            self.assertNotIn("validate_json(", src, f"Found 'validate_json(' in code cell {idx}")
            self.assertNotIn("validate_csv(", src, f"Found 'validate_csv(' in code cell {idx}")

    def test_run_stage_stops_on_nonzero_exit_code(self):
        """Confirm a non-zero production script correctly stops the pipeline with BenchmarkStageException."""
        # Extract Cell 3 helper environment
        cell_3_code = "".join(self.nb["cells"][3]["source"])
        env = {
            "PROJECT_ROOT": PROJECT_ROOT,
            "OUTPUTS_DIR": PROJECT_ROOT / "outputs",
            "SCRIPTS_DIR": PROJECT_ROOT / "scripts",
            "__name__": "__main__",
        }
        exec(cell_3_code, env)
        run_stage = env["run_stage"]
        BenchmarkStageException = env["BenchmarkStageException"]

        # Run non-existent / failing script or python -c "sys.exit(1)"
        with tempfile.TemporaryDirectory() as tmpdir:
            fail_script = Path(tmpdir) / "fail_script.py"
            fail_script.write_text("import sys; print('Simulated script failure', file=sys.stderr); sys.exit(42)", encoding="utf-8")
            
            with self.assertRaises(BenchmarkStageException) as ctx:
                run_stage(
                    stage_id="TEST_FAIL",
                    script_rel_path=str(fail_script.relative_to(PROJECT_ROOT)) if fail_script.is_relative_to(PROJECT_ROOT) else str(fail_script),
                    expected_outputs=[]
                )
            self.assertIn("42", str(ctx.exception))

    def test_artifact_inventory_detection(self):
        """Confirm missing output artifact and zero-byte artifact are properly identified."""
        cell_3_code = "".join(self.nb["cells"][3]["source"])
        env = {
            "PROJECT_ROOT": PROJECT_ROOT,
            "OUTPUTS_DIR": PROJECT_ROOT / "outputs",
            "SCRIPTS_DIR": PROJECT_ROOT / "scripts",
            "__name__": "__main__",
        }
        exec(cell_3_code, env)
        inventory_artifact = env["inventory_artifact"]
        inventory_outputs = env["inventory_outputs"]
        BenchmarkStageException = env["BenchmarkStageException"]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            missing_file = tmppath / "nonexistent.json"
            empty_file = tmppath / "empty.json"
            valid_file = tmppath / "valid.json"
            empty_dir = tmppath / "empty_dir"
            populated_dir = tmppath / "populated_dir"

            empty_file.write_text("", encoding="utf-8")
            valid_file.write_text('{"status": "ok"}', encoding="utf-8")
            empty_dir.mkdir()
            populated_dir.mkdir()
            (populated_dir / "item.txt").write_text("hello", encoding="utf-8")

            # Check missing
            inv_missing = inventory_artifact(missing_file)
            self.assertFalse(inv_missing["exists"])
            self.assertEqual(inv_missing["type"], "MISSING")

            # Check empty file
            inv_empty = inventory_artifact(empty_file)
            self.assertTrue(inv_empty["exists"])
            self.assertEqual(inv_empty["type"], "FILE")
            self.assertEqual(inv_empty["size_bytes"], 0)

            # Check valid file
            inv_valid = inventory_artifact(valid_file)
            self.assertTrue(inv_valid["exists"])
            self.assertGreater(inv_valid["size_bytes"], 0)

            # Check empty dir
            inv_empty_dir = inventory_artifact(empty_dir)
            self.assertTrue(inv_empty_dir["exists"])
            self.assertEqual(inv_empty_dir["type"], "DIRECTORY")
            self.assertEqual(inv_empty_dir["items_count"], 0)

            # Check populated dir
            inv_pop_dir = inventory_artifact(populated_dir)
            self.assertTrue(inv_pop_dir["exists"])
            self.assertEqual(inv_pop_dir["type"], "DIRECTORY")
            self.assertEqual(inv_pop_dir["items_count"], 1)

            # inventory_outputs accurately catalogs outputs
            inv_all = inventory_outputs([missing_file, empty_file, valid_file, empty_dir, populated_dir])
            self.assertEqual(len(inv_all), 5)
            self.assertFalse(inv_all[0]["exists"])
            self.assertEqual(inv_all[1]["size_bytes"], 0)
            self.assertGreater(inv_all[2]["size_bytes"], 0)
            self.assertEqual(inv_all[3]["items_count"], 0)
            self.assertEqual(inv_all[4]["items_count"], 1)

            # run_stage raises BenchmarkStageException on missing or empty expected outputs
            dummy_script = tmppath / "dummy_success.py"
            dummy_script.write_text("import sys; sys.exit(0)", encoding="utf-8")
            run_stage = env["run_stage"]

            with self.assertRaises(BenchmarkStageException):
                run_stage("TEST_MISSING", str(dummy_script), expected_outputs=[missing_file])

            with self.assertRaises(BenchmarkStageException):
                run_stage("TEST_EMPTY_FILE", str(dummy_script), expected_outputs=[empty_file])

            with self.assertRaises(BenchmarkStageException):
                run_stage("TEST_EMPTY_DIR", str(dummy_script), expected_outputs=[empty_dir])

            # Passes on valid outputs (either PASS or RESUMED when artifacts already present)
            res = run_stage("TEST_VALID", str(dummy_script), expected_outputs=[valid_file, populated_dir])
            self.assertIn(res["status"], ["PASS", "RESUMED (OUTPUTS_EXIST)"])
            self.assertEqual(len(res["artifacts"]), 2)

    def test_zero_exit_unusual_valid_output_accepted(self):
        """Confirm a zero-return production script with unusual but valid output is accepted without schema rejection."""
        cell_3_code = "".join(self.nb["cells"][3]["source"])
        env = {
            "PROJECT_ROOT": PROJECT_ROOT,
            "OUTPUTS_DIR": PROJECT_ROOT / "outputs",
            "SCRIPTS_DIR": PROJECT_ROOT / "scripts",
            "__name__": "__main__",
        }
        exec(cell_3_code, env)
        run_stage = env["run_stage"]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            custom_out = tmppath / "unusual_output.json"
            test_script = tmppath / "unusual_script.py"
            # Emits valid JSON with arbitrary schema that old validator would fail on
            test_script.write_text(f"""
import json
with open(r"{custom_out}", "w") as f:
    json.dump({{"unexpected_key": 123, "custom_field": "test"}}, f)
""", encoding="utf-8")

            # Must succeed because exit code == 0 and output exists & size > 0
            res = run_stage(
                stage_id="UNUSUAL_STAGE",
                script_rel_path=str(test_script),
                expected_outputs=[custom_out]
            )
            self.assertEqual(res["status"], "PASS")
            self.assertEqual(res["return_code"], 0)
            self.assertEqual(len(res["artifacts"]), 1)
            self.assertTrue(res["artifacts"][0]["exists"])
            self.assertGreater(res["artifacts"][0]["size_bytes"], 0)

    def test_stage_14_gpu_gate_and_baseline_cli_preserved(self):
        """Confirm GPU gate, ALLOW_CPU_SMOKE_TEST logic, and Stage 14 baseline CLI arguments are preserved."""
        cell_31_code = "".join(self.nb["cells"][31]["source"])
        self.assertIn("--masking_mode", cell_31_code)
        self.assertIn("MLM_MASKING_MODE", cell_31_code)
        self.assertIn("--evaluation_unit", cell_31_code)
        self.assertIn("MLM_EVALUATION_UNIT", cell_31_code)
        self.assertIn("--device", cell_31_code)
        self.assertIn("ALLOW_CPU_SMOKE_TEST", cell_31_code)
        self.assertIn("BLOCKED_GPU_REQUIRED", cell_31_code)
        self.assertIn("SMOKE_TEST_ONLY", cell_31_code)

        # Cell 3 defaults
        cell_3_code = "".join(self.nb["cells"][3]["source"])
        self.assertIn('MLM_MASKING_MODE = "subword"', cell_3_code)
        self.assertIn('MLM_EVALUATION_UNIT = "subword"', cell_3_code)
        self.assertIn("ALLOW_CPU_SMOKE_TEST = False", cell_3_code)

if __name__ == "__main__":
    unittest.main()
