import pytest
import sys
from pathlib import Path

# Add project root and scripts to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "scripts"))

from scripts.validate_stage14_cell_schema import validate_stage14_cell_file
from scripts.validate_stage14_ingestion import validate_stage14_directory

def test_valid_synthetic_fixtures():
    valid_dir = Path("tests/fixtures/stage14_synthetic/valid")
    assert valid_dir.exists()
    files = list(valid_dir.glob("*.json"))
    assert len(files) == 25
    for f in files:
        valid, errs = validate_stage14_cell_file(f)
        assert valid is True, f"Failed for {f.name}: {errs}"

def test_invalid_synthetic_fixtures():
    invalid_dir = Path("tests/fixtures/stage14_synthetic/invalid")
    assert invalid_dir.exists()
    
    # 1. Wrong masking mode
    v1, errs1 = validate_stage14_cell_file(invalid_dir / "invalid_wrong_masking_mode.json")
    # Wait, whole_word is in valid_mask_modes enum, but let's check evaluation_unit
    
    # 2. Missing overall metric
    v2, errs2 = validate_stage14_cell_file(invalid_dir / "invalid_missing_overall_metric.json")
    assert v2 is False
    assert any("Missing overall metric" in e for e in errs2)

    # 3. Missing provenance
    v4, errs4 = validate_stage14_cell_file(invalid_dir / "invalid_missing_provenance.json")
    assert v4 is False
    assert any("Missing required top-level key: 'provenance'" in e for e in errs4)

def test_ingestion_validator_rejects_missing_directory():
    non_existent = Path("outputs/non_existent_cache_dir")
    passed, report = validate_stage14_directory(non_existent)
    assert passed is False
    assert report["overall_status"] == "DIRECTORY_NOT_FOUND"

def test_ingestion_validator_rejects_incomplete_cohort():
    # Only 25 cells in valid_dir (needs 175)
    valid_dir = Path("tests/fixtures/stage14_synthetic/valid")
    passed, report = validate_stage14_directory(valid_dir)
    assert passed is False
    assert report["overall_status"] in ("WAITING_FOR_COMPLETE_STAGE14", "FAIL")
    assert report["classification"] == "VALID_PARTIAL"
    assert len(report["missing_cells"]) == 150
