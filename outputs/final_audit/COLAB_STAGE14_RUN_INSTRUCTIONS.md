# Google Colab Execution Instructions for Stage 14 Benchmark
**MaritimeBench / MaritimeBERT Pipeline Version 2.1**
**Date**: October 8, 2026

---

## 1. Overview & Objective
Stage 14 evaluates **7 pretrained language models** across **5 textual representations** and **5 informativeness subsets** (175 evaluation cells $\times$ 200 documents = 35,000 document evaluations).

To avoid local compute exhaustion, Stage 14 is designed to run in **Google Colab** with GPU acceleration (Nvidia T4, V100, or A100).
Once completed, the resulting cache directory `cache_legacy_subword_15/` is downloaded and placed into your local repository to unlock Stages 15 through 18.

---

## 2. Step-by-Step Colab Instructions

### Step 1: Open Google Colab & Select GPU Runtime
1. Open [Google Colab](https://colab.research.google.com).
2. Create a new notebook.
3. In the menu, go to **Runtime** $\to$ **Change runtime type**.
4. Select **T4 GPU** (or A100 if Colab Pro is available) and click **Save**.

### Step 2: Upload or Clone the Pipeline
In a Colab notebook cell:
```python
# Option A: If cloning from GitHub
!git clone https://github.com/AbhishekYadav2207/TSBC-MaritimePipeline-PR.git repo
%cd repo

# Option B: Or mount Google Drive where you have the repo
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/TSBC-MaritimePipeline-Version2.1
```

### Step 3: Install Required Dependencies
```python
!pip install -q transformers>=4.36.0 torch>=2.1.0 scipy>=1.11.0 pandas>=2.0.0 jsonschema>=4.19.0
```

Verify GPU availability:
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
```

### Step 4: Run the Primary Legacy Stage 14 Benchmark
Execute Stage 14 using the primary restored protocol (Random-15% Subword MLM):
```python
!python scripts/14_mlm_evaluation.py \
    --masking_mode subword \
    --evaluation_unit subword \
    --masking_strategy random_15 \
    --sample_size 200 \
    --cache_dir outputs/stage-14/evaluations/cache_legacy_subword_15 \
    --device cuda
```

*Note*: Default CLI values already point to `subword`, `subword`, `random_15`, and `outputs/stage-14/evaluations/cache_legacy_subword_15`, so running simply `!python scripts/14_mlm_evaluation.py --device cuda` also works identically.

**Expected Runtime**:
- Nvidia T4: ~1.5 to 2.5 hours
- Nvidia A100: ~30 to 45 minutes

### Step 5: Verify Cache Completeness
Check that all 175 cells were generated:
```python
import glob
files = glob.glob("outputs/stage-14/evaluations/cache_legacy_subword_15/*.json")
print(f"Generated Cell Count: {len(files)} / 175")
assert len(files) == 175, "Expected exactly 175 cell files!"
```

### Step 6: Compress and Download the Cache
```python
!zip -q -r cache_legacy_subword_15.zip outputs/stage-14/evaluations/cache_legacy_subword_15

from google.colab import files
files.download("cache_legacy_subword_15.zip")
```

---

## 3. Local Workspace Ingestion & Downstream Execution

Once `cache_legacy_subword_15.zip` is downloaded to your local machine:
1. Extract the contents into:
   `outputs/stage-14/evaluations/cache_legacy_subword_15/`
   (Ensuring all 175 `.json` files reside in this folder).
2. In PowerShell or Command Prompt, run:
   ```powershell
   python scripts/continue_after_stage14.py --continue
   ```

This command will automatically:
- Ingest and validate all 175 cell schemas.
- Run independent calculation audits ($\epsilon < 10^{-6}$).
- Execute Stage 15 (Cross-Model Benchmarking).
- Execute Stage 16 (Statistical Validation & Crossed ANOVA).
- Execute Stage 17 (Evidence Decision Engine).
- Execute Stage 18 (Corpus Linting).
- Output the final publication report.
