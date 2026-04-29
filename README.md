# Deep Learning MCQ Image Solver — GNR638 Project 2

**Team:**

- Souparna Bhowmik — 25D1386
- Vivekananda Giri — 25D1381

A vision-language pipeline that reads PNG images of multiple-choice questions about deep learning and emits `submission.csv` with one option in `{1, 2, 3, 4, 5}` per row. Option `5` is an explicit abstention. Inference runs fully offline on a 48 GB L40s GPU.

## Grader sequence

```bash
bash setup.bash
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>
python <grading_script> --submission_file submission.csv
conda remove --name gnr_project_env --all -y
```

`setup.bash` runs with internet **on**: it clones this repo into the current directory, creates the conda env `gnr_project_env` (Python 3.11), installs the pip dependencies, and downloads the model weights (~5 GB) into `./models/Qwen2.5-VL-7B-Instruct-AWQ/`.

`inference.py` runs with internet **off**: it sets `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` before any heavy imports, reads `test.csv` and `images/` from `--test_dir`, and writes `submission.csv` to the **project directory** (cwd at grader-time), not to `--test_dir`.

## Local development

```bash
bash setup.bash
conda activate gnr_project_env
python inference.py --test_dir /absolute/path/to/test_dir
```

To verify the produced CSV is structurally valid:

```bash
python verify.py ./
```

## Hardware requirements

| Resource | Value |
|----------|-------|
| GPU | NVIDIA L40s (Ada SM 8.9), 48 GB VRAM |
| CUDA | 12.6 driver (R555+); pip wheels target the cu126 channel |
| RAM | 16 GB system memory |
| Internet at inference | None |
| Wall-clock budget | ≤ 1 hour for ≤ 50 images |

## Repository layout

```
.
├── inference.py              # Grader entry point (--test_dir <abs>)
├── setup.bash                # Internet-on bootstrap
├── verify.py                 # Offline structural CSV check
├── download_weights.py       # HuggingFace snapshot_download (internet required)
├── requirements.txt          # pip lockfile
├── environment.yml           # conda env (gnr_project_env, python 3.11)
├── tau_sweep.py              # τ grid-search calibration
├── dev_eval.py               # Rubric eval harness (dev-only)
├── src/solver/               # Pipeline modules
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── reasoner.py
│   ├── output_validator.py
│   ├── confidence.py
│   ├── submission_writer.py
│   ├── calibration_writer.py
│   └── pipeline.py
└── dev_set/                  # 70-image dev set
    ├── labels.csv
    ├── dev_margins.csv
    ├── tau_sweep.csv
    ├── render_mmlu.py
    └── images/
```

## Architecture

The pipeline is a single forward pass per image:

1. **Data loader** — reads `test.csv` (column `image_name`) and resolves PNGs from `images/`, `image/`, or `Images/` (case-tolerant).
2. **Preprocess** — opens each PNG with Pillow, normalises to RGB.
3. **Reasoner** — runs Qwen2.5-VL-7B-Instruct-AWQ via vLLM with chain-of-thought prompting (`<transcribe>` / `<analyze>` / `<answer>` tags). The answer slot is regex-extracted from the response.
4. **Output validator** — coerces the extracted digit to `int ∈ {1, 2, 3, 4, 5}`. Anything outside that set becomes `5` (abstain).
5. **Confidence gate** — computes a softmax-normalised margin over the four option-token logits. If margin ≥ τ the gate emits the option; otherwise it emits `5`. Default τ is 0.5, calibrated offline via `tau_sweep.py` on the 70-image dev set.
6. **Submission writer** — writes `id,image_name,option` to `submission.csv` in the project cwd, ASCII no-BOM, LF line endings.

The output validator is the only path to disk; nothing else writes the CSV. This eliminates the −1 hallucination penalty by construction.

## Citations

- Hendrycks et al. 2021. *Measuring Massive Multitask Language Understanding.* ICLR. (MMLU dataset, used for 50 of 70 dev-set rows. CC-BY-4.0.) https://huggingface.co/datasets/cais/mmlu
- Bai et al. 2025. *Qwen2.5-VL Technical Report.* (Primary VLM, Apache-2.0.) https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ
- Lin et al. 2023. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* https://arxiv.org/abs/2306.00978
- Kwon et al. 2023. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP. (vLLM inference engine and structured-output choice mask.) https://github.com/vllm-project/vllm
- The remaining 20 dev-set MCQs are original questions written by the team.
