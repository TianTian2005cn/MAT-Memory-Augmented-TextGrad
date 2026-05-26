# MAT: Memory-Augmented TextGrad

> **Learning to Optimize Faster: Multi-Timescale Prompt Optimization with Episodic Gradient Memory**
> Tian Tian, Ziqi Xu, Renqiang Luo · ICONIP 2026

[![Paper](https://img.shields.io/badge/Paper-ICONIP%202026-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]()

MAT augments [TextGrad](https://github.com/zou-group/textgrad) with an **episodic memory of successful optimization trajectories**, inspired by the multi-timescale principle of Nested Learning.
Across GSM8K, BBH, and MMLU Math, MAT consistently reduces API calls by **29–34%** compared to Vanilla TextGrad while maintaining or slightly improving accuracy.
The entire experimental pipeline costs **under 30 USD**.

---

## Key Idea

MAT operates at three timescales:

- **Fast inner loop** — standard textual gradient descent (per iteration).
- **Medium-timescale controller** — retrieves semantically similar past experiences and predicts the optimal number of iterations (per problem).
- **Slow outer loop** — accumulates an episodic memory of successful trajectories (across problems).

The core technical contribution is **gradient-level injection**: retrieved insights are wrapped as a `Variable` and added to TextGrad's gradient set via `set.add`, enriching the computation graph itself rather than prepending memory to the input prompt.

---

## Repository Structure

```
.
├── mat.py                          # Core MAT implementation
├── prepare_datasets.py             # Download & convert BBH / MMLU Math
├── train_memory_parallel.py        # Build episodic memory from training set
├── retry_failed.py                 # Retry failed/untrained examples
├── run_tests_parallel.py           # Evaluate Vanilla TextGrad vs MAT
├── run_experiments_planB.py        # Cross-benchmark experiment runner
├── run_extra_baselines.py          # CoT / Zero-shot CoT / Reflexion / TSGD-M baselines
├── collect_extra_baseline_table.py # Aggregate baseline results into tables
├── scripts/                        # One-line reproduction scripts
├── results/                        # Final result JSONs (per benchmark)
├── requirements.txt
└── README.md
```

Pre-trained memory files (`memory_*.json`) are released via [GitHub Releases](../../releases) so reviewers can skip the training phase.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

We recommend Python 3.9+ and a fresh virtual environment.

### 2. Configure API access

MAT uses **DeepSeek-V4-Flash** as the backbone LLM (cheap and fast) and a local **MiniLM** model for embeddings (zero API cost).

```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

Set up via `setup_textgrad_with_deepseek()` in `mat.py`. You can replace the backbone with any TextGrad-compatible LLM by editing this function.

### 3. (Optional) Hugging Face mirror

`prepare_datasets.py` automatically sets `HF_ENDPOINT=https://hf-mirror.com` for users in regions where Hugging Face is slow. Override with:

```bash
export HF_ENDPOINT=https://huggingface.co
```

---

## Reproduce the Paper

### Option A — Quick (~5 USD): use released memory

If you only want to verify the test-time numbers, download our pre-trained memory files from [Releases](../../releases) into the project root, then jump to **Step 3**.

### Option B — Full pipeline (~30 USD): train from scratch

#### Step 1. Prepare datasets

```bash
python prepare_datasets.py --dataset all --output_root prepared
```

This downloads and normalizes BBH (23 tasks) and MMLU Math (7 subjects) into a unified schema. For GSM8K, follow the standard HuggingFace loader.

#### Step 2. Train episodic memory

```bash
# BBH
python train_memory_parallel.py \
  --train_file prepared/bbh/train.json \
  --memory_file memory_bbh.json \
  --benchmark bbh \
  --threads 8 \
  --capacity 3000 \
  --max_iterations 3

# MMLU Math
python train_memory_parallel.py \
  --train_file prepared/mmlu_math/train.json \
  --memory_file memory_mmlu_math.json \
  --benchmark mmlu \
  --threads 8 \
  --capacity 3000 \
  --max_iterations 3
```

To retry failed/untrained problems:

```bash
python retry_failed.py \
  --train_file prepared/bbh/train.json \
  --memory_file memory_bbh.json \
  --max_retries 3
```

#### Step 3. Evaluate on test sets

```bash
# BBH
python run_tests_parallel.py \
  --test_file prepared/bbh/test.json \
  --memory_file memory_bbh.json \
  --output_file results/results_bbh.json \
  --benchmark bbh \
  --threads 8

# MMLU Math
python run_tests_parallel.py \
  --test_file prepared/mmlu_math/test.json \
  --memory_file memory_mmlu_math.json \
  --output_file results/results_mmlu_math.json \
  --benchmark mmlu \
  --threads 8
```

This evaluates **Vanilla TextGrad** and **MAT** on the same set of problems using a **shared initial-solution cache** (`initial_solutions_test.json`) to ensure fair comparison.

#### Step 4. Extra baselines (CoT / Zero-shot CoT / Reflexion / TSGD-M)

```bash
python run_extra_baselines.py \
  --test_file prepared/bbh/test.json \
  --output_file summary_extra_bbh.json

python collect_extra_baseline_table.py \
  --summaries summary_extra_bbh.json summary_extra_mmlu.json \
  --output_md table_extra_baselines.md \
  --output_tex table_extra_baselines.tex
```

---

## Ablations

To disable individual MAT components, use the corresponding flags in `run_tests_parallel.py`:

```bash
# w/o retrieval — equivalent to Vanilla TextGrad
python run_tests_parallel.py ... --no_retrieval

# w/o adaptive iteration prediction
python run_tests_parallel.py ... --no_adaptive_iter

# w/o gradient-level injection
python run_tests_parallel.py ... --no_gradient_injection
```

### Similarity threshold sweep

```bash
for theta in 0.3 0.4 0.5 0.6 0.7; do
  python run_tests_parallel.py \
    --test_file prepared/bbh/test.json \
    --memory_file memory_bbh.json \
    --output_file results/threshold_bbh_${theta}.json \
    --sim_threshold $theta
done
```

---

## Hyperparameters

| Hyperparameter           | Value | Flag                     |
|--------------------------|-------|--------------------------|
| Similarity threshold $\theta$ | 0.4   | `--sim_threshold`        |
| Top-$k$ retrieval        | 1     | `--top_k`                |
| Max iterations $N_{\max}$ | 3     | `--max_iterations`       |
| Memory capacity          | 3000  | `--capacity`             |
| Embedding model          | `all-MiniLM-L6-v2` (local CPU) | hard-coded |
| Backbone LLM             | DeepSeek-V4-Flash | `setup_textgrad_with_deepseek` |
| Parallel threads         | 8     | `--threads`              |

These match the defaults reported in the paper and are held constant across all experiments unless explicitly varied in ablations.

---

## Cost Breakdown

| Stage                          | API Cost (approx.) |
|--------------------------------|--------------------|
| Memory training (BBH)          | ~12 USD            |
| Memory training (MMLU Math)    | ~3 USD             |
| Memory training (GSM8K)        | ~8 USD             |
| Test-time evaluation (all 3)   | ~5 USD             |
| **Total**                      | **~28 USD**        |

Embedding (MiniLM) runs entirely on local CPU with zero API cost.

---

## Results

See `results/` for raw JSON outputs from our experiments. Headline numbers (test sets):

| Benchmark   | Vanilla TG Acc. | MAT Acc. | API ↓   | Time ↓ |
|-------------|-----------------|----------|---------|--------|
| GSM8K       | 94.3%           | 94.4%    | -34%    | -26%   |
| BBH         | 50.7%           | 50.8%    | -33%    | -31%   |
| MMLU Math   | 89.3%           | 89.3%    | -29%    | -36%   |

---

## Citation

If you find MAT useful, please cite:

```bibtex
@inproceedings{tian2026mat,
  title     = {Learning to Optimize Faster: Multi-Timescale Prompt Optimization with Episodic Gradient Memory},
  author    = {Tian, Tian and Xu, Ziqi and Luo, Renqiang},
  booktitle = {Proceedings of the 33rd International Conference on Neural Information Processing (ICONIP)},
  year      = {2026}
}
```

---

## Acknowledgements

MAT builds on [TextGrad](https://github.com/zou-group/textgrad) (Yuksekgonul et al., 2025) and draws conceptually from [Nested Learning](https://arxiv.org/abs/2511.xxxxx) (Behrouz et al., 2025). We thank the authors for releasing their work openly.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contact

For questions, please open an issue or contact `tiantian5523@mails.jlu.edu.cn`.
