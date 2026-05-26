#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_datasets.py

Support: BBH / MMLU Math

- MMLU Math: covers abstract_algebra, college_mathematics, elementary_mathematics,
  high_school_mathematics, high_school_statistics, formal_logic, logical_fallacies
- Train : Test = 4:1 (train_ratio=0.8), with samples for each problem type
"""

import os
import re
import json
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from datasets import load_dataset, get_dataset_config_names
except ImportError as e:
    raise ImportError("pip install -U datasets huggingface_hub") from e


# =========================================================
# Basic Utilities
# =========================================================

def ensure_hf_mirror():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: str):
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_space(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def sample_list(data: List[Dict], limit: int, seed: int) -> List[Dict]:
    if limit is None or limit <= 0 or limit >= len(data):
        return data
    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    return [data[i] for i in idx[:limit]]


def split_by_train_ratio(
    data: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    if not data:
        return [], []
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1.")

    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)

    train_size = max(1, min(int(round(len(data) * train_ratio)), len(data) - 1))
    train_ids = set(idx[:train_size])

    train_data, test_data = [], []
    for i, item in enumerate(data):
        (train_data if i in train_ids else test_data).append(item)
    return train_data, test_data


def get_first(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def to_letter(idx: int) -> str:
    return chr(ord("A") + idx)


def stable_shuffle_options(question: str, options: List[str]) -> List[str]:
    seed = abs(hash(question)) % (2 ** 32)
    rng = random.Random(seed)
    copied = list(options)
    rng.shuffle(copied)
    return copied


def make_record(
    question: str,
    answer: Any,
    benchmark: str,
    task: str = "",
    subject: str = "",
    category: str = "",
    choices: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record = {
        "question": normalize_space(question),
        "answer": answer,
        "benchmark": benchmark,
        "task": task,
        "subject": subject,
        "category": category,
    }
    if choices:
        record["choices"] = [normalize_space(x) for x in choices]
    if extra:
        record.update(extra)
    return record


# =========================================================
# MMLU Common Sample Normalization
# =========================================================

def normalize_mmlu_example(
    ex: Dict[str, Any], subject_name: str,
    category: str,
) -> Optional[Dict[str, Any]]:
    question = get_first(ex, ["question", "input", "prompt"], "")
    question = normalize_space(question)
    if not question:
        return None

    choices = get_first(ex, ["choices", "options"], None)
    if not choices:
        for keys in [["A", "B", "C", "D"], ["a", "b", "c", "d"]]:
            if all(k in ex for k in keys):
                choices = [ex[k] for k in keys]
                break

    if not choices:
        return None

    answer = get_first(ex, ["answer", "label", "target"], None)
    if isinstance(answer, int):
        answer = to_letter(answer)
    elif isinstance(answer, str):
        ans = answer.strip()
        answer = to_letter(int(ans)) if ans.isdigit() else ans.upper()

    return make_record(
        question=question,
        choices=choices,
        answer=answer,
        benchmark="mmlu",
        task="mmlu",
        subject=subject_name,
        category=category,
    )


# =========================================================
# BBH
# =========================================================

def prepare_bbh(
    output_root: str,
    repo_id: str = "lukaemon/bbh",
    test_ratio: float = 0.2,
    seed: int = 42,
    max_per_task: int = 0,
    include_tasks: Optional[List[str]] = None,
):
    print(f"\n📦 Preparing BBH from repo: {repo_id}")

    try:
        config_names = get_dataset_config_names(repo_id)
    except Exception as e:
        raise RuntimeError(f"Unable to fetch BBH config list: {e}")

    if include_tasks:
        include_set = {x.strip() for x in include_tasks if x.strip()}
        config_names = [c for c in config_names if c in include_set]

    all_train, all_test = [], []

    for task_name in config_names:
        print(f"  - Loading BBH task: {task_name}")
        try:
            ds = load_dataset(repo_id, task_name)
        except Exception as e:
            print(f"    ⚠️ Skipping {task_name}: {e}")
            continue

        task_examples = []
        for split_name in ds.keys():
            for ex in ds[split_name]:
                question = get_first(ex, ["input", "question", "prompt"], "")
                answer = get_first(ex, ["target", "answer", "label"], "")
                if isinstance(answer, list):
                    answer = answer[0] if answer else ""
                task_examples.append(make_record(
                    question=question,
                    answer=normalize_space(answer),
                    benchmark="bbh",
                    task=task_name,
                    extra={"source_split": split_name},
                ))

        if max_per_task and max_per_task > 0:
            task_examples = sample_list(task_examples, max_per_task, seed)

        train_part, test_part = split_by_train_ratio(
            task_examples, train_ratio=1.0 - test_ratio, seed=seed
        )
        all_train.extend(train_part)
        all_test.extend(test_part)
        print(f"    ✅ {task_name}: total={len(task_examples)} train={len(train_part)} test={len(test_part)}")

    out_dir = Path(output_root) / "bbh"
    ensure_dir(str(out_dir))
    save_json(all_train, str(out_dir / "train.json"))
    save_json(all_test, str(out_dir / "test.json"))
    save_json({
        "benchmark": "bbh", "repo_id": repo_id,
        "num_train": len(all_train), "num_test": len(all_test),
        "test_ratio": test_ratio, "seed": seed, "tasks": config_names,
    }, str(out_dir / "meta.json"))

    print(f"✅ BBH done: train={len(all_train)}, test={len(all_test)}")
    print(f"📁 Output: {out_dir}")


# =========================================================
# MMLU Math
# =========================================================

MMLU_MATH_SUBJECTS = [
    "abstract_algebra",
    "college_mathematics",
    "elementary_mathematics",
    "high_school_mathematics",
    "high_school_statistics",
    "formal_logic",
    "logical_fallacies",
]

MMLU_MATH_SUBJECT_CATEGORY = {
    "abstract_algebra":        "algebra",
    "college_mathematics":     "college_math",
    "elementary_mathematics":  "elementary_math",
    "high_school_mathematics": "high_school_math",
    "high_school_statistics":  "statistics",
    "formal_logic":            "logic",
    "logical_fallacies":       "logic",
}


def prepare_mmlu_math(
    output_root: str,
    repo_id: str = "cais/mmlu",
    subjects: Optional[List[str]] = None,
    seed: int = 42,
    train_ratio: float = 0.8,
    train_limit: int = 0,
    test_limit: int = 0,
    min_per_subject_test: int = 5,
):
    """
    Prepare MMLU math subset.

    Design goals:
    - Cover all math/logic-related subjects, with samples for each problem type in both train and test sets
    - Train : Test = 4:1 (train_ratio=0.8)
    - Split each subject independently to ensure every problem type appears in the test set for per-category analysis
    - Output format is fully compatible with mat.py normalize_problem()
    """
    print(f"\n📦 Preparing MMLU Math from repo: {repo_id}")

    if not subjects:
        subjects = MMLU_MATH_SUBJECTS

    out_dir = Path(output_root) / "mmlu_math"
    ensure_dir(str(out_dir))

    all_train, all_test = [], []
    subject_stats = {}

    for subject_name in subjects:
        print(f"  - Loading MMLU subject: {subject_name}")
        try:
            ds = load_dataset(repo_id, subject_name)
        except Exception as e:
            print(f"    ⚠️ Skipping {subject_name}: {e}")
            continue

        category = MMLU_MATH_SUBJECT_CATEGORY.get(subject_name, "math")
        subject_records = []

        for split_name in ds.keys():
            for ex in ds[split_name]:
                item = normalize_mmlu_example(ex, subject_name, category=category)
                if item is not None:
                    item["source_split"] = split_name
                    subject_records.append(item)

        # Deduplication
        dedup = {}
        for item in subject_records:
            key = (item.get("question", ""), str(item.get("choices", "")))
            dedup[key] = item
        subject_records = list(dedup.values())

        if not subject_records:
            print(f"    ⚠️ {subject_name}: No valid samples, skipping")
            continue

        n = len(subject_records)
        # Split each subject independently, ensuring at least min_per_subject_test samples in the test set
        n_test = max(min_per_subject_test, int(round(n * (1.0 - train_ratio))))
        n_test = min(n_test, n - 1)
        n_train = n - n_test

        rng = random.Random(seed + hash(subject_name) % 10000)
        idx = list(range(n))
        rng.shuffle(idx)

        train_idx = set(idx[:n_train])
        s_train = [subject_records[i] for i in range(n) if i in train_idx]
        s_test  = [subject_records[i] for i in range(n) if i not in train_idx]

        all_train.extend(s_train)
        all_test.extend(s_test)

        subject_stats[subject_name] = {
            "total": n, "train": len(s_train), "test": len(s_test), "category": category,
        }
        print(f"    ✅ {subject_name}: total={n} | train={len(s_train)} | test={len(s_test)}")

    if train_limit > 0:
        all_train = sample_list(all_train, train_limit, seed)
    if test_limit > 0:
        all_test = sample_list(all_test, test_limit, seed)

    save_json(all_train, str(out_dir / "train.json"))
    save_json(all_test,  str(out_dir / "test.json"))

    ratio = round(len(all_train) / max(1, len(all_test)), 3)
    save_json({
        "benchmark": "mmlu", "subset": "math", "repo_id": repo_id,
        "subjects": subjects, "subject_stats": subject_stats,
        "num_total": len(all_train) + len(all_test),
        "num_train": len(all_train), "num_test": len(all_test),
        "train_ratio": train_ratio, "actual_train_test_ratio": ratio, "seed": seed,
    }, str(out_dir / "meta.json"))

    print(f"\n✅ MMLU Math done:")
    print(f"   Total : {len(all_train) + len(all_test)}")
    print(f"   Train : {len(all_train)}")
    print(f"   Test  : {len(all_test)}")
    print(f"   Train:Test = {ratio}:1")
    print(f"📁 Output: {out_dir}")
    return subject_stats


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download, filter and convert BBH / MMLU-Math into MAT-ready train.json/test.json"
    )

    parser.add_argument(
        "--dataset", default="all",
        choices=["all", "bbh", "mmlu_math"],
        help="Dataset to prepare. mmlu_math = MMLU math subset",
    )
    parser.add_argument("--output_root", default="prepared")
    parser.add_argument("--seed", type=int, default=42)

    # BBH
    parser.add_argument("--bbh_repo", default="lukaemon/bbh")
    parser.add_argument("--bbh_test_ratio", type=float, default=0.2)
    parser.add_argument("--bbh_max_per_task", type=int, default=0)
    parser.add_argument("--bbh_tasks", default="")

    # MMLU Math
    parser.add_argument(
        "--mmlu_math_subjects",
        default=",".join(MMLU_MATH_SUBJECTS),
        help="Comma-separated list of MMLU math subjects",
    )
    parser.add_argument(
        "--mmlu_math_train_ratio", type=float, default=0.8,
        help="MMLU Math training set ratio, default 0.8 (train:test=4:1)",
    )
    parser.add_argument("--mmlu_math_train_limit", type=int, default=0)
    parser.add_argument("--mmlu_math_test_limit", type=int, default=0)
    parser.add_argument(
        "--mmlu_math_min_per_subject_test", type=int, default=5,
        help="Minimum number of test samples to keep per subject",
    )

    args = parser.parse_args()

    ensure_hf_mirror()
    ensure_dir(args.output_root)

    print("🚀 Preparing datasets with Hugging Face mirror")
    print(f"🌐 HF_ENDPOINT = {os.environ.get('HF_ENDPOINT')}")
    print(f"📁 output_root = {args.output_root}")

    if args.dataset in ["all", "bbh"]:
        include_tasks = [x.strip() for x in args.bbh_tasks.split(",") if x.strip()]
        prepare_bbh(
            output_root=args.output_root,
            repo_id=args.bbh_repo,
            test_ratio=args.bbh_test_ratio,
            seed=args.seed,
            max_per_task=args.bbh_max_per_task,
            include_tasks=include_tasks if include_tasks else None,
        )

    if args.dataset in ["all", "mmlu_math"]:
        math_subjects = [x.strip() for x in args.mmlu_math_subjects.split(",") if x.strip()]
        prepare_mmlu_math(
            output_root=args.output_root,
            subjects=math_subjects,
            seed=args.seed,
            train_ratio=args.mmlu_math_train_ratio,
            train_limit=args.mmlu_math_train_limit,
            test_limit=args.mmlu_math_test_limit,
            min_per_subject_test=args.mmlu_math_min_per_subject_test,
        )

    print("\n✅ All requested datasets are prepared.")


if __name__ == "__main__":
    main()