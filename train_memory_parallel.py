"""
Parallel memory training script for MAT on BBH / MMLU .

This script builds long-term experience memory from training problems.

Examples:

BBH:
python train_memory_parallel.py \
  --train_file bbh_train.json \
  --memory_file memory_bbh.json \
  --benchmark bbh

MMLU math:
python train_memory_parallel.py \
  --train_file mmlu_train.json \
  --memory_file memory_mmlu_math.json \

"""

import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from mat import (
    setup_textgrad_with_deepseek,
    ExperienceMemory,
    run_single_problem,
    load_problem_file,
    filter_problems,
    save_json,
    SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def train_one(problem, memory, args):
    return run_single_problem(
        problem=problem,
        method="mat",
        memory=memory,
        max_iterations=args.max_iterations,
        is_training=True,
        sim_threshold=args.sim_threshold,
        top_k_experiences=args.top_k,
        use_retrieval=args.use_retrieval,
        use_adaptive_iter=args.use_adaptive_iter,
        use_gradient_injection=args.use_gradient_injection,
        initial_temperature=args.initial_temperature,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", default="train.json")
    parser.add_argument("--memory_file", default="memory_after_training.json")
    parser.add_argument("--benchmark", default="mixed", choices=["mixed", "bbh", "mmlu"])

    parser.add_argument("--capacity", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max_iterations", type=int, default=3)
    parser.add_argument("--sim_threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--initial_temperature", type=float, default=0.0)

    parser.add_argument("--include_subjects", default=None)
    parser.add_argument("--include_categories", default=None)
    parser.add_argument("--include_tasks", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_retrieval", action="store_true", default=True)
    parser.add_argument("--no_retrieval", dest="use_retrieval", action="store_false")

    parser.add_argument("--use_adaptive_iter", action="store_true", default=True)
    parser.add_argument("--no_adaptive_iter", dest="use_adaptive_iter", action="store_false")

    parser.add_argument("--use_gradient_injection", action="store_true", default=True)
    parser.add_argument(
        "--no_gradient_injection",
        dest="use_gradient_injection",
        action="store_false",
    )

    args = parser.parse_args()

    setup_textgrad_with_deepseek()

    raw_train = load_problem_file(args.train_file)
    train_data = filter_problems(
        raw_train,
        include_subjects=args.include_subjects,
        include_categories=args.include_categories,
        include_tasks=args.include_tasks,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    memory = ExperienceMemory(
        capacity=args.capacity,
        similarity_threshold=args.sim_threshold,
    )

    print("🚀 Starting parallel MAT memory training")
    print(f"📌 Benchmark: {args.benchmark}")
    print(f"📚 Raw training problems: {len(raw_train)}")
    print(f"📚 Selected training problems: {len(train_data)}")
    print(f"🧵 Threads: {args.threads}")
    print(f"🧠 Memory capacity: {args.capacity}")

    t0 = time.time()
    results = [None] * len(train_data)

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(train_one, p, memory, args): i
            for i, p in enumerate(train_data)
        }

        completed = 0

        for future in as_completed(futures):
            idx = futures[future]
            completed += 1

            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {
                    "success": False,
                    "error": str(e),
                    "problem_index": idx,
                }

            if completed % 50 == 0 or completed == len(train_data):
                print(
                    f"✅ Progress: {completed}/{len(train_data)} | "
                    f"Stored memories: {len(memory.experiences)}"
                )

    memory.save(args.memory_file)
    save_json(results, "train_memory_results.json")

    elapsed = (time.time() - t0) / 60.0

    print("🎉 Training completed")
    print(f"⏱️ Runtime: {elapsed:.2f} minutes")
    print(f"🧠 Final stored experiences: {len(memory.experiences)}")
    print(f"💾 Memory saved to: {args.memory_file}")
    print("📄 Training logs saved to: train_memory_results.json")


if __name__ == "__main__":
    main()