"""
Retry failed or untrained MAT memory-building examples.

This script reloads an existing memory file, finds training problems whose
normalized problem ids are not yet stored, and retries them.

Examples:

python retry_failed.py \
  --train_file mmlu_train.json \
  --memory_file memory_mmlu_physics.json \
  --include_subjects high_school_physics,college_physics
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
    normalize_problem,
    ExperienceMemory,
    SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def train_with_retry(problem, memory, args, idx):
    last_error = None

    for attempt in range(1, args.max_retries + 1):
        try:
            result = run_single_problem(
                problem=problem,
                method="mat",
                memory=memory,
                max_iterations=args.max_iterations,
                is_training=True,
                sim_threshold=args.sim_threshold,
                top_k_experiences=args.top_k,
                use_retrieval=True,
                use_adaptive_iter=True,
                use_gradient_injection=True,
                initial_temperature=args.initial_temperature,
            )
            result["retry_attempts"] = attempt
            return idx, result

        except Exception as e:
            last_error = str(e)

            if attempt < args.max_retries:
                time.sleep(args.retry_sleep)

    return idx, {
        "success": False,
        "error": last_error,
        "problem_index": idx,
        "retry_attempts": args.max_retries,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", default="train.json")
    parser.add_argument("--memory_file", default="memory_after_training.json")
    parser.add_argument("--output_file", default="memory_after_training.json")
    parser.add_argument("--benchmark", default="mixed", choices=["mixed", "bbh", "mmlu", "gpqa"])

    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=3000)
    parser.add_argument("--max_iterations", type=int, default=3)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=3.0)

    parser.add_argument("--sim_threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--initial_temperature", type=float, default=0.0)

    parser.add_argument("--include_subjects", default=None)
    parser.add_argument("--include_categories", default=None)
    parser.add_argument("--include_tasks", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

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
    memory.load(args.memory_file)

    trained_ids = {exp.problem_id for exp in memory.experiences}

    failed_or_untrained = []
    for p in train_data:
        n = normalize_problem(p)
        pid = ExperienceMemory.problem_id(n["question"])

        if pid not in trained_ids:
            failed_or_untrained.append(p)

    print("🔁 Retry failed/untrained examples")
    print(f"📌 Benchmark: {args.benchmark}")
    print(f"📚 Raw training problems: {len(raw_train)}")
    print(f"📚 Selected training problems: {len(train_data)}")
    print(f"🧠 Already stored memories: {len(memory.experiences)}")
    print(f"🔁 Problems to retry: {len(failed_or_untrained)}")

    if not failed_or_untrained:
        print("✅ No failed or untrained problems found.")
        return

    t0 = time.time()
    retry_results = [None] * len(failed_or_untrained)

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(train_with_retry, p, memory, args, i): i
            for i, p in enumerate(failed_or_untrained)
        }

        completed = 0

        for future in as_completed(futures):
            completed += 1
            idx = futures[future]

            try:
                ridx, res = future.result()
                retry_results[ridx] = res
            except Exception as e:
                retry_results[idx] = {
                    "success": False,
                    "error": str(e),
                    "problem_index": idx,
                }

            if completed % 20 == 0 or completed == len(failed_or_untrained):
                print(
                    f"✅ Retry progress: {completed}/{len(failed_or_untrained)} | "
                    f"Stored memories: {len(memory.experiences)}"
                )

    memory.save(args.output_file)
    save_json(retry_results, "retry_failed_results.json")

    elapsed = (time.time() - t0) / 60.0

    print("🎉 Retry completed")
    print(f"⏱️ Runtime: {elapsed:.2f} minutes")
    print(f"🧠 Final stored experiences: {len(memory.experiences)}")
    print(f"💾 Memory saved to: {args.output_file}")
    print("📄 Retry logs saved to: retry_failed_results.json")


if __name__ == "__main__":
    main()