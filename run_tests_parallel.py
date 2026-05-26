"""
Parallel testing script for Vanilla TextGrad and MAT on BBH / MMLU .

This script supports fair evaluation by caching initial solutions.
Both Vanilla and MAT can use exactly the same initial solution for each problem.

Examples:

BBH:
python run_tests_parallel.py \
  --test_file bbh_test.json \
  --memory_file memory_bbh.json \
  --output_file results_bbh.json

MMLU math:
python run_tests_parallel.py \
  --test_file mmlu_test.json \
  --memory_file memory_mmlu_math.json \
  --output_file results_mmlu_math.json

"""

import os
import time
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from mat import (
    setup_textgrad_with_deepseek,
    ExperienceMemory,
    run_single_problem,
    generate_initial_solution,
    load_problem_file,
    filter_problems,
    save_json,
    SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def metrics(results):
    if not results:
        return {
            "accuracy": 0.0,
            "avg_iterations": 0.0,
            "avg_api_calls": 0.0,
            "avg_time": 0.0,
            "retrieval_rate": 0.0,
            "total": 0,
            "correct": 0,
        }

    total = len(results)
    correct = sum(1 for r in results if r.get("success"))
    retrieval_hits = sum(1 for r in results if r.get("retrieved_count", 0) > 0)

    return {
        "accuracy": round(100.0 * correct / total, 2),
        "avg_iterations": round(float(np.mean([r.get("num_iterations", 0) for r in results])), 3),
        "avg_api_calls": round(float(np.mean([r.get("api_calls", 0) for r in results])), 3),
        "avg_time": round(float(np.mean([r.get("time", 0.0) for r in results])), 3),
        "retrieval_rate": round(100.0 * retrieval_hits / total, 2),
        "total": total,
        "correct": correct,
    }


def build_or_load_initial_cache(test_data, cache_file, threads, temperature):
    if cache_file and os.path.exists(cache_file):
        print(f"📦 Loading initial solution cache: {cache_file}")
        from mat import load_problem_file as _unused  # keep import style stable
        import json
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    print("🧪 Building initial solution cache for fair comparison...")
    cache = [None] * len(test_data)

    def task(i, problem):
        return i, generate_initial_solution(
            problem,
            temperature=temperature,
        )

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {
            executor.submit(task, i, p): i
            for i, p in enumerate(test_data)
        }

        completed = 0

        for future in as_completed(futures):
            completed += 1
            i, sol = future.result()
            cache[i] = sol

            if completed % 50 == 0 or completed == len(test_data):
                print(f"✅ Initial cache progress: {completed}/{len(test_data)}")

    if cache_file:
        save_json(cache, cache_file)
        print(f"💾 Initial solution cache saved to: {cache_file}")

    return cache


def run_parallel_test(
    test_data,
    method,
    memory,
    args,
    initial_cache=None,
):
    results = [None] * len(test_data)

    def task(i, problem):
        init_sol = initial_cache[i] if initial_cache is not None else None

        return run_single_problem(
            problem=problem,
            method=method,
            memory=memory,
            max_iterations=args.max_iterations,
            is_training=False,
            initial_solution=init_sol,
            sim_threshold=args.sim_threshold,
            top_k_experiences=args.top_k,
            use_retrieval=args.use_retrieval,
            use_adaptive_iter=args.use_adaptive_iter,
            use_gradient_injection=args.use_gradient_injection,
            initial_temperature=args.initial_temperature,
        )

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(task, i, p): i
            for i, p in enumerate(test_data)
        }

        completed = 0

        for future in as_completed(futures):
            idx = futures[future]
            completed += 1

            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {
                    "method": method,
                    "success": False,
                    "error": str(e),
                    "problem_index": idx,
                }

            if completed % 50 == 0 or completed == len(test_data):
                print(f"✅ {method} progress: {completed}/{len(test_data)}")

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_file", default="test.json")
    parser.add_argument("--memory_file", default="memory_after_training.json")
    parser.add_argument("--output_file", default="results_main.json")
    parser.add_argument("--initial_cache_file", default="initial_solutions_test.json")
    parser.add_argument("--benchmark", default="mixed", choices=["mixed", "bbh", "mmlu"])

    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=3000)
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

    raw_test = load_problem_file(args.test_file)
    test_data = filter_problems(
        raw_test,
        include_subjects=args.include_subjects,
        include_categories=args.include_categories,
        include_tasks=args.include_tasks,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    print("🚀 Parallel benchmark testing")
    print(f"📌 Benchmark: {args.benchmark}")
    print(f"📚 Raw test problems: {len(raw_test)}")
    print(f"📚 Selected test problems: {len(test_data)}")

    initial_cache = build_or_load_initial_cache(
        test_data=test_data,
        cache_file=args.initial_cache_file,
        threads=args.threads,
        temperature=args.initial_temperature,
    )

    memory = ExperienceMemory(
        capacity=args.capacity,
        similarity_threshold=args.sim_threshold,
    )
    memory.load(args.memory_file)

    print(f"🧠 Loaded memories: {len(memory.experiences)}")

    print("\n🚀 Testing Vanilla TextGrad...")
    t0 = time.time()

    vanilla_results = run_parallel_test(
        test_data=test_data,
        method="vanilla",
        memory=None,
        args=args,
        initial_cache=initial_cache,
    )

    vanilla_metrics = metrics(vanilla_results)

    print(f"📊 Vanilla: {vanilla_metrics}")
    print(f"⏱️ Vanilla runtime: {time.time() - t0:.1f}s")

    print("\n🚀 Testing MAT...")
    t0 = time.time()

    mat_results = run_parallel_test(
        test_data=test_data,
        method="mat",
        memory=memory,
        args=args,
        initial_cache=initial_cache,
    )

    mat_metrics = metrics(mat_results)

    print(f"📊 MAT: {mat_metrics}")
    print(f"⏱️ MAT runtime: {time.time() - t0:.1f}s")

    output = {
        "benchmark": args.benchmark,
        "vanilla": vanilla_results,
        "mat": mat_results,
        "summary": {
            "vanilla": vanilla_metrics,
            "mat": mat_metrics,
            "delta_accuracy": round(
                mat_metrics["accuracy"] - vanilla_metrics["accuracy"],
                2,
            ),
            "delta_avg_iterations": round(
                mat_metrics["avg_iterations"] - vanilla_metrics["avg_iterations"],
                3,
            ),
            "delta_avg_api_calls": round(
                mat_metrics["avg_api_calls"] - vanilla_metrics["avg_api_calls"],
                3,
            ),
        },
    }

    save_json(output, args.output_file)
    print(f"💾 Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()