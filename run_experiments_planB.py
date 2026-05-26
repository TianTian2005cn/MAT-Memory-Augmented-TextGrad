"""
MAT full experiment pipeline for BBH / MMLU / GSM8K.

Experiments:
1. Main experiment:
   - Vanilla TextGrad vs Full MAT.

2. Ablation study:
   - Vanilla TextGrad
   - MAT without retrieval
   - MAT without adaptive iteration
   - MAT without gradient-level memory injection
   - Full MAT

3. Similarity threshold sensitivity.

4. Per-category analysis:
   - BBH: by task
   - MMLU: by subject

Examples:

BBH:
python run_experiments_planB.py \
  --benchmark bbh \
  --train_file bbh_train.json \
  --test_file bbh_test.json \
  --memory_file memory_bbh.json

MMLU :
python run_experiments_planB.py \
  --benchmark mmlu \
  --train_file mmlu_train.json \
  --test_file mmlu_test.json \
  --memory_file memory_mmlu_physics.json \
  --include_subjects high_school_physics,college_physics

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
    infer_problem_type,
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
        import json
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    print("🧪 Building shared initial solution cache...")
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
        print(f"💾 Initial cache saved to: {cache_file}")

    return cache


def parallel_run(
    test_data,
    method,
    memory,
    args,
    initial_cache=None,
    sim_threshold=None,
    use_retrieval=True,
    use_adaptive_iter=True,
    use_gradient_injection=True,
):
    results = [None] * len(test_data)
    threshold = args.sim_threshold if sim_threshold is None else sim_threshold

    def task(i, problem):
        return run_single_problem(
            problem=problem,
            method=method,
            memory=memory,
            max_iterations=args.max_iterations,
            is_training=False,
            initial_solution=initial_cache[i] if initial_cache is not None else None,
            sim_threshold=threshold,
            top_k_experiences=args.top_k,
            use_retrieval=use_retrieval,
            use_adaptive_iter=use_adaptive_iter,
            use_gradient_injection=use_gradient_injection,
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


def train_memory_if_needed(train_data, args):
    memory = ExperienceMemory(
        capacity=args.capacity,
        similarity_threshold=args.sim_threshold,
    )

    if os.path.exists(args.memory_file):
        print(f"📦 Loading existing memory: {args.memory_file}")
        memory.load(args.memory_file)
        print(f"🧠 Loaded memories: {len(memory.experiences)}")
        return memory

    print("🧠 No memory file found. Starting sequential memory training...")
    print("⚠️ For large-scale training, train_memory_parallel.py is recommended.")

    for i, p in enumerate(train_data):
        try:
            run_single_problem(
                problem=p,
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
        except Exception as e:
            print(f"⚠️ Training example {i} failed: {e}")

        if (i + 1) % 50 == 0 or (i + 1) == len(train_data):
            print(
                f"✅ Training progress: {i + 1}/{len(train_data)} | "
                f"Stored memories: {len(memory.experiences)}"
            )

    memory.save(args.memory_file)
    print(f"💾 Memory saved to: {args.memory_file}")

    return memory


def exp1_main(train_data, test_data, args, initial_cache):
    print("\n" + "=" * 80)
    print("🧪 Experiment 1: Main comparison")
    print("=" * 80)

    memory = train_memory_if_needed(train_data, args)

    print("\n🚀 Running Vanilla TextGrad...")
    vanilla = parallel_run(
        test_data=test_data,
        method="vanilla",
        memory=None,
        args=args,
        initial_cache=initial_cache,
    )
    vanilla_m = metrics(vanilla)
    print(f"📊 Vanilla metrics: {vanilla_m}")

    print("\n🚀 Running Full MAT...")
    mat = parallel_run(
        test_data=test_data,
        method="mat",
        memory=memory,
        args=args,
        initial_cache=initial_cache,
        use_retrieval=True,
        use_adaptive_iter=True,
        use_gradient_injection=True,
    )
    mat_m = metrics(mat)
    print(f"📊 MAT metrics: {mat_m}")

    results = {
        "benchmark": args.benchmark,
        "vanilla": vanilla,
        "mat": mat,
        "summary": {
            "vanilla": vanilla_m,
            "mat": mat_m,
            "delta_accuracy": round(mat_m["accuracy"] - vanilla_m["accuracy"], 2),
            "delta_avg_iterations": round(
                mat_m["avg_iterations"] - vanilla_m["avg_iterations"],
                3,
            ),
            "delta_avg_api_calls": round(
                mat_m["avg_api_calls"] - vanilla_m["avg_api_calls"],
                3,
            ),
        },
    }

    save_json(results, args.main_output_file)
    print(f"💾 Saved: {args.main_output_file}")

    return results, memory


def exp2_ablation(test_data, memory, args, initial_cache):
    print("\n" + "=" * 80)
    print("🧪 Experiment 2: Component ablation")
    print("=" * 80)

    subset = test_data[:args.ablation_size]
    subset_cache = initial_cache[:args.ablation_size] if initial_cache else None

    configs = [
        {
            "name": "Vanilla TextGrad",
            "method": "vanilla",
            "use_retrieval": False,
            "use_adaptive_iter": False,
            "use_gradient_injection": False,
        },
        {
            "name": "MAT w/o retrieval",
            "method": "mat",
            "use_retrieval": False,
            "use_adaptive_iter": True,
            "use_gradient_injection": True,
        },
        {
            "name": "MAT w/o adaptive iteration",
            "method": "mat",
            "use_retrieval": True,
            "use_adaptive_iter": False,
            "use_gradient_injection": True,
        },
        {
            "name": "MAT w/o gradient-level injection",
            "method": "mat",
            "use_retrieval": True,
            "use_adaptive_iter": True,
            "use_gradient_injection": False,
        },
        {
            "name": "Full MAT",
            "method": "mat",
            "use_retrieval": True,
            "use_adaptive_iter": True,
            "use_gradient_injection": True,
        },
    ]

    all_results = {}

    for cfg in configs:
        print(f"\n🚀 Running ablation: {cfg['name']}")

        res = parallel_run(
            test_data=subset,
            method=cfg["method"],
            memory=memory if cfg["method"] == "mat" else None,
            args=args,
            initial_cache=subset_cache,
            use_retrieval=cfg["use_retrieval"],
            use_adaptive_iter=cfg["use_adaptive_iter"],
            use_gradient_injection=cfg["use_gradient_injection"],
        )

        m = metrics(res)

        all_results[cfg["name"]] = {
            "metrics": m,
            "results": res,
        }

        print(f"📊 {cfg['name']}: {m}")

    compact = {k: v["metrics"] for k, v in all_results.items()}

    save_json(compact, args.ablation_output_file)
    save_json(all_results, args.ablation_full_output_file)

    print(f"💾 Saved: {args.ablation_output_file}")
    print(f"💾 Saved: {args.ablation_full_output_file}")

    return all_results


def exp3_threshold(test_data, memory_file, args, initial_cache):
    print("\n" + "=" * 80)
    print("🧪 Experiment 3: Similarity threshold sensitivity")
    print("=" * 80)

    subset = test_data[:args.threshold_size]
    subset_cache = initial_cache[:args.threshold_size] if initial_cache else None

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    output = {}

    for th in thresholds:
        print(f"\n🚀 Running threshold θ={th}")

        mem = ExperienceMemory(
            capacity=args.capacity,
            similarity_threshold=th,
        )
        mem.load(memory_file)
        mem.similarity_threshold = th

        res = parallel_run(
            test_data=subset,
            method="mat",
            memory=mem,
            args=args,
            initial_cache=subset_cache,
            sim_threshold=th,
            use_retrieval=True,
            use_adaptive_iter=True,
            use_gradient_injection=True,
        )

        m = metrics(res)

        output[str(th)] = {
            "metrics": m,
            "results": res,
        }

        print(f"📊 θ={th}: {m}")

    compact = {k: v["metrics"] for k, v in output.items()}

    save_json(compact, args.threshold_output_file)
    save_json(output, args.threshold_full_output_file)

    print(f"💾 Saved: {args.threshold_output_file}")
    print(f"💾 Saved: {args.threshold_full_output_file}")

    return output


def category_key(result, benchmark):
    metadata = result.get("metadata", {}) or {}

    if benchmark == "bbh":
        return metadata.get("task") or result.get("problem_type") or "unknown_task"

    if benchmark == "mmlu":
        return metadata.get("subject") or result.get("problem_type") or "unknown_subject"

    q = result.get("question", "")
    return infer_problem_type(q, metadata)


def exp4_category(main_results, args):
    print("\n" + "=" * 80)
    print("🧪 Experiment 4: Per-category analysis")
    print("=" * 80)

    cats = {}

    for method in ["vanilla", "mat"]:
        for r in main_results[method]:
            key = category_key(r, args.benchmark)
            cats.setdefault(key, {"vanilla": [], "mat": []})
            cats[key][method].append(r)

    output = {}

    for key, data in cats.items():
        vanilla_m = metrics(data["vanilla"])
        mat_m = metrics(data["mat"])

        output[key] = {
            "vanilla": vanilla_m,
            "mat": mat_m,
            "delta_accuracy": round(
                mat_m["accuracy"] - vanilla_m["accuracy"],
                2,
            ),
            "delta_avg_iterations": round(
                mat_m["avg_iterations"] - vanilla_m["avg_iterations"],
                3,
            ),
            "delta_avg_api_calls": round(
                mat_m["avg_api_calls"] - vanilla_m["avg_api_calls"],
                3,
            ),
        }

    save_json(output, args.category_output_file)
    print(f"💾 Saved: {args.category_output_file}")

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--benchmark", default="mixed", choices=["mixed", "bbh", "mmlu"])

    parser.add_argument("--train_file", default="train.json")
    parser.add_argument("--test_file", default="test.json")
    parser.add_argument("--memory_file", default="memory_after_training.json")
    parser.add_argument("--initial_cache_file", default="initial_solutions_test.json")

    parser.add_argument("--main_output_file", default="results_main.json")
    parser.add_argument("--ablation_output_file", default="results_ablation.json")
    parser.add_argument("--ablation_full_output_file", default="results_ablation_full.json")
    parser.add_argument("--threshold_output_file", default="results_threshold.json")
    parser.add_argument("--threshold_full_output_file", default="results_threshold_full.json")
    parser.add_argument("--category_output_file", default="results_per_category.json")
    parser.add_argument("--summary_output_file", default="all_results_summary.json")

    parser.add_argument("--capacity", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max_iterations", type=int, default=3)
    parser.add_argument("--sim_threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--initial_temperature", type=float, default=0.0)

    parser.add_argument("--include_subjects", default=None)
    parser.add_argument("--include_categories", default=None)
    parser.add_argument("--include_tasks", default=None)

    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--test_limit", type=int, default=0)
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--shuffle_test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--ablation_size", type=int, default=200)
    parser.add_argument("--threshold_size", type=int, default=200)
    parser.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7")

    args = parser.parse_args()

    setup_textgrad_with_deepseek()

    raw_train = load_problem_file(args.train_file)
    raw_test = load_problem_file(args.test_file)

    train_data = filter_problems(
        raw_train,
        include_subjects=args.include_subjects,
        include_categories=args.include_categories,
        include_tasks=args.include_tasks,
        limit=args.train_limit,
        shuffle=args.shuffle_train,
        seed=args.seed,
    )

    test_data = filter_problems(
        raw_test,
        include_subjects=args.include_subjects,
        include_categories=args.include_categories,
        include_tasks=args.include_tasks,
        limit=args.test_limit,
        shuffle=args.shuffle_test,
        seed=args.seed,
    )

    print("🚀 MAT full experiment pipeline")
    print(f"📌 Benchmark: {args.benchmark}")
    print(f"📚 Raw train problems: {len(raw_train)}")
    print(f"📚 Selected train problems: {len(train_data)}")
    print(f"📚 Raw test problems: {len(raw_test)}")
    print(f"📚 Selected test problems: {len(test_data)}")

    initial_cache = build_or_load_initial_cache(
        test_data=test_data,
        cache_file=args.initial_cache_file,
        threads=args.threads,
        temperature=args.initial_temperature,
    )

    t0 = time.time()

    main_results, memory = exp1_main(
        train_data=train_data,
        test_data=test_data,
        args=args,
        initial_cache=initial_cache,
    )

    ablation = exp2_ablation(
        test_data=test_data,
        memory=memory,
        args=args,
        initial_cache=initial_cache,
    )

    threshold = exp3_threshold(
        test_data=test_data,
        memory_file=args.memory_file,
        args=args,
        initial_cache=initial_cache,
    )

    category = exp4_category(main_results, args)

    summary = {
        "benchmark": args.benchmark,
        "main": main_results["summary"],
        "ablation": {k: v["metrics"] for k, v in ablation.items()},
        "threshold": {k: v["metrics"] for k, v in threshold.items()},
        "category": category,
        "runtime_minutes": round((time.time() - t0) / 60.0, 2),
    }

    save_json(summary, args.summary_output_file)

    print(f"💾 Saved: {args.summary_output_file}")
    print("\n✅ All experiments completed successfully!")


if __name__ == "__main__":
    main()