# -*- coding: utf-8 -*-
"""
run_extra_baselines.py

Extra baselines for MAT paper:
- Chain-of-Thought CoT
- Zero-shot CoT
- Reflexion
- TSGD-M-style textual-gradient descent with momentum

This script is designed to be compatible with your existing project.
It tries to reuse mat.py utilities if available:
- load_problem_file
- filter_problems
- normalize_problem
- extract_predicted_answer
- is_correct

If not available, it falls back to robust local implementations.

Author: adapted for fast deadline experiments.
"""

import os
import re
import json
import time
import math
import random
import argparse
import hashlib
import string
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from openai import OpenAI

# ============================================================
# Optional import from original mat.py
# ============================================================

try:
    import mat
except Exception:
    mat = None


# ============================================================
# Environment
# ============================================================

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

MODEL_FORWARD = os.environ.get("MODEL_FORWARD", "deepseek-v4-flash")
MODEL_BACKWARD = os.environ.get("MODEL_BACKWARD", MODEL_FORWARD)

CHOICE_LETTERS = list(string.ascii_uppercase)


# ============================================================
# Basic utils
# ============================================================

def require_api_key():
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Please set DEEPSEEK_API_KEY first.")


def get_client():
    require_api_key()
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )


def chat_completion(
    messages: List[Dict[str, str]],
    model: str = MODEL_FORWARD,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    retries: int = 3,
    sleep: float = 2.0,
) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            client = get_client()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(sleep * (attempt + 1))
    raise last_err


def save_json(data: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(x: Any) -> str:
    x = "" if x is None else str(x)
    x = x.replace("\r\n", "\n").replace("\r", "\n").strip()
    x = re.sub(r"[ \t]+", " ", x)
    x = re.sub(r"\n{3,}", "\n\n", x)
    return x.strip()


def stable_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


# ============================================================
# Data loading / filtering
# ============================================================

def fallback_load_problem_file(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for k in ["data", "examples", "questions", "items", "records"]:
            if k in data and isinstance(data[k], list):
                return data[k]

    raise ValueError(f"Unsupported data format: {path}")


def load_problem_file(path: str) -> List[Dict[str, Any]]:
    if mat is not None and hasattr(mat, "load_problem_file"):
        return mat.load_problem_file(path)
    return fallback_load_problem_file(path)


def lower_set(csv: Optional[str]) -> Optional[set]:
    if not csv:
        return None
    return {x.strip().lower() for x in csv.split(",") if x.strip()}


def fallback_filter_problems(
    problems: List[Dict[str, Any]],
    include_subjects: Optional[str] = None,
    include_categories: Optional[str] = None,
    include_tasks: Optional[str] = None,
    limit: int = 0,
    shuffle: bool = False,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    subj_set = lower_set(include_subjects)
    cat_set = lower_set(include_categories)
    task_set = lower_set(include_tasks)

    out = []

    for p in problems:
        n = normalize_problem(p)
        meta = n.get("metadata", {}) or {}
        subject = str(meta.get("subject", "")).lower()
        category = str(meta.get("category", "")).lower()
        task = str(meta.get("task", "")).lower()

        if subj_set and subject not in subj_set:
            continue

        if cat_set:
            joined = f"{subject} {category} {task}".lower()
            if not any(c in joined for c in cat_set):
                continue

        if task_set and task not in task_set:
            continue

        out.append(p)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(out)

    if limit and limit > 0:
        out = out[:limit]

    return out


def filter_problems(
    problems: List[Dict[str, Any]],
    include_subjects: Optional[str] = None,
    include_categories: Optional[str] = None,
    include_tasks: Optional[str] = None,
    limit: int = 0,
    shuffle: bool = False,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    if mat is not None and hasattr(mat, "filter_problems"):
        try:
            return mat.filter_problems(
                problems,
                include_subjects=include_subjects,
                include_categories=include_categories,
                include_tasks=include_tasks,
                limit=limit,
                shuffle=shuffle,
                seed=seed,
            )
        except TypeError:
            pass

    return fallback_filter_problems(
        problems,
        include_subjects=include_subjects,
        include_categories=include_categories,
        include_tasks=include_tasks,
        limit=limit,
        shuffle=shuffle,
        seed=seed,
    )


# ============================================================
# Problem normalization
# ============================================================

def normalize_choices(raw_choices: Any) -> List[str]:
    if raw_choices is None:
        return []

    if isinstance(raw_choices, list):
        out = []
        for c in raw_choices:
            if isinstance(c, dict):
                out.append(str(c.get("text", c.get("content", c.get("choice", c)))))
            else:
                out.append(str(c))
        return out

    if isinstance(raw_choices, dict):
        ordered = []
        for letter in CHOICE_LETTERS:
            if letter in raw_choices:
                ordered.append(str(raw_choices[letter]))
            elif letter.lower() in raw_choices:
                ordered.append(str(raw_choices[letter.lower()]))
        if ordered:
            return ordered
        return [str(v) for _, v in sorted(raw_choices.items())]

    return []


def looks_numeric(s: str) -> bool:
    s = normalize_text(s).replace(",", "")
    return bool(re.fullmatch(r"[-+]?\d+(\.\d+)?", s))


def answer_to_letter(answer: Any, choices: List[str]) -> Optional[str]:
    if answer is None or not choices:
        return None

    n = len(choices)

    if isinstance(answer, int):
        if 0 <= answer < n:
            return CHOICE_LETTERS[answer]
        if 1 <= answer <= n:
            return CHOICE_LETTERS[answer - 1]

    ans = str(answer).strip()

    if len(ans) == 1 and ans.upper() in CHOICE_LETTERS[:n]:
        return ans.upper()

    if re.fullmatch(r"\d+", ans):
        idx = int(ans)
        if 0 <= idx < n:
            return CHOICE_LETTERS[idx]
        if 1 <= idx <= n:
            return CHOICE_LETTERS[idx - 1]

    ans_norm = normalize_text(ans).lower()
    for i, c in enumerate(choices):
        if normalize_text(c).lower() == ans_norm:
            return CHOICE_LETTERS[i]

    return None


def fallback_normalize_problem(raw: Dict[str, Any]) -> Dict[str, Any]:
    question = (
        raw.get("question")
        or raw.get("Question")
        or raw.get("input")
        or raw.get("prompt")
        or raw.get("query")
        or raw.get("problem")
        or raw.get("stem")
        or ""
    )

    choices = normalize_choices(
        raw.get("choices")
        or raw.get("options")
        or raw.get("Options")
        or raw.get("answer_choices")
        or raw.get("candidate_answers")
    )

    answer = (
        raw.get("answer")
        or raw.get("Answer")
        or raw.get("target")
        or raw.get("label")
        or raw.get("gold")
        or raw.get("gold_answer")
        or raw.get("correct_answer")
        or raw.get("Correct Answer")
        or raw.get("correct")
        or ""
    )

    # MMLU style: Correct Answer + Incorrect Answer 1/2/3
    correct_answer = raw.get("Correct Answer", raw.get("correct_answer", None))
    incorrects = []
    for k in [
        "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3",
        "incorrect_answer_1", "incorrect_answer_2", "incorrect_answer_3",
    ]:
        if raw.get(k):
            incorrects.append(str(raw[k]))

    if correct_answer is not None and incorrects and not choices:
        all_options = [str(correct_answer)] + incorrects
        rng = random.Random(int(stable_hash(str(question))[:8], 16))
        rng.shuffle(all_options)
        choices = all_options
        answer = answer_to_letter(str(correct_answer), choices)

    formatted = normalize_text(question)

    if choices:
        formatted += "\nChoices:\n"
        for i, c in enumerate(choices):
            formatted += f"{CHOICE_LETTERS[i]}. {normalize_text(c)}\n"
        answer_type = "mcq"
        letter = answer_to_letter(answer, choices)
        if letter is not None:
            answer = letter
    else:
        answer = normalize_text(answer)
        answer_type = "numeric" if looks_numeric(answer) else "exact"

    metadata = {
        "subject": str(raw.get("subject", raw.get("Subject", ""))),
        "category": str(raw.get("category", raw.get("Category", raw.get("domain", "")))),
        "task": str(raw.get("task", raw.get("Task", raw.get("task_name", raw.get("bbh_task", ""))))),
        "benchmark": str(raw.get("benchmark", raw.get("dataset", raw.get("source", "")))),
    }

    return {
        "question": formatted.strip(),
        "raw_question": normalize_text(question),
        "answer": normalize_text(answer),
        "answer_type": answer_type,
        "choices": choices,
        "metadata": metadata,
        "raw": raw,
    }


def normalize_problem(raw: Dict[str, Any]) -> Dict[str, Any]:
    if mat is not None and hasattr(mat, "normalize_problem"):
        try:
            return mat.normalize_problem(raw)
        except Exception:
            pass
    return fallback_normalize_problem(raw)


# ============================================================
# Answer extraction / scoring
# ============================================================

def fallback_extract_predicted_answer(
    solution_text: str,
    answer_type: str,
    choices: Optional[List[str]] = None,
) -> str:
    text = normalize_text(solution_text)
    choices = choices or []

    if answer_type == "mcq":
        patterns = [
            r"(?:final answer|answer|答案)\s*[:：]\s*([A-Z])\b",
            r"\boption\s*([A-Z])\b",
            r"\bchoice\s*([A-Z])\b",
            r"\b([A-Z])\b\s*$",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
            if m:
                ans = m.group(1).upper()
                if not choices or ans in CHOICE_LETTERS[:len(choices)]:
                    return ans

        for letter in CHOICE_LETTERS[:len(choices)]:
            if re.search(rf"\b{letter}\b", text):
                return letter

        return text[-1:].upper() if text else ""

    if answer_type == "numeric":
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
        return nums[-1] if nums else text

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    for line in reversed(lines):
        if re.search(r"(final answer|answer|答案)\s*[:：]", line, re.IGNORECASE):
            return re.sub(r"(?i)(final answer|answer|答案)\s*[:：]\s*", "", line).strip()
    return lines[-1] if lines else text


def extract_predicted_answer(solution_text: str, answer_type: str, choices: Optional[List[str]] = None) -> str:
    if mat is not None and hasattr(mat, "extract_predicted_answer"):
        try:
            return mat.extract_predicted_answer(solution_text, answer_type, choices)
        except TypeError:
            try:
                return mat.extract_predicted_answer(solution_text, answer_type)
            except Exception:
                pass
        except Exception:
            pass

    return fallback_extract_predicted_answer(solution_text, answer_type, choices)


def norm_num(s: str) -> str:
    s = normalize_text(s).replace(",", "")
    try:
        v = float(s)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return str(v)
    except Exception:
        return s.lower()


def fallback_is_correct(pred: str, gold: str, answer_type: str) -> bool:
    pred = normalize_text(pred)
    gold = normalize_text(gold)

    if answer_type == "mcq":
        return pred.upper() == gold.upper()

    if answer_type == "numeric":
        return norm_num(pred) == norm_num(gold)

    return pred.lower() == gold.lower()


def is_correct(pred: str, gold: str, answer_type: str) -> bool:
    if mat is not None and hasattr(mat, "is_correct"):
        try:
            return bool(mat.is_correct(pred, gold, answer_type))
        except Exception:
            pass
    return fallback_is_correct(pred, gold, answer_type)


# ============================================================
# Prompt templates
# ============================================================

BASE_SYSTEM = "You are a careful reasoning assistant. Solve the task accurately."

PROMPTS = {
    "cot": """
You are a careful reasoning assistant.

Solve the following problem step by step.
Then end with a final answer line.

Problem:
{question}

Requirements:
- Think step by step.
- If it is multiple-choice, output exactly one option letter.
- End with: Final Answer: <answer>
""".strip(),

    "zero_shot_cot": """
Solve the following problem.

Problem:
{question}

Let's think step by step.

Requirements:
- Provide concise reasoning.
- If it is multiple-choice, output exactly one option letter.
- End with: Final Answer: <answer>
""".strip(),

    "direct": """
Solve the following problem.

Problem:
{question}

Requirements:
- Be concise.
- If it is multiple-choice, output exactly one option letter.
- End with: Final Answer: <answer>
""".strip(),
}


def build_prompt(problem: Dict[str, Any], prompt_text: str) -> str:
    n = normalize_problem(problem)
    result = prompt_text
    result = result.replace("{question}", n["question"])
    result = result.replace("{answer_type}", n["answer_type"])
    return result


# ============================================================
# Solvers
# ============================================================

def solve_with_prompt(
    problem: Dict[str, Any],
    prompt_text: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    n = normalize_problem(problem)
    t0 = time.time()

    user_prompt = build_prompt(problem, prompt_text)

    output = chat_completion(
        messages=[
            {"role": "system", "content": BASE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    pred = extract_predicted_answer(output, n["answer_type"], n.get("choices", []))
    success = is_correct(pred, n["answer"], n["answer_type"])

    return {
        "success": success,
        "question": n["question"],
        "ground_truth": n["answer"],
        "predicted_answer": pred,
        "answer_type": n["answer_type"],
        "choices": n.get("choices", []),
        "metadata": n.get("metadata", {}),
        "final_solution": output,
        "num_iterations": 0,
        "api_calls": 1,
        "time": round(time.time() - t0, 4),
    }


def solve_reflexion(
    problem: Dict[str, Any],
    model: str,
    temperature: float,
    max_tokens: int,
    max_reflections: int = 2,
) -> Dict[str, Any]:
    """
    Test-time Reflexion-style baseline:
    - initial solution
    - self-critique without gold label
    - revise
    """
    n = normalize_problem(problem)
    t0 = time.time()
    api_calls = 0

    init_prompt = PROMPTS["zero_shot_cot"].format(question=n["question"], answer_type=n["answer_type"])

    solution = chat_completion(
        messages=[
            {"role": "system", "content": BASE_SYSTEM},
            {"role": "user", "content": init_prompt},
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    api_calls += 1

    reflections = []

    for i in range(max_reflections):
        critique_prompt = f"""
You are a self-reflection module.

Problem:
{n['question']}

Current solution:
{solution}

Task:
Critically examine the solution without seeing the gold answer.
Identify possible mistakes, overlooked constraints, arithmetic errors, distractor traps,
or answer-format issues.

Return concise reflection only.
""".strip()

        reflection = chat_completion(
            messages=[
                {"role": "system", "content": "You are a strict reasoning critic."},
                {"role": "user", "content": critique_prompt},
            ],
            model=model,
            temperature=0.0,
            max_tokens=1024,
        )
        api_calls += 1
        reflections.append(reflection)

        revise_prompt = f"""
Revise the solution using the reflection.

Problem:
{n['question']}

Current solution:
{solution}

Reflection:
{reflection}

Requirements:
- Produce an improved final solution.
- If multiple-choice, output exactly one option letter.
- End with: Final Answer: <answer>
""".strip()

        solution = chat_completion(
            messages=[
                {"role": "system", "content": BASE_SYSTEM},
                {"role": "user", "content": revise_prompt},
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        api_calls += 1

    pred = extract_predicted_answer(solution, n["answer_type"], n.get("choices", []))
    success = is_correct(pred, n["answer"], n["answer_type"])

    return {
        "success": success,
        "question": n["question"],
        "ground_truth": n["answer"],
        "predicted_answer": pred,
        "answer_type": n["answer_type"],
        "choices": n.get("choices", []),
        "metadata": n.get("metadata", {}),
        "initial_solution": "",
        "final_solution": solution,
        "reflections": reflections,
        "num_iterations": max_reflections,
        "api_calls": api_calls,
        "time": round(time.time() - t0, 4),
    }


# ============================================================
# Prompt optimization baselines
# ============================================================

DEFAULT_OPT_PROMPT = """
You are a careful reasoning assistant.

Solve the problem accurately. Use concise reasoning. Pay attention to all constraints,
distractor options, units, and final answer format.

Problem:
{question}

End with: Final Answer: <answer>
""".strip()


def evaluate_prompt_on_dev(
    prompt_text: str,
    dev_data: List[Dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int,
    threads: int,
) -> Tuple[float, List[Dict[str, Any]]]:
    results = [None] * len(dev_data)

    def task(i, p):
        return i, solve_with_prompt(p, prompt_text, model, temperature, max_tokens)

    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = {ex.submit(task, i, p): i for i, p in enumerate(dev_data)}
        for fut in as_completed(futures):
            i, r = fut.result()
            results[i] = r

    acc = sum(1 for r in results if r and r.get("success")) / max(1, len(results))
    return acc, results


def summarize_errors(dev_results: List[Dict[str, Any]], max_errors: int = 8) -> str:
    errors = [r for r in dev_results if r and not r.get("success")]
    errors = errors[:max_errors]

    parts = []
    for i, r in enumerate(errors, 1):
        parts.append(
            f"""
Error #{i}
Question:
{r.get('question', '')[:1000]}

Gold answer: {r.get('ground_truth', '')}
Predicted answer: {r.get('predicted_answer', '')}

Model solution:
{r.get('final_solution', '')[:1200]}
""".strip()
        )

    return "\n\n".join(parts) if parts else "No errors."


def tsgdm_update_prompt(
    current_prompt: str,
    dev_acc: float,
    dev_results: List[Dict[str, Any]],
    momentum_memory: List[str],
    model: str,
) -> Tuple[str, str]:
    error_summary = summarize_errors(dev_results)
    momentum_text = "\n".join(momentum_memory[-3:]) if momentum_memory else "(none)"

    gradient_prompt = f"""
You are a textual-gradient generator with momentum.

Current prompt:
{current_prompt}

Current dev accuracy:
{dev_acc:.4f}

Recent momentum gradients:
{momentum_text}

Failure cases:
{error_summary}

Task:
Generate an updated textual gradient. It should be consistent with useful previous gradients
but correct any stale or harmful directions.
""".strip()

    gradient = chat_completion(
        messages=[
            {"role": "system", "content": "You generate momentum textual gradients for prompt optimization."},
            {"role": "user", "content": gradient_prompt},
        ],
        model=model,
        temperature=0.0,
        max_tokens=1024,
    )

    combined_momentum = "\n".join((momentum_memory + [gradient])[-4:])

    update_prompt = f"""
Update the prompt using textual gradient descent with momentum.

Current prompt:
{current_prompt}

Momentum gradient history:
{combined_momentum}

Requirements:
- Keep the prompt concise.
- It must contain the literal placeholder {{question}}.
- It must require final output line: Final Answer: <answer>.
- Return only the improved prompt.
""".strip()

    new_prompt = chat_completion(
        messages=[
            {"role": "system", "content": "You are a prompt editor using gradient descent with momentum."},
            {"role": "user", "content": update_prompt},
        ],
        model=model,
        temperature=0.3,
        max_tokens=2048,
    ).strip()

    if "{question}" not in new_prompt:
        new_prompt = current_prompt

    return new_prompt, gradient


def optimize_prompt(
    method: str,
    train_data: List[Dict[str, Any]],
    args,
) -> Dict[str, Any]:
    """
    Optimize prompt for TSGD-M-style.

    Returns:
    {
      "best_prompt": str,
      "best_acc": float,
      "history": list
    }
    """
    rng = random.Random(args.seed)

    data = list(train_data)
    rng.shuffle(data)

    dev_size = min(args.dev_size, len(data))
    dev_data = data[:dev_size]

    current_prompt = DEFAULT_OPT_PROMPT
    best_prompt = current_prompt
    best_acc = -1.0
    history = []
    momentum_memory = []

    print(f"🧪 Optimizing prompt with {method} on dev size={len(dev_data)}")

    for step in range(args.optimize_steps):
        print(f"\n🔧 Optimization step {step + 1}/{args.optimize_steps} | method={method}")

        if method == "tsgd_m":
            cur_acc, cur_results = evaluate_prompt_on_dev(
                current_prompt,
                dev_data,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                threads=args.threads,
            )

            if cur_acc > best_acc:
                best_acc = cur_acc
                best_prompt = current_prompt

            new_prompt, gradient = tsgdm_update_prompt(
                current_prompt=current_prompt,
                dev_acc=cur_acc,
                dev_results=cur_results,
                momentum_memory=momentum_memory,
                model=args.model,
            )

            momentum_memory.append(gradient)

            new_acc, _ = evaluate_prompt_on_dev(
                new_prompt,
                dev_data,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                threads=args.threads,
            )

            if new_acc >= cur_acc:
                current_prompt = new_prompt

            if new_acc > best_acc:
                best_acc = new_acc
                best_prompt = new_prompt

            history.append({
                "step": step + 1,
                "current_acc": cur_acc,
                "new_acc": new_acc,
                "best_acc": best_acc,
                "gradient": gradient,
                "momentum_size": len(momentum_memory),
            })

            print(f"  📊 cur_acc={cur_acc:.4f} | new_acc={new_acc:.4f} | best_acc={best_acc:.4f}")

        else:
            raise ValueError(f"Unknown optimization method: {method}")

    return {
        "method": method,
        "best_prompt": best_prompt,
        "best_acc": best_acc,
        "history": history,
    }

# ============================================================
# Evaluation
# ============================================================

def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "accuracy": 0.0,
            "avg_iterations": 0.0,
            "avg_api_calls": 0.0,
            "avg_time": 0.0,
            "total": 0,
            "correct": 0,
        }

    total = len(results)
    correct = sum(1 for r in results if r.get("success"))

    api_calls = [r.get("api_calls", 0) for r in results if isinstance(r.get("api_calls", 0), (int, float))]
    times = [r.get("time", 0.0) for r in results]
    iters = [r.get("num_iterations", 0) for r in results]

    return {
        "accuracy": round(100.0 * correct / total, 2),
        "avg_iterations": round(float(np.mean(iters)), 3),
        "avg_api_calls": round(float(np.mean(api_calls)), 3) if api_calls else None,
        "avg_time": round(float(np.mean(times)), 3),
        "total": total,
        "correct": correct,
    }


def run_parallel_eval(
    method: str,
    test_data: List[Dict[str, Any]],
    args,
    learned_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    results = [None] * len(test_data)

    def task(i, p):
        if method == "cot":
            r = solve_with_prompt(
                p,
                PROMPTS["cot"],
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        elif method == "zero_shot_cot":
            r = solve_with_prompt(
                p,
                PROMPTS["zero_shot_cot"],
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        elif method == "direct":
            r = solve_with_prompt(
                p,
                PROMPTS["direct"],
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        elif method == "reflexion":
            r = solve_reflexion(
                p,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_reflections=args.reflexion_iters,
            )
        elif method in ["tsgd_m"]:
            assert learned_prompt is not None
            r = solve_with_prompt(
                p,
                learned_prompt,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

        r["method"] = method
        return i, r

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = {ex.submit(task, i, p): i for i, p in enumerate(test_data)}

        completed = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            completed += 1
            try:
                i, r = fut.result()
                results[i] = r
            except Exception as e:
                results[idx] = {
                    "method": method,
                    "success": False,
                    "error": repr(e),
                    "problem_index": idx,
                    "api_calls": 0,
                    "time": 0,
                    "num_iterations": 0,
                }

            if completed % 25 == 0 or completed == len(test_data):
                print(f"✅ {method} progress: {completed}/{len(test_data)}")

    return results


# ============================================================
# Main
# ============================================================

def parse_methods(s: str) -> List[str]:
    if s.lower() == "all":
        return ["cot", "zero_shot_cot", "reflexion", "tsgd_m"]
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--benchmark", default="mixed", choices=["mixed", "mmlu", "bbh"])
    parser.add_argument("--train_file", default="")
    parser.add_argument("--test_file", required=True)

    parser.add_argument("--methods", default="cot,zero_shot_cot,reflexion,tsgd_m")
    parser.add_argument("--output_file", default="results_extra_baselines.json")
    parser.add_argument("--summary_output_file", default="summary_extra_baselines.json")
    parser.add_argument("--prompt_cache_file", default="optimized_prompts_extra_baselines.json")

    parser.add_argument("--include_subjects", default=None)
    parser.add_argument("--include_categories", default=None)
    parser.add_argument("--include_tasks", default=None)

    parser.add_argument("--test_limit", type=int, default=200)
    parser.add_argument("--train_limit", type=int, default=300)
    parser.add_argument("--shuffle_test", action="store_true")
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--model", default=MODEL_FORWARD)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2048)

    # Reflexion
    parser.add_argument("--reflexion_iters", type=int, default=2)

    # Prompt optimization
    parser.add_argument("--dev_size", type=int, default=40)
    parser.add_argument("--optimize_steps", type=int, default=3)


    args = parser.parse_args()

    methods = parse_methods(args.methods)

    print("🚀 Running extra baselines")
    print(f"📌 Benchmark: {args.benchmark}")
    print(f"📌 Methods: {methods}")
    print(f"🧵 Threads: {args.threads}")
    print(f"🤖 Model: {args.model}")

    raw_test = load_problem_file(args.test_file)
    test_data = filter_problems(
        raw_test,
        include_subjects=args.include_subjects,
        include_categories=args.include_categories,
        include_tasks=args.include_tasks,
        limit=args.test_limit,
        shuffle=args.shuffle_test,
        seed=args.seed,
    )

    print(f"📚 Raw test: {len(raw_test)} | selected test: {len(test_data)}")

    train_data = []
    if args.train_file:
        raw_train = load_problem_file(args.train_file)
        train_data = filter_problems(
            raw_train,
            include_subjects=args.include_subjects,
            include_categories=args.include_categories,
            include_tasks=args.include_tasks,
            limit=args.train_limit,
            shuffle=args.shuffle_train,
            seed=args.seed,
        )
        print(f"📚 Raw train: {len(raw_train)} | selected train: {len(train_data)}")

    all_results = {}
    summary = {
        "benchmark": args.benchmark,
        "methods": methods,
        "test_total": len(test_data),
        "metrics": {},
        "optimized_prompts": {},
    }

    optimized_prompts = {}
    if args.prompt_cache_file and os.path.exists(args.prompt_cache_file):
        try:
            optimized_prompts = load_json(args.prompt_cache_file)
            print(f"📦 Loaded optimized prompt cache: {args.prompt_cache_file}")
        except Exception:
            optimized_prompts = {}

    for method in methods:
        print("\n" + "=" * 90)
        print(f"🚀 Method: {method}")
        print("=" * 90)

        t0 = time.time()

        if method in ["cot", "zero_shot_cot", "direct", "reflexion"]:
            results = run_parallel_eval(method, test_data, args)
            m = compute_metrics(results)

            all_results[method] = {
                "metrics": m,
                "results": results,
            }
            summary["metrics"][method] = m

            print(f"📊 {method}: {m}")

        elif method in ["tsgd_m"]:
            if not train_data:
                print(f"⚠️ Skip {method}: --train_file is required.")
                all_results[method] = {
                    "error": "--train_file is required for prompt optimization baselines.",
                    "metrics": None,
                    "results": [],
                }
                summary["metrics"][method] = None
                continue

            if method in optimized_prompts:
                print(f"📦 Using cached optimized prompt for {method}")
                best_prompt = optimized_prompts[method]["best_prompt"]
                opt_info = optimized_prompts[method]
            else:
                opt_info = optimize_prompt(method, train_data, args)
                best_prompt = opt_info["best_prompt"]
                optimized_prompts[method] = opt_info
                save_json(optimized_prompts, args.prompt_cache_file)

            results = run_parallel_eval(method, test_data, args, learned_prompt=best_prompt)
            m = compute_metrics(results)

            all_results[method] = {
                "optimization": opt_info,
                "metrics": m,
                "results": results,
            }
            summary["metrics"][method] = m
            summary["optimized_prompts"][method] = {
                "best_acc": opt_info.get("best_acc"),
                "best_prompt": best_prompt,
            }

            print(f"📊 {method}: {m}")


        else:
            print(f"⚠️ Unknown method skipped: {method}")
            continue

        print(f"⏱️ Method runtime: {(time.time() - t0) / 60:.2f} minutes")

        save_json(all_results, args.output_file)
        save_json(summary, args.summary_output_file)

    save_json(all_results, args.output_file)
    save_json(summary, args.summary_output_file)

    print("\n🎉 Extra baselines completed")
    print(f"💾 Full results: {args.output_file}")
    print(f"💾 Summary: {args.summary_output_file}")
    print(f"💾 Prompt cache: {args.prompt_cache_file}")


if __name__ == "__main__":
    main()