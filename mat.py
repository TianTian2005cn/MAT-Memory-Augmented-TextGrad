"""
Memory-Augmented TextGrad (MAT) for BBH / MMLU 
=====================================================

This module adapts MAT from GSM8K-style numeric math problems to broader
reasoning and knowledge benchmarks, including:

- BBH / Big Bench Hard:
  Usually free-form or exact-match tasks with fields such as input/target.

- MMLU:
  Multiple-choice academic questions with fields such as question, choices,
  answer, subject.


Core MAT idea:
- First run native TextGrad feedback on the current task.
- Retrieve similar successful optimization trajectories from long-term memory.
- Inject retrieved memory as an additional TextGrad variable into the gradient
  set. This is gradient-level memory injection, not simple prompt concatenation.
"""

import os
import re
import json
import time
import math
import random
import hashlib
import threading
import string
from dataclasses import dataclass, field, asdict
from fractions import Fraction
from typing import List, Optional, Dict, Any, Tuple, Union

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

import textgrad as tg
from textgrad.engine import EngineLM


# ---------------------------------------------------------------------
# Environment and model configuration
# ---------------------------------------------------------------------

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

MODEL_FORWARD = os.environ.get("MODEL_FORWARD", "deepseek-v4-flash")
MODEL_BACKWARD = os.environ.get("MODEL_BACKWARD", "deepseek-v4-flash")

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.4"))
DEFAULT_TOP_K = int(os.environ.get("TOP_K_EXPERIENCES", "3"))

CHOICE_LETTERS = list(string.ascii_uppercase)
_thread_state = threading.local()


# ---------------------------------------------------------------------
# API utilities
# ---------------------------------------------------------------------

def _require_api_key() -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. "
            "Please set it as an environment variable before running."
        )
    return DEEPSEEK_API_KEY


def get_deepseek_client() -> OpenAI:
    return OpenAI(
        api_key=_require_api_key(),
        base_url=DEEPSEEK_BASE_URL,
    )


def _begin_api_count() -> None:
    _thread_state.api_calls = 0


def _inc_api_count() -> None:
    if not hasattr(_thread_state, "api_calls"):
        _thread_state.api_calls = 0
    _thread_state.api_calls += 1


def _get_api_count() -> int:
    return int(getattr(_thread_state, "api_calls", 0))


def chat_completion(
    messages: List[Dict[str, str]],
    model: str = MODEL_FORWARD,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    client = get_deepseek_client()
    _inc_api_count()

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return resp.choices[0].message.content or ""


class DeepSeekEngine(EngineLM):
    DEFAULT_SYSTEM_PROMPT = (
        "You are a careful reasoning evaluator. "
        "Evaluate whether the candidate solution correctly answers the task. "
        "Give concise, actionable feedback for improvement."
    )

    def __init__(
        self,
        model_string: str = MODEL_BACKWARD,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.model_string = model_string
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = get_deepseek_client()
        self.is_multimodal = False

    def generate(self, content, system_prompt=None, **kwargs):
        sys_prompt = system_prompt or self.system_prompt

        if isinstance(content, str):
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": content},
            ]
        else:
            messages = content

        _inc_api_count()

        resp = self.client.chat.completions.create(
            model=self.model_string,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return resp.choices[0].message.content or ""

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)


def setup_textgrad_with_deepseek() -> DeepSeekEngine:
    engine = DeepSeekEngine()
    tg.set_backward_engine(engine, override=True)
    return engine


# ---------------------------------------------------------------------
# Data loading and benchmark adaptation
# ---------------------------------------------------------------------

def load_problem_file(path: str) -> List[Dict[str, Any]]:
    """
    Load problems from .json or .jsonl.

    Supported JSON shapes:
    - list[dict]
    - {"data": list[dict]}
    - {"examples": list[dict]}
    - {"questions": list[dict]}
    """
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
        for key in ["data", "examples", "questions", "items", "records"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(f"Unsupported data format in file: {path}")


def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _lower_set(csv: Optional[str]) -> Optional[set]:
    if not csv:
        return None
    return {x.strip().lower() for x in csv.split(",") if x.strip()}


def filter_problems(
    problems: List[Dict[str, Any]],
    include_subjects: Optional[str] = None,
    include_categories: Optional[str] = None,
    include_tasks: Optional[str] = None,
    limit: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Filter examples for MMLU / BBH style metadata.

    Examples:
    - MMLU physics:
      include_subjects="high_school_physics,college_physics"

    - BBH selected tasks:
      include_tasks="date_understanding,boolean_expressions"
    """
    subj_set = _lower_set(include_subjects)
    cat_set = _lower_set(include_categories)
    task_set = _lower_set(include_tasks)

    output = []

    for p in problems:
        n = normalize_problem(p)

        subject = str(n["metadata"].get("subject", "")).lower()
        category = str(n["metadata"].get("category", "")).lower()
        task = str(n["metadata"].get("task", "")).lower()

        if subj_set and subject not in subj_set:
            continue

        if cat_set:
            joined = f"{category} {subject} {task}".lower()
            if not any(c in joined for c in cat_set):
                continue

        if task_set and task not in task_set:
            continue

        output.append(p)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(output)

    if limit is not None and limit > 0:
        output = output[:limit]

    return output


def _first_existing(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _normalize_choices(raw_choices: Any) -> List[str]:
    if raw_choices is None:
        return []

    if isinstance(raw_choices, list):
        out = []
        for c in raw_choices:
            if isinstance(c, dict):
                val = _first_existing(c, ["text", "content", "answer", "choice", "label"], "")
                out.append(str(val))
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


def _stable_shuffle_options(question: str, options: List[str]) -> List[str]:
    seed = int(hashlib.sha256(question.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    copied = list(options)
    rng.shuffle(copied)
    return copied


def _answer_to_letter(answer: Any, choices: List[str]) -> Optional[str]:
    if answer is None:
        return None

    n = len(choices)
    if n <= 0:
        return None

    # Integer index, common in MMLU: 0,1,2,3
    if isinstance(answer, int):
        if 0 <= answer < n:
            return CHOICE_LETTERS[answer]
        if 1 <= answer <= n:
            return CHOICE_LETTERS[answer - 1]

    ans = str(answer).strip()

    # Letter
    if len(ans) == 1 and ans.upper() in CHOICE_LETTERS[:n]:
        return ans.upper()

    # Numeric string index
    if re.fullmatch(r"\d+", ans):
        idx = int(ans)
        if 0 <= idx < n:
            return CHOICE_LETTERS[idx]
        if 1 <= idx <= n:
            return CHOICE_LETTERS[idx - 1]

    # Match option text
    ans_norm = normalize_text(ans)
    for i, c in enumerate(choices):
        if normalize_text(c) == ans_norm:
            return CHOICE_LETTERS[i]

    return None


def normalize_problem(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert BBH / MMLU  examples into a unified schema.

    Unified schema:
    {
      "question": str,
      "answer": str,
      "answer_type": "mcq" | "numeric" | "exact",
      "choices": list[str],
      "metadata": {...},
      "raw": original dict
    }
    """
    # Common metadata
    subject = _first_existing(
        raw,
        ["subject", "Subject", "mmlu_subject", "subdomain", "Subdomain"],
        "",
    )
    category = _first_existing(
        raw,
        ["category", "Category", "discipline", "Discipline", "domain", "Domain"],
        "",
    )
    task = _first_existing(
        raw,
        ["task", "Task", "task_name", "bbh_task", "name"],
        "",
    )
    benchmark = _first_existing(
        raw,
        ["benchmark", "dataset", "source", "src"],
        "",
    )

    correct_answer = _first_existing(
        raw,
        [
            "Correct Answer",
            "correct_answer",
            "correct",
            "answer_correct",
            "gold",
        ],
        None,
    )

    incorrects = []
    for k in [
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
        "incorrect_answer_1",
        "incorrect_answer_2",
        "incorrect_answer_3",
        "incorrect1",
        "incorrect2",
        "incorrect3",
    ]:
        if k in raw and raw[k]:
            incorrects.append(str(raw[k]))

    # Question field
    question = _first_existing(
        raw,
        [
            "question",
            "Question",
            "input",
            "prompt",
            "query",
            "problem",
            "stem",
        ],
        "",
    )
    question = str(question).strip()

    # Choices
    choices = _normalize_choices(
        _first_existing(
            raw,
            ["choices", "options", "Options", "answer_choices", "candidate_answers"],
            None,
        )
    )

    if correct_answer is not None and incorrects and not choices:
        all_options = [str(correct_answer)] + incorrects
        choices = _stable_shuffle_options(question, all_options)
        answer = _answer_to_letter(str(correct_answer), choices)
        answer_type = "mcq"
    else:
        answer = _first_existing(
            raw,
            [
                "answer",
                "Answer",
                "target",
                "targets",
                "label",
                "gold_answer",
                "correct",
            ],
            correct_answer,
        )

        if isinstance(answer, list):
            answer = answer[0] if answer else ""

        if choices:
            letter = _answer_to_letter(answer, choices)
            answer = letter if letter is not None else str(answer)
            answer_type = "mcq"
        else:
            answer = str(answer).strip()
            if looks_numeric(answer):
                answer_type = "numeric"
            else:
                answer_type = "exact"

    formatted_question = format_question_for_model(
        question=question,
        choices=choices,
        metadata={
            "subject": subject,
            "category": category,
            "task": task,
            "benchmark": benchmark,
        },
    )

    return {
        "question": formatted_question,
        "raw_question": question,
        "answer": str(answer).strip(),
        "answer_type": answer_type,
        "choices": choices,
        "metadata": {
            "subject": str(subject),
            "category": str(category),
            "task": str(task),
            "benchmark": str(benchmark),
        },
        "raw": raw,
    }


def format_question_for_model(
    question: str,
    choices: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    metadata = metadata or {}
    parts = []

    subject = metadata.get("subject", "")
    category = metadata.get("category", "")
    task = metadata.get("task", "")

    meta_line = []
    if subject:
        meta_line.append(f"Subject: {subject}")
    if category:
        meta_line.append(f"Category: {category}")
    if task:
        meta_line.append(f"Task: {task}")

    if meta_line:
        parts.append(" | ".join(meta_line))

    parts.append(str(question).strip())

    if choices:
        parts.append("Choices:")
        for i, c in enumerate(choices):
            if i >= len(CHOICE_LETTERS):
                break
            parts.append(f"{CHOICE_LETTERS[i]}. {c}")

    return "\n".join(parts).strip()


def get_problem_id(problem: Union[Dict[str, Any], str]) -> str:
    if isinstance(problem, str):
        text = problem.strip()
    else:
        n = normalize_problem(problem)
        text = n["question"].strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------
# Experience memory
# ---------------------------------------------------------------------

@dataclass
class OptimizationExperience:
    problem_id: str
    problem_text: str
    problem_type: str = ""
    answer_type: str = ""
    initial_solution: str = ""
    final_solution: str = ""
    textual_gradients: List[str] = field(default_factory=list)
    key_insight: str = ""
    num_iterations: int = 0
    success: bool = False
    improvement_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {k: v for k, v in data.items() if k in allowed}
        return cls(**clean)


class ExperienceMemory:
    def __init__(
        self,
        capacity: int = 3000,
        embedding_model: str = EMBEDDING_MODEL,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold

        self.experiences: List[OptimizationExperience] = []
        self.embeddings: List[np.ndarray] = []

        self.stats = {
            "total_stored": 0,
            "total_retrieved": 0,
            "cache_hits": 0,
            "duplicate_skipped": 0,
        }

        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._seen_problem_ids = set()
        self._embedder = SentenceTransformer(embedding_model)
        self.lock = threading.RLock()

    @staticmethod
    def problem_id(text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def _memory_text(self, exp: OptimizationExperience) -> str:
        return (
            f"Problem type: {exp.problem_type}\n"
            f"Answer type: {exp.answer_type}\n"
            f"Metadata: {json.dumps(exp.metadata, ensure_ascii=False)}\n"
            f"Problem:\n{exp.problem_text}\n"
            f"Reusable insight:\n{exp.key_insight}\n"
            f"Final solution sketch:\n{exp.final_solution[:1000]}"
        )

    def _query_text(self, problem_text: str) -> str:
        return (
            f"Problem type: {infer_problem_type(problem_text)}\n"
            f"Problem:\n{problem_text}"
        )

    def _get_embedding(self, text: str) -> np.ndarray:
        text = text[:4000]
        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()

        with self.lock:
            if cache_key in self._embedding_cache:
                self.stats["cache_hits"] += 1
                return self._embedding_cache[cache_key]

        emb = self._embedder.encode(text, normalize_embeddings=True)

        with self.lock:
            self._embedding_cache[cache_key] = emb

        return emb

    def store(self, experience: OptimizationExperience) -> bool:
        if not experience.success:
            return False

        if not experience.problem_id:
            experience.problem_id = self.problem_id(experience.problem_text)

        with self.lock:
            if experience.problem_id in self._seen_problem_ids:
                self.stats["duplicate_skipped"] += 1
                return False

            if len(self.experiences) >= self.capacity:
                old = self.experiences.pop(0)
                self.embeddings.pop(0)
                self._seen_problem_ids.discard(old.problem_id)

            emb = self._get_embedding(self._memory_text(experience))

            self.experiences.append(experience)
            self.embeddings.append(emb)
            self._seen_problem_ids.add(experience.problem_id)
            self.stats["total_stored"] += 1

        return True

    def retrieve(
        self,
        query_problem: str,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: Optional[float] = None,
    ) -> Tuple[List[OptimizationExperience], List[float]]:
        threshold = self.similarity_threshold if min_similarity is None else min_similarity

        with self.lock:
            if not self.embeddings:
                return [], []

            embeddings = np.stack(self.embeddings)
            experiences = list(self.experiences)

        query_emb = self._get_embedding(self._query_text(query_problem))
        similarities = embeddings @ query_emb
        valid_idx = np.where(similarities >= threshold)[0]

        if len(valid_idx) == 0:
            return [], []

        sorted_idx = valid_idx[np.argsort(similarities[valid_idx])[::-1]]
        top_idx = sorted_idx[:top_k]

        exps = [experiences[i] for i in top_idx]
        sims = [float(similarities[i]) for i in top_idx]

        with self.lock:
            self.stats["total_retrieved"] += len(exps)

        return exps, sims

    def save(self, filepath: str) -> None:
        with self.lock:
            data = {
                "capacity": self.capacity,
                "similarity_threshold": self.similarity_threshold,
                "experiences": [e.to_dict() for e in self.experiences],
                "embeddings": [e.tolist() for e in self.embeddings],
                "stats": self.stats,
            }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        with self.lock:
            self.capacity = int(data.get("capacity", self.capacity))
            self.similarity_threshold = float(
                data.get("similarity_threshold", self.similarity_threshold)
            )
            self.experiences = [
                OptimizationExperience.from_dict(e)
                for e in data.get("experiences", [])
            ]
            self.embeddings = [np.array(e) for e in data.get("embeddings", [])]
            self.stats = data.get("stats", self.stats)
            self._seen_problem_ids = {e.problem_id for e in self.experiences}


# ---------------------------------------------------------------------
# MAT optimizer
# ---------------------------------------------------------------------

class MemoryAugmentedTGD:
    def __init__(
        self,
        parameters,
        memory: Optional[ExperienceMemory],
        top_k_experiences: int = DEFAULT_TOP_K,
        sim_threshold: float = SIMILARITY_THRESHOLD,
        use_retrieval: bool = True,
        use_adaptive_iter: bool = True,
        use_gradient_injection: bool = True,
    ):
        self.parameters = parameters
        self.memory = memory
        self.top_k = top_k_experiences
        self.sim_threshold = sim_threshold

        self.use_retrieval = use_retrieval
        self.use_adaptive_iter = use_adaptive_iter
        self.use_gradient_injection = use_gradient_injection

        self.base_optimizer = tg.TGD(parameters=parameters)

        self.current_problem: Optional[str] = None
        self.current_answer_type: str = ""
        self.current_metadata: Dict[str, Any] = {}

        self.gradient_history: List[str] = []
        self.retrieved_experiences: List[OptimizationExperience] = []
        self.retrieved_similarities: List[float] = []
        self.predicted_iters: Optional[int] = None

    def set_problem(
        self,
        problem_text: str,
        answer_type: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.current_problem = problem_text
        self.current_answer_type = answer_type
        self.current_metadata = metadata or {}

        self.gradient_history = []
        self.retrieved_experiences = []
        self.retrieved_similarities = []
        self.predicted_iters = None

        if self.use_retrieval and self.memory is not None:
            exps, sims = self.memory.retrieve(
                query_problem=problem_text,
                top_k=self.top_k,
                min_similarity=self.sim_threshold,
            )
            self.retrieved_experiences = exps
            self.retrieved_similarities = sims

        if self.use_adaptive_iter and self.retrieved_experiences:
            self.predicted_iters = self.predict_required_iterations()

    def predict_required_iterations(self) -> int:
        if not self.retrieved_experiences:
            return 3

        weights = np.array(self.retrieved_similarities, dtype=float)
        iters = np.array(
            [max(1, int(e.num_iterations)) for e in self.retrieved_experiences],
            dtype=float,
        )

        if weights.sum() <= 0:
            return 3

        pred = int(round(float(np.average(iters, weights=weights))))
        return int(np.clip(pred, 1, 5))

    def _format_memory_gradient(self) -> str:
        if not self.retrieved_experiences:
            return ""

        parts = []

        for rank, (exp, sim) in enumerate(
            zip(self.retrieved_experiences, self.retrieved_similarities),
            start=1,
        ):
            if not exp.key_insight:
                continue

            parts.append(
                f"[Retrieved experience #{rank}; similarity={sim:.3f}; "
                f"type={exp.problem_type}; answer_type={exp.answer_type}] "
                f"{exp.key_insight[:700]}"
            )

        if not parts:
            return ""

        return (
            "Memory-augmented textual gradient guidance:\n"
            "Use these retrieved successful trajectories as additional optimization "
            "signals. They may reveal common traps, answer-format constraints, "
            "or reasoning strategies for similar tasks.\n"
            + "\n".join(parts)
        )

    @staticmethod
    def _gradient_to_text(gradients) -> str:
        if gradients is None:
            return ""

        if isinstance(gradients, (set, list, tuple)):
            return " ".join(str(getattr(g, "value", g)) for g in gradients)

        return str(getattr(gradients, "value", gradients))

    def step(self) -> None:
        for param in self.parameters:
            if not hasattr(param, "gradients") or param.gradients is None:
                continue

            original_gradient = self._gradient_to_text(param.gradients)
            self.gradient_history.append(original_gradient)

            if self.use_gradient_injection and self.retrieved_experiences:
                memory_gradient_text = self._format_memory_gradient()

                if memory_gradient_text:
                    memory_gradient_var = tg.Variable(
                        memory_gradient_text,
                        role_description=(
                            "retrieved long-term memory as an additional textual gradient"
                        ),
                        requires_grad=False,
                    )

                    if not isinstance(param.gradients, set):
                        if isinstance(param.gradients, list):
                            param.gradients = set(param.gradients)
                        else:
                            param.gradients = {param.gradients}

                    param.gradients.add(memory_gradient_var)

        self.base_optimizer.step()

    def record_success(
        self,
        final_solution: str,
        initial_solution: str,
    ) -> bool:
        if not self.current_problem or self.memory is None:
            return False

        insight = extract_reusable_insight(
            problem=self.current_problem,
            answer_type=self.current_answer_type,
            metadata=self.current_metadata,
            initial_solution=initial_solution,
            final_solution=final_solution,
            textual_gradients=self.gradient_history,
        )

        exp = OptimizationExperience(
            problem_id=ExperienceMemory.problem_id(self.current_problem),
            problem_text=self.current_problem,
            problem_type=infer_problem_type(self.current_problem, self.current_metadata),
            answer_type=self.current_answer_type,
            initial_solution=initial_solution,
            final_solution=final_solution,
            textual_gradients=self.gradient_history.copy(),
            key_insight=insight,
            num_iterations=len(self.gradient_history),
            success=True,
            improvement_score=1.0,
            metadata=self.current_metadata,
        )

        return self.memory.store(exp)


# ---------------------------------------------------------------------
# Reasoning type, loss, generation
# ---------------------------------------------------------------------

def infer_problem_type(
    question: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    metadata = metadata or {}

    subject = str(metadata.get("subject", "")).lower()
    category = str(metadata.get("category", "")).lower()
    task = str(metadata.get("task", "")).lower()

    joined_meta = f"{subject} {category} {task}".strip()
    if joined_meta:
        if "physics" in joined_meta:
            return "science_physics"
        if "biology" in joined_meta:
            return "science_biology"
        if "chemistry" in joined_meta:
            return "science_chemistry"
        if "date" in joined_meta:
            return "bbh_date_understanding"
        if "boolean" in joined_meta:
            return "bbh_boolean_logic"
        if "causal" in joined_meta:
            return "bbh_causal_judgment"
        if "dyck" in joined_meta:
            return "bbh_formal_language"
        if "tracking" in joined_meta:
            return "bbh_object_tracking"

    q = question.lower()

    if any(w in q for w in ["force", "velocity", "acceleration", "energy", "momentum", "charge", "field"]):
        return "science_physics"

    if any(w in q for w in ["cell", "gene", "protein", "enzyme", "organism", "evolution", "dna", "rna"]):
        return "science_biology"

    if any(w in q for w in ["molecule", "reaction", "acid", "base", "compound", "electron", "bond"]):
        return "science_chemistry"

    if any(w in q for w in ["true", "false", "and", "or", "not"]):
        return "logic_boolean"

    if any(w in q for w in ["date", "day", "month", "year"]):
        return "date_reasoning"

    if any(w in q for w in ["choose", "which of the following", "choices:"]):
        return "multiple_choice_reasoning"

    if any(w in q for w in ["calculate", "compute", "how many", "what is the value"]):
        return "quantitative_reasoning"

    return "general_reasoning"


def create_loss_function(answer_type: str = "exact") -> tg.TextLoss:
    instruction = f"""
Evaluate the candidate solution for this benchmark task.

Answer type: {answer_type}

Check:
1. Whether the reasoning is faithful to the question.
2. Whether the candidate selected or produced the correct final answer.
3. Whether there are hidden traps, distractors, unit errors, or logical errors.
4. For multiple-choice tasks, the final answer must be a single option letter.
5. For exact-match tasks, the final answer should match the required target.
6. If incorrect, provide specific actionable feedback for the next optimization step.
7. If correct, confirm briefly.

The feedback will be used as a TextGrad textual gradient.
"""
    return tg.TextLoss(instruction)


def generate_initial_solution(
    problem_or_question: Union[Dict[str, Any], str],
    temperature: float = 0.0,
) -> str:
    if isinstance(problem_or_question, dict):
        p = normalize_problem(problem_or_question)
        question = p["question"]
        answer_type = p["answer_type"]
    else:
        question = str(problem_or_question)
        answer_type = "exact"

    if answer_type == "mcq":
        final_instruction = (
            "Solve carefully. End with exactly one line: Final answer: <letter>. "
            "The letter must be one of the provided choices."
        )
    elif answer_type == "numeric":
        final_instruction = (
            "Solve carefully. End with exactly one line: Final answer: <number>."
        )
    else:
        final_instruction = (
            "Solve carefully. End with exactly one line: Final answer: <short answer>."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful benchmark-solving assistant. "
                "Reason step by step, avoid overconfidence, and follow the required answer format. "
                + final_instruction
            ),
        },
        {"role": "user", "content": question},
    ]

    try:
        return chat_completion(
            messages=messages,
            model=MODEL_FORWARD,
            temperature=temperature,
            max_tokens=2048,
        ).strip()
    except Exception as e:
        return f"Error while generating initial solution: {e}"


def extract_reusable_insight(
    problem: str,
    answer_type: str,
    metadata: Dict[str, Any],
    initial_solution: str,
    final_solution: str,
    textual_gradients: List[str],
    use_llm: bool = False,
) -> str:
    if not use_llm:
        last_grad = textual_gradients[-1].strip() if textual_gradients else ""
        if last_grad:
            return last_grad[:700]

        ptype = infer_problem_type(problem, metadata)
        return (
            f"For {ptype} / {answer_type} tasks, identify the target, check "
            "distractors, preserve answer format, and verify the final response."
        )

    prompt = f"""
Summarize the reusable optimization insight from this successful MAT trajectory.

Metadata:
{json.dumps(metadata, ensure_ascii=False)}

Answer type:
{answer_type}

Problem:
{problem}

Initial solution:
{initial_solution[:1000]}

Final correct solution:
{final_solution[:1000]}

Recent textual gradients:
{chr(10).join(textual_gradients[-3:])[:2000]}

Return one concise reusable strategy sentence.
"""

    try:
        return chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You extract concise reusable benchmark-solving strategies.",
                },
                {"role": "user", "content": prompt},
            ],
            model=MODEL_FORWARD,
            temperature=0.0,
            max_tokens=256,
        ).strip()
    except Exception:
        return "Check the task type, avoid distractors, and follow the required final-answer format."


# ---------------------------------------------------------------------
# Answer extraction and checking
# ---------------------------------------------------------------------

def normalize_text(s: str) -> str:
    """
    Normalize text for answer comparison.

    Designed for:
    - BBH exact answers
    - MMLU option text comparison
    - short free-form answers
    """
    s = "" if s is None else str(s)
    s = s.strip().lower()

    # Remove markdown/code/latex-ish wrappers
    s = s.replace("```", " ")
    s = re.sub(r"\\boxed\s*\{([^{}]+)\}", r"\1", s)
    s = re.sub(r"\\\((.*?)\\\)", r"\1", s)
    s = re.sub(r"\\\[(.*?)\\\]", r"\1", s)

    # Remove common final-answer prefixes
    s = re.sub(
        r"^(final\s+answer|answer|the\s+answer\s+is|therefore\s*,?\s*the\s+answer\s+is|答案|最终答案)\s*[:：=\-]*\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )

    # Remove leading option markers: A. xxx / (A) xxx / A) xxx
    s = re.sub(r"^\(?[a-z]\)?[\.\):：]\s+", "", s)

    # Normalize quotes and punctuation
    s = s.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    s = s.strip(" \t\n\r.。,:;；!！?？\"'`")

    # Normalize articles lightly for English exact match
    s = re.sub(r"\b(a|an|the)\b", " ", s)

    # Normalize spaces
    s = re.sub(r"\s+", " ", s).strip()

    return s


def looks_numeric(text: str) -> bool:
    text = str(text).strip()
    if not text:
        return False

    return bool(
        re.search(
            r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*)?%?",
            text,
        )
    )


def _clean_number_string(s: str) -> str:
    return s.replace(",", "").replace("$", "").replace("%", "").strip()


def _to_float_maybe(s: str) -> Optional[float]:
    s = _clean_number_string(str(s))

    if not s:
        return None

    try:
        if "/" in s and re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", s):
            return float(Fraction(s.replace(" ", "")))
        return float(s)
    except Exception:
        return None


def extract_final_number(text: str) -> Optional[float]:
    if not text:
        return None

    patterns = [
        r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*)?%?)",
        r"final answer\s*[:=]?\s*([-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*)?%?)",
        r"the answer is\s*([-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*)?%?)",
        r"answer\s*[:=]\s*([-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*)?%?)",
    ]

    lower = text.lower()
    for p in patterns:
        matches = re.findall(p, lower, flags=re.IGNORECASE)
        if matches:
            val = _to_float_maybe(matches[-1])
            if val is not None:
                return val

    candidates = re.findall(
        r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*)?%?",
        text,
    )

    for cand in reversed(candidates):
        val = _to_float_maybe(cand)
        if val is not None:
            return val

    return None


def extract_final_choice(text: str, n_choices: int) -> Optional[str]:
    """
    Extract final MCQ choice letter robustly.

    Supports:
    - Final answer: C
    - Final answer: (C)
    - Answer: C.
    - The correct answer is C
    - Option C
    - choose C / answer：C
    - \\boxed{C}
    """
    if not text or n_choices <= 0:
        return None

    valid = CHOICE_LETTERS[:n_choices]
    valid_group = "".join(valid)

    raw = str(text)
    lower = raw.lower()

    # Handle boxed answers first
    boxed_patterns = [
        rf"\\boxed\s*\{{\s*([{valid_group}{valid_group.lower()}])\s*\}}",
        rf"boxed\s*\{{\s*([{valid_group}{valid_group.lower()}])\s*\}}",
    ]
    for p in boxed_patterns:
        matches = re.findall(p, raw, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # Strong final-answer patterns
    patterns = [
        rf"final\s+answer\s*[:：=]?\s*\(?\s*([{valid_group}])\s*\)?\b",
        rf"final\s+choice\s*[:：=]?\s*\(?\s*([{valid_group}])\s*\)?\b",
        rf"answer\s*[:：=]?\s*\(?\s*([{valid_group}])\s*\)?\b",
        rf"答案\s*[:：=]?\s*\(?\s*([{valid_group}])\s*\)?\b",
        rf"最终答案\s*[:：=]?\s*\(?\s*([{valid_group}])\s*\)?\b",
        rf"the\s+answer\s+is\s*\(?\s*([{valid_group}])\s*\)?\b",
        rf"the\s+correct\s+answer\s+is\s*\(?\s*([{valid_group}])\s*\)?\b",
        rf"correct\s+answer\s*[:：=]?\s*\(?\s*([{valid_group}])\s*\)?\b",
        rf"option\s*\(?\s*([{valid_group}])\s*\)?\b",
        rf"choice\s*\(?\s*([{valid_group}])\s*\)?\b",
        rf"选\s*\(?\s*([{valid_group}])\s*\)?",
        rf"选择\s*\(?\s*([{valid_group}])\s*\)?",
        rf"\b([{valid_group}])\b\s+is\s+correct",
        rf"\b([{valid_group}])\b\s+is\s+the\s+correct\s+answer",
    ]

    for p in patterns:
        matches = re.findall(p, raw, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # Inspect last few non-empty lines.
    # This catches outputs like:
    # reasoning...
    # C
    lines = [x.strip() for x in raw.strip().splitlines() if x.strip()]
    tail_lines = lines[-6:]

    for line in reversed(tail_lines):
        clean = line.strip()

        # Examples: C / (C) / C. / C) / C:
        m = re.fullmatch(
            rf"\(?\s*([{valid_group}])\s*\)?[\.\):：]?",
            clean,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1).upper()

        # Examples: Final answer: **C**
        clean2 = re.sub(r"[*_`]", "", clean)
        m = re.search(
            rf"(final\s+answer|answer|答案|最终答案)\s*[:：=]?\s*\(?\s*([{valid_group}])\s*\)?",
            clean2,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(2).upper()

    # Conservative fallback:
    # only look at final 500 chars, not whole solution,
    # to avoid matching option labels in the question.
    tail = raw[-500:]
    fallback_patterns = [
        rf"\(([{valid_group}])\)",
        rf"\b([{valid_group}])\b",
    ]

    for p in fallback_patterns:
        matches = re.findall(p, tail, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    return None


def extract_final_text(text: str) -> str:
    """
    Extract final free-form answer text.

    More robust for:
    - Final answer: xxx
    - Answer: xxx
    - The answer is xxx
    - 答案：xxx
    - boxed answers
    """
    if not text:
        return ""

    raw = str(text).strip()

    # Boxed answer
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", raw, flags=re.IGNORECASE)
    if boxed:
        return boxed[-1].strip()

    patterns = [
        r"final\s+answer\s*[:：=]\s*(.+)",
        r"final\s+answer\s+is\s+(.+)",
        r"answer\s*[:：=]\s*(.+)",
        r"the\s+answer\s+is\s+(.+)",
        r"therefore\s*,?\s*the\s+answer\s+is\s+(.+)",
        r"答案\s*[:：=]\s*(.+)",
        r"最终答案\s*[:：=]\s*(.+)",
    ]

    for p in patterns:
        matches = re.findall(p, raw, flags=re.IGNORECASE)
        if matches:
            ans = matches[-1].strip()
            # Only take first line after the answer prefix
            ans = ans.splitlines()[0].strip()
            return ans.strip(" \t\n\r.。,:;；!！?？")

    lines = [x.strip() for x in raw.splitlines() if x.strip()]
    if not lines:
        return raw

    # Prefer the final non-empty line.
    return lines[-1].strip(" \t\n\r.。,:;；!！?？")


def check_answer(
    solution: str,
    ground_truth: str,
    answer_type: str = "exact",
    choices: Optional[List[str]] = None,
    tolerance: float = 1e-2,
) -> bool:
    """
    Robust but not overly permissive answer checker.

    Main goals:
    - MCQ: reliably extract final option letter.
    - Numeric: compare numeric values with tolerance.
    - Exact: support BBH-style short answers, booleans, simple text normalization.
    """
    choices = choices or []
    solution = "" if solution is None else str(solution)
    ground_truth = "" if ground_truth is None else str(ground_truth)

    # ------------------------------------------------------------
    # Multiple choice
    # ------------------------------------------------------------
    if answer_type == "mcq":
        n_choices = len(choices)

        # Normalize gold letter
        gold_letter = None
        if n_choices > 0:
            gold_letter = _answer_to_letter(ground_truth, choices)

        if gold_letter is None:
            gt = ground_truth.strip()
            if gt:
                # Supports "C", "(C)", "C.", "answer: C"
                m = re.search(
                    rf"\b([{''.join(CHOICE_LETTERS[:max(n_choices, 4)])}])\b",
                    gt,
                    flags=re.IGNORECASE,
                )
                if m:
                    gold_letter = m.group(1).upper()
                else:
                    gold_letter = gt[:1].upper()

        if not gold_letter:
            return False

        # Extract predicted letter
        pred_letter = extract_final_choice(solution, n_choices if n_choices > 0 else 4)

        if pred_letter is not None:
            return pred_letter == gold_letter

        # Fallback: compare final text with correct option text
        if choices and gold_letter in CHOICE_LETTERS[:len(choices)]:
            gold_idx = CHOICE_LETTERS.index(gold_letter)
            gold_text = normalize_text(choices[gold_idx])
            pred_text = normalize_text(extract_final_text(solution))

            if pred_text == gold_text:
                return True

            # Allow the final answer line to contain the correct option text
            # but avoid allowing extremely short accidental matches.
            if len(gold_text) >= 8 and gold_text in pred_text:
                return True

        return False

    # ------------------------------------------------------------
    # Numeric
    # ------------------------------------------------------------
    if answer_type == "numeric":
        pred = extract_final_number(solution)
        gold = extract_final_number(str(ground_truth))

        if pred is None or gold is None:
            return False

        return math.isclose(pred, gold, abs_tol=tolerance, rel_tol=1e-4)

    # ------------------------------------------------------------
    # Exact / free-form
    # ------------------------------------------------------------
    pred_raw = extract_final_text(solution)
    gold_raw = ground_truth

    pred = normalize_text(pred_raw)
    gold = normalize_text(gold_raw)

    if not pred or not gold:
        return False

    # Exact normalized match
    if pred == gold:
        return True

    # Boolean normalization
    bool_map = {
        "yes": "true",
        "y": "true",
        "true": "true",
        "t": "true",
        "1": "true",
        "no": "false",
        "n": "false",
        "false": "false",
        "f": "false",
        "0": "false",
    }

    if pred in bool_map and gold in bool_map:
        return bool_map[pred] == bool_map[gold]

    # Numeric fallback for exact-type tasks
    pred_num = extract_final_number(pred_raw)
    gold_num = extract_final_number(gold_raw)
    if pred_num is not None and gold_num is not None:
        if math.isclose(pred_num, gold_num, abs_tol=tolerance, rel_tol=1e-4):
            return True

    # Strip common option-like wrappers for exact answers
    pred2 = re.sub(r"^\(?[a-z]\)?[\.\):：]\s*", "", pred).strip()
    gold2 = re.sub(r"^\(?[a-z]\)?[\.\):：]\s*", "", gold).strip()

    if pred2 == gold2:
        return True

    # Short-answer containment.
    # This is useful for BBH tasks where final line may be:
    # "Final answer: the correct object is the apple"
    # while gold is "apple".
    #
    # Keep it conservative:
    # - gold <= 5 words
    # - gold length >= 2
    # - match as phrase boundary when possible
    if len(gold2) >= 2 and len(gold2.split()) <= 5:
        escaped = re.escape(gold2)
        if re.search(rf"(^|[^a-zA-Z0-9]){escaped}([^a-zA-Z0-9]|$)", pred2):
            return True

        # Also allow final text ending with gold.
        if pred2.endswith(gold2):
            return True

    # Some BBH answers are comma-separated or list-like.
    # Normalize simple punctuation variants.
    pred_compact = re.sub(r"[\s,;，；]+", " ", pred2).strip()
    gold_compact = re.sub(r"[\s,;，；]+", " ", gold2).strip()

    if pred_compact == gold_compact:
        return True

    return False

def extract_predicted_answer(
    solution: str,
    answer_type: str,
    choices: Optional[List[str]] = None,
):
    choices = choices or []

    if answer_type == "mcq":
        return extract_final_choice(solution, len(choices))

    if answer_type == "numeric":
        return extract_final_number(solution)

    return extract_final_text(solution)


# ---------------------------------------------------------------------
# Main single-problem runner
# ---------------------------------------------------------------------

def run_single_problem(
    problem: Dict[str, Any],
    method: str,
    memory: Optional[ExperienceMemory],
    max_iterations: int = 3,
    is_training: bool = False,
    initial_solution: Optional[str] = None,
    sim_threshold: float = SIMILARITY_THRESHOLD,
    top_k_experiences: int = DEFAULT_TOP_K,
    use_retrieval: bool = True,
    use_adaptive_iter: bool = True,
    use_gradient_injection: bool = True,
    initial_temperature: float = 0.0,
) -> Dict[str, Any]:
    _begin_api_count()
    start_total = time.time()

    p = normalize_problem(problem)

    question = p["question"]
    ground_truth = p["answer"]
    answer_type = p["answer_type"]
    choices = p["choices"]
    metadata = p["metadata"]

    if initial_solution is None:
        initial_solution = generate_initial_solution(
            problem,
            temperature=initial_temperature,
        )

    init_time = time.time() - start_total

    solution = tg.Variable(
        initial_solution,
        role_description="candidate benchmark solution",
        requires_grad=True,
    )

    loss_fn = create_loss_function(answer_type=answer_type)

    predicted_iters = None
    retrieved_count = 0
    retrieved_similarities = []

    if method == "vanilla":
        optimizer = tg.TGD(parameters=[solution])
        actual_max_iters = max_iterations

    elif method == "mat":
        optimizer = MemoryAugmentedTGD(
            parameters=[solution],
            memory=memory,
            top_k_experiences=top_k_experiences,
            sim_threshold=sim_threshold,
            use_retrieval=use_retrieval,
            use_adaptive_iter=use_adaptive_iter,
            use_gradient_injection=use_gradient_injection,
        )

        optimizer.set_problem(
            problem_text=question,
            answer_type=answer_type,
            metadata=metadata,
        )

        predicted_iters = optimizer.predicted_iters
        retrieved_count = len(optimizer.retrieved_experiences)
        retrieved_similarities = optimizer.retrieved_similarities

        if predicted_iters is not None:
            actual_max_iters = min(max_iterations, max(1, predicted_iters + 1))
        else:
            actual_max_iters = max_iterations

    else:
        raise ValueError(f"Unknown method: {method}")

    success = False
    final_solution = initial_solution
    num_iterations = 0
    iterations_log = []

    loop_start = time.time()

    for i in range(actual_max_iters):
        if check_answer(
            solution=solution.value,
            ground_truth=ground_truth,
            answer_type=answer_type,
            choices=choices,
        ):
            success = True
            final_solution = solution.value
            num_iterations = i
            break

        loss = loss_fn(solution)
        loss.backward()

        iterations_log.append(
            {
                "iteration": i + 1,
                "solution_preview": str(solution.value)[:700],
                "gradient_preview": str(solution.gradients)[:700],
            }
        )

        optimizer.step()

    if not success:
        final_solution = solution.value
        num_iterations = actual_max_iters
        success = check_answer(
            solution=solution.value,
            ground_truth=ground_truth,
            answer_type=answer_type,
            choices=choices,
        )

    loop_time = time.time() - loop_start
    total_time = init_time + loop_time

    stored = False

    if is_training and success and method == "mat":
        stored = optimizer.record_success(
            final_solution=final_solution,
            initial_solution=initial_solution,
        )

    return {
        "method": method,
        "question": question,
        "question_preview": question[:300],
        "raw_question": p["raw_question"],
        "ground_truth": ground_truth,
        "answer_type": answer_type,
        "choices": choices,
        "metadata": metadata,
        "problem_type": infer_problem_type(question, metadata),
        "initial_solution": initial_solution,
        "final_solution": final_solution,
        "predicted_answer": extract_predicted_answer(final_solution, answer_type, choices),
        "success": success,
        "num_iterations": num_iterations,
        "actual_max_iterations": actual_max_iters,
        "api_calls": _get_api_count(),
        "time": round(total_time, 4),
        "predicted_iter": predicted_iters,
        "retrieved_count": retrieved_count,
        "retrieved_similarities": retrieved_similarities,
        "stored_to_memory": stored,
        "use_retrieval": use_retrieval,
        "use_adaptive_iter": use_adaptive_iter,
        "use_gradient_injection": use_gradient_injection,
        "sim_threshold": sim_threshold,
        "iterations_log": iterations_log,
    }