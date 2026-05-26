# -*- coding: utf-8 -*-
"""
collect_extra_baseline_table.py

Collect extra baseline summaries into Markdown and LaTeX tables.

Example:
python collect_extra_baseline_table.py \
  --summaries summary_extra_mmlu.json summary_extra_bbh.json \
  --output_md table_extra_baselines.md \
  --output_tex table_extra_baselines.tex
"""

import os
import json
import argparse


METHOD_ORDER = [
    "direct",
    "zero_shot_cot",
    "cot",
    "reflexion",
    "tsgd_m",
]

METHOD_NAME = {
    "direct": "Direct",
    "zero_shot_cot": "Zero-shot CoT",
    "cot": "CoT",
    "reflexion": "Reflexion",
    "tsgd_m": "TSGD-M-style",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(text, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", nargs="+", required=True)
    parser.add_argument("--output_md", default="table_extra_baselines.md")
    parser.add_argument("--output_tex", default="table_extra_baselines.tex")
    args = parser.parse_args()

    rows = []

    for path in args.summaries:
        s = load_json(path)
        benchmark = s.get("benchmark", os.path.basename(path))
        metrics = s.get("metrics", {})

        for method in METHOD_ORDER:
            if method not in metrics:
                continue
            m = metrics[method]
            if not m:
                continue

            rows.append({
                "benchmark": benchmark,
                "method": METHOD_NAME.get(method, method),
                "accuracy": m.get("accuracy"),
                "avg_iterations": m.get("avg_iterations"),
                "avg_api_calls": m.get("avg_api_calls"),
                "avg_time": m.get("avg_time"),
                "total": m.get("total"),
            })

    # Markdown
    md = []
    md.append("| Benchmark | Method | Accuracy ↑ | Avg Iter ↓ | Avg API Calls ↓ | Avg Time ↓ | N |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['benchmark']} | {r['method']} | {fmt(r['accuracy'])} | "
            f"{fmt(r['avg_iterations'])} | {fmt(r['avg_api_calls'])} | "
            f"{fmt(r['avg_time'])} | {fmt(r['total'])} |"
        )
    md_text = "\n".join(md)

    # LaTeX
    tex = []
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\small")
    tex.append("\\begin{tabular}{llrrrrr}")
    tex.append("\\toprule")
    tex.append("Benchmark & Method & Acc. $\\uparrow$ & Iter. $\\downarrow$ & API $\\downarrow$ & Time $\\downarrow$ & N \\\\")
    tex.append("\\midrule")

    current_benchmark = None
    for r in rows:
        if current_benchmark is not None and r["benchmark"] != current_benchmark:
            tex.append("\\midrule")
        current_benchmark = r["benchmark"]

        tex.append(
            f"{r['benchmark']} & {r['method']} & {fmt(r['accuracy'])} & "
            f"{fmt(r['avg_iterations'])} & {fmt(r['avg_api_calls'])} & "
            f"{fmt(r['avg_time'])} & {fmt(r['total'])} \\\\"
        )

    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\caption{Additional baseline results.}")
    tex.append("\\label{tab:extra-baselines}")
    tex.append("\\end{table}")

    tex_text = "\n".join(tex)

    save_text(md_text, args.output_md)
    save_text(tex_text, args.output_tex)

    print(f"✅ Markdown table saved to: {args.output_md}")
    print(f"✅ LaTeX table saved to: {args.output_tex}")


if __name__ == "__main__":
    main()