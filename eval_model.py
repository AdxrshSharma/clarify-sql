"""Step 3 & 5: evaluate any backend on the test set and write results/<run>.json.

Metrics
- exec_acc (clear / resolved): predicted SQL returns the same rows as the gold SQL.
- clarify precision: of the times the model asked, how often was the question truly ambiguous?
- clarify recall:    of the truly ambiguous questions, how many did the model catch?
- latency and token counts (multiply tokens by your price/1M to get cost).

Examples
  python eval_model.py --backend gold --run oracle
  python eval_model.py --backend hf --model Qwen/Qwen2.5-1.5B-Instruct --run base
  python eval_model.py --backend hf --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/lora --run finetuned
  python eval_model.py --backend anthropic --model claude-sonnet-5-5 --run prompted_api
"""
import argparse
import json
import time
from pathlib import Path

from backends import make_backend
from common import parse_output
from sql_guard import execute_safe


def norm(v):
    return float(f"{v:.4g}") if isinstance(v, float) else v


def canon(rows, ordered):
    rows = [tuple(norm(x) for x in r) for r in rows]
    return rows if ordered else sorted(rows, key=repr)


def same_result(pred_sql, gold_sql):
    """Execution match: same rows (order only matters when the gold query has LIMIT)."""
    try:
        _, pred = execute_safe(pred_sql, max_rows=100_000)
    except Exception:
        return False
    _, gold = execute_safe(gold_sql, max_rows=100_000)
    ordered = "LIMIT" in gold_sql.upper()
    return canon(pred, ordered) == canon(gold, ordered)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(results):
    answerable = [r for r in results if r["kind"] in ("clear", "resolved")]
    amb = [r for r in results if r["kind"] == "ambiguous"]
    tp = sum(r["pred_kind"] == "clarify" for r in amb)
    fp = sum(r["pred_kind"] == "clarify" for r in answerable)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(amb) if amb else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {
        "n": len(results),
        "exec_acc_clear": mean(r["correct"] for r in results if r["kind"] == "clear"),
        "exec_acc_resolved": mean(r["correct"] for r in results if r["kind"] == "resolved"),
        "clarify_precision": precision, "clarify_recall": recall, "clarify_f1": f1,
        "over_clarify_rate": fp / len(answerable) if answerable else 0.0,
        "overall_acc": mean(r["correct"] for r in results),
        "avg_latency_s": mean(r["latency_s"] for r in results),
        "total_tokens_in": sum(r["tokens_in"] for r in results),
        "total_tokens_out": sum(r["tokens_out"] for r in results),
        "by_phrasing": {p: mean(r["correct"] for r in results if r["phrasing"] == p)
                        for p in ("seen", "heldout")},
        "by_category": {c: mean(r["correct"] for r in results if r["category"] == c)
                        for c in sorted({r["category"] for r in results})},
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["gold", "hf", "anthropic"])
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no_4bit", action="store_true")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(f"data/{args.split}.jsonl")]
    if args.limit:
        records = records[: args.limit]
    backend = make_backend(args.backend, args.model, args.adapter, records, not args.no_4bit)

    results = []
    for i, rec in enumerate(records, 1):
        t0 = time.perf_counter()
        raw, tin, tout = backend.predict(rec["question"])
        latency = time.perf_counter() - t0
        pred_kind, payload = parse_output(raw)
        if rec["kind"] == "ambiguous":
            correct = pred_kind == "clarify"
        else:
            correct = pred_kind == "sql" and same_result(payload, rec["target"])
        results.append({**rec, "raw": raw, "pred_kind": pred_kind, "correct": bool(correct),
                        "latency_s": latency, "tokens_in": tin, "tokens_out": tout})
        if i % 25 == 0:
            print(f"  {i}/{len(records)}  running acc={mean(r['correct'] for r in results):.3f}")

    summary = summarize(results)
    Path("results").mkdir(exist_ok=True)
    json.dump({"args": vars(args), "summary": summary, "results": results},
              open(f"results/{args.run}.json", "w"), indent=1)
    print(f"\n=== {args.run} ===")
    for k, v in summary.items():
        print(f"{k:20s} {v if isinstance(v, (int, dict)) else round(v, 4)}")


if __name__ == "__main__":
    main()
