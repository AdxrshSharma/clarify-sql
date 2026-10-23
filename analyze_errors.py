"""Error analysis for an evaluation run. No GPU needed.

  python analyze_errors.py                      # looks at trend + vague_time in results/finetuned.json
  python analyze_errors.py top_n budget         # other categories
  python analyze_errors.py trend --run base     # a different results file
"""
import argparse
import json
from collections import Counter

from common import parse_output
from sql_guard import execute_safe

ap = argparse.ArgumentParser()
ap.add_argument("categories", nargs="*", default=["trend", "vague_time"])
ap.add_argument("--run", default="finetuned")
ap.add_argument("--show", type=int, default=8)
args = ap.parse_args()

results = json.load(open(f"results/{args.run}.json"))["results"]
wrong = [r for r in results if not r["correct"]]

print(f"{len(wrong)} wrong out of {len(results)}\n")
print("wrong by category:", Counter(r["category"] for r in wrong).most_common())
print("wrong by kind    :", Counter(r["kind"] for r in wrong).most_common())
print("wrong by phrasing:", Counter(r["phrasing"] for r in wrong).most_common())
print("wrong by type of failure:", Counter(
    "asked to clarify on an answerable question" if r["kind"] != "ambiguous" and r["pred_kind"] == "clarify"
    else "did not ask on an ambiguous question" if r["kind"] == "ambiguous"
    else "wrong or invalid SQL" for r in wrong).most_common())


def run_sql(sql):
    try:
        cols, rows = execute_safe(sql, max_rows=1000)
    except Exception as e:
        return f"ERROR: {e}"
    return f"{len(rows)} rows, first: {rows[0] if rows else None}"


shown = 0
for r in wrong:
    if r["category"] not in args.categories or shown >= args.show:
        continue
    shown += 1
    print("\n" + "=" * 90)
    print(f"Q    : {r['question']!r}   [{r['kind']}, {r['phrasing']}]")
    print("GOLD :", r["target"])
    print("PRED :", r["raw"])
    kind, sql = parse_output(r["raw"])
    if r["kind"] != "ambiguous" and kind == "sql":
        print("gold result:", run_sql(r["target"]))
        print("pred result:", run_sql(sql))