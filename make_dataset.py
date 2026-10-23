"""Step 2: generate the labelled dataset (clear / ambiguous / resolved questions).

kind = "clear"      -> target is SQL
kind = "ambiguous"  -> target is "CLARIFY: <question>"
kind = "resolved"   -> question + "\\nClarification: <answer>" -> target is SQL

Each phrasing list ends with ONE held-out phrasing that is never used for
training. The test set mixes seen and held-out phrasings so you can report how
well the model generalises beyond templates (not just memorises them).
"""
import json
import random
import sqlite3

from common import DB_PATH, TODAY, METRICS, PLATFORMS, CHANNELS
from sql_guard import execute_safe

SEED = 42
WINDOWS = {  # text used in questions -> SQL predicate
    "the last 7 days": f"dp.report_date >= date('{TODAY}', '-6 day')",
    "the last 30 days": f"dp.report_date >= date('{TODAY}', '-29 day')",
    "July 2026": "dp.report_date BETWEEN '2026-07-01' AND '2026-07-31'",
    "August 2026": "dp.report_date BETWEEN '2026-08-01' AND '2026-08-31'",
    "September 2026": "dp.report_date BETWEEN '2026-09-01' AND '2026-09-30'",
}
BASE_FROM = ("FROM daily_performance dp JOIN line_items li ON dp.line_item_id = li.line_item_id")
FULL_FROM = (BASE_FROM + " JOIN campaigns c ON li.campaign_id = c.campaign_id"
             " JOIN advertisers a ON c.advertiser_id = a.advertiser_id")

con = sqlite3.connect(DB_PATH)
CAMPAIGNS = [r[0] for r in con.execute("SELECT name FROM campaigns")]
ADVERTISERS = [r[0] for r in con.execute("SELECT name FROM advertisers")]
con.close()


def pick(phrasings, pool, rng):
    """Last phrasing is reserved for the held-out pool."""
    return phrasings[-1] if pool == "heldout" else rng.choice(phrasings[:-1])


def scope(rng):
    """A random scope: (subject text, SQL condition)."""
    kind = rng.choice(["platform", "channel", "campaign", "advertiser"])
    if kind == "platform":
        v = rng.choice(PLATFORMS); return v, f"li.platform = '{v}'"
    if kind == "channel":
        v = rng.choice(CHANNELS); return v, f"li.channel = '{v}'"
    if kind == "campaign":
        v = rng.choice(CAMPAIGNS); return v, f"c.name = '{v}'"
    v = rng.choice(ADVERTISERS); return v, f"a.name = '{v}'"


def rank_sql(cond, pred, mk, n, direction):
    having = " HAVING SUM(dp.conversions) > 0" if mk == "cpa" else ""
    return (f"SELECT li.name, {METRICS[mk][1]} AS value {FULL_FROM} WHERE {cond} AND {pred} "
            f"GROUP BY li.line_item_id{having} ORDER BY value {direction} LIMIT {n}")


# ----------------------------- CLEAR templates -----------------------------
def t_metric_by_dim(rng, pool):
    mk = rng.choice(list(METRICS)); label, expr, _ = METRICS[mk]
    dim, dim_label = rng.choice([("platform", "platform"), ("channel", "channel"), ("deal_type", "deal type")])
    w, pred = rng.choice(list(WINDOWS.items()))
    q = pick(["What is the {m} by {d} in {w}?", "Show {m} per {d} for {w}.",
              "Break down {m} by {d} over {w}.",
              "Across each {d}, how does {m} look in {w}?"], pool, rng).format(m=label, d=dim_label, w=w)
    sql = (f"SELECT li.{dim}, {expr} AS value {BASE_FROM} WHERE {pred} "
           f"GROUP BY li.{dim} ORDER BY li.{dim}")
    return q, sql


def t_top_n(rng, pool):
    mk = rng.choice(["ctr", "cpm", "cpa", "roas", "spend", "conversions"]); label = METRICS[mk][0]
    plat = rng.choice(PLATFORMS); n = rng.choice([3, 5, 10]); w, pred = rng.choice(list(WINDOWS.items()))
    word, order = rng.choice([("highest", "DESC"), ("lowest", "ASC")])
    q = pick(["Which {n} line items on {p} had the {x} {m} in {w}?",
              "Show the {n} line items on {p} with the {x} {m} for {w}.",
              "List {n} line items on {p} ranked by {x} {m} over {w}.",
              "Give me the {x} {m} line items on {p} in {w}, {n} of them."],
             pool, rng).format(n=n, p=plat, x=word, m=label, w=w)
    return q, rank_sql(f"li.platform = '{plat}'", pred, mk, n, order)


def t_budget(rng, pool):
    c = rng.choice(CAMPAIGNS)
    q = pick(["How much has {c} spent so far compared with its budget?",
              "What is the total spend and budget for the {c} campaign?",
              "Show spend against budget for {c}.",
              "{c}: total spend vs total budget?"], pool, rng).format(c=c)
    sql = ("SELECT c.name, c.total_budget, SUM(dp.spend) AS spent FROM campaigns c "
           "JOIN line_items li ON li.campaign_id = c.campaign_id "
           "JOIN daily_performance dp ON dp.line_item_id = li.line_item_id "
           f"WHERE c.name = '{c}' GROUP BY c.campaign_id")
    return q, sql


def t_count(rng, pool):
    plat = rng.choice(PLATFORMS); dim, lab = rng.choice([("deal_type", "deal type"), ("channel", "channel")])
    q = pick(["How many line items does {p} have by {d}?", "Count the {p} line items per {d}.",
              "Give the number of line items on {p} for each {d}.",
              "On {p}, what is the line item count split by {d}?"], pool, rng).format(p=plat, d=lab)
    sql = f"SELECT {dim}, COUNT(*) FROM line_items WHERE platform = '{plat}' GROUP BY {dim} ORDER BY {dim}"
    return q, sql


def t_trend(rng, pool):
    mk = rng.choice(["spend", "impressions", "clicks", "ctr", "cpm", "conversions"]); label, expr, _ = METRICS[mk]
    c = rng.choice(CAMPAIGNS); w, pred = rng.choice(list(WINDOWS.items())[:2])
    q = pick(["Show daily {m} for {c} in {w}.", "What is the day-by-day {m} of {c} over {w}?",
              "Give me the daily trend of {m} for campaign {c} for {w}.",
              "Daily {m} for {c} during {w}, please."], pool, rng).format(m=label, c=c, w=w)
    sql = (f"SELECT dp.report_date, {expr} AS value {FULL_FROM} WHERE c.name = '{c}' AND {pred} "
           "GROUP BY dp.report_date ORDER BY dp.report_date")
    return q, sql


THRESHOLDS = {"ctr": [(0.001, "0.1%"), (0.002, "0.2%"), (0.003, "0.3%"), (0.005, "0.5%")],
              "roas": [(1, "1x"), (2, "2x"), (3, "3x"), (5, "5x")],
              "cpm": [(3, "$3"), (5, "$5"), (10, "$10"), (15, "$15")],
              "cpa": [(10, "$10"), (25, "$25"), (50, "$50"), (100, "$100")]}


def t_threshold(rng, pool):
    mk = rng.choice(list(THRESHOLDS)); label, expr, _ = METRICS[mk]
    thr, txt = rng.choice(THRESHOLDS[mk]); w, pred = rng.choice(list(WINDOWS.items()))
    q = pick(["Which line items had {m} below {t} in {w}?", "List line items with {m} under {t} for {w}.",
              "Find line items whose {m} was lower than {t} over {w}.",
              "Any line items with {m} < {t} in {w}?"], pool, rng).format(m=label, t=txt, w=w)
    sql = (f"SELECT li.name, {expr} AS value {BASE_FROM} WHERE {pred} "
           f"GROUP BY li.line_item_id HAVING {expr} < {thr} ORDER BY li.name")
    return q, sql


def t_advertiser(rng, pool):
    mk = rng.choice(list(METRICS)); label, expr, _ = METRICS[mk]
    a = rng.choice(ADVERTISERS); w, pred = rng.choice(list(WINDOWS.items()))
    q = pick(["What was the {m} for {a} in {w}?", "Give me {m} for advertiser {a} over {w}.",
              "Report {m} for {a} for {w}.", "{a}'s {m} in {w}, please."],
             pool, rng).format(m=label, a=a, w=w)
    sql = f"SELECT {expr} AS value {FULL_FROM} WHERE a.name = '{a}' AND {pred}"
    return q, sql


CLEAR = [t_metric_by_dim, t_top_n, t_budget, t_count, t_trend, t_threshold, t_advertiser]


# --------------------------- AMBIGUOUS templates ---------------------------
def a_vague_metric(rng, pool):
    subj, cond = scope(rng); w, pred = rng.choice(list(WINDOWS.items()))
    q = pick(["How did {s} perform in {w}?", "How is {s} doing in {w}?",
              "Was {s} doing well in {w}?", "Give me a performance summary of {s} for {w}."],
             pool, rng).format(s=subj, w=w)
    res = [(METRICS[mk][0], f"SELECT {METRICS[mk][1]} AS value {FULL_FROM} WHERE {cond} AND {pred}")
           for mk in ["ctr", "cpa", "roas", "spend"]]
    return dict(question=q, clarify="Which metric should I use for performance: CTR, CPA, ROAS, or total spend?",
                resolutions=res)


def a_vague_time(rng, pool):
    subj, cond = scope(rng); col = rng.choice(["spend", "impressions", "clicks", "conversions"])
    q = pick(["What was the total {m} for {s} recently?", "Show me {s} {m} lately.",
              "How much {m} did {s} deliver recently?",
              "{s} {m}: what is the latest figure?"], pool, rng).format(m=col, s=subj)
    res = [(w, f"SELECT SUM(dp.{col}) AS value {FULL_FROM} WHERE {cond} AND {pred}")
           for w, pred in WINDOWS.items()]
    return dict(question=q, clarify="What time range should I use: the last 7 days, the last 30 days, or a specific month?",
                resolutions=res)


def a_vague_ranking(rng, pool):
    subj, cond = scope(rng); w, pred = rng.choice(list(WINDOWS.items()))
    q = pick(["Which are the top 5 line items for {s} in {w}?", "Show the best 5 line items for {s} over {w}.",
              "List the 5 best performing line items for {s} in {w}.",
              "Who are the five leading line items for {s} in {w}?"], pool, rng).format(s=subj, w=w)
    res = [(METRICS[mk][0], rank_sql(cond, pred, mk, 5, "ASC" if METRICS[mk][2] else "DESC"))
           for mk in ["ctr", "cpa", "roas", "spend"]]
    return dict(question=q, clarify="What should I rank line items by: CTR, CPA, ROAS, or total spend?",
                resolutions=res)


def a_vague_cost(rng, pool):
    subj, cond = scope(rng); w, pred = rng.choice(list(WINDOWS.items()))
    q = pick(["What was the cost of {s} in {w}?", "What did {s} cost us in {w}?",
              "Tell me the cost for {s} over {w}.", "{s} costs in {w}?"], pool, rng).format(s=subj, w=w)
    res = [(METRICS[mk][0], f"SELECT {METRICS[mk][1]} AS value {FULL_FROM} WHERE {cond} AND {pred}")
           for mk in ["spend", "cpm", "cpa"]]
    return dict(question=q, clarify="Which cost do you mean: total spend, CPM, or CPA?", resolutions=res)


AMBIGUOUS = [a_vague_metric, a_vague_time, a_vague_ranking, a_vague_cost]


# -------------------------------- assembly ---------------------------------
def informative(sql):
    """Reject gold queries that return nothing: an empty result would 'match' any empty prediction."""
    _, rows = execute_safe(sql, max_rows=5)
    return bool(rows) and not all(v is None for v in rows[0])


def build_split(rng, n, heldout_frac, exclude):
    records, seen, tries = [], set(), 0
    while len(records) < n and tries < n * 80:
        tries += 1
        pool = "heldout" if rng.random() < heldout_frac else "train"
        r = rng.random()
        if r < 0.70:
            fn = rng.choice(CLEAR); q, sql = fn(rng, pool)
            rec = dict(question=q, target=sql, kind="clear", category=fn.__name__[2:])
        else:
            fn = rng.choice(AMBIGUOUS); item = fn(rng, pool)
            if r < 0.85:
                rec = dict(question=item["question"], target="CLARIFY: " + item["clarify"],
                           kind="ambiguous", category=fn.__name__[2:])
            else:
                ans, sql = rng.choice(item["resolutions"])
                ans = rng.choice([ans, f"Use {ans}", f"I mean {ans}"])
                rec = dict(question=f"{item['question']}\nClarification: {ans}", target=sql,
                           kind="resolved", category=fn.__name__[2:])
        if rec["question"] in seen or rec["question"] in exclude:
            continue
        if rec["kind"] != "ambiguous" and not informative(rec["target"]):
            continue
        seen.add(rec["question"])
        rec["phrasing"] = pool if pool == "heldout" else "seen"
        records.append(rec)
    for i, rec in enumerate(records):
        rec["id"] = i
    return records


def main():
    rng = random.Random(SEED)
    train = build_split(rng, 3000, 0.0, set())
    train_q = {r["question"] for r in train}
    val = build_split(rng, 200, 0.0, train_q)
    test = build_split(rng, 400, 0.5, train_q | {r["question"] for r in val})
    for name, data in [("train", train), ("val", val), ("test", test)]:
        with open(f"data/{name}.jsonl", "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
        kinds = {k: sum(r["kind"] == k for r in data) for k in ("clear", "ambiguous", "resolved")}
        print(f"{name}: {len(data)} examples {kinds}")


if __name__ == "__main__":
    main()
