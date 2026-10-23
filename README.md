# ClarifySQL: clarification-aware text-to-SQL for ad-tech analytics

A small open model (Qwen2.5-1.5B-Instruct) fine-tuned with QLoRA to do two jobs on a campaign-performance database:

1. **Answer** clear questions with SQL.
2. **Ask back** when a question is ambiguous (no metric, no time range, no ranking criterion) instead of guessing.

Wrong-but-confident SQL is the most common failure of text-to-SQL systems. In ad-tech, "how did DV360 perform?" could mean CTR, CPA, ROAS or spend, so guessing silently gives the wrong number.

| Question | Model output |
|---|---|
| `What was the CTR for DV360 in July 2026?` | `SELECT SUM(dp.clicks) * 1.0 / SUM(dp.impressions) ...` |
| `How did DV360 perform in July 2026?` | `CLARIFY: Which metric should I use for performance: CTR, CPA, ROAS, or total spend?` |
| `How did DV360 perform in July 2026?` + `Clarification: ROAS` | `SELECT SUM(dp.revenue) * 1.0 / SUM(dp.spend) ...` |

## Results

Fine-tuned model on a 400-question test set the model never saw (execution accuracy = the query returns the same rows as the gold query):

| Metric | Result |
|---|---|
| Overall accuracy | 85.0% |
| SQL accuracy, clear questions | 83.3% |
| SQL accuracy, after a clarification | 79.8% |
| Clarification precision / recall / F1 | 98.6% / 98.6% / 98.6% |
| Needless clarification on answerable questions | 0.3% |
| Accuracy on training-style phrasings (136 questions) | 100% |
| **Accuracy on unseen phrasings (264 questions)** | **77.3%** |
| Avg latency | 9.3 s/question (Colab T4, 4-bit, unbatched, not optimised) |

**Not yet measured:** the un-tuned base model and a prompted API model on the same test set. Until those runs exist, this README makes no claim about how much fine-tuning improved accuracy.

Training: 3,000 train / 200 validation examples, QLoRA (rank 16, all linear layers), 2 epochs = 376 steps, fp16 on a free Colab T4.

## What the error analysis found

All 60 test errors were on phrasings the model never saw in training (0 errors on seen phrasings), so the model learned the templates but generalises imperfectly to new wording:

- **`trend` (30 errors):** for "Daily X for Y during Z, please", every inspected failure returned one total instead of a day-by-day series (dropped `report_date` and `GROUP BY`).
- **`vague_time` (22 SQL errors + 1 missed clarification):** after a clarification, "what is the latest figure?" made the model use `MAX(...)` instead of `SUM(...)`; one query invented a non-existent column.
- **`top_n` (3 errors):** `MAX`/`MIN(conversions)` instead of `SUM`.
- **`budget` (2 errors):** one truncated a campaign name; one asked for a time range on a question that has none (my data labels and prompt rules disagree there, a dataset flaw).
- **`threshold` (2 errors):** not inspected yet.

The held-out score is lumpy: each template has only one held-out phrasing, so a single confusing sentence causes many errors at once. It measures sensitivity to those wordings, not a general paraphrase rate.

## How it works

1. `make_db.py` builds a synthetic ad-tech database (24 campaigns, 103 line items, ~7k daily rows). No client data.
2. `make_dataset.py` generates three kinds of examples: *clear* (SQL), *ambiguous* (`CLARIFY: ...`) and *resolved* (question + clarification answer, then SQL). The last phrasing of every template is held out of training.
3. `finetune.py` trains a LoRA adapter on the 4-bit base model; loss is computed on the answer only.
4. `eval_model.py` scores any backend by executing predicted and gold SQL and comparing rows, plus clarification precision/recall.
5. `serve.py` exposes a FastAPI `/ask` endpoint: the model either asks a clarifying question or returns SQL that is executed through `sql_guard.py` (read-only connection, SQLite authorizer, timeout).

## Design decisions

| Decision | Why |
|---|---|
| Synthetic data, fixed "today" (2026-09-30) | Reproducible; relative dates like "last 7 days" have one correct answer. |
| Execution accuracy, not string match | Different SQL can be equally correct. |
| Held-out phrasings in the test set | Separates memorising templates from generalising. |
| `resolved` examples in training | Without them the model keeps asking instead of using the user's answer. |
| Gold queries with empty results removed | An empty result would "match" any empty prediction. |
| QLoRA + loss on the answer only | Fits a free T4; the long schema prompt is not "learned". |
| Read-only DB + authorizer + timeout | The SQL is written by an LLM; it never gets write access. |
| Same prompt for training, evaluation and serving | The model only sees the behaviour it was trained on, and comparisons are fair. |

## Limitations

- Training data is template-generated, so real user questions will be messier.
- Ambiguity labels follow one convention ("cost" is ambiguous; "performance" needs a named metric). Another team would label differently.
- Only ambiguity detectable from the question text is covered; entity ambiguity ("the Summer campaign" matching several) needs database access.
- A 1.5B model may lose to large API models on raw accuracy; the realistic win is cost and latency on a narrow domain, and that is still to be measured.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
python make_db.py            # data/campaigns.db
python make_dataset.py       # data/{train,val,test}.jsonl
python tests/test_basic.py   # sanity checks
python eval_model.py --backend gold --run oracle   # harness check (must be 100%)
```

Training and model evaluation need a GPU (a free Colab T4 works):

```bash
pip install -U transformers peft trl bitsandbytes datasets accelerate
python finetune.py --model Qwen/Qwen2.5-1.5B-Instruct --out outputs/lora
python eval_model.py --backend hf --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/lora --run finetuned
python analyze_errors.py                      # inspect failures (no GPU needed)
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct ADAPTER_DIR=outputs/lora uvicorn serve:app --port 8000
```

## Roadmap

- Run the base-model and prompted-API baselines on the same test set.
- v2 data: 8-10 training phrasings per template and a fresh held-out set with several unseen phrasings per template; fix the budget labelling.
- Cut latency: merge the adapter, batch requests, shorten the prompt.

## Process note

Code scaffolding was written with an AI assistant. Training runs, evaluation and error analysis were run and interpreted by the author.
