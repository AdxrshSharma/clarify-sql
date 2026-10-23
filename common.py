"""Shared constants, the system prompt, metric definitions and output parsing.

Everything that the dataset generator, the evaluator, the trainer and the API
must agree on lives here, so the model sees an identical prompt everywhere.
"""
import re
from pathlib import Path

ROOT = Path(__file__).parent
DB_PATH = ROOT / "data" / "campaigns.db"
TODAY = "2026-09-30"  # fixed "today" so relative dates ("last 7 days") are reproducible

PLATFORMS = ["DV360", "TTD", "Google Ads"]
CHANNELS = ["Display", "Video", "CTV", "Audio"]
DEAL_TYPES = ["Open Auction", "PMP", "PG"]

# key -> (label used in questions, SQL expression, lower_is_better)
METRICS = {
    "ctr": ("CTR", "SUM(dp.clicks) * 1.0 / SUM(dp.impressions)", False),
    "cpm": ("CPM", "SUM(dp.spend) * 1000.0 / SUM(dp.impressions)", True),
    "cpa": ("CPA", "SUM(dp.spend) * 1.0 / SUM(dp.conversions)", True),
    "roas": ("ROAS", "SUM(dp.revenue) * 1.0 / SUM(dp.spend)", False),
    "spend": ("total spend", "SUM(dp.spend)", False),
    "impressions": ("total impressions", "SUM(dp.impressions)", False),
    "clicks": ("total clicks", "SUM(dp.clicks)", False),
    "conversions": ("total conversions", "SUM(dp.conversions)", False),
}

SCHEMA = """\
advertisers(advertiser_id, name, industry)
campaigns(campaign_id, advertiser_id, name, objective, start_date, end_date, total_budget)
line_items(line_item_id, campaign_id, name, platform, channel, deal_type, daily_budget)
daily_performance(report_date, line_item_id, impressions, clicks, spend, conversions, revenue, video_completions, viewable_impressions)

Values:
- platform: 'DV360', 'TTD', 'Google Ads'
- channel: 'Display', 'Video', 'CTV', 'Audio'
- deal_type: 'Open Auction', 'PMP', 'PG'
- objective: 'Awareness', 'Consideration', 'Conversion'
- dates are 'YYYY-MM-DD' text; spend and revenue are in USD."""

SYSTEM_PROMPT = f"""You are a SQL assistant for an ad-tech campaign database (SQLite).
Today's date is {TODAY}.

Schema:
{SCHEMA}

Metric definitions (aggregate first, then divide):
- CTR = SUM(clicks) * 1.0 / SUM(impressions)
- CPM = SUM(spend) * 1000.0 / SUM(impressions)
- CPA = SUM(spend) * 1.0 / SUM(conversions)
- ROAS = SUM(revenue) * 1.0 / SUM(spend)

Conventions:
- Aliases: dp = daily_performance, li = line_items, c = campaigns, a = advertisers.
- "last N days" = the N days ending today, inclusive: dp.report_date >= date('{TODAY}', '-(N-1) day').
- A month name means that calendar month of 2026.
- Use only SELECT. Never modify data.

Output rules:
- If one clear SQL query answers the question, reply with ONLY the SQL (no markdown, no explanation).
- Ask for clarification when the question (1) says "performance", "best" or "doing well" without naming a metric,
  (2) says "cost" without saying spend, CPM or CPA, or (3) gives no time range ("recently", "lately").
  Reply with exactly one line: CLARIFY: <short question that offers options>.
- Never ask for clarification when metric, time range and scope are all given.
- If the message contains a line starting with "Clarification:", use it and reply with SQL."""


def build_messages(question: str):
    """Chat messages used identically for training, evaluation and serving."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def parse_output(text: str):
    """Return ('clarify', question) or ('sql', query) from raw model output."""
    t = text.strip()
    t = re.sub(r"^```(?:sql)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t).strip()
    m = re.match(r"(?is)^CLARIFY\s*:\s*(.+)$", t)
    if m:
        return "clarify", m.group(1).strip()
    return "sql", t.rstrip(";").strip()
