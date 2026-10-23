"""Step 1: build a synthetic ad-tech campaign database (no client data involved)."""
import datetime as dt
import random
import sqlite3

from common import DB_PATH, TODAY, PLATFORMS

random.seed(7)
START = dt.date(2026, 7, 1)
END = dt.date.fromisoformat(TODAY)

ADVERTISERS = [
    ("Nutrio Infant", "Nutrition"), ("BrightBite Foods", "Food & Beverage"),
    ("Urban Threads", "Fashion"), ("PeakFit Gear", "Sports"),
    ("CloudNest Bank", "Finance"), ("TripWing Travel", "Travel"),
]
THEMES = ["Back to School", "Festive Push", "Summer Launch", "Always On",
          "Loyalty Reactivation", "New Product Teaser"]
OBJECTIVES = ["Awareness", "Consideration", "Conversion"]
DEAL_TYPES = ["Open Auction", "PMP", "PG"]
PLATFORM_CHANNELS = {
    "DV360": ["Display", "Video", "CTV", "Audio"],
    "TTD": ["Display", "Video", "CTV", "Audio"],
    "Google Ads": ["Display", "Video"],
}
# channel -> (base CPM $, base CTR, click->conv rate, video completion rate, viewability)
PROFILE = {
    "Display": (3.0, 0.0040, 0.020, 0.00, 0.55),
    "Video": (8.0, 0.0060, 0.015, 0.55, 0.70),
    "CTV": (18.0, 0.0010, 0.010, 0.85, 0.95),
    "Audio": (6.0, 0.0020, 0.012, 0.00, 0.80),
}

DDL = """
CREATE TABLE advertisers(advertiser_id INTEGER PRIMARY KEY, name TEXT, industry TEXT);
CREATE TABLE campaigns(campaign_id INTEGER PRIMARY KEY, advertiser_id INTEGER, name TEXT UNIQUE,
    objective TEXT, start_date TEXT, end_date TEXT, total_budget REAL);
CREATE TABLE line_items(line_item_id INTEGER PRIMARY KEY, campaign_id INTEGER, name TEXT UNIQUE,
    platform TEXT, channel TEXT, deal_type TEXT, daily_budget REAL);
CREATE TABLE daily_performance(report_date TEXT, line_item_id INTEGER, impressions INTEGER,
    clicks INTEGER, spend REAL, conversions INTEGER, revenue REAL,
    video_completions INTEGER, viewable_impressions INTEGER);
CREATE INDEX idx_dp_date ON daily_performance(report_date);
CREATE INDEX idx_dp_li ON daily_performance(line_item_id);
"""


def main():
    DB_PATH.parent.mkdir(exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    con.executescript(DDL)

    perf_rows, li_id, camp_id = [], 0, 0
    for adv_id, (adv_name, industry) in enumerate(ADVERTISERS, start=1):
        con.execute("INSERT INTO advertisers VALUES (?,?,?)", (adv_id, adv_name, industry))
        aov = random.uniform(20, 120)  # average order value drives revenue
        for theme in random.sample(THEMES, 4):
            camp_id += 1
            start = START + dt.timedelta(days=random.randint(0, 30))
            end = start + dt.timedelta(days=random.randint(45, 120))
            daily_total = 0.0
            for _ in range(random.randint(3, 6)):
                li_id += 1
                platform = random.choice(PLATFORMS)
                channel = random.choice(PLATFORM_CHANNELS[platform])
                deal = random.choice(DEAL_TYPES)
                budget = round(random.uniform(150, 1500), 2)
                daily_total += budget
                con.execute("INSERT INTO line_items VALUES (?,?,?,?,?,?,?)", (
                    li_id, camp_id, f"LI{li_id:03d} {platform} {channel} {deal}",
                    platform, channel, deal, budget))
                cpm0, ctr0, cr, vcr, view = PROFILE[channel]
                cpm_f, q_f = random.uniform(0.8, 1.3), random.uniform(0.7, 1.4)
                day = max(start, START)
                while day <= min(end, END):
                    spend = budget * random.uniform(0.7, 1.1)
                    cpm = cpm0 * cpm_f * random.uniform(0.9, 1.1)
                    imps = int(spend / cpm * 1000)
                    clicks = int(imps * ctr0 * q_f * random.uniform(0.85, 1.15))
                    convs = int(clicks * cr * random.uniform(0.7, 1.3) + imps * 0.00001)
                    perf_rows.append((
                        day.isoformat(), li_id, imps, clicks, round(spend, 2), convs,
                        round(convs * aov * random.uniform(0.8, 1.2), 2),
                        int(imps * vcr * random.uniform(0.9, 1.05)),
                        int(imps * view * random.uniform(0.95, 1.0))))
                    day += dt.timedelta(days=1)
            total_budget = round(daily_total * (end - start).days * random.uniform(0.9, 1.2), -2)
            con.execute("INSERT INTO campaigns VALUES (?,?,?,?,?,?,?)", (
                camp_id, adv_id, f"{adv_name} - {theme}", random.choice(OBJECTIVES),
                start.isoformat(), end.isoformat(), total_budget))

    con.executemany("INSERT INTO daily_performance VALUES (?,?,?,?,?,?,?,?,?)", perf_rows)
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM line_items").fetchone()[0]
    print(f"DB ready: {camp_id} campaigns, {n} line items, {len(perf_rows)} daily rows -> {DB_PATH}")
    con.close()


if __name__ == "__main__":
    main()
