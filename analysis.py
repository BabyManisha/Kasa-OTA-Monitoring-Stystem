# analysis.py
import io
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import requests


# =========================
# OLLAMA – LLM INSIGHTS
# =========================

def generate_ollama_insights(
    week_label: str,
    alerts_df: pd.DataFrame,
    best_df: pd.DataFrame,
    model: str = "gemma3:latest",
) -> str:
    """
    Call Ollama running locally to generate a rich narrative.
    Ollama must be running on http://localhost:11434 with the given model pulled.
    """

    alerts_csv = alerts_df.head(60).to_csv(index=False)
    best_csv = best_df.head(30).to_csv(index=False)

    prompt = f"""
You are an OTA performance and revenue optimization specialist.

Week: {week_label}

[ALERT_LISTINGS]
{alerts_csv}

[BEST_PERFORMERS]
{best_csv}

Using ONLY this data:

1) EXECUTIVE SUMMARY
   - 4–6 bullet points summarizing the overall health of the portfolio this week.
   - Explicitly mention:
     • Conversion Rate patterns
     • Search Visibility patterns
     • Velocity trends (week-over-week bookings)
     • Listing Stability (stable vs volatile listings).

2) PRIORITY ALERTS (P1 / P2 / P3)
   - For each priority level:
     • Describe common patterns you see (e.g. “many P1 listings lost bookings but kept traffic”).
     • Call out likely ROOT CAUSES conceptually (pricing, content, fees, search ranking, restrictions, etc.).
     • Suggest 2–3 targeted actions per priority level.

3) BEST PERFORMERS
   - 3–5 bullets on what the top listings seem to be doing well.
   - Add a short “playbook-style” list of what we can replicate from these listings.

4) GENERAL SUGGESTIONS & EXPERIMENT IDEAS
   - 5–7 concrete ideas that are not tied to a single listing, for the next 1–2 weeks
     (pricing tests, policy tweaks, photo/description updates, promotions, etc).

5) NOTES & WATCHOUTS
   - Mention any data quality concerns or gaps you infer.
   - Suggest what additional data would help future analyses.

Style:
- Under ~550 words.
- Use short paragraphs and bullet points.
- Be practical and specific, not fluffy.
"""

    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


# =========================
# METRICS + HELPERS
# =========================

def load_all_weeks(excel_bytes: bytes) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load one Excel workbook (multi-sheet: each sheet is a week).
    Returns a single DataFrame with a 'week' column + list of sheet names.
    """
    xls = pd.ExcelFile(io.BytesIO(excel_bytes))
    sheets = xls.sheet_names
    frames = []
    for s in sheets:
        df = pd.read_excel(xls, sheet_name=s)
        df["week"] = s
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    return data, sheets


def compute_basic_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
    - conversion_rate = bookings / total_listing_views
    - search_visibility = total_listing_views / appearance_in_search
    """
    df = df.copy()

    df["conversion_rate"] = df["bookings"] / df["total_listing_views"].replace({0: np.nan})
    df["conversion_rate"] = df["conversion_rate"].fillna(0)

    df["search_visibility"] = df["total_listing_views"] / df["appearance_in_search"].replace({0: np.nan})
    df["search_visibility"] = df["search_visibility"].fillna(0)

    return df


def add_stability_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a simple stability index per listing across ALL weeks:
    - calculate mean/std for bookings and views
    - derive volatility, then stability_index = max(0, 1 - volatility)
    """
    base = df.copy()

    grouped = base.groupby("id_listing").agg(
        bookings_mean=("bookings", "mean"),
        bookings_std=("bookings", "std"),
        views_mean=("total_listing_views", "mean"),
        views_std=("total_listing_views", "std"),
    ).reset_index()

    grouped["bookings_vol"] = grouped["bookings_std"] / (grouped["bookings_mean"] + 1)
    grouped["views_vol"] = grouped["views_std"] / (grouped["views_mean"] + 1)
    grouped["volatility"] = (grouped["bookings_vol"] + grouped["views_vol"]) / 2
    grouped["stability_index"] = (1 - grouped["volatility"]).clip(0, 1)

    merged = base.merge(grouped[["id_listing", "stability_index"]], on="id_listing", how="left")
    return merged


def add_multi_week_wow(df: pd.DataFrame, weeks: List[str]) -> Tuple[pd.DataFrame, str, str]:
    """
    Compute week-over-week deltas for *all* weeks (except the first),
    but still return last_week and previous_week labels for convenience.

    For each listing + week we attach:
    - *_prev columns from the prior week (if exists)
    - delta_* and rel_change_* columns
    - velocity_bookings / velocity_bookings_pct
    """
    week_order = {w: i for i, w in enumerate(weeks)}

    df = df.copy()
    df["week_index"] = df["week"].map(week_order)

    # Current rows
    cur = df.copy()

    # Previous week rows (shift week_index by +1 so they align when merging)
    prev = df.copy()
    prev["week_index"] = prev["week_index"] + 1

    cols_to_prev = [
        "id_listing",
        "week_index",
        "appearance_in_search",
        "total_listing_views",
        "bookings",
        "conversion_rate",
        "search_visibility",
    ]

    prev = prev[cols_to_prev].rename(columns={
        "week_index": "week_index_prev",
        "appearance_in_search": "appearance_in_search_prev",
        "total_listing_views": "total_listing_views_prev",
        "bookings": "bookings_prev",
        "conversion_rate": "conversion_rate_prev",
        "search_visibility": "search_visibility_prev",
    })

    merged = cur.merge(
        prev,
        left_on=["id_listing", "week_index"],
        right_on=["id_listing", "week_index_prev"],
        how="left",
    )

    # Deltas + relative changes for ALL weeks that have a previous week
    for col in ["appearance_in_search", "total_listing_views", "bookings",
                "conversion_rate", "search_visibility"]:
        base_prev = merged.get(col + "_prev")
        merged[f"delta_{col}"] = merged[col] - base_prev
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = merged[f"delta_{col}"] / base_prev.replace({0: np.nan})
        merged[f"rel_change_{col}"] = rel.replace([np.inf, -np.inf], np.nan)

    # Velocity = bookings change WoW
    merged["velocity_bookings"] = merged["delta_bookings"]
    merged["velocity_bookings_pct"] = merged["rel_change_bookings"]

    last_week = weeks[-1]
    prev_week = weeks[-2] if len(weeks) > 1 else None

    return merged, last_week, prev_week


def score_alert(row: pd.Series) -> Dict:
    """
    Classify a listing row into P1 / P2 / P3 alert, or 0 for no alert.
    """
    reasons: List[str] = []
    priority: Optional[int] = None

    views = row["total_listing_views"]
    views_prev = row.get("total_listing_views_prev") or 0
    conv = row["conversion_rate"]
    conv_prev = row.get("conversion_rate_prev") or 0
    searches = row["appearance_in_search"]
    searches_prev = row.get("appearance_in_search_prev") or 0
    bookings = row["bookings"]
    bookings_prev = row.get("bookings_prev") or 0

    # --- P1: severe problems ---
    if views >= 50 and bookings_prev >= 1 and bookings == 0:
        reasons.append("Bookings dropped to 0 despite decent traffic")
        priority = 1

    if conv_prev > 0 and conv < conv_prev * 0.3:
        reasons.append("Conversion rate dropped >70% WoW")
        priority = 1 if priority is None or priority > 1 else priority

    if searches_prev >= 100 and searches < searches_prev * 0.3:
        reasons.append("Search impressions dropped >70% WoW")
        priority = 1 if priority is None or priority > 1 else priority

    # --- P2: medium problems ---
    if priority is None:
        if conv_prev > 0 and conv < conv_prev * 0.7:
            reasons.append("Conversion rate dropped 30–70% WoW")
            priority = 2
        elif searches_prev > 0 and searches < searches_prev * 0.7:
            reasons.append("Search impressions dropped 30–70% WoW")
            priority = 2

    # --- P3: watchlist ---
    if priority is None and views >= 50 and conv < 0.02:
        reasons.append("Low conversion with meaningful traffic")
        priority = 3

    return {
        "priority": priority or 0,
        "reasons": "; ".join(reasons) if reasons else "",
    }


def find_best_performers(wow_df: pd.DataFrame) -> pd.DataFrame:
    """
    Best performers for this week:
    - decent traffic
    - strong conversion
    - reasonably stable
    """
    df = wow_df.copy()

    df = df[df["total_listing_views"] >= 50]
    df = df[df["conversion_rate"] >= 0.05]
    df = df[df["stability_index"].fillna(0) >= 0.5]

    df = df.sort_values(
        ["conversion_rate", "velocity_bookings", "stability_index"],
        ascending=[False, False, False],
    )

    cols = [
        "week",
        "id_listing",
        "bookings",
        "total_listing_views",
        "conversion_rate",
        "search_visibility",
        "velocity_bookings",
        "velocity_bookings_pct",
        "stability_index",
    ]

    existing_cols = [c for c in cols if c in df.columns]
    return df[existing_cols].head(20)


# =========================
# MAIN ENTRYPOINT FOR API
# =========================

def build_insights_excel(excel_bytes: bytes) -> Tuple[bytes, Dict]:
    """
    Main orchestration:
    - read input Excel (multi-sheet, weekly)
    - compute metrics & stability
    - compute WoW for ALL weeks
    - compute alerts & best performers for latest week
    - call Ollama for narrative
    - return:
        * Excel (as bytes) with:
          - per-week sheets (with metrics + WoW)
          - 'alerts' sheet
          - 'best_performers' sheet
          - 'llm_insights' sheet
        * JSON summary dict for API / Slack
    """
    # 1) Load + metrics
    all_df, weeks = load_all_weeks(excel_bytes)
    all_df = compute_basic_metrics(all_df)
    all_df = add_stability_index(all_df)

    # 2) WoW for ALL weeks, and get last/prev labels
    df_with_wow, last_week, prev_week = add_multi_week_wow(all_df, weeks)

    # We'll use last_week_rows for alerts & best performers
    last_week_rows = df_with_wow[df_with_wow["week"] == last_week].copy()

    # 3) Alerts (only latest week)
    alert_rows: List[Dict] = []
    for _, row in last_week_rows.iterrows():
        scored = score_alert(row)
        if scored["priority"] == 0:
            continue

        alert_rows.append({
            "week": last_week,
            "id_listing": row["id_listing"],
            "priority": scored["priority"],
            "reasons": scored["reasons"],

            "appearance_in_search": row["appearance_in_search"],
            "appearance_in_search_prev": row.get("appearance_in_search_prev"),
            "delta_appearance_in_search": row.get("delta_appearance_in_search"),
            "rel_change_appearance_in_search": row.get("rel_change_appearance_in_search"),

            "total_listing_views": row["total_listing_views"],
            "total_listing_views_prev": row.get("total_listing_views_prev"),
            "delta_total_listing_views": row.get("delta_total_listing_views"),
            "rel_change_total_listing_views": row.get("rel_change_total_listing_views"),

            "bookings": row["bookings"],
            "bookings_prev": row.get("bookings_prev"),
            "delta_bookings": row.get("delta_bookings"),
            "rel_change_bookings": row.get("rel_change_bookings"),

            "conversion_rate": row["conversion_rate"],
            "conversion_rate_prev": row.get("conversion_rate_prev"),
            "delta_conversion_rate": row.get("delta_conversion_rate"),
            "rel_change_conversion_rate": row.get("rel_change_conversion_rate"),

            "search_visibility": row["search_visibility"],
            "search_visibility_prev": row.get("search_visibility_prev"),
            "delta_search_visibility": row.get("delta_search_visibility"),
            "rel_change_search_visibility": row.get("rel_change_search_visibility"),

            "velocity_bookings": row.get("velocity_bookings"),
            "velocity_bookings_pct": row.get("velocity_bookings_pct"),
            "stability_index": row.get("stability_index"),
        })

    alerts_df = pd.DataFrame(alert_rows)
    if not alerts_df.empty:
        alerts_df.sort_values(
            ["priority", "velocity_bookings"],
            ascending=[True, True],
            inplace=True,
        )

    # 4) Best performers (latest week, with WoW + stability)
    best_df = find_best_performers(last_week_rows) if not last_week_rows.empty else pd.DataFrame()

    # 5) LLM (Ollama) insights
    insights_text = ""
    if not alerts_df.empty or not best_df.empty:
        insights_text = generate_ollama_insights(last_week, alerts_df, best_df)

    # 6) Build Excel in memory
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        # Week-by-week sheets (same names as input, truncated to 31 chars)
        for w in weeks:
            week_df = df_with_wow[df_with_wow["week"] == w].copy()
            sheet_name = w[:31]
            week_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Alerts sheet
        if not alerts_df.empty:
            alerts_df.to_excel(writer, sheet_name="alerts", index=False)

        # Best performers sheet
        if not best_df.empty:
            best_df.to_excel(writer, sheet_name="best_performers", index=False)

        # LLM insights sheet
        pd.DataFrame(
            [{"week": last_week, "insights": insights_text}]
        ).to_excel(writer, sheet_name="llm_insights", index=False)

    buffer.seek(0)
    excel_out = buffer.getvalue()

    # 7) JSON summary for API (grouped alerts)
    grouped: Dict[str, List[Dict]] = {"1": [], "2": [], "3": []}
    for _, r in alerts_df.iterrows():
        p = int(r["priority"])
        if p not in (1, 2, 3):
            continue
        grouped[str(p)].append({
            "id_listing": str(r["id_listing"]),
            "reasons": r["reasons"],
            "conversion_rate": float(r.get("conversion_rate") or 0),
            "conversion_rate_prev": float(r.get("conversion_rate_prev") or 0),
            "velocity_bookings": int(r.get("velocity_bookings") or 0),
            "stability_index": float(r.get("stability_index") or 0),
        })

    summary = {
        "last_week": last_week,
        "previous_week": prev_week,
        "alerts": grouped,
        "llm_insights": insights_text,
    }

    return excel_out, summary


# Optional: quick CLI test on a local file
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python analysis.py input.xlsx output_insights.xlsx")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    with open(in_path, "rb") as f:
        excel_bytes = f.read()

    excel_out, summary = build_insights_excel(excel_bytes)
    with open(out_path, "wb") as f:
        f.write(excel_out)

    print("Wrote:", out_path)
    print("Summary keys:", summary.keys())
