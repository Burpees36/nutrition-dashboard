import pandas as pd


ADHERENCE_MAP = {
    "Most days": 1.0,
    "Some days": 0.5,
    "Very few days": 0.0
}

SLEEP_BIN_MAP = {
    "Less than 5": 4.5,
    "5-6": 5.5,
    "6-7": 6.5,
    "7-8": 7.5,
    "8+": 8.5
}

# Expected headers you provided (used for validation)
REQUIRED_INTAKE = {
    "timestamp", "email",
    "bodyweight_lbs_baseline", "rhr_bpm_baseline",
    "sleep_quality_baseline", "energy_baseline"
}

REQUIRED_WEEKLY_MIN = {"timestamp", "email", "week_number"}


def normalize_common(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "email" in df.columns:
        df["email"] = df["email"].astype(str).str.strip().str.lower()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def parse_week_number(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace("Week", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def cast_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_config(path: str) -> dict:
    """
    Optional config.csv columns:
      challenge_name, start_date, end_date, week_count, coach_email
    """
    cfg = {"challenge_name": "Nutrition Challenge"}
    try:
        df = pd.read_csv(path)
        if len(df) >= 1:
            row = df.iloc[0].to_dict()
            name = str(row.get("challenge_name", "")).strip()
            if name:
                cfg["challenge_name"] = name
            cfg["start_date"] = pd.to_datetime(row.get("start_date", None), errors="coerce")
            cfg["end_date"] = pd.to_datetime(row.get("end_date", None), errors="coerce")
            if pd.notna(row.get("week_count", None)):
                cfg["week_count"] = int(row.get("week_count"))
            cfg["coach_email"] = str(row.get("coach_email", "")).strip()
    except Exception:
        pass
    return cfg


def dedupe_weekly_keep_latest(weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep latest submission per email+week_number (based on timestamp).
    Returns: (weekly_deduped, duplicates_table)
    """
    if not {"email", "week_number"}.issubset(weekly.columns):
        return weekly, pd.DataFrame()

    wk = weekly.copy()
    wk = wk.sort_values("timestamp") if "timestamp" in wk.columns else wk.sort_index()

    dup_mask = wk.duplicated(subset=["email", "week_number"], keep=False)
    dup_cols = [c for c in ["email", "week_number", "timestamp"] if c in wk.columns]
    duplicates = wk.loc[dup_mask, dup_cols].sort_values(
        ["email", "week_number"] + (["timestamp"] if "timestamp" in wk.columns else [])
    )

    wk = wk.drop_duplicates(subset=["email", "week_number"], keep="last")
    return wk, duplicates


def validate_intake(intake: pd.DataFrame) -> list[str]:
    missing = sorted(list(REQUIRED_INTAKE - set(intake.columns)))
    return missing


def validate_weekly(weekly: pd.DataFrame) -> list[str]:
    missing = sorted(list(REQUIRED_WEEKLY_MIN - set(weekly.columns)))
    return missing


def prep_data(intake: pd.DataFrame, weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      merged_df: weekly rows left-joined to intake by email
      weekly_clean: cleaned/deduped weekly
    """
    intake = normalize_common(intake)
    weekly = normalize_common(weekly)

    weekly["week_number"] = parse_week_number(weekly["week_number"])

    # Map categorical bins -> numeric where relevant
    if "nutrition_adherence_weekly" in weekly.columns:
        weekly["adherence_score_weekly"] = weekly["nutrition_adherence_weekly"].map(ADHERENCE_MAP)

    if "sleep_hours_weekly" in weekly.columns:
        weekly["sleep_hours_numeric_weekly"] = weekly["sleep_hours_weekly"].map(SLEEP_BIN_MAP)

    if "sleep_hours_baseline" in intake.columns:
        intake["sleep_hours_numeric_baseline"] = intake["sleep_hours_baseline"].map(SLEEP_BIN_MAP)

    # Numeric casts
    intake = cast_numeric(intake, [
        "bodyweight_lbs_baseline", "rhr_bpm_baseline",
        "sleep_quality_baseline", "energy_baseline",
        "stress_baseline", "classes_per_week_baseline",
        "whole_food_days_per_week_baseline", "alcohol_days_per_week_baseline",
        "takeout_per_week_baseline"
    ])

    weekly = cast_numeric(weekly, [
        "bodyweight_lbs_weekly", "rhr_bpm_weekly",
        "energy_weekly", "sleep_quality_weekly",
        "stress_weekly", "alcohol_days_weekly",
        "class_attended_weekly"
    ])

    merged = weekly.merge(intake, on="email", how="left", suffixes=("", "_intake"))

    # Deltas vs intake baseline
    if {"bodyweight_lbs_weekly", "bodyweight_lbs_baseline"}.issubset(merged.columns):
        merged["delta_bodyweight_lbs"] = merged["bodyweight_lbs_weekly"] - merged["bodyweight_lbs_baseline"]
    if {"rhr_bpm_weekly", "rhr_bpm_baseline"}.issubset(merged.columns):
        merged["delta_rhr_bpm"] = merged["rhr_bpm_weekly"] - merged["rhr_bpm_baseline"]
    if {"energy_weekly", "energy_baseline"}.issubset(merged.columns):
        merged["delta_energy"] = merged["energy_weekly"] - merged["energy_baseline"]

    return merged, weekly


def cohort_weekly_summary(merged: pd.DataFrame) -> pd.DataFrame:
    if merged is None or merged.empty or "week_number" not in merged.columns:
        return pd.DataFrame(columns=[
            "week_number", "n_participants",
            "bodyweight_mean", "rhr_mean", "energy_mean",
            "adherence_mean", "sleep_hours_mean", "stress_mean"
        ])

    g = merged.dropna(subset=["week_number"]).groupby("week_number")
    out = g.agg(
        n_participants=("email", "nunique"),
        bodyweight_mean=("bodyweight_lbs_weekly", "mean"),
        rhr_mean=("rhr_bpm_weekly", "mean"),
        energy_mean=("energy_weekly", "mean"),
        adherence_mean=("adherence_score_weekly", "mean"),
        sleep_hours_mean=("sleep_hours_numeric_weekly", "mean"),
        stress_mean=("stress_weekly", "mean"),
    ).reset_index().sort_values("week_number")
    return out


def compute_total_weight_lost(merged: pd.DataFrame) -> float:
    if merged is None or merged.empty:
        return 0.0
    needed = {"email", "week_number", "bodyweight_lbs_weekly", "bodyweight_lbs_baseline"}
    if not needed.issubset(merged.columns):
        return 0.0

    latest_rows = (
        merged.dropna(subset=["week_number"])
        .sort_values("week_number")
        .groupby("email", as_index=False)
        .tail(1)
    )
    weight_change = latest_rows["bodyweight_lbs_weekly"] - latest_rows["bodyweight_lbs_baseline"]
    return float(weight_change[weight_change < 0].abs().sum())


def current_week(merged: pd.DataFrame) -> int | None:
    if merged is None or merged.empty or "week_number" not in merged.columns:
        return None
    weeks = pd.to_numeric(merged["week_number"], errors="coerce").dropna()
    if len(weeks) == 0:
        return None
    return int(weeks.max())


def coaching_action_list(intake: pd.DataFrame, merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      missing_checkin_df: emails missing latest-week submission
      at_risk_df: subset of latest-week submitters flagged by simple rules
    """
    wk = current_week(merged)
    if wk is None:
        return pd.DataFrame(columns=["email"]), pd.DataFrame()

    this_week = merged[merged["week_number"] == wk].copy()
    submitted_emails = set(this_week["email"].dropna()) if "email" in this_week.columns else set()
    all_emails = set(intake["email"].dropna()) if "email" in intake.columns else set()

    missing = sorted(list(all_emails - submitted_emails))
    missing_df = pd.DataFrame({"email": missing})

    # At-risk rules
    at_risk = this_week.copy()
    rule_adherence = pd.Series([False] * len(at_risk))
    if "nutrition_adherence_weekly" in at_risk.columns:
        rule_adherence = at_risk["nutrition_adherence_weekly"].astype(str).str.strip().eq("Very few days")

    rule_stress_sleep = pd.Series([False] * len(at_risk))
    if {"stress_weekly", "sleep_quality_weekly"}.issubset(at_risk.columns):
        rule_stress_sleep = (at_risk["stress_weekly"] >= 8) & (at_risk["sleep_quality_weekly"] <= 4)

    at_risk["risk_flag"] = rule_adherence | rule_stress_sleep
    at_risk = at_risk[at_risk["risk_flag"]].copy()

    if "stress_weekly" in at_risk.columns:
        at_risk = at_risk.sort_values("stress_weekly", ascending=False)

    return missing_df, at_risk


def member_latest_snapshot(merged: pd.DataFrame, member_email: str) -> pd.DataFrame:
    """
    Returns last submission row for the selected member.
    """
    if merged is None or merged.empty or "email" not in merged.columns:
        return pd.DataFrame()

    df = merged[merged["email"] == member_email].copy()
    df = df.dropna(subset=["week_number"]).sort_values("week_number")
    if df.empty:
        return pd.DataFrame()
    return df.tail(1)
