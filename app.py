import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import backend as be

st.set_page_config(page_title="Nutrition Challenge Dashboard", layout="wide")

# -----------------
# Hardcoded paths (coaches should never edit these)
# -----------------
INTAKE_PATH = "data/intake_responses.csv"
WEEKLY_PATH = "data/weekly_responses.csv"
CONFIG_PATH = "data/config.csv"


def plot_line(x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    ax.plot(x, y, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def plot_hist(series, xlabel, title):
    fig, ax = plt.subplots()
    ax.hist(series.dropna(), bins=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


# -----------------
# Load data
# -----------------
cfg = be.load_config(CONFIG_PATH)

try:
    intake_df = pd.read_csv(INTAKE_PATH)
except Exception as e:
    st.error(f"Could not load intake CSV at '{INTAKE_PATH}': {e}")
    st.stop()

try:
    weekly_df = pd.read_csv(WEEKLY_PATH)
    weekly_available = True
except FileNotFoundError:
    weekly_available = False
    weekly_df = pd.DataFrame()
except Exception as e:
    st.error(f"Could not load weekly CSV at '{WEEKLY_PATH}': {e}")
    weekly_available = False
    weekly_df = pd.DataFrame()

# Normalize & validate intake
intake_df = be.normalize_common(intake_df)
missing_intake = be.validate_intake(intake_df)
if missing_intake:
    st.error(f"Intake CSV missing required columns: {missing_intake}")
    st.stop()

# Weekly pipeline (optional)
duplicates_df = pd.DataFrame()
merged_df = pd.DataFrame()
weekly_clean = pd.DataFrame()

if weekly_available and not weekly_df.empty:
    weekly_df = be.normalize_common(weekly_df)

    missing_weekly = be.validate_weekly(weekly_df)
    if missing_weekly:
        st.error(f"Weekly CSV missing required columns: {missing_weekly}")
        st.stop()

    weekly_df["week_number"] = be.parse_week_number(weekly_df["week_number"])
    weekly_df, duplicates_df = be.dedupe_weekly_keep_latest(weekly_df)
    merged_df, weekly_clean = be.prep_data(intake_df, weekly_df)

# -----------------
# Header + KPIs
# -----------------
challenge_name = cfg.get("challenge_name", "Nutrition Challenge")
st.title(f"{challenge_name} — Coach Dashboard")

total_weight_lost = be.compute_total_weight_lost(merged_df)
weeks_tracked = int(merged_df["week_number"].nunique()) if "week_number" in merged_df.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Participants (intake)", int(intake_df["email"].nunique()))
c2.metric("Weekly submissions", int(len(weekly_clean)) if isinstance(weekly_clean, pd.DataFrame) else 0)
c3.metric("Weeks tracked", weeks_tracked)
c4.metric("Total Weight Lost (lbs)", f"{total_weight_lost:.1f}")

st.divider()

# -----------------
# Member selection (core coach feature)
# -----------------
member_email = None
if not merged_df.empty and "email" in merged_df.columns:
    all_members = sorted(merged_df["email"].dropna().unique().tolist())
    member_email = st.selectbox("Select a member to review", options=["(All members)"] + all_members)
else:
    st.info("No weekly data loaded yet — member selection will appear once check-ins start.")

# -----------------
# Tabs
# -----------------
tab1, tab2, tab3, tab4 = st.tabs(["Trends Over Time", "Before/After", "Coaching Action List", "Data"])

# --- Trends Over Time ---
with tab1:
    st.subheader("Gym trends over time")

    if merged_df.empty:
        st.info("No weekly data yet. Once weekly check-ins come in, trends will appear here.")
    else:
        # Filter for member if selected
        if member_email and member_email != "(All members)":
            view_df = merged_df[merged_df["email"] == member_email].copy()
            st.caption(f"Showing trends for: {member_email}")
        else:
            view_df = merged_df.copy()
            st.caption("Showing trends for: All members")

        summary = be.cohort_weekly_summary(view_df)

        colA, colB, colC = st.columns(3)
        with colA:
            if not summary.empty and "bodyweight_mean" in summary.columns:
                plot_line(summary["week_number"], summary["bodyweight_mean"], "Week", "Bodyweight (lbs)", "Avg Bodyweight by Week")
            else:
                st.info("Bodyweight trend unavailable.")
        with colB:
            if not summary.empty and "rhr_mean" in summary.columns:
                plot_line(summary["week_number"], summary["rhr_mean"], "Week", "RHR (bpm)", "Avg Resting HR by Week")
            else:
                st.info("RHR trend unavailable.")
        with colC:
            if not summary.empty and "energy_mean" in summary.columns:
                plot_line(summary["week_number"], summary["energy_mean"], "Week", "Energy (1–10)", "Avg Energy by Week")
            else:
                st.info("Energy trend unavailable.")

        colD, colE, colF = st.columns(3)
        with colD:
            if not summary.empty and "adherence_mean" in summary.columns:
                plot_line(summary["week_number"], summary["adherence_mean"], "Week", "Adherence (0–1)", "Avg Adherence by Week")
            else:
                st.info("Adherence trend unavailable.")
        with colE:
            if not summary.empty and "sleep_hours_mean" in summary.columns:
                plot_line(summary["week_number"], summary["sleep_hours_mean"], "Week", "Sleep Hours", "Avg Sleep Hours by Week")
            else:
                st.info("Sleep hours trend unavailable.")
        with colF:
            if not summary.empty and "stress_mean" in summary.columns:
                plot_line(summary["week_number"], summary["stress_mean"], "Week", "Stress (1–10)", "Avg Stress by Week")
            else:
                st.info("Stress trend unavailable.")

        # Member detail: show their most recent row + key deltas
        if member_email and member_email != "(All members)":
            st.subheader("Member snapshot (latest check-in)")
            snap = be.member_latest_snapshot(merged_df, member_email)
            if snap.empty:
                st.info("No check-ins found for this member.")
            else:
                show_cols = [
                    "week_number",
                    "bodyweight_lbs_weekly", "delta_bodyweight_lbs",
                    "rhr_bpm_weekly", "delta_rhr_bpm",
                    "energy_weekly", "delta_energy",
                    "nutrition_adherence_weekly",
                    "sleep_quality_weekly", "stress_weekly", "sleep_hours_weekly",
                    "notes_weekly", "weekly_win", "weekly_help"
                ]
                show_cols = [c for c in show_cols if c in snap.columns]
                st.dataframe(snap[show_cols], use_container_width=True)

# --- Before/After ---
with tab2:
    st.subheader("Before vs After (baseline → final week)")

    if merged_df.empty or "week_number" not in merged_df.columns:
        st.info("No weekly data yet.")
    else:
        df = merged_df.copy()
        if member_email and member_email != "(All members)":
            df = df[df["email"] == member_email].copy()
            st.caption(f"Comparing for: {member_email}")
        else:
            st.caption("Comparing for: All members")

        weeks = np.sort(pd.to_numeric(df["week_number"], errors="coerce").dropna().astype(int).unique())
        if len(weeks) == 0:
            st.info("No valid week numbers detected yet.")
        else:
            final_week = int(weeks.max())
            st.caption(f"Final week detected: Week {final_week}")

            final_rows = df[df["week_number"] == final_week].copy()

            col1, col2, col3 = st.columns(3)
            with col1:
                if "delta_bodyweight_lbs" in final_rows.columns:
                    plot_hist(final_rows["delta_bodyweight_lbs"], "Δ Bodyweight (final - baseline)", "Bodyweight Change Distribution")
                else:
                    st.info("Bodyweight delta unavailable.")
            with col2:
                if "delta_rhr_bpm" in final_rows.columns:
                    plot_hist(final_rows["delta_rhr_bpm"], "Δ RHR (final - baseline)", "RHR Change Distribution")
                else:
                    st.info("RHR delta unavailable.")
            with col3:
                if "delta_energy" in final_rows.columns:
                    plot_hist(final_rows["delta_energy"], "Δ Energy (final - baseline)", "Energy Change Distribution")
                else:
                    st.info("Energy delta unavailable.")

# --- Coaching Action List ---
with tab3:
    st.subheader("Coaching action list (latest week)")

    if merged_df.empty:
        st.info("No weekly data yet.")
    else:
        wk = be.current_week(merged_df)
        if wk is None:
            st.info("No valid week numbers detected yet.")
        else:
            st.caption(f"Current week: Week {wk}")

            missing_df, at_risk_df = be.coaching_action_list(intake_df, merged_df)

            st.markdown("### Missing weekly check-in")
            if not missing_df.empty:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("No missing check-ins for the current week.")

            st.markdown("### At-risk (among those who checked in)")
            if not at_risk_df.empty:
                risk_cols = [
                    "email", "week_number",
                    "nutrition_adherence_weekly",
                    "stress_weekly", "sleep_quality_weekly", "energy_weekly",
                    "sleep_hours_weekly",
                    "bodyweight_lbs_weekly", "rhr_bpm_weekly",
                    "weekly_help", "notes_weekly"
                ]
                risk_cols = [c for c in risk_cols if c in at_risk_df.columns]
                st.dataframe(at_risk_df[risk_cols], use_container_width=True)
            else:
                st.success("No at-risk flags triggered this week (based on current rules).")

# --- Data ---
with tab4:
    st.subheader("Data preview (debug)")
    with st.expander("Merged (weekly + intake)", expanded=False):
        st.dataframe(merged_df, use_container_width=True)

    with st.expander("Intake responses", expanded=False):
        st.dataframe(intake_df, use_container_width=True)

    with st.expander("Weekly responses (deduped)", expanded=False):
        st.dataframe(weekly_clean, use_container_width=True)

    if isinstance(duplicates_df, pd.DataFrame) and len(duplicates_df):
        with st.expander("Duplicates detected (informational)", expanded=False):
            st.dataframe(duplicates_df, use_container_width=True)

st.caption("v3: Coach-only dashboard (no sidebar, split frontend/backend). Next: live Google Sheets + exports.")
