# ─────────────────────────────────────────────────────────────────────────────
# streamlit_app.py
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# ─────────────────────────────────────────────────────────────────────────────
# 0) PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="2025 Cash-Flow Scenario Dashboard")

# ─────────────────────────────────────────────────────────────────────────────
# 1) LOAD & PREPARE DATA (CACHED)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data(path="cash_flow_data.csv"):
 
    df = pd.read_csv(path)
    df["Month_dt"] = pd.to_datetime(df["Month"], format="%b-%Y")

    inflow_cols = [
        "Tuition Revenue", "State Appropriation", "Federal Grants",
        "Research Grants", "Donations & Endowments", "Auxiliary Services Income",
        "Continuing Education Fees"
    ]
    outflow_cols = [
        "Faculty Salaries", "Staff Salaries", "Scholarship Disbursements",
        "Facilities Maintenance", "Utility Expenses", "IT Infrastructure Costs",
        "Academic Program Costs", "Student Services", "Administrative Expenses",
        "Marketing & Outreach", "Debt Service"
    ]

    # Compute aggregates if missing
    if "Total Inflow" not in df.columns:
        df["Total Inflow"] = df[inflow_cols].sum(axis=1)
    if "Total Outflow" not in df.columns:
        df["Total Outflow"] = df[outflow_cols].sum(axis=1)
    if "Net Cash Flow" not in df.columns:
        df["Net Cash Flow"] = df["Total Inflow"] - df["Total Outflow"]

    # Fit Prophet on Total Inflow / Total Outflow (historical 2023-2024)
    inflow_df  = df[["Month_dt", "Total Inflow"]].rename(columns={"Month_dt": "ds", "Total Inflow": "y"})
    outflow_df = df[["Month_dt", "Total Outflow"]].rename(columns={"Month_dt": "ds", "Total Outflow": "y"})

    m_in  = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m_in.fit(inflow_df)
    m_out = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m_out.fit(outflow_df)

    # Forecast 12 months into 2025, using freq="MS" for month‐start alignment
    future_in  = m_in.make_future_dataframe(periods=12, freq="MS")
    future_out = m_out.make_future_dataframe(periods=12, freq="MS")

    forecast_in  = m_in.predict(future_in)
    forecast_out = m_out.predict(future_out)

    pred_in  = forecast_in[["ds", "yhat"]].rename(columns={"yhat": "Inflow_Pred"})
    pred_out = forecast_out[["ds", "yhat"]].rename(columns={"yhat": "Outflow_Pred"})

    pred = pred_in.merge(pred_out, on="ds")
    pred["Net_Cash_Pred"] = pred["Inflow_Pred"] - pred["Outflow_Pred"]

    # Keep only 2025 rows
    pred_2025 = pred[pred["ds"] >= pd.to_datetime("2025-01-01")].reset_index(drop=True)

    # Identify “flexible” outflows by Relative Amplitude (2023-2024)
    df["Month_Num"] = df["Month_dt"].dt.month
    season_amp = {}
    mean_vals  = {}
    for cat in outflow_cols:
        monthly_avg     = df.groupby("Month_Num")[cat].mean()
        season_amp[cat] = monthly_avg.max() - monthly_avg.min()
        mean_vals[cat]  = df[cat].mean()
    season_amp   = pd.Series(season_amp)
    mean_vals    = pd.Series(mean_vals)
    relative_amp = (season_amp / mean_vals).sort_values(ascending=False)
    threshold    = relative_amp.median()
    flexible_by_rel = list(relative_amp[relative_amp > threshold].index)

    return df, inflow_cols, outflow_cols, pred_2025, flexible_by_rel

df, inflow_cols, outflow_cols, pred_2025, flexible_by_rel = load_and_prepare_data()

# ─────────────────────────────────────────────────────────────────────────────
# 2) DEFINE PRESCRIPTIVE SCENARIO FUNCTION (CACHED)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def scenario_2025(inflow_cuts: dict, min_net_cash: float):
    """
    inflow_cuts: dict {inflow_category: fraction_to_cut (0.0-1.0)}
    min_net_cash: required minimum net cash each month (dollars)

    Returns:
      - pa (DataFrame): 12 rows (Jan-Dec 2025) with columns:
           ds, Inflow_Pred, Outflow_Pred, Net_Cash_Pred,
           Inflow_Adj, Net_Cash_Adj, Shortfall
      - cuts_df (DataFrame): indexed by ds (month), columns = [Shortfall] + flexible_by_rel
           showing recommended dollar cuts for each “flexible” outflow.
    """
    # 2.1) Historical share of each inflow category (2023-2024)
    inflow_shares = df[inflow_cols].sum() / df["Total Inflow"].sum()

    # 2.2) Copy pred_2025 → pa, then compute Inflow_Adj based on cuts
    pa = pred_2025[["ds", "Inflow_Pred", "Outflow_Pred", "Net_Cash_Pred"]].copy()
    pa["Inflow_Adj"] = pa["Inflow_Pred"]
    for cat, pct in inflow_cuts.items():
        share = inflow_shares.get(cat, 0.0)
        pa["Inflow_Adj"] = pa["Inflow_Adj"] * (1 - pct * share)

    # 2.3) Adjusted Net Cash
    pa["Net_Cash_Adj"] = pa["Inflow_Adj"] - pa["Outflow_Pred"]

    # 2.4) Shortfall below min_net_cash
    pa["Shortfall"] = np.where(
        pa["Net_Cash_Adj"] < min_net_cash,
        min_net_cash - pa["Net_Cash_Adj"],
        0
    )

    # 2.5) Allocate each shortfall across flexible outflows by 2023-2024 average weights
    avg_vals  = df[flexible_by_rel].mean()
    total_avg = avg_vals.sum()

    cuts_list = []
    for _, row in pa.iterrows():
        month = row["ds"]
        short = row["Shortfall"]
        recs  = {}
        if short <= 0:
            for c in flexible_by_rel:
                recs[c] = 0.0
        else:
            for c in flexible_by_rel:
                recs[c] = short * (avg_vals[c] / total_avg)
        recs["Shortfall"] = short
        recs["ds"]        = month
        cuts_list.append(recs)

    cuts_df = pd.DataFrame(cuts_list).set_index("ds")
    return pa, cuts_df

# ─────────────────────────────────────────────────────────────────────────────
# 3) MAIN PAGE: TWO TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([" Predictions ", " What-If Predictions "])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: “Predictions” – show historical + forecast plots/tables
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("1. 2025 Prophet Forecasts & Historical Comparison")
    st.markdown(
        "Below are three separate charts—one each for Inflow, Outflow, and Net Cash—"
        "covering the historical period (2023–2024) followed by the 2025 forecast.  \n"
        "Historical data is shown as a solid line; 2025 forecast is shown as a dashed line."
    )

    # Prepare historical vs 2025‐forecast data
    hist = df[["Month_dt", "Total Inflow", "Total Outflow", "Net Cash Flow"]].sort_values("Month_dt")
    pred = pred_2025.copy()

    # --- 1.1) Inflow: Historical vs Forecast (solid+overlay dashed) ---
    st.subheader("1.1 Inflow: Historical (2023–2024) vs 2025 Forecast")
    fig_inf, ax_inf = plt.subplots(figsize=(10, 4))

    # 1a) Draw solid line for entire period (2023-2025)
    all_dates  = pd.concat([hist["Month_dt"], pred["ds"]])
    all_values = pd.concat([hist["Total Inflow"], pred["Inflow_Pred"]])
    ax_inf.plot(
        all_dates, all_values,
        color="tab:blue", linestyle="-", linewidth=2,
        label="Inflow (Historic + Forecast)", zorder=1
    )

    # 1b) Overlay dashed for 2025 only
    ax_inf.plot(
        pred["ds"], pred["Inflow_Pred"],
        color="tab:green", linestyle="-", linewidth=2,
        label="Forecast Inflow (2025)", zorder=2
    )

    ax_inf.set_title("Inflow: Historical vs 2025 Forecast")
    ax_inf.set_xlabel("Month")
    ax_inf.set_ylabel("Inflow ($)")
    ax_inf.legend(loc="upper left")
    ax_inf.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig_inf)

    # Combined table for Inflow (Historic + Forecast)
    df_inflow_table = pd.DataFrame({
        "Month": pd.concat([hist["Month_dt"], pred["ds"]]),
        "Inflow Value": pd.concat([hist["Total Inflow"], pred["Inflow_Pred"]])
    }).reset_index(drop=True)
    df_inflow_table["Month"] = df_inflow_table["Month"].dt.strftime("%b-%Y")
    st.dataframe(
        df_inflow_table.style.format({"Inflow Value": "{:,.0f}"}),
        use_container_width=True
    )

    # --- 1.2) Outflow: Historical vs Forecast (solid+overlay dashed) ---
    st.subheader("1.2 Outflow: Historical (2023–2024) vs 2025 Forecast")
    fig_out, ax_out = plt.subplots(figsize=(10, 4))

    all_dates_o  = pd.concat([hist["Month_dt"], pred["ds"]])
    all_values_o = pd.concat([hist["Total Outflow"], pred["Outflow_Pred"]])
    ax_out.plot(
        all_dates_o, all_values_o,
        color="tab:orange", linestyle="-", linewidth=2,
        label="Outflow (Historic + Forecast)", zorder=1
    )
    ax_out.plot(
        pred["ds"], pred["Outflow_Pred"],
        color="tab:purple", linestyle="-", linewidth=2,
        label="Forecast Outflow (2025)", zorder=2
    )

    ax_out.set_title("Outflow: Historical vs 2025 Forecast")
    ax_out.set_xlabel("Month")
    ax_out.set_ylabel("Outflow ($)")
    ax_out.legend(loc="upper left")
    ax_out.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig_out)

    # Combined table for Outflow (Historic + Forecast)
    df_outflow_table = pd.DataFrame({
        "Month": pd.concat([hist["Month_dt"], pred["ds"]]),
        "Outflow Value": pd.concat([hist["Total Outflow"], pred["Outflow_Pred"]])
    }).reset_index(drop=True)
    df_outflow_table["Month"] = df_outflow_table["Month"].dt.strftime("%b-%Y")
    st.dataframe(
        df_outflow_table.style.format({"Outflow Value": "{:,.0f}"}),
        use_container_width=True
    )

    # --- 1.3) Net Cash: Historical vs Forecast (solid+overlay dashed) ---
    st.subheader("1.3 Net Cash: Historical (2023–2024) vs 2025 Forecast")
    fig_net, ax_net = plt.subplots(figsize=(10, 4))

    all_dates_n  = pd.concat([hist["Month_dt"], pred["ds"]])
    all_values_n = pd.concat([hist["Net Cash Flow"], pred["Net_Cash_Pred"]])
    ax_net.plot(
        all_dates_n, all_values_n,
        color="tab:green", linestyle="-", linewidth=2,
        label="Net Cash (Historic + Forecast)", zorder=1
    )
    ax_net.plot(
        pred["ds"], pred["Net_Cash_Pred"],
        color="tab:cyan", linestyle="-", linewidth=2,
        label="Forecast Net Cash (2025)", zorder=2
    )

    ax_net.set_title("Net Cash: Historical vs 2025 Forecast")
    ax_net.set_xlabel("Month")
    ax_net.set_ylabel("Net Cash ($)")
    ax_net.legend(loc="upper left")
    ax_net.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig_net)

    # Combined table for Net Cash (Historic + Forecast)
    df_netcash_table = pd.DataFrame({
        "Month": pd.concat([hist["Month_dt"], pred["ds"]]),
        "Net Cash Value": pd.concat([hist["Net Cash Flow"], pred["Net_Cash_Pred"]])
    }).reset_index(drop=True)
    df_netcash_table["Month"] = df_netcash_table["Month"].dt.strftime("%b-%Y")
    st.dataframe(
        df_netcash_table.style.format({"Net Cash Value": "{:,.0f}"}),
        use_container_width=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: “What-If Predictions” – sidebar sliders + adjusted results
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    # 2.1) SIDEBAR inputs (only shown on Tab 2)
    st.sidebar.header("Scenario Inputs: Inflow Cuts & Minimum Cash")
    st.sidebar.markdown(
        "Adjust sliders to simulate percentage cuts in each revenue line for all of 2025."
    )

    inflow_cuts = {}
    for cat in inflow_cols:
        cut_pct = st.sidebar.slider(f"{cat} cut (%)", min_value=0, max_value=100, value=0, step=1)
        inflow_cuts[cat] = cut_pct / 100.0

    min_cash = st.sidebar.number_input(
        "Minimum Net Cash required each month ($)", min_value=0, value=0, step=100000
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Flexible outflow categories (by relative amplitude):")
    for cat in flexible_by_rel:
        st.sidebar.text(f"• {cat}")

    # 2.2) Main content for Tab 2
    st.header("2. 2025 What-If Scenario")
    st.markdown(
        "Use the sliders to cut any portion of your forecasted inflows, "
        "then view how your Net Cash changes and which outflow lines must be trimmed."
    )

    pred_2025_adj, cuts_2025 = scenario_2025(inflow_cuts, min_cash)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(
        pred_2025_adj["ds"], pred_2025_adj["Net_Cash_Pred"],
        "g--", linewidth=2, label="Base Net Cash"
    )
    ax2.plot(
        pred_2025_adj["ds"], pred_2025_adj["Net_Cash_Adj"],
        color="magenta", linewidth=2, alpha=0.6, label="Adjusted Net Cash"
    )
    ax2.axhline(
        min_cash,
        color="red", linestyle="--", linewidth=1.5,
        label=f"Min Cash Floor (${min_cash:,.0f})"
    )
    ax2.set_title("2025 Net Cash: Baseline (green dashed) vs Scenario (magenta)")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Net Cash ($)")
    ax2.legend(loc="upper left")
    ax2.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

    st.subheader("Months with Shortfalls & Prescribed Outflow Cuts")
    shortfall_months = cuts_2025[cuts_2025["Shortfall"] > 0]
    if shortfall_months.empty:
        st.success("No shortfalls under these assumptions. No cuts needed in 2025.")
    else:
        st.info("Below are the shortfall amounts and recommended cuts per flexible category:")
        display_cols = ["Shortfall"] + flexible_by_rel
        st.dataframe(
            shortfall_months[display_cols]
                          .rename_axis("Month")
                          .style.format("{:,.0f}"),
            use_container_width=True
        )

    with st.expander("Show full 2025 adjusted forecast data"):
        df_show = pred_2025_adj.copy()
        df_show["Month"] = df_show["ds"].dt.strftime("%b-%Y")
        df_show = df_show[
            ["Month", "Inflow_Pred", "Inflow_Adj",
             "Outflow_Pred", "Net_Cash_Pred", "Net_Cash_Adj", "Shortfall"]
        ]
        st.dataframe(
            df_show.style.format({
                "Inflow_Pred":   "{:,.0f}",
                "Inflow_Adj":    "{:,.0f}",
                "Outflow_Pred":  "{:,.0f}",
                "Net_Cash_Pred": "{:,.0f}",
                "Net_Cash_Adj":  "{:,.0f}",
                "Shortfall":     "{:,.0f}"
            }),
            use_container_width=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# 4) FOOTER NOTE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Use the “What-If Predictions” tab to experiment with cuts in any revenue line. "
    "The green dashed curve is the baseline forecast, the magenta curve is your adjusted net cash, "
    "and the red dashed horizontal line marks your required minimum cash floor. "
    "Any month where Adjusted Net Cash < minimum floor appears in the table with recommended cuts."
)
