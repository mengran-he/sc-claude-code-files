"""
E-commerce Business Analytics Dashboard
A professional Streamlit dashboard for business performance analysis.
"""

import calendar
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_and_process_data

warnings.filterwarnings("ignore")

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-commerce Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.main > div { padding-top: 1.5rem; }

/* KPI cards */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
    height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 0;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e293b;
    margin: 0;
    line-height: 1.1;
}
.kpi-trend {
    font-size: 0.8rem;
    font-weight: 500;
    margin: 0;
}

/* Bottom row cards */
.bottom-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
    height: 160px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
}
.bottom-value {
    font-size: 2.8rem;
    font-weight: 700;
    color: #1e293b;
    margin: 0;
    line-height: 1.1;
}
.bottom-subtitle {
    font-size: 0.8rem;
    color: #64748b;
    font-weight: 500;
    margin: 0.3rem 0 0 0;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.stars {
    font-size: 1.5rem;
    color: #f59e0b;
    margin: 0.2rem 0;
    letter-spacing: 0.05em;
}

/* Trend colors */
.trend-up   { color: #16a34a; }
.trend-down { color: #dc2626; }

/* Section spacing */
.section-gap { margin-top: 1.25rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    loader, processed = load_and_process_data("ecommerce_data/")
    return loader, processed


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_short(value: float) -> str:
    """Format a number as $1.2M / $300K / $500."""
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:.0f}"


def fmt_trend(current: float, previous: float, lower_is_better: bool = False) -> str:
    """Return colored trend HTML with two decimal places."""
    if not previous:
        return "<span style='color:#94a3b8'>N/A</span>"
    pct = (current - previous) / previous * 100
    positive_outcome = (pct < 0) if lower_is_better else (pct > 0)
    arrow = "▲" if pct > 0 else "▼"
    cls = "trend-up" if positive_outcome else "trend-down"
    return f'<span class="{cls}">{arrow} {abs(pct):.2f}%</span>'


def currency_ticks(max_val: float):
    """Compute nice tick values and $K/$M labels for a Plotly axis."""
    if max_val <= 0:
        return [0], ["$0"]
    use_millions = max_val >= 1_000_000
    mag = 1_000_000 if use_millions else 1_000
    sfx = "M" if use_millions else "K"

    raw_step = max_val / 5 / mag
    exp = 10 ** np.floor(np.log10(max(raw_step, 1e-10)))
    nice_step = np.ceil(raw_step / exp) * exp * mag

    ticks, t = [], 0.0
    while t <= max_val * 1.01:
        ticks.append(t)
        t += nice_step

    def label(v: float) -> str:
        if v == 0:
            return "$0"
        v_m = v / mag
        return f"${int(v_m)}{sfx}" if v_m == int(v_m) else f"${v_m:.1f}{sfx}"

    return ticks, [label(t) for t in ticks]


# ── Chart constants ───────────────────────────────────────────────────────────
CHART_H = 380
GRID_COLOR = "#f1f5f9"
PRIMARY = "#1565C0"
PREV_COLOR = "#94a3b8"
MONTH_ABBR = {i: calendar.month_abbr[i] for i in range(1, 13)}


def base_layout(**overrides) -> dict:
    layout = dict(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#374151"),
        height=CHART_H,
    )
    layout.update(overrides)
    if "margin" not in layout:
        layout["margin"] = dict(t=55, b=50, l=60, r=30)
    return layout


# ── Chart builders ────────────────────────────────────────────────────────────

def revenue_trend_chart(
    cur: pd.DataFrame, prev: pd.DataFrame | None, cur_year: int, prev_year: int
) -> go.Figure:
    fig = go.Figure()

    cur_m = (
        cur.groupby("purchase_month")["price"]
        .sum()
        .reset_index()
        .assign(label=lambda d: d["purchase_month"].map(MONTH_ABBR))
    )

    all_max = cur_m["price"].max()
    if prev is not None and not prev.empty:
        prev_m = (
            prev.groupby("purchase_month")["price"]
            .sum()
            .reset_index()
            .assign(label=lambda d: d["purchase_month"].map(MONTH_ABBR))
        )
        all_max = max(all_max, prev_m["price"].max())
    else:
        prev_m = None

    tickvals, ticktext = currency_ticks(all_max * 1.1)

    fig.add_trace(
        go.Scatter(
            x=cur_m["label"],
            y=cur_m["price"],
            name=str(cur_year),
            mode="lines+markers",
            line=dict(color=PRIMARY, width=2.5, dash="solid"),
            marker=dict(size=7, color=PRIMARY),
            text=[fmt_short(v) for v in cur_m["price"]],
            hovertemplate="%{x}: %{text}<extra></extra>",
        )
    )

    if prev_m is not None:
        fig.add_trace(
            go.Scatter(
                x=prev_m["label"],
                y=prev_m["price"],
                name=str(prev_year),
                mode="lines+markers",
                line=dict(color=PREV_COLOR, width=2, dash="dash"),
                marker=dict(size=6, color=PREV_COLOR),
                text=[fmt_short(v) for v in prev_m["price"]],
                hovertemplate="%{x}: %{text}<extra></extra>",
            )
        )

    fig.update_layout(
        **base_layout(
            title=dict(text="Revenue Trend", font_size=14, x=0),
            xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False),
            yaxis=dict(
                showgrid=True,
                gridcolor=GRID_COLOR,
                zeroline=False,
                tickvals=tickvals,
                ticktext=ticktext,
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"
            ),
            hovermode="x unified",
        )
    )
    return fig


def category_chart(sales: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "product_category_name" not in sales.columns:
        fig.add_annotation(
            text="Product category data not available",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
        )
        return fig

    # Top 10 sorted descending; reverse for horizontal-bar top-to-bottom display
    top10 = (
        sales.groupby("product_category_name")["price"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .sort_values(ascending=True)  # plotly horizontal: ascending → highest at top
    )

    tickvals, ticktext = currency_ticks(top10.max() * 1.2)

    fig.add_trace(
        go.Bar(
            y=top10.index,
            x=top10.values,
            orientation="h",
            marker=dict(
                color=top10.values,
                colorscale="Blues",
                cmin=top10.min() * 0.3,
                cmax=top10.max(),
                showscale=False,
            ),
            text=[fmt_short(v) for v in top10.values],
            textposition="outside",
            hovertemplate="%{y}<br>Revenue: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        **base_layout(
            title=dict(text="Top 10 Product Categories by Revenue", font_size=14, x=0),
            xaxis=dict(
                showgrid=True,
                gridcolor=GRID_COLOR,
                zeroline=False,
                tickvals=tickvals,
                ticktext=ticktext,
            ),
            yaxis=dict(showgrid=False, automargin=True),
            margin=dict(t=55, b=50, l=190, r=90),
        )
    )
    return fig


def state_map_chart(sales: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "customer_state" not in sales.columns:
        fig.add_annotation(
            text="Geographic data not available",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
        )
        return fig

    st_rev = (
        sales.groupby("customer_state")["price"]
        .sum()
        .reset_index()
        .rename(columns={"customer_state": "state", "price": "revenue"})
    )
    st_rev["label"] = st_rev["revenue"].apply(fmt_short)

    rev_min, rev_max = st_rev["revenue"].min(), st_rev["revenue"].max()

    fig.add_trace(
        go.Choropleth(
            locations=st_rev["state"],
            z=st_rev["revenue"],
            locationmode="USA-states",
            colorscale="Blues",
            showscale=True,
            colorbar=dict(
                title=dict(text="Revenue", side="right"),
                tickvals=[rev_min, (rev_min + rev_max) / 2, rev_max],
                ticktext=[
                    fmt_short(rev_min),
                    fmt_short((rev_min + rev_max) / 2),
                    fmt_short(rev_max),
                ],
                len=0.65,
                thickness=14,
            ),
            customdata=st_rev["label"],
            hovertemplate="%{location}<br>Revenue: %{customdata}<extra></extra>",
        )
    )

    fig.update_layout(
        **base_layout(
            title=dict(text="Revenue by State", font_size=14, x=0),
            geo=dict(
                scope="usa",
                showframe=False,
                showcoastlines=True,
                coastlinecolor="#cbd5e1",
                bgcolor="white",
                lakecolor="white",
            ),
            margin=dict(t=55, b=10, l=10, r=10),
        )
    )
    return fig


def satisfaction_delivery_chart(sales: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "delivery_days" not in sales.columns or "review_score" not in sales.columns:
        fig.add_annotation(
            text="Delivery or review data not available",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
        )
        return fig

    def bucket(d: float) -> str | None:
        if pd.isna(d):
            return None
        if d <= 3:
            return "1-3 days"
        if d <= 7:
            return "4-7 days"
        if d <= 14:
            return "8-14 days"
        return "15+ days"

    orders = (
        sales.drop_duplicates("order_id")
        .dropna(subset=["delivery_days", "review_score"])
        .copy()
    )
    orders["bucket"] = orders["delivery_days"].apply(bucket)
    orders = orders.dropna(subset=["bucket"])

    bucket_order = ["1-3 days", "4-7 days", "8-14 days", "15+ days"]
    agg = (
        orders.groupby("bucket")["review_score"]
        .mean()
        .reindex(bucket_order)
        .dropna()
        .reset_index()
    )

    fig.add_trace(
        go.Bar(
            x=agg["bucket"],
            y=agg["review_score"],
            marker_color=PRIMARY,
            width=0.5,
            text=[f"{v:.2f}" for v in agg["review_score"]],
            textposition="outside",
            hovertemplate="%{x}<br>Avg Score: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        **base_layout(
            title=dict(text="Satisfaction vs Delivery Time", font_size=14, x=0),
            xaxis=dict(showgrid=False, title="Delivery Time"),
            yaxis=dict(
                showgrid=True,
                gridcolor=GRID_COLOR,
                zeroline=False,
                range=[0, 5.5],
                title="Average Review Score",
                dtick=1,
            ),
        )
    )
    return fig


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    loader, processed = load_data()
    if loader is None:
        st.error("Failed to load data. Please check that ecommerce_data/ exists.")
        return

    orders_df = processed["orders"]
    available_years = sorted(orders_df["purchase_year"].unique(), reverse=True)

    # ── Header: title left, filters right ────────────────────────────────────
    h_title, h_year, h_month = st.columns([3, 1, 1])

    with h_title:
        st.markdown("## E-commerce Analytics Dashboard")

    with h_year:
        default_idx = available_years.index(2023) if 2023 in available_years else 0
        selected_year = st.selectbox("Year", available_years, index=default_idx)

    with h_month:
        month_names = ["All Months"] + [calendar.month_name[i] for i in range(1, 13)]
        month_sel = st.selectbox("Month", month_names, index=0)
        selected_month = (
            None if month_sel == "All Months"
            else list(calendar.month_name).index(month_sel)
        )

    # ── Filtered datasets ─────────────────────────────────────────────────────
    cur = loader.create_sales_dataset(
        year_filter=selected_year,
        month_filter=selected_month,
        status_filter="delivered",
    )

    prev_year = selected_year - 1
    prev = (
        loader.create_sales_dataset(
            year_filter=prev_year,
            month_filter=selected_month,
            status_filter="delivered",
        )
        if prev_year in available_years
        else None
    )

    # ── KPI calculations ──────────────────────────────────────────────────────
    total_rev = cur["price"].sum()
    total_orders = cur["order_id"].nunique()
    aov = cur.groupby("order_id")["price"].sum().mean() if total_orders > 0 else 0.0

    prev_rev = prev["price"].sum() if prev is not None else 0.0
    prev_orders = prev["order_id"].nunique() if prev is not None else 0
    prev_aov = (
        prev.groupby("order_id")["price"].sum().mean()
        if prev is not None else 0.0
    )

    monthly = cur.groupby("purchase_month")["price"].sum()
    monthly_growth = monthly.pct_change().mean() * 100 if len(monthly) > 1 else 0.0
    mg_arrow = "▲" if monthly_growth >= 0 else "▼"
    mg_cls = "trend-up" if monthly_growth >= 0 else "trend-down"

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <p class="kpi-label">Total Revenue</p>
                <p class="kpi-value">{fmt_short(total_rev)}</p>
                <p class="kpi-trend">{fmt_trend(total_rev, prev_rev)}</p>
            </div>""",
            unsafe_allow_html=True,
        )

    with k2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <p class="kpi-label">Monthly Growth</p>
                <p class="kpi-value">{monthly_growth:.2f}%</p>
                <p class="kpi-trend"><span class="{mg_cls}">{mg_arrow}</span></p>
            </div>""",
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <p class="kpi-label">Average Order Value</p>
                <p class="kpi-value">{fmt_short(aov)}</p>
                <p class="kpi-trend">{fmt_trend(aov, prev_aov)}</p>
            </div>""",
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <p class="kpi-label">Total Orders</p>
                <p class="kpi-value">{total_orders:,}</p>
                <p class="kpi-trend">{fmt_trend(total_orders, prev_orders)}</p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

    # ── Charts Grid (2 × 2) ───────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            revenue_trend_chart(cur, prev, selected_year, prev_year),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(category_chart(cur), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(state_map_chart(cur), use_container_width=True)
    with c4:
        st.plotly_chart(satisfaction_delivery_chart(cur), use_container_width=True)

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

    # ── Bottom Row ────────────────────────────────────────────────────────────
    b1, b2 = st.columns(2)

    with b1:
        if "delivery_days" in cur.columns:
            avg_del = cur.drop_duplicates("order_id")["delivery_days"].dropna().mean()
            prev_del = (
                prev.drop_duplicates("order_id")["delivery_days"].dropna().mean()
                if prev is not None else 0.0
            )
            del_trend = fmt_trend(avg_del, prev_del, lower_is_better=True)
            st.markdown(
                f"""
                <div class="bottom-card">
                    <p class="kpi-label">Average Delivery Time</p>
                    <p class="bottom-value">{avg_del:.1f} days</p>
                    <p class="kpi-trend">{del_trend}</p>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="bottom-card">
                    <p class="kpi-label">Average Delivery Time</p>
                    <p class="bottom-value">N/A</p>
                </div>""",
                unsafe_allow_html=True,
            )

    with b2:
        if "review_score" in cur.columns:
            avg_rev = cur.drop_duplicates("order_id")["review_score"].dropna().mean()
            full = int(round(avg_rev))
            stars = "★" * full + "☆" * (5 - full)
            st.markdown(
                f"""
                <div class="bottom-card">
                    <p class="bottom-value">{avg_rev:.2f}</p>
                    <p class="stars">{stars}</p>
                    <p class="bottom-subtitle">Average Review Score</p>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="bottom-card">
                    <p class="bottom-value">N/A</p>
                    <p class="bottom-subtitle">Average Review Score</p>
                </div>""",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
