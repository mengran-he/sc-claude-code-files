"""
Business metrics calculation module for e-commerce data analysis.

This module provides:
- BusinessMetricsCalculator: Calculates revenue, product, geographic, satisfaction,
  and delivery metrics from processed sales data.
- MetricsVisualizer: Creates business-oriented charts and plots from report data.
- Utility functions: Formatting and summary printing helpers.
"""

import calendar
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.express as px
import plotly.graph_objects as go

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from data_loader import categorize_delivery_speed


# Business-oriented color palette
COLORS = {
    "primary":   "#1565C0",  # Dark blue
    "secondary": "#1976D2",  # Medium blue
    "accent":    "#42A5F5",  # Light blue
    "success":   "#2E7D32",  # Green
    "warning":   "#E65100",  # Deep orange
    "danger":    "#B71C1C",  # Dark red
    "neutral":   "#546E7A",  # Blue-grey
    "purple":    "#7B1FA2",
    "teal":      "#00838F",
    "olive":     "#558B2F",
}

COLOR_PALETTE = list(COLORS.values())


class BusinessMetricsCalculator:
    """
    Calculates business performance metrics from a processed e-commerce sales dataset.

    The input dataset is expected to contain delivered orders only and should include
    columns: order_id, price, purchase_year, purchase_month.
    Optional enrichment columns: product_category_name, customer_state,
    review_score, delivery_days.
    """

    def __init__(self, sales_data: pd.DataFrame):
        """
        Initialize the calculator.

        Args:
            sales_data: Processed sales dataset containing delivered orders.
        """
        self.sales_data = sales_data.copy()
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate that required columns are present."""
        required = ["price", "order_id", "purchase_year"]
        missing = [c for c in required if c not in self.sales_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # ------------------------------------------------------------------
    # Revenue metrics
    # ------------------------------------------------------------------

    def calculate_revenue_metrics(
        self,
        current_year: int,
        previous_year: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Calculate total revenue, order volume, and average order value.

        Optionally computes year-over-year growth rates when previous_year
        is supplied.

        Args:
            current_year: Year to analyze.
            previous_year: Baseline year for growth calculations.

        Returns:
            Dictionary of revenue metrics and, when applicable, growth rates.
        """
        cur = self.sales_data[self.sales_data["purchase_year"] == current_year]

        metrics: Dict[str, Any] = {
            "total_revenue": cur["price"].sum(),
            "total_orders": cur["order_id"].nunique(),
            "average_order_value": cur.groupby("order_id")["price"].sum().mean(),
            "total_items_sold": len(cur),
        }

        if previous_year is not None:
            prev = self.sales_data[self.sales_data["purchase_year"] == previous_year]
            prev_rev = prev["price"].sum()
            prev_ord = prev["order_id"].nunique()
            prev_aov = prev.groupby("order_id")["price"].sum().mean()

            def growth(new: float, old: float) -> float:
                return ((new - old) / old * 100) if old > 0 else 0.0

            metrics.update(
                {
                    "previous_year_revenue": prev_rev,
                    "previous_year_orders": prev_ord,
                    "previous_year_aov": prev_aov,
                    "revenue_growth_rate": growth(metrics["total_revenue"], prev_rev),
                    "order_growth_rate": growth(metrics["total_orders"], prev_ord),
                    "aov_growth_rate": growth(metrics["average_order_value"], prev_aov),
                }
            )

        return metrics

    # ------------------------------------------------------------------
    # Monthly trends
    # ------------------------------------------------------------------

    def calculate_monthly_trends(self, year: int) -> pd.DataFrame:
        """
        Compute month-over-month revenue, order count, and average order value.

        Args:
            year: Year to analyze.

        Returns:
            DataFrame with columns: month, revenue, orders, avg_order_value,
            revenue_growth (%), order_growth (%), aov_growth (%).
        """
        year_data = self.sales_data[self.sales_data["purchase_year"] == year]

        monthly_revenue = year_data.groupby("purchase_month")["price"].sum()
        monthly_orders = year_data.groupby("purchase_month")["order_id"].nunique()
        monthly_aov = (
            year_data.groupby(["purchase_month", "order_id"])["price"]
            .sum()
            .groupby("purchase_month")
            .mean()
        )

        monthly = pd.DataFrame(
            {
                "month": monthly_revenue.index,
                "revenue": monthly_revenue.values,
                "orders": monthly_orders.values,
                "avg_order_value": monthly_aov.values,
            }
        )

        monthly["revenue_growth"] = monthly["revenue"].pct_change() * 100
        monthly["order_growth"] = monthly["orders"].pct_change() * 100
        monthly["aov_growth"] = monthly["avg_order_value"].pct_change() * 100

        return monthly

    # ------------------------------------------------------------------
    # Product performance
    # ------------------------------------------------------------------

    def analyze_product_performance(
        self, year: int, top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze revenue and volume by product category.

        Args:
            year: Year to analyze.
            top_n: Number of top categories to include in the summary table.

        Returns:
            Dictionary with keys:
              - all_categories: Full category metrics sorted by revenue.
              - top_categories: Top-N categories by revenue.
            Returns {"error": <message>} if product data is unavailable.
        """
        year_data = self.sales_data[self.sales_data["purchase_year"] == year]

        if "product_category_name" not in year_data.columns:
            return {"error": "Product category data not available"}

        agg = (
            year_data.groupby("product_category_name")
            .agg(
                total_revenue=("price", "sum"),
                avg_item_price=("price", "mean"),
                items_sold=("price", "count"),
                unique_orders=("order_id", "nunique"),
            )
            .round(2)
            .reset_index()
        )

        total = agg["total_revenue"].sum()
        agg["revenue_share"] = (agg["total_revenue"] / total * 100).round(2)
        agg = agg.sort_values("total_revenue", ascending=False).reset_index(drop=True)

        return {
            "all_categories": agg,
            "top_categories": agg.head(top_n),
        }

    # ------------------------------------------------------------------
    # Geographic performance
    # ------------------------------------------------------------------

    def analyze_geographic_performance(self, year: int) -> pd.DataFrame:
        """
        Calculate revenue and order metrics aggregated by US state.

        Args:
            year: Year to analyze.

        Returns:
            DataFrame sorted by revenue with columns: state, revenue, orders,
            avg_order_value. Returns a DataFrame with an 'error' column when
            geographic data is unavailable.
        """
        year_data = self.sales_data[self.sales_data["purchase_year"] == year]

        if "customer_state" not in year_data.columns:
            return pd.DataFrame({"error": ["Geographic data not available"]})

        state_revenue = year_data.groupby("customer_state")["price"].sum().rename("revenue")
        state_orders = year_data.groupby("customer_state")["order_id"].nunique().rename("orders")
        state_aov = (
            year_data.groupby(["customer_state", "order_id"])["price"]
            .sum()
            .groupby("customer_state")
            .mean()
            .rename("avg_order_value")
        )

        state_metrics = (
            pd.concat([state_revenue, state_orders, state_aov], axis=1)
            .reset_index()
            .rename(columns={"customer_state": "state"})
        )

        return state_metrics.sort_values("revenue", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Customer satisfaction
    # ------------------------------------------------------------------

    def analyze_customer_satisfaction(self, year: int) -> Dict[str, Any]:
        """
        Compute review score statistics and the full score distribution.

        Args:
            year: Year to analyze.

        Returns:
            Dictionary with avg_review_score, total_reviews, percentage metrics
            for star ratings, and score_distribution (proportion per score 1-5).
            Returns {"error": <message>} if review data is unavailable.
        """
        year_data = self.sales_data[self.sales_data["purchase_year"] == year]

        if "review_score" not in year_data.columns:
            return {"error": "Review data not available"}

        # One review per order
        order_data = year_data.drop_duplicates("order_id").dropna(subset=["review_score"])

        score_distribution = (
            order_data["review_score"].value_counts(normalize=True).sort_index()
        )

        return {
            "avg_review_score": order_data["review_score"].mean(),
            "total_reviews": int(order_data["review_score"].count()),
            "score_5_percentage": (order_data["review_score"] == 5).mean() * 100,
            "score_4_plus_percentage": (order_data["review_score"] >= 4).mean() * 100,
            "score_1_2_percentage": (order_data["review_score"] <= 2).mean() * 100,
            "score_distribution": score_distribution,
        }

    # ------------------------------------------------------------------
    # Delivery performance
    # ------------------------------------------------------------------

    def analyze_delivery_performance(self, year: int) -> Dict[str, Any]:
        """
        Calculate delivery time metrics and their correlation with review scores.

        Delivery time is grouped into three buckets:
          - 1-3 days (fast)
          - 4-7 days (standard)
          - 8+ days (slow)

        Args:
            year: Year to analyze.

        Returns:
            Dictionary with avg/median delivery days, speed-tier percentages,
            and optionally delivery_satisfaction (DataFrame with avg review score
            per delivery bucket). Returns {"error": <message>} when delivery
            data is unavailable.
        """
        year_data = self.sales_data[self.sales_data["purchase_year"] == year]

        if "delivery_days" not in year_data.columns:
            return {"error": "Delivery data not available"}

        order_data = (
            year_data.drop_duplicates("order_id")
            .dropna(subset=["delivery_days"])
            .copy()
        )

        metrics: Dict[str, Any] = {
            "avg_delivery_days": order_data["delivery_days"].mean(),
            "median_delivery_days": order_data["delivery_days"].median(),
            "fast_delivery_percentage": (order_data["delivery_days"] <= 3).mean() * 100,
            "slow_delivery_percentage": (order_data["delivery_days"] > 7).mean() * 100,
        }

        if "review_score" in order_data.columns:
            order_data["delivery_bucket"] = order_data["delivery_days"].apply(
                categorize_delivery_speed
            )
            bucket_order = ["1-3 days", "4-7 days", "8+ days"]

            delivery_sat = (
                order_data.groupby("delivery_bucket")["review_score"]
                .agg(avg_review_score="mean", order_count="count")
                .reindex(bucket_order)
                .reset_index()
                .rename(columns={"delivery_bucket": "delivery_bucket"})
                .dropna(subset=["avg_review_score"])
            )

            metrics["delivery_satisfaction"] = delivery_sat

        return metrics

    # ------------------------------------------------------------------
    # Comprehensive report
    # ------------------------------------------------------------------

    def generate_comprehensive_report(
        self,
        current_year: int,
        previous_year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a full business performance report for the specified period.

        Args:
            current_year: Primary analysis year.
            previous_year: Optional baseline year for growth comparisons.

        Returns:
            Dictionary containing analysis_period, comparison_period,
            revenue_metrics, monthly_trends, product_performance,
            geographic_performance, customer_satisfaction, and
            delivery_performance.
        """
        return {
            "analysis_period": current_year,
            "comparison_period": previous_year,
            "revenue_metrics": self.calculate_revenue_metrics(current_year, previous_year),
            "monthly_trends": self.calculate_monthly_trends(current_year),
            "product_performance": self.analyze_product_performance(current_year),
            "geographic_performance": self.analyze_geographic_performance(current_year),
            "customer_satisfaction": self.analyze_customer_satisfaction(current_year),
            "delivery_performance": self.analyze_delivery_performance(current_year),
        }


# ==========================================================================
# Visualizer
# ==========================================================================


class MetricsVisualizer:
    """
    Creates business-oriented visualizations from a BusinessMetricsCalculator report.

    All matplotlib figures use a consistent corporate blue color scheme.
    Plotly figures are used for interactive geographic maps.
    """

    def __init__(self, report_data: Dict[str, Any]):
        """
        Initialize the visualizer.

        Args:
            report_data: Report dictionary returned by
                         BusinessMetricsCalculator.generate_comprehensive_report().
        """
        self.report_data = report_data
        self.color_palette = COLOR_PALETTE

    # ------------------------------------------------------------------
    # Revenue trend
    # ------------------------------------------------------------------

    def plot_revenue_trend(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot monthly revenue as a line chart with data point labels.

        X-axis shows abbreviated month names; y-axis is formatted as USD.

        Args:
            figsize: Figure dimensions (width, height) in inches.

        Returns:
            Matplotlib Figure object.
        """
        monthly_data = self.report_data["monthly_trends"]
        year = self.report_data["analysis_period"]
        month_labels = [calendar.month_abbr[m] for m in monthly_data["month"]]

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            monthly_data["month"],
            monthly_data["revenue"],
            marker="o",
            linewidth=2,
            markersize=8,
            color=COLORS["primary"],
        )
        ax.fill_between(
            monthly_data["month"],
            monthly_data["revenue"],
            alpha=0.1,
            color=COLORS["primary"],
        )

        ax.set_title(
            f"Monthly Revenue Trend (Jan {year} - Dec {year})",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Revenue (USD)", fontsize=12)
        ax.set_xticks(monthly_data["month"])
        ax.set_xticklabels(month_labels, rotation=45, ha="right")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(True, alpha=0.3)

        for _, row in monthly_data.iterrows():
            ax.annotate(
                f"${row['revenue']:,.0f}",
                (row["month"], row["revenue"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Product category performance
    # ------------------------------------------------------------------

    def plot_category_performance(
        self, top_n: int = 10, figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot a horizontal bar chart of the top N product categories by revenue.

        Each bar is labeled with revenue and revenue share percentage.

        Args:
            top_n: Number of top categories to display.
            figsize: Figure dimensions (width, height) in inches.

        Returns:
            Matplotlib Figure object.
        """
        if "error" in self.report_data["product_performance"]:
            return self._empty_plot(figsize, "Product category data not available")

        category_data = (
            self.report_data["product_performance"]["top_categories"].head(top_n)
        )
        year = self.report_data["analysis_period"]

        fig, ax = plt.subplots(figsize=figsize)

        colors = (self.color_palette * ((len(category_data) // len(self.color_palette)) + 1))[
            : len(category_data)
        ]
        bars = ax.barh(
            range(len(category_data)),
            category_data["total_revenue"],
            color=colors,
        )

        ax.set_yticks(range(len(category_data)))
        ax.set_yticklabels(category_data["product_category_name"], fontsize=11)
        ax.invert_yaxis()

        ax.set_title(
            f"Top {top_n} Product Categories by Revenue ({year})",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Revenue (USD)", fontsize=12)
        ax.set_ylabel("Product Category", fontsize=12)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        for bar, row in zip(bars, category_data.itertuples()):
            ax.text(
                bar.get_width() * 1.01,
                bar.get_y() + bar.get_height() / 2,
                f"${row.total_revenue:,.0f}  ({row.revenue_share:.1f}%)",
                va="center",
                ha="left",
                fontsize=9,
            )

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Geographic heatmap
    # ------------------------------------------------------------------

    def plot_geographic_heatmap(self) -> go.Figure:
        """
        Create an interactive US choropleth map of revenue by state.

        Returns:
            Plotly Figure object.
        """
        geo_data = self.report_data["geographic_performance"]
        year = self.report_data["analysis_period"]

        if "error" in geo_data.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Geographic data not available",
                x=0.5,
                y=0.5,
                showarrow=False,
                font_size=16,
            )
            return fig

        fig = px.choropleth(
            geo_data,
            locations="state",
            color="revenue",
            locationmode="USA-states",
            scope="usa",
            title=f"Revenue by State ({year})",
            color_continuous_scale="Blues",
            labels={"revenue": "Revenue (USD)"},
        )

        fig.update_layout(
            title_font_size=16,
            title_x=0.5,
            geo=dict(showframe=False, showcoastlines=True),
        )

        return fig

    # ------------------------------------------------------------------
    # Review score distribution
    # ------------------------------------------------------------------

    def plot_review_score_distribution(
        self, figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot the distribution of customer review scores (1-5) as a horizontal bar chart.

        Each score is color-coded from red (1) to green (5).

        Args:
            figsize: Figure dimensions (width, height) in inches.

        Returns:
            Matplotlib Figure object.
        """
        year = self.report_data["analysis_period"]
        satisfaction = self.report_data["customer_satisfaction"]

        if "error" in satisfaction:
            return self._empty_plot(figsize, "Review data not available")

        if "score_distribution" not in satisfaction:
            return self._empty_plot(figsize, "Score distribution data not available")

        score_dist = satisfaction["score_distribution"]
        scores = score_dist.index.astype(int)
        percentages = score_dist.values * 100

        score_color_map = {
            1: COLORS["danger"],
            2: COLORS["warning"],
            3: COLORS["neutral"],
            4: COLORS["secondary"],
            5: COLORS["success"],
        }
        colors = [score_color_map.get(s, COLORS["primary"]) for s in scores]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(scores, percentages, color=colors)

        ax.set_title(
            f"Customer Review Score Distribution ({year})",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Percentage of Reviews (%)", fontsize=12)
        ax.set_ylabel("Review Score", fontsize=12)
        ax.set_yticks(scores)
        ax.set_yticklabels([f"Score {s}" for s in scores], fontsize=11)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

        for bar, pct in zip(bars, percentages):
            ax.text(
                pct + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%",
                va="center",
                ha="left",
                fontsize=10,
            )

        plt.tight_layout()
        return fig

    # Backward-compatible alias
    def plot_review_distribution(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Alias for plot_review_score_distribution."""
        return self.plot_review_score_distribution(figsize=figsize)

    # ------------------------------------------------------------------
    # Delivery-satisfaction correlation
    # ------------------------------------------------------------------

    def plot_delivery_satisfaction(
        self, figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot average review score by delivery time bucket (1-3, 4-7, 8+ days).

        Includes an overall average reference line and order count annotations.

        Args:
            figsize: Figure dimensions (width, height) in inches.

        Returns:
            Matplotlib Figure object.
        """
        year = self.report_data["analysis_period"]
        delivery_metrics = self.report_data["delivery_performance"]

        if "error" in delivery_metrics or "delivery_satisfaction" not in delivery_metrics:
            return self._empty_plot(figsize, "Delivery satisfaction data not available")

        delivery_sat = delivery_metrics["delivery_satisfaction"].reset_index(drop=True)

        bucket_colors = [COLORS["success"], COLORS["secondary"], COLORS["warning"]]
        bar_colors = bucket_colors[: len(delivery_sat)]

        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.bar(
            delivery_sat["delivery_bucket"],
            delivery_sat["avg_review_score"],
            color=bar_colors,
            width=0.5,
        )

        ax.set_title(
            f"Average Review Score by Delivery Time ({year})",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Delivery Time Category", fontsize=12)
        ax.set_ylabel("Average Review Score (1-5)", fontsize=12)
        ax.set_ylim(0, 5.5)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.05,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        if "order_count" in delivery_sat.columns:
            for i, row in delivery_sat.iterrows():
                ax.text(
                    i,
                    0.15,
                    f"n={row['order_count']:,}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )

        # Overall average reference line
        if "error" not in self.report_data["customer_satisfaction"]:
            overall_avg = self.report_data["customer_satisfaction"]["avg_review_score"]
        else:
            overall_avg = delivery_sat["avg_review_score"].mean()

        ax.axhline(
            y=overall_avg,
            color=COLORS["neutral"],
            linestyle="--",
            alpha=0.7,
            label=f"Overall average ({overall_avg:.2f})",
        )
        ax.legend(fontsize=10)

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _empty_plot(self, figsize: Tuple[int, int], message: str) -> plt.Figure:
        """Return a blank figure displaying a message."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, message, ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_axis_off()
        return fig


# ==========================================================================
# Formatting helpers
# ==========================================================================


def format_currency(value: float) -> str:
    """Return a value formatted as a USD currency string (e.g., '$1,234.56')."""
    return f"${value:,.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Return a value formatted as a percentage string (e.g., '12.3%')."""
    return f"{value:.{decimals}f}%"


def print_metrics_summary(report: Dict[str, Any]) -> None:
    """
    Print a concise, formatted summary of key business metrics.

    Args:
        report: Report dictionary from BusinessMetricsCalculator.generate_comprehensive_report().
    """
    print("=" * 60)
    print(f"BUSINESS METRICS SUMMARY - {report['analysis_period']}")
    print("=" * 60)

    rev = report["revenue_metrics"]
    print(f"\nREVENUE PERFORMANCE:")
    print(f"  Total Revenue:        {format_currency(rev['total_revenue'])}")
    print(f"  Total Orders:         {rev['total_orders']:,}")
    print(f"  Average Order Value:  {format_currency(rev['average_order_value'])}")
    if "revenue_growth_rate" in rev:
        print(f"  Revenue Growth:       {format_percentage(rev['revenue_growth_rate'])}")
        print(f"  Order Growth:         {format_percentage(rev['order_growth_rate'])}")

    satisfaction = report["customer_satisfaction"]
    if "error" not in satisfaction:
        print(f"\nCUSTOMER SATISFACTION:")
        print(f"  Average Review Score: {satisfaction['avg_review_score']:.2f}/5.0")
        print(f"  High Satisfaction (4+): {format_percentage(satisfaction['score_4_plus_percentage'])}")

    delivery = report["delivery_performance"]
    if "error" not in delivery:
        print(f"\nDELIVERY PERFORMANCE:")
        print(f"  Average Delivery Time:   {delivery['avg_delivery_days']:.1f} days")
        print(f"  Fast Delivery (<=3 days): {format_percentage(delivery['fast_delivery_percentage'])}")

    print("=" * 60)
