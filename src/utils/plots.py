"""Plotting utilities."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_actuarial_profile(dfa: pd.DataFrame, feature_type: str, feature: str, bins: Optional[List[int]]) -> None:
    """Plot univariate distribution against frequency of claims"""
    assert feature_type in ["num", "cat"], "feature type must be 'num' or 'cat'"

    df = dfa.copy()

    # Create bins
    if feature_type == "num":
        df["bin"] = pd.cut(df[feature], bins=bins)
    else:
        df["bin"] = df[feature]

    # Aggregate exposure and number of claims according to bins
    profile = (
        df.groupby("bin", observed=True).agg(Exposure=("Exposure", "sum"), Claims=("ClaimNb", "sum")).reset_index()
    )

    # Sort
    profile = profile.sort_values("bin")

    # Now convert to string for the x-axis labels
    profile["bin_label"] = profile["bin"].astype(str)

    # Calculate Frequency
    profile["Frequency"] = profile["Claims"] / profile["Exposure"]

    # Initialize figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Exposure as bars (volume)
    fig.add_trace(
        go.Bar(
            x=profile["bin_label"],
            y=profile["Exposure"],
            name="Total Exposure (Years)",
            marker_color="lightslategrey",
            opacity=0.6,
        ),
        secondary_y=False,
    )

    # Add Frequency (risk)
    if feature_type == "num":
        mode = "lines+markers"
    else:
        mode = "markers"
    fig.add_trace(
        go.Scatter(
            x=profile["bin_label"],
            y=profile["Frequency"],
            name="Claim Frequency",
            mode=mode,
            line=dict(color="firebrick", width=3),
        ),
        secondary_y=True,
    )

    # Styling
    fig.update_layout(
        title_text=f"<b>Actuarial Profile: {feature} vs Risk</b>",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis2_range=[0, max(0.2, 1.1 * max(profile["Frequency"]))],
    )

    fig.update_xaxes(title_text=feature)
    fig.update_yaxes(title_text="Volume (Exposure years)", secondary_y=False)
    fig.update_yaxes(title_text="Annualized Frequency", secondary_y=True)

    fig.show()


def plot_interaction_heatmap(
    dfa: pd.DataFrame, x_feature: str, x_bins: List[int], y_feature: str, y_bins: List[int]
) -> None:
    """Plot bivariate distributions against frequency of claims."""
    df = dfa.copy()

    # Create bins for both features
    df[f"{x_feature}_bin"] = pd.cut(df[x_feature], bins=x_bins)
    df[f"{y_feature}_bin"] = pd.cut(df[y_feature], bins=y_bins)

    # Aggregate Frequency
    interaction = (
        df.groupby([f"{x_feature}_bin", f"{y_feature}_bin"], observed=True)
        .agg(Freq=("ClaimNb", "sum"), Exp=("Exposure", "sum"))
        .reset_index()
    )

    # Convert to strings
    interaction[f"{x_feature}_bin"] = interaction[f"{x_feature}_bin"].astype(str)
    interaction[f"{y_feature}_bin"] = interaction[f"{y_feature}_bin"].astype(str)

    # Calculate Frequency
    interaction["Frequency"] = interaction["Freq"] / interaction["Exp"]

    # Plot Heatmap
    fig = px.density_heatmap(
        interaction,
        x=f"{x_feature}_bin",
        y=f"{y_feature}_bin",
        z="Frequency",
        title=f"<b>Interaction: {x_feature} vs {y_feature} on Claim Frequency</b>",
        color_continuous_scale="Reds",
        text_auto=".3f",  # Shows 3 decimal places on the heatmap
        category_orders={
            x_feature: interaction[f"{x_feature}_bin"].unique().tolist(),
            y_feature: interaction[f"{y_feature}_bin"].unique().tolist()[::-1],
        },
        labels={f"{x_feature}_bin": x_feature, f"{y_feature}_bin": y_feature},
    )

    fig.show()


def plot_exposure_by_claim_boxplot(df):
    "Boxplot to compare claims and exposure (together determine frequency of claims)."
    fig = px.box(
        df,
        x="ClaimNb",
        y="Exposure",
        title="<b>Exposure distribution by ClaimNb</b>",
        labels={"ClaimNb": "Number of Claims", "Exposure": "Policy Exposure (Years)"},
        points=False,  # Hides outliers to keep the chart clean and fast
    )

    fig.update_layout(
        showlegend=False,
        template="plotly_white",
        yaxis_range=[0, 2.1],  # Focuses on our 0-2 year range
    )

    fig.show()


def _get_lorenz_coords(actual: np.ndarray, predicted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Sort actual values based on predictions (highest to lowest)
    order = np.argsort(predicted)[::-1]
    actual_sorted = actual.iloc[order].values

    # Cumulative sums
    cum_actual = np.cumsum(actual_sorted) / np.sum(actual_sorted)
    cum_pop = np.arange(1, len(actual) + 1) / len(actual)

    # Prepend (0,0)
    return np.insert(cum_pop, 0, 0), np.insert(cum_actual, 0, 0)


def plot_lorenz_curves(df: pd.DataFrame) -> None:
    # Calculate coordinates for models
    pop_glm, actual_glm = _get_lorenz_coords(df["ClaimNb"], df["glm_pred"])
    pop_xgb, actual_xgb = _get_lorenz_coords(df["ClaimNb"], df["xgb_pred"])
    pop_bmk, actual_bmk = _get_lorenz_coords(df["ClaimNb"], df["bmk_pred"])

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=pop_glm, y=actual_glm, name="GLM Lorenz Curve", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=pop_xgb, y=actual_xgb, name="XGBoost Lorenz Curve", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=pop_bmk, y=actual_bmk, name="Benchmark Lorenz Curve", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random Selection (Null)", line=dict(color="gray", dash="dash")))

    fig.update_layout(
        title="Lorenz Curve: Benchmark vs GLM vs XGBoost (Test Set)",
        xaxis_title="Cumulative Share of Population (Highest to Lowest Risk)",
        yaxis_title="Cumulative Share of Actual Claims Captured",
        template="plotly_white",
        height=600,
        width=800,
        xaxis_range=[0.0, 1.0],
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
    )

    fig.show()


def plot_double_lift_chart(df_test: pd.DataFrame) -> None:
    # Calculate ratio
    df_test["ratio"] = df_test["xgb_pred"] / df_test["glm_pred"]

    # Sort into 10 buckets (deciles)
    df_test["decile"] = pd.qcut(df_test["ratio"], 10, labels=False)

    # Aggregate actual vs predicted frequency
    lift_data = (
        df_test.groupby("decile")
        .agg({"ClaimNb": "sum", "exposure_stable": "sum", "glm_pred": "sum", "xgb_pred": "sum"})
        .reset_index()
    )

    lift_data["actual_freq"] = lift_data["ClaimNb"] / lift_data["exposure_stable"]
    lift_data["glm_freq"] = lift_data["glm_pred"] / lift_data["exposure_stable"]
    lift_data["xgb_freq"] = lift_data["xgb_pred"] / lift_data["exposure_stable"]

    # Plot
    fig = go.Figure()

    fig.add_trace(
        go.Bar(x=lift_data["decile"], y=lift_data["actual_freq"], name="Actual Frequency", marker_color="lightgrey")
    )
    fig.add_trace(
        go.Scatter(
            x=lift_data["decile"], y=lift_data["glm_freq"], name="GLM Pred Frequency", line=dict(color="blue", width=3)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=lift_data["decile"],
            y=lift_data["xgb_freq"],
            name="XGBoost Pred Frequency",
            line=dict(color="orange", width=3),
        )
    )

    fig.update_layout(
        title="Double Lift Chart: Deciles of (XGB / GLM) Ratio",
        xaxis_title="Decile (1=GLM overestimates relative to XGB, 10=XGB overestimates relative to GLM)",
        yaxis_title="Claim Frequency (actual)",
        template="plotly_white",
        height=600,
        barmode="group",
    )

    fig.show()
