"""Plotting utilities."""

from typing import List, Optional

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
