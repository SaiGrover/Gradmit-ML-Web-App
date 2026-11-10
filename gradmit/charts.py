"""Plotly figure builders shared by the Streamlit pages.

Colour carries one meaning throughout: teal for the selected model, slate for
other models, ochre for baselines and brick for errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from .baselines import BASELINES
from .style import CORAL, GOLD, GRID, MINT, MINT_SOFT, MUT, PANEL_2, SLATE, TXT

# Colour carries one meaning on every page: mint for the selected model, slate for
# other models, gold for baselines and coral for errors.
INK = TXT
TEAL = MINT
TEAL_SOFT = MINT_SOFT
OCHRE = GOLD
BRICK = CORAL
FONT = "Inter, system-ui, sans-serif"

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "brier": "Brier score",
}


def _layout(fig: go.Figure, height: int = 360, **kwargs) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=28, b=8),
        font=dict(family=FONT, size=13, color=MUT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font=dict(family=FONT, size=13, color=TXT), bgcolor=PANEL_2, bordercolor=GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=MUT)),
        **kwargs,
    )
    axis = dict(gridcolor=GRID, linecolor=GRID, zeroline=False, automargin=True,
                tickfont=dict(color=MUT), title_font=dict(color=MUT))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def model_colour(model: str, selected: str) -> str:
    if model == selected:
        return TEAL
    return OCHRE if model in BASELINES else SLATE


def interval_chart(comparison: pd.DataFrame, metric: str, selected: str) -> go.Figure:
    """Mean and 95% CI of a repeated-CV metric for every model (forest plot)."""
    data = comparison.sort_values(f"{metric}_mean")
    fig = go.Figure()
    for _, row in data.iterrows():
        mean = row[f"{metric}_mean"]
        fig.add_trace(
            go.Scatter(
                x=[mean],
                y=[row["model"]],
                mode="markers",
                marker=dict(size=13, color=model_colour(row["model"], selected)),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[row[f"{metric}_ci_high"] - mean],
                    arrayminus=[mean - row[f"{metric}_ci_low"]],
                    thickness=2,
                    width=6,
                ),
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['model']}</b><br>{METRIC_LABELS[metric]} {mean:.3f}"
                    f"<br>95% CI {row[f'{metric}_ci_low']:.3f} to {row[f'{metric}_ci_high']:.3f}"
                    "<extra></extra>"
                ),
            )
        )
    fig.update_xaxes(title=f"{METRIC_LABELS[metric]} (mean of 50 folds, 95% CI)")
    return _layout(fig, height=60 + 44 * len(data))


def fold_distribution(folds: pd.DataFrame, metric: str, selected: str, order: list[str]) -> go.Figure:
    """Distribution of the 50 validation-fold scores per model."""
    fig = go.Figure()
    for model in order[::-1]:
        values = folds.loc[folds["model"] == model, metric]
        fig.add_trace(
            go.Box(
                x=values,
                name=model,
                orientation="h",
                boxpoints="all",
                jitter=0.45,
                pointpos=0,
                marker=dict(size=4, opacity=0.55, color=model_colour(model, selected)),
                line=dict(color=model_colour(model, selected), width=1.5),
                fillcolor="rgba(0,0,0,0)",
                showlegend=False,
                hovertemplate=f"{model}<br>%{{x:.3f}}<extra></extra>",
            )
        )
    fig.update_xaxes(title=f"{METRIC_LABELS[metric]} per validation fold")
    return _layout(fig, height=70 + 52 * len(order))


def paired_forest(paired: pd.DataFrame) -> go.Figure:
    """Difference in balanced accuracy (selected model minus baseline) with bootstrap CI."""
    fig = go.Figure()
    for _, row in paired.iterrows():
        colour = TEAL if row["significant"] else SLATE
        fig.add_trace(
            go.Scatter(
                x=[row["difference"]],
                y=[f"vs {row['baseline']}"],
                mode="markers",
                marker=dict(size=13, color=colour),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[row["ci_high"] - row["difference"]],
                    arrayminus=[row["difference"] - row["ci_low"]],
                    thickness=2,
                    width=6,
                ),
                showlegend=False,
                hovertemplate=(
                    f"<b>vs {row['baseline']}</b><br>Difference {row['difference']:+.3f}"
                    f"<br>95% CI {row['ci_low']:+.3f} to {row['ci_high']:+.3f}"
                    f"<br>Holm p = {row['p_value_holm']:.4f}<extra></extra>"
                ),
            )
        )
    fig.add_vline(x=0, line=dict(color=INK, width=1, dash="dot"))
    fig.update_xaxes(title="Model minus baseline, balanced accuracy")
    return _layout(fig, height=190)


def research_chart(research: pd.DataFrame) -> go.Figure:
    """Share of each Research group above the Chance-of-Admit threshold, Wilson 95% CI."""
    data = research.sort_values("research", ascending=False)
    labels = [f"{g}<br>n = {n}" for g, n in zip(data["group"], data["n"])]
    rate = data["positive_rate"] * 100
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=rate,
            marker_color=[TEAL, TEAL_SOFT],
            width=0.5,
            error_y=dict(
                type="data",
                symmetric=False,
                array=data["ci_high"] * 100 - rate,
                arrayminus=rate - data["ci_low"] * 100,
                color=INK,
                thickness=1.5,
                width=8,
            ),
            customdata=np.stack(
                [data["n_positive"], data["n"], data["ci_low"] * 100, data["ci_high"] * 100], axis=1
            ),
            hovertemplate=(
                "%{customdata[0]} of %{customdata[1]} above threshold"
                "<br>Wilson 95% CI %{customdata[2]:.1f}% to %{customdata[3]:.1f}%<extra></extra>"
            ),
        )
    )
    for label, r, high in zip(labels, rate, data["ci_high"] * 100):
        fig.add_annotation(
            x=label, y=high, text=f"<b>{r:.1f}%</b>", showarrow=False, yshift=14,
            font=dict(size=15, color=INK),
        )
    fig.update_yaxes(title="% above Chance-of-Admit threshold", range=[0, 100])
    return _layout(fig, height=380)


def class_balance_chart(sensitivity: pd.DataFrame) -> go.Figure:
    labels = [f"{t:.2f}" for t in sensitivity["target_threshold"]]
    pos = sensitivity["positive_share"] * 100
    fig = go.Figure()
    fig.add_bar(
        x=labels, y=pos, name="Proxy-positive", marker_color=TEAL,
        text=[f"{p:.1f}%" for p in pos], textposition="inside",
    )
    fig.add_bar(
        x=labels, y=100 - pos, name="Proxy-negative", marker_color="rgba(186,214,220,.16)",
        text=[f"{100 - p:.1f}%" for p in pos], textposition="inside",
    )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Target threshold on Chance of Admit", type="category")
    fig.update_yaxes(title="% of 500 records", range=[0, 100])
    return _layout(fig, height=320)


def sensitivity_chart(sensitivity: pd.DataFrame, selected: str) -> go.Figure:
    """Balanced accuracy of the majority rule, CGPA cutoff and selected model per threshold."""
    labels = [f"{t:.2f}" for t in sensitivity["target_threshold"]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels, y=sensitivity["majority_balanced_accuracy"], name="Majority baseline",
            mode="lines+markers", line=dict(color=OCHRE, dash="dot"), marker=dict(size=9, symbol="square"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels, y=sensitivity["cgpa_balanced_accuracy"], name="CGPA cutoff baseline",
            mode="lines+markers", line=dict(color=OCHRE), marker=dict(size=10, symbol="diamond"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels, y=sensitivity["model_balanced_accuracy"], name=selected,
            mode="lines+markers", line=dict(color=TEAL, width=3), marker=dict(size=11),
            error_y=dict(
                type="data", symmetric=False,
                array=sensitivity["model_balanced_accuracy_ci_high"] - sensitivity["model_balanced_accuracy"],
                arrayminus=sensitivity["model_balanced_accuracy"] - sensitivity["model_balanced_accuracy_ci_low"],
                thickness=1.5, width=6,
            ),
        )
    )
    fig.update_xaxes(title="Target threshold on Chance of Admit", type="category")
    fig.update_yaxes(title="Balanced accuracy (mean of 50 folds)", range=[0.45, 1.0])
    return _layout(fig, height=380)


def confusion_chart(cm: dict) -> go.Figure:
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    names = [["True negative", "False positive"], ["False negative", "True positive"]]
    text = [[f"{names[i][j]}<br><b>{z[i][j]}</b>" for j in range(2)] for i in range(2)]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Predicted below", "Predicted above"],
            y=["Proxy-negative", "Proxy-positive"],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=14, color=TXT),
            colorscale=[[0, PANEL_2], [1, TEAL]],
            showscale=False,
            hoverinfo="skip",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height=320)


def calibration_chart(y_true, prob, n_bins: int = 8) -> go.Figure:
    frac, mean_pred = calibration_curve(y_true, prob, n_bins=n_bins, strategy="quantile")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                   line=dict(color=SLATE, dash="dot"))
    )
    fig.add_trace(
        go.Scatter(x=mean_pred, y=frac, mode="lines+markers", name="Cross-fitted predictions",
                   line=dict(color=TEAL, width=3), marker=dict(size=9))
    )
    fig.update_xaxes(title="Mean predicted probability (quantile bin)", range=[0, 1])
    fig.update_yaxes(title="Observed proxy-positive share", range=[0, 1])
    return _layout(fig, height=360)


def roc_chart(y_true, prob) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_true, prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                             line=dict(color=SLATE, dash="dot")))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Cross-fitted ROC",
                             line=dict(color=TEAL, width=3)))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    return _layout(fig, height=360)


def pr_chart(y_true, prob) -> go.Figure:
    precision, recall, _ = precision_recall_curve(y_true, prob)
    prevalence = float(np.mean(y_true))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[prevalence, prevalence], mode="lines",
                             name="Prevalence", line=dict(color=SLATE, dash="dot")))
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="Cross-fitted PR",
                             line=dict(color=TEAL, width=3)))
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1.02])
    return _layout(fig, height=360)


def feature_histogram(df: pd.DataFrame, feature: str, target: pd.Series, label: str) -> go.Figure:
    fig = go.Figure()
    for value, name, colour in ((1, "Proxy-positive", TEAL), (0, "Proxy-negative", OCHRE)):
        fig.add_trace(
            go.Histogram(x=df.loc[target == value, feature], name=name, marker_color=colour,
                         opacity=0.7, nbinsx=24)
        )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title=label)
    fig.update_yaxes(title="Records")
    return _layout(fig, height=340)


def cutoff_histogram(cutoffs: pd.Series) -> go.Figure:
    fig = go.Figure(go.Histogram(x=cutoffs, marker_color=OCHRE, nbinsx=20))
    fig.update_xaxes(title="CGPA cutoff chosen on the training fold")
    fig.update_yaxes(title="Folds (of 50)")
    return _layout(fig, height=300)


def probability_bar(probability: float, threshold: float) -> go.Figure:
    """Estimated probability on a 0-1 track with the classifier threshold marked."""
    fig = go.Figure(
        go.Bar(x=[probability], y=[""], orientation="h", marker_color=TEAL, width=0.5,
               hoverinfo="skip")
    )
    fig.add_vline(x=threshold, line=dict(color=INK, width=2, dash="dot"),
                  annotation_text=f"classifier threshold {threshold:.2f}",
                  annotation_position="top")
    fig.update_xaxes(range=[0, 1], tickformat=".0%", dtick=0.25)
    fig.update_yaxes(showticklabels=False)
    return _layout(fig, height=130)


def correlation_chart(corr: pd.DataFrame, labels: dict) -> go.Figure:
    names = [labels.get(c, c) for c in corr.columns]
    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(),
            x=names,
            y=names,
            zmin=-1,
            zmax=1,
            colorscale=[[0, OCHRE], [0.5, PANEL_2], [1, TEAL]],
            text=corr.round(2).to_numpy(),
            texttemplate="%{text}",
            textfont=dict(size=11, color=TXT),
            colorbar=dict(title="r", thickness=10),
            hovertemplate="%{y} and %{x}<br>Pearson r = %{z:.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height=460)
