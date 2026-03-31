#!/usr/bin/env python3
"""
model_registry_visualizer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Production-grade, class-based Plotly visualisation suite for model_bundle_registry.csv.

Usage — standalone:
    python model_registry_visualizer.py \
        --csv /var/app/data/models/production_ml/tables/model_bundle_registry.csv \
        --out /tmp/metamp_publication_figures

Usage — as stdin payload via docker exec:
    docker exec -i <container> python - < model_registry_visualizer.py

Usage — host-side import:
    from model_registry_visualizer import ModelRegistryVisualizer
    viz = ModelRegistryVisualizer.from_csv("model_bundle_registry.csv")
    viz.display_summary()
    viz.render_all(output_dir="./figures")
"""

from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────────────────────
# Palette & Constants
# ──────────────────────────────────────────────────────────────────────────────

FIGURE_WIDTH  = 1400
FIGURE_HEIGHT = 820

TRAINING_MODE_PALETTE: dict[str, str] = {
    "supervised":      "#01696f",
    "semi_supervised": "#d19900",
}

CLASSIFIER_PALETTE: dict[str, str] = {
    "Logistic Regression":          "#01696f",
    "Decision Tree":                "#d19900",
    "Random Forest":                "#a13544",
    "SVM":                          "#006494",
    "Gradient Boosting Classifier": "#7a39bb",
    "KNeighbors Classifier":        "#da7101",
    "Gaussian NB":                  "#437a22",
}

DR_PALETTE: dict[str, str] = {
    "no_dr": "#01696f",
    "pca":   "#d19900",
    "tsne":  "#5f4bb6",
    "umap":  "#a13544",
}

DR_DISPLAY: dict[str, str] = {
    "no_dr": "No-DR",
    "pca":   "PCA",
    "tsne":  "t-SNE",
    "umap":  "UMAP",
}

FONT_FAMILY   = "Arial Black, Arial, Helvetica Neue, sans-serif"
FONT_COLOR_FG = "#28251d"
FONT_COLOR_MU = "#7a7974"
BG_PAPER      = "#ffffff"
BG_PLOT       = "#ffffff"
GRID_COLOR    = "#d7dde5"
DR_ORDER      = ["no_dr", "pca", "tsne", "umap"]


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelRecord:
    """Strongly-typed representation of a single registry row."""

    artifact_id:               str
    training_mode:             str
    reduction_key:             str
    classifier_name:           str
    artifact_path:             str
    feature_columns:           list[str]
    selected_for_upload:       bool
    selected_for_mode:         bool
    cv_mean_accuracy:          float
    cv_mean_precision:         float
    cv_mean_recall:            float
    cv_mean_f1:                float
    expert_row_count:          int
    expert_accuracy:           float
    expert_precision_weighted: float
    expert_recall_weighted:    float
    expert_f1_weighted:        float
    expert_accuracy_exact:     float
    expert_f1_weighted_exact:  float
    expert_predictions_path:   str
    discrepancy_predictions_path: str

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def dr_display(self) -> str:
        return DR_DISPLAY.get(self.reduction_key, self.reduction_key.upper())

    @property
    def short_label(self) -> str:
        return f"{self.dr_display} · {self.classifier_name}"

    @property
    def cv_expert_f1_gap(self) -> float:
        return float(self.cv_mean_f1) - float(self.expert_f1_weighted)

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ModelRecord":
        def _bool(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() in {"true", "yes", "1"}

        def _float(v: Any) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")

        def _int(v: Any) -> int:
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return 0

        raw = row.get("feature_columns", "[]")
        if isinstance(raw, str):
            try:
                features = json.loads(raw)
            except json.JSONDecodeError:
                features = [f.strip().strip('"') for f in raw.strip("[]").split(",")]
        else:
            features = list(raw)

        return cls(
            artifact_id=str(row.get("artifact_id", "")),
            training_mode=str(row.get("training_mode", "")),
            reduction_key=str(row.get("reduction_key", "")),
            classifier_name=str(row.get("classifier_name", "")),
            artifact_path=str(row.get("artifact_path", "")),
            feature_columns=features,
            selected_for_upload=_bool(row.get("selected_for_upload", False)),
            selected_for_mode=_bool(row.get("selected_for_mode", False)),
            cv_mean_accuracy=_float(row.get("cv_mean_accuracy")),
            cv_mean_precision=_float(row.get("cv_mean_precision")),
            cv_mean_recall=_float(row.get("cv_mean_recall")),
            cv_mean_f1=_float(row.get("cv_mean_f1")),
            expert_row_count=_int(row.get("expert_row_count", 0)),
            expert_accuracy=_float(row.get("expert_accuracy")),
            expert_precision_weighted=_float(row.get("expert_precision_weighted")),
            expert_recall_weighted=_float(row.get("expert_recall_weighted")),
            expert_f1_weighted=_float(row.get("expert_f1_weighted")),
            expert_accuracy_exact=_float(row.get("expert_accuracy_exact")),
            expert_f1_weighted_exact=_float(row.get("expert_f1_weighted_exact")),
            expert_predictions_path=str(row.get("expert_predictions_path", "")),
            discrepancy_predictions_path=str(row.get("discrepancy_predictions_path", "")),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Layout helper
# ──────────────────────────────────────────────────────────────────────────────

def _base_layout(title: str, subtitle: str = "") -> dict[str, Any]:
    emphasized_title = f"<b>{title}</b>"
    full_title = (
        f"{emphasized_title}<br><span style='font-size:17px;font-weight:600;"
        f"color:{FONT_COLOR_MU};'>{subtitle}</span>"
        if subtitle else emphasized_title
    )
    return dict(
        title=dict(
            text=full_title,
            font=dict(size=24, color=FONT_COLOR_FG, family=FONT_FAMILY),
            x=0.5, xanchor="center",
        ),
        paper_bgcolor=BG_PAPER,
        plot_bgcolor=BG_PLOT,
        font=dict(family=FONT_FAMILY, color=FONT_COLOR_FG, size=18),
        margin=dict(l=90, r=70, t=135, b=145),
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.16,
            yanchor="top",
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor=GRID_COLOR,
            borderwidth=1,
            font=dict(size=16),
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Visualiser
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelRegistryVisualizer:
    """
    Production-grade Plotly visualisation suite for model_bundle_registry.csv.

    Construction
    ------------
    viz = ModelRegistryVisualizer.from_csv("model_bundle_registry.csv")
    viz = ModelRegistryVisualizer.from_dataframe(df)

    Display
    -------
    viz.display_summary()
    viz.display_record("supervised_pca_logistic_regression")

    Figures (individual)
    --------------------
    fig = viz.fig_cv_vs_expert_scatter()
    fig.show()

    Batch export
    ------------
    viz.render_all(output_dir="./figures", formats=("png", "pdf", "html", "json"))
    """

    records:  list[ModelRecord] = field(default_factory=list)
    df:       pd.DataFrame      = field(default_factory=pd.DataFrame)
    csv_path: str               = ""

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "ModelRegistryVisualizer":
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Registry CSV not found: {path}")
        df = pd.read_csv(path)
        records = [ModelRecord.from_dict(row) for row in df.to_dict(orient="records")]
        inst = cls(records=records, df=df, csv_path=str(path))
        inst._enrich_df()
        return inst

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame,
                       csv_path: str = "<dataframe>") -> "ModelRegistryVisualizer":
        records = [ModelRecord.from_dict(row) for row in df.to_dict(orient="records")]
        inst = cls(records=records, df=df.copy(), csv_path=csv_path)
        inst._enrich_df()
        return inst

    def _enrich_df(self) -> None:
        df = self.df

        def _bool_col(col: str) -> None:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: v if isinstance(v, bool)
                    else str(v).strip().lower() in {"true", "yes", "1"}
                )

        _bool_col("selected_for_upload")
        _bool_col("selected_for_mode")

        numeric_cols = [
            "cv_mean_accuracy", "cv_mean_precision", "cv_mean_recall", "cv_mean_f1",
            "expert_accuracy", "expert_precision_weighted", "expert_recall_weighted",
            "expert_f1_weighted", "expert_accuracy_exact", "expert_f1_weighted_exact",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "reduction_key" in df.columns:
            df["dr_display"] = df["reduction_key"].map(DR_DISPLAY).fillna(
                df["reduction_key"].str.upper()
            )

        if "cv_mean_f1" in df.columns and "expert_f1_weighted" in df.columns:
            df["cv_expert_f1_gap"] = df["cv_mean_f1"] - df["expert_f1_weighted"]

        if "classifier_name" in df.columns and "dr_display" in df.columns:
            df["short_label"] = df["dr_display"] + " · " + df["classifier_name"]

    def _registry_dr_keys(self) -> list[str]:
        """Return only dimensionality-reduction keys present in the registry."""
        if "reduction_key" not in self.df.columns:
            return [key for key in DR_ORDER if key != "tsne"]
        available = {
            str(value).strip().lower()
            for value in self.df["reduction_key"].dropna().tolist()
            if str(value).strip()
        }
        ordered = [key for key in DR_ORDER if key in available]
        return ordered or [key for key in DR_ORDER if key != "tsne"]

    # ── Console display ───────────────────────────────────────────────────────

    def display_summary(self) -> None:
        """Print structured summary of the full registry to stdout."""
        df  = self.df
        sep = "─" * 80

        print(f"\n{sep}")
        print(f"  MODEL BUNDLE REGISTRY  —  {self.csv_path}")
        print(sep)
        print(f"  Total models       : {len(self.records)}")
        print(f"  Training modes     : {', '.join(sorted(df['training_mode'].unique()))}")
        print(f"  Reduction methods  : {', '.join(sorted(df['reduction_key'].unique()))}")
        print(f"  Classifier types   : {', '.join(sorted(df['classifier_name'].unique()))}")
        print(f"  Expert row count   : {int(df['expert_row_count'].iloc[0])}")
        print(sep)

        upload_sel = [r for r in self.records if r.selected_for_upload]
        mode_sel   = [r for r in self.records if r.selected_for_mode]

        if upload_sel:
            r = upload_sel[0]
            print(f"\n  ★  SELECTED FOR UPLOAD  →  {r.artifact_id}")
            print(f"     Expert F1 : {r.expert_f1_weighted:.4f}  |  "
                  f"CV F1 : {r.cv_mean_f1:.4f}  |  "
                  f"Expert Acc : {r.expert_accuracy:.4f}")

        if mode_sel:
            print(f"\n  ✓  SELECTED FOR MODE ({len(mode_sel)}):")
            for r in mode_sel:
                print(f"     {r.artifact_id:<60}  Expert F1: {r.expert_f1_weighted:.4f}")

        top5 = sorted(self.records, key=lambda r: r.expert_f1_weighted, reverse=True)[:5]
        print(f"\n{sep}")
        print("  TOP 5 BY EXPERT F1")
        hdr = f"  {'Rank':<5} {'Artifact ID':<55} {'Mode':<16} {'DR':<8} {'Expert F1':>10} {'CV F1':>8} {'Gap':>8}"
        print(hdr)
        print(f"  {'-'*5} {'-'*55} {'-'*16} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
        for i, r in enumerate(top5, 1):
            gap = r.cv_mean_f1 - r.expert_f1_weighted
            print(f"  {i:<5} {r.artifact_id:<55} {r.training_mode:<16} "
                  f"{r.reduction_key:<8} {r.expert_f1_weighted:>10.4f} "
                  f"{r.cv_mean_f1:>8.4f} {gap:>+8.4f}")

        metrics = [
            "expert_f1_weighted", "expert_accuracy",
            "cv_mean_f1", "expert_accuracy_exact",
        ]
        print(f"\n{sep}")
        print("  METRIC SUMMARY  (all models)")
        print(f"  {'Metric':<32} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'-'*32} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for m in metrics:
            if m in df.columns:
                s = df[m].dropna()
                print(f"  {m:<32} {s.mean():>8.4f} {s.median():>8.4f} "
                      f"{s.min():>8.4f} {s.max():>8.4f}")

        print(f"\n{sep}")
        print("  BREAKDOWN BY TRAINING MODE")
        print(f"  {'Mode':<18} {'Count':>6} {'Avg Expert F1':>14} {'Avg CV F1':>10}")
        print(f"  {'-'*18} {'-'*6} {'-'*14} {'-'*10}")
        for mode, grp in df.groupby("training_mode"):
            print(f"  {mode:<18} {len(grp):>6} "
                  f"{grp['expert_f1_weighted'].mean():>14.4f} "
                  f"{grp['cv_mean_f1'].mean():>10.4f}")

        print(f"\n{sep}\n")

    def display_record(self, artifact_id: str) -> None:
        """Pretty-print every field for one model record."""
        matches = [r for r in self.records if r.artifact_id == artifact_id]
        if not matches:
            print(f"No record found for artifact_id='{artifact_id}'")
            return
        r   = matches[0]
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  {r.artifact_id}")
        print(sep)
        rows = [
            ("Training mode",     r.training_mode),
            ("Reduction",         r.reduction_key),
            ("Classifier",        r.classifier_name),
            ("Selected upload",   r.selected_for_upload),
            ("Selected mode",     r.selected_for_mode),
            ("CV Accuracy",       f"{r.cv_mean_accuracy:.4f}"),
            ("CV Precision",      f"{r.cv_mean_precision:.4f}"),
            ("CV Recall",         f"{r.cv_mean_recall:.4f}"),
            ("CV F1",             f"{r.cv_mean_f1:.4f}"),
            ("Expert rows",       r.expert_row_count),
            ("Expert Accuracy",   f"{r.expert_accuracy:.4f}"),
            ("Expert Precision",  f"{r.expert_precision_weighted:.4f}"),
            ("Expert Recall",     f"{r.expert_recall_weighted:.4f}"),
            ("Expert F1",         f"{r.expert_f1_weighted:.4f}"),
            ("Exact Accuracy",    f"{r.expert_accuracy_exact:.4f}"),
            ("Exact F1",          f"{r.expert_f1_weighted_exact:.4f}"),
            ("CV→Expert Gap",     f"{r.cv_expert_f1_gap:+.4f}"),
            ("Features",          ", ".join(r.feature_columns)),
        ]
        for label, value in rows:
            print(f"  {label:<22}: {value}")
        print(sep + "\n")

    # ── Figure 1 ──────────────────────────────────────────────────────────────

    def fig_cv_vs_expert_scatter(self) -> go.Figure:
        """CV F1 vs Expert F1 scatter — identity line + selected model annotation."""
        df = self.df.copy()
        df["marker_size"] = df["expert_accuracy_exact"].fillna(0) * 40 + 8
        df["hover"] = (
            df["artifact_id"] + "<br>"
            + "CV F1: "     + df["cv_mean_f1"].round(3).astype(str) + "<br>"
            + "Expert F1: " + df["expert_f1_weighted"].round(3).astype(str) + "<br>"
            + "Mode: "      + df["training_mode"]
        )
        fig = go.Figure()
        axis_min, axis_max = 0.70, 1.00

        fig.add_trace(go.Scatter(
            x=[axis_min, axis_max], y=[axis_min, axis_max],
            mode="lines",
            line=dict(color=GRID_COLOR, width=1.5, dash="dot"),
            name="Perfect transfer (CV = Expert)",
        ))

        for mode, colour in TRAINING_MODE_PALETTE.items():
            sub = df[df["training_mode"] == mode]
            fig.add_trace(go.Scatter(
                x=sub["cv_mean_f1"], y=sub["expert_f1_weighted"],
                mode="markers",
                marker=dict(color=colour, size=sub["marker_size"], opacity=0.82,
                            line=dict(color="white", width=0.8)),
                name=mode.replace("_", "-").title(),
                text=sub["hover"],
                hovertemplate="%{text}<extra></extra>",
            ))

        upload = df[df["selected_for_upload"] == True]
        if not upload.empty:
            r = upload.iloc[0]
            fig.add_annotation(
                x=r["cv_mean_f1"], y=r["expert_f1_weighted"],
                text=f"  ★ {r['artifact_id'].replace('_', ' ')}",
                showarrow=True, arrowhead=2,
                arrowcolor=TRAINING_MODE_PALETTE["supervised"],
                font=dict(size=14, color=TRAINING_MODE_PALETTE["supervised"]),
                xanchor="left",
            )

        fig.update_layout(**_base_layout(
            "CV F1 vs Expert F1 — All Models",
            "Dot size ∝ exact accuracy  ·  Identity line = perfect CV→expert transfer",
        ))
        fig.update_xaxes(title_text="Cross-Validation F1", range=[axis_min, axis_max],
                         tickformat=".2f", showgrid=True, gridcolor=GRID_COLOR)
        fig.update_yaxes(title_text="Expert F1 (Weighted)", range=[0.70, 0.94],
                         tickformat=".2f", showgrid=True, gridcolor=GRID_COLOR)
        return fig

    # ── Figure 2 ──────────────────────────────────────────────────────────────

    def fig_top_n_ranked_bar(self, n: int = 15) -> go.Figure:
        """Horizontal ranked bar chart — top-N by Expert F1 with CV F1 overlay."""
        df  = self.df.copy()
        top = df.nlargest(n, "expert_f1_weighted").sort_values("expert_f1_weighted")
        colours = [TRAINING_MODE_PALETTE.get(m, "#888") for m in top["training_mode"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top["short_label"], x=top["expert_f1_weighted"],
            orientation="h", name="Expert F1",
            marker=dict(color=colours, opacity=0.88,
                        line=dict(color="white", width=0.6)),
            hovertemplate="<b>%{y}</b><br>Expert F1: %{x:.4f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            y=top["short_label"], x=top["cv_mean_f1"],
            mode="markers", name="CV F1",
            marker=dict(symbol="diamond", size=9, color=FONT_COLOR_FG, opacity=0.7,
                        line=dict(color="white", width=0.8)),
            hovertemplate="<b>%{y}</b><br>CV F1: %{x:.4f}<extra></extra>",
        ))

        upload = top[top["selected_for_upload"] == True]
        if not upload.empty:
            r = upload.iloc[0]
            fig.add_annotation(
                x=r["expert_f1_weighted"] + 0.003, y=r["short_label"],
                text="★ selected", showarrow=False,
                font=dict(size=14, color=TRAINING_MODE_PALETTE["supervised"]),
                xanchor="left",
            )

        for mode, colour in TRAINING_MODE_PALETTE.items():
            fig.add_trace(go.Bar(
                y=[None], x=[None], orientation="h",
                marker=dict(color=colour),
                name=mode.replace("_", "-").title(), showlegend=True,
            ))

        fig.update_layout(**_base_layout(
            f"Top {n} Models by Expert F1",
            "Bars = Expert F1 (colour = training mode)  ·  ◆ = CV F1",
        ))
        fig.update_xaxes(title_text="F1 Score", range=[0.78, 0.96],
                         tickformat=".2f", showgrid=True, gridcolor=GRID_COLOR)
        fig.update_yaxes(title_text="", showgrid=False, automargin=True)
        fig.update_layout(height=max(FIGURE_HEIGHT, n * 44 + 160))
        return fig

    # ── Figure 3 ──────────────────────────────────────────────────────────────

    def fig_grouped_bar_classifier_mode(self) -> go.Figure:
        """Grouped bar — avg Expert F1 per classifier × training mode."""
        df  = self.df.copy()
        agg = (
            df.groupby(["classifier_name", "training_mode"])["expert_f1_weighted"]
            .mean().reset_index()
            .rename(columns={"expert_f1_weighted": "mean_expert_f1"})
        )
        classifiers = (
            agg.groupby("classifier_name")["mean_expert_f1"]
            .mean().sort_values(ascending=True).index.tolist()
        )

        fig = go.Figure()
        for mode, colour in TRAINING_MODE_PALETTE.items():
            sub = (
                agg[agg["training_mode"] == mode]
                .set_index("classifier_name")
                .reindex(classifiers)
            )
            fig.add_trace(go.Bar(
                y=classifiers, x=sub["mean_expert_f1"],
                orientation="h",
                name=mode.replace("_", "-").title(),
                marker=dict(color=colour, opacity=0.85,
                            line=dict(color="white", width=0.6)),
                hovertemplate="<b>%{y}</b><br>" + mode + ": %{x:.4f}<extra></extra>",
            ))

        fig.update_layout(
            **_base_layout(
                "Average Expert F1 by Classifier and Training Mode",
                "Averaged across all dimensionality reduction methods",
            ),
            barmode="group",
        )
        fig.update_xaxes(title_text="Mean Expert F1 (Weighted)", range=[0.72, 0.94],
                         tickformat=".2f", showgrid=True, gridcolor=GRID_COLOR)
        fig.update_yaxes(title_text="", automargin=True)
        return fig

    # ── Figure 4 ──────────────────────────────────────────────────────────────

    def fig_heatmap_classifier_dr(self) -> go.Figure:
        """Annotated heatmap — Expert F1 keyed by classifier × DR method."""
        df    = self.df.copy()
        dr_keys = self._registry_dr_keys()
        pivot = (
            df.groupby(["classifier_name", "reduction_key"])["expert_f1_weighted"]
            .mean().reset_index()
            .pivot(index="classifier_name", columns="reduction_key",
                   values="expert_f1_weighted")
        )
        pivot     = pivot.reindex(columns=dr_keys)
        row_order = pivot.mean(axis=1).sort_values(ascending=False).index.tolist()
        pivot     = pivot.loc[row_order]

        annotations = []
        for i, row_label in enumerate(pivot.index):
            for j, col_label in enumerate(pivot.columns):
                val = pivot.loc[row_label, col_label]
                annotations.append(dict(
                    x=j, y=i, text="n/a" if pd.isna(val) else f"{val:.3f}", showarrow=False,
                    font=dict(
                        size=15,
                        color=FONT_COLOR_MU if pd.isna(val) else ("white" if val < 0.84 else FONT_COLOR_FG),
                    ),
                    xref="x", yref="y",
                ))

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[DR_DISPLAY.get(c, c) for c in pivot.columns],
            y=pivot.index.tolist(),
            colorscale="Blues", zmin=0.74, zmax=0.92,
            colorbar=dict(title="Expert F1", tickformat=".3f", len=0.85),
            hovertemplate="<b>%{y}</b><br>%{x}<br>Expert F1: %{z:.4f}<extra></extra>",
        ))
        fig.update_layout(
            **_base_layout(
                "Expert F1 Heatmap — Classifier × Dimensionality Reduction",
                "Cell values = mean Expert F1 (weighted)  ·  Sorted by row average",
            ),
            annotations=annotations,
        )
        fig.update_xaxes(title_text="Dimensionality Reduction", side="bottom")
        fig.update_yaxes(title_text="Classifier", automargin=True)
        return fig

    # ── Figure 5 ──────────────────────────────────────────────────────────────

    def fig_cv_expert_gap_boxplot(self) -> go.Figure:
        """Boxplot — CV-to-Expert F1 transfer gap by DR method."""
        df  = self.df.copy()
        fig = go.Figure()
        dr_keys = self._registry_dr_keys()

        for dr_key in dr_keys:
            colour = DR_PALETTE[dr_key]
            sub = df[df["reduction_key"] == dr_key]["cv_expert_f1_gap"].dropna()
            fig.add_trace(go.Box(
                y=sub, name=DR_DISPLAY.get(dr_key, dr_key),
                marker_color=colour, boxmean="sd",
                line=dict(width=1.8), width=0.45,
                hovertemplate=(
                    DR_DISPLAY.get(dr_key, dr_key)
                    + "<br>Gap: %{y:.4f}<extra></extra>"
                ),
            ))

        fig.add_hline(
            y=0, line=dict(color="#a12c7b", width=1.4, dash="dash"),
            annotation_text="CV = Expert", annotation_position="top right",
        )
        fig.update_layout(**_base_layout(
            "CV-to-Expert F1 Transfer Gap by Dimensionality Reduction",
            "Gap = CV F1 − Expert F1  ·  Positive = CV optimistic  ·  Box shows SD",
        ))
        fig.update_yaxes(title_text="CV F1 − Expert F1", tickformat="+.3f",
                         showgrid=True, gridcolor=GRID_COLOR)
        fig.update_xaxes(title_text="Dimensionality Reduction", showgrid=False)
        return fig

    # ── Figure 6 ──────────────────────────────────────────────────────────────

    def fig_bubble_cv_expert(self) -> go.Figure:
        """Bubble chart — CV F1 × Expert F1, size = exact accuracy, colour = classifier."""
        df = self.df.copy()
        df["bubble_size"] = (
            df["expert_accuracy_exact"].fillna(0) * 60 + 8
        ).clip(lower=8, upper=50)

        fig = go.Figure()
        for clf, colour in CLASSIFIER_PALETTE.items():
            sub = df[df["classifier_name"] == clf]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["cv_mean_f1"], y=sub["expert_f1_weighted"],
                mode="markers", name=clf,
                marker=dict(size=sub["bubble_size"], color=colour, opacity=0.78,
                            line=dict(color="white", width=0.8)),
                text=(sub["artifact_id"] + "<br>Exact Acc: "
                      + sub["expert_accuracy_exact"].round(3).astype(str)),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "CV F1: %{x:.4f}<br>"
                    "Expert F1: %{y:.4f}<extra></extra>"
                ),
            ))

        fig.add_trace(go.Scatter(
            x=[0.70, 1.00], y=[0.70, 1.00],
            mode="lines", line=dict(color=GRID_COLOR, width=1.2, dash="dot"),
            name="Identity", showlegend=True,
        ))
        fig.update_layout(**_base_layout(
            "CV F1 × Expert F1 Bubble Chart — Classifier Families",
            "Bubble size ∝ exact accuracy  ·  Colour = classifier  ·  ---- = identity",
        ))
        fig.update_xaxes(title_text="Cross-Validation F1", range=[0.70, 1.00],
                         tickformat=".2f", showgrid=True, gridcolor=GRID_COLOR)
        fig.update_yaxes(title_text="Expert F1 (Weighted)", range=[0.70, 0.96],
                         tickformat=".2f", showgrid=True, gridcolor=GRID_COLOR)
        return fig

    # ── Figure 7 ──────────────────────────────────────────────────────────────

    def fig_parallel_coordinates(self) -> go.Figure:
        """Parallel coordinates — four metrics, Viridis colour = Expert F1."""
        df = self.df.copy()
        df["mode_idx"] = (
            df["training_mode"]
            .map({"supervised": 0, "semi_supervised": 1})
            .fillna(0)
        )
        metrics = [
            ("cv_mean_f1",            "CV F1"),
            ("expert_accuracy",       "Expert Acc"),
            ("expert_f1_weighted",    "Expert F1"),
            ("expert_accuracy_exact", "Exact Acc"),
        ]
        dimensions = [
            dict(range=[0.70, 1.00], label=label,
                 values=df[col].fillna(0).tolist())
            for col, label in metrics
        ]
        dimensions.insert(0, dict(
            range=[0, 1], label="Mode",
            values=df["mode_idx"].tolist(),
            tickvals=[0, 1],
            ticktext=["supervised", "semi-supervised"],
        ))

        fig = go.Figure(go.Parcoords(
            line=dict(
                color=df["expert_f1_weighted"].fillna(0),
                colorscale="Viridis", showscale=True,
                cmin=df["expert_f1_weighted"].min(),
                cmax=df["expert_f1_weighted"].max(),
                colorbar=dict(title="Expert F1", tickformat=".3f", len=0.75),
            ),
            dimensions=dimensions,
            labelside="bottom",
            labelangle=0,
        ))
        fig.update_layout(**_base_layout(
            "Parallel Coordinates — CV & Expert Metrics",
            "Line colour = Expert F1 (Viridis)  ·  Drag axes to filter",
        ))
        return fig

    # ── Figure 8 ──────────────────────────────────────────────────────────────

    def fig_dr_subplot_comparison(self) -> go.Figure:
        """1×3 subplot panel — Expert F1 per DR method, colour = classifier."""
        df      = self.df.copy()
        dr_keys = self._registry_dr_keys()

        fig = make_subplots(
            rows=1, cols=len(dr_keys),
            subplot_titles=[DR_DISPLAY.get(k, k) for k in dr_keys],
            shared_yaxes=True,
        )

        for col_idx, dr_key in enumerate(dr_keys, start=1):
            sub     = df[df["reduction_key"] == dr_key].sort_values(
                "expert_f1_weighted", ascending=True
            )
            if sub.empty:
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    xref=f"x{col_idx} domain",
                    yref=f"y{col_idx} domain",
                    text="No production<br>registry bundle",
                    showarrow=False,
                    font=dict(size=15, color=FONT_COLOR_MU),
                    align="center",
                )
                continue
            colours = [CLASSIFIER_PALETTE.get(c, "#888") for c in sub["classifier_name"]]
            fig.add_trace(
                go.Bar(
                    x=sub["expert_f1_weighted"],
                    y=sub["short_label"],
                    orientation="h",
                    marker=dict(color=colours, opacity=0.84,
                                line=dict(color="white", width=0.5)),
                    hovertemplate="<b>%{y}</b><br>Expert F1: %{x:.4f}<extra></extra>",
                    showlegend=(col_idx == len(dr_keys)),
                    name="",
                ),
                row=1, col=col_idx,
            )

        for clf, colour in CLASSIFIER_PALETTE.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, color=colour),
                name=clf, showlegend=True,
            ))

        max_rows = max(max(len(df[df["reduction_key"] == k]) for k in dr_keys), 1)
        fig.update_layout(
            **_base_layout(
                "Expert F1 by DR Method — Side-by-Side Panel",
                "Shared y-axis  ·  Colour = classifier  ·  Sorted ascending within panel",
            ),
            height=max(FIGURE_HEIGHT, 70 * max_rows + 200),
        )
        fig.update_xaxes(tickformat=".2f", range=[0.72, 0.96],
                         showgrid=True, gridcolor=GRID_COLOR)
        fig.update_yaxes(showgrid=False, automargin=True)
        return fig

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_figure(
        self,
        fig: go.Figure,
        stem: str,
        output_dir: Path,
        formats: tuple[str, ...] = ("png", "pdf", "html", "json"),
    ) -> dict[str, str]:
        paths: dict[str, str] = {}
        for fmt in formats:
            dest = output_dir / f"{stem}.{fmt}"
            if fmt in ("png", "pdf"):
                fig.write_image(str(dest), scale=2 if fmt == "png" else 1)
            elif fmt == "html":
                fig.write_html(
                    str(dest),
                    include_plotlyjs="cdn",
                    include_mathjax=False,
                    full_html=True,
                )
                html_text = dest.read_text(encoding="utf-8")
                html_text = html_text.replace(
                    "<script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>",
                    "",
                )
                dest.write_text(html_text, encoding="utf-8")
            elif fmt == "json":
                dest.write_text(fig.to_json())
            paths[fmt] = str(dest)
        return paths

    def render_all(
        self,
        output_dir: str | Path = "/tmp/metamp_publication_figures",
        formats: tuple[str, ...] = ("png", "pdf", "html", "json"),
        top_n: int = 15,
    ) -> dict[str, Any]:
        """Render all 8 figures and write to output_dir. Returns manifest dict."""
        out = Path(output_dir)
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        builders: list[tuple[str, Any]] = [
            ("fig1_cv_vs_expert_scatter",        self.fig_cv_vs_expert_scatter),
            ("fig2_top_n_ranked_bar",            lambda: self.fig_top_n_ranked_bar(n=top_n)),
            ("fig3_grouped_bar_classifier_mode", self.fig_grouped_bar_classifier_mode),
            ("fig4_heatmap_classifier_dr",       self.fig_heatmap_classifier_dr),
            ("fig5_cv_expert_gap_boxplot",       self.fig_cv_expert_gap_boxplot),
            ("fig6_bubble_cv_expert",            self.fig_bubble_cv_expert),
            ("fig7_parallel_coordinates",        self.fig_parallel_coordinates),
            ("fig8_dr_subplot_comparison",       self.fig_dr_subplot_comparison),
        ]

        manifest: dict[str, Any] = {"out_dir": str(out), "figures": {}}
        for stem, builder in builders:
            print(f"  → rendering {stem} …", flush=True)
            fig   = builder()
            paths = self._export_figure(fig, stem, out, formats=formats)
            manifest["figures"][stem] = paths

        manifest_path = out / "figure_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"\n  ✓ {len(builders)} figures written to {out}")
        print(f"  ✓ Manifest : {manifest_path}\n")
        return manifest


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Production Plotly visualiser for model_bundle_registry.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python model_registry_visualizer.py \\
                  --csv /var/app/data/models/production_ml/tables/model_bundle_registry.csv

              python model_registry_visualizer.py \\
                  --csv ./model_bundle_registry.csv \\
                  --out ./figures --formats png html

              python model_registry_visualizer.py \\
                  --csv ./model_bundle_registry.csv \\
                  --artifact-id supervised_pca_logistic_regression
        """),
    )
    parser.add_argument(
        "--csv",
        default="/var/app/data/models/production_ml/tables/model_bundle_registry.csv",
        help="Path to model_bundle_registry.csv",
    )
    parser.add_argument(
        "--out", default="/tmp/metamp_publication_figures",
        help="Output directory for rendered figures",
    )
    parser.add_argument(
        "--formats", nargs="+", default=["png", "pdf", "html", "json"],
        choices=["png", "pdf", "html", "json"],
        help="Export formats (space-separated)",
    )
    parser.add_argument(
        "--top-n", type=int, default=15,
        help="Number of models in the ranked bar chart (fig 2)",
    )
    parser.add_argument(
        "--artifact-id", default=None,
        help="Print detailed record for one artifact_id then exit",
    )
    parser.add_argument(
        "--no-summary", action="store_true",
        help="Skip the console summary table",
    )
    args = parser.parse_args()

    viz = ModelRegistryVisualizer.from_csv(args.csv)

    if not args.no_summary:
        viz.display_summary()

    if args.artifact_id:
        viz.display_record(args.artifact_id)

    manifest = viz.render_all(
        output_dir=args.out,
        formats=tuple(args.formats),
        top_n=args.top_n,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    _cli()
