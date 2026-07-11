"""
train_model.py
--------------
Trains the model, evaluates it honestly, checks whether it has real
stock-selection value, and saves everything.

Run it with:   python train_model.py

What it does, in order:
  1. Build the dataset (from data_pipeline.py)
  2. Split by TIME (never randomly — that would leak the future)
  3. Train a HistGradientBoosting classifier (fast on millions of rows)
  4. Report accuracy, ROC-AUC and a confusion matrix
  5. The important part: a decile backtest showing whether top-ranked
     stocks actually outperform — this is what a portfolio system needs
  6. Permutation importance: which features mattered
  7. Save the model, a metrics file, and a results chart
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # save plots to file without needing a screen
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix,
)

import config
from data_pipeline import prepare_dataset


def time_split(df):
    """Split into train (past) and test (future) by date. This is the single
    most important correctness step in financial machine learning."""
    cutoff = pd.Timestamp(config.TRAIN_END_DATE)
    train = df[df["Date"] < cutoff]
    test = df[df["Date"] >= cutoff]
    print(f"\nTime split at {config.TRAIN_END_DATE}")
    print(f"  Train: {len(train):,} rows  ({train['Date'].min().date()} -> {train['Date'].max().date()})")
    print(f"  Test : {len(test):,} rows  ({test['Date'].min().date()} -> {test['Date'].max().date()})")
    return train, test


def train(train_df):
    """Fit the gradient-boosting model on the training years."""
    print("\nTraining model ... (a minute or so on the full dataset)")
    model = HistGradientBoostingClassifier(**config.MODEL_PARAMS)
    model.fit(train_df[config.FEATURES], train_df["target"])
    return model


def evaluate(model, test_df):
    """Standard classification metrics — and a frank note on what they mean."""
    X = test_df[config.FEATURES]
    y = test_df["target"]
    proba = model.predict_proba(X)[:, 1]   # probability the stock beats the market
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, proba)
    cm = confusion_matrix(y, pred)
    report = classification_report(y, pred, digits=3)

    lines = []
    lines.append("=== CLASSIFICATION METRICS (out-of-sample / future data) ===")
    lines.append(f"Accuracy : {acc:.3f}")
    lines.append(f"ROC-AUC  : {auc:.3f}")
    lines.append("")
    lines.append("Reading these honestly:")
    lines.append(" - Accuracy near 0.50 is EXPECTED. Markets are close to efficient,")
    lines.append("   so predicting any single stock is nearly a coin flip.")
    lines.append(" - If you ever see 0.90+ accuracy here, you have a bug (usually")
    lines.append("   lookahead bias). High accuracy on stocks is a red flag, not a win.")
    lines.append(" - The real test is the decile backtest below, not accuracy.")
    lines.append("")
    lines.append("Confusion matrix [rows=true, cols=pred]:")
    lines.append(str(cm))
    lines.append("")
    lines.append("Full report:")
    lines.append(report)

    out = "\n".join(lines)
    print("\n" + out)
    return proba, auc, acc, cm


def portfolio_backtest(test_df, proba):
    """
    The heart of a 'smart portfolio system'.

    Sort every test-set prediction into 10 buckets (deciles) by the model's
    predicted probability. If the model is useful, the top decile (stocks it
    liked most) should earn higher actual forward returns than the bottom decile.
    The gap between them is the model's stock-selection 'edge'.
    """
    df = test_df.copy()
    df["pred_proba"] = proba
    df["decile"] = pd.qcut(df["pred_proba"], 10, labels=False, duplicates="drop")
    decile_returns = df.groupby("decile")["fwd_ret"].mean()

    top = decile_returns.iloc[-1]
    bottom = decile_returns.iloc[0]
    market = df["fwd_ret"].mean()
    spread = top - bottom

    lines = []
    lines.append("=== PORTFOLIO DECILE BACKTEST (the metric that matters) ===")
    lines.append(f"Average forward {config.HORIZON}-day return by predicted decile:")
    for d, r in decile_returns.items():
        bar = "#" * max(0, int(r * 400))
        lines.append(f"  decile {int(d):2d}: {r:+.4f}  {bar}")
    lines.append("")
    lines.append(f"Top decile (model's favourites): {top:+.4f}")
    lines.append(f"Bottom decile (model's dislikes): {bottom:+.4f}")
    lines.append(f"Market average:                   {market:+.4f}")
    lines.append(f"Long-short edge (top - bottom):   {spread:+.4f}")
    lines.append("")
    if top > market and spread > 0:
        lines.append("=> The model adds value: its top picks beat the market and beat its")
        lines.append("   bottom picks. A portfolio overweighting the top decile and avoiding")
        lines.append("   the bottom decile would have outperformed an equal-weight basket.")
    else:
        lines.append("=> No reliable edge in this run. Try a different HORIZON, more")
        lines.append("   features, or a different train cutoff.")

    out = "\n".join(lines)
    print("\n" + out)
    return decile_returns, out


def feature_importance(model, test_df):
    """Which features drove the predictions? Permutation importance shuffles
    each feature and measures how much performance drops. Computed on a sample
    so it stays fast."""
    print("\nComputing permutation importance (on a sample) ...")
    sample = test_df.sample(min(40000, len(test_df)), random_state=42)
    result = permutation_importance(
        model, sample[config.FEATURES], sample["target"],
        n_repeats=5, random_state=42, scoring="roc_auc", n_jobs=-1,
    )
    imp = (pd.Series(result.importances_mean, index=config.FEATURES)
           .sort_values(ascending=False))
    print("Feature importance (drop in ROC-AUC when shuffled):")
    for f, v in imp.items():
        print(f"  {f:13s} {v:.4f}")
    return imp


def make_plots(cm, decile_returns, importance):
    """Save a single results.png with three panels for your report."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1) confusion matrix
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Underperform", "Beat market"])
    ax.set_yticklabels(["Underperform", "Beat market"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")

    # 2) decile returns — the money chart
    ax = axes[1]
    ax.bar(decile_returns.index, decile_returns.values, color="steelblue")
    ax.axhline(decile_returns.mean(), color="red", ls="--", lw=1, label="market avg")
    ax.set_title(f"Avg {config.HORIZON}-day Return by Predicted Decile")
    ax.set_xlabel("Decile (0 = model dislikes, 9 = model likes)")
    ax.set_ylabel("Forward return")
    ax.legend()

    # 3) feature importance
    ax = axes[2]
    importance.sort_values().plot.barh(ax=ax, color="seagreen")
    ax.set_title("Feature Importance (permutation)")
    ax.set_xlabel("Drop in ROC-AUC when shuffled")

    fig.tight_layout()
    fig.savefig(config.PLOT_OUT, dpi=120)
    print(f"\nSaved chart to {config.PLOT_OUT}")


def main():
    df = prepare_dataset()
    train_df, test_df = time_split(df)

    model = train(train_df)
    proba, auc, acc, cm = evaluate(model, test_df)
    decile_returns, backtest_text = portfolio_backtest(test_df, proba)
    importance = feature_importance(model, test_df)
    make_plots(cm, decile_returns, importance)

    # Save the trained model so you can load it later for predictions
    joblib.dump({"model": model, "features": config.FEATURES,
                 "horizon": config.HORIZON}, config.MODEL_OUT)
    print(f"Saved trained model to {config.MODEL_OUT}")

    # Save a metrics report for your dissertation appendix
    with open(config.METRICS_OUT, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nROC-AUC: {auc:.4f}\n\n")
        f.write(backtest_text + "\n\n")
        f.write("Feature importance:\n")
        f.write(importance.to_string())
    print(f"Saved metrics to {config.METRICS_OUT}")
    print("\nDone. Use predict.py to score the latest data and build a portfolio.")


if __name__ == "__main__":
    main()
