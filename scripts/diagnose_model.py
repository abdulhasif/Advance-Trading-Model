"""
scripts/diagnose_model.py — XGBoost Directional Bias Diagnostic
================================================================
Run from the repo root:
    .venv\Scripts\python.exe scripts/diagnose_model.py

Covers:
  D1: Target variable distribution (why you get 0 short trades)
  D2: predict_proba confidence score analysis per direction
  D3: Feature importance — identify price-level overfit
  D4: Exact cure code — sample_weights + asymmetric thresholds
"""

import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")   # headless — saves PNG instead of showing window
import matplotlib.pyplot as plt
from pathlib import Path

# ── Add project root to path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

DIVIDER = "=" * 70

# ── Load training data ────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print(f"\n{DIVIDER}")
    print("  LOADING FEATURE DATA")
    print(DIVIDER)
    frames = []
    for sector_dir in config.FEATURES_DIR.iterdir():
        if not sector_dir.is_dir():
            continue
        for pf in sorted(sector_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(pf)
                df["_sector"] = sector_dir.name
                df["_symbol"] = pf.stem
                frames.append(df)
            except Exception as e:
                print(f"  [SKIP] {pf.name}: {e}")
    if not frames:
        print("  [ERROR] No parquet files found in storage/features/")
        sys.exit(1)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("brick_timestamp").reset_index(drop=True)
    print(f"  Total bricks loaded : {len(combined):,}")
    print(f"  Symbols             : {combined['_symbol'].nunique()}")
    print(f"  Date range          : {combined['brick_timestamp'].min().date()} → "
          f"{combined['brick_timestamp'].max().date()}")
    return combined

# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    b1_path = config.BRAIN1_MODEL_PATH
    if not b1_path.exists():
        print(f"  [ERROR] Brain1 not found at {b1_path}. Run: python main.py train")
        sys.exit(1)
    b1 = xgb.XGBClassifier()
    b1.load_model(str(b1_path))
    print(f"  Brain1 loaded from  : {b1_path}")
    return b1

FEATURE_COLS = [
    "velocity", "wick_pressure", "relative_strength",
    "brick_size", "duration_seconds", "direction",
    "consecutive_same_dir", "brick_oscillation_rate",
    "fracdiff_price", "hurst", "is_trending_regime",
]

def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce brain_trainer.py target logic."""
    out = df.copy()
    out["_date"] = out["brick_timestamp"].dt.date
    out = out.sort_values(["_symbol", "brick_timestamp"]).reset_index(drop=True)
    next_dir = out.groupby(["_symbol", "_date"])["direction"].shift(-1)
    out["direction_target"] = np.where(next_dir.isna(), np.nan, (next_dir > 0).astype(float))
    return out.dropna(subset=["direction_target"]).reset_index(drop=True)

# =============================================================================
# DELIVERABLE 1: Target Variable Distribution
# =============================================================================
def deliverable_1_target_distribution(df: pd.DataFrame):
    print(f"\n{DIVIDER}")
    print("  D1: TARGET VARIABLE DISTRIBUTION")
    print(f"{DIVIDER}")

    df = make_targets(df)

    # Binary target: 1 = next brick UP (LONG setup), 0 = next brick DOWN (SHORT setup)
    counts = df["direction_target"].value_counts().sort_index()
    total  = len(df)

    print(f"\n  Binary target (direction_target):")
    print(f"  {'Label':<10} {'Meaning':<20} {'Count':>10} {'Pct':>8}")
    print(f"  {'-'*52}")

    labels = {0.0: "SHORT setup (next brick DN)", 1.0: "LONG setup  (next brick UP)"}
    for val, count in counts.items():
        label = labels.get(float(val), str(val))
        print(f"  {str(val):<10} {label:<20} {count:>10,} {count/total*100:>7.1f}%")

    imbalance = counts.get(0.0, 0) / max(counts.get(1.0, 1), 1)
    print(f"\n  Imbalance ratio (SHORT:LONG) = {imbalance:.2f}:1")

    # Diagnosis
    print(f"\n  DIAGNOSIS:")
    if imbalance > 1.5:
        print(f"  ⚠ LONG-biased training data ({imbalance:.1f}x more LONG labels).")
        print(f"    The model learned to predict UP to minimize training loss.")
        print(f"    This is the PRIMARY cause of 0 short trades.")
    elif imbalance < 0.67:
        print(f"  ⚠ SHORT-biased training data. Check Renko label construction.")
    else:
        print(f"  ✓ Data is roughly balanced ({imbalance:.2f}:1). Problem is likely")
        print(f"    in the execution threshold — proceed to D2.")

    # Save distribution chart
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["SHORT (0)", "LONG (1)"],
           [counts.get(0.0, 0), counts.get(1.0, 0)],
           color=["#e74c3c", "#2ecc71"], edgecolor="white")
    ax.set_title("Training Target Distribution\n(D1: Are LONG and SHORT setups balanced?)")
    ax.set_ylabel("Count"); ax.set_xlabel("Direction")
    for i, v in enumerate([counts.get(0.0, 0), counts.get(1.0, 0)]):
        ax.text(i, v + total * 0.005, f"{v:,}\n({v/total*100:.1f}%)", ha="center", fontsize=9)
    out = config.LOGS_DIR / "diag_d1_target_dist.png"
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"\n  Chart saved → {out}")

# =============================================================================
# DELIVERABLE 2: predict_proba Confidence Analyzer
# =============================================================================
def deliverable_2_proba_analysis(df: pd.DataFrame, model: xgb.XGBClassifier):
    print(f"\n{DIVIDER}")
    print("  D2: predict_proba CONFIDENCE SCORE ANALYSIS")
    print(f"{DIVIDER}")

    df = make_targets(df)
    # Use test split (data after TEST_START_DATE)
    cutoff = pd.Timestamp(config.TEST_START_DATE, tz="Asia/Kolkata")
    test   = df[df["brick_timestamp"] >= cutoff].reset_index(drop=True)
    if len(test) < 100:
        print(f"  [WARN] Only {len(test)} test rows after {config.TEST_START_DATE}. Using full dataset.")
        test = df

    # Drop rows missing any feature
    test_clean = test.dropna(subset=FEATURE_COLS)
    X = test_clean[FEATURE_COLS].fillna(0)

    proba = model.predict_proba(X)          # shape: (N, 2)
    p_up   = proba[:, 1]                   # P(next brick UP)  → LONG signal
    p_down = 1 - p_up                      # P(next brick DOWN) → SHORT signal

    # Classify what signal would fire at common thresholds
    results = {}
    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75]:
        long_signals  = (p_up   >= thresh).sum()
        short_signals = (p_down >= thresh).sum()
        results[thresh] = (long_signals, short_signals)

    print(f"\n  Test set size: {len(test_clean):,} bricks  "
          f"(cutoff: {config.TEST_START_DATE})")
    print(f"\n  Raw probability stats:")
    print(f"  {'Metric':<35} {'LONG (p_up)':>12} {'SHORT (1-p_up)':>14}")
    print(f"  {'-'*63}")
    print(f"  {'Max confidence':<35} {p_up.max():>12.4f} {p_down.max():>14.4f}")
    print(f"  {'Mean confidence':<35} {p_up.mean():>12.4f} {p_down.mean():>14.4f}")
    print(f"  {'Median confidence':<35} {np.median(p_up):>12.4f} {np.median(p_down):>14.4f}")
    print(f"  {'% of bricks where signal > 0.50':<35} "
          f"{(p_up>0.5).sum()/len(p_up)*100:>10.1f}% {(p_down>0.5).sum()/len(p_down)*100:>12.1f}%")

    print(f"\n  Signal count by threshold:")
    print(f"  {'Threshold':<12} {'LONG signals':>14} {'SHORT signals':>14} {'SHORT%':>8}")
    print(f"  {'-'*52}")
    for thresh, (ls, ss) in results.items():
        total_sigs = ls + ss
        short_pct = ss / max(total_sigs, 1) * 100
        marker = " ← your current setting" if thresh == 0.70 else ""
        print(f"  {thresh:<12.2f} {ls:>14,} {ss:>14,} {short_pct:>7.1f}%{marker}")

    # Probability histogram
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(p_up,   bins=50, alpha=0.7, label="P(LONG)",  color="#2ecc71")
    ax.hist(p_down, bins=50, alpha=0.7, label="P(SHORT)", color="#e74c3c")
    ax.axvline(0.70, color="gold",  ls="--", lw=1.5, label="threshold=0.70")
    ax.axvline(0.50, color="white", ls=":",  lw=1.0, label="decision boundary")
    ax.set_title("D2: Probability Distribution — LONG vs SHORT\n"
                 "If SHORT bars are mostly left of the threshold → threshold trap")
    ax.set_xlabel("Confidence score"); ax.set_ylabel("Brick count")
    ax.legend(facecolor="#1a1a2e")
    out = config.LOGS_DIR / "diag_d2_proba_distribution.png"
    plt.tight_layout(); plt.savefig(out, dpi=120, facecolor="#1a1a2e"); plt.close()
    print(f"\n  Histogram saved → {out}")

    # Key diagnostic
    short_max = p_down.max()
    print(f"\n  DIAGNOSIS:")
    if short_max < 0.70:
        print(f"  ⚠ THRESHOLD TRAP: Model's max SHORT confidence = {short_max:.4f}")
        print(f"    It NEVER breaches your 0.70 threshold → zero short trades.")
        print(f"    → Solution: Apply asymmetric threshold (D4, Fix 2)")
    else:
        print(f"  ✓ Model CAN produce SHORT confidence > 0.70 (max={short_max:.4f})")
        print(f"    But SHORT count at 0.70 threshold = {results[0.70][1]}.")
        print(f"    → Problem is in gate logic or feature bias. Check D3.")

# =============================================================================
# DELIVERABLE 3: Feature Importance / Overfit Audit
# =============================================================================
def deliverable_3_feature_importance(model: xgb.XGBClassifier):
    print(f"\n{DIVIDER}")
    print("  D3: FEATURE IMPORTANCE — OVERFIT AUDIT")
    print(f"{DIVIDER}")

    booster = model.get_booster()
    importance = booster.get_score(importance_type="gain")
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    top10 = list(importance.items())[:10]
    total_gain = sum(importance.values())

    print(f"\n  Top 10 features by Split Gain (higher = more decisive splits):")
    print(f"\n  {'Rank':<6} {'Feature':<30} {'Gain':>10} {'% of Total':>12}  ⚠ Flag")
    print(f"  {'-'*68}")

    # Features that indicate price-level overfit
    PRICE_LEVEL_FEATURES = {"brick_close", "brick_open", "brick_size_abs",
                            "price", "ltp", "close", "open", "high", "low"}
    STATIONARY_FEATURES  = {"velocity", "relative_strength", "wick_pressure",
                            "fracdiff_price", "hurst", "consecutive_same_dir",
                            "brick_oscillation_rate", "is_trending_regime",
                            "duration_seconds", "direction"}

    for rank, (feat, gain) in enumerate(top10, 1):
        pct   = gain / total_gain * 100
        flag  = ""
        if feat in PRICE_LEVEL_FEATURES:
            flag = "⚠ RAW PRICE LEVEL — OVERFIT RISK"
        elif feat in STATIONARY_FEATURES:
            flag = "✓ stationary/momentum"
        print(f"  {rank:<6} {feat:<30} {gain:>10.1f} {pct:>10.1f}%  {flag}")

    # Save importance chart
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [f[0] for f in top10]
    gains = [f[1] for f in top10]
    colors = ["#e74c3c" if n in PRICE_LEVEL_FEATURES else "#2ecc71" for n in names]
    ax.barh(names[::-1], gains[::-1], color=colors[::-1])
    ax.set_title("D3: Feature Importance (Gain)\nRed = raw price → overfit risk | Green = stationary")
    ax.set_xlabel("Gain")
    out = config.LOGS_DIR / "diag_d3_feature_importance.png"
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"\n  Chart saved → {out}")

    # Diagnosis
    price_features_in_top10 = [f for f, _ in top10 if f in PRICE_LEVEL_FEATURES]
    print(f"\n  DIAGNOSIS:")
    if price_features_in_top10:
        print(f"  ⚠ RAW PRICE FEATURES in top 10: {price_features_in_top10}")
        print(f"    The model learned absolute price levels, not momentum patterns.")
        print(f"    It will fail on regime changes and be LONG-biased in uptrends.")
        print(f"    → Fix: Remove raw price from FEATURE_COLS, retrain on fracdiff_price only.")
    else:
        print(f"  ✓ No raw price features dominating. Model is using momentum indicators.")
        print(f"    Bias is likely from class imbalance — proceed to D4.")

# =============================================================================
# DELIVERABLE 4: The Cure
# =============================================================================
def deliverable_4_cure(df: pd.DataFrame):
    print(f"\n{DIVIDER}")
    print("  D4: THE CURE — EXACT CODE FOR YOUR CODEBASE")
    print(f"{DIVIDER}")

    df = make_targets(df)
    y = df["direction_target"]
    n_short = int((y == 0).sum())
    n_long  = int((y == 1).sum())
    ratio   = n_short / max(n_long, 1)

    print(f"""
  ┌─ FIX 1: scale_pos_weight in brain_trainer.py ──────────────────────┐
  │ Already implemented in run_brain_trainer()!                         │
  │ Current imbalance: SHORT={n_short:,}  LONG={n_long:,}  ratio={ratio:.3f}      │
  │ Your XGBClassifier already uses scale_pos_weight = {n_short/max(n_long,1):.2f}         │
  │ If SHORT count is still 0, the issue is the threshold, not weights. │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ FIX 2: Asymmetric Execution Thresholds (paste into paper_trader.py) ┐
  │                                                                       │
  │  BEFORE (symmetric — wrong):                                          │
  │    ENTRY_PROB_THRESH = 0.70  # applied to BOTH LONG and SHORT         │
  │                                                                       │
  │  AFTER (asymmetric — correct):                                        │
  │    LONG_PROB_THRESH  = 0.75  # Raise the bar for LONG (more frequent) │
  │    SHORT_PROB_THRESH = 0.62  # Lower the bar for SHORT (suppressed)   │
  │                                                                       │
  │  Gate 1 logic change in run_paper_trader():                           │
  │    # OLD:                                                             │
  │    entry_prob_ok = b1p >= eff_prob_thresh                             │
  │                                                                       │
  │    # NEW (asymmetric):                                                │
  │    if signal == "LONG":                                               │
  │        entry_prob_ok = b1p >= LONG_PROB_THRESH                        │
  │    else:  # SHORT                                                     │
  │        entry_prob_ok = (1 - b1p) >= SHORT_PROB_THRESH                 │
  └───────────────────────────────────────────────────────────────────────┘
""")

    # Compute optimal SHORT threshold from observed max short confidence
    print(f"  To find the right SHORT_PROB_THRESH, run D2 first and look at:")
    print(f"  • MAX short confidence → set SHORT_PROB_THRESH just below it")
    print(f"  • The 'Signal count by threshold' table → pick a thresh that gives")
    print(f"    ~20-30% short signal share (realistic for a trending market)")
    print(f"\n  ┌─ FIX 3: Retrain with separate class weights per sample ────────┐")
    print(f"  │ In brain_trainer.py → train_brain1() → m.fit():               │")
    print(f"  │                                                                │")
    print( "  │   from sklearn.utils.class_weight import compute_sample_weight │")
    print( "  │   sw = compute_sample_weight('balanced', y_tr)                 │")
    print( "  │   m.fit(X_tr, y_tr, sample_weight=sw,                         │")
    print( "  │          eval_set=[(X_va, y_va)], verbose=50)                  │")
    print(f"  │                                                                │")
    print(f"  │ This is STRONGER than scale_pos_weight — it adjusts per-row,  │")
    print(f"  │ not just globally, giving SHORT bricks same total weight.      │")
    print(f"  └────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─ DECISION TREE ────────────────────────────────────────────────┐")
    print(f"  │                                                                │")
    print(f"  │  D2 max SHORT conf < 0.65 → Apply Fix 2 (lower threshold)     │")
    print(f"  │  D1 imbalance > 1.5:1     → Apply Fix 3 + retrain (Fix 1+3)  │")
    print(f"  │  D3 raw price in top 3    → Remove price features + retrain   │")
    print(f"  │  All of the above         → Fix all 3 in order                │")
    print(f"  └────────────────────────────────────────────────────────────────┘")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print(f"\n{DIVIDER}")
    print("  XGBoost Directional Bias Diagnostic")
    print("  Institutional Fortress | NSE Intraday Renko Engine")
    print(DIVIDER)

    df    = load_data()
    model = load_model()

    deliverable_1_target_distribution(df)
    deliverable_2_proba_analysis(df, model)
    deliverable_3_feature_importance(model)
    deliverable_4_cure(df)

    print(f"\n{DIVIDER}")
    print("  DIAGNOSTIC COMPLETE")
    print(f"  Charts saved to: {config.LOGS_DIR}/diag_*.png")
    print(DIVIDER + "\n")
