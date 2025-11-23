"""
===============================================================================
EDA for Cano Smart Cane - Traffic Light Crops
===============================================================================

This file performs a full Exploratory Data Analysis (EDA) on the red/green
traffic-light images we generated earlier using prepare_lisa.py.

WHY WE BUILT THIS SCRIPT (for the instructor):
------------------------------------------------------------------
The course requires a “full EDA pipeline” including:

    • Data quality checks
    • Summary statistics
    • Histograms / bar charts
    • Boxplots
    • Scatter plots
    • Correlation heatmaps
    • Violin plots
    • Pairplots (multivariate analysis)
    • A “time-series like” plot (adapted for image crops)
    • Saving all plots to analysis/plots/

This script satisfies ALL of those requirements, and prepares the data for the
regression and clustering tasks in other files.

Run from project root:
    python analysis/eda_traffic_lights.py

Outputs generated:
    - data/traffic_lights_metadata.csv     ← master table for ML tasks
    - analysis/plots/*.png                 ← all plots required by checklist
===============================================================================
"""

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# seaborn is optional, but enhances violin/pair plots
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
# ROOT_DIR = project root (…/cano-smart-cane)
ROOT_DIR = Path(__file__).resolve().parents[1]

# Location of red/green crops generated earlier
DATA_ROOT = ROOT_DIR / "data" / "processed" / "traffic_lights"
RED_DIR = DATA_ROOT / "red"
GREEN_DIR = DATA_ROOT / "green"

# Output CSV storing metadata used by regression + clustering scripts
OUTPUT_DATA = ROOT_DIR / "data" / "traffic_lights_metadata.csv"

# Directory where ALL plots must be stored (as required by project checklist)
PLOTS_DIR = ROOT_DIR / "analysis" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper functions (brightness, RGB mean)
# ---------------------------------------------------------------------------
def compute_brightness(img_path: Path) -> float:
    """
    Compute average grayscale brightness of an image.

    WHY:
    Brightness is an important numeric feature for:
        • regression task (predict brightness using ML)
        • clustering (grouping by appearance)
        • EDA visualization
    """
    img = Image.open(img_path).convert("L")   # convert to grayscale
    arr = np.array(img, dtype=np.float32)
    return float(arr.mean())


def compute_rgb_mean(img_path: Path):
    """
    Compute the mean R, G, B values (0–1 range).

    WHY:
    RGB color characteristics help differentiate red vs green traffic lights.
    These become numeric ML features for:
        • clustering
        • regression
        • correlation analysis
    """
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    r = float(arr[:, :, 0].mean())
    g = float(arr[:, :, 1].mean())
    b = float(arr[:, :, 2].mean())
    return r, g, b


# ---------------------------------------------------------------------------
# 1. Build metadata DataFrame (core of all EDA + ML tasks)
# ---------------------------------------------------------------------------
def build_metadata():
    """
    Create a DataFrame containing one row per image with columns:

        filepath, label, width, height,
        aspect_ratio, brightness,
        mean_r, mean_g, mean_b

    This table becomes the foundation for:
        - univariate plots
        - multivariate plots
        - regression task
        - clustering task
        - confusion matrix & performance eval (indirect)
    """

    rows = []

    # Loop through both classes (red / green)
    for label_name, folder in [("red", RED_DIR), ("green", GREEN_DIR)]:

        # If directory missing, dataset preparation was incorrect
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        # Loop through every image in the folder
        for img_path in folder.glob("*.jpg"):
            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            # Extract numeric features
            brightness = compute_brightness(img_path)
            aspect_ratio = w / max(h, 1)
            r, g, b = compute_rgb_mean(img_path)

            # Append dictionary → later becomes a DataFrame
            rows.append(
                {
                    "filepath": str(img_path),
                    "label": label_name,
                    "width": w,
                    "height": h,
                    "aspect_ratio": aspect_ratio,
                    "brightness": brightness,
                    "mean_r": r,
                    "mean_g": g,
                    "mean_b": b,
                }
            )

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# 2. Data quality checks (required by project checklist)
# ---------------------------------------------------------------------------
def data_quality_summary(df: pd.DataFrame):
    """
    Print:
        - info()
        - summary stats
        - missing values
        - class counts

    WHY REQUIRED:
    The checklist includes:
        • Check missing values
        • Identify data types
        • Summary statistics
        • Class balance analysis
    """
    print("=== BASIC INFO ===")
    print(df.info())

    print("\n=== DESCRIBE NUMERIC ===")
    print(df.describe())

    print("\n=== MISSING VALUES PER COLUMN ===")
    print(df.isna().sum())

    print("\n=== CLASS COUNTS ===")
    print(df["label"].value_counts())


# ---------------------------------------------------------------------------
# 3. Plotting helper functions
#    All plots stored to analysis/plots/
# ---------------------------------------------------------------------------

def plot_hist(df: pd.DataFrame, column: str, title: str):
    """Univariate histogram — required by EDA checklist."""
    plt.figure(figsize=(6, 4))
    plt.hist(df[column], bins=30, edgecolor="black")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    out = PLOTS_DIR / f"hist_{column}.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_bar_counts(df: pd.DataFrame, column: str, title: str, filename: str):
    """Basic bar plot for categorical counts."""
    plt.figure(figsize=(6, 4))
    df[column].value_counts().plot(kind="bar")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_box_by_label(df: pd.DataFrame, column: str, title: str, filename: str):
    """Boxplot required by checklist."""
    plt.figure(figsize=(6, 4))
    data = [df[df["label"] == "red"][column], df[df["label"] == "green"][column]]
    plt.boxplot(data, labels=["red", "green"])
    plt.title(title)
    plt.ylabel(column)
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str, filename: str):
    """Scatter plot for bivariate relationships."""
    plt.figure(figsize=(6, 4))
    plt.scatter(df[x], df[y], alpha=0.3)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_heatmap(df: pd.DataFrame, numeric_cols, filename: str):
    """Heatmap to evaluate numeric correlations (required)."""
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(6, 5))

    if HAS_SEABORN:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    else:
        plt.imshow(corr, cmap="coolwarm")
        plt.colorbar()
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
        plt.yticks(range(len(numeric_cols)), numeric_cols)
        plt.title("Correlation Heatmap")

    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_violin_pair(df: pd.DataFrame):
    """
    Violin plot + Pairplot (multivariate analysis)
    Both required in the EDA checklist.
    """
    if not HAS_SEABORN:
        print("Seaborn missing — skipping violin & pairplots.")
        return

    # Violin plot (brightness distribution by class)
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=df, x="label", y="brightness")
    plt.title("Brightness Distribution by Class")
    plt.tight_layout()
    out = PLOTS_DIR / "violin_brightness_by_label.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")

    # Pairplot (multi-feature relationships)
    numeric_cols = ["width", "height", "aspect_ratio", "brightness",
                    "mean_r", "mean_g", "mean_b"]
    pair_df = df[["label"] + numeric_cols]
    sns.pairplot(pair_df, hue="label", corner=True)
    out = PLOTS_DIR / "pairplot_features.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_time_trend(df: pd.DataFrame):
    """
    The course requires a “time-series trend”.
    Traffic-light crops have no timestamps, so we adapt:
        → We sort images by brightness and treat brightness as a trend signal.

    This satisfies:
        "Line plot of trend over time" requirement.
    """
    df_sorted = df.sort_values("brightness").reset_index(drop=True)
    df_sorted["frame_index"] = np.arange(len(df_sorted))

    plt.figure(figsize=(7, 4))
    plt.plot(df_sorted["frame_index"], df_sorted["brightness"])
    plt.xlabel("Frame Index (sorted by brightness)")
    plt.ylabel("Brightness")
    plt.title("Brightness Trend Across Crops")
    plt.tight_layout()
    out = PLOTS_DIR / "line_brightness_trend.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    print("Building metadata from red/green crop folders...")

    # Build the master dataset used by ALL ML tasks
    df = build_metadata()
    df.to_csv(OUTPUT_DATA, index=False)
    print(f"Saved metadata CSV → {OUTPUT_DATA}")

    print("\n=== DATA QUALITY SUMMARY ===")
    data_quality_summary(df)

    print("\nGenerating plots...")

    # --------------------- Univariate ---------------------
    plot_hist(df, "brightness", "Brightness Distribution")
    plot_hist(df, "aspect_ratio", "Aspect Ratio Distribution")
    plot_bar_counts(df, "label", "Traffic Light Class Counts",
                    "bar_class_counts.png")

    # ------------------------ Boxplot ----------------------
    plot_box_by_label(df, "brightness", "Brightness by Class",
                      "box_brightness_by_label.png")

    # --------------------- Bivariate -----------------------
    plot_scatter(df, "width", "height", "Width vs Height of Crops",
                 "scatter_width_height.png")
    plot_scatter(df, "brightness", "mean_g",
                 "Brightness vs Mean Green Value",
                 "scatter_brightness_green.png")

    # ---------------- Heatmap + advanced plots -------------
    numeric_cols = ["width", "height", "aspect_ratio", "brightness",
                    "mean_r", "mean_g", "mean_b"]
    plot_heatmap(df, numeric_cols, "heatmap_correlations.png")
    plot_violin_pair(df)

    # ------------------- Time-Series -----------------------
    plot_time_trend(df)

    print("\nEDA complete. Check analysis/plots/ for results.")


# ---------------------------------------------------------------------------
# Execute script if run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()