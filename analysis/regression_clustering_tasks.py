"""
===============================================================================
Regression & Clustering Tasks for Cano Smart Cane (CS316 Machine Learning)
===============================================================================

This script implements the *additional required ML tasks* from the CS316
project checklist:

    1) Regression Task
        → Predict brightness (numeric variable) from image-based features.

    2) Clustering Task
        → Group traffic-light crops using unsupervised learning (KMeans)
          based on visual similarity (brightness + RGB means).

These tasks use the metadata generated in:
    analysis/eda_traffic_lights.py
which extracted:
    - width, height, aspect_ratio
    - brightness
    - mean_r, mean_g, mean_b
for every red/green traffic-light crop.

This script produces:
    • regression_results.csv
    • clustering_assignments.csv

Run from project root:

    python analysis/regression_clustering_tasks.py
===============================================================================
"""
# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import pandas as pd

# scikit-learn tools for regression, clustering, metrics, preprocessing, splits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Locate the project root directory
ROOT_DIR = Path(__file__).resolve().parents[1]

# Metadata CSV generated during EDA (contains brightness, RGB means, etc.)
METADATA_CSV = ROOT_DIR / "data" / "traffic_lights_metadata.csv"

# Save outputs (regression results + clustering assignments)
RESULTS_DIR = ROOT_DIR / "analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: Compute regression metrics
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    """
    Compute all regression metrics required for CS316:

        - MAE   (Mean Absolute Error)
        - RMSE  (Root Mean Squared Error)
        - R2    (Coefficient of Determination)
        - MAPE  (Mean Absolute Percentage Error)

    WHY THESE METRICS?
    --------------------
    The project checklist explicitly asks for:

        • MAE
        • RMSE
        • R²
        • MAPE (optional but recommended)

    These metrics provide:
        - Absolute error (MAE)
        - Penalized squared error (RMSE)
        - Goodness of fit (R²)
        - Percentage-based interpretability (MAPE)
    """

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # RMSE is not always given directly in sklearn

    r2 = r2_score(y_true, y_pred)

    # MAPE formula:
    #     mean(|y_true - y_pred| / |y_true|)
    # Add epsilon (1e-3) to avoid division by zero.
    mape = np.mean(
        np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-3))
    ) * 100

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

# ---------------------------------------------------------------------------
# REGRESSION TASK (Predict brightness)
# ---------------------------------------------------------------------------
def regression_task(df: pd.DataFrame):
    """
    Regression = Predict a numeric value.

    In our case:
        Target (y)   = brightness of each crop
        Features (X) = width, height, aspect_ratio, mean_r, mean_g, mean_b

    WHY THIS FITS THE CHECKLIST:
        ✔ Required: "Regression task predicting numerical values"
        ✔ Uses multiple numerical features extracted during EDA
        ✔ Produces metrics table saved for the final report
    """
    # These are all numeric image-derived features
    features = ["width", "height", "aspect_ratio", "mean_r", "mean_g", "mean_b"]

    # Extract X (inputs) and y (target brightness)
    X = df[features].values
    y = df["brightness"].values

    # Split into train/test, 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Standardize features → improves learning performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Two regression models:
    models = {
        "Linear Regression": LinearRegression(),  # simple baseline model
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,  # use all CPU cores
        ),
    }
    rows = []
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)

        # Predict on test set
        y_pred = model.predict(X_test_scaled)

        # Compute metrics
        mets = regression_metrics(y_test, y_pred)

        rows.append({"Model": name, **mets})

    # Convert metrics to DataFrame
    results_df = pd.DataFrame(rows)

    # Save metrics to CSV
    out_csv = RESULTS_DIR / "regression_results.csv"
    results_df.to_csv(out_csv, index=False)

    print("\n=== Regression Results (brightness prediction) ===")
    print(results_df.to_string(index=False))
    print(f"Saved regression results to {out_csv}")

# ---------------------------------------------------------------------------
# CLUSTERING TASK (Unsupervised learning)
# ---------------------------------------------------------------------------
def clustering_task(df: pd.DataFrame):
    """
    Unsupervised learning → group similar traffic-light crops.

    Features used:
        brightness, mean_r, mean_g, mean_b

    WHY THIS SATISFIES THE CHECKLIST:
        ✔ Required: "Clustering task"
        ✔ KMeans is a classic algorithm for unsupervised grouping
        ✔ Uses scaled numeric features
        ✔ Produces cluster summary + assignments CSV
    """

    features = ["brightness", "mean_r", "mean_g", "mean_b"]
    X = df[features].values

    # Scale features for KMeans (important for fair cluster distances)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans with 3 clusters (choice justified by brightness/color patterns)
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10,  # required for sklearn>=1.4 compatibility
    )

    clusters = kmeans.fit_predict(X_scaled)

    # Add cluster assignment to DataFrame
    df_clusters = df.copy()
    df_clusters["cluster"] = clusters

    # Print analysis (required in checklist)
    print("\n=== Cluster Sizes ===")
    print(df_clusters["cluster"].value_counts())

    print("\n=== Cluster Means (brightness & RGB) ===")
    print(df_clusters.groupby("cluster")[features].mean())

    # Save clustering results
    out_csv = RESULTS_DIR / "clustering_assignments.csv"
    df_clusters.to_csv(out_csv, index=False)

    print(f"\nSaved clustering assignments to {out_csv}")

# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def main():
    """
    Main task runner:
        1. Load metadata
        2. Run regression
        3. Run clustering
    """
    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found at {METADATA_CSV}. "
            "Run eda_traffic_lights.py first."
        )
    # Load image feature metadata (brightness, RGB, width, height, etc.)
    df = pd.read_csv(METADATA_CSV)

    # Perform regression + clustering tasks required by CS316
    regression_task(df)
    clustering_task(df)

# Execute when run directly
if __name__ == "__main__":
    main()