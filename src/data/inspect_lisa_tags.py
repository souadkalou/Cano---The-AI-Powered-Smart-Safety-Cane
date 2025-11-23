"""
This script is part of our data-understanding phase for  Cano project. 
Before training any model on the LISA Traffic Light Dataset, we needed to
understand:

    • What annotation labels (tags) actually appear in the dataset
    • How frequently each tag occurs
    • Whether there are additional tags beyond simple "go" / "stop" traffic lights

We wrote this small inspection tool to systematically scan several annotation CSV files
and summarize the distribution of the "Annotation tag" column. The results guide the
choice of classes for our traffic-light classifier and help us document the dataset
in the report.
"""
from pathlib import Path
import pandas as pd

# -------------------------------------------------------------------------
# We define the root folder where we unzipped the LISA Traffic Light Dataset.
# This must point to the *top-level* "archive" directory on our machine.
# -------------------------------------------------------------------------
LISA_ROOT = Path("/Users/roaa.hamdan/Downloads/archive")

# The annotation CSV files are stored in a nested path:
#   archive/Annotations/Annotations/...
# We store this directory so that we can search it recursively for CSVs.
ANNOTATIONS_DIR = LISA_ROOT / "Annotations" / "Annotations"


def read_annotation_csv(csv_path: Path) -> pd.DataFrame:
    """
    We use this helper function to robustly load a single LISA annotation CSV.

    Some CSV files in the dataset use ';' as a separator, while others use ','.
    To avoid format-related errors, we first try to read using sep=';'.
    If that fails (e.g., due to a parsing error), we fall back to the default
    CSV reader settings.

    Parameters
    ----------
    csv_path : Path
        Path object pointing to one annotation CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all rows and columns of the annotation file.
    """
    try:
        # First attempt: use semicolon as delimiter (common in LISA dataset).
        df = pd.read_csv(csv_path, sep=";")
    except Exception:
        # Fallback: rely on pandas' default delimiter detection.
        df = pd.read_csv(csv_path)
    return df


def main():
    """
    Main inspection routine.

    In this function we:
        1. Discover all annotation CSV files inside ANNOTATIONS_DIR.
        2. Load a small subset of them (first 5 files) to keep the run fast.
        3. For each file:
            - print its path (for traceability)
            - show the column names (schema)
            - compute and display the frequency of each 'Annotation tag'
        4. Aggregate all labels from the inspected files and print a global
           frequency table (top 50 tags).

    The output is used in the report to justify which labels we treat as
    traffic lights (e.g., 'go', 'stop', 'goLeft', 'stopLeft', 'warning').
    """

    # ---------------------------------------------------------------------
    # Step 1: recursively collect all CSV annotation files.
    # There are many files, but we do not need to inspect all of them here.
    # ---------------------------------------------------------------------
    csv_files = list(ANNOTATIONS_DIR.rglob("frameAnnotationsBOX.csv"))
    print(f"Found {len(csv_files)} CSV files")

    # We will keep track of every label we see, across multiple files,
    # so that we can compute a global frequency distribution at the end.
    all_labels = []

    # ---------------------------------------------------------------------
    # Step 2: Loop over only the first few CSV files.
    #
    # We intentionally restrict ourselves to csv_files[:5] so that this
    # script runs quickly during development. For a full analysis we could
    # remove the slice and process all files.
    # ---------------------------------------------------------------------
    for csv_path in csv_files[:5]:
        print(f"\n--- {csv_path} ---")

        # Load the CSV into a DataFrame using our robust reader.
        df = read_annotation_csv(csv_path)

        # Display the column names so we can verify the schema is consistent.
        print("Columns:", list(df.columns))

        # We focus on the "Annotation tag" column which encodes the label.
        # We cast to string to avoid potential issues with mixed types.
        labels = df["Annotation tag"].astype(str)

        # For each individual file, we show how many times each tag appears.
        # This helps us notice rare labels or typos.
        print("Unique tags in this file:")
        print(labels.value_counts())

        # Extend the global list with all labels from this file,
        # so we can compute overall statistics later on.
        all_labels.extend(labels.tolist())

    # ---------------------------------------------------------------------
    # Step 3: After looping over the subset of files, we compute a global
    #         frequency distribution of all observed labels.
    #
    # We limit the printed output to the top 50 labels to keep the console
    # readable, while still revealing the most common tags.
    # ---------------------------------------------------------------------
    print("\n===== GLOBAL UNIQUE TAGS (first 50) =====")
    print(pd.Series(all_labels).value_counts().head(50))


# Standard Python entry point to ensure main() runs only if
# the script is executed directly (and not imported as a module).
if __name__ == "__main__":
    main()