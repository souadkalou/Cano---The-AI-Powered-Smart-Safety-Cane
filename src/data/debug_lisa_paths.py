"""
This script was developed as part of our dataset preparation workflow for Cano project.

Its purpose is to systematically debug the LISA Traffic Light Dataset file
structure. We discovered early in the project that the annotation CSV files
use filename paths that often do NOT match the actual folder structure in the
dataset. This caused our cropping script to save 0 images initially.

To resolve this, we designed this diagnostic tool to validate:
    • whether annotation CSV files are being read correctly
    • which columns they contain
    • what filenames the annotations reference
    • whether those filenames physically exist in the dataset
    • and if not, whether the images can be located elsewhere by searching recursively

These insights allowed us to rewrite our dataset preparation script correctly.
"""

from pathlib import Path
import pandas as pd

# -------------------------------------------------------------------------
# We begin by defining the root directory of the LISA dataset.
# This path points to the unpacked dataset location on our system.
# -------------------------------------------------------------------------
LISA_ROOT = Path("/Users/roaa.hamdan/Downloads/archive")

# The annotation CSV files are stored inside a nested folder structure:
#     archive/Annotations/Annotations/
# We define this location so that we can scan it for all annotation files.
ANNOTATIONS_DIR = LISA_ROOT / "Annotations" / "Annotations"


def read_annotation_csv(csv_path: Path) -> pd.DataFrame:
    """
    We created this helper function to robustly read annotation CSV files.

    The LISA dataset uses mixed delimiters (some CSVs use semicolons, others commas).
    To handle both cases cleanly, we first attempt to read using ';'.
    If this fails, we fall back to reading with the default delimiter.

    By doing this, we eliminate file-format inconsistencies as a source of failure.
    """
    try:
        return pd.read_csv(csv_path, sep=";")
    except Exception:
        return pd.read_csv(csv_path)

def main():
    """
    This is the main debugging workflow.

    Here we perform multiple validation steps:
        1. detect all CSV annotation files
        2. load one file and inspect its structure
        3. extract one traffic-light annotation entry
        4. check whether the referenced image actually exists
        5. if not, search recursively to locate the real file

    These diagnostics help us correct the image-path resolution logic
    in our dataset preparation module.
    """
    # ---------------------------------------------------------------------
    # Step 1: Locate all annotation CSV files.
    # We use rglob() because the dataset contains multiple nested sequences.
    # ---------------------------------------------------------------------
    csv_files = list(ANNOTATIONS_DIR.rglob("frameAnnotationsBOX.csv"))
    print(f"Found {len(csv_files)} CSV files")

    # If no CSV files are detected, the dataset path is incorrect,
    # so we stop early and notify the user.
    if not csv_files:
        print("ERROR: No CSV files found. Check ANNOTATIONS_DIR path.")
        return

    # For initial debugging, analyzing a single CSV file is sufficient.
    csv_path = csv_files[0]
    print(f"\nUsing CSV: {csv_path}\n")

    # ---------------------------------------------------------------------
    # Step 2: Load the annotation CSV.
    # ---------------------------------------------------------------------
    df = read_annotation_csv(csv_path)

    # Display the detected columns so we understand the structure.
    print("Columns:", list(df.columns))

    # Preview the first few rows for contextual understanding.
    print("\nFirst 3 rows:")
    print(df.head(3))

    # ---------------------------------------------------------------------
    # Step 3: Identify one annotation entry containing a traffic-light label.
    #
    # Not all entries correspond to traffic lights (some involve “warning” tags).
    # We filter by tags relevant to our project (red/green light recognition).
    # ---------------------------------------------------------------------
    valid_labels = ["go", "goLeft", "goForward", "stop", "stopLeft"]

    # Extract the first matching row.
    row = df[df["Annotation tag"].isin(valid_labels)].iloc[0]

    # Retrieve the filename path used inside the annotation CSV.
    filename = str(row["Filename"])
    print("\nFilename from CSV:", filename)

    # ---------------------------------------------------------------------
    # Step 4: Attempt a direct file lookup.
    #
    # This tests whether the CSV's relative file path matches the actual dataset.
    # In most cases, this fails due to missing "frames" subdirectories.
    # ---------------------------------------------------------------------
    direct = LISA_ROOT / filename
    print("\nChecking direct path:", direct)
    print("Exists?", direct.exists())

    # ---------------------------------------------------------------------
    # Step 5: If the direct path is invalid, search by basename.
    #
    # The CSV may reference:
    #     daySequence1/dayClip1--00123.jpg
    #
    # But the true path may be:
    #     daySequence1/dayClip1/frames/dayClip1--00123.jpg
    #
    # Searching by basename allows us to locate the correct file even when
    # directory structure differs.
    # ---------------------------------------------------------------------
    basename = Path(filename).name
    print(f"\nSearching for basename '{basename}' inside archive...")

    matches = list(LISA_ROOT.rglob(basename))
    print("Matches found:", len(matches))

    # We limit the output to the first 10 matches for readability.
    for m in matches[:10]:
        print("Match:", m)

# -------------------------------------------------------------------------
# Standard Python entry point to run main() only when executed directly.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()