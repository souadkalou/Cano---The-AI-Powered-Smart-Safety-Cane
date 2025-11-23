"""
prepare_lisa.py

This script is responsible for converting the raw LISA Traffic Light Dataset into
a clean, machine-learning-ready dataset of **cropped red and green traffic lights**.

We wrote this script because the LISA dataset provides:
    • full-frame driving images
    • with bounding-box annotations
    • inside many separate CSV files
    • with inconsistent file paths

Our classifier requires:
    • ONLY the traffic-light region
    • saved as small, clean image crops
    • grouped into “red” and “green” classes

Therefore, this script:
    1) Reads all annotation CSVs
    2) Finds the correctly labeled traffic-light bounding boxes
    3) Locates the original image files (even when the paths are inconsistent)
    4) Crops ONLY the bounding box region
    5) Saves the crop into:
           data/processed/traffic_lights/red/
           data/processed/traffic_lights/green/
    6) Counts and reports the dataset quality (missing files, total crops, etc.)

This is a critical preprocessing stage for our traffic-light classifier training.
"""

from pathlib import Path
import pandas as pd
from PIL import Image

# -------------------------------------------------------------------------
# 1. Dataset paths
# We point to the root folder where the LISA dataset was extracted.
# NOTE: This must be changed to the student's own machine path if needed.
# -------------------------------------------------------------------------
LISA_ROOT = Path("/Users/roaa.hamdan/Downloads/archive")

# Annotation CSVs inside:
#   archive/Annotations/Annotations/<many folders>/frameAnnotationsBOX.csv
ANNOTATIONS_DIR = LISA_ROOT / "Annotations" / "Annotations"

# -------------------------------------------------------------------------
# 2. Output directories for the processed crops
# We create separate folders for red and green class images.
# They will contain thousands of small JPG crops.
# -------------------------------------------------------------------------
OUTPUT_RED = Path("data/processed/traffic_lights/red")
OUTPUT_GREEN = Path("data/processed/traffic_lights/green")

# Ensure the output folders exist (create if missing)
OUTPUT_RED.mkdir(parents=True, exist_ok=True)
OUTPUT_GREEN.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 3. Label mapping
# After inspecting the dataset (via inspect_lisa_tags.py), we discovered
# that these specific strings represent RED and GREEN states.
# These labels come directly from the LISA annotation format.
# -------------------------------------------------------------------------
RED_LABELS = ["stop", "stopLeft"]
GREEN_LABELS = ["go", "goLeft", "goForward"]


def read_annotation_csv(csv_path: Path) -> pd.DataFrame:
    """
    Helper function for reliably reading LISA CSV files.
    Some CSV files use ';' as a delimiter and others use ','.
    To avoid errors, we try reading with ';' first, then fallback.

    This improves preprocessing robustness and prevents silent failures.
    """
    try:
        return pd.read_csv(csv_path, sep=";")
    except Exception:
        return pd.read_csv(csv_path)

def main():
    """
    Main procedure that loops through all annotation CSV files, extracts
    bounding-box crops, and saves them to disk.

    Key improvements implemented here:
        ✓ We build a global index of ALL .jpg filenames so we can look them up
          even when the CSV refers to paths that do not actually exist.
        ✓ We skip invalid bounding boxes.
        ✓ We count missing or corrupt images for later reporting.
    """

    red_count = 0      # number of red traffic-light crops saved
    green_count = 0    # number of green traffic-light crops saved
    not_found_images = 0

    # -----------------------------------------------------
    # STEP 1: Build a fast lookup dictionary for ALL .jpg files
    # -----------------------------------------------------
    print("Building image index (scanning all .jpg under archive)...")

    # Because LISA paths are inconsistent (sometimes missing folders),
    # we decided to create a dictionary mapping:
    #     "image_filename.jpg" --> full absolute file path
    # This allows us to always locate the image efficiently.
    image_index: dict[str, Path] = {}

    for img_path in LISA_ROOT.rglob("*.jpg"):
        image_index[img_path.name] = img_path  # key = just the filename

    print(f"Indexed {len(image_index)} .jpg images.\n")

    # -----------------------------------------------------
    # STEP 2: Load all annotation CSV files from dataset
    # -----------------------------------------------------
    csv_files = list(ANNOTATIONS_DIR.rglob("frameAnnotationsBOX.csv"))
    print(f"Found {len(csv_files)} annotation CSV files.")

    # Loop through every annotation file
    for csv_path in csv_files:
        print(f"\nProcessing {csv_path}")
        df = read_annotation_csv(csv_path)

        # -----------------------------------------------------
        # STEP 3: Iterate row-by-row through each annotation file
        # -----------------------------------------------------
        for _, row in df.iterrows():

            # The annotation tag tells us what type of traffic-light signal it is.
            raw_label = str(row["Annotation tag"]).strip()

            # check if this is a RED label
            if raw_label in RED_LABELS:
                is_red = True
            # check if this is a GREEN label
            elif raw_label in GREEN_LABELS:
                is_red = False
            else:
                # Ignore non-traffic-light labels (like “warning”)
                continue

            # Extract only the filename portion (basename)
            filename = str(row["Filename"])
            basename = Path(filename).name

            # Look up this basename in our global image index
            img_path = image_index.get(basename)

            # If the image wasn’t found, something is wrong with the CSV
            if img_path is None:
                not_found_images += 1
                continue

            # Try to load the image from disk
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                # If the file is corrupted, unreadable, or missing
                not_found_images += 1
                continue

            # -----------------------------------------------------
            # STEP 4: Extract and validate bounding-box coordinates
            # -----------------------------------------------------
            x1 = int(row["Upper left corner X"])
            y1 = int(row["Upper left corner Y"])
            x2 = int(row["Lower right corner X"])
            y2 = int(row["Lower right corner Y"])

            # Skip invalid or zero-area bounding boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Crop the image ONLY to the traffic-light bounding box
            crop = img.crop((x1, y1, x2, y2))

            # -----------------------------------------------------
            # STEP 5: Save the crop into the correct class folder
            # -----------------------------------------------------
            if is_red:
                out_path = OUTPUT_RED / f"red_{red_count:05d}.jpg"
                red_count += 1
            else:
                out_path = OUTPUT_GREEN / f"green_{green_count:05d}.jpg"
                green_count += 1

            crop.save(out_path)

    # -----------------------------------------------------
    # Final summary for the instructor
    # -----------------------------------------------------
    print("\nDone!")
    print(f"Saved {red_count} RED crops")
    print(f"Saved {green_count} GREEN crops")
    print(f"Images not found / failed to open: {not_found_images}")
    print(f"Red dir:   {OUTPUT_RED.resolve()}")
    print(f"Green dir: {OUTPUT_GREEN.resolve()}")


# Standard Python entry point
if __name__ == "__main__":
    main()