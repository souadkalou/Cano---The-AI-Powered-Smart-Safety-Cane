"""
train_traffic_light.py

This script trains our custom TrafficLightCNN model on the cropped LISA traffic light
images that we prepared earlier using prepare_lisa.py.

High-level goals of this script:
    • Load red/green traffic-light crops from:
          data/processed/traffic_lights/red
          data/processed/traffic_lights/green
    • Build a labeled DataFrame with image paths and class names
    • Convert the DataFrame into a PyTorch Dataset and DataLoader
    • Train our CNN using a standard supervised learning loop
    • Evaluate on a validation split (F1-score + classification report)
    • Save the best model weights to disk for later use in Streamlit

We structured the code in a modular way so it can be reused and inspected easily.
"""
import sys, os

# -------------------------------------------------------------------------
# We compute the ROOT_DIR of the project so that "src" can be imported
# correctly, regardless of where this script is executed from.
#
# Explanation:
#   os.path.abspath(__file__)   -> absolute path of THIS file
#   os.path.dirname(...)        -> go up one level (folder containing this file)
#   repeated os.path.dirname    -> go up more levels to reach project root
# Here we go up 3 levels: /.../src/training/train_traffic_light.py -> project root
# -------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)  # Add project root to Python path so `src` is importable

from pathlib import Path  # Path class for platform-independent filesystem paths

import torch
from torch.utils.data import Dataset, DataLoader  # Base Dataset + batching utility
from torchvision import transforms               # Image transforms (resize, normalize)
from PIL import Image                            # Image loading (Pillow)
import pandas as pd                              # DataFrame to hold filepaths + labels
from sklearn.metrics import classification_report, f1_score  # Evaluation metrics
from sklearn.model_selection import train_test_split          # Train/validation split

# Import our custom CNN model defined in src/models/traffic_light_model.py
from src.models.traffic_light_model import TrafficLightCNN

# -------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# -------------------------------------------------------------------------

# Directory containing the processed red/green crops:
#   data/processed/traffic_lights/red/*.jpg
#   data/processed/traffic_lights/green/*.jpg
DATA_DIR = Path("data/processed/traffic_lights")

# Path where we will save the best-performing model weights
MODEL_OUT = Path("models/traffic_light_classifier/best_traffic_light.pt")

# Class names used for labeling and reporting
CLASSES = ["red", "green"]

# Device to run training on. For this course project we use CPU.
# If a GPU is available, this could be "cuda" instead.
DEVICE = "cpu"

# To control dataset size and keep training time reasonable,
# we cap each class at MAX_PER_CLASS samples.
MAX_PER_CLASS = 5000   # sample size per class for training


def build_dataframe() -> pd.DataFrame:
    """
    Build a DataFrame listing all image paths and their labels.

    This function:
        1. Walks through each class folder under DATA_DIR (red/green).
        2. For each .jpg file, records:
               - full filepath
               - class label ("red" or "green")
        3. Balances the dataset by:
               - limiting to MAX_PER_CLASS per class
               - shuffling the combined DataFrame for randomness

    Returns
    -------
    pd.DataFrame
        A shuffled, balanced DataFrame with columns:
            'filepath' (Path objects)
            'label'    (string: "red" or "green")
    """

    rows = []  # We collect all rows here before converting into a DataFrame.

    # Iterate over each class name defined in CLASSES: ["red", "green"]
    for label in CLASSES:
        # Construct the directory path for this class:
        #   data/processed/traffic_lights/red
        #   data/processed/traffic_lights/green
        class_dir = DATA_DIR / label

        # For every .jpg file in the class directory, create a row.
        for img_path in class_dir.glob("*.jpg"):
            rows.append({"filepath": img_path, "label": label})

    # Convert the list of rows into a pandas DataFrame
    df = pd.DataFrame(rows)

    # ---------------------------------------------------------------------
    # We now apply balancing logic so that:
    #   • each class has at most MAX_PER_CLASS samples
    #   • the final dataset is shuffled
    # ---------------------------------------------------------------------
    dfs = []

    for label in CLASSES:
        # Filter rows belonging to the current class
        sub = df[df["label"] == label]

        # If this class has more than MAX_PER_CLASS samples,
        # randomly downsample to that limit for fairness & speed.
        if len(sub) > MAX_PER_CLASS:
            sub = sub.sample(MAX_PER_CLASS, random_state=42)

        # Append the (possibly downsampled) subset
        dfs.append(sub)

    # Concatenate all class subsets into one DataFrame
    df_balanced = pd.concat(dfs)

    # Shuffle all rows randomly (sample(frac=1)) and reset the index
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced

class TrafficLightDataset(Dataset):
    """
    Custom PyTorch Dataset wrapping our traffic-light image DataFrame.

    Responsibilities:
        • Read images from disk using file paths in the DataFrame.
        • Apply consistent preprocessing transforms (resize, normalize).
        • Return (image_tensor, label_index) for each sample.

    This Dataset is then passed into a DataLoader to provide mini-batches.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Constructor to store the DataFrame and define image transforms.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'filepath' and 'label' columns.
        """
        # We reset index to ensure __getitem__ works with 0..N-1
        self.df = df.reset_index(drop=True)

        # Define image preprocessing:
        #   • Resize all images to 224×224 (expected by our CNN)
        #   • Convert PIL image to tensor
        #   • Normalize using ImageNet-like mean/std (standard practice)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # typical RGB means
                std=[0.229, 0.224, 0.225]    # typical RGB stds
            )
        ])

    def __len__(self):
        """
        Return the total number of samples in this dataset.
        Required by PyTorch's Dataset interface.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Fetch one sample (image + label) by index.

        Steps:
            1. Look up row 'idx' in the DataFrame.
            2. Open the image from disk and convert to RGB.
            3. Apply the predefined transforms.
            4. Convert the string label ("red"/"green") to numeric index (0/1).

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        (img, label_idx) : (torch.Tensor, int)
            • img        : image tensor of shape [3, 224, 224]
            • label_idx  : integer class index (0 or 1)
        """
        # Get the row corresponding to this index
        row = self.df.iloc[idx]

        # Open the image file and ensure it is in RGB mode
        img = Image.open(row["filepath"]).convert("RGB")

        # Apply resize + normalization transforms
        img = self.transform(img)

        # Map label string ("red"/"green") to its index in CLASSES
        label_idx = CLASSES.index(row["label"])

        return img, label_idx


def train():
    """
    Entry point for training the TrafficLightCNN model.

    This function:
        1. Builds a balanced DataFrame of images and labels.
        2. Splits the data into train and validation sets.
        3. Wraps them into Datasets and DataLoaders.
        4. Constructs the model, loss function, and optimizer.
        5. Trains the model for several epochs.
        6. Evaluates using F1-score + classification report.
        7. Saves the best model weights (by validation F1-score).

    This training script is run offline once to produce the .pt model file
    that we later load in the Streamlit app.
    """

    # ---------------------------------------------------------
    # STEP 1: Build balanced dataset DataFrame
    # ---------------------------------------------------------
    df = build_dataframe()

    print("Total samples used:", len(df))          # total number of images
    print(df["label"].value_counts())             # class distribution summary

    # ---------------------------------------------------------
    # STEP 2: Train/validation split (80/20), stratified by label
    # ---------------------------------------------------------
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,                 # 20% of data for validation
        stratify=df["label"],          # preserve label distribution
        random_state=42                # fixed seed for reproducibility
    )

    # Wrap the DataFrames into our custom Dataset objects
    train_ds = TrafficLightDataset(train_df)
    val_ds = TrafficLightDataset(val_df)

    # DataLoader turns the Dataset into mini-batches:
    #   • batch_size=32 is a reasonable trade-off for CPU training
    #   • shuffle=True for training to randomize sample order
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # ---------------------------------------------------------
    # STEP 3: Model, loss function, and optimizer
    # ---------------------------------------------------------

    # Initialize our CNN with 2 output classes (red & green)
    model = TrafficLightCNN(num_classes=len(CLASSES)).to(DEVICE)

    # CrossEntropyLoss is standard for multi-class classification
    criterion = torch.nn.CrossEntropyLoss()

    # Adam optimizer is a good default choice for CNN training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # We will track the best validation F1-score to decide which model to save
    best_f1 = 0.0

    # ---------------------------------------------------------
    # STEP 4: Training loop over epochs
    # ---------------------------------------------------------
    for epoch in range(8):  # We train for 8 epochs (tunable hyperparameter)
        model.train()       # Set model to training mode (enables dropout, etc.)
        running_loss = 0.0  # To accumulate loss across the epoch

        # -------------------------------
        # 4a. Iterate over all batches
        # -------------------------------
        for images, labels in train_loader:
            # Move the mini-batch tensors to the chosen device (CPU here)
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Reset gradients from the previous step
            optimizer.zero_grad()

            # Forward pass: compute model predictions for this batch
            outputs = model(images)

            # Compute classification loss
            loss = criterion(outputs, labels)

            # Backward pass: compute gradients w.r.t. parameters
            loss.backward()

            # Update model parameters using the optimizer
            optimizer.step()

            # Accumulate scaled loss for epoch average
            running_loss += loss.item() * images.size(0)

        # Compute average loss over all training samples this epoch
        epoch_loss = running_loss / len(train_loader.dataset)

        # ---------------------------------------------------------
        # STEP 5: Validation phase (no gradient computation)
        # ---------------------------------------------------------
        model.eval()  # Set model to evaluation mode (disables dropout)
        all_labels, all_preds = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                # Forward pass only (no backprop)
                outputs = model(images)

                # Predicted class = index of max logit
                preds = outputs.argmax(dim=1)

                # Move to CPU and store for metric computation
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

        # ---------------------------------------------------------
        # STEP 6: Compute F1-score and classification report
        # ---------------------------------------------------------
        f1 = f1_score(all_labels, all_preds, average="macro")
        print(f"\nEpoch {epoch+1}/8 - Loss: {epoch_loss:.4f} - F1: {f1:.3f}")

        # classification_report provides per-class precision, recall, F1, support
        print(classification_report(all_labels, all_preds, target_names=CLASSES))

        # If this epoch's F1 is better than any previous one, we save the model
        if f1 > best_f1:
            best_f1 = f1

            # Ensure the output directory exists
            MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

            # Save only the model's state_dict (weights, not full object)
            torch.save(model.state_dict(), MODEL_OUT)

            print(f"✅ Saved new best model with F1={best_f1:.3f} -> {MODEL_OUT}")

    # ---------------------------------------------------------
    # Final summary after all epochs
    # ---------------------------------------------------------
    print("\nTraining finished. Best F1:", best_f1)


# Standard Python entry point: run training only if executed directly
if __name__ == "__main__":
    train()