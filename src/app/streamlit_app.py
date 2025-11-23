# ============================
# Main Streamlit Application
# ============================

# Standard library imports
from pathlib import Path  # For building OS-independent file paths
import pandas as pd       # For reading CSV results in the analysis dashboard

import sys                # To modify Python path so we can import from src/
import time               # For small delays (e.g., let camera warm up)

# Third-party libraries
import cv2                # OpenCV – for camera access and image processing
import numpy as np        # Numerical operations on arrays (e.g., depth maps)
import streamlit as st    # Streamlit – for building the web UI
import torch              # PyTorch – deep learning framework
from torchvision import transforms  # For CNN image preprocessing
from PIL import Image     # Pillow – for image loading and conversion
import pyttsx3            # Text-to-speech on the local machine

# -------------------------------------------------------
# Make sure Python can see the src/ package
# -------------------------------------------------------

# ROOT_DIR is the project root folder: <project>/, two levels above this file
# (since this file is in src/app/streamlit_app.py)
ROOT_DIR = Path(__file__).resolve().parents[2]

# Add the project root directory to sys.path so imports like `from src.models...`
# will work correctly when running with `streamlit run src/app/streamlit_app.py`
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Now that src/ is on the path, we can import our custom model wrappers
from src.models.yolo_detector import YoloDetector          # Wrapper for YOLOv8
from src.models.midas_estimator import MidasDepthEstimator # Wrapper for MiDaS depth model
from src.models.traffic_light_model import TrafficLightCNN # CNN classifier for RED/GREEN lights


# -------------------------------------------------------
# Text-to-speech helpers
# -------------------------------------------------------
def init_tts():
    """
    Initialize the pyttsx3 text-to-speech engine and slightly slow down the rate
    to make spoken messages clearer for the user.
    """
    engine = pyttsx3.init()               # Create TTS engine
    rate = engine.getProperty("rate")     # Get current speaking rate
    engine.setProperty("rate", int(rate * 0.9))  # Slow it down by ~10%
    return engine


def speak(engine, text: str):
    """
    Speak a given text string using the initialized TTS engine.
    We call engine.stop() first to avoid overlapping audio.
    Any error is silently ignored so the app does not crash if audio is not available.
    """
    try:
        engine.stop()    # Stop any previous phrase still in the queue
        engine.say(text) # Queue the new text to be spoken
        engine.runAndWait()  # Block until speaking finishes
    except Exception:
        # Fail silently if TTS is not available (e.g., no audio device)
        pass


# -------------------------------------------------------
# Model loading
# -------------------------------------------------------
@st.cache_resource
def load_models():
    """
    Load all heavy models exactly once and cache them with Streamlit.
    This is a major optimization: YOLO, MiDaS, and the CNN are expensive to load,
    so caching avoids re-loading them for every rerun or user interaction.
    """

    # Build absolute path to YOLOv8 weights: e.g., <project_root>/yolov8n.pt
    yolo_weights = ROOT_DIR / "yolov8n.pt"
    # Initialize our YoloDetector wrapper on CPU
    yolo = YoloDetector(str(yolo_weights), device="cpu")

    # Initialize MiDaS depth estimator (also on CPU)
    midas = MidasDepthEstimator(device="cpu")

    # Initialize our traffic light CNN classifier with 2 output classes: [red, green]
    traffic_model = TrafficLightCNN(num_classes=2)
    # Path to the trained CNN weights file
    tl_weights = ROOT_DIR / "models" / "traffic_light_classifier" / "best_traffic_light.pt"
    # Load the trained weights into the model
    state = torch.load(tl_weights, map_location="cpu")
    traffic_model.load_state_dict(state)
    traffic_model.eval()  # Set model to evaluation mode (no dropout, no training)

    # Define preprocessing pipeline for traffic-light crops:
    # resize → tensor → normalize with ImageNet-like mean/std
    traffic_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),           # Resize to 224x224
            transforms.ToTensor(),                   # Convert PIL image to PyTorch tensor
            transforms.Normalize(                    # Normalize to improve model stability
                mean=[0.485, 0.456, 0.406],          # Standard ImageNet mean
                std=[0.229, 0.224, 0.225],           # Standard ImageNet std
            ),
        ]
    )

    # Return all four: YOLO model, MiDaS model, CNN model, and its transform
    return yolo, midas, traffic_model, traffic_transform


# -------------------------------------------------------
# Analysis dashboard (EDA + metrics)
# -------------------------------------------------------
def show_analysis_dashboard():
    """
    Streamlit page that displays:
    - Model comparison table for the traffic light classifier
    - Confusion matrix & ROC curve plots
    - Regression results (e.g., brightness prediction)
    - Clustering summary
    - EDA plots (distributions, correlations, trends)
    This satisfies the "EDA" and "Model Evaluation" parts of the project checklist.
    """
    st.markdown("## 📊 Data & Model Analysis")

    # Directories where analysis scripts have saved plots and result CSVs
    plots_dir = ROOT_DIR / "analysis" / "plots"
    results_dir = ROOT_DIR / "analysis" / "results"

    # ---------- Traffic Light Model Evaluation ----------
    st.markdown("### 🚦 Traffic Light Classifier – Performance")

    # Comparison table CSV – output from evaluate_traffic_model.py
    cmp_path = ROOT_DIR / "analysis" / "traffic_light_model_comparison.csv"
    if cmp_path.exists():
        cmp_df = pd.read_csv(cmp_path)    # Load comparison table (CNN vs baselines)
        st.markdown("**Model Comparison Table**")
        st.dataframe(cmp_df)              # Show it as an interactive table in Streamlit
    else:
        st.warning("traffic_light_model_comparison.csv not found. Run evaluate_traffic_model.py first.")

    # Paths to confusion matrix and ROC curve plots for the CNN model
    cm_path = plots_dir / "cm_traffic_light_cnn.png"
    roc_path = plots_dir / "roc_traffic_light_cnn.png"

    # Display the two plots side by side using columns
    cols = st.columns(2)
    with cols[0]:
        if cm_path.exists():
            st.markdown("**Confusion Matrix (TrafficLightCNN)**")
            st.image(str(cm_path))
    with cols[1]:
        if roc_path.exists():
            st.markdown("**ROC Curve (TrafficLightCNN)**")
            st.image(str(roc_path))

    # Horizontal rule to separate sections
    st.markdown("---")

    # ---------- Regression Results ----------
    st.markdown("### 📈 Regression Task – Predict Brightness")

    # Regression metrics CSV – output from regression_clustering_tasks.py
    reg_path = results_dir / "regression_results.csv"
    if reg_path.exists():
        reg_df = pd.read_csv(reg_path)   # Contains MAE, RMSE, R², etc. for multiple models
        st.dataframe(reg_df)             # Show as table
    else:
        st.warning("regression_results.csv not found. Run regression_clustering_tasks.py first.")

    st.markdown("---")

    # ---------- Clustering Summary ----------
    st.markdown("### 🧩 Clustering – Traffic Light Groups")

    # Clustering assignments CSV – each crop assigned to a cluster
    cluster_path = results_dir / "clustering_assignments.csv"
    if cluster_path.exists():
        cluster_df = pd.read_csv(cluster_path)

        # Show cluster size (how many samples in each cluster)
        st.write("**Cluster sizes**")
        st.write(cluster_df["cluster"].value_counts())

        # Show mean brightness & RGB per cluster to interpret clusters
        st.write("**Cluster means (brightness & RGB)**")
        st.write(cluster_df.groupby("cluster")[["brightness", "mean_r", "mean_g", "mean_b"]].mean())
    else:
        st.warning("clustering_assignments.csv not found. Run regression_clustering_tasks.py first.")

    st.markdown("---")

    # ---------- EDA Plots ----------
    st.markdown("### 🔍 EDA – Traffic Light Crops")

    # List of (title, filename) pairs for key EDA plots generated by eda_traffic_lights.py
    plot_files = [
        ("Brightness histogram", "hist_brightness.png"),
        ("Aspect ratio histogram", "hist_aspect_ratio.png"),
        ("Class counts", "bar_class_counts.png"),
        ("Brightness by label (boxplot)", "box_brightness_by_label.png"),
        ("Brightness distribution (violin)", "violin_brightness_by_label.png"),
        ("Feature correlations", "heatmap_correlations.png"),
        ("Brightness trend (line plot)", "line_brightness_trend.png"),
    ]

    # Loop over each plot filename and display it if it exists
    for title, fname in plot_files:
        path = plots_dir / fname
        if path.exists():
            st.markdown(f"**{title}**")
            st.image(str(path))


# -------------------------------------------------------
# Streamlit UI + CSS
# -------------------------------------------------------

# Configure Streamlit page: title in browser tab, and layout set to "wide"
st.set_page_config(page_title="Cano Smart Cane", layout="wide")

# Sidebar radio button to choose between two modes:
# 1) Live Demo (camera + real-time detection)
# 2) Data & Model Analysis (EDA & metrics)
mode = st.sidebar.radio("Mode", ["Live Demo", "Data & Model Analysis"])

# Custom CSS to mimic the dark UI design with colored boxes
# We inject raw HTML <style> into Streamlit; unsafe_allow_html=True is required.
st.markdown(
    """
<style>
/* Main Streamlit app background (dark theme) */
.stApp {
    background-color: #111111 !important;
}

/* Make main heading text white for contrast */
h1, h2, h3, h4, h5 {
    color: #ffffff !important;
}

/* Title label above each panel box */
.label-title {
    font-size: 18px;
    font-weight: 600;
    color: #00bcd4;
    margin-bottom: 4px;
}

/* Large status boxes (e.g., Distance Alert, Traffic Light Status) */
.big-box {
    border: 2px solid #666666;
    border-radius: 6px;
    padding: 12px 16px;
    text-align: center;
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 10px;
    background-color: #222222;
    color: #ffffff;
}

/* Color variants for danger / warning / safe */
.big-box.danger {
    background-color: #b00020;
    color: #ffffff;
}
.big-box.warning {
    background-color: #ff9800;
    color: #000000;
}
.big-box.safe {
    background-color: #1b5e20;
    color: #ffffff;
}

/* Smaller info boxes (vehicles list, other objects, etc.) */
.small-box {
    border: 1px solid #666666;
    border-radius: 4px;
    padding: 8px 10px;
    font-size: 14px;
    min-height: 48px;
    margin-bottom: 10px;
    background-color: #222222;
    color: #ffffff;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# PAGE 1: LIVE DEMO
# -------------------------------------------------------
if mode == "Live Demo":
    # Main title inside the app for the live demo mode
   # Load logo
    logo_path = ROOT_DIR / "src" / "app" / "CanoLogoStr.png"

    # Two-column header (logo + title)
    col_logo, col_title = st.columns([1, 9])

    with col_logo:
        st.image(str(logo_path), width=200)   # Adjust size as needed

    with col_title:
        st.markdown(
            "<h2 style='color: white; margin-top: 15px;'>Cano - Smart Safety Cane</h2>",
            unsafe_allow_html=True
        )

    # Two-column layout: left for camera feed, right for detection results
    col_cam, col_panel = st.columns([3, 2])

    # Left column: camera feed
    with col_cam:
        st.markdown("### Camera Feed")
        FRAME = st.empty()  # Placeholder where we will show video frames

    # Right column: detection and status panels
    with col_panel:
        st.markdown("### Detection Results")
        traffic_box = st.empty()   # Traffic light status panel
        distance_box = st.empty()  # Distance alert panel
        vehicles_box = st.empty()  # Detected vehicles info
        people_box = st.empty()    # People count
        other_box = st.empty()     # Other detected labels

    # Button to start the camera and models
    start = st.button("Start Camera")

    if start:
        # Load YOLO, MiDaS, CNN model, and preprocessing only once (cached)
        yolo, midas, traffic_model, traffic_transform = load_models()
        # Initialize text-to-speech engine
        tts_engine = init_tts()

        # Variables to store last spoken states (to avoid repeating speech)
        last_distance_alert = None   # "SAFE" / "WARNING" / "DANGER"
        last_tl_message = None       # "Safe to cross" / "Do not cross" / None

        # Open the default camera (index 0)
        cap = cv2.VideoCapture(0)

        # If camera is not accessible, show error in UI
        if not cap.isOpened():
            st.error("Cannot open camera")
        else:
            # Small delay so camera exposure can auto-adjust
            time.sleep(0.5)

            # Main loop: read frames continuously from the camera
            while True:
                ret, frame = cap.read()  # ret=True if frame was grabbed correctly
                if not ret:
                    st.error("Failed to read frame.")
                    break  # Exit the loop if no frame is available

                # ---------------------------------------------------
                # (1) Depth map (used for info & vehicles panel)
                # ---------------------------------------------------
                # Use MiDaS to estimate depth values for the entire frame.
                # depth_map is a 2D array with relative depth values.
                depth_map = midas.estimate(frame)
                # Minimum depth across the frame (closest point, for debugging/info)
                min_dist = float(np.min(depth_map))

                # ---------------------------------------------------
                # (2) YOLO detection
                # ---------------------------------------------------
                # Run YOLOv8 detection on the current frame
                result = yolo.detect(frame)
                boxes = result.boxes  # All detected bounding boxes

                traffic_crop = None   # Will store traffic light crop for CNN
                traffic_conf = 0.0    # Best confidence for detected traffic light

                vehicles_info = []    # List of strings describing vehicles + distance
                people_count = 0      # Counter for detected people
                other_labels = set()  # Set of other object labels

                # Classes we consider as "vehicles"
                VEHICLE_LABELS = {"car", "truck", "bus", "motorbike", "motorcycle"}

                # Will store relative size of the largest detected object (0..1)
                # Used as a proxy for how close the nearest obstacle is.
                max_rel_size = 0.0

                # Iterate over each detected object (box)
                for box in boxes:
                    # Extract bounding box coordinates as integers
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    # Class ID and confidence
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = result.names[cls_id]  # Class name as string

                    # Estimate object "depth" by taking the minimum depth within the box area
                    obj_depth = float(
                        np.min(depth_map[max(y1, 0):max(y2, 1), max(x1, 0):max(x2, 1)])
                    )

                    # Draw bounding box and label on the frame for visualization
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f} {obj_depth:.1f}m",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

                    # Compute relative area of this box compared to the whole frame
                    h, w, _ = frame.shape
                    box_area = max((x2 - x1), 1) * max((y2 - y1), 1)
                    frame_area = max(w * h, 1)
                    rel_size = box_area / frame_area

                    # Keep track of the largest relative size (closest obstacle)
                    if rel_size > max_rel_size:
                        max_rel_size = rel_size

                    # Collect stats for side panels
                    if label in VEHICLE_LABELS:
                        # Add vehicle with its estimated depth
                        vehicles_info.append(f"{label} - {obj_depth:.1f}m")
                    elif label == "person":
                        # Count number of people detected
                        people_count += 1
                    elif label != "traffic light":
                        # Add any other label (except traffic light) to "Other Objects" set
                        other_labels.add(label)

                    # If YOLO detects a traffic light, keep the best-confidence box as crop
                    if label == "traffic light" and conf > traffic_conf:
                        traffic_conf = conf
                        traffic_crop = frame[y1:y2, x1:x2].copy()

                # ---------------------------------------------------
                # (3) Distance alert box + audio based on YOLO box size
                # ---------------------------------------------------
                # max_rel_size is:
                #   - 0.00 : no objects detected
                #   - 0.05 : small object, fairly far
                #   - 0.30 : large object, quite close (we treat as danger)

                CLOSE_THRESH = 0.30   # Threshold for DANGER
                MID_THRESH = 0.15     # Threshold for WARNING

                # Decide the alert level based on max_rel_size
                if max_rel_size == 0.0:
                    # No objects detected
                    alert_text = "SAFE"
                    alert_class = "safe"
                    distance_desc = "No obstacles detected"
                    distance_voice = None
                elif max_rel_size >= CLOSE_THRESH:
                    # Very close large object
                    alert_text = "DANGER"
                    alert_class = "danger"
                    distance_desc = "Obstacle very close"
                    distance_voice = "Danger! Obstacle very close."
                elif max_rel_size >= MID_THRESH:
                    # Medium-sized object: warn user
                    alert_text = "WARNING"
                    alert_class = "warning"
                    distance_desc = "Obstacle ahead"
                    distance_voice = "Warning. Obstacle ahead."
                else:
                    # Objects are relatively small: considered safe
                    alert_text = "SAFE"
                    alert_class = "safe"
                    distance_desc = "Obstacles are far"
                    distance_voice = None

                # Speak the distance alert message only when the alert changes
                # This avoids repeating the same speech every frame.
                if distance_voice is not None and alert_text != last_distance_alert:
                    speak(tts_engine, distance_voice)
                # Update last alert state
                last_distance_alert = alert_text

                # Update Distance Alert panel in the UI
                distance_box.markdown(
                    f"""
<div class="label-title">Distance Alert</div>
<div class="big-box {alert_class}">{alert_text}</div>
<div class="small-box">{distance_desc}</div>
""",
                    unsafe_allow_html=True,
                )

                # ---------------------------------------------------
                # (4) Traffic light classification (RED vs GREEN)
                # ---------------------------------------------------
                if traffic_crop is not None:
                    # Convert BGR (OpenCV) to RGB and then to PIL for transforms
                    crop_rgb = cv2.cvtColor(traffic_crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(crop_rgb)
                    # Apply the same preprocessing used during training
                    input_tensor = traffic_transform(pil_img).unsqueeze(0)

                    # Run the CNN in evaluation mode (no gradient computation)
                    with torch.no_grad():
                        outputs = traffic_model(input_tensor)
                        probs = torch.softmax(outputs, dim=1)[0]   # Softmax over 2 classes
                        pred_idx = int(torch.argmax(probs))        # Class index 0 or 1
                        classes = ["red", "green"]                 # Label mapping
                        pred_label = classes[pred_idx]             # Label string
                        pred_prob = float(probs[pred_idx])         # Confidence score

                    # Text for the panel, e.g., "GREEN (0.98)"
                    tl_text = f"{pred_label.upper()} ({pred_prob:.2f})"

                    # If the model is confident, generate audio message
                    if pred_prob > 0.90:
                        if pred_label == "green":
                            tl_msg = "Safe to cross"
                        else:
                            tl_msg = "Do not cross"

                        # Speak only when message changes so user doesn't get spammed
                        if tl_msg != last_tl_message:
                            speak(tts_engine, tl_msg)
                            last_tl_message = tl_msg
                    else:
                        tl_msg = None
                else:
                    # No traffic light crop detected this frame
                    tl_text = "NONE"

                # Update Traffic Light Status panel
                traffic_box.markdown(
                    f"""
<div class="label-title">Traffic Light Status</div>
<div class="big-box">{tl_text}</div>
""",
                    unsafe_allow_html=True,
                )

                # ---------------------------------------------------
                # (5) Vehicles / People / Other objects panel
                # ---------------------------------------------------

                # Vehicles panel: list all detected vehicles with their approximate depth
                if vehicles_info:
                    vehicles_text = "<br>".join(vehicles_info)
                else:
                    vehicles_text = "None detected"

                vehicles_box.markdown(
                    f"""
<div class="label-title">Vehicles Detected</div>
<div class="small-box">{vehicles_text}</div>
""",
                    unsafe_allow_html=True,
                )

                # People panel: show count of detected persons
                people_box.markdown(
                    f"""
<div class="label-title">People Detected</div>
<div class="small-box">{people_count}</div>
""",
                    unsafe_allow_html=True,
                )

                # Other objects panel: show any other labels not in vehicles or traffic light
                if other_labels:
                    other_text = ", ".join(sorted(other_labels))
                else:
                    other_text = "None detected"

                other_box.markdown(
                    f"""
<div class="label-title">Other Objects</div>
<div class="small-box">{other_text}</div>
""",
                    unsafe_allow_html=True,
                )

                # ---------------------------------------------------
                # (6) Show camera feed
                # ---------------------------------------------------
                # Convert BGR frame (OpenCV default) to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Display the current frame in the left column
                FRAME.image(frame_rgb, channels="RGB")
    else:
        # Before clicking "Start Camera", show an info box
        st.info("Click **Start Camera** to begin the live demo.")

# -------------------------------------------------------
# PAGE 2: ANALYSIS DASHBOARD
# -------------------------------------------------------
else:
    # If the user selects "Data & Model Analysis" in the sidebar,
    # we show all EDA plots and model evaluation tables.
    show_analysis_dashboard()