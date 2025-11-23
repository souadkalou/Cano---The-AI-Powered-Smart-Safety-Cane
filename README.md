## VisionGuide – Smart Safety Cane
AI-Powered Obstacle Detection, Distance Estimation & Traffic Light Recognition  
**CS316 – Machine Learning Project**

---------

## **1. Project Overview**

VisionGuide is an intelligent assistive system designed to support visually impaired users during urban navigation.  
The system integrates:

- **YOLOv8** for real-time obstacle & vehicle detection  
- **MiDaS** for monocular depth estimation  
- **A custom CNN classifier** trained on the **LISA Traffic Light Dataset**  
- **Text-to-speech** feedback for safe, hands-free navigation  

The application is implemented in **Python**, with an interactive UI built using **Streamlit**, and includes a full pipeline for data preprocessing, model training, evaluation, and live inference.

---------

## **2. Project Structure**

cano-smart-cane/
│
├── src/
│   ├── app/
│   │   └── streamlit_app.py        # Main UI application (live demo + analysis)
│   ├── models/
│   │   ├── yolov8n.pt              # YOLO model weights
│   │   ├── midas_estimator.py      # Depth model wrapper
│   │   ├── yolo_detector.py        # YOLO wrapper class
│   │   └── traffic_light_model.py  # CNN classifier architecture
│   ├── training/
│   │   └── train_traffic_light.py  # Training script for CNN
│   └── data/
│       ├── prepare_lisa.py         # Preprocessing script for LISA dataset
│
├── analysis/
│   ├── eda_traffic_lights.py       # Full dataset exploratory analysis
│   ├── evaluate_traffic_model.py   # Metrics, ROC, confusion matrix
│   └── regression_clustering_tasks.py  # ML tasks (regression + clustering)
│
├── models/
│   └── traffic_light_classifier/
│       └── best_traffic_light.pt
│
├── data/
│   └── processed/
│       └── traffic_lights/
│           ├── red/
│           └── green/
│
├── analysis/plots/                 # All EDA visualizations
│
├── REPORT.md                       # (This file)
└── requirements.txt

---------

## **3. Project Objectives**

### ✔️ **1. Traffic Light Recognition**  
Train & deploy a CNN that classifies traffic lights as **RED** or **GREEN** with >97% accuracy.

### ✔️ **2. Obstacle & Vehicle Detection**  
Integrate YOLOv8 to recognize:
- people  
- cars  
- trucks  
- buses  
- motorcycles  
- miscellaneous obstacles

### ✔️ **3. Distance Estimation**  
Use MiDaS depth estimation + bounding box heuristics to produce:
- **SAFE**
- **WARNING**
- **DANGER**

### ✔️ **4. System Integration**  
A unified UI where the system:
- detects objects  
- estimates distance  
- recognizes traffic lights  
- speaks audio alerts  
- displays EDA and results  

---------

## **4. Datasets Used**

### **LISA Traffic Light Dataset (Kaggle)**  
Used to extract & train the RED/GREEN classifier  
Processed into over **106,000** cropped samples.

### **COCO Dataset**  
YOLOv8 pretrained model for detection:
- person  
- car  
- bus  
- motorcycle  
- truck  
- traffic light  

### **MiDaS Pretraining Datasets**  
Trained on 12 multi-dataset mixtures including:
- NYU Depth V2  
- ReDWeb  
- MegaDepth  

---------

## **5. How to Run the Project**

  ### **Step 1: Clone the Repository**
    git clone 
    cd cano-smart-cane

  ### **Step 2: Create Virtual Environment**
    python3 -m venv .venv
    source .venv/bin/activate

  ### **Step 3: Install Dependencies**
    pip install -r requirements.txt

  ### **Step 4: Run Streamlit App**
    streamlit run src/app/streamlit_app.py

---------    

## **6. How the System Works**

### **YOLOv8 – Obstacle Detection**
- Detects vehicles, people, objects  
- Returns bounding boxes and labels  
- Used for proximity estimation  

### **MiDaS – Depth Estimation**
- Produces depth map  
- Helps determine near/far obstacles  

### **CNN – Traffic Light Recognition**
- Trained on cropped LISA dataset  
- Output: RED vs GREEN  
- Accuracy ~ **97–99%**  

### **Audio Alerts**
- “Danger! Obstacle very close.”  
- “Warning. Obstacle ahead.”  
- “Safe to cross.” / “Do not cross.”  

---------

## **7. Exploratory Data Analysis (EDA)**

Scripts generate:
- Brightness histogram  
- Class distributions  
- Boxplots & violin plots  
- RGB averages  
- Correlation matrices  
- Pairplots  
- Trends over dataset  

All plots are visible inside **Streamlit → Data & Model Analysis**.

---------

## **8. Model Evaluation**

### **Traffic Light CNN**
| Metric | Score |
|--------|-------|
| **Accuracy** | 0.997 |
| **Precision** | 0.996 |
| **Recall** | 0.998 |
| **F1 Score** | 0.997 |
| **ROC AUC** | 0.9997 |

### **Baselines**
- Random guess baseline: **≈50%**
- Logistic regression on brightness: **≈53%**

### **Regression Tasks**
Predict brightness (numerical):
- MAE ≈ 0.017  
- RMSE ≈ 0.13  
- R² ≈ 0.95  

### **Clustering**
KMeans (k=3) based on:
- brightness  
- RGB means  

Produces meaningful groupings of lighting conditions.

---------

## **9. Features of the Final Application**

### **Mode 1 – Live Demo**
- Camera feed  
- Real-time detection  
- Distance alerts  
- Traffic-light recognition  
- Audio messages  

### **Mode 2 – Data & Model Analysis**
- Model comparison tables  
- Confusion matrix & ROC  
- Regression results  
- Clustering  
- All EDA plots  

---------

## **10. Requirements**

Main libraries:
- Ultralytics (YOLOv8)  
- PyTorch  
- scikit-learn  
- Streamlit  
- OpenCV  
- Matplotlib & Seaborn  

Full list in `requirements.txt`.

---------

## **11. Team Contributions**

- Dataset creation & preprocessing  
- CNN training  
- EDA & ML tasks  
- UI development  
- Full system integration  
- Final documentation & presentation  

---------

## **12.Optimization & Memory Efficiency**

Several optimizations were implemented to ensure the system runs efficiently in real time on CPU-only devices:

1. **Streamlit model caching**  
   All heavy models (YOLOv8, MiDaS, TrafficLightCNN) are loaded only once using `@st.cache_resource`.

2. **Single MiDaS computation per frame**  
   Depth estimation is computed once instead of per object, significantly reducing CPU load.

3. **Bounding box size heuristics**  
   Distance detection uses lightweight geometric features rather than expensive per-object depth predictions.

4. **Optimized traffic-light classification pipeline**  
   CNN classification is performed only when YOLO detects a traffic-light box.

5. **Reduced redundant TTS calls**  
   Audio messages are triggered only when the alert state changes, avoiding CPU spikes.

6. **Efficient dataset preprocessing**  
   A global image index avoids repeated disk scanning, improving speed and reducing IO overhead.

These optimizations allow the entire system to run smoothly in real-time on a standard laptop CPU.

---------

## **13. Conclusion**

Cano successfully integrates three advanced AI models into a unified, fully functional assistive technology system.  
The project satisfies **all CS316 requirements**, including:

- Dataset preprocessing  
- EDA  
- Multiple ML tasks  
- Model evaluation  
- System integration  
- Full documentation  

This README serves as the official documentation and run guide.
