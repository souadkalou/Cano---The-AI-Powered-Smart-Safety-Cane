"""
yolo_detector.py

This module provides a clean wrapper around the YOLOv8 object detection model
from the Ultralytics library. Instead of interacting directly with the raw YOLO
API throughout the project, we encapsulate detection functionality in this
YoloDetector class.

Why we designed this wrapper:

    • To simplify usage inside the Streamlit app — instead of repeatedly
      configuring YOLO, we load it *once* and expose a simple `.detect()` method.

    • To make our code more readable and modular. If we ever switch YOLO models
      (e.g., yolov8s.pt or a custom model), we only change this file.

    • To control device placement (CPU vs GPU) explicitly, which is important
      for reproducibility in the student project environment.

YOLOv8 is responsible for detecting:
    • people
    • vehicles
    • traffic lights (bounding box only — classification is handled by our CNN)
    • other common objects present in urban street scenes

The output of YOLO feeds directly into:
    • the distance-alert system (using bounding-box relative size)
    • the traffic-light classifier (via cropped ROI)
"""

from ultralytics import YOLO   # Modern YOLOv8 implementation
import cv2                     # Included only for interface consistency (not required here)


class YoloDetector:
    """
    A simple abstraction layer for YOLOv8 detection.

    This class handles:
        • loading the YOLO model from a given weight file
        • moving the model to CPU/GPU
        • running predictions on a single video frame
        • returning only the first (and only) result for our use case

    The Streamlit app initializes this class *once*, and re-uses it for every
    incoming camera frame.
    """

    def __init__(self, model_path="yolov8n.pt", device="cpu"):
        """
        Constructor: we load the YOLOv8 model from the specified weight file.

        Parameters
        ----------
        model_path : str
            Path to the YOLOv8 weight file. For this project, we use 'yolov8n.pt',
            which is the nano model — chosen intentionally because it is lightweight
            and suitable for real-time CPU inference, which is required in our demo.

        device : str
            Compute device to run the model on. For our course project, we use "cpu".
            If CUDA were available, this could be "cuda" for faster inference.

        Key design note:
        We load YOLO **once** and store it inside the class instance.
        Reloading YOLO repeatedly would drastically reduce performance.
        """
        # Load YOLO model from the given weights
        self.model = YOLO(model_path)

        # Move the model to CPU or GPU (CPU in our case)
        self.model.to(device)

    def detect(self, frame, conf=0.4):
        """
        Run YOLOv8 object detection on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            A BGR image frame (from OpenCV). YOLOv8 handles preprocessing internally.

        conf : float
            Confidence threshold for retaining a detection. We set a default of 0.4,
            which balances sensitivity and precision for common traffic objects.

        Returns
        -------
        results[0] : ultralytics.engine.results.Results
            YOLO returns a list of results (one per frame). Since we only submit
            one frame at a time, we return the first element.

            The result includes:
                • bounding boxes
                • class indices
                • confidence scores
                • class names
                • segmentation masks (if applicable)
                • tracking IDs (if running tracking)

        Why we return results[0]:
        YOLOv8’s `.predict()` always returns a list, even for single images.
        Returning results[0] keeps our Streamlit app simple and consistent.
        """

        # Run YOLO prediction with:
        #   - our frame as input
        #   - selected confidence threshold
        #   - verbose=False to avoid console logs during real-time inference
        results = self.model.predict(
            source=frame,
            conf=conf,
            verbose=False
        )

        # Since only one image was provided, return the first (and only) result
        return results[0]