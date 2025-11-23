"""
midas_estimator.py

In this module, we wrap the MiDaS depth-estimation model in a small, clean
Python class that we can easily use inside our main Streamlit application.

Why we need this:

    • Our project requires a monocular depth-estimation component to approximate
      the relative distance of obstacles from a single RGB frame.

    • Rather than implementing a depth model from scratch, we leverage the
      pre-trained MiDaS models provided by the Intel ISL team via `torch.hub`.

    • We encapsulate all of the model loading, device configuration, and
      input preprocessing inside a single class (MidasDepthEstimator), so that
      the rest of the codebase can call `estimate(frame)` without worrying
      about model internals.

Design decisions:

    • We default to "MiDaS_small" for speed, since we are running on CPU and
      need near real-time performance during the live demo.

    • We automatically select the appropriate transform function depending on
      which MiDaS variant is loaded ("DPT_Large", "DPT_Hybrid", or "MiDaS_small").

    • The output `depth_map` is a 2D NumPy array with relative depth values.
      Although these values are not calibrated to real-world meters, they are
      sufficient to detect “near vs far” obstacles.
"""

import cv2      # OpenCV for color space conversion (BGR → RGB)
import torch    # PyTorch for loading and running the MiDaS model
import numpy as np  # NumPy for handling the depth map as an array


class MidasDepthEstimator:
    """
    This class provides a simple interface for performing monocular depth
    estimation using one of the MiDaS models.

    Usage pattern inside the project:
        • We create a single instance of this class (cached in Streamlit).
        • For each camera frame, we call `estimate(frame_bgr)` to obtain a
          depth map.
        • We then use this depth map to reason about proximity of detected
          objects, and to support audio-based distance alerts.
    """

    def __init__(self, device: str = "cpu"):
        """
        Constructor: we initialize and configure the MiDaS model here.

        Parameters
        ----------
        device : str
            The compute device we want to use, typically "cpu" in this project.
            If a GPU is available and configured, this could be set to "cuda"
            to accelerate inference.

        Steps we perform:
            1. Store the device setting.
            2. Choose which MiDaS variant to load ("MiDaS_small" by default).
            3. Use torch.hub to load the pre-trained model weights.
            4. Move the model to the correct device and set it to eval() mode.
            5. Load the corresponding input transform pipeline.
        """

        # Store the target device ("cpu" for this project)
        self.device = device

        # ------------------------------------------------------------------
        # MiDaS model selection
        # Available options include:
        #   "DPT_Large"  : highest quality, but slowest (not ideal for real-time)
        #   "DPT_Hybrid" : intermediate quality/speed
        #   "MiDaS_small": fastest model, suitable for live demos on CPU
        #
        # For our use case (real-time guidance on a laptop), we prioritize
        # speed over perfect depth accuracy, so we select "MiDaS_small".
        # ------------------------------------------------------------------
        self.model_type = "MiDaS_small"

        # ------------------------------------------------------------------
        # Load the MiDaS model via torch.hub.
        #
        # We use the official repository: "intel-isl/MiDaS".
        # torch.hub.load() automatically downloads the pre-trained weights
        # if they are not already cached on the local machine.
        # ------------------------------------------------------------------
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)

        # Move the model to the specified device (CPU or GPU).
        self.model.to(self.device)

        # Set the model to evaluation mode to disable dropout, etc.
        self.model.eval()

        # ------------------------------------------------------------------
        # Load the corresponding transformation functions for MiDaS.
        #
        # The transforms handle tasks such as:
        #   • resizing the input image
        #   • normalizing pixel values
        #   • converting NumPy arrays / PIL images into PyTorch tensors
        #
        # MiDaS provides specific transforms depending on whether we are
        # using a DPT model or a small model.
        # ------------------------------------------------------------------
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        # If we choose a DPT-based model, we use the DPT-specific transform
        if "DPT" in self.model_type:
            self.transform = midas_transforms.dpt_transform
        else:
            # Otherwise, we select the transformation pipeline for small models
            self.transform = midas_transforms.small_transform

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Estimate depth from a single BGR image frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            A BGR image as returned by OpenCV (shape H x W x 3).

        Returns
        -------
        depth_map : np.ndarray
            A 2D NumPy array (H x W) where each element represents
            a relative depth value. Larger values are either closer
            or farther depending on MiDaS scaling, but for our use,
            we mainly care about relative comparisons (e.g., nearest point).

        Processing steps:
            1. Convert the frame from BGR (OpenCV default) to RGB.
            2. Apply the MiDaS input transform to normalize and resize.
            3. Run the model in no-grad mode to get the raw depth prediction.
            4. Resize (interpolate) the predicted depth back to original
               image resolution.
            5. Move the prediction to CPU and convert it into a NumPy array.
        """

        # ------------------------------------------------------------------
        # 1) Convert BGR image (OpenCV) to RGB format.
        #
        # MiDaS (and most PyTorch-based vision models) expects images in RGB
        # channel order, whereas OpenCV gives us BGR by default.
        # ------------------------------------------------------------------
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ------------------------------------------------------------------
        # 2) Apply MiDaS transform to prepare the input for the model.
        #
        # The transform returns a tensor of the correct size and normalization.
        # We then move this tensor to the correct device (CPU or GPU).
        # ------------------------------------------------------------------
        input_batch = self.transform(img_rgb).to(self.device)

        # We disable gradient tracking since we are only performing inference,
        # not training the model. This reduces memory usage and speeds up
        # computation.
        with torch.no_grad():
            # ------------------------------------------------------------------
            # 3) Run the model to obtain a raw depth prediction.
            #
            # The output is typically a 2D map with lower resolution than
            # the input image, so we will upsample it in the next step.
            # ------------------------------------------------------------------
            prediction = self.model(input_batch)

            # ------------------------------------------------------------------
            # 4) Interpolate the low-resolution depth map back to the original
            #    input image size (height x width).
            #
            # We:
            #   • Insert a channel dimension using unsqueeze(1)
            #   • Use bicubic interpolation for smoother depth maps
            #   • Disable align_corners to avoid artifacts
            #   • Remove the added dimension with squeeze()
            # ------------------------------------------------------------------
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],  # (height, width) of original image
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # ------------------------------------------------------------------
        # 5) Convert the depth map to a NumPy array on CPU.
        #
        # This format is easier to manipulate with standard numerical tools
        # (NumPy, OpenCV) and to integrate with the rest of our pipeline.
        # ------------------------------------------------------------------
        depth_map = prediction.cpu().numpy()

        # The resulting depth_map is a dense 2D array of floats. We use
        # this in the main app to infer which regions of the frame are closer
        # and trigger warnings accordingly.
        return depth_map