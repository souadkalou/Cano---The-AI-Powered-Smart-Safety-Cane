"""
traffic_light_model.py

This module defines the convolutional neural network (CNN) we designed for
the binary classification task in our project: distinguishing between
RED and GREEN traffic light states.

Why we built our own CNN instead of using a pre-trained model:

    • Our cropped traffic-light images are small (224×224) and highly specific.
    • We only need to classify **2 classes**, not the 1000 classes in ImageNet.
    • A lightweight model trains faster, avoids overfitting, and runs smoothly on CPU.
    • Our dataset (LISA crops) is clean and does not require a heavy architecture.

The architecture below is intentionally simple and efficient:
    • Two convolutional blocks
    • MaxPooling for spatial downsampling
    • Dropout to reduce overfitting
    • A fully connected layer for feature projection
    • A final classification layer for RED/GREEN prediction

This model achieved very high accuracy (>97%) on our processed dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrafficLightCNN(nn.Module):
    """
    We implemented a compact CNN specifically tailored for the small and
    visually simple traffic-light crops extracted from the LISA dataset.

    The goal of this architecture is NOT to compete with very deep networks
    like ResNet — instead, we aim for:
        • speed (runs in real-time on CPU)
        • simplicity (easy to train and debug)
        • strong performance on a narrow, well-defined task
    """

    def __init__(self, num_classes: int = 2):
        """
        Constructor for the CNN.

        Parameters
        ----------
        num_classes : int
            Number of output classes. For our project:
                0 → red
                1 → green

        Architecture overview:
            Input: 3-channel RGB image, 224×224
            Conv1 → ReLU → Pool → Conv2 → ReLU → Pool → Dropout → FC layers
        """
        super().__init__()

        # ------------------------------------------------------------------
        # Convolution Block 1:
        #   3 input channels (RGB)
        #   32 output feature maps
        #   kernel size = 3 (small conv for spatial details)
        # padding=1 preserves spatial resolution
        #
        # Output size remains 224×224 before pooling.
        # ------------------------------------------------------------------
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        # ------------------------------------------------------------------
        # Convolution Block 2:
        #   32 → 64 feature maps
        #   Same kernel size and padding
        #
        # Helps extract higher-level features after first pooling layer.
        # Output before pooling: 112×112
        # ------------------------------------------------------------------
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Max pooling layer used after each convolution block:
        #   reduces spatial size by 50%
        #   provides translation invariance
        #   reduces compute cost significantly
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout layer:
        #   reduces overfitting by randomly masking neurons during training
        #   especially helpful since LISA crops are highly repetitive
        self.dropout = nn.Dropout(0.3)

        # ------------------------------------------------------------------
        # After two pooling operations:
        #   224×224 → 112×112 → 56×56
        #
        # At this point:
        #   • channels = 64
        #   • height = 56
        #   • width = 56
        #
        # Total flattened features = 64 * 56 * 56
        # ------------------------------------------------------------------
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # dense projection layer

        # Final classification layer:
        # Output = num_classes (2 for red/green)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        This is the computation graph defining how an input image tensor
        flows through the layers to produce a prediction.

        Steps:
            1. Convolution → ReLU → Pool
            2. Convolution → ReLU → Pool
            3. Dropout
            4. Flatten
            5. Fully connected → ReLU
            6. Dropout
            7. Output layer (logits)
        """

        # First conv block: extract low-level edges & colors
        x = F.relu(self.conv1(x))
        x = self.pool(x)   # 224 → 112

        # Second conv block: extract more abstract features
        x = F.relu(self.conv2(x))
        x = self.pool(x)   # 112 → 56

        # Dropout before flattening to reduce overfitting
        x = self.dropout(x)

        # Flatten from (batch, channels, H, W) to (batch, features)
        x = x.view(x.size(0), -1)

        # Fully connected projection layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Second dropout for additional regularization
        x = self.dropout(x)

        # Final linear layer produces logits for each class
        x = self.fc2(x)

        return x