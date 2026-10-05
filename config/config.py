"""Shared project configuration for the recycling detector."""

CLASS_NAMES = ("plastic", "glass", "paper", "metal", "cardboard")
NUM_CLASSES = len(CLASS_NAMES)
DATA_CONFIG_PATH = "dataset/data.yaml"
MODEL_WEIGHTS_PATH = "models/best.pt"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
