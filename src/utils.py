"""Input validation and small presentation helpers."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_TYPES = {"jpg", "jpeg", "png", "webp"}
SUPPORTED_VIDEO_TYPES = {"mp4", "avi", "mov"}
# Preserve Pillow's decoder before Ultralytics optionally patches Image.open.
_PIL_IMAGE_OPEN = Image.open


class InputValidationError(ValueError):
    """Raised when user-provided media cannot be opened safely."""


def load_uploaded_image(data: bytes) -> Image.Image:
    """Decode and fully load an uploaded image, returning an RGB copy."""
    if not data:
        raise InputValidationError("The uploaded image is empty.")
    try:
        with _PIL_IMAGE_OPEN(BytesIO(data)) as image:
            image.verify()
        with _PIL_IMAGE_OPEN(BytesIO(data)) as image:
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InputValidationError("This file is not a readable JPG, JPEG, PNG, or WEBP image.") from exc
