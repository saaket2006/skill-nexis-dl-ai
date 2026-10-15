"""
Preprocessing Pipeline for MNIST Digit Recognition
===================================================
Prepares raw images (drawn canvas, uploaded files, or dataset samples)
into the exact format required by the MNIST-trained model:
- Shape: (1, 28, 28)
- Type: float32
- Range: [0.0, 1.0]
- Background: 0.0 (black)
- Foreground: 1.0 (white digit)
- Centering: Center of Mass aligned to (14.0, 14.0)
"""

from typing import Tuple, Optional
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, shift


def preprocess_digit_image(
    pil_image: Image.Image,
    target_box_size: int = 20,
    canvas_size: int = 28,
    ink_threshold: float = 25.0
) -> Tuple[Optional[np.ndarray], np.ndarray, bool]:
    """
    Transforms any PIL image into a standard MNIST-compatible (1, 28, 28) array.

    Processing Steps:
    1. Transparency handling: composites RGBA/LA images onto a clean white background.
    2. Grayscale conversion: converts to single-channel luminance.
    3. Adaptive polarity inversion: converts black-ink-on-white-canvas into
       white-digit-on-black-background (MNIST convention) while preserving images
       that are already dark-background.
    4. Ink validation: detects empty or near-empty inputs.
    5. Bounding box cropping: isolates the active digit region.
    6. Aspect-ratio preserving resize: scales the digit to fit inside target_box_size (20x20)
       using Lanczos interpolation.
    7. Initial placement: places the 20x20 crop into a 28x28 canvas.
    8. Center-of-mass alignment: calculates the center of mass of the active pixels and translates
       the image so that the center of mass is placed at (14.0, 14.0) — exactly matching
       Yann LeCun's official MNIST normalization algorithm.
    9. Range normalization: clips values to [0.0, 255.0] and scales to [0.0, 1.0].

    Returns:
        tensor_input: np.ndarray of shape (1, 28, 28) and dtype float32, or None if blank.
        debug_canvas: np.ndarray of shape (28, 28) and dtype float32 for UI visualization.
        is_blank: bool indicating whether the input was blank / no ink detected.
    """
    empty_canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)

    if pil_image is None:
        return None, empty_canvas, True

    # Step 1: Transparency / Alpha Handling
    if pil_image.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", pil_image.size, (255, 255, 255))
        alpha = pil_image.split()[-1]
        bg.paste(pil_image, mask=alpha)
        pil_image = bg
    elif pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Step 2: Grayscale Conversion
    gray = pil_image.convert("L")
    arr = np.array(gray, dtype=np.float32)

    # Step 3: Adaptive Polarity Detection
    # If average luminance > 127, background is white/light -> invert so background is dark
    if arr.mean() > 127.0:
        arr = 255.0 - arr

    # Step 4: Ink Validation (blank canvas check)
    if arr.max() < ink_threshold:
        return None, empty_canvas, True

    # Step 5: Bounding Box Detection
    rows = np.any(arr > ink_threshold, axis=1)
    cols = np.any(arr > ink_threshold, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None, empty_canvas, True

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = arr[rmin:rmax + 1, cmin:cmax + 1]

    # Step 6: Aspect-Preserving Resize into target_box_size (20x20)
    h, w = cropped.shape
    if h <= 0 or w <= 0:
        return None, empty_canvas, True

    scale = float(target_box_size) / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    cr_pil = Image.fromarray(np.clip(cropped, 0, 255).astype(np.uint8))
    rs_pil = cr_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized = np.array(rs_pil, dtype=np.float32)

    # Step 7: Initial Bounding-Box Centering on 28x28 Frame
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    top = (canvas_size - new_h) // 2
    left = (canvas_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    # Step 8: Center of Mass Translation to (14.0, 14.0)
    # This matches the NIST-to-MNIST normalization standard (LeCun et al.)
    total_mass = np.sum(canvas)
    if total_mass > 1e-4:
        cy, cx = center_of_mass(canvas)
        if not np.isnan(cy) and not np.isnan(cx):
            target_center = canvas_size / 2.0  # 14.0 for 28x28
            shift_y = target_center - cy
            shift_x = target_center - cx
            # Order 1 (bilinear) maintains antialiasing without artificial ripples
            canvas = shift(canvas, (shift_y, shift_x), order=1, mode="constant", cval=0.0)

    # Step 9: Normalization to [0.0, 1.0]
    canvas = np.clip(canvas, 0.0, 255.0) / 255.0
    tensor_input = canvas.reshape(1, canvas_size, canvas_size).astype(np.float32)

    return tensor_input, canvas, False
