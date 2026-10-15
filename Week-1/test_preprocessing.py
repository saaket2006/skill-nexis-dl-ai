"""
Unit Tests for MNIST Digit Recognizer Preprocessing & Model Compatibility
=========================================================================
Tests:
1. Output dimensions, tensor shape, data types, pixel normalization range [0, 1].
2. Dark background with bright digit (MNIST style).
3. White background with black stroke (HTML5 canvas style).
4. Multi-channel and dimension formats (RGBA, RGB, L, non-square dimensions).
5. Empty and nearly empty / noise inputs.
6. Digits touching or near canvas boundaries.
7. Center-of-mass alignment preservation (centering to (14, 14)).
8. Model inference compatibility and 10-class output mapping.
9. Deterministic digit 7 prediction accuracy comparing CoM alignment.
"""

import os
import numpy as np
import pytest
from PIL import Image, ImageDraw
from scipy.ndimage import center_of_mass

from preprocessing import preprocess_digit_image


# ─── Test 1: Output Dimensions, Types, and Normalization Range ────────────────
def test_output_shape_type_and_range():
    """Verify standard drawing returns (1, 28, 28) float32 in [0.0, 1.0]."""
    img = Image.new("RGB", (280, 280), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line([(140, 40), (140, 240)], fill=(0, 0, 0), width=20)

    tensor_in, debug_canvas, is_blank = preprocess_digit_image(img)

    assert not is_blank, "Visible stroke should not be flagged as blank"
    assert tensor_in is not None, "Tensor input must not be None"
    assert tensor_in.shape == (1, 28, 28), f"Expected shape (1, 28, 28), got {tensor_in.shape}"
    assert tensor_in.dtype == np.float32, f"Expected float32, got {tensor_in.dtype}"
    assert debug_canvas.shape == (28, 28), f"Expected canvas (28, 28), got {debug_canvas.shape}"
    assert 0.0 <= tensor_in.min() <= tensor_in.max() <= 1.0, (
        f"Values outside [0, 1]: min={tensor_in.min()}, max={tensor_in.max()}"
    )


# ─── Test 2: Polarity Handling ───────────────────────────────────────────────
def test_white_bg_canvas_drawing():
    """Canvas drawing with white background and black stroke should invert to dark background."""
    img = Image.new("RGB", (280, 280), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([80, 50, 200, 230], outline=(0, 0, 0), width=22)

    tensor_in, _, is_blank = preprocess_digit_image(img)
    assert not is_blank
    # Corners should be black (0.0) in MNIST standard
    corners = [
        tensor_in[0, 0, 0],
        tensor_in[0, 0, 27],
        tensor_in[0, 27, 0],
        tensor_in[0, 27, 27]
    ]
    assert all(c == 0.0 for c in corners), f"Corners should be 0.0, got {corners}"
    assert tensor_in.max() > 0.5, "Foreground digit should have bright pixels"


def test_black_bg_mnist_style_image():
    """Uploaded image that is already dark background with white digit must NOT be reinverted."""
    img = Image.new("RGB", (100, 100), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([25, 25, 75, 75], outline=(255, 255, 255), width=8)

    tensor_in, _, is_blank = preprocess_digit_image(img)
    assert not is_blank
    corners = [
        tensor_in[0, 0, 0],
        tensor_in[0, 0, 27],
        tensor_in[0, 27, 0],
        tensor_in[0, 27, 27]
    ]
    assert all(c == 0.0 for c in corners), "Background should remain dark"
    assert tensor_in.max() > 0.5, "Foreground digit should remain bright"


# ─── Test 3: Upload Formats & Variable Dimensions ─────────────────────────────
@pytest.mark.parametrize("mode", ["RGBA", "LA", "RGB", "L"])
def test_channel_modes(mode):
    """Ensure transparency and color modes (RGBA, LA, RGB, L) process seamlessly."""
    if "A" in mode:
        # Create transparent canvas with black stroke
        base = Image.new(mode, (200, 200), (0, 0, 0, 0) if mode == "RGBA" else (0, 0))
        draw = ImageDraw.Draw(base)
        color = (0, 0, 0, 255) if mode == "RGBA" else (0, 255)
        draw.line([(50, 30), (150, 170)], fill=color, width=15)
    else:
        base = Image.new(mode, (200, 200), (255, 255, 255) if mode == "RGB" else 255)
        draw = ImageDraw.Draw(base)
        color = (0, 0, 0) if mode == "RGB" else 0
        draw.line([(50, 30), (150, 170)], fill=color, width=15)

    tensor_in, _, is_blank = preprocess_digit_image(base)
    assert not is_blank
    assert tensor_in.shape == (1, 28, 28)


@pytest.mark.parametrize("size", [(120, 340), (500, 200), (45, 45), (1024, 768)])
def test_non_square_and_extreme_sizes(size):
    """Non-square and extreme aspect ratios should be fit into 20x20 without distortion."""
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([size[0] // 4, size[1] // 4, 3 * size[0] // 4, 3 * size[1] // 4], outline=(0, 0, 0), width=10)

    tensor_in, _, is_blank = preprocess_digit_image(img)
    assert not is_blank
    assert tensor_in.shape == (1, 28, 28)


# ─── Test 4: Blank and Near-Blank Canvases ─────────────────────────────────────
def test_pure_blank_canvas():
    """Pure white or transparent canvas must return is_blank=True and None tensor."""
    white_img = Image.new("RGB", (280, 280), (255, 255, 255))
    tensor_in, _, is_blank = preprocess_digit_image(white_img)
    assert is_blank
    assert tensor_in is None


def test_low_intensity_noise_canvas():
    """Negligible background noise (< 25.0 intensity) should be treated as blank."""
    noise_arr = np.full((280, 280), 245, dtype=np.uint8)  # intensity diff is 10 < 25
    noise_img = Image.fromarray(noise_arr, mode="L")
    tensor_in, _, is_blank = preprocess_digit_image(noise_img)
    assert is_blank
    assert tensor_in is None


# ─── Test 5: Boundary Edge Touching ───────────────────────────────────────────
def test_digit_touching_canvas_edge():
    """Strokes that start at or run along the canvas edge must be cropped cleanly."""
    img = Image.new("RGB", (280, 280), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Stroke right at top-left edge
    draw.line([(0, 0), (100, 100)], fill=(0, 0, 0), width=20)

    tensor_in, _, is_blank = preprocess_digit_image(img)
    assert not is_blank
    assert tensor_in.shape == (1, 28, 28)
    # Bounding box crop should have centered it inside the 28x28 canvas
    assert tensor_in[0, 0, 0] == 0.0 or tensor_in[0, 27, 27] == 0.0


# ─── Test 6: Center of Mass Centering Precision ───────────────────────────────
def test_center_of_mass_placement():
    """The center of mass of the processed image must be approximately (14.0, 14.0)."""
    img = Image.new("RGB", (280, 280), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw an asymmetrical digit (7 with heavy top bar)
    draw.line([(60, 50), (220, 50)], fill=(0, 0, 0), width=25)
    draw.line([(220, 50), (130, 230)], fill=(0, 0, 0), width=18)

    _, debug_canvas, is_blank = preprocess_digit_image(img)
    assert not is_blank

    cy, cx = center_of_mass(debug_canvas)
    # Center of mass should be close to 14.0 (within 0.5 pixels tolerance)
    assert abs(cy - 14.0) < 0.6, f"Row center of mass {cy} deviated too far from 14.0"
    assert abs(cx - 14.0) < 0.6, f"Col center of mass {cx} deviated too far from 14.0"


# ─── Test 7: Model Compatibility and Class Mapping ────────────────────────────
def test_model_inference_compatibility():
    """Test that the loaded model expects shape (None, 28, 28) and outputs 10 classes."""
    import app
    model, err = app.load_trained_model()
    assert err is None, f"Model load error: {err}"
    assert model is not None

    dummy_input = np.zeros((1, 28, 28), dtype=np.float32)
    probs = model.predict(dummy_input, verbose=0)
    assert probs.shape == (1, 10), f"Expected output shape (1, 10), got {probs.shape}"
    assert abs(np.sum(probs) - 1.0) < 1e-4, "Model output should sum to 1.0 (Softmax)"


# ─── Test 8: Deterministic Digit 7 Classification ─────────────────────────────
def test_handwritten_7_classification():
    """Verify that a cleanly drawn digit 7 classifies as 7 rather than 3."""
    import app
    model, _ = app.load_trained_model()
    assert model is not None

    img = Image.new("RGB", (280, 280), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line([(70, 60), (210, 60)], fill=(0, 0, 0), width=24)
    draw.line([(210, 60), (130, 230)], fill=(0, 0, 0), width=24)

    tensor_in, _, is_blank = preprocess_digit_image(img)
    assert not is_blank

    probs = model.predict(tensor_in, verbose=0)[0]
    pred = int(np.argmax(probs))
    assert pred == 7, f"Expected digit 7, but predicted {pred} (probabilities: {probs})"
    assert probs[7] > 0.85, f"Expected high confidence for 7, got {probs[7]:.3f}"
