"""A changed input or scale invalidates old restoration/download state."""

import numpy as np
from PIL import Image

from app import preview_input


def test_input_preview_never_fakes_reconstruction_and_clears_download():
    image = Image.fromarray(np.random.default_rng(2).integers(0, 256, (64, 80, 3), dtype=np.uint8))
    slider, output, download, dimensions, notice = preview_input(image, 4)
    left, right = slider["value"]
    assert np.array_equal(left, right) and np.array_equal(left, image)
    assert output is None and download["value"] is None and not download["interactive"]
    assert "320 × 256" in dimensions and "Processing time: —" in dimensions
    assert "no neural output generated" in notice


def test_empty_input_clears_comparison_and_large_preview_is_bounded():
    slider, output, download, _, _ = preview_input(None, 2)
    assert slider["value"] is None and output is None and not download["interactive"]
    slider, _, _, _, _ = preview_input(Image.new("RGB", (1500, 1000)), 2)
    assert slider["value"][0].width <= 1280
