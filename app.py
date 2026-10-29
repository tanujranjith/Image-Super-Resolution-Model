"""Dark restoration workbench; input previews never masquerade as neural outputs."""

from functools import lru_cache
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import quote
import gradio as gr
from PIL import Image, ImageOps
from superres.common import resolve

ASSETS = resolve("assets")
SAMPLE = ASSETS / "waterfall-preview.png"


def asset_url(name):
    return "/gradio_api/file=" + quote((ASSETS / name).as_posix(), safe="/:")


def icon(name):
    return f'<img class="ui-icon" src="{asset_url(name + ".svg")}" alt="" aria-hidden="true">'


@lru_cache(maxsize=2)
def upscaler(scale):
    directory = resolve(os.getenv("SR_ARTIFACTS", "artifacts")) / f"x{scale}"
    if not (directory / "model.safetensors").is_file():
        raise FileNotFoundError(
            "Trained weights are not available yet. Training and export must be completed first."
        )
    from superres.inference import Upscaler

    return Upscaler(
        resolve(os.getenv("SR_ARTIFACTS", "artifacts")) / f"x{scale}",
        os.getenv("SR_DEVICE", "auto"),
    )


def enhance(image, scale):
    if image is None:
        raise gr.Error("Upload an image to begin.")
    scale = int(scale)
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.width * image.height * scale**2 > 16_000_000:
        raise gr.Error("Output exceeds 16 megapixels. Resize your input first.")
    start = time.perf_counter()
    try:
        result = upscaler(scale).upscale(image, tile_size=128)
        elapsed = time.perf_counter() - start
        display_input = image.resize(result.size, Image.Resampling.BICUBIC)
        cache = Path(tempfile.gettempdir()) / "local-neural-sr"
        cache.mkdir(exist_ok=True)
        for previous in cache.glob("neural-*.png"):
            if time.time() - previous.stat().st_mtime > 3600:
                previous.unlink(missing_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=cache, prefix=f"neural-x{scale}-", suffix=".png", delete=False
        ) as output:
            output_path = output.name
        result.save(output_path)
        info = f'<div class="dimension-strip"><span><strong>{image.width} × {image.height}</strong> → <strong>{result.width} × {result.height}</strong> · {scale}×</span><span>Processing time: <strong>{elapsed:.2f}s</strong></span></div>'
        return (display_input, result), result, output_path, info
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise gr.Error(str(error)) from error


def preview_input(image, scale):
    """Display only the original input twice, with no synthetic neural preview."""
    if image is None:
        return (
            gr.update(value=None, label="Upload an image to compare"),
            None,
            gr.update(value=None, interactive=False),
            '<div class="dimension-strip"><span>Input dimensions: —</span><span>Processing time: —</span></div>',
            '<p class="comparison-note">Upload an image to begin. No reconstruction has been generated.</p>',
        )
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    display = image.copy()
    display.thumbnail((1280, 960))
    info = f'<div class="dimension-strip"><span>Input: <strong>{width} × {height}</strong> · Target: <strong>{width * int(scale)} × {height * int(scale)}</strong></span><span>Processing time: —</span></div>'
    notice = (
        '<p class="comparison-note">Input preview on both sides · no neural output generated.</p>'
    )
    return (
        gr.update(value=(display, display), label="Input preview ↔ output pending"),
        None,
        gr.update(value=None, interactive=False),
        info,
        notice,
    )


def restore(image, scale):
    pair, output, path, info = enhance(image, scale)
    notice = '<p class="comparison-note">Left: bicubic reference · Right: neural reconstruction. Reconstructed detail is estimated.</p>'
    return (
        gr.update(value=pair, label="Bicubic reference ↔ neural reconstruction"),
        output,
        gr.update(value=path, interactive=True),
        info,
        notice,
    )


def sample_photo():
    with Image.open(SAMPLE) as source:
        sample = source.convert("RGB")
    sample.thumbnail((960, 640))
    return sample


def build_app():
    gr.set_static_paths(paths=[ASSETS])
    root = resolve(os.getenv("SR_ARTIFACTS", "artifacts"))
    scales = [f"{s}×" for s in (2, 4) if (root / f"x{s}/model.safetensors").exists()]
    status = (
        "RESEARCH DEMO · AVAILABLE: " + ", ".join(scales)
        if scales
        else "IMPLEMENTATION PREVIEW · MODEL NOT TRAINED"
    )
    note = (
        "Reconstruction estimates detail; it does not recover ground truth."
        if scales
        else "Results will be available after approved training and export."
    )
    sample = sample_photo() if SAMPLE.exists() else None
    initial = preview_input(sample, 4)
    css = (
        (ASSETS / "interface.css")
        .read_text(encoding="utf-8")
        .replace("__FONT_URL__", asset_url("Inter.ttf"))
    )
    with gr.Blocks(
        title="Image Super-Resolution",
        css=css,
        theme=gr.themes.Base(font=["Inter", "system-ui", "sans-serif"]),
        delete_cache=(3600, 3600),
    ) as demo:
        gr.HTML(
            f'<header class="app-header"><div class="brand">{icon("photo")}<span>Image Super-Resolution</span><small>Research preview</small></div><nav aria-label="Page navigation"><a href="#about-model">How it works</a><a href="#full-output">Full output</a></nav></header>'
        )
        gr.HTML(f'<div class="status-strip"><strong>{status}</strong><span>{note}</span></div>')
        gr.HTML(
            '<div class="page-heading"><h1>See the difference, pixel by pixel.</h1><p>Upload an image and compare a learned 2× or 4× reconstruction with its bicubic reference.</p></div>'
        )
        with gr.Row(elem_id="sr-workspace", equal_height=False):
            with gr.Column(scale=1, min_width=240, elem_id="tools-panel"):
                gr.HTML('<h2 class="tool-heading">Input image</h2>')
                photo = gr.Image(
                    value=sample,
                    type="pil",
                    label="Input image",
                    show_label=False,
                    height=180,
                    sources=["upload"],
                    show_download_button=False,
                    show_share_button=False,
                    elem_id="input-image",
                )
                use_sample = gr.Button(
                    "Use sample image",
                    icon=str(ASSETS / "photo.svg"),
                    size="sm",
                    elem_id="sample-photo",
                )
                scale = gr.Radio(
                    choices=[("2×", 2), ("4×", 4)],
                    value=4,
                    label="Upscale factor",
                    elem_id="scale-control",
                )
                button = gr.Button(
                    "Restore image",
                    icon=str(ASSETS / "upload.svg"),
                    variant="primary",
                    elem_id="restore-button",
                )
                download = gr.DownloadButton(
                    "Download PNG",
                    icon=str(ASSETS / "download.svg"),
                    interactive=False,
                    elem_id="download-button",
                )
                gr.HTML(
                    '<p class="panel-footnote">AI-generated sample image. Upload your own photo using the image toolbar.</p>'
                )
            with gr.Column(scale=4, min_width=320, elem_id="comparison-column"):
                slider = gr.ImageSlider(
                    value=initial[0]["value"],
                    label=initial[0]["label"],
                    type="pil",
                    height=640,
                    show_download_button=False,
                    elem_id="comparison-slider",
                )
                notice = gr.HTML(initial[4], elem_id="comparison-notice")
        info = gr.HTML(initial[3], elem_id="image-info")
        with gr.Accordion("Full neural output", open=False, elem_id="full-output"):
            preview = gr.Image(type="pil", label="Neural reconstruction", show_share_button=False)
        gr.HTML(
            '<section class="about-section" id="about-model"><h2>More room for detail.</h2><p>Separate learned models reconstruct images at 2× and 4×. Once weights exist, drag the divider to compare a bicubic display reference with the neural result, then download a lossless PNG. Until then, both sides show the same original input.</p><p>Textures, text, faces and license plates may be inaccurate. Output is limited to 16 megapixels. No training or weight download starts from this interface. Temporary downloads expire after one hour.</p></section>'
        )
        outputs = [slider, preview, download, info, notice]
        button.click(restore, [photo, scale], outputs, concurrency_limit=1)
        photo.change(preview_input, [photo, scale], outputs, queue=False)
        scale.change(preview_input, [photo, scale], outputs, queue=False)
        use_sample.click(sample_photo, outputs=photo, queue=False)
    return demo


if __name__ == "__main__":
    build_app().queue(max_size=8).launch(
        server_name=os.getenv(
            "GRADIO_SERVER_NAME", "0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1"
        ),
        server_port=int(os.getenv("PORT", "7861")),
        show_error=False,
    )
