"""CPU-output tiled inference bounds GPU activations with a receptive-field halo."""

from PIL import ImageOps
from safetensors.torch import load_file
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from .common import choose_device, digest, read_json, resolve
from .models import build_model


@torch.inference_mode()
def tiled(model, image, device, tile_size=128, halo=None):
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3 or tile_size < 1:
        raise ValueError("Expected one RGB BCHW image and positive tile_size")
    scale = model.scale
    halo = model.required_halo if halo is None else halo
    if halo < model.required_halo:
        raise ValueError(f"Halo must be at least {model.required_halo} LR pixels")
    h, w = image.shape[-2:]
    if h * w * scale**2 > 16_000_000:
        raise ValueError("Output would exceed 16 megapixels; use a smaller input")
    result = torch.empty((1, 3, h * scale, w * scale), dtype=torch.float32, device="cpu")
    model.eval()
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y1, x1 = min(y + tile_size, h), min(x + tile_size, w)
            top, left, bottom, right = (
                max(0, y - halo),
                max(0, x - halo),
                min(h, y1 + halo),
                min(w, x1 + halo),
            )
            patch = image[..., top:bottom, left:right].to(device)
            prediction = model(patch).float().cpu()
            result[..., y * scale : y1 * scale, x * scale : x1 * scale] = prediction[
                ...,
                (y - top) * scale : (y1 - top) * scale,
                (x - left) * scale : (x1 - left) * scale,
            ]
    return result


class Upscaler:
    def __init__(self, directory, device="auto"):
        directory = resolve(directory)
        if not (directory / "model.safetensors").is_file():
            raise FileNotFoundError(
                "Trained weights for this scale are not available yet. Train and export the model first."
            )
        self.config = read_json(directory / "config.json")
        if (directory / "checksums.json").exists():
            for filename, expected in read_json(directory / "checksums.json").items():
                if (
                    filename not in ("model.safetensors", "config.json")
                    or digest(directory / filename) != expected
                ):
                    raise ValueError("Exported artifact checksum mismatch")
        if not self.config.get("trained"):
            raise ValueError("Artifact is not a trained model")
        self.device = choose_device(device)
        self.model = build_model(self.config).to(self.device).eval()
        self.model.load_state_dict(
            load_file(str(directory / "model.safetensors"), device=str(self.device))
        )

    def upscale(self, image, tile_size=128):
        if image is None:
            raise ValueError("Upload an image first")
        image = ImageOps.exif_transpose(image).convert("RGB")
        output = tiled(self.model, to_tensor(image).unsqueeze(0), self.device, tile_size)
        return to_pil_image(output[0].clamp(0, 1).mul(255).round().div(255))


def main():
    import argparse
    from PIL import Image

    p = argparse.ArgumentParser(description="Neural 2x/4x upscaling using exported trained weights")
    p.add_argument("--image", required=True)
    p.add_argument("--scale", type=int, choices=[2, 4], required=True)
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--output", required=True)
    p.add_argument("--tile-size", type=int, default=128)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    model = Upscaler(resolve(a.artifacts) / f"x{a.scale}", a.device)
    with Image.open(a.image) as image:
        output = model.upscale(image, a.tile_size)
    output.save(a.output)
    print(f"Saved {output.width}x{output.height} image to {a.output}")


if __name__ == "__main__":
    main()
