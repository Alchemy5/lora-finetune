import torch.nn as nn
from torchvision import transforms
import PIL
from PIL import Image
import numpy as np
import torch


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def pil_to_pt(images):
    """
    Convert a PIL image or a batch of images to a torch image.
    """
    if isinstance(images, Image.Image):
        images = [images]

    images = [np.array(image) for image in images]
    images = np.stack(images)
    images = torch.from_numpy(images).float().permute(0, 3, 1, 2)

    # normalizes images to [-1, 1]
    images = images / 255 * 2 - 1

    return images


def pt_to_pil(images):
    """
    Convert a torch image to a PIL image.
    """
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = numpy_to_pil(images)
    return images


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def pil_to_numpy(images):
    """
    Convert a PIL image or a batch of images to a numpy image.
    """
    if isinstance(images, Image.Image):
        images = [images]

    return [np.array(image) for image in images]


def make_image_grid(
    images: list[PIL.Image.Image], rows: int, cols: int, resize: int = None
) -> PIL.Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
