#!/usr/bin/env python3
"""Generate Depth Pro depth maps and save them as NPZ files."""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from depth_pro import create_model_and_transforms, load_rgb
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
DEFAULT_OUTPUT_DIR = Path("data/output")
DEFAULT_INPUT_DIR = Path("data/datasets/validation_road_obstacle_21")


def get_torch_device() -> torch.device:
    """Return the best available Torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_image_paths(input_path: Path) -> Tuple[List[Path], Path]:
    """Return input images and the path used to preserve relative directories."""
    if input_path.is_dir():
        image_paths = sorted(
            path
            for path in input_path.glob("**/*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        return image_paths, input_path
    return [input_path], input_path.parent


def generate_depth_maps(input_path: Path, output_path: Path) -> None:
    """Generate one compressed NPZ depth map for each input image."""
    device = get_torch_device()
    LOGGER.info("Using device: %s", device)

    model, transform = create_model_and_transforms(
        device=device,
        precision=torch.half,
    )
    model.eval()

    image_paths, relative_root = get_image_paths(input_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_paths, desc="Generating depth maps", unit="image"):
        try:
            image, _, f_px = load_rgb(image_path)
            prediction = model.infer(transform(image), f_px=f_px)

            depth = prediction["depth"].detach().cpu().numpy().squeeze()
            output_file = (
                output_path / image_path.relative_to(relative_root).parent / image_path.stem
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if f_px is not None:
                predicted_fx = f_px
            elif prediction["focallength_px"] is not None:
                predicted_fx = prediction["focallength_px"].detach().cpu().item()
            else:
                raise ValueError("Depth Pro did not return a focal length")

            fx = float(predicted_fx)
            height, width = depth.shape
            cx = (width - 1) / 2.0
            cy = (height - 1) / 2.0
            intrinsics = np.array(
                [
                    [fx, 0.0, cx],
                    [0.0, fx, cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

            np.savez_compressed(
                output_file,
                depth=depth,
                intrinsics=intrinsics,
            )
            tqdm.write(
                f"Saved: {output_file}.npz | depth: {depth.shape} | intrinsics: {intrinsics.shape}"
            )
        except Exception as error:
            LOGGER.error("Failed to process %s: %s", image_path, error)


def main() -> None:
    """Parse arguments and generate depth maps."""
    parser = argparse.ArgumentParser(
        description="Generate Depth Pro depth maps as compressed NPZ files."
    )
    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Path to an input image or directory of images.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where NPZ depth maps will be stored.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show verbose output.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    generate_depth_maps(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
