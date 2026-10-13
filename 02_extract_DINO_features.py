# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:19:29 2026

@author: Gail McConnell, SIPBS, University of Strathclyde
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import tifffile
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

MEDIA = ["LB", "M9"]

OUTPUT = ROOT / "DINO_results"
OUTPUT.mkdir(exist_ok=True)

TILE_QC = OUTPUT / "tile_QC"
TILE_QC.mkdir(exist_ok=True)

TILE_SIZE = 224

# This threshold is NOT used to reject tiles.
# It is retained only to record the fraction of pixels above
# this intensity as potentially useful metadata.
SIGNAL_THRESHOLD = 0.08


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


# ============================================================
# LOAD DINOV2
# ============================================================

print("Loading DINOv2...")

model = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14"
)

model = model.to(device)
model.eval()

print("DINOv2 loaded.")


# ============================================================
# DINO TRANSFORM
# ============================================================

# DINOv2 expects three-channel images.
#
# Our microscopy images are grayscale, so each grayscale
# tile will later be copied into R, G and B identically.
#
# ImageNet normalisation is then applied before DINOv2.

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


# ============================================================
# FUNCTIONS
# ============================================================

def get_strain(folder_name):
    """
    Convert folder names into convenient strain labels.

    Expected folder names include:
        BW25113
        BW25113 amiA
        BW25113 ompR
        BW25113 ydgD
    """

    name = folder_name.lower()

    if "amia" in name:
        return "amiA"

    elif "ompr" in name:
        return "ompR"

    elif "ydgd" in name:
        return "ydgD"

    else:
        return "BW25113"


def get_replicate(filename):
    """
    Extract replicate number from filenames such as:

        BW25113_5um_stack1.tif
        BW25113_amiA_5um_stack3.tif
    """

    match = re.search(
        r"stack(\d+)",
        filename.lower()
    )

    if match:
        return int(match.group(1))

    return None


def make_projection(stack):
    """
    Generate a maximum-intensity projection through the
    complete z-stack.
    """

    return np.max(
        stack,
        axis=0
    )


def normalise_projection(image):
    """
    Robustly normalise the WHOLE maximum-intensity projection.

    The 1st and 99.8th percentiles are used rather than the
    absolute minimum and maximum so that occasional extreme
    pixels do not determine the contrast range.

    Importantly, normalisation is performed before tiling.
    Individual tiles are NOT independently contrast stretched.

    Returns a float32 image with values from 0 to 1.
    """

    image = image.astype(
        np.float32
    )

    low, high = np.percentile(
        image,
        [1, 99.8]
    )

    if high <= low:

        return np.zeros_like(
            image,
            dtype=np.float32
        )

    image = (
        (image - low) /
        (high - low)
    )

    image = np.clip(
        image,
        0,
        1
    )

    return image


def generate_tiles(image):
    """
    Generate a centred grid of non-overlapping
    224 x 224 pixel tiles.

    For a 2048 x 2048 image:

        2048 // 224 = 9

    Therefore:

        9 x 9 = 81 tiles

    9 x 224 = 2016 pixels.

    This leaves 32 unused pixels in each dimension.

    Rather than always losing these pixels from the right and
    bottom edges, the grid is centred.

    Therefore 16 pixels are omitted from each side.

    No tiles are rejected.
    """

    height, width = image.shape

    n_y = height // TILE_SIZE
    n_x = width // TILE_SIZE

    used_height = (
        n_y * TILE_SIZE
    )

    used_width = (
        n_x * TILE_SIZE
    )

    y_offset = (
        height - used_height
    ) // 2

    x_offset = (
        width - used_width
    ) // 2

    for row in range(n_y):

        for col in range(n_x):

            y = (
                y_offset +
                row * TILE_SIZE
            )

            x = (
                x_offset +
                col * TILE_SIZE
            )

            tile = image[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ]

            yield (
                row,
                col,
                x,
                y,
                tile
            )


def tile_to_rgb(tile):
    """
    Convert a normalised grayscale tile into a three-channel
    image.

    All three channels are identical:

        R = grayscale
        G = grayscale
        B = grayscale

    Therefore no artificial colour information is introduced.
    """

    tile8 = (
        tile * 255
    ).astype(
        np.uint8
    )

    rgb = np.stack(
        [
            tile8,
            tile8,
            tile8
        ],
        axis=-1
    )

    return rgb


def extract_embedding(rgb_tile):
    """
    Extract a 384-dimensional DINOv2 ViT-S/14 embedding
    from one 224 x 224 RGB tile.
    """

    pil_image = Image.fromarray(
        rgb_tile
    )

    tensor = transform(
        pil_image
    )

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(
        device
    )

    with torch.inference_mode():

        embedding = model(
            tensor
        )

    embedding = (
        embedding
        .squeeze(0)
        .cpu()
        .numpy()
    )

    return embedding


def save_tile_qc(
    projection,
    tile_positions,
    title,
    output_path
):
    """
    Save a QC image showing exactly which regions were
    presented to DINOv2.

    Since every tile is retained, the image should show a
    centred 9 x 9 grid for a 2048 x 2048 image.
    """

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.imshow(
        projection,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    for x, y in tile_positions:

        rectangle = plt.Rectangle(
            (x, y),
            TILE_SIZE,
            TILE_SIZE,
            fill=False,
            linewidth=0.8
        )

        ax.add_patch(
            rectangle
        )

    ax.set_title(
        title
    )

    ax.axis(
        "off"
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# PROCESS DATA
# ============================================================

feature_records = []

summary_records = []

total_samples = 0

total_tiles = 0


for medium in MEDIA:

    medium_dir = (
        ROOT /
        medium
    )

    if not medium_dir.exists():

        print(
            f"WARNING: cannot find {medium_dir}"
        )

        continue


    strain_folders = sorted(
        [
            p
            for p in medium_dir.iterdir()
            if p.is_dir()
        ]
    )


    for strain_folder in strain_folders:

        strain = get_strain(
            strain_folder.name
        )

        tif_files = sorted(
            list(
                strain_folder.glob(
                    "*.tif"
                )
            )
            +
            list(
                strain_folder.glob(
                    "*.tiff"
                )
            )
        )


        for tif_path in tif_files:

            replicate = get_replicate(
                tif_path.name
            )

            total_samples += 1


            print()
            print(
                "=" * 70
            )

            print(
                f"Processing sample {total_samples}: "
                f"{tif_path.name}"
            )

            print(
                f"Medium: {medium}"
            )

            print(
                f"Strain: {strain}"
            )

            print(
                f"Replicate: {replicate}"
            )


            # =================================================
            # LOAD TIFF STACK
            # =================================================

            stack = tifffile.imread(
                tif_path
            )

            if stack.ndim != 3:

                print(
                    f"WARNING: unexpected shape "
                    f"{stack.shape}. Skipping."
                )

                del stack

                continue


            z_planes = stack.shape[0]

            height = stack.shape[1]

            width = stack.shape[2]


            print(
                f"Stack shape: {stack.shape}"
            )


            # =================================================
            # MAXIMUM-INTENSITY PROJECTION
            # =================================================

            projection_raw = make_projection(
                stack
            )

            # We no longer need the full z-stack.
            # Delete it immediately to minimise RAM usage.

            del stack


            # =================================================
            # NORMALISE WHOLE PROJECTION
            # =================================================

            projection = normalise_projection(
                projection_raw
            )

            del projection_raw


            # =================================================
            # GENERATE TILES AND RUN DINO
            # =================================================

            tile_positions = []

            sample_embeddings = []

            sample_signal_fractions = []

            tile_number = 0


            for (
                row,
                col,
                x,
                y,
                tile
            ) in generate_tiles(
                projection
            ):

                tile_number += 1

                total_tiles += 1


                # ---------------------------------------------
                # Record signal fraction
                #
                # IMPORTANT:
                # This is metadata only.
                #
                # It is NOT used to reject the tile.
                # ---------------------------------------------

                signal_fraction = float(
                    np.mean(
                        tile >
                        SIGNAL_THRESHOLD
                    )
                )

                sample_signal_fractions.append(
                    signal_fraction
                )


                # ---------------------------------------------
                # Save tile location for QC
                # ---------------------------------------------

                tile_positions.append(
                    (x, y)
                )


                # ---------------------------------------------
                # Convert grayscale to RGB
                # ---------------------------------------------

                rgb = tile_to_rgb(
                    tile
                )


                # ---------------------------------------------
                # DINOv2
                # ---------------------------------------------

                embedding = extract_embedding(
                    rgb
                )


                sample_embeddings.append(
                    embedding
                )


                # ---------------------------------------------
                # Metadata for this tile
                # ---------------------------------------------

                record = {

                    "medium":
                        medium,

                    "strain":
                        strain,

                    "replicate":
                        replicate,

                    "filename":
                        tif_path.name,

                    "z_planes":
                        z_planes,

                    "image_height_px":
                        height,

                    "image_width_px":
                        width,

                    "tile":
                        tile_number,

                    "tile_row":
                        row,

                    "tile_col":
                        col,

                    "x":
                        x,

                    "y":
                        y,

                    "signal_fraction":
                        signal_fraction
                }


                # ---------------------------------------------
                # Add the 384 DINO features
                # ---------------------------------------------

                for i, value in enumerate(
                    embedding
                ):

                    record[
                        f"dino_{i:03d}"
                    ] = float(
                        value
                    )


                feature_records.append(
                    record
                )


            # =================================================
            # CREATE SAMPLE-LEVEL EMBEDDING
            # =================================================

            if len(
                sample_embeddings
            ) > 0:

                embedding_matrix = np.vstack(
                    sample_embeddings
                )


                # Mean of all 81 tile embeddings
                #
                # This gives one 384-dimensional representation
                # for the complete biological replicate.

                mean_embedding = np.mean(
                    embedding_matrix,
                    axis=0
                )


                summary = {

                    "medium":
                        medium,

                    "strain":
                        strain,

                    "replicate":
                        replicate,

                    "filename":
                        tif_path.name,

                    "z_planes":
                        z_planes,

                    "tiles":
                        len(
                            sample_embeddings
                        ),

                    "mean_signal_fraction":
                        float(
                            np.mean(
                                sample_signal_fractions
                            )
                        )
                }


                for i, value in enumerate(
                    mean_embedding
                ):

                    summary[
                        f"dino_{i:03d}"
                    ] = float(
                        value
                    )


                summary_records.append(
                    summary
                )


            # =================================================
            # SAVE TILE QC IMAGE
            # =================================================

            if replicate is None:

                replicate_text = "unknown"

            else:

                replicate_text = (
                    f"{replicate:02d}"
                )


            qc_name = (

                f"{medium}_"
                f"{strain}_"
                f"{replicate_text}_"
                f"tiles.png"

            )


            qc_title = (

                f"{medium} | "
                f"{strain} | "
                f"replicate {replicate}\n"
                f"Tiles: {len(tile_positions)}"

            )


            save_tile_qc(

                projection,

                tile_positions,

                qc_title,

                TILE_QC /
                qc_name

            )


            print(
                f"Tiles processed: "
                f"{len(tile_positions)}"
            )


            print(
                "Mean signal fraction: "
                f"{np.mean(sample_signal_fractions):.3f}"
            )


            # Projection no longer needed.

            del projection


# ============================================================
# SAVE TILE-LEVEL FEATURES
# ============================================================

print()
print(
    "Saving tile-level features..."
)

features_df = pd.DataFrame(
    feature_records
)

features_file = (
    OUTPUT /
    "DINO_tile_features.csv"
)

features_df.to_csv(
    features_file,
    index=False
)


# ============================================================
# SAVE SAMPLE-LEVEL FEATURES
# ============================================================

print(
    "Saving sample-level features..."
)

summary_df = pd.DataFrame(
    summary_records
)

summary_file = (
    OUTPUT /
    "DINO_sample_features.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL CHECKS
# ============================================================

print()
print(
    "=" * 70
)

print(
    "FINISHED"
)

print(
    "=" * 70
)

print()

print(
    f"Biological samples processed: "
    f"{len(summary_df)}"
)

print(
    f"Total DINO tiles processed: "
    f"{len(features_df)}"
)


if len(summary_df) > 0:

    print()

    print(
        "Tiles per biological sample:"
    )

    print(
        summary_df[
            "tiles"
        ].value_counts().sort_index()
    )


print()

print(
    "Tile features saved to:"
)

print(
    features_file
)

print()

print(
    "Sample features saved to:"
)

print(
    summary_file
)

print()

print(
    "Tile QC images saved to:"
)

print(
    TILE_QC
)


# ============================================================
# EXPECTED RESULT
# ============================================================

expected_samples = 40

expected_tiles_per_sample = 81

expected_total_tiles = (
    expected_samples *
    expected_tiles_per_sample
)


print()
print(
    "=" * 70
)

print(
    "EXPECTED DATASET"
)

print(
    "=" * 70
)

print(
    f"Expected biological samples: "
    f"{expected_samples}"
)

print(
    f"Expected tiles per sample: "
    f"{expected_tiles_per_sample}"
)

print(
    f"Expected total tiles: "
    f"{expected_total_tiles}"
)


if (
    len(summary_df) ==
    expected_samples
    and
    len(features_df) ==
    expected_total_tiles
):

    print()
    print(
        "CHECK PASSED:"
    )

    print(
        "40 samples and 3240 tiles "
        "were processed."
    )

else:

    print()
    print(
        "CHECK DATA:"
    )

    print(
        "The number of samples or tiles "
        "differs from the expected "
        "40 samples / 3240 tiles."
    )