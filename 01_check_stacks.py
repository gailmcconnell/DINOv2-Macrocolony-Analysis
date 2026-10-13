# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:06:12 2026

@author: Gail McConnell, SIPBS, University of Strathclyde
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

MEDIA = ["LB", "M9"]

OUTPUT = ROOT / "QC_results"
OUTPUT.mkdir(exist_ok=True)

PREVIEW_DIR = OUTPUT / "previews"
PREVIEW_DIR.mkdir(exist_ok=True)


# ============================================================
# FUNCTIONS
# ============================================================

def get_strain(folder_name):
    """
    Convert your folder names into convenient strain labels.
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


def normalise_for_display(image, low=1, high=99.8):
    """
    Percentile normalisation for DISPLAY ONLY.
    Original TIFF data are never modified.
    """
    image = image.astype(np.float32)

    lo, hi = np.percentile(image, [low, high])

    if hi <= lo:
        return np.zeros_like(image)

    image = (image - lo) / (hi - lo)
    return np.clip(image, 0, 1)


def make_25d(stack):
    z = stack.shape[0]

    edges = np.linspace(0, z, 4, dtype=int)

    shallow = np.max(stack[edges[0]:edges[1]], axis=0)
    middle = np.max(stack[edges[1]:edges[2]], axis=0)
    deep = np.max(stack[edges[2]:edges[3]], axis=0)

    shallow = normalise_for_display(shallow)
    middle = normalise_for_display(middle)
    deep = normalise_for_display(deep)

    return np.dstack([shallow, middle, deep])


# ============================================================
# INSPECT TIFF STACKS
# ============================================================

records = []

for medium in MEDIA:

    medium_dir = ROOT / medium

    if not medium_dir.exists():
        print(f"WARNING: cannot find {medium_dir}")
        continue

    strain_folders = [p for p in medium_dir.iterdir() if p.is_dir()]

    for strain_folder in strain_folders:

        strain = get_strain(strain_folder.name)

        tif_files = sorted(
            list(strain_folder.glob("*.tif")) +
            list(strain_folder.glob("*.tiff"))
        )

        for replicate, tif_path in enumerate(tif_files, start=1):

            print()
            print("=" * 70)
            print(f"Reading: {tif_path.name}")
            print(f"Medium: {medium}")
            print(f"Strain: {strain}")

            stack = tifffile.imread(tif_path)

            # Expect a grayscale z-stack: Z x Y x X
            if stack.ndim != 3:
                print(
                    f"WARNING: {tif_path.name} has shape {stack.shape}; "
                    "expected a 3-D Z x Y x X stack."
                )
                continue

            z, y, x = stack.shape

            # ------------------------------------------------
            # Intensity statistics
            # ------------------------------------------------

            minimum = float(np.min(stack))
            maximum = float(np.max(stack))
            mean = float(np.mean(stack))
            median = float(np.median(stack))

            p1, p5, p95, p99, p998 = np.percentile(
                stack,
                [1, 5, 95, 99, 99.8]
            )

            print(f"Shape: {stack.shape}")
            print(f"Data type: {stack.dtype}")
            print(f"Z planes: {z}")
            print(f"Intensity range: {minimum:g} - {maximum:g}")

            records.append(
                {
                    "medium": medium,
                    "strain": strain,
                    "replicate": replicate,
                    "filename": tif_path.name,
                    "folder": strain_folder.name,
                    "z_planes": z,
                    "height_px": y,
                    "width_px": x,
                    "dtype": str(stack.dtype),
                    "min": minimum,
                    "p1": float(p1),
                    "p5": float(p5),
                    "median": median,
                    "mean": mean,
                    "p95": float(p95),
                    "p99": float(p99),
                    "p99.8": float(p998),
                    "max": maximum,
                }
            )

            # =================================================
            # MAKE QC PREVIEW
            # =================================================

            middle_plane = stack[z // 2]

            max_projection = np.max(stack, axis=0)

            rgb_25d = make_25d(stack)

            middle_display = normalise_for_display(middle_plane)
            max_display = normalise_for_display(max_projection)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(middle_display, cmap="gray")
            axes[0].set_title(f"Middle plane (z={z // 2})")
            axes[0].axis("off")

            axes[1].imshow(max_display, cmap="gray")
            axes[1].set_title("Maximum projection")
            axes[1].axis("off")

            axes[2].imshow(rgb_25d)
            axes[2].set_title(
                "Depth encoding\nR=shallow, G=middle, B=deep"
            )
            axes[2].axis("off")

            fig.suptitle(
                f"{medium} | {strain} | {tif_path.name}",
                fontsize=12
            )

            plt.tight_layout()

            preview_name = (
                f"{medium}_{strain}_{replicate:02d}_QC.png"
            )

            fig.savefig(
                PREVIEW_DIR / preview_name,
                dpi=150,
                bbox_inches="tight"
            )

            plt.close(fig)

            # Explicitly release this stack before loading next
            del stack


# ============================================================
# SAVE QC TABLE
# ============================================================

df = pd.DataFrame(records)

csv_path = OUTPUT / "stack_QC.csv"
df.to_csv(csv_path, index=False)

print()
print("=" * 70)
print("FINISHED")
print("=" * 70)

print(f"\nStacks found: {len(df)}")

if len(df) > 0:

    print("\nStacks by condition:")
    print(
        df.groupby(["medium", "strain"])
        .size()
        .to_string()
    )

    print("\nZ-plane summary:")
    print(
        df.groupby(["medium", "strain"])["z_planes"]
        .agg(["min", "max", "mean"])
        .round(1)
        .to_string()
    )

print(f"\nQC table saved to:\n{csv_path}")
print(f"\nPreview images saved to:\n{PREVIEW_DIR}")