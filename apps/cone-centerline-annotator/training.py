"""Weak 3-D texture segmentation from explicit GUI scribbles.

The output is a preliminary probability map for annotation assistance. It is
not anatomical ground truth and must not be used as an independent test set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude
from sklearn.ensemble import RandomForestClassifier


def feature_volume(volume: np.ndarray) -> np.ndarray:
    image = volume.astype(np.float32) / 255.0
    smooth_1 = gaussian_filter(image, (1.0, 1.0, 0.5))
    smooth_2 = gaussian_filter(image, (2.0, 2.0, 1.0))
    smooth_4 = gaussian_filter(image, (4.0, 4.0, 2.0))
    gradient = gaussian_gradient_magnitude(image, 1.0)
    return np.stack(
        (image, smooth_1, smooth_2, smooth_4, smooth_1 - smooth_4, gradient),
        axis=-1,
    )


def train_from_scribbles(
    volume_path: Path,
    scribble_path: Path,
    output_dir: Path,
    random_state: int = 48,
) -> dict:
    volume = np.load(volume_path)
    scribbles = np.load(scribble_path)
    if volume.shape != scribbles.shape:
        raise ValueError("Volume and scribbles have different shapes")
    foreground = np.argwhere(scribbles == 1)
    background = np.argwhere(scribbles == 2)
    if len(foreground) < 25:
        raise ValueError("Mark more cone centre-line points before training")
    if len(background) < 25:
        raise ValueError("Add explicit background points before training")

    features = feature_volume(volume)
    rng = np.random.default_rng(random_state)
    limit = 15000
    if len(foreground) > limit:
        foreground = foreground[rng.choice(len(foreground), limit, replace=False)]
    if len(background) > limit:
        background = background[rng.choice(len(background), limit, replace=False)]
    coordinates = np.vstack((foreground, background))
    target = np.r_[np.ones(len(foreground), dtype=np.uint8), np.zeros(len(background), dtype=np.uint8)]
    design = features[tuple(coordinates.T)]

    model = RandomForestClassifier(
        n_estimators=160,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced",
        oob_score=True,
        n_jobs=max(1, min(4, os.cpu_count() or 1)),
        random_state=random_state,
    )
    model.fit(design, target)
    probability = model.predict_proba(features.reshape(-1, features.shape[-1]))[:, 1]
    probability = probability.reshape(volume.shape).astype(np.float32)
    mask = probability >= 0.5

    mask_fraction = float(np.mean(mask))
    qc_status = "pass" if mask_fraction <= 0.20 else "fail: implausibly broad mask"

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "cone_probability.npy", probability)
    np.save(output_dir / "preliminary_cone_mask.npy", mask)
    summary = {
        "status": "annotation-assistance model; not anatomical validation",
        "foreground_training_voxels": int(len(foreground)),
        "background_training_voxels": int(len(background)),
        "random_forest_oob_score": float(model.oob_score_),
        "probability_threshold": 0.5,
        "predicted_mask_fraction": mask_fraction,
        "quality_control": qc_status,
        "warning": (
            "Training and evaluation use the same manually annotated volume. "
            "The mask must be reviewed and corrected before centre-line measurement."
        ),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    selected = np.linspace(0, volume.shape[2] - 1, 9).round().astype(int)
    figure, axes = plt.subplots(3, 3, figsize=(9, 9), dpi=150)
    for axis, depth in zip(axes.ravel(), selected):
        image = volume[:, :, depth]
        lo, hi = np.percentile(image, (1, 99))
        axis.imshow(np.clip((image - lo) / (hi - lo + 1e-6), 0, 1), cmap="gray")
        axis.contour(probability[:, :, depth], levels=[0.5], colors="#00D4FF", linewidths=0.6)
        axis.set_title(f"depth {depth}")
        axis.axis("off")
    figure.suptitle("Preliminary cone probability: manual review required")
    figure.tight_layout()
    figure.savefig(output_dir / "preliminary_segmentation_preview.png", bbox_inches="tight")
    plt.close(figure)
    return summary
