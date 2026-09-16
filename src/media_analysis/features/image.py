"""Native still measurements; no temporal tracks or editing decisions."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from media_analysis.features.exposure import bt709_luma_from_bgr
from media_analysis.features.ppocr import run_det_inference
from media_analysis.features.yunet import normalize_face_box

IMAGE_MEASUREMENT_VERSION = "image-measurements-v1"
SALIENCY_VERSION = "lab-global-local-contrast-v1"
FOCUS_VERSION = "saliency-region-centroid-v1"


def quality(bgr: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "laplacianVariance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "darkClipFraction": float(np.mean(gray <= 5)),
        "brightClipFraction": float(np.mean(gray >= 250)),
        "clipPolicy": "sRGB-gray-uint8<=5-or>=250",
        "scoreType": "measurement",
        "workingWidth": bgr.shape[1],
        "workingHeight": bgr.shape[0],
        "policyVersion": IMAGE_MEASUREMENT_VERSION,
    }


def exposure(bgr: np.ndarray) -> dict[str, Any]:
    luma = bt709_luma_from_bgr(bgr)
    return {
        "meanLuma": float(np.mean(luma)),
        "lumaStdDev": float(np.std(luma)),
        "lumaPercentiles": [float(x) for x in np.percentile(luma, [5, 50, 95])],
        "meanSrgb": [float(x) / 255 for x in bgr.mean(axis=(0, 1))[::-1]],
        "lumaColorSpace": "linear-sRGB-BT709",
        "policyVersion": IMAGE_MEASUREMENT_VERSION,
    }


def saliency(bgr: np.ndarray) -> dict[str, Any]:
    """Reproducible contrast heuristic, explicitly not a learned attention probability."""
    h, w = bgr.shape[:2]
    scale = min(1, 256 / max(h, w))
    small = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))))
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    blurred = cv2.GaussianBlur(lab, (0, 0), 3)
    contrast = np.linalg.norm(blurred - np.median(lab, axis=(0, 1)), axis=2)
    contrast += np.linalg.norm(lab - blurred, axis=2)
    peak = float(contrast.max())
    regions = []
    if peak > 3:
        mask = (contrast >= max(3, peak * 0.45)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        mh, mw = mask.shape
        for label in range(1, count):
            x, y, rw, rh, area = stats[label]
            if area < max(2, mh * mw * 0.001):
                continue
            weights = np.where(labels == label, contrast, 0)
            ys, xs = np.indices(weights.shape)
            total = float(weights.sum())
            regions.append(
                {
                    "box": {
                        "x": float(x / mw),
                        "y": float(y / mh),
                        "width": float(rw / mw),
                        "height": float(rh / mh),
                    },
                    "point": {
                        "x": float(((xs + 0.5) * weights).sum() / total / mw),
                        "y": float(((ys + 0.5) * weights).sum() / total / mh),
                    },
                    "score": float(contrast[labels == label].mean() / peak),
                    "contrastMass": total,
                }
            )
        regions.sort(key=lambda region: region["contrastMass"], reverse=True)
    return {
        "regions": regions[:8],
        "algorithmVersion": SALIENCY_VERSION,
        "model": None,
        "scoreType": "heuristic",
        "scoreMeaning": "region contrast / image peak contrast",
        "validity": "contrast-evidence" if regions else "insufficient-contrast",
        "productionQualified": False,
    }


def focus(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "point": region["point"],
                "box": region["box"],
                "evidence": {"feature": "saliency", "regionIndex": index},
                "selection": "inferred",
            }
            for index, region in enumerate(evidence["regions"])
        ],
        "policyVersion": FOCUS_VERSION,
        "evidenceAlgorithmVersion": SALIENCY_VERSION,
        "validity": evidence["validity"],
    }


def regions(feature: str, bgr: np.ndarray, runtime: Any) -> list[dict[str, Any]]:
    h, w = bgr.shape[:2]
    if feature == "subjects":
        detections, _ = runtime.subject_detector.detect_normalized(bgr)
        return [
            {"type": "person", "box": box, "score": detection.score, "scoreType": "raw_model"}
            for detection, box in detections
            if box["width"] > 0 and box["height"] > 0
        ]
    if feature == "faces":
        runtime.face_detector.set_input_size(w, h)
        found = []
        for detection in runtime.face_detector.detect(bgr):
            box = normalize_face_box(detection, frame_width=w, frame_height=h)
            if box["width"] > 0 and box["height"] > 0:
                found.append({"box": box, "score": detection.score, "scoreType": "raw_model"})
        return found
    return [
        {
            "polygon": [
                {"x": float(np.clip(x / w, 0, 1)), "y": float(np.clip(y / h, 0, 1))}
                for x, y in polygon.points
            ],
            "score": polygon.score,
            "scoreType": "raw_model",
        }
        for polygon in run_det_inference(runtime.ocr_session, bgr)
    ]
