"""ISPRS Potsdam loading + evaluation shared by all satellite benchmarks.

Data: HF mirror `aamante/potsdam` (official 512x512 crops of the 6000x6000
true orthophotos, RGB+NIR, *eroded-boundary* ground truth "gt_no_boundary").
Label ids on disk: 0 = eroded boundary (ignored), 1..6 = classes below.

Split (by *tile*, never by patch -> no spatial leakage):
  train : 2_10 3_11 4_12 5_10 6_8 6_11      (6 x 144 patches)
  val   : 7_9 7_12                          (2 x 144 patches)
  test  : official test tiles 2_13 3_14 5_15 (full 6000x6000)
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import tifffile
import cv2

DATA = os.environ.get("POTSDAM", "/home/user/data/potsdam")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "docs", "figures")
CLASSES = ["Impervious", "Building", "Low veg.", "Tree", "Car", "Clutter"]
COLORS = np.array([[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]], np.uint8)
IGNORE = 255
TRAIN_TILES = ["2_10", "3_11", "4_12", "5_10", "6_8", "6_11"]
VAL_TILES = ["7_9", "7_12"]
TEST_TILES = ["2_13", "3_14", "5_15"]
# fixed evaluation crops on the test tiles (y, x, size) - used by EVERY method
TEST_CROPS = [(y, x, 1024) for y in (500, 3500) for x in (500, 3500)]


def _lab(gt):
    out = gt.astype(np.int16) - 1
    out[gt == 0] = IGNORE
    return out.astype(np.uint8)


def load_patches(tiles, with_nir=False, limit=None):
    imgs, labs = [], []
    for t in tiles:
        files = sorted(glob.glob(f"{DATA}/train/RGBIR/potsdam_{t}_*.tif"))[:limit]
        for f in files:
            a = tifffile.imread(f)
            imgs.append(a if with_nir else a[..., :3])
            labs.append(_lab(cv2.imread(f.replace("RGBIR", "gt").replace(".tif", ".png"), cv2.IMREAD_UNCHANGED)))
    return np.stack(imgs), np.stack(labs)


def load_test_tile(t, with_nir=False):
    a = tifffile.imread(f"{DATA}/test/RGBIR/potsdam_{t}.tif")
    g = _lab(tifffile.imread(f"{DATA}/test/gt/potsdam_{t}.tif"))
    return (a if with_nir else a[..., :3]), g


def confusion(pred, gt, n=6):
    m = gt != IGNORE
    return np.bincount(n * gt[m].astype(np.int64) + pred[m], minlength=n * n).reshape(n, n)


def scores(cm, name=""):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return dict(method=name, per_class_iou=dict(zip(CLASSES, iou.round(4).tolist())),
                per_class_f1=dict(zip(CLASSES, f1.round(4).tolist())),
                mIoU_6=float(iou.mean()), mF1_6=float(f1.mean()),
                mIoU_5_no_clutter=float(iou[:5].mean()), mF1_5_no_clutter=float(f1[:5].mean()),
                OA=float(tp.sum() / cm.sum()))


def save(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(obj, open(os.path.join(RESULTS, name), "w"), indent=2)


def colorize(lab):
    out = np.zeros(lab.shape + (3,), np.uint8)
    for i, c in enumerate(COLORS):
        out[lab == i] = c
    return out
