"""YOLOv8n-seg for CAR instance segmentation on Potsdam (time-boxed CPU fine-tune).

Instances are derived from the semantic ground truth: connected components of
the car class -> polygons (YOLO-seg format). Evaluated (a) with Ultralytics
mask mAP on validation patches and (b) as *pixel* car IoU on the same 12 test
crops used for every other method, so it is comparable with the U-Net / RF.
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from benchmarks.potsdam import IGNORE, TEST_CROPS, TEST_TILES, TRAIN_TILES, VAL_TILES, load_patches, load_test_tile, save

CAR = 4
ROOT = "/home/user/data/potsdam/yolo"


def write_split(X, Y, split):
    os.makedirs(f"{ROOT}/images/{split}", exist_ok=True)
    os.makedirs(f"{ROOT}/labels/{split}", exist_ok=True)
    n_inst = 0
    for i, (x, y) in enumerate(zip(X, Y)):
        m = ((y == CAR) & (y != IGNORE)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))   # bridge eroded-boundary gaps
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lines = []
        for c in cnts:
            if cv2.contourArea(c) < 80 or len(c) < 3:
                continue
            c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2) / np.array([x.shape[1], x.shape[0]])
            lines.append("0 " + " ".join(f"{v:.5f}" for v in c.ravel()))
        n_inst += len(lines)
        cv2.imwrite(f"{ROOT}/images/{split}/{i:05d}.png", cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
        open(f"{ROOT}/labels/{split}/{i:05d}.txt", "w").write("\n".join(lines))
    return n_inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=25)
    ap.add_argument("--model", default="yolov8n-seg.pt")
    args = ap.parse_args()
    from ultralytics import YOLO
    X, Y = load_patches(TRAIN_TILES)
    has_car = np.array([(y == CAR).sum() > 200 for y in Y])
    idx = np.r_[np.where(has_car)[0], np.where(~has_car)[0][:60]]
    n_tr = write_split(X[idx], Y[idx], "train")
    Xv, Yv = load_patches(VAL_TILES, limit=60)
    n_va = write_split(Xv, Yv, "val")
    open(f"{ROOT}/data.yaml", "w").write(f"path: {ROOT}\ntrain: images/train\nval: images/val\nnames:\n  0: car\n")
    print(f"train imgs {len(idx)} instances {n_tr}; val instances {n_va}", flush=True)

    model = YOLO(args.model)
    t0 = time.time()
    model.train(data=f"{ROOT}/data.yaml", imgsz=512, epochs=100, time=args.minutes / 60, batch=8, device="cpu",
                workers=2, project="/home/user/data/potsdam/yolo_runs", name="car", exist_ok=True, verbose=False,
                plots=False, patience=100)
    train_min = (time.time() - t0) / 60
    best = "/home/user/data/potsdam/yolo_runs/car/weights/best.pt"
    model = YOLO(best if os.path.exists(best) else args.model)
    met = model.val(data=f"{ROOT}/data.yaml", imgsz=512, device="cpu", verbose=False, plots=False)

    # pixel-level car IoU on the shared test crops
    tp = fp = fn = 0
    t = time.perf_counter()
    npx = 0
    for tl in TEST_TILES:
        img, gt = load_test_tile(tl)
        for (y0, x0, s) in TEST_CROPS:
            c, g = img[y0:y0 + s, x0:x0 + s], gt[y0:y0 + s, x0:x0 + s]
            r = model.predict(cv2.cvtColor(c, cv2.COLOR_RGB2BGR), imgsz=1024, conf=0.25, device="cpu", verbose=False)[0]
            pm = np.zeros(g.shape, bool)
            if r.masks is not None:
                for m in r.masks.data.cpu().numpy():
                    pm |= cv2.resize(m, (s, s), interpolation=cv2.INTER_NEAREST) > 0.5
            valid = g != IGNORE
            gc = (g == CAR) & valid
            pm &= valid
            tp += (pm & gc).sum()
            fp += (pm & ~gc).sum()
            fn += (~pm & gc).sum()
            npx += g.size
    ms_mp = 1000 * (time.perf_counter() - t) / (npx / 1e6)
    out = dict(model=args.model, train_minutes=train_min, train_images=int(len(idx)), train_instances=int(n_tr),
               mask_mAP50=float(met.seg.map50), mask_mAP50_95=float(met.seg.map), box_mAP50=float(met.box.map50),
               car_pixel_IoU_test_crops=float(tp / max(tp + fp + fn, 1)), car_pixel_F1_test_crops=float(2 * tp / max(2 * tp + fp + fn, 1)),
               ms_per_megapixel=ms_mp)
    print(out, flush=True)
    save(out, "yolo_car.json")


if __name__ == "__main__":
    main()
