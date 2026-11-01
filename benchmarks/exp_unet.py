"""Train & evaluate the repository's Residual U-Net (TensorFlow/Keras) on Potsdam.

Time-boxed CPU training (no GPU available): the point is a *real, honest*
number for the interview, not SOTA. Uses the repo's own `get_unet_model`,
`jaccard_distance`-free sparse CE with the eroded-boundary pixels ignored,
and the repo's (bug-fixed) `get_extract_pred_scatter` / TTA for inference.

Usage: python -m benchmarks.exp_unet --minutes 35 --filters 16
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from benchmarks.potsdam import (IGNORE, TEST_CROPS, TEST_TILES, TRAIN_TILES, VAL_TILES, confusion, load_patches,  # noqa: E402
                                load_test_tile, save, scores)
from src.inference.predictor import get_extract_pred_scatter  # noqa: E402
from src.models.unet import get_unet_model  # noqa: E402

tf.random.set_seed(0)
np.random.seed(0)


def batches(X, Y, bs, size=256):
    while True:
        idx = np.random.randint(0, len(X), bs)
        xb = np.empty((bs, size, size, 3), np.float32)
        yb = np.empty((bs, size, size), np.int32)
        for k, i in enumerate(idx):
            y0, x0 = np.random.randint(0, X.shape[1] - size + 1, 2)
            a, b = X[i, y0:y0 + size, x0:x0 + size], Y[i, y0:y0 + size, x0:x0 + size]
            r = np.random.randint(4)                          # geometric aug: rot90 x flips
            a, b = np.rot90(a, r), np.rot90(b, r)
            if np.random.rand() < .5:
                a, b = a[:, ::-1], b[:, ::-1]
            # photometric aug: brightness/contrast jitter
            a = a.astype(np.float32) / 255.0
            a = np.clip((a - .5) * np.random.uniform(.85, 1.15) + .5 + np.random.uniform(-.06, .06), 0, 1)
            xb[k], yb[k] = a, b
        yield xb, yb


def masked_ce(y, logits_softmax):
    y = tf.cast(y, tf.int32)
    m = tf.cast(y != IGNORE, tf.float32)
    yc = tf.where(y == IGNORE, 0, y)
    ce = tf.keras.losses.sparse_categorical_crossentropy(yc, logits_softmax)
    return tf.reduce_sum(ce * m) / (tf.reduce_sum(m) + 1.0)


class Wrapped:
    """predictor expects uint8 RGB -> model; normalise inside."""
    def __init__(self, m):
        self.m = m
        self.output_shape = m.output_shape

    def predict(self, x, batch_size=8, verbose=0):
        return self.m.predict(x.astype(np.float32) / 255.0, batch_size=batch_size, verbose=0)


def evaluate(model, tta=False, crops=True, stride=256):
    from src.inference.tta import get_tile_tta_pred
    cm = np.zeros((6, 6), np.int64)
    cfg = dict(patch_size=256, patch_stride=stride, batch_size=16, num_classes=6)
    w = Wrapped(model)
    t0 = time.perf_counter()
    npx = 0
    for t in TEST_TILES:
        img, gt = load_test_tile(t)
        regions = [(y, x, s) for (y, x, s) in TEST_CROPS] if crops else [(0, 0, 6000)]
        for (y, x, s) in regions:
            c, g = img[y:y + s, x:x + s], gt[y:y + s, x:x + s]
            if tta:
                p = get_tile_tta_pred(c, w, cfg, from_disk=False)
            else:
                p = get_extract_pred_scatter(c, w, 256, stride, 16, 6).argmax(-1)
            cm += confusion(p.astype(np.int64), g)
            npx += g.size
    return cm, 1000 * (time.perf_counter() - t0) / (npx / 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=35)
    ap.add_argument("--filters", type=int, default=16)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    X, Y = load_patches(TRAIN_TILES)
    Xv, Yv = load_patches(VAL_TILES, limit=40)
    model = get_unet_model((256, 256, 3), filters=args.filters, n_block=4, n_class=6, mode="serial", residual=True)
    print("params", model.count_params(), flush=True)
    opt = tf.keras.optimizers.Adam(1e-3)

    @tf.function
    def step(x, y):
        with tf.GradientTape() as tape:
            loss = masked_ce(y, model(x, training=True))
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss

    gen = batches(X, Y, args.bs)
    t0, it, hist = time.time(), 0, []
    budget = args.minutes * 60
    while time.time() - t0 < budget:
        frac = (time.time() - t0) / budget                   # one-cycle-ish: warmup then cosine decay
        lr = 1e-3 * min(1.0, frac / 0.1) * (0.5 * (1 + np.cos(np.pi * min(1, frac))) + 0.02)
        opt.learning_rate.assign(lr)
        xb, yb = next(gen)
        loss = float(step(xb, yb))
        hist.append(loss)
        it += 1
        if it % 50 == 0:
            pv = model.predict(Xv[:, :256, :256].astype(np.float32) / 255.0, verbose=0).argmax(-1)
            s = scores(confusion(pv, Yv[:, :256, :256]))
            print(f"it {it} t={(time.time()-t0)/60:.1f}m loss={np.mean(hist[-50:]):.3f} val mIoU5={s['mIoU_5_no_clutter']:.3f}", flush=True)
    model.save_weights("/home/user/data/potsdam/unet_f%d.weights.h5" % args.filters)
    out = dict(params=int(model.count_params()), iterations=it, minutes=args.minutes, batch=args.bs,
               patches_seen=it * args.bs, loss_curve=hist[::10])
    cm, ms = evaluate(model)
    out["no_tta"] = scores(cm, "Residual U-Net (repo), no TTA")
    out["no_tta"]["ms_per_megapixel"] = ms
    print("no TTA", out["no_tta"]["mIoU_5_no_clutter"], out["no_tta"]["per_class_iou"], flush=True)
    cm, ms = evaluate(model, tta=True)
    out["tta8"] = scores(cm, "Residual U-Net (repo), 8x TTA")
    out["tta8"]["ms_per_megapixel"] = ms
    print("TTA", out["tta8"]["mIoU_5_no_clutter"], flush=True)
    cm, ms = evaluate(model, stride=128)
    out["overlap50"] = scores(cm, "Residual U-Net (repo), 50% overlap stitching")
    out["overlap50"]["ms_per_megapixel"] = ms
    print("overlap", out["overlap50"]["mIoU_5_no_clutter"], flush=True)
    # full-tile inference on one 6000x6000 tile (the repo's headline use-case)
    img, gt = load_test_tile(TEST_TILES[0])
    t = time.perf_counter()
    p = get_extract_pred_scatter(img, Wrapped(model), 256, 256, 32, 6).argmax(-1)
    out["full_tile_seconds_cpu"] = time.perf_counter() - t
    out["full_tile"] = scores(confusion(p.astype(np.int64), gt), f"full tile {TEST_TILES[0]}")
    np.save("/home/user/data/potsdam/unet_pred_2_13.npy", p.astype(np.uint8))
    save(out, f"unet_f{args.filters}.json")
    print("done", out["full_tile_seconds_cpu"], flush=True)


if __name__ == "__main__":
    main()
