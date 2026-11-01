"""Classical / statistical pixel-wise segmentation baselines on Potsdam.

All methods use the same per-pixel descriptor (colour + local texture) and
are evaluated on the same fixed 1024x1024 crops of the 3 official test tiles.

 1. K-means (k=6), unsupervised, clusters mapped to classes with the
    Hungarian algorithm on the train confusion (best-case mapping).
 2. GMM-Bayes: one Gaussian mixture p(x | class) per class (generative
    classifier), prediction = argmax_c log p(x|c) + log pi_c.
 3. GMM-Bayes + spatial smoothing of the class posteriors (cheap CRF/MRF
    surrogate: Gaussian-blurred log-posteriors).
 4. Random Forest on the same features (classic remote-sensing baseline).
 5. Random Forest + NIR / NDVI (multispectral physics prior).
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture

from benchmarks.potsdam import (CLASSES, IGNORE, TEST_CROPS, TEST_TILES, TRAIN_TILES, confusion, load_patches,
                                load_test_tile, save, scores)


def pixel_features(img, nir=None):
    """(H,W,3) uint8 -> (H*W, D) float32. Colour spaces + multi-scale texture."""
    f = img.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32) / np.array([180, 255, 255], np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    exg = 2 * g - r - b                                   # excess-green vegetation index
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    feats = [f, hsv, lab, exg[..., None]]
    for s in (2, 6):
        mu = cv2.GaussianBlur(gray, (0, 0), s)
        sd = np.sqrt(np.maximum(cv2.GaussianBlur(gray ** 2, (0, 0), s) - mu ** 2, 0))
        gx = cv2.Sobel(cv2.GaussianBlur(gray, (0, 0), s / 2), cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(cv2.GaussianBlur(gray, (0, 0), s / 2), cv2.CV_32F, 0, 1)
        feats += [mu[..., None], sd[..., None], np.sqrt(gx ** 2 + gy ** 2)[..., None]]
        feats.append(cv2.GaussianBlur(f, (0, 0), s))      # smoothed colour context
    if nir is not None:
        n = nir.astype(np.float32) / 255.0
        ndvi = (n - r) / (n + r + 1e-6)
        feats += [n[..., None], ndvi[..., None], cv2.GaussianBlur(ndvi, (0, 0), 4)[..., None]]
    return np.concatenate(feats, axis=2).reshape(-1, sum(x.shape[2] for x in feats))


def sample_train(n_per_class=12000, with_nir=False, seed=0):
    rng = np.random.default_rng(seed)
    X, Y = load_patches(TRAIN_TILES, with_nir=True)
    Xs, Ys = [], []
    for i in rng.choice(len(X), 220, replace=False):
        F = pixel_features(X[i][..., :3], X[i][..., 3] if with_nir else None)
        lab = Y[i].ravel()
        keep = np.where(lab != IGNORE)[0]
        pick = rng.choice(keep, min(1500, len(keep)), replace=False)
        Xs.append(F[pick])
        Ys.append(lab[pick])
    Xs, Ys = np.concatenate(Xs), np.concatenate(Ys)
    # class-balance (cap each class)
    idx = np.concatenate([rng.permutation(np.where(Ys == c)[0])[:n_per_class] for c in range(6)])
    return Xs[idx], Ys[idx], Xs, Ys


def test_crops(with_nir=False):
    for t in TEST_TILES:
        img, gt = load_test_tile(t, with_nir=True)
        for (y, x, s) in TEST_CROPS:
            c = img[y:y + s, x:x + s]
            yield c[..., :3], (c[..., 3] if with_nir else None), gt[y:y + s, x:x + s]


class GMMBayes:
    def __init__(self, k=4):
        self.k = k

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.g = [GaussianMixture(self.k, covariance_type="full", reg_covar=1e-3, random_state=0).fit(Z[y == c])
                  for c in range(6)]
        self.logprior = np.log(np.bincount(y, minlength=6) / len(y))
        return self

    def log_post(self, X):
        Z = (X - self.mu) / self.sd
        return np.stack([g.score_samples(Z) for g in self.g], 1) + self.logprior


def main():
    res, timings = [], {}
    Xb, Yb, Xall, Yall = sample_train()
    print("train pixels", Xb.shape, flush=True)

    # ---- fit
    km = MiniBatchKMeans(6, random_state=0, n_init=3, batch_size=4096).fit((Xall - Xall.mean(0)) / Xall.std(0))
    cm_map = np.zeros((6, 6))
    lab_tr = km.predict((Xall - Xall.mean(0)) / Xall.std(0))
    for c, k in zip(Yall, lab_tr):
        cm_map[c, k] += 1
    r_, c_ = linear_sum_assignment(-cm_map)
    k2c = dict(zip(c_, r_))
    gb = GMMBayes(4).fit(Xb, Yb)
    rf = RandomForestClassifier(120, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=0).fit(Xb, Yb)
    Xn, Yn, _, _ = sample_train(with_nir=True)
    rfn = RandomForestClassifier(120, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=0).fit(Xn, Yn)

    cms = {k: np.zeros((6, 6), np.int64) for k in
           ["K-means (k=6, unsupervised, Hungarian-mapped)", "GMM-Bayes (4 comps/class)",
            "GMM-Bayes + posterior smoothing", "Random Forest (RGB features)", "Random Forest (RGB + NIR/NDVI)"]}
    names = list(cms)
    t_acc = {k: 0.0 for k in names}
    mu_all, sd_all = Xall.mean(0), Xall.std(0)
    keep_vis = None
    for rgb, nir, gt in test_crops(with_nir=True):
        H, W = gt.shape
        F = pixel_features(rgb)
        t = time.perf_counter()
        p = np.vectorize(k2c.get)(km.predict((F - mu_all) / sd_all)).reshape(H, W)
        t_acc[names[0]] += time.perf_counter() - t
        cms[names[0]] += confusion(p.astype(np.int64), gt)
        t = time.perf_counter()
        lp = gb.log_post(F).reshape(H, W, 6)
        p1 = lp.argmax(-1)
        t_acc[names[1]] += time.perf_counter() - t
        cms[names[1]] += confusion(p1, gt)
        t = time.perf_counter()
        post = np.exp(lp - lp.max(-1, keepdims=True))
        post /= post.sum(-1, keepdims=True)
        p2 = cv2.GaussianBlur(post, (0, 0), 3).argmax(-1)
        t_acc[names[2]] += time.perf_counter() - t + t_acc[names[1]] * 0
        cms[names[2]] += confusion(p2, gt)
        t = time.perf_counter()
        p3 = rf.predict(F).reshape(H, W)
        t_acc[names[3]] += time.perf_counter() - t
        cms[names[3]] += confusion(p3, gt)
        Fn = pixel_features(rgb, nir)
        t = time.perf_counter()
        p4 = rfn.predict(Fn).reshape(H, W)
        t_acc[names[4]] += time.perf_counter() - t
        cms[names[4]] += confusion(p4, gt)
        if keep_vis is None:
            keep_vis = dict(rgb=rgb, gt=gt, kmeans=p, gmm=p2, rf=p3, rfn=p4)
        print(".", end="", flush=True)
    print()
    n_px = len(TEST_TILES) * len(TEST_CROPS) * 1024 * 1024
    for k in names:
        s = scores(cms[k], k)
        s["ms_per_megapixel"] = 1000 * t_acc[k] / (n_px / 1e6)
        res.append(s)
        print(f"{k:48s} mIoU5={s['mIoU_5_no_clutter']:.3f} mF1_5={s['mF1_5_no_clutter']:.3f} OA={s['OA']:.3f} "
              f"| " + " ".join(f"{c}:{v:.2f}" for c, v in s['per_class_iou'].items()), flush=True)
    save(dict(results=res, eval="12 fixed 1024x1024 crops of test tiles " + ",".join(TEST_TILES),
              confusions={k: v.tolist() for k, v in cms.items()}), "classical_seg.json")
    np.savez_compressed("/home/user/data/potsdam/vis_classical.npz", **keep_vis)


if __name__ == "__main__":
    main()
