#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from visual_helpers import convert_to_contrast_3chnl

NO_LABEL = 0


# ============================================================
# SPYDI DBSCAN
# ============================================================

def my_spydi_dbscan(selected_events, eps_val, min_samples_val):

    if selected_events.shape[0] == 0:
        return np.array([])

    X = selected_events[:, :2].astype(int)

    H = int(X[:, 0].max()) + 1
    W = int(X[:, 1].max()) + 1

    binary_img = np.zeros((H, W), dtype=np.uint8)

    for x, y in X:
        binary_img[x, y] = 1

    label_img = np.zeros((H, W), dtype=np.int32)
    core_map = np.zeros((H, W), dtype=np.uint8)

    cluster_counter = 1

    def neighbor_count(y, x):
        y0 = max(0, y - eps_val)
        y1 = min(H, y + eps_val + 1)
        x0 = max(0, x - eps_val)
        x1 = min(W, x + eps_val + 1)
        return np.sum(binary_img[y0:y1, x0:x1])

    for y in range(H):
        for x in range(W):

            if binary_img[y, x] == 0:
                continue

            cnt = neighbor_count(y, x)
            is_core = cnt >= min_samples_val
            core_map[y, x] = 1 if is_core else 0

            if not is_core:
                continue

            min_label = None

            for yy in range(max(0, y - eps_val), min(H, y + eps_val + 1)):
                for xx in range(max(0, x - eps_val), min(W, x + eps_val + 1)):
                    if core_map[yy, xx]:
                        lbl = label_img[yy, xx]
                        if lbl != NO_LABEL:
                            if min_label is None or lbl < min_label:
                                min_label = lbl

            if min_label is not None:
                label_img[y, x] = min_label
            else:
                label_img[y, x] = cluster_counter
                cluster_counter += 1

    labels = np.zeros(len(X), dtype=np.int32)

    for i, (x, y) in enumerate(X):
        labels[i] = label_img[x, y]

    labels[labels == 0] = -1

    return labels


# ============================================================
# IOU
# ============================================================

def _compute_IOU_(box, gt):

    x1, y1, x2, y2 = box
    gx1, gy1, gx2, gy2, _, _ = gt

    inter_w = max(0, min(x2, gx2) - max(x1, gx1))
    inter_h = max(0, min(y2, gy2) - max(y1, gy1))

    inter = inter_w * inter_h

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (gx2 - gx1) * (gy2 - gy1)

    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


# ============================================================
# CLUSTER EXTRACTION
# ============================================================

def getboundaries(event_img, eps, minpts, mindiag):

    xs, ys = np.where(event_img > 0)

    if len(xs) == 0:
        return []

    pts = np.vstack((xs, ys)).T

    labels = DBSCAN(eps=eps, min_samples=minpts).fit_predict(pts)

    boxes = []

    for lbl in set(labels):

        if lbl == -1:
            continue

        cluster = pts[labels == lbl]

        if cluster.shape[0] == 0:
            continue

        x = cluster[:, 0]
        y = cluster[:, 1]

        diag = (x.max()-x.min())**2 + (y.max()-y.min())**2

        if diag >= mindiag:
            boxes.append((y.min(), x.min(), y.max(), x.max()))

    return boxes


# ============================================================
# MAIN FUNCTION
# ============================================================

def compare_all(model,
                evnt_img_3c,
                gray_img_3c,
                recovered_3c,
                evnt_img,
                eps_dbscan=15,
                eps_spydi=13,
                min_samples_val=10,
                mindiagonalsquared=100):

    results = model(gray_img_3c)
    dets = results.xyxy[0].cpu().numpy()

    yolo_boxes = [
        (int(x1), int(y1), int(x2), int(y2), conf, cls)
        for x1, y1, x2, y2, conf, cls in dets
    ]

    event_img = recovered_3c[:, :, 0]

    db_boxes = getboundaries(event_img, eps_dbscan, min_samples_val, mindiagonalsquared)

    xs, ys = np.where(event_img > 0)
    pts = np.vstack((xs, ys)).T

    spydi_labels = my_spydi_dbscan(pts, eps_spydi, min_samples_val)

    sp_boxes = []

    for lbl in set(spydi_labels):

        if lbl == -1:
            continue

        cluster = pts[spydi_labels == lbl]

        if cluster.shape[0] == 0:
            continue

        x = cluster[:, 0]
        y = cluster[:, 1]

        diag = (x.max()-x.min())**2 + (y.max()-y.min())**2

        if diag >= mindiagonalsquared:
            sp_boxes.append((y.min(), x.min(), y.max(), x.max()))

    DBSCAN_sc = []
    SPYDI_sc = []

    DBSCAN_img = evnt_img_3c.copy()
    SPYDI_img = evnt_img_3c.copy()

    for gt in yolo_boxes:

        gx1, gy1, gx2, gy2, _, _ = gt

        cv2.rectangle(gray_img_3c, (gx1, gy1), (gx2, gy2), (0,255,255), 2)

        best_db = 0
        best_sp = 0

        for b in db_boxes:
            best_db = max(best_db, _compute_IOU_(b, gt))

        for b in sp_boxes:
            best_sp = max(best_sp, _compute_IOU_(b, gt))

        DBSCAN_sc.append(best_db)
        SPYDI_sc.append(best_sp)

    for b in db_boxes:
        cv2.rectangle(DBSCAN_img, (b[0], b[1]), (b[2], b[3]), (0,255,0), 2)

    for b in sp_boxes:
        cv2.rectangle(SPYDI_img, (b[0], b[1]), (b[2], b[3]), (255,0,0), 2)

    return gray_img_3c, DBSCAN_img, SPYDI_img, DBSCAN_sc, SPYDI_sc
