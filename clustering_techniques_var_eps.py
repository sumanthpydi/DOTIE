#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import DBSCAN

NO_LABEL = 0

# ============================================================
# SPYDI
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
            if cnt < min_samples_val:
                continue

            core_map[y, x] = 1

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

def compute_iou(box1, box2):
    x1,y1,x2,y2 = box1
    gx1,gy1,gx2,gy2,_,_ = box2

    ix1 = max(x1, gx1)
    iy1 = max(y1, gy1)
    ix2 = min(x2, gx2)
    iy2 = min(y2, gy2)

    inter = max(0, ix2-ix1) * max(0, iy2-iy1)

    a1 = (x2-x1)*(y2-y1)
    a2 = (gx2-gx1)*(gy2-gy1)

    union = a1 + a2 - inter
    return inter/union if union > 0 else 0


# ============================================================
# BOX EXTRACTION
# ============================================================

def get_boxes(events, labels, mindiagonal=2300):
    boxes = []

    for lbl in set(labels):
        if lbl == -1:
            continue

        pts = events[labels == lbl]
        x = pts[:,0]
        y = pts[:,1]

        diag = (x.max()-x.min())**2 + (y.max()-y.min())**2

        if diag >= mindiagonal:
            boxes.append((int(y.min()), int(x.min()),
                          int(y.max()), int(x.max())))
    return boxes


# ============================================================
# MAIN
# ============================================================

def compare_all(model, evnt_frame_3chnl, gray_img_3chnl,
                recovered_3chnl, eps_sp, min_sp):

    results = model(gray_img_3chnl)
    detections = results.xyxy[0].cpu().numpy()

    yolo_boxes = [(int(x1),int(y1),int(x2),int(y2),conf,cls)
                  for x1,y1,x2,y2,conf,cls in detections]

    img = recovered_3chnl[:,:,0]
    Xx, Xy = np.where(img > 0)

    if len(Xx) == 0:
        return 0, 0

    events = np.vstack((Xx, Xy)).T

    # DBSCAN fixed
    db_labels = DBSCAN(eps=15, min_samples=10).fit_predict(events)
    db_boxes = get_boxes(events, db_labels)

    # SPYDI variable
    sp_labels = my_spydi_dbscan(events, eps_sp, min_sp)
    sp_boxes = get_boxes(events, sp_labels)

    best_db, best_sp = 0, 0

    for gt in yolo_boxes:
        best_db = max(best_db, max([compute_iou(b, gt) for b in db_boxes] or [0]))
        best_sp = max(best_sp, max([compute_iou(b, gt) for b in sp_boxes] or [0]))

    return best_db, best_sp
