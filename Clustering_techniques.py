#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helper codes for clustering
DBSCAN vs SPYDI comparison
"""

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

    # Forward scan
    for y in range(H):
        for x in range(W):
            if binary_img[y, x] == 0:
                continue

            cnt = neighbor_count(y, x)
            is_core = cnt >= min_samples_val
            core_map[y, x] = 1 if is_core else 0

            if not is_core:
                continue

            min_neighbor_label = None

            for yy in range(max(0, y - eps_val), min(H, y + eps_val + 1)):
                for xx in range(max(0, x - eps_val), min(W, x + eps_val + 1)):
                    if core_map[yy, xx]:
                        lbl = label_img[yy, xx]
                        if lbl != NO_LABEL:
                            if min_neighbor_label is None or lbl < min_neighbor_label:
                                min_neighbor_label = lbl

            if min_neighbor_label is not None:
                label_img[y, x] = min_neighbor_label
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

def _compute_IOU_(event_box, gt_box):
    xes, yes, xee, yee = event_box
    xgs, ygs, xge, yge, _, _ = gt_box

    aeb = max(0, (xee - xes)) * max(0, (yee - yes))
    agb = max(0, (xge - xgs)) * max(0, (yge - ygs))

    xis = max(xes, xgs)
    yis = max(yes, ygs)
    xie = min(xee, xge)
    yie = min(yee, yge)

    aib = max(0, (xie - xis)) * max(0, (yie - yis))
    aub = aeb + agb - aib

    if aub == 0:
        return 0.0

    return aib / aub


def _best_iou_with_any_gt(box, yolo_boxes):
    """
    Returns:
        best_iou (float)
        best_gt_box (tuple or None)
    """
    if len(yolo_boxes) == 0:
        return 0.0, None

    best_iou = 0.0
    best_gt = None

    for gt in yolo_boxes:
        iou = _compute_IOU_(box, gt)
        if iou > best_iou:
            best_iou = iou
            best_gt = gt

    return best_iou, best_gt


# ============================================================
# CLUSTER BOUNDARY EXTRACTION
# ============================================================

def getboundaries_other(event_input_3chnl,
                        eps_dbscan=15,
                        eps_spydi=13,
                        min_samples_val=10,
                        mindiagonalsquared=100):

    image_array = event_input_3chnl[:, :, 0]
    Ximage_x, Ximage_y = np.where(image_array > 0)

    if len(Ximage_x) == 0:
        return [], []

    selected_events = np.vstack((Ximage_x, Ximage_y)).T

    if selected_events.shape[0] == 0:
        return [], []

    ev_box_dbscan = []
    ev_box_spydi = []

    # DBSCAN
    clustering_dbscan = DBSCAN(
        eps=eps_dbscan,
        min_samples=min_samples_val
    ).fit_predict(selected_events)

    for lbl in set(clustering_dbscan):
        if lbl == -1:
            continue

        pts = selected_events[clustering_dbscan == lbl]
        if pts.shape[0] == 0:
            continue

        x_vals = pts[:, 0]
        y_vals = pts[:, 1]

        diagon = ((x_vals.max() - x_vals.min()) ** 2) + ((y_vals.max() - y_vals.min()) ** 2)

        if diagon >= mindiagonalsquared:
            ev_box_dbscan.append((
                int(y_vals.min()),
                int(x_vals.min()),
                int(y_vals.max()),
                int(x_vals.max())
            ))

    # SPYDI
    clustering_spydi = my_spydi_dbscan(
        selected_events,
        eps_spydi,
        min_samples_val
    )

    if clustering_spydi.size > 0:
        for lbl in set(clustering_spydi):
            if lbl == -1:
                continue

            pts = selected_events[clustering_spydi == lbl]
            if pts.shape[0] == 0:
                continue

            x_vals = pts[:, 0]
            y_vals = pts[:, 1]

            diagon = ((x_vals.max() - x_vals.min()) ** 2) + ((y_vals.max() - y_vals.min()) ** 2)

            if diagon >= mindiagonalsquared:
                ev_box_spydi.append((
                    int(y_vals.min()),
                    int(x_vals.min()),
                    int(y_vals.max()),
                    int(x_vals.max())
                ))

    return ev_box_dbscan, ev_box_spydi


# ============================================================
# MAIN COMPARISON
# ============================================================

def compare_all(model,
                evnt_inp_3chnl,
                gray_image_3chnl,
                isolated_evnts_3chnl,
                evnt_inp,
                eps_dbscan=15,
                eps_spydi=13,
                min_samples_val=10,
                mindiagonalsquared=100):

    # YOLO detection
    results = model(gray_image_3chnl)
    detections = results.xyxy[0].cpu().numpy()

    yolo_boxes = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        yolo_boxes.append((int(x1), int(y1), int(x2), int(y2), conf, cls))

    # Frame-level IoU lists
    DBSCAN_sc = []
    SPYDI_sc = []

    # Base image for visualization
    contrasted_inp = convert_to_contrast_3chnl(evnt_inp)
    DBSCAN_img = contrasted_inp.copy()
    SPYDI_img = contrasted_inp.copy()
    gray_image_vis = gray_image_3chnl.copy()

    # Get cluster boxes
    dbscan_boxes, spydi_boxes = getboundaries_other(
        isolated_evnts_3chnl,
        eps_dbscan,
        eps_spydi,
        min_samples_val,
        mindiagonalsquared
    )

    # ------------------------------------------------
    # Draw YOLO GT boxes on grayscale image
    # ------------------------------------------------
    for gt in yolo_boxes:
        x1, y1, x2, y2, conf, cls = gt
        cv2.rectangle(gray_image_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            gray_image_vis,
            f"GT {conf:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

    # ------------------------------------------------
    # Draw DBSCAN boxes and per-box IoU
    # ------------------------------------------------
    for idx, box in enumerate(dbscan_boxes):
        x1, y1, x2, y2 = box
        box_iou, _ = _best_iou_with_any_gt(box, yolo_boxes)

        cv2.rectangle(DBSCAN_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            DBSCAN_img,
            f"IoU={box_iou:.3f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    # ------------------------------------------------
    # Draw SPYDI boxes and per-box IoU
    # ------------------------------------------------
    for idx, box in enumerate(spydi_boxes):
        x1, y1, x2, y2 = box
        box_iou, _ = _best_iou_with_any_gt(box, yolo_boxes)

        cv2.rectangle(SPYDI_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            SPYDI_img,
            f"IoU={box_iou:.3f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

    # ------------------------------------------------
    # Frame-level max IoU for CSV
    # ------------------------------------------------
    for gt in yolo_boxes:
        best_db = 0.0
        best_sp = 0.0

        for box in dbscan_boxes:
            best_db = max(best_db, _compute_IOU_(box, gt))

        for box in spydi_boxes:
            best_sp = max(best_sp, _compute_IOU_(box, gt))

        DBSCAN_sc.append(best_db)
        SPYDI_sc.append(best_sp)

    return gray_image_vis, DBSCAN_img, SPYDI_img, DBSCAN_sc, SPYDI_sc
