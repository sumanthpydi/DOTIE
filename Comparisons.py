#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import snntorch as snn
import torch
import torch.nn as nn
import h5py
import numpy as np
import cv2
import os
import pandas as pd

from visual_helpers import convert_to_contrast_3chnl, recover_fast_inputs, convert_to_3chnl
from Clustering_techniques import compare_all


os.makedirs("results", exist_ok=True)
os.makedirs("results/frames", exist_ok=True)


if __name__ == "__main__":

    evnts_enc = np.load('datasets/DOTIE_Encoding/count_data/500.npy')
    gray_idx = np.load('datasets/DOTIE_Encoding/count_data/gray_ind.npy')

    d_set = h5py.File('datasets/outdoor_day2_data.hdf5', 'r')
    gray_imgs = d_set['davis']['left']['image_raw']

    print("Loading YOLO...")
    model = torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)
    model.eval()

    conv1 = nn.Conv2d(1,1,3,1,1,bias=False)
    conv1.weight = torch.nn.Parameter(torch.ones_like(conv1.weight)*0.15)
    with torch.no_grad():
        conv1.weight[0,0,1,1] = 0.2

    snn1 = snn.Leaky(beta=0.3, reset_mechanism="subtract")
    mem_dir = snn1.init_leaky()

    indx_for_gray = 0
    while int(gray_idx[indx_for_gray]) < 300:
        indx_for_gray += 1

    gryimg = np.array(gray_imgs[indx_for_gray], dtype=np.uint8)

    DBSCAN_all, SPYDI_all, frame_list = [], [], []

    total_frames = evnts_enc.shape[-1]
    start_frame = 300
    end_frame = min(total_frames, 1200)   # LIMIT (CHANGE IF NEEDED)

    print(f"\nProcessing {start_frame} to {end_frame}")

    for curr_pos in range(start_frame, end_frame):

        print(f"Frame {curr_pos}")

        ########################################################
        # Better frame construction
        ########################################################
        frame = evnts_enc[0,:,:,curr_pos] + evnts_enc[1,:,:,curr_pos]

        inp_img = torch.tensor(frame).float().unsqueeze(0).unsqueeze(0)
        con_out = conv1(inp_img)
        spk_dir, mem_dir = snn1(con_out, mem_dir)

        ########################################################
        # Update grayscale
        ########################################################
        if indx_for_gray < len(gray_idx):
            if int(gray_idx[indx_for_gray]) == curr_pos:
                gryimg = np.array(gray_imgs[indx_for_gray], dtype=np.uint8)
                indx_for_gray += 1

        gryimg_3chnl = convert_to_3chnl(gryimg)

        ########################################################
        # Normalize
        ########################################################
        denom = frame.max() - frame.min()
        if denom == 0:
            evnt_frame = np.zeros_like(frame, dtype=np.uint8)
        else:
            evnt_frame = ((frame - frame.min())*(255/denom)).astype('uint8')

        evnt_frame_3chnl = convert_to_contrast_3chnl(evnt_frame)

        ########################################################
        # Stronger reconstruction
        ########################################################
        spk_frame = torch.squeeze(spk_dir.detach()).cpu().numpy().astype(np.uint8)
        spk_frame[spk_frame>0] = 255

        recovered_inputs = recover_fast_inputs(
            evnt_frame,
            spk_frame,
            recovery_neighborhood=18   # 🔥 IMPORTANT CHANGE
        )

        ########################################################
        # Skip empty frames
        ########################################################
        if np.sum(recovered_inputs) < 50:
            continue

        recovered_inputs_3chnl = convert_to_3chnl(recovered_inputs)

        ########################################################
        # Clustering
        ########################################################
        gray_img, DBSCAN_img, SPYDI_img, DBSCAN_sc, SPYDI_sc = compare_all(
            model,
            evnt_frame_3chnl,
            gryimg_3chnl,
            recovered_inputs_3chnl,
            evnt_frame,
            eps_dbscan=15,
            eps_spydi=11,
            min_samples_val=10,
            mindiagonalsquared=2300
        )

        ########################################################
        # IoU
        ########################################################
        db = max(DBSCAN_sc) if DBSCAN_sc else 0
        sp = max(SPYDI_sc) if SPYDI_sc else 0

        DBSCAN_all.append(db)
        SPYDI_all.append(sp)
        frame_list.append(curr_pos)

        print(f"DB:{db:.3f} SP:{sp:.3f}")

        ########################################################
        # Text
        ########################################################
        cv2.putText(DBSCAN_img,f"DB:{db:.3f}",(10,30),0,0.7,(0,0,255),2)
        cv2.putText(SPYDI_img,f"SP:{sp:.3f}",(10,30),0,0.7,(0,0,255),2)

        ########################################################
        # Save
        ########################################################
        h = min(gray_img.shape[0], DBSCAN_img.shape[0], SPYDI_img.shape[0])
        combined = np.concatenate((gray_img[:h], DBSCAN_img[:h], SPYDI_img[:h]),axis=1)

        cv2.imwrite(f"results/frames/frame_{curr_pos}.png", combined)
        cv2.imshow("Result", combined)
        cv2.waitKey(1)

    ############################################################
    # TABLE
    ############################################################

    df = pd.DataFrame({
        "Frame": frame_list,
        "DBSCAN_IoU": DBSCAN_all,
        "SPYDI_IoU": SPYDI_all
    })

    eps = 1e-6

    df["Winner"] = np.where(
        np.abs(df["SPYDI_IoU"]-df["DBSCAN_IoU"])<eps,
        "TIE",
        np.where(df["SPYDI_IoU"]>df["DBSCAN_IoU"],"SPYDI","DBSCAN")
    )

    sp_acc, db_acc = [], []

    for db, sp in zip(DBSCAN_all, SPYDI_all):

        if db == 0 and sp == 0:
            sp_acc.append(100)
            db_acc.append(100)
        elif db == 0:
            sp_acc.append(100)
            db_acc.append(0)
        else:
            sp_acc.append((sp/db)*100)
            db_acc.append(100)

    df["SPYDI_Accuracy_%"] = np.round(sp_acc,2)
    df["DBSCAN_Accuracy_%"] = np.round(db_acc,2)

    df.to_csv("results/iou_comparison.csv", index=False)

    print("\nSaved → results/iou_comparison.csv")

    cv2.destroyAllWindows()
