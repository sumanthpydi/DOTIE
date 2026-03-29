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


# ------------------------------------------------------------
# Create output folders
# ------------------------------------------------------------

os.makedirs("results", exist_ok=True)


if __name__ == "__main__":

    ############################################################
    # Load Data
    ############################################################

    evnts_enc = np.load('datasets/DOTIE_Encoding/count_data/500.npy')
    gray_idx = np.load('datasets/DOTIE_Encoding/count_data/gray_ind.npy')

    d_set = h5py.File('datasets/outdoor_day2_data.hdf5', 'r')
    gray_imgs = d_set['davis']['left']['image_raw']


    ############################################################
    # Load YOLO once
    ############################################################

    print("Loading YOLO model (Torch Hub)...")

    model = torch.hub.load(
        'ultralytics/yolov5',
        'yolov5s',
        pretrained=True
    )

    model.eval()


    ############################################################
    # Spiking architecture
    ############################################################

    conv1 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1,bias=False)

    conv1.weight = torch.nn.Parameter(
        torch.ones_like(conv1.weight) * 0.15
    )

    with torch.no_grad():
        conv1.weight[0,0,1,1] = 0.2

    snn1 = snn.Leaky(beta=0.3, reset_mechanism="subtract")
    mem_dir = snn1.init_leaky()


    ############################################################
    # Initialize grayscale pointer
    ############################################################

    indx_for_gray = 0
    while int(gray_idx[indx_for_gray]) < 300:
        indx_for_gray += 1

    gryimg = np.array(gray_imgs[indx_for_gray], dtype=np.uint8)


    ############################################################
    # PARAMETER GRID
    ############################################################

    spydi_eps_list = [13,12,11,10]
    spydi_minpts_list = [10,9,8,7]

    results_table = []


    ############################################################
    # Frame Loop
    ############################################################

    start_frame = 300
    end_frame = min(evnts_enc.shape[-1], 400)   # start small

    print("\n====================================================")
    print(f"Processing Frames {start_frame}–{end_frame-1}")
    print("====================================================")


    for curr_pos in range(start_frame, end_frame):

        print(f"\nFrame {curr_pos}")

        ########################################################
        # Extract frame
        ########################################################

        frame = evnts_enc[:,:,:,curr_pos]
        frame = np.sum(frame,axis=0)

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
        # Normalize event frame
        ########################################################

        denom = frame.max() - frame.min()

        if denom == 0:
            evnt_frame = np.zeros_like(frame, dtype=np.uint8)
        else:
            evnt_frame = ((frame - frame.min()) * (255/denom)).astype(np.uint8)

        evnt_frame_3chnl = convert_to_contrast_3chnl(evnt_frame)


        ########################################################
        # DOTIE output
        ########################################################

        spk_frame = torch.squeeze(spk_dir.detach()).cpu().numpy().astype(np.uint8)
        spk_frame[spk_frame>0] = 255

        recovered_inputs = recover_fast_inputs(
            evnt_frame,
            spk_frame,
            recovery_neighborhood=12
        )

        recovered_inputs_3chnl = convert_to_3chnl(recovered_inputs)


        ########################################################
        # FIXED DBSCAN
        ########################################################

        _, _, _, DBSCAN_sc, _ = compare_all(
            model,
            evnt_frame_3chnl,
            gryimg_3chnl,
            recovered_inputs_3chnl,
            evnt_frame,
            eps_dbscan=15,
            eps_spydi=13,
            min_samples_val=10,
            mindiagonalsquared=2300
        )

        db_iou = max(DBSCAN_sc) if DBSCAN_sc else 0


        ########################################################
        # SPYDI GRID SEARCH
        ########################################################

        for eps in spydi_eps_list:
            for minpts in spydi_minpts_list:

                _, _, _, _, SPYDI_sc = compare_all(
                    model,
                    evnt_frame_3chnl,
                    gryimg_3chnl,
                    recovered_inputs_3chnl,
                    evnt_frame,
                    eps_dbscan=15,
                    eps_spydi=eps,
                    min_samples_val=minpts,
                    mindiagonalsquared=2300
                )

                sp_iou = max(SPYDI_sc) if SPYDI_sc else 0

                results_table.append({
                    "Frame": curr_pos,
                    "SPYDI_eps": eps,
                    "SPYDI_minpts": minpts,
                    "DBSCAN_IoU": db_iou,
                    "SPYDI_IoU": sp_iou
                })

                print(f"eps={eps}, minpts={minpts} → SPYDI={sp_iou:.3f}, DBSCAN={db_iou:.3f}")


    ############################################################
    # CREATE DATAFRAME
    ############################################################

    df = pd.DataFrame(results_table)


    ############################################################
    # WINNER WITH TIE
    ############################################################

    epsilon = 1e-6

    df["Winner"] = np.where(
        np.abs(df["SPYDI_IoU"] - df["DBSCAN_IoU"]) < epsilon,
        "TIE",
        np.where(df["SPYDI_IoU"] > df["DBSCAN_IoU"], "SPYDI", "DBSCAN")
    )


    ############################################################
    # ACCURACY %
    ############################################################

    df["SPYDI_Accuracy_%"] = np.where(
        df["DBSCAN_IoU"] == 0,
        0,
        (df["SPYDI_IoU"] / df["DBSCAN_IoU"]) * 100
    )

    df["DBSCAN_Accuracy_%"] = 100


    ############################################################
    # SAVE CSV
    ############################################################

    df.to_csv("results/spydi_grid_search.csv", index=False)

    print("\nSaved → results/spydi_grid_search.csv")
