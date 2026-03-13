#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized DOTIE Comparison Script
DBSCAN (eps=15) vs SPYDI (eps=13)
YOLO loaded once using Torch Hub
"""

import snntorch as snn
import torch
import torch.nn as nn
import h5py
import numpy as np
import cv2
import os
os.makedirs("results", exist_ok=True)
from visual_helpers import convert_to_contrast_3chnl, recover_fast_inputs, convert_to_3chnl
from Clustering_techniques import compare_all


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

    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

    conv1.weight = torch.nn.Parameter(
        torch.ones_like(conv1.weight) * 0.15
    )

    with torch.no_grad():
        conv1.weight[0, 0, 1, 1] = 0.2

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
    # IoU storage
    ############################################################

    DBSCAN_all = []
    SPYDI_all = []

    print("\n====================================================")
    print("Processing Frames 300–399")
    print("====================================================")

    ############################################################
    # Main loop
    ############################################################

    for curr_pos in range(300, 400):

        print(f"\nFrame {curr_pos}")

        ########################################################
        # Extract frame from encoded tensor
        ########################################################

        frame = evnts_enc[:, :, :, curr_pos]
        frame = np.sum(frame, axis=0)

        inp_img = torch.tensor(frame).float().unsqueeze(0).unsqueeze(0)

        con_out = conv1(inp_img)
        spk_dir, mem_dir = snn1(con_out, mem_dir)

        ########################################################
        # Update grayscale image
        ########################################################

        if indx_for_gray < len(gray_idx):
            if int(gray_idx[indx_for_gray]) == curr_pos:
                gryimg = np.array(gray_imgs[indx_for_gray], dtype=np.uint8)
                indx_for_gray += 1

        gryimg_3chnl = convert_to_3chnl(gryimg)

        ########################################################
        # Normalize event frame
        ########################################################

        visual_frame = frame

        denom = visual_frame.max() - visual_frame.min()

        if denom == 0:
            evnt_frame = np.zeros_like(visual_frame, dtype=np.uint8)
        else:
            evnt_frame = ((visual_frame - visual_frame.min()) *
                          (255 / denom)).astype('uint8')

        evnt_frame_3chnl = convert_to_contrast_3chnl(evnt_frame)

        ########################################################
        # DOTIE recovered output
        ########################################################

        spk_frame = torch.squeeze(spk_dir.detach())
        spk_frame[spk_frame > 0] = 255
        spk_frame = np.array(spk_frame, dtype=np.uint8)

        recovered_inputs = recover_fast_inputs(
            evnt_frame,
            spk_frame,
            recovery_neighborhood=12
        )

        recovered_inputs_3chnl = convert_to_3chnl(recovered_inputs)

        ########################################################
        # Compare DBSCAN vs SPYDI
        ########################################################

        gray_image_3chnl, DBSCAN_img, SPYDI_img, DBSCAN_sc, SPYDI_sc = compare_all(
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

        ########################################################
        # Store frame IoU
        ########################################################

        db_frame_iou = max(DBSCAN_sc) if len(DBSCAN_sc) > 0 else 0
        sp_frame_iou = max(SPYDI_sc) if len(SPYDI_sc) > 0 else 0

        DBSCAN_all.append(db_frame_iou)
        SPYDI_all.append(sp_frame_iou)

        print(f"DBSCAN: {db_frame_iou:.4f} | SPYDI: {sp_frame_iou:.4f}")

        ########################################################
        # Visualization
        ########################################################

        visual = np.concatenate(
            (gray_image_3chnl, DBSCAN_img, SPYDI_img),
            axis=1
        )
        
        # Save images
        cv2.imwrite(f"results/frame_{curr_pos}.png", visual)
        cv2.imwrite(f"results/dbscan_{curr_pos}.png", DBSCAN_img)
        cv2.imwrite(f"results/spydi_{curr_pos}.png", SPYDI_img)
        
        # Show preview
        cv2.imshow("DBSCAN vs SPYDI", cv2.cvtColor(visual, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

    ############################################################
    # Final results
    ############################################################

    print("\n====================================================")
    print("Final Results (Frames 300–399)")
    print("----------------------------------------------------")

    db_mean = np.mean(DBSCAN_all) if len(DBSCAN_all) > 0 else 0
    sp_mean = np.mean(SPYDI_all) if len(SPYDI_all) > 0 else 0

    print(f"DBSCAN Mean IoU: {db_mean:.4f}")
    print(f"SPYDI   Mean IoU: {sp_mean:.4f}")

    if sp_mean > db_mean:
        print("SPYDI performs better.")
    elif db_mean > sp_mean:
        print("DBSCAN performs better.")
    else:
        print("Both perform equally.")

    print("====================================================")
