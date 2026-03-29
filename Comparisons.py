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

if __name__ == "__main__":

    evnts_enc = np.load('datasets/DOTIE_Encoding/count_data/500.npy')
    gray_idx = np.load('datasets/DOTIE_Encoding/count_data/gray_ind.npy')

    d_set = h5py.File('datasets/outdoor_day2_data.hdf5', 'r')
    gray_imgs = d_set['davis']['left']['image_raw']

    model = torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)
    model.eval()

    conv1 = nn.Conv2d(1,1,3,1,1,bias=False)
    conv1.weight = torch.nn.Parameter(torch.ones_like(conv1.weight)*0.15)

    with torch.no_grad():
        conv1.weight[0,0,1,1] = 0.2

    snn1 = snn.Leaky(beta=0.3)
    mem_dir = snn1.init_leaky()

    start_frame = 1900
    end_frame = 1950

    indx_for_gray = 0
    while int(gray_idx[indx_for_gray]) < start_frame:
        indx_for_gray += 1

    gryimg = np.array(gray_imgs[indx_for_gray], dtype=np.uint8)

    eps_list = [13,12,11,10]
    minpts_list = [10,9,8]

    results = []

    for curr_pos in range(start_frame, end_frame):

        frame = evnts_enc[:,:,:,curr_pos]
        frame = np.sum(frame,axis=0)

        inp = torch.tensor(frame).float().unsqueeze(0).unsqueeze(0)
        spk, mem_dir = snn1(conv1(inp), mem_dir)

        if int(gray_idx[indx_for_gray]) == curr_pos:
            gryimg = np.array(gray_imgs[indx_for_gray], dtype=np.uint8)
            indx_for_gray += 1

        gry3 = convert_to_3chnl(gryimg)

        denom = frame.max()-frame.min()
        ev = np.zeros_like(frame) if denom==0 else ((frame-frame.min())*(255/denom)).astype(np.uint8)

        ev3 = convert_to_contrast_3chnl(ev)

        spk = torch.squeeze(spk.detach()).cpu().numpy().astype(np.uint8)
        spk[spk>0]=255

        rec = recover_fast_inputs(ev,spk,12)
        rec3 = convert_to_3chnl(rec)

        row = {"Frame":curr_pos}

        _,_,_,db_sc,_ = compare_all(model,ev3,gry3,rec3,ev,15,13,10,2300)
        row["DBSCAN"] = max(db_sc) if db_sc else 0

        for e in eps_list:
            for m in minpts_list:

                _,_,_,_,sp_sc = compare_all(model,ev3,gry3,rec3,ev,15,e,m,2300)

                row[f"S_{e}_{m}"] = max(sp_sc) if sp_sc else 0

        results.append(row)

    pd.DataFrame(results).to_csv("results/grid.csv",index=False)
