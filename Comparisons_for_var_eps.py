#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
import h5py
import pandas as pd

from visual_helpers import recover_fast_inputs, convert_to_3chnl, convert_to_contrast_3chnl
from clustering_techniques_var_eps import compare_all

eps_list = [13,12,11,10,9,8,7,6]
minpts_list = [10,9,8,7,6,5]

results = []

if __name__ == "__main__":

    evnts_enc = np.load('datasets/DOTIE_Encoding/count_data/500.npy')
    gray_idx = np.load('datasets/DOTIE_Encoding/count_data/gray_ind.npy')

    d_set = h5py.File('datasets/outdoor_day2_data.hdf5', 'r')
    gray_imgs = d_set['davis']['left']['image_raw']

    print("Loading YOLO...")
    model = torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)
    model.eval()

    conv1 = nn.Conv2d(1,1,3,1,1,bias=False)
    conv1.weight.data.fill_(0.15)
    conv1.weight.data[0,0,1,1] = 0.2

    snn1 = snn.Leaky(beta=0.3)
    mem = snn1.init_leaky()

    idx = 0
    while int(gray_idx[idx]) < 1900:
        idx += 1

    for frame_id in range(1900, 2001):

        print("Frame:", frame_id)

        frame = evnts_enc[0,:,:,frame_id] + evnts_enc[1,:,:,frame_id]

        inp = torch.tensor(frame).float().unsqueeze(0).unsqueeze(0)
        spk, mem = snn1(conv1(inp), mem)

        if int(gray_idx[idx]) == frame_id:
            gry = np.array(gray_imgs[idx], dtype=np.uint8)
            idx += 1

        gry_3 = convert_to_3chnl(gry)

        denom = frame.max()-frame.min()
        evnt = ((frame-frame.min())*(255/denom)).astype('uint8') if denom!=0 else np.zeros_like(frame)

        spk_frame = torch.squeeze(spk).numpy()
        spk_frame[spk_frame>0]=255

        rec = recover_fast_inputs(evnt, spk_frame, recovery_neighborhood=18)

        if np.sum(rec) < 50:
            continue

        rec3 = convert_to_3chnl(rec)
        ev3 = convert_to_contrast_3chnl(evnt)

        for eps in eps_list:
            for minpts in minpts_list:

                db, sp = compare_all(
                    model,
                    ev3,
                    gry_3,
                    rec3,
                    eps,
                    minpts
                )

                results.append({
                    "Frame": frame_id,
                    "eps_spydi": eps,
                    "minpts_spydi": minpts,
                    "DBSCAN_IoU": db,
                    "SPYDI_IoU": sp
                })

    df = pd.DataFrame(results)
    df.to_csv("results/spydi_sweep_1900_2000.csv", index=False)

    print("Saved results/spydi_sweep_1900_2000.csv")
