# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 13:04:26 2023

@author: b.vries1

Computes the Area Over the Perturbation Curve (AOPC) for a set of PET scans
and their corresponding XAI attribution maps, by progressively occluding the
voxels highlighted by the attribution map and measuring the resulting change
in the classifier's predicted probability.
"""
import data_process as dps
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# Update the paths below to point to your local data/weights directories
PET_DIR = 'path/to/Scans'
ATTRIBUTION_MAP_DIR = 'path/to/Attribution_maps'
MODEL_WEIGHTS_PATH = 'path/to/weights.h5'

filenames_PETS = dps.getfilenames(PET_DIR, ext1='.nii')
PETS, _, _ = dps.getlistimage(filenames_PETS)

filenames_XAIS = dps.getfilenames(ATTRIBUTION_MAP_DIR, ext1='.nii')
XAIS, _, _ = dps.getlistimage(filenames_XAIS)

classifier = keras.models.load_model(MODEL_WEIGHTS_PATH)
classifier.load_weights(MODEL_WEIGHTS_PATH)

AOPCS_L = []
AOPCS_H = []
for PET, XAI in zip(PETS, XAIS):
    PET_trans = (PET>(PET.max()*0.005))*PET
    PET_trans =  np.rot90(np.transpose(PET_trans,(0,2,1)),1)

    baseline = classifier.predict(PET_trans[None,...])
    idx_baseline = np.argmax(baseline)
    baseline = baseline[0,idx_baseline]
    print(f'Baseline probability: {baseline}')
    cc = len(np.where(XAI>0)[0])

    prob = classifier.predict(PET_trans[None,...])[0,idx_baseline]
    AOPC = baseline - prob
    print(f'AOPC: {AOPC}')
    if idx_baseline==0:
        AOPCS_L.append(AOPC)   
    else:
        AOPCS_H.append(AOPC)
    
    coor = np.where(PET_trans>0)
    idxs = list(np.linspace(0,len(coor[0]),int(cc/64),dtype='int'))
    for i in idxs[:-1]:
        x, y, z = coor[0][i], coor[1][i],coor[2][i]
        PET_trans[x-2:x+2,y-2:y+2,z-2:z+2] = 0
    plt.imshow(PET_trans[:,:,70])
    plt.show()
    prob = classifier.predict(PET_trans[None,...])[0,idx_baseline]
    AOPC = baseline - prob
    print(f'AOPC: {AOPC}')
    if idx_baseline==0:
        AOPCS_L.append(AOPC)   
    else:
        AOPCS_H.append(AOPC)

        
