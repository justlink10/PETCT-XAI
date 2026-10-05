# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 09:15:53 2022

@author: b.vries1

Computes SUV-based and spatial occlusion attribution maps for a PET/CT
classifier, per anatomical structure and as a combined global attribution
map, using an AOPC (Area Over the Perturbation Curve) sampling approach.
"""
import data_process as dps
import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow.keras as keras
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import scipy
import itertools

def local_window(image, coor, stepSize_x,stepSize_y,stepSize_z, windowSize):
    # slide a window across the 3D-image
    coor_x, coor_y, coor_z = coor[0],coor[1],coor[2]
    x_min, x_max = np.min(coor_x), np.max(coor_x)
    y_min, y_max = np.min(coor_y), np.max(coor_y)
    z_min, z_max = np.min(coor_z), np.max(coor_z)    
    for x in range(x_min,x_max,stepSize_x):
        for y in range(y_min,y_max,stepSize_y):
            for z in range(z_min,z_max,stepSize_z):
                yield (x, y, z, image[int(x):int(x + windowSize[0]),
                                      int(y):int(y + windowSize[1]), 
                                      int(z):int(z + windowSize[2])])


# Update the paths below to point to your local weights/data directories
MODEL_WEIGHTS_PATH = 'path/to/weights.h5'
PET_DIR = 'path/to/Test_data/PET'
SEG_DIR = 'path/to/Test_data/SEG'

classifier = keras.models.load_model(MODEL_WEIGHTS_PATH)
classifier.summary()

# load weights into new model
classifier.load_weights(MODEL_WEIGHTS_PATH)
print("Loaded model from disk")

filenames_PETS = dps.getfilenames(PET_DIR, ext1='.nii')
PETS, affine_imgs, nr_scans = dps.getlistimage(filenames_PETS)

filenames_SEGS = dps.getfilenames(SEG_DIR, ext1='.nii')
SEGS, affine_imgs, nr_scans = dps.getlistimage(filenames_SEGS)

for PET, SEG, filename, PET_affine in zip(PETS, SEGS, filenames_PETS, affine_imgs):
    print(filename[-11:-4])
    PET = (PET>(PET.max()*0.005))*PET
    PET =  np.rot90(np.transpose(PET,(0,2,1)),1)
    SEG =  np.rot90(np.transpose(SEG,(0,2,1)),1)

    max_SEG = np.max(SEG)
    print(f'Max labels is: {max_SEG}')
    structures = list(np.linspace(1,max_SEG,max_SEG).astype(int))
    
    if (max_SEG==103) & (len(np.where(SEG==4)[0])>10000):
        print('Galbladder not in data, adding zeros')
        SEG = np.where(SEG>3,SEG+1,SEG)
        max_SEG = np.max(SEG)
        print(f'Max labels is: {max_SEG} after correction')
    if (max_SEG==102) & (len(np.where(SEG==12)[0])>1000):
        print('Adrenal gland and galbladder not in data, adding zeros')
        SEG = np.where(SEG>3,SEG+1,SEG)
        SEG = np.where(SEG>11,SEG+1,SEG)
        max_SEG = np.max(SEG)
        print(f'Max labels is: {max_SEG} after correction')
    else:
        print('Data correct')
    
    PET_target = PET*np.where(SEG>0,0,1)
    SEG+=np.where(PET_target>0,max_SEG + 1,0)

    max_SEG = np.max(SEG)
    print(f'Max labels is: {max_SEG}')
    structures = list(np.linspace(1,max_SEG,max_SEG).astype(int))
    #Get the benchmark probability
    baseline = classifier.predict(PET[None,...])
    idx_baseline = np.argmax(baseline)
    baseline = baseline[0,idx_baseline]
    print(f'Baseline probability: {baseline}')

    kernel = np.ones((5, 5, 5), np.uint8)
    attribution_map_SUV = np.zeros(np.shape(SEG))
    #Stores the SUV conditions used in the sensitivity analysis
    mask_whole_image = np.zeros(np.shape(SEG))
    structs = []

    for organ in structures[:105]:
        print(f'Organ: {organ}')
        attribution_map_organ = np.zeros(np.shape(SEG))
        non_target = np.where(SEG==organ,0,1)    
        target = np.where(SEG==organ,1,0)
        coor = np.where(SEG==organ)
        volume_organ = len(coor[0])
        print(f'Volume organ: {volume_organ}')
        count=0
        if volume_organ>0:        
            PET_non_target = PET*non_target
            PET_target = PET*target
            
            SUV_prev = 0        
            max_target = int(np.max(PET_target))+1
            SUV_range_target = list(np.linspace(1, max_target, max_target-1, endpoint=False))
            for SUV in SUV_range_target:

                PET_th = np.where((PET_target>=SUV_prev) & (PET_target<SUV), 0, PET_target)
                background_and_healthy = PET_th + PET_non_target
                background_and_healthy = scipy.ndimage.gaussian_filter(background_and_healthy,sigma = 1)
    
                mask = np.where((PET_target>=SUV_prev) & (PET_target<SUV), 1, 0)
                mask = mask * np.where(SEG==organ,1,0)
                
                if (np.shape(mask)[0] > np.shape(kernel)[0])&(np.shape(mask)[1] > np.shape(kernel)[1])&(np.shape(mask)[2] > np.shape(kernel)[2])&(organ!=105):
                    mask_dil_healthy = scipy.ndimage.morphology.binary_dilation(mask, kernel, iterations=1)
                    mask_dil_healthy = mask_dil_healthy * np.where(SEG==organ,1,0)
                else:
                    mask_dil_healty = mask
                
                PET_sampled = np.where(mask_dil_healthy==1,background_and_healthy,PET)
                AOPC_healthy = (baseline - (classifier.predict(PET_sampled[None,...])[0,idx_baseline]))
    
                if AOPC_healthy > 0:
                    if count==0:
                        structs.append(organ)

                    imp = (AOPC_healthy/baseline)*100
                    print(f'Percentage interpreted {imp}% with current condition with SUV: {SUV_prev}-{SUV}' )

                    attribution_map_organ += np.where(mask_dil_healthy == 1 , imp, 0)
                    
                    mask_whole_image += mask
    
                    mask_dil_condition = scipy.ndimage.morphology.binary_dilation(mask_whole_image, kernel, iterations=1)
                    mask_dil_condition = mask_dil_condition *  np.isin(SEG, [structs],invert=False)
    
                    background_and_conditions = np.where(mask_whole_image == 1, 0, PET)
                    background_and_conditions = scipy.ndimage.gaussian_filter(background_and_conditions,sigma = 1)
                    
                    PET_2_abl = np.where(mask_dil_condition==1,background_and_conditions,PET)
    
                    pred_local = baseline - (classifier.predict(PET_2_abl[None,...])[0,idx_baseline])
                    imp = (pred_local/baseline)*100
                    print(f'Percentage interpreted {imp}% with current conditions')
                    count+=1
                SUV_prev = SUV
        
            attribution_map_SUV += attribution_map_organ

    PET_XAI_local =  np.transpose(np.rot90(PET_2_abl,-1),(0,2,1))
    PET_XAI_local = nib.Nifti1Image(PET_XAI_local.astype('float32'), PET_affine)
    nib.save(PET_XAI_local, f'{filename}_sampled.nii')
    
    PET_XAI_local =  np.transpose(np.rot90(attribution_map_SUV,-1),(0,2,1))
    PET_XAI_local = nib.Nifti1Image(PET_XAI_local.astype('float32'), PET_affine)
    nib.save(PET_XAI_local, f'{filename[:-4]}_sampled_SUV_attribution_map.nii')

    attribution_map_SUV_spat = np.zeros(np.shape(SEG))
    kernel = np.ones((5, 5, 5), np.uint8)

    for organ in structures[:105]:
        PET_spat_2_abl = PET_2_abl.copy()
        attribution_map_organ_spat= np.zeros(np.shape(SEG))    
        print(f'Organ: {organ}')
        target = np.where(SEG==organ,1,0)
        coor = np.where(((attribution_map_SUV>0)*target)>0)
        volume_organ = len(coor[0])
        print(f'Volume organ after SUV sampling: {volume_organ}')
        
        if len(coor[0])>0:
            PET_target = PET*target
            (winX,winY,winZ) = (4,4,4)   
            
            stepSize_x=winX
            stepSize_y=winY
            stepSize_z=winZ
    
            if  len(coor[0])>50000:
                print('Increasing stepsize because of high workload')
                stepSize_x=winX*4
                stepSize_y=winY*4
                stepSize_z=winZ*4
                
            for (x,y,z,window) in local_window(PET,coor,stepSize_x=stepSize_x,stepSize_y=stepSize_y,stepSize_z=stepSize_z,windowSize=(winX,winY,winZ)):
                if window.shape[0] != winX or window.shape[1] != winY or window.shape[2] != winZ:        
                    continue
                
                if window.shape[0] == winX and window.shape[1] == winY and window.shape[2] == winZ: 
                    if np.nanpercentile(PET_target[int(x-winX):int(x + winX),int(y-winX):int(y + winX),int(z-winX):int(z + winX)],q=25)>0:
                        volume_kernel = winX+1*winY+1*winZ+1
                        PET2sample = PET.copy()

                        PET2sample[int(x-winX):int(x + winX),
                                   int(y-winX):int(y + winX),
                                   int(z-winX):int(z + winX)] = 0

                        PET_spat_2_abl[int(x-(winX+2)):int(x + (winX+2)),
                                       int(y-(winX+2)):int(y + (winX+2)),
                                       int(z-(winX+2)):int(z + (winX+2))] = PET2sample[int(x-(winX+2)):int(x + (winX+2)),
                                                                                       int(y-(winX+2)):int(y + (winX+2)),
                                                                                       int(z-(winX+2)):int(z + (winX+2))]

                                                                                                                             
                        attribution_map_organ_spat[int(x-winX):int(x + winX),
                                                    int(y-winX):int(y + winX),
                                                    int(z-winX):int(z + winX)] = (baseline - (classifier.predict(PET2sample[None,...])[0,idx_baseline]))/volume_kernel
    
                    elif np.nanpercentile(PET_target[int(x-winX):int(x + winX/2),int(y-winX/2):int(y + winX/2),int(z-winX/2):int(z + winX/2)],q=25)>0:
                        winX_temp, winY_temp, winZ_temp = winX/2, winY/2, winZ/2
                        volume_kernel = winX_temp+1*winY_temp+1*winZ_temp+1
                        PET2sample = PET.copy()

                        PET2sample[int(x-winX_temp):int(x + winX_temp),
                                   int(y-winX_temp):int(y + winX_temp),
                                   int(z-winX_temp):int(z + winX_temp)] = 0

                        PET_spat_2_abl[int(x-(winX+2)):int(x + (winX+2)),
                                       int(y-(winX+2)):int(y + (winX+2)),
                                       int(z-(winX+2)):int(z + (winX+2))] = PET2sample[int(x-(winX+2)):int(x + (winX+2)),
                                                                                       int(y-(winX+2)):int(y + (winX+2)),
                                                                                       int(z-(winX+2)):int(z + (winX+2))]
            
                                                                                                                    
                        attribution_map_organ_spat[int(x-winX_temp):int(x + winX_temp),
                                                    int(y-winX_temp):int(y + winX_temp),
                                                    int(z-winX_temp):int(z + winX_temp)] = (baseline - (classifier.predict(PET2sample[None,...])[0,idx_baseline]))/volume_kernel
    
                    elif np.nanpercentile(PET_target[int(x-winX):int(x + winX/4),int(y-winX/4):int(y + winX/4),int(z-winX/4):int(z + winX/4)],q=25)>0:
                        winX_temp, winY_temp, winZ_temp = winX/4, winY/4, winZ/4
                        volume_kernel = winX_temp+1* winY_temp+1 * winZ_temp+1
                        PET2sample = PET.copy()

                        PET2sample[int(x-winX_temp):int(x + winX_temp),
                                   int(y-winX_temp):int(y + winX_temp),
                                   int(z-winX_temp):int(z + winX_temp)] = 0

                        PET_spat_2_abl[int(x-(winX+2)):int(x + (winX+2)),
                                       int(y-(winX+2)):int(y + (winX+2)),
                                       int(z-(winX+2)):int(z + (winX+2))] = PET2sample[int(x-(winX+2)):int(x + (winX+2)),
                                                                                       int(y-(winX+2)):int(y + (winX+2)),
                                                                                       int(z-(winX+2)):int(z + (winX+2))]
            
                                                                                                                    
                        attribution_map_organ_spat[int(x-winX_temp):int(x + winX_temp),
                                                   int(y-winX_temp):int(y + winX_temp),
                                                   int(z-winX_temp):int(z + winX_temp)] = (baseline - (classifier.predict(PET2sample[None,...])[0,idx_baseline]))/volume_kernel
    
                    else:
                        volume_kernel = 1
                        PET2sample = PET.copy()

                        PET2sample[int(x):int(x),
                                   int(y):int(y),
                                   int(z):int(z)] = 0
                        
                        PET_spat_2_abl[int(x):int(x),int(y):int(y), int(z):int(z)] = PET2sample[int(x):int(x),int(y):int(y), int(z):int(z)]
    
                        
                        attribution_map_organ_spat[int(x):int(x),
                                                    int(y):int(y),
                                                    int(z):int(z)] = (baseline - (classifier.predict(PET2sample[None,...])[0,idx_baseline]))/volume_kernel
            #Optimize attribution map per organ          
            pred_local = 0
            def_attribute_value = 0
            percentage = np.linspace(0,1,10,endpoint=False).tolist()
            for perc in percentage:
                local_temp_bin = np.where(attribution_map_organ_spat > np.max(attribution_map_organ_spat)*perc, 1,0).astype('uint8')      
                local_temp = np.where(local_temp_bin==1, 0, PET)
                local_temp_bin_dil = scipy.ndimage.morphology.binary_dilation(local_temp_bin, kernel, iterations=1)
                
                coor_x, _, _ = np.where(attribution_map_organ_spat > np.max(attribution_map_organ_spat)*perc)
                if len(coor_x)>0:
                    local_temp = np.where(local_temp_bin_dil==1,local_temp,PET)
                    pred_local_temp = baseline - (classifier.predict(local_temp[None,...])[0,idx_baseline])
            
                    if pred_local_temp>pred_local:
                        def_perc = perc
                        def_attribute_value = np.max(attribution_map_organ_spat)*perc
                        local = local_temp
                        pred_local = pred_local_temp
                        imp_spat = (pred_local/baseline)*100       
                        print(f'Percentage interpreted {imp_spat}% with {def_perc*100}% of max of current organ')
        
            attribution_map_organ_spat = np.where((attribution_map_organ_spat > def_attribute_value) | (attribution_map_organ_spat < 0), attribution_map_organ_spat, 0)
    
                     
        attribution_map_SUV_spat += attribution_map_organ_spat
    
    #Optimize attribution map for all organss
    pred_local = 0
    def_attribute_value = 0
    percentage = np.linspace(0,1,10,endpoint=False).tolist()
    for perc in percentage:
        print(perc)
        local_temp_bin = np.where(attribution_map_SUV_spat > np.max(attribution_map_SUV_spat)*perc, 1,0).astype('uint8')      
        local_temp = np.where(local_temp_bin==1, 0, PET)
        local_temp_bin_dil = scipy.ndimage.morphology.binary_dilation(local_temp_bin, kernel, iterations=1)
    
        coor_x, _, _ = np.where(attribution_map_SUV_spat > np.max(attribution_map_SUV_spat)*perc)
        if len(coor_x)>0:
            local_temp = np.where(local_temp_bin_dil==1,local_temp,PET)
            pred_local_temp = baseline - (classifier.predict(local_temp[None,...])[0,idx_baseline])
            if pred_local_temp>pred_local:
                def_perc = perc
                def_attribute_value = np.max(attribution_map_SUV_spat)*perc
                local = local_temp
                pred_local = pred_local_temp
                imp = (pred_local/baseline)*100       
                print(f'Percentage interpreted {imp}% with {def_perc*100}% of max of all organs')
    
    attribution_map_SUV_spat = np.where((attribution_map_SUV_spat > def_attribute_value) | (attribution_map_SUV_spat < 0), attribution_map_SUV_spat, 0)
    
    PET_2_abl = local
                 
    PET_XAI_local =  np.transpose(np.rot90(PET_2_abl,-1),(0,2,1))
    PET_XAI_local = nib.Nifti1Image(PET_XAI_local.astype('float32'), PET_affine)
    nib.save(PET_XAI_local, f'{filename}_sampled_SUV_spat.nii')
        
    PET_XAI_local =  np.transpose(np.rot90(attribution_map_SUV_spat,-1),(0,2,1))
    PET_XAI_local = nib.Nifti1Image(PET_XAI_local.astype('float32'), PET_affine)
    nib.save(PET_XAI_local, f'{filename[:-4]}_SUV_spat_attribution_map.nii')
    
    #Find instances/organs where attribution > 0
    new_structures = []
    for organ in structures[:105]:
        attribution = np.where(SEG==organ,attribution_map_SUV_spat,0)
        if np.max(attribution) > 0:
            new_structures.append(organ)
            
    print(f'Positive structures: {new_structures}')
    
    combinations_organs = []
    coalition = [1,2,len(new_structures)-1]
    for r in coalition:
        for combination in itertools.combinations(new_structures, r):
            combinations_organs.append(combination)
        
    preds_global = []
    for coalition in combinations_organs:
        binary = np.isin(SEG, [coalition],invert=False)
        local_temp = np.where(binary!=0,PET_2_abl,PET)
        AOPC = baseline - (classifier.predict(local_temp[None,...])[0,idx_baseline])
        print(f'Coalition {coalition} with AOPC: {AOPC*100}%')                                                                              
        preds_global.append(AOPC)
    
    idx = np.argmax(preds_global)
    
    print(f'Best AOPC: {(preds_global[idx]/baseline)*100}%')
    print(f'Best coalition: {combinations_organs[idx]}')
    AOPC = (preds_global[idx]/baseline)*100
    
    global_scores= np.zeros(len(new_structures))
    for coalition, score in zip(combinations_organs, preds_global):
        for organ in coalition:
            global_scores[new_structures.index(organ)] += score

    
    global_scores = global_scores/np.max(global_scores)
    print(global_scores)
    
    attribution_map_global = np.zeros(np.shape(SEG))
    for idx, organ in enumerate(new_structures):
        attribution_map_global += np.where(SEG==organ,(attribution_map_SUV_spat*global_scores[idx]), 0)
    
    PET_XAI_local =  np.transpose(np.rot90(attribution_map_global,-1),(0,2,1))
    PET_XAI_local = nib.Nifti1Image(PET_XAI_local.astype('float32'), PET_affine)
    nib.save(PET_XAI_local, f"{filename[:-4]}_global_attribution_map.nii")
    
    df = pd.DataFrame(data = global_scores).T
    df.columns = new_structures
    df.to_excel(excel_writer = f"{filename[:-4]}_XAI_global_scores.xlsx")
    
    SUV_ranges_max = list(np.linspace(0, 100, 100, endpoint=False))
    columns = []
    volumes = []
    for organ in structures[:105]:
        print(f'Organ: {organ}')
        PET_organ = PET*np.where(SEG==organ,1,0)
        PET_organ = PET_organ*np.where(attribution_map_SUV_spat>0,1,0)
        SUV_prev = 0
        for SUV in SUV_ranges_max:
            volume = len((np.where((PET_organ>=SUV_prev) & (PET_organ<SUV)))[0])
            columns.append(f'{organ};{SUV_prev};{SUV}')
            volumes.append(volume)
                
            SUV_prev = SUV
    columns.append('Class')
    volumes.append(idx_baseline)
    columns.append('AOPC SUV')
    volumes.append(AOPC)
    df = pd.DataFrame(data = volumes).T
    df.columns = columns
    
    df.to_excel(excel_writer = f"{filename[:-4]}_XAI_SUV.xlsx")
