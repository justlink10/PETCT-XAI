# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:21:06 2021

@author: b.vries
"""


def getfilenames(data_path=None,ext1='.none',ext2='.none'):
    """The function walks through the specified directory and
    extracts the paths of the files based on user defined extension
    
    Input:

    :param data_path: data path to directory where data is stored
    :type data_path: str
    :param ext1, ext2: extension to find path to filenames
    :type ext1, ext2: str

    Output:

    :param list_paths: data paths to directory where data is stored
    :type list_paths: list

    """
    from os import walk
    import sys
    if data_path==None:
        print('TypeError: data_path is not a valid path')
        sys.exit()
    
    for (dirpath, dirnames, filenames) in walk(data_path):
        filenames = sorted(filenames)
        break
    list_paths = []
    for i in filenames:
        if i.endswith(ext1):
            path = data_path + '/' + i
            list_paths.append(path)
        elif i.endswith(ext2):
            path = data_path + '/' + i
            list_paths.append(path)
    list_paths = sorted(list_paths)
    
    if not list_paths:
        print('DataError: list is empty')
        sys.exit()
    
    return list_paths

def getlistimage(list_paths_img):
    """The function obtains the data from the list of paths
    
    Input:
        
    :param list_path_img: data paths to directory where image data is stored
    :type list_path_img: list
    
    Output:
        
    :param list_img: images 
    :type list_img: lists
    """
    
    import numpy as np
    import nibabel as nib

    print('Loading patient data')
    nr_scans = len(list_paths_img)
    print(f'Number of subjects: {nr_scans}' )

    list_imgs = []
    affine_imgs = []
    
    default_shape_img_4mm = (160, 160, 500)
    default_shape_img_2mm = (320, 320, 1000)
    default_shape_img_4mm_prostate = (64, 64, 64)

    for path_img in list_paths_img:
        # Load Image data
        print('-----------------------------------------')
        print(f'Path of image data: {path_img}')
        img = nib.load(path_img)
        affine = img.affine
        img = img.get_fdata()
        #Remove noise
        img = (img>(img.max()*0.005))*img
        img = np.rot90(np.transpose(img,(0,2,1)),1)
        
        shape_img = np.shape(img)
        if shape_img == default_shape_img_4mm or default_shape_img_2mm or default_shape_img_4mm_prostate:
            print(f'Img shape: {shape_img}')
        else:
            print(f'ShapeError: Shape Seg {shape_img} is not similar to default shape {default_shape_img_4mm} or {default_shape_img_2mm} or {default_shape_img_4mm_prostate}')
            print('Please correct shape of the input images or contact the developer')

        list_imgs.append(img)
        affine_imgs.append(affine)
    print(f'Finished loading the {nr_scans} scans')
    return list_imgs, affine_imgs, nr_scans

def getlistSegs(list_paths_segs):
    """The function obtains the data from the list of paths
    
    Input:
    :param list_path_seg: data paths to directory where segmentation data is stored
    :type list_path_seg: list
    
    Output:
        
    :param list_seg: segmentation
    :type list_seg: lists
    """
    
    import numpy as np
    import nibabel as nib
    import sys
    
    print('Loading patient data')
    nr_segs = len(list_paths_segs)
    print(f'Number of subjects: {nr_segs}' )

    list_seg = []
       
    default_shape_seg= (320, 320, 1000)
    

    for path_seg in list_paths_segs:   
        # Load Segmentation data                
        print(f'Path of segmentation data: {path_seg}')
        seg = nib.load(path_seg)
        seg = seg.get_fdata()
        shape_seg = np.shape(seg)
        
        if shape_seg==default_shape_seg:
            print(f'Img shape: {shape_seg}')
        else:
            print(f'ShapeError: Shape Seg {shape_seg} is not similar to default shape {default_shape_seg}')
            sys.exit()

    list_seg.append(seg)
    
    return list_seg, nr_segs


def sliding_window(image, stepSize_x,stepSize_y,stepSize_z, windowSize):
    # slide a window across the 3D-image
    for x in range(0, image.shape[0], stepSize_x):
        for y in range(0, image.shape[1], stepSize_y):
            for z in range(0, image.shape[2], stepSize_z):
                yield (x, y, z, image[x:x + windowSize[0], y:y + windowSize[1], z:z + windowSize[2]])
                
def patching(images, winX, winY, winZ):                
    import numpy as np
    
    print('Patching.....')  
    list_patches = []
    for image in images:        
        for (x,y,z,window) in sliding_window(image,stepSize_x=int(winX),stepSize_y=int(winY),
                                             stepSize_z=int(winZ),windowSize=(winX,winY,winZ)):
                
            if window.shape[0] != winX or window.shape[1] != winY or window.shape[2] != winZ:        
                continue
            
            if window.shape[0] == winX and window.shape[1] == winY and window.shape[2] == winZ:        

                patch = image[x:x+window.shape[0],y:y+window.shape[1],z:z+window.shape[2]]
                
                list_patches.append(patch)
                   
    list_patches = np.array(list_patches) 
    
    return list_patches

def ratio(labels,cv,oversample):
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    from sklearn.utils import shuffle
    kfold = StratifiedKFold(n_splits=cv, shuffle=False)    

    train_list = []
    test_list = []
    #Walks through the KFold
    for train, test in kfold.split(labels,labels):
        positive_scans_idx = np.where(labels[train]==1)
        negative_scans_idx = np.where(labels[train]==0)
            
        positive_scans_idx = np.array(positive_scans_idx)
        negative_scans_idx = np.array(negative_scans_idx)

        positive_scans_idx = positive_scans_idx[0,:]
        negative_scans_idx = negative_scans_idx[0,:]
        positive_scans_nr = len(positive_scans_idx)
        negative_scans_nr = len(negative_scans_idx)
        
        #Calculates the ratio (int) which is used to overfit the minority
        if negative_scans_nr > positive_scans_nr:
            ratio = np.round(negative_scans_nr*(1-(positive_scans_nr/negative_scans_nr))).astype('int64')             
        else:
            ratio = np.round(positive_scans_nr*(1-(negative_scans_nr/positive_scans_nr))).astype('int64') 

        train  = train.tolist()
        if oversample==True:
            for x in range(ratio):
                idx_over = positive_scans_idx[x]
                train.append(idx_over)
            train = shuffle(train, random_state=5)

        train_list.append(np.array(train))
        test_list.append(np.array(test))
    return train_list, test_list            

def oversampling_class(list_imgs, labels, train, test):
    import numpy as np
    import scipy
    X_train_over = []
    X_test = []
    Y_train_over = []
    Y_test = []
    
    for nr in test:            
        X_test.append(list_imgs[nr])
        Y_test.append(labels[nr])
        
    (unique, counts) = np.unique(train, return_counts=True)
    for nr, count in zip(unique,counts):
        X_train_over.append(list_imgs[nr])
        Y_train_over.append(labels[nr])
        if count > 1:
            print(f'Oversampling {unique} - {count} times')
            for i in range(count-1):
                img = list_imgs[nr]
                img = scipy.ndimage.interpolation.shift(img, (np.random.randint(-4,4),np.random.randint(-4,4), 0),order=0, mode='nearest')
                img = scipy.ndimage.rotate(img,float(np.around(np.random.uniform(-6.0,6.0,size=1),2)),reshape=False,order=0, mode='nearest')
                X_train_over.append(img)
                Y_train_over.append(labels[nr])
        
    return X_train_over, X_test, Y_train_over, Y_test 

def axial_coronal_sagital(list_imgs, axial, coronal, sagital):
    import numpy as np
    list_imgs_orien = []
    if axial == True:
        list_imgs_orien = list_imgs
    elif coronal == True:
        for img in list_imgs:
            img = np.rot90(np.transpose(img,(0,2,1)),1)
            list_imgs_orien.append(img)
    elif sagital == True:
        for img in list_imgs:
            img = np.rot90(np.transpose(img,(1,2,0)),1)
            list_imgs_orien.append(img) 
    return list_imgs_orien
            
    
def oversampling_seg(list_img, list_seg, labels):
    """The function oversamples the minority classes/segments
    
    Input:
        
    :param list_img, list_seg: images and segmentation
    :type list_img, list_seg: lists
    :param ratio_os: the ratio to oversample minority class
    :type ratio_os: int
    :param split_ratio: the ratio to split the data into train and validation data
    :type lsplit_ratio: float

    Output:
        
    :param X_train_os, X_test, Y_train_os, Y_test: (oversampled) images and segmentation split into train and val data
    :type list_img, list_seg: numpy arrays
    
    """
    from sklearn.model_selection import train_test_split
    import numpy as np
    import scipy

    # Perform train test split
    X_train, X_test, Y_train, Y_test = train_test_split(list_img, list_seg,
                                                        test_size = 0.2,
                                                        random_state=2)
    
    
    (unique, counts) = np.unique(Y_train, return_counts=True)
    
    counts_labels = [counts[label] for label in labels]
    max_label = np.max(counts_labels)
    ratios_os = counts_labels/max_label
    
    X_train_os = []
    Y_train_os = []
    X_train_os.append(X_train)
    Y_train_os.append(Y_train)
    
    print('Oversampling the minority class')
    for x, y in zip(X_train,Y_train):
        for label, ratio_os in zip(labels,ratios_os):
            X_train_os.append(x)
            Y_train
        
        
        
        if np.max(y[:,:,1:]) == 1:
           for i in range(int(ratio_os)):           
               if i == 0:
                   X_train_os.append(x)
                   Y_train_os.append(y.astype('bool'))
               else:
                   rot = float(np.around(np.random.uniform(-12.0,12.0,size=1),2))
                       
                   x = scipy.ndimage.rotate(x,rot,reshape=False,order=0, mode='nearest')
                   X_train_os.append(x)
                       
                   y = scipy.ndimage.rotate(y,rot,reshape=False,order=0, mode='nearest')
                   Y_train_os.append(y.astype('bool'))
    
    return np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)
