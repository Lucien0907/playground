from lupy import *
import os
import sys
import math
import shutil
import numpy as np
import scipy.io as sio
import SimpleITK as sitk
import matplotlib.pyplot as plt

# find files
files = fsearch(path = '../data/ISLES2018/TRAINING/', suffix='.nii', include='4DPWI', sort_level=-3)
print('\n'.join(files))

if len(sys.argv) == 2:
    if not sys.argv[1] == "all":
        case = int(sys.argv[1])
        for x in files:
            if x.find('case_'+str(case)) >= 0:
                files = [x]
                print('selected: '+files[0].split('/')[-3])
                break
    for fin in files:
        print()
        print(fin.split('/')[-3])
        img = sitk.ReadImage(fin)
        arr = sitk.GetArrayFromImage(img)
        print('shape: ', arr.shape)
        print('spacing: ', img.GetSpacing())
        print('origin: ', img.GetOrigin())
        print('direction: ', img.GetDirection())
        print("raw data: max="+str(np.max(arr))+", min="+str(np.min(arr))+", dtype="+str(arr.dtype))

        # normalization
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        arr_ori =arr
        print("normalized: max="+str(np.max(arr))+", min="+str(np.min(arr))+", dtype="+str(arr.dtype))

        # specify the location and size of the window to search for aif and vof
        xroi = arr.shape[2]//2
        yroi = arr.shape[3]//2
        w = 200
        h = 200

        # create roi segments from original data
        arr_roi = arr[:,:,xroi-h//2:xroi+h//2, yroi-w//2:yroi+w//2]
        print('arr_roi shape:', arr_roi.shape)

        # segments to store max and max-min functions
        arr_max = np.amax(arr_roi, axis=0)
        arr_diff = np.amax(arr_roi, axis=0) - np.amin(arr_roi, axis=0)
        print('arr_max shape:', arr_max.shape, 'arr_diff shape:', arr_diff.shape)

        # select specific slice for aif and vof
        s = 0
        arr_roi = arr_roi[:,s,:,:]
        arr_max = arr_max[s]
        arr_diff = arr_diff[s]
        print('arr_roi shape: ', arr_roi.shape)

        # find locations of voxels after thresholding
        th = np.max(arr_diff)*0.31
        print('th =', th)

        for i in range(0, arr_roi.shape[1], 10):
            for j in range(0, arr_roi.shape[2], 10):
                v = arr_diff[i,j]
                if v > th:
                    plt.plot(arr_roi[:,i,j])
        plt.show()
"""
        # subfolder for storing heat maps
        folder = '../out/'+fin.split('/')[-3]
        #folder = os.path.dirname(fin)+ '/out'
        if not os.path.exists(folder):
            os.makedirs(folder)

        # copy raw data
        shutil.copyfile(fin, folder+'/4DPWI_'+fin.split('/')[-3]+'.nii')

        # extract and save aif and vof signals
        aif = arr_aif[:,0,aif_loc[0], aif_loc[1]]
        vof = arr_vof[:,-1,vof_loc[0], vof_loc[1]]
        print('aif shape: ',aif.shape, 'vof shape: ', vof.shape)
        sio.savemat(folder+'/av_'+fin.split('/')[-3]+'.mat', mdict={'AIF': aif,'VOF': vof})

        # plot aif and vof
        plt.cla()
        plt.clf()
        plt.plot(aif)
        plt.plot(vof)
        if len(files) == 1:
            plt.show()
        plt.savefig(folder+'/aif_vof_'+fin.split('/')[-3]+'.png')

        # plot heat maps
        arr_diff = np.amax(arr, axis=0) - np.amin(arr, axis=0)
        for i in range(arr.shape[1]):
            plt.cla()
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_yticks(range(0,256,20))
            ax.set_xticks(range(0,256,20))
            im = ax.imshow(arr_diff[i], cmap=plt.cm.hot_r)
            plt.colorbar(im)
            plt.title("heatmap, max-min")
            fig.savefig(folder+'/slice '+str(i+1))

        # print heat maps according to maximum difference
        #plt.imshow(aif_diff[0,:,:],cmap=plt.cm.hot_r)
        #plt.show()
        #plt.imshow(vof_diff[7,:,:],cmap=plt.cm.hot_r)
        #plt.show()
"""
