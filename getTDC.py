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
        xaif = arr.shape[2]//2
        yaif = arr.shape[3]//2
        xvof = arr.shape[2]*1//6
        yvof = yaif
        wa = 35
        ha = 35
        wv = 60
        hv = 25

        # create aif/vof segments from original data
        arr_aif = arr[:,:,xaif-ha:xaif+ha, yaif-wa:yaif+wa]
        arr_vof = arr[:,:,xvof-hv:xvof+hv, yvof-wv:yvof+wv]
        print('aif shape:', arr_aif.shape, ',vof shape:', arr_vof.shape)

        # segments to store max and max-min functions
        aif_max = np.amax(arr_aif, axis=0)
        aif_diff = np.amax(arr_aif, axis=0) - np.amin(arr_aif, axis=0)
        vof_max = np.amax(arr_vof, axis=0)
        vof_diff = np.amax(arr_vof, axis=0) - np.amin(arr_vof, axis=0)
        print('aif_max shape:', aif_max.shape, 'aif_diff shape:', aif_diff.shape)
        print('vof_max shape:', vof_max.shape, ',vof_diff shape:', vof_diff.shape)

        # select specific slices for aif and vof
        sa = 0
        sv = -1
        aif_max = aif_max[sa]
        aif_diff = aif_diff[sa]
        vof_max = vof_max[sv]
        vof_diff = vof_diff[sv]

        # find location of maximum difference
        aif_loc = np.unravel_index(np.argmax(aif_diff), aif_diff.shape)
        vof_loc = np.unravel_index(np.argmax(vof_diff), vof_diff.shape)
        print('aif_loc: ', aif_loc, 'vof_loc: ', vof_loc)

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
        #name_aif = 'aif_' + case
        #name_vof = 'vof_' + case
        #sio.savemat('aif_vof.mat', mdict={name_aif: aif, name_vof: vof})
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

