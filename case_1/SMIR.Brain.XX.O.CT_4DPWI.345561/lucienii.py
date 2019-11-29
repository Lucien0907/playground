import os
import cv2
import numpy as np
import SimpleITK as sitk
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim

def rescale_slices_cxy(slices, shape):
    """resize all slices of an array nii file, channels first"""
    resized = np.empty((slices.shape[0], shape[0], shape[1]), dtype= np.float32)
    for i in range(slices.shape[0]):
        resized[i] = cv2.resize(slices[i], shape)
    return resized

def rescale_slices_xyc(slices, shape):
    """resize all slices of an array nii file, channels last"""
    resized = np.empty((shape[0], shape[1], slices.shape[2]), dtype = np.float32)
    for i in range(slices.shape[2]):
        resized[:,:,i] = cv2.resize(slices[:,:,i], shape)
    return resized

def pad(img, shape):
    """pad an image to transorm it into a specified shape, does not cut the image
    if output size is smaller"""
    delta_h = shape[0]-img.shape[0]
    delta_w = shape[1]-img.shape[1]
    if delta_h > 0:
        up = delta_h//2
        down = delta_h-up
        img = np.vstack((np.zeros((up,img.shape[1]), dtype=np.float32), img))
        img = np.vstack((img, np.zeros((down,img.shape[1]), dtype=np.float32)))
    if delta_w > 0:
        left = delta_w//2
        right = delta_w-left
        img = np.hstack((np.zeros((img.shape[0],left), dtype=np.float32), img))
        img = np.hstack((img, np.zeros((img.shape[0], right), dtype=np.float32)))
    return img

def crop(img, shape):
    """pad an image to transorm it into a specified shape, does not cut the image
    if output size is smaller"""
    delta_h = img.shape[0]-shape[0]
    delta_w = img.shape[1]-shape[1]
    if delta_h > 0:
        up = delta_h//2
        down = delta_h-up
        img = img[up:-down,:]
    if delta_w > 0:
        left = delta_w//2
        right = delta_w-left
        img = img[:,left:-right]
    return img

def pad_crop(img, shape):
    """aplly padding and cropping to resize the current image without rescaling"""
    img = pad(img, shape)
    img = crop(img, shape)
    return img

def resize_slices_cxy(slices, shape):
    resized = np.empty((slices.shape[0], shape[0], shape[1]), dtype=np.float32)
    for i in range(slices.shape[0]):
        resized[i] = pad_crop(slices[i], shape)
    return resized

def resize_slices_xyc(slices, shape):
    resized = np.empty((shape[0], shape[1], slices.shape[2]), dtype=np.float32)
    for i in range(slices.shape[2]):
        resized[:,:,i] = pad_crop(slices[:,:,i], shape)
    return resized

def nii_to_png(nii_file, dst_path=None, name_idx=-2):
    itkimg = sitk.ReadImage(nii_file)
    img = sitk.GetArrayFromImage(itkimg)
    if dst_path == None:
        dst_path = nii_file.rstrip(nii_file.split('/')[-1]).rstrip('/')
    for i in range(len(img)):
        plt.imsave(dst_path+'/slice'+str(i)+'.png',img[i], cmap='gray')
    print("File #"+nii_file.split('/')[-2]+" saved "+str(i)+" slices")
    return dst_path

def niis_to_png(nii_files,dst_path=None):
    print("start converting "+str(len(nii_files))+" nii files")
    if dst_path!=None:
        dst = dst_path
    for x in nii_files:
        nii_to_png(x, dst_path=dst_path)
    print(str(len(nii_files))+" files converted")
    return

def correct_bias(in_file, out_file=None, image_type=sitk.sitkFloat64):
    """
    Corrects the bias using ANTs N4BiasFieldCorrection. If this fails, will then attempt to correct bias using SimpleITK
    :param in_file: nii文件的输入路径
    :param out_file: 校正后的文件保存路径名
    :return: 校正后的nii文件全路径名
    """
    if out_file == None:
        out_file = in_file.rstrip('.nii')+"_bias_corrected.nii"
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    try:
        done = correct.run()
        return done.outputs.output_image
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found."))

def normalization(x):
    mean = np.mean(x)
    std = np.std(x)
    out = (x-mean)/std
    print("Normalization done: mean="+str(mean)+", std="+str(std)+", dtype="+str(np.dtype(out[0][0][0])))
    return out

def nii_to_array(nii_path):
    itkimage = sitk.ReadImage(nii_path)
    img = sitk.GetArrayFromImage(itkimage)
    spacing = itkimage.GetSpacing()
    origin = itkimage.GetOrigin()
    direction = itkimage.GetDirection()
    print("Read array from: " + nii_path + ", dtype= " + str(img.dtype))
    return img, spacing, origin, direction

def save_as_nii(img, path, spacing=None, origin=None, direction=None):
    itkimage = sitk.GetImageFromArray(img)
    if spacing != None:
        itkimage.SetSpacing(spacing)
    if origin != None:
        itkimage.SetOrigin(origin)
    if direction != None:
        itkimage.SetDirection(direction)
    sitk.WriteImage(itkimage, path)
    print("Images saved as: "+path)
    return path

def mse(img1, img2):
    err = np.sum((img1.astype("float") - img2.astype("float"))**2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err
