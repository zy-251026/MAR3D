import os
import nibabel as nib
import SimpleITK as stik
import glob
import numpy
import shutil
path1 = glob.glob('/home/lab/2023brain_age/CORR/*')
for i in range(len(path1)):
        path2 = glob.glob(path1[i]+'/*')
        for j in range(len(path2)):
                name = 'CORR_'+path2[j][-7:]+'_'
                path3 = path2[j]+'/session_1/anat_1/anat.nii.gz'
                name2 = name + 'e.nii.gz'
                new = '/home/lab/braintransformer/dataset/CORR/' + name2
                try:
                     shutil.copyfile(path3, new)
                except:
                     pass
                     
                path3 = path2[j]+'/session_1/anat_1/anat_inv1.nii.gz'
                name2 = name + '1.nii.gz'
                new = '/home/lab/braintransformer/dataset/CORR/' + name2
                try:
                     shutil.copyfile(path3, new)
                except:
                     pass
                     
                path3 = path2[j]+'/session_1/anat_1/anat_inv2.nii.gz'
                name2 = name + '2.nii.gz'
                new = '/home/lab/braintransformer/dataset/CORR/' + name2
                try:
                     shutil.copyfile(path3, new)
                except:
                     pass
                
                path3 = path2[j]+'/session_1/anat_1/anat_uni.nii.gz'
                name2 = name + '3.nii.gz'
                new = '/home/lab/braintransformer/dataset/CORR/' + name2
                try:
                     shutil.copyfile(path3, new)
                except:
                     pass
