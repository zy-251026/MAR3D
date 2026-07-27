import os
import nibabel as nib
import SimpleITK as stik
import glob
import numpy
import shutil
path1 = glob.glob('/home/lab/2023brain_age/ADNI1_Complete/ADNI2_part4/*')
num = 1
for i in range(len(path1)):
        path2 = glob.glob(path1[i]+'/*')
        for j in range(len(path2)):
                path3 = glob.glob(path2[j]+'/*')
                for k in range(len(path3)):
                        print(path3[k])
                        path4 = glob.glob(path3[k]+'/*')
                        name = path4[0][-6:]
                        path5 = glob.glob(path4[0]+'/*')
                        #img = nib.load(path5[0])
                        new = '/home/lab/braintransformer/dataset/ADNI1/' + 'ADNI1_' + name + '.nii'
                        print('image'+str(num)+'is copyed!')
                        num += 1
                        try:
                                shutil.copyfile(path5[0], new)
                        except:
                                pass
                        
