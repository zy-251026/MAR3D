import os
import nibabel as nib
import SimpleITK as stik
import glob
import numpy
import shutil
path1 = glob.glob('/home/lab/2023brain_age/CORR/SWU_1*')
for i in range(len(path1)):
        path2 = glob.glob(path1[i]+'/*')
        for j in range(len(path2)):
                name = 'CORR_'+path2[j][-7:]+'_'
                path3 = glob.glob(path2[j]+'/*')
                for k in range(len(path3)):
                        num = path3[k][-1]
                        if num == '0':
                                num = '9'
                        elif num == '1':
                                num = '3'
                        else:
                                num = str(int(num)-1+3)
                        name2 = name + num + '.nii.gz'
                        path4 = path3[k]+'/anat_2/anat.nii.gz'
                        #img = nib.load(path5[0])
                        new = '/home/lab/braintransformer/dataset/CORR/' + name2
                        try:
                                shutil.copyfile(path4, new)
                        except:
                                pass
                        #print('image'+str(num)+'is copyed!')
                        #shutil.copyfile(path4, new)
