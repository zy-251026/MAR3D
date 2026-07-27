import os
import nibabel as nib
import SimpleITK as stik
import glob
import numpy
import shutil
path1 = glob.glob('/home/lab/braintransformer/preprocessing/normal/*002611*')
for i in range(len(path1)):
                name = path1[i][-21:]
                name1 = list(name)
                name1[-8] = '1'
                name1 = ''.join(name1)
                name2 = list(name)
                name2[-8] = '2'
                name2 = ''.join(name2)
                new1 = '/home/lab/braintransformer/preprocessing/normal/' + name1
                #new2 = '/home/lab/braintransformer/preprocessing/normal/' + name2
                #print(path1[i], new)
                try:
                        shutil.copyfile(path1[i], new1)
                        #shutil.copyfile(path1[i], new2)
                except:
                        pass
                        #print('image'+str(num)+'is copyed!')
                        #shutil.copyfile(path4, new)
