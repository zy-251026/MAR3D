import os
import nibabel as nib
import SimpleITK as stik
import glob
import numpy
import pandas as pd
in_path = './dataset/label/new_CoRR.csv'
allpath = glob.glob('./preprocessing/normal/*')
k = pd.read_csv(in_path,header=None)
for i in range(len(allpath)):
        flag = 0
        for j in range(1,len(k)):
                if allpath[i][-21:-7] == k.iloc[j][1]:
                        flag = 1
                        break
        if flag == 0:
                print('file:'+allpath[i][-21:-7]+'has no corresponding label!')
