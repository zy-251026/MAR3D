import os
import nibabel as nib
import SimpleITK as stik
import glob
import numpy
import pandas as pd
in_path = './dataset/label/ADNI1_1.5T.csv.csv'
out_path = './dataset/label/ADNI.csv'
data_name = []
data_age = []
k = pd.read_csv(in_path,header=None)
for i in range(len(k)):
        if len(k.iloc[i][0]) == 4:
                index_name = 'CORR_000'+ k.iloc[i][0] + '_' + k.iloc[i][2][-1]
        elif len(k.iloc[i][0]) == 5:
                index_name = 'CORR_00'+ k.iloc[i][0]+ '_' + k.iloc[i][2][-1]
        else:
                index_name = 'CORR_'+ k.iloc[i][0] + '_' + k.iloc[i][2][-1]
        data_name.append(index_name)
        kk = 0
        while k.iloc[i-kk][4] == '#':
                kk += 1
        data_age.append(k.iloc[i-kk][4])
'''
for i in range(12):
    origin_path = '/home/lab/2023brain_age/oasis_cross-sectional_disc' + str(i+1) + '/disc' + str(i+1) + '/'
    dir_paths = glob.glob(origin_path+'OAS1_*')
    for j in range(len(dir_paths)):
        dirs = glob.glob(dir_paths[j] + '/*.txt')
        k = pd.read_csv(dirs[0],header=None)
        data_name.append(k.iloc[0].values[0][-13:])
        data_age.append(k.iloc[1].values[0][-2:])
        #print(k.iloc[0].values[0][-14:],k.iloc[1].values[0][-3:])
'''
data = pd.DataFrame([])
data['name'] = data_name
data['age'] = data_age
data.to_csv(out_path)
