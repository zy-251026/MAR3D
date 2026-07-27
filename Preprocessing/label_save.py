import os
import nibabel as nib
import SimpleITK as stik
import glob
import numpy
import pandas as pd
out_path = './dataset/OASIS.csv'
data_name = []
data_age = []
for i in range(12):
    origin_path = '/home/lab/2023brain_age/oasis_cross-sectional_disc' + str(i+1) + '/disc' + str(i+1) + '/'
    dir_paths = glob.glob(origin_path+'OAS1_*')
    for j in range(len(dir_paths)):
        dirs = glob.glob(dir_paths[j] + '/*.txt')
        k = pd.read_csv(dirs[0],header=None)
        data_name.append(k.iloc[0].values[0][-13:])
        data_age.append(k.iloc[1].values[0][-2:])
        #print(k.iloc[0].values[0][-14:],k.iloc[1].values[0][-3:])
data = pd.DataFrame([])
data['name'] = data_name
data['age'] = data_age
data.to_csv(out_path)
