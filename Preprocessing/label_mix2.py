import os
import nibabel as nib
import SimpleITK as stik
import glob
import numpy
import pandas as pd
in_path1 = './dataset/label/IXI.csv'
in_path2 = './dataset/label/OASIS.csv'
in_path3 = './dataset/label/sub_information.csv'
in_path4 = './dataset/label/Subject_Information.csv'
in_path5 = './dataset/label/ADNI1_1.5T.csv'
in_path6 = './dataset/label/ADNI1_1.5T2.csv'
in_path7 = './dataset/label/ADNI1_1.5T3.csv'
in_path8 = './dataset/label/CoRR.csv'
out_path = './dataset/all_label.csv'

data_name = []
data_age = []
'''
k = pd.read_csv(in_path1,header=None)
for i in range(1,len(k)):
        data_name.append(k.iloc[i].values[2])
        data_age.append(k.iloc[i].values[3])
k2 = pd.read_csv(in_path2,header=None)
for i in range(1,len(k2)):
        data_name.append(k2.iloc[i].values[1][:9])
        data_age.append(k2.iloc[i].values[2])
k3 = pd.read_csv(in_path3,header=None)
for i in range(1,len(k3)):
        SALD_name = 'sub-'+ k3.iloc[i].values[0]
        data_name.append(SALD_name)
        data_age.append(k3.iloc[i].values[2])
k4 = pd.read_csv(in_path4,header=None)
for i in range(1,len(k4)):
        DLBS_name = '00'+k4.iloc[i].values[0]
        data_name.append(DLBS_name)
        data_age.append(k4.iloc[i].values[1])
'''
k5 = pd.read_csv(in_path5,header=None)
for i in range(1,len(k5)):
        ADNI_name = 'ADNI1_'+k5.iloc[i].values[0][-6:]
        #print(ADNI_name)
        data_name.append(ADNI_name)
        if k5.iloc[i].values[2] == 'CN':
                data_age.append(0)
        elif k5.iloc[i].values[2] == 'MCI':
                data_age.append(1)
        else:
                data_age.append(2)

k6 = pd.read_csv(in_path6,header=None)
for i in range(1,len(k6)):
        ADNI_name = 'ADNI1_'+k6.iloc[i].values[0][-6:]
        #print(ADNI_name)
        data_name.append(ADNI_name)
        if k6.iloc[i].values[2] == 'CN':
                data_age.append(0)
        elif k6.iloc[i].values[2] == 'MCI':
                data_age.append(1)
        else:
                data_age.append(2)
k7 = pd.read_csv(in_path7,header=None)
for i in range(1,len(k7)):
        ADNI_name = 'ADNI1_'+k7.iloc[i].values[0][-6:]
        #print(ADNI_name)
        data_name.append(ADNI_name)
        if k7.iloc[i].values[2] == 'CN':
                data_age.append(0)
        elif k7.iloc[i].values[2] == 'MCI':
                data_age.append(1)
        else:
                data_age.append(2)
# k8 = pd.read_csv(in_path8,header=None)
# for i in range(1,len(k8)):
#         if len(k8.iloc[i].values[0][:]) == 4:
#                 index_name = '000'+ k8.iloc[i].values[0][:]
#         elif len(k8.iloc[i].values[0][:]) == 5:
#                 index_name = '00'+ k8.iloc[i].values[0][:]
#         else:
#                 index_name = k8.iloc[i].values[0][:]
#         CORR_name = 'CORR_'+index_name+'_'+k8.iloc[i].values[2][-1]
#         print(CORR_name)
        #print(ADNI_name)
        #data_name.append(CORR_name)
        #data_age.append(k8.iloc[i].values[4])

data = pd.DataFrame([])
data['name'] = data_name
data['age'] = data_age
data.to_csv(out_path)
