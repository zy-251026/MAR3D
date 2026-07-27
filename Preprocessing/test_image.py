import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
#path = glob.glob('./preprocessing/normal/corr_0003001_e.nii.gz')
path3 = '/home/lab/Downloads/BMB_1_0003001_0003042/0003001/session_1/anat_1/anat.nii.gz'
path4 = '/home/lab/Downloads/BMB_1_0003001_0003042/0003001/session_1/rest_1/rest.nii.gz'
path5 = '/home/lab/Downloads/BMB_1_0003001_0003042/0003001/session_1/rest_2/rest.nii.gz'
#path3 = '/home/lab/2023brain_age/adnitest/1.nii'
#path4 = '/home/lab/2023brain_age/adnitest/2.nii'
#path5 = '/home/lab/2023brain_age/adnitest/3.nii'
alldata = []
alldata.append(np.array(nib.load(path3).dataobj))
alldata.append(np.array(nib.load(path4).dataobj))
alldata.append(np.array(nib.load(path5).dataobj))
num = 3
print(alldata[0].shape)
print(alldata[1].shape)
print(alldata[2].shape)
for i in range(num):
    plt.subplot(3,num,0*num + i + 1)
    plt.imshow(alldata[i][100,:,:])
    plt.subplot(3,num,1*num + i + 1)
    plt.imshow(alldata[i][:,130,:], cmap='gray')
    plt.subplot(3,num,2*num + i + 1)
    plt.imshow(alldata[i][:,:,100], cmap='gray')
    
    
# print(data_y)
#np.array(data_y)
#np.savetxt("age.csv", data_y, fmt="%f")
    #j = j + 1
# for i in range(num):
#     plt.subplot(3,num,j)
#     plt.imshow(data[i+100,:,:])
#     plt.subplot(3,num,j+num)
#     plt.imshow(data[:,i+100,:])
#     plt.subplot(3,num,j+num*2)
#     plt.imshow(data[:,:,i+100])
#     j = j + 1
plt.show()
