import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
path = './dataset/'
paths = glob.glob(path+'IXI*.nii.gz')
pathc = '/home/lab/2023brain_age/IXI-T1/_bet/'
pathp = []
path_label = path + 'label/IXI.csv'
for i in range(len(paths)):
    pathp.append(paths[i][10:16])
j = 1
num = 4
a = pd.read_csv(path_label, header=0)
# a = np.load(path_label)
# print(pathp)
data_y = []
for i in range(num):
    data = np.array(nib.load(paths[i]).dataobj)
    #dataori = np.array(nib.load(pathss).dataobj)
    pathcom = pathc + paths[i][10:]
    #print(pathcom)
    data2 = np.array(nib.load(pathcom).dataobj)
    # fig = plt.figure(i+1)
    numage = a.loc[a.name==pathp[i]].age
    #if len(numage.values)!=0:
    #    data_y.append(numage.values[0])
    #numage = a[:,4]
    #plt.title("title:{}".format(numage.values[0]))
    #age.csv
    plt.subplot(6,num,j)
    plt.imshow(data[100,:,:])
    plt.subplot(6,num,j+num)
    plt.imshow(data[:,100,:])
    plt.subplot(6,num,j+num*2)
    plt.imshow(data[:,:,100])
    plt.subplot(6,num,j+num*3)
    plt.imshow(data2[100,:,:])
    plt.subplot(6,num,j+num*4)
    plt.imshow(data2[:,100,:])
    plt.subplot(6,num,j+num*5)
    plt.imshow(data2[:,:,100])
    
    
# print(data_y)
#np.array(data_y)
#np.savetxt("age.csv", data_y, fmt="%f")
    j = j + 1
# for i in range(num):
#     plt.subplot(3,num,j)
#     plt.imshow(data[i+100,:,:])
#     plt.subplot(3,num,j+num)
#     plt.imshow(data[:,i+100,:])
#     plt.subplot(3,num,j+num*2)
#     plt.imshow(data[:,:,i+100])
#     j = j + 1
plt.show()
