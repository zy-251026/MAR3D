import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
path = './dataset/'
paths = glob.glob(path+'IXI*.nii.gz')
image_name = 'IXI358-Guys-0919-T1.nii.gz'
pathc = '/home/lab/braintransformer/dataset/'+image_name
pathrs = '/home/lab/braintransformer/preprocessing/resample/'+image_name
pathss = '/home/lab/braintransformer/preprocessing/skullstrip/'+image_name
pathn4 = '/home/lab/braintransformer/preprocessing/n4/'+image_name
pathrg = '/home/lab/braintransformer/preprocessing/registration/'+image_name
pathnm = '/home/lab/braintransformer/preprocessing/normal/'+image_name
pathp = []
path_label = path + 'label/IXI.csv'
for i in range(len(paths)):
    pathp.append(paths[i][10:16])
j = 1
num = 5
a = pd.read_csv(path_label, header=0)
# a = np.load(path_label)
# print(pathp)
data_y = []
alldata = []
#alldata.append(np.array(nib.load(pathc).dataobj))
alldata.append(np.array(nib.load(pathrs).dataobj))
alldata.append(np.array(nib.load(pathss).dataobj))
alldata.append(np.array(nib.load(pathn4).dataobj))
alldata.append(np.array(nib.load(pathrg).dataobj))
alldata.append(np.array(nib.load(pathnm).dataobj))
for i in range(num):
    #data = np.array(nib.load(paths[i]).dataobj)
    #dataori = np.array(nib.load(pathss).dataobj)
    #pathcom = pathc + paths[i][10:]
    #print(pathcom)
    #data1 = np.array(nib.load(pathc).dataobj)
    #data2 = np.array(nib.load(pathrs).dataobj)
    #data3 = np.array(nib.load(pathss).dataobj)
    #data4 = np.array(nib.load(pathn4).dataobj)
    #data5 = np.array(nib.load(pathrg).dataobj)
    #data6 = np.array(nib.load(pathnm).dataobj)
    # fig = plt.figure(i+1)
    numage = a.loc[a.name==pathp[i]].age
    #if len(numage.values)!=0:
    #    data_y.append(numage.values[0])
    #numage = a[:,4]
    #plt.title("title:{}".format(numage.values[0]))
    plt.subplot(3,num,0*num + i + 1)
    plt.imshow(alldata[i][100,:,:])
    plt.subplot(3,num,1*num + i + 1)
    plt.imshow(alldata[i][:,100,:], cmap='gray')
    plt.subplot(3,num,2*num + i + 1)
    plt.imshow(alldata[i][:,:,100], cmap='gray')
    
    
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
