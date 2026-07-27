import nibabel as nib
import glob
paths = glob.glob('./preprocessing/normal2/IXI*.nii.gz')
errorp = []
for i in range(len(paths)):
    try:
        b = nib.load(paths[i])
    except:
        errorp.append(paths[i])
print(len(errorp))
'''
paths = './preprocessing/normal/IXI389-Guys-0930-T1.nii.gz'
nib.load(paths)
'''
