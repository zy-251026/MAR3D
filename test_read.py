import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
path = glob.glob('E:/code/convert/data/orgin_nii/*')
for i in range(len(path)):
    data = np.array(nib.load(path[i]).dataobj)
    print(data.shape)
