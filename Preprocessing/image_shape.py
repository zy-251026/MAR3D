import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
paths = glob.glob('./preprocessing/n4/*.nii.gz')
for i in range(len(paths)):
    data = np.array(nib.load(paths[i]).dataobj)
    print(paths[i])
    print(data.shape)
