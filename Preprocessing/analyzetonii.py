import os
import nibabel as nib
import SimpleITK as stik
import glob
import numpy
for i in range(12):
    origin_path = '/home/lab/2023brain_age/oasis_cross-sectional_disc' + str(i+1) + '/disc' + str(i+1) + '/'
    dir_paths = glob.glob(origin_path+'OAS1_*')
    for j in range(len(dir_paths)):
        dirs = glob.glob(dir_paths[j] + '/RAW/' + '*mpr-1_anon.hdr')
        out_path = './dataset/OASIS/' + dirs[0][-28:-19] + '.nii.gz'
        img = nib.load(dirs[0])
        temp_data = img.get_fdata()
        temp_data = numpy.squeeze(temp_data)
        temp_data = temp_data.transpose(2,1,0)
        data = stik.GetImageFromArray(temp_data)
        stik.WriteImage(data, out_path)
