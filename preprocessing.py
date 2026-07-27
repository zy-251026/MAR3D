# -*- coding: utf-8 -*-
# @Time    : 2019/7/17 12:42 PM
# @Author  : weiziyang
# @FileName: preprocessing.py
# @Software: PyCharm
import os
import copy

import deepbrain
import numpy as np
from sklearn.mixture import GaussianMixture
import nibabel as nib
import SimpleITK as sitk
import nibabel.processing
import subprocess
from skimage import morphology
#from intensity_normalization.normalize import gmm
import glob

class Processor(object):
    def __init__(self, file_name, skull_strip=False):
        self.template_path = './MNI152_T1_1mm_brain.nii.gz'
        self.environment_path = 'enviorment.txt'
        self.file_name = file_name
        self._skull_strip = skull_strip
        self.obj_name = self.file_name
        self.agent = nib.load(self.file_name)
        self.resample_path = './preprocessing/resample/'+file_name[10:]
        self.n4_bias_path = './preprocessing/n4/'+file_name[10:]
        self.registration_path = './preprocessing/registration/'+file_name[10:]
        self.segmentation_path = './preprocessing/segmentation/'+file_name[10:-7]
        self.skullstrip_path = './preprocessing/skullstrip/'+file_name[10:]
        self.matrix_path = './preprocessing/registration/'+file_name[10:-6]+'mat'
        self.normalisation_path = './preprocessing/normal/'+file_name[10:]

    def init_env(self):
        f = open(self.environment_path, 'r')
        print(self.registration_path)
        #config_file = f.read()
        content = f.readlines()
        f.close()
        config_dict = dict()
        for line in content:
            print(line)
            left, right = line.split('=')
            config_dict[left] = right
        os.environ.update(config_dict)

    def resample(self, resolution=(1, 1, 1)):
        self.agent = nib.processing.resample_to_output(self.agent, resolution)
        nib.save(self.agent, self.resample_path)

    def n4_bias_correction(self):
        inputImage = sitk.ReadImage(self.skullstrip_path)
        #inputImage = np.array(self.agent.dataobj)
        maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
        inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected_image = corrector.Execute(inputImage, maskImage)
        #log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
        #output = inputImage/sitk.Exp(log_bias_field)
        sitk.WriteImage(corrected_image, self.n4_bias_path)

    def skull_strip(self):
            copy_img = copy.copy(np.array(self.agent.dataobj))
            #copyagent = copy.copy(self.agent)
            extractor = deepbrain.Extractor()
            prob = extractor.run(copy_img)
            copy_img[prob < 0.5] = 0
            binary = copy.copy(copy_img)
            binary[binary > 0] = 1
            labels = morphology.label(binary)
            labels_num = [len(labels[labels == each]) for each in np.unique(labels)]
            rank = np.argsort(np.argsort(labels_num))
            index = list(rank).index(len(rank) - 2)
            new_img = copy.copy(copy_img)
            #new_img[labels != index] = 0
            #sitk.WriteImage(new_img, self.skullstrip_path)
            #copyagent.dataobj = new_img
            self.agent.dataobj[labels != index] = 0
            #print(copy_img.shape, new_img.shape)
            nib.save(self.agent, self.skullstrip_path)

    def template_registration(self):
        #subprocess.run('/home/lab/fsl/share/fsl/bin/flirt -ref {} -in {} -out {} -omat {}'.format(self.template_path, self.n4_bias_path, self.registration_path, self.matrix_path), shell=True)
        #os.environ["PATH"] += '/home/lab/fsl/bin/flirt'
        #print('/home/lab/fsl/bin/flirt -ref {} -in {} -out {} -omat {}'.format(self.template_path, self.n4_bias_path, self.registration_path, self.matrix_path))
        os.system("flirt -ref {} -in {} -omat {}".format(self.template_path, self.n4_bias_path, self.matrix_path))
        os.system("flirt -ref {} -in {} -applyxfm -init {} -out {}".format(self.template_path, self.n4_bias_path, self.matrix_path, self.registration_path))
        #os.system(f"/home/lab/fsl/bin/flirt -ref {self.template_path} -in {self.n4_bias_path}"
                  #f"  -out {self.registration_path} -omat {self.matrix_path}")
        #outputs = os.popen(f"/home/lab/fsl/bin/flirt -ref {self.template_path} -in {self.n4_bias_path} -out {self.registration_path}")
        #command = "/home/lab/fsl/bin/flirt -in {} -ref {} -schedule /home/lab/fsl/etc/flirtsch/ztransonly.sch -out {} -omat {}".format(self.n4_bias_path, self.template_path, self.registration_path, self.matrix_path)
        #result = subprocess.run(command, shell=True, capture_output=True)
        #print(result.returncode)
        #command2 = "/home/lab/fsl/bin/flirt -ref {} -in {} -applyxfm -init {} -out {}".format(self.template_path, self.n4_bias_path, self.matrix_path, self.registration_path)
        #result2 = subprocess.run(command2, shell=True, capture_output=True)
        #print(result2.returncode)
        #nib.save(result2, self.registration_path)
    
    def gmm(self, agent):
        data = np.array(agent.dataobj)
        brain = np.expand_dims(data[data > np.mean(data)].flatten(), 1)
        gmm = GaussianMixture(3)
        gmm.fit(brain)
        means = sorted(gmm.means_.T.squeeze())
        grey_matter, white_matter = means[1], means[2]
        data = data/white_matter
        C = grey_matter / white_matter
        a = (0.75 - C ** 2) / (C - C ** 2)
        data[data < 1] = a * data[data < 1] + (1 - a) * data[data < 1]**2
        x = data[data>0]
        # value, grid = np.histogram(x.flatten(), bins=100, range=(0.2, 1.5), density=True)
        # plt.plot(grid[:-1], value)
        # plt.show()
        return nib.Nifti1Image(data, agent.affine, agent.header)
        
    def gmm_normalisation(self):
        agent = nib.load(self.registration_path)
        new_agent = self.gmm(agent)
        nib.save(new_agent, self.normalisation_path)

    def auto_segmentation(self):
        os.system(f"fast -o {self.segmentation_path} {self.registration_path}")

    def clean(self):
        pass

    def start(self):
        self.init_env()
        self.resample()
        self.skull_strip()
        self.n4_bias_correction()
        self.template_registration()
        self.gmm_normalisation()
        #self.auto_segmentation()
        #self.clean()


if __name__ == "__main__":
    '''
        paths = glob.glob('./dataset/IXI*.nii.gz')
        for i in range(len(paths)):
            Processor(paths[i]).start()
            print("Image{}/{}:preprocessing finish.".format(i+1,len(paths)))
    '''
    paths = glob.glob('./preprocessing/normal/IXI*.nii.gz')
    errorp = []
    for i in range(len(paths)):
        try:
            b = nib.load(paths[i])
        except:
            errorp.append(paths[i][23:])
    for i in range(len(errorp)):
            path = './dataset/' + errorp[i]
            Processor(path).start()
            print("Image{}/{}:preprocessing finish.".format(i+1,len(errorp)))
    #path = './dataset/IXI389-Guys-0930-T1.nii.gz'
    #Processor(path).start()
    
