import os
import torch
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import nibabel as nib
from torch import optim
from datetime import datetime
from model.Resnet3D import resnet10,resnet18, resnet34, resnet50, resnet101
from model.MAR3D import MAR3D_10, MAR3D_18, MAR3D_34, MAR3D_50, MAR3D_101
from model.swin_transformer import SwinTransformer
from model.RelationTransformer import RelationNet
from model.vit_3d import ViT3D
from model.novel_model.models.DenseNet import generate_model
from model.novel_model.models.EfficientNet import EfficientNet3D
from model.novel_model.models import ShuffleNetV2
from model.M3T.M3T import M3T
from model.models.pcrlv2_model_3d import PCRLv23d
import matplotlib.pyplot as plt
import glob

from torch.utils.data import Dataset, DataLoader

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_num', help="Input the amount of GPU you need", default=1, type=int)
    parser.add_argument('--GPU_no', help="Input the No of GPU you want", default='', type=str)
    parser.add_argument('--batch_size', help="Input the batch number", default=1, type=int)
    parser.add_argument('--epoch', help="Input the number of epoch number", default=50, type=int)
    parser.add_argument('--pretrain', help="Input if you need a pre trained model", default=False, type=bool)
    parser.add_argument('--model', help="Input your model", default='Aresnet3D', type=str)
    parser.add_argument('--model_depth', help="Input your resnet depth", default=34, type=int)
    parser.add_argument('--lr', help='Input learning rate', default=0.001, type=float)
    parser.add_argument('--norm', help='Input your normalisation method', default=' ', type=str)
    
    parser.add_argument('--flag', default='', type=str)
    args = parser.parse_args()
    return args


class MyDataset(Dataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        #self.y = np.loadtxt(os.path.join(dir_name, 'label', 'age.csv'))
        self.paths = glob.glob('./preprocessing/normal/'+self.dataset_name+'*.nii.gz')
        self.ypath = os.path.join('dataset', 'label', 'label2.csv')
        self.yy = pd.read_csv(self.ypath, header=0)
        self.y = []
        self.defic = []
        for i in range(len(self.paths)):
            pathp = self.paths[i]
            #print(self.yy.values[:][1])
            #print(self.yy.loc[self.yy.name==pathp].age)
            #print([j in pathp for j in self.yy.name])
            numage = self.yy.loc[[j in pathp for j in self.yy.name]].age
            #numage = self.yy.loc[self.yy.name==pathp].age
            if len(numage.values)!=0:
                self.y.append(numage.values[0])
                #print(self.yy[i].name)  
            else:
                #print(pathp)
                self.defic.append(i)
        self.paths = np.array(self.paths)
        self.paths = np.delete(self.paths, self.defic)
        self.y = np.array(self.y)

    def __getitem__(self, index):
        #print(self.paths[index])
        self.agent = nib.load(self.paths[index])
        data = np.array(self.agent.dataobj)
        data = data[np.newaxis, :, :, :]
        data = torch.from_numpy(data)
        data = data.unsqueeze(0)
        data = torch.nn.functional.interpolate(data, size=[192,192,192], mode='trilinear', align_corners=False) #将原本的182*216*182的图像扩展成了192*192*192的标准正方体图像，2维也可用
        data = data.squeeze(0)
        return data, torch.tensor(self.y[index]).float()

    def __len__(self):
        return self.y.shape[0]


if __name__ == "__main__":
    data_index = ['CORR', 'IXI', 'OAS1', 'sub', '0028', 'ADNI1', '']
    name_index = ['0', '1', '2', '3', '4', '', '_all']
    model_index = ['Aresnet']
    for jj in range(1):
        #index = ['10', '18', '34', '50', '101']
        #index = ['18_2024-04-06']
        for ii in range(7):
            dindex = data_index[ii]
            args = init()
            path_index = model_index[jj] + '3d' + '_model'
            #pathi = glob.glob('./model/' + path_index +'*')
            #args.model_depth = int(index[ii])
            #print(args.model_depth)
            max_loss = 1000
            dataset_name = 'Alldataset_'
            model_surname = model_index[jj] + '3d'
            os.makedirs('data/'+dataset_name+model_surname+str(args.model_depth)+'/', exist_ok = True)
            para_name = '_'.join([key + ':' + str(value) for key, value in args.__dict__.items()])
            t = datetime.now()
            str_time = t.strftime("%Y-%m-%d_%H:%M:%S")
            train_save = 'data/'+dataset_name+model_surname+str(args.model_depth)+'/predict'+name_index[ii]+'.csv'
            test_save = 'data/'+dataset_name+model_surname+str(args.model_depth)+'/true'+name_index[ii]+'.csv'
            model_name = './model/trained_model/'+model_surname+'_model.pkl'
            #if not os.path.exists('logging'):
            #    os.mkdir('logging')
            #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename='logging/'+dataset_name+model_surname+str(args.model_depth)+'_'+str_time+'.log')
            #logger = logging.getLogger(__name__)
            #logging.info('Start training, parameter:'+para_name)
            print("start training!")

            assert args.GPU_num <= torch.cuda.device_count(), 'GPU exceed the maximum num'
            if torch.cuda.is_available():
                if args.GPU_no:
                    device = torch.device("cuda:"+args.GPU_no[0])
                else:
                    device = torch.device("cuda:0")
            else:
                device = torch.device('cpu')
            if jj == 3:
            #model = eval(str(args.model)+str(args.model_depth)+'()')
                model = generate_model(121)
            elif jj == 4:
                model = EfficientNet3D.from_name('efficientnet-b4', override_params={'num_classes': 1}, in_channels=1)
            elif jj == 5:
                model = M3T(1, 32, 256, 8, 1)
            elif jj == 0:
                model = PCRLv23d()
            elif jj == 1:
                model = ShuffleNetV2.get_model(sample_size=192, num_classes=1, width_mult=1., in_channels=1)

            #model = ViT3D(image_size=(192,192,192), patch_size=16, num_classes=1, dim=16, depth=8, heads=16, mlp_dim=3072, dropout=0.1, emb_dropout=0)
            old_model = torch.load(model_name, map_location=None if torch.cuda.is_available() else 'cpu')
            old_model_dict = {key.replace('module.', ''): value for key, value in old_model.state_dict().items()}
            model_dict = {key: value for key, value in old_model_dict.items() if key in model.state_dict().keys()}
            model.load_state_dict(model_dict)
            model.to(device)
            #model = ViT(image_size=224, image_patch_size=16, frames=224, frame_patch_size=16, num_classes=1, dim=2048, depth=6, heads=24, mlp_dim = 4096).to(device)
            #model = SwinTransformer(img_size = 224, patch_size = 4, frame_size = 224, frame_patch_size = 4).to(device)
            #model = RelationNet(in_dim = 1, num_classes=1, num_transformer_blocks=2, drop_rate=0, im_dim='3d', max_pool_on_image=True, share_backbone=True)
            if args.pretrain:
                device = torch.device('cpu')
                old_model = torch.load('pretrain/resnet34.pkl', map_location=None if torch.cuda.is_available() else 'cpu')
                old_model_dict = {key.replace('module.', ''): value for key, value in old_model.state_dict().items()}
                model_dict = {key: value for key, value in old_model_dict.items() if key in model.state_dict().keys()}
                model.load_state_dict(model_dict)
                model.to(device)

            if device.type == 'cuda' and args.GPU_num > 1:
                if args.GPU_no:
                    assert len(args.GPU_no) == args.GPU_num
                    model = nn.DataParallel(model, [int(each) for each in args.GPU_no])
                else:
                    model = nn.DataParallel(model, list(range(args.GPU_num)))

            optimiser = optim.Adam(model.parameters(), lr=args.lr)
            dataset = MyDataset(dindex)
            #train_num = int(len(dataset)*0.9)+1
            train_num = 0
            test_num = len(dataset) - train_num
            print(test_num)
            training_set, test_set = torch.utils.data.random_split(dataset, [train_num, test_num])
            #train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
            loss_function = nn.MSELoss()
            train_loss_list = []
            test_loss_list = []
            model.eval()
            with torch.no_grad():
                test_loss = 0
                predict_list = []
                true_list = []
                for data in tqdm(test_loader):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.unsqueeze(1).to(device)
                    outputs = model(inputs)
                    #print(outputs.items())
                    try:
                        predict_list.extend(list(outputs.detach().cpu().squeeze().numpy()))
                    except:
                        predict_list.extend([outputs.detach().cpu().squeeze().numpy()])
                    try:
                        true_list.extend(list(labels.detach().cpu().squeeze().numpy()))
                    except:
                        true_list.extend([labels.detach().cpu().squeeze().numpy()])


                    test_loss += torch.sum(torch.abs(labels - outputs))
                    #logging.info('Epoch %d, true:' % epoch + str(true_list) + 'predict:' + str(predict_list))
                #logging.info('Epoch %d Test loss %4.2f' % (epoch, test_loss/test_num))
            print("Test loss %4.2f" % (test_loss/test_num))
            print('**** Finished****')
            train_loss_list = pd.DataFrame(predict_list)
            test_loss_list = pd.DataFrame(true_list)
            train_loss_list.to_csv(train_save,header=False,index=False)
            test_loss_list.to_csv(test_save,header=False,index=False)
    
