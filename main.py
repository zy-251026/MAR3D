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
from model.Resnet3D import resnet10,resnet18, resnet34, resnet50, resnet101, resnet152, resnet200
from model.Aresnet3D import aresnet10, aresnet18, aresnet34, aresnet50, aresnet101, aresnet152, aresnet200
from model.swin_transformer import SwinTransformer
from model.RelationTransformer import RelationNet
from model.vit_3d import ViT3D
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
    parser.add_argument('--model', help="Input your model", default='aresnet', type=str)
    parser.add_argument('--model_depth', help="Input your resnet depth", default=34, type=int)
    parser.add_argument('--lr', help='Input learning rate', default=0.001, type=float)
    parser.add_argument('--norm', help='Input your normalisation method', default='dataset', type=str)
    
    parser.add_argument('--flag', default='', type=str)
    args = parser.parse_args()
    return args


class MyDataset(Dataset):
    def __init__(self, dir_name):
        self.dir_name = dir_name
        #self.y = np.loadtxt(os.path.join(dir_name, 'label', 'age.csv'))
        self.paths = glob.glob('./preprocessing/normal/*.nii.gz')
        self.ypath = os.path.join(dir_name, 'label', 'label2.csv')
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
        #data = torch.cat((data, torch.zeros(1,21,218, 182)), 1)
        #data = torch.cat((torch.zeros(1,21,218, 182), data), 1)
        #data = torch.cat((data, torch.zeros(1,224,3,182)), 2)
        #data = torch.cat((torch.zeros(1,224,3,182), data), 2)
        #data = torch.cat((data, torch.zeros(1,224,224,21)), 3)
        #data = torch.cat((torch.zeros(1,224,224,21), data), 3)
        #print(data.shape)
        #data = torch.from_numpy(data).view(182,218,182)
        #data2, d1 = data.split([180,2],dim=0)
        #data3, d1 = data2.split([216,2],dim=1)
        #data4, d1 = data3.split([180,2],dim=2)
        #data5 = data4.view(1,180,216,180)
        return data, torch.tensor(self.y[index]).float()

    def __len__(self):
        return self.y.shape[0]


if __name__ == "__main__":
    args = init()
    max_loss = 1000
    dataset_name = 'Alldataset_'
    model_surname = args.model + '3d'
    os.makedirs('data/'+dataset_name+model_surname+str(args.model_depth)+'/', exist_ok = True)
    para_name = '_'.join([key + ':' + str(value) for key, value in args.__dict__.items()])
    t = datetime.now()
    str_time = t.strftime("%Y-%m-%d_%H:%M:%S")
    train_save = 'data/'+dataset_name+model_surname+str(args.model_depth)+'/train_'+str_time+'.csv'
    test_save = 'data/'+dataset_name+model_surname+str(args.model_depth)+'/test_'+str_time+'.csv'
    model_name = './model/'+model_surname+'_model_' + str(args.model_depth) + '_' + str_time + '.pkl'
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
    model = eval(str(args.model)+str(args.model_depth)+'()').to(device)
    old_model = torch.load('model/aresnet3d_model_34_2024-01-31_18_26_44.pkl', map_location=None if torch.cuda.is_available() else 'cpu')
    old_model_dict = {key.replace('module.', '') : value for key, value in old_model.state_dict().items()}
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
    dataset = MyDataset(args.norm)
    train_num = int(len(dataset)*0.9)+1
    test_num = len(dataset) - train_num
    #print(train_num, test_num)
    training_set, test_set = torch.utils.data.random_split(dataset, [train_num, test_num])
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    loss_function = nn.MSELoss()
    train_loss_list = []
    test_loss_list = []
    for epoch in range(args.epoch):
        #logging.info('Epoch '+str(epoch))
        running_loss = 0.0
        real_loss = 0.0
        model.train()
        for data in tqdm(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.unsqueeze(1).to(device)
            optimiser.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            real_loss += torch.sum(torch.abs(labels - outputs))
        #logging.info("Epoch %d, Training loss %4.2f" % (epoch, real_loss/train_num))
        print("Epoch %d, Training loss %4.2f" % (epoch, real_loss/train_num))
        train_loss_list.append(real_loss/train_num)
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
        print("Epoch %d, Test loss %4.2f" % (epoch, test_loss/test_num))
        if test_loss/test_num < max_loss:
            max_loss = test_loss/test_num
            torch.save(model, model_name)
        test_loss_list.append(test_loss/test_num)
    print('**** Finished****')
    #train_loss_list = pd.DataFrame(train_loss_list)
    #test_loss_list = pd.DataFrame(test_loss_list)
    #train_loss_list.to_csv(train_save,header=False,index=False)
    #test_loss_list.to_csv(test_save,header=False,index=False)
    
