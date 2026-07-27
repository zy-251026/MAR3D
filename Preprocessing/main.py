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
from Resnet3D import resnet10,resnet18, resnet34, resnet50, resnet101
from Relation_Transformer import RelationNet
import matplotlib.pyplot as plt
import glob

from torch.utils.data import Dataset, DataLoader

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_num', help="Input the amount of GPU you need", default=0, type=int)
    parser.add_argument('--GPU_no', help="Input the No of GPU you want", default='', type=str)
    parser.add_argument('--batch_size', help="Input the batch number", default=2, type=int)
    parser.add_argument('--epoch', help="Input the number of epoch number", default=40, type=int)
    parser.add_argument('--pretrain', help="Input if you need a pre trained model", default=False, type=bool)
    parser.add_argument('--model_depth', help="Input your resnet depth", default=10, type=int)
    parser.add_argument('--lr', help='Input learning rate', default=0.0001, type=float)
    parser.add_argument('--norm', help='Input your normalisation method', default='dataset', type=str)
    parser.add_argument('--flag', default='', type=str)
    args = parser.parse_args()
    return args


class MyDataset(Dataset):
    def __init__(self, dir_name):
        self.dir_name = dir_name
        #self.y = np.loadtxt(os.path.join(dir_name, 'label', 'age.csv'))
        self.paths = glob.glob('./preprocessing/normal/IXI*.nii.gz')
        self.ypath = os.path.join(dir_name, 'label', 'IXI.csv')
        self.yy = pd.read_csv(self.ypath, header=0)
        self.y = []
        self.defic = []
        for i in range(len(self.paths)):
            pathp = self.paths[i][23:29]
            #print(self.yy.loc[self.yy.name==pathp].age)
            numage = self.yy.loc[self.yy.name==pathp].age
            if len(numage.values)!=0:
                self.y.append(numage.values[0])
            else:
                self.defic.append(i)
        self.paths = np.array(self.paths)
        self.paths = np.delete(self.paths, self.defic)
        self.y = np.array(self.y)

    def __getitem__(self, index):
        self.agent = nib.load(self.paths[index])
        data = np.array(self.agent.dataobj)
        data = data[np.newaxis, :, :, :]
        return torch.from_numpy(data), torch.tensor(self.y[index]).float()

    def __len__(self):
        return self.y.shape[0]


if __name__ == "__main__":
    args = init()
    para_name = '-'.join([key + ':' + str(value) for key, value in args.__dict__.items()])
    if not os.path.exists('logging'):
        os.mkdir('logging')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename='logging/'+para_name+'.log')
    logger = logging.getLogger(__name__)
    logging.info('Start training, parameter:'+para_name)

    assert args.GPU_num <= torch.cuda.device_count(), 'GPU exceed the maximum num'
    if torch.cuda.is_available():
        if args.GPU_no:
            device = torch.device("cuda:"+args.GPU_no[0])
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    #model = eval('resnet'+str(args.model_depth)+'()').to(device)
    model = RelationNet(in_dim=1,
                 num_classes=1,
                 num_transformer_blocks=2,
                 drop_rate=0,
                 im_dim='3d',
                 max_pool_on_image=True,
                 share_backbone=True).to(device)

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
    train_num = int(len(dataset)*0.9)
    test_num = len(dataset) - train_num
    training_set, test_set = torch.utils.data.random_split(dataset, [train_num, test_num])
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    loss_function = nn.MSELoss()
    train_loss_list = []
    test_loss_list = []
    for epoch in range(args.epoch):
        logging.info('Epoch '+str(epoch) + 'start training')
        running_loss = 0.0
        real_loss = 0.0
        model.train()
        for data in tqdm(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.unsqueeze(1).to(device)
            optimiser.zero_grad()
            relations = model(inputs[0].view(1,1,182,218,182), inputs[1].view(1,1,182,218,182))
            age_sum = labels[0] + labels[1]
            loss_sum = torch.mean(torch.abs(relations[0]-age_sum))
            
            age_sub = labels[0] - labels[1]
            loss_sub = torch.mean(torch.abs(relations[1]-age_sub))
            
            age_max = torch.max(labels[0], labels[1])
            loss_max = torch.mean(torch.abs(relations[2]-age_max))
            
            age_min = torch.min(labels[0], labels[1])
            loss_min = torch.mean(torch.abs(relations[3]-age_min))
            
            loss = loss_sum + loss_sub + loss_max + loss_min
            
            #loss = loss_function(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            #real_loss += torch.sum(torch.abs(labels - outputs))
            real_loss += loss
        logging.info("Epoch %d, training loss %4.2f" % (epoch, real_loss/train_num))
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
                outputs = model(inputs, inputs)
                age_sum = labels*2
                loss_sum = torch.mean(torch.abs(relations[0]-age_sum))
            
                age_sub = 0
                loss_sub = torch.mean(torch.abs(relations[1]-age_sub))
            
                age_max = labels
                loss_max = torch.mean(torch.abs(relations[2]-age_max))
            
                age_min = labels
                loss_min = torch.mean(torch.abs(relations[3]-age_min))
            
                loss = loss_sum + loss_sub + loss_max + loss_min
                predict_list.extend(list(outputs.detach().cpu().squeeze().numpy()))
                true_list.extend(list(labels.detach().cpu().squeeze().numpy()))

                #test_loss += torch.sum(torch.abs(labels - outputs))
                test_loss += loss
            logging.info('Epoch %d, true:' % epoch + str(true_list) + 'predict:' + str(predict_list))
        logging.info('Epoch %d Test loss %4.2f' % (epoch, test_loss/test_num))
        test_loss_list.append(test_loss/test_num)
    print('**** Finished Training ****')

    now = str(datetime.today())
    plt.figure(figsize=(20, 10))
    plt.plot(train_loss_list, 'r--', label='train')
    plt.plot(test_loss_list, 'b--', label='valid')
    plt.xlabel("Epoch")
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss of  Resnet{} model with learning rate {} (pretrained:{}, normalisation:{}):'.format(args.model_depth,
                                                                                                        args.lr,
                                                                                                        args.pretrain,
                                                                                                        args.norm))
    plt.savefig("output/loss_history" + para_name + ".png")

    model_name = 'output/model' + para_name + '.pkl'
    torch.save(model, model_name)

