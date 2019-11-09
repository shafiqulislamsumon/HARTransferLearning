# encoding=utf-8

import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 3)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * 98, out_features=100),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=2)
        )

    def forward(self, src, tar):
        x_src = self.conv1(src)
        x_tar = self.conv1(tar)
        
        x_src = self.conv2(x_src)
        x_tar = self.conv2(x_tar)
        #print(x_src.shape)
        x_src = x_src.reshape(-1, 64 * 98)
        x_tar = x_tar.reshape(-1, 64 * 98)
        
        x_src_mmd = self.fc1(x_src)
        x_tar_mmd = self.fc1(x_tar)
        
        #x_src = self.fc1(x_src)
        #x_tar = self.fc1(x_tar)
        
        #x_src_mmd = self.fc2(x_src)
        #x_tar_mmd = self.fc2(x_tar)
        
        y_src = self.fc2(x_src_mmd)
        
        return y_src, x_src_mmd, x_tar_mmd
    
