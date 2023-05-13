import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import cv2

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 3 * 3, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float,requires_grad=True))


    def stn(self,x):
        xs = self.localization(x)
        xs = xs.view(-1, 64 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1,2,3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self,x):
        x = self.stn(x) #Return a 48 x 48 grid 
        return x
        

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.stn = STN()
        self.conv = nn.Conv2d(1, 64, kernel_size= 5)    #42 / 2
        
        self.b1 = Block2(64, 32, 128, [3,5])                     #out 128
        self.b2 =  Block2(128, 64, 128, [3,5])
        self.b3 =  Block2(128, 64, 128, [3,5])                #out 128


        self.b4 = Block2(128, 64, 256, [3,5])                 #out 256
        self.b5 = Block2(256, 128, 256, [3,5])                 #out 256
        self.b6 = Block2(256, 128, 256, [3,5])               #out 256


        self.b7 = Block2(256, 128, 512,  [3,5])               #out 512
        self.b8 =  Block2(512, 256, 512, [3,5])         #out 256
        self.b9 = Block2(512, 256, 512, [3,5])            #out 
        #self.b10 = Block2(512, 256, 512, [3,5])              #out 256
        #self.b11 =   Block2(512, 256, 512, [3,5])                #out 512

        
        self.b12 = Block2(512, 256, 1024, [3,5])              #1024
        self.b13 =   Block2(1024, 512, 1024, [3,5])       #512
        #self.b14 =   Block2(1024, 512, 1024, [3,5])
        #self.b15 = Block2(1024, 512, 1024, [3,5])


        #self.b14 = Block2(1024, 512, [3,5])            #1024
        self.bn1 = nn.BatchNorm1d(2 * 2 * 1024)
        self.drop1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(2 * 2 * 1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 7)

        

    def forward(self, x):
        x = self.stn(x)
        x = self.conv(x)               # 44
        
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)

        x = F.max_pool2d(x,2)              #22

        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)

        x = F.max_pool2d(x,2)                 #11

        x = self.b7(x)
        x = self.b8(x)
        x = self.b9(x)
        #x = self.b10(x)
        #x = self.b11(x)
        
        x = F.max_pool2d(x,2)                          #5

        x = self.b12(x)
        x = self.b13(x)
        #x = self.b14(x)
        #x = self.b15(x)

        x = F.avg_pool2d(x,2)                       #2
        
        x = x.view(-1, 2 * 2 * 1024)
        x = self.drop1(self.bn1(x))
        x = F.leaky_relu(self.bn2(self.fc1(x)))
        x = self.drop(x)
        x = self.fc2(x)
        
        return x
    
    def count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 



class Block2(nn.Module):
    def __init__(self, in_channel, filter, out_channel, ks):
        super(Block2, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, filter, kernel_size= ks[0], padding= 'same')
        self.bn1 = nn.BatchNorm2d(filter)
        self.conv2 = nn.Conv2d(in_channel, filter, kernel_size= ks[1], padding= 'same')
        self.bn2 = nn.BatchNorm2d(filter)
        
        self.conv3 = nn.Conv2d(filter*2+in_channel, out_channel, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channel)
        #self.sq = SEBlock(filter*2 + in_channel, 16)

    
    def forward(self, x):
        x1 = F.leaky_relu(self.bn1(self.conv1(x)))
        x2 = F.leaky_relu(self.bn2(self.conv2(x)))

        res = torch.cat((x1,x2,x), 1)         #filter * 2 + in_channel
        res = self.bn3(self.conv3(res))
        res = F.leaky_relu(res)
        return res
    



class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        # squeeze function
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
       
       # excitation function
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        
        return inputs * x


print(Net2().count())

