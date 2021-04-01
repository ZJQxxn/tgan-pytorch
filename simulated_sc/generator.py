import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
import copy
import numpy as np

class FrameSeedGenerator(nn.Module):
    # def __init__(self, z_slow_dim, z_fast_dim, time_points, bath_size):
    #     super().__init__()
    #     self.z_slow_dim = z_slow_dim * bath_size
    #     self.z_fast_dim = z_fast_dim * bath_size
    #     self.time_points = time_points
    #
    #     self.dc0 = nn.Linear(self.z_slow_dim, 1024)
    #     self.dc1 = nn.Linear(1024, 512)
    #     self.dc2 = nn.Linear(512, 256)
    #     self.dc3 = nn.Linear(256, 128)
    #     self.dc4 = nn.Linear(128, self.z_fast_dim)
    #     self.bn0 = nn.BatchNorm1d(1024)
    #     self.bn1 = nn.BatchNorm1d(512)
    #     self.bn2 = nn.BatchNorm1d(256)
    #     self.bn3 = nn.BatchNorm1d(128)
    #
    # def forward(self, z_slow):
    #     # h = z_slow.view(z_slow.size(0),-1, 1) #
    #     # repeat_z_slow = z_slow.repeat(time_points, 1).view(time_points, -1)
    #     # h = copy.deepcopy(repeat_z_slow) # h.permute(1, 0, 2)
    #     h = copy.deepcopy(z_slow) # h.permute(1, 0, 2)
    #     h = F.relu(self.bn0(self.dc0(h)))
    #     h = F.relu(self.bn1(self.dc1(h)))
    #     h = F.relu(self.bn2(self.dc2(h)))
    #     h = F.relu(self.bn3(self.dc3(h)))
    #     z_fast = F.tanh(self.dc4(h))
    #     return z_fast

    def __init__(self, z_slow_dim, z_fast_dim):
        super().__init__()
        self.z_slow_dim = z_slow_dim
        self.z_fast_dim = z_fast_dim

        self.dc0 = nn.ConvTranspose1d(z_slow_dim, 512, 1, 1, 0)
        self.dc1 = nn.ConvTranspose1d(512, 256, 4, 2, 1)
        self.dc2 = nn.ConvTranspose1d(256, 128, 4, 2, 1)
        self.dc3 = nn.ConvTranspose1d(128, z_fast_dim, 4, 2, 1)
        # self.dc4 = nn.ConvTranspose1d(128, z_fast_dim, 4, 2, 1)
        self.bn0 = nn.BatchNorm1d(512)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(128)

    def forward(self, z_slow):
        h = z_slow.view(z_slow.size(0), -1, 1)
        h = F.relu(self.bn0(self.dc0(h)))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        # h = F.relu(self.bn3(self.dc3(h)))
        z_fast = F.tanh(self.dc3(h))
        return z_fast



class LinearFrameSeedGenerator(nn.Module):
    # def __init__(self, z_slow_dim, z_fast_dim, time_points, bath_size):
    #     super().__init__()
    #     self.z_slow_dim = z_slow_dim * bath_size
    #     self.z_fast_dim = z_fast_dim * bath_size
    #     self.time_points = time_points
    #
    #     self.dc0 = nn.Linear(self.z_slow_dim, 1024)
    #     self.dc1 = nn.Linear(1024, 512)
    #     self.dc2 = nn.Linear(512, 256)
    #     self.dc3 = nn.Linear(256, 128)
    #     self.dc4 = nn.Linear(128, self.z_fast_dim)
    #     self.bn0 = nn.BatchNorm1d(1024)
    #     self.bn1 = nn.BatchNorm1d(512)
    #     self.bn2 = nn.BatchNorm1d(256)
    #     self.bn3 = nn.BatchNorm1d(128)
    #
    # def forward(self, z_slow):
    #     # h = z_slow.view(z_slow.size(0),-1, 1) #
    #     # repeat_z_slow = z_slow.repeat(time_points, 1).view(time_points, -1)
    #     # h = copy.deepcopy(repeat_z_slow) # h.permute(1, 0, 2)
    #     h = copy.deepcopy(z_slow) # h.permute(1, 0, 2)
    #     h = F.relu(self.bn0(self.dc0(h)))
    #     h = F.relu(self.bn1(self.dc1(h)))
    #     h = F.relu(self.bn2(self.dc2(h)))
    #     h = F.relu(self.bn3(self.dc3(h)))
    #     z_fast = F.tanh(self.dc4(h))
    #     return z_fast

    def __init__(self, z_slow_dim, z_fast_dim, time_points):
        super().__init__()
        self.z_slow_dim = z_slow_dim
        self.z_fast_dim = z_fast_dim
        self.time_points = time_points

        self.l0 = nn.Linear(self.z_slow_dim, 1024)
        self.l1 = nn.Linear(1024, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, self.z_fast_dim)
        self.bn0 = nn.BatchNorm1d(1024)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)


    def forward(self, z_slow):
        # h = z_slow.view(z_slow.size(0), -1, 1)
        batch_size = z_slow.size()[0]
        time_points = self.time_points
        h = z_slow.repeat(time_points, 1)
        h = F.relu(self.bn0(self.l0(h)))
        h = F.relu(self.bn1(self.l1(h)))
        h = F.relu(self.bn2(self.l2(h)))
        h = F.relu(self.bn3(self.l3(h)))
        z_fast = F.tanh(self.l4(h))
        z_fast = z_fast.view(batch_size, time_points, -1).permute(0, 2, 1)
        return z_fast



class VideoGenerator(nn.Module): #TODO: for atch data; for every time step; for every cell
    def __init__(self, z_slow_dim, z_fast_dim, genes_dim):
        super().__init__()
        self.genes_dim = genes_dim
        self.slow_dim = z_slow_dim
        self.fast_dim = z_fast_dim
        self.middle_dim = 32
        self.z_concat_dim = 64

        self.ls = nn.Linear(self.slow_dim, self.middle_dim)
        self.lf = nn.Linear(self.fast_dim, self.middle_dim)

        self.dc1 = nn.Linear(self.z_concat_dim, 256)
        self.dc2 = nn.Linear(256, 512)
        self.dc3 = nn.Linear(512, 1024)
        self.dc4 = nn.Linear(1024, self.genes_dim)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
    
    def forward(self, z_slow, z_fast):
        h_slow = F.relu(self.ls(z_slow))
        h_fast = F.relu(self.ls(z_fast))
        h = torch.cat((h_slow, h_fast), 1)
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        x = F.tanh(self.dc4(h))
        return x


class Model(nn.Module):
    def __init__(self, z_slow_dim=128, z_fast_dim=128, genes_dim = 1000, time_points= 5, batch_size = 1):
        super().__init__()
        self.z_slow_dim = z_slow_dim
        self.z_fast_dim = z_fast_dim
        self.genes_dim = genes_dim
        self.time_points = time_points
        self.batch_size = batch_size


        # self._fsgen = FrameSeedGenerator(self.z_slow_dim, self.z_fast_dim)
        self._fsgen = LinearFrameSeedGenerator(self.z_slow_dim, self.z_fast_dim, self.time_points)
        self._vgen = VideoGenerator(self.z_slow_dim, self.z_fast_dim, self.genes_dim)

    def generate_input(self, batch_size=16, time_points = 5):
        """
        Generates latent vector from normal distribution
        """
        z_slow = torch.randn(batch_size, self.z_slow_dim)
        # z_slow = z_slow.repeat(time_points, 1).view(time_points, -1)
        return z_slow

    def forward(self, z_slow):
        z_fast = self._fsgen(z_slow)
        B, n_z_fast, time_points = z_fast.size()
        z_fast = z_fast.permute(0, 2, 1).contiguous().view(B * time_points, n_z_fast) #squash time dimension in batch dimension

        B, n_z_slow = z_slow.size()
        z_slow = z_slow.unsqueeze(1).repeat(1, time_points, 1)
        z_slow = z_slow.contiguous().view(B * time_points, n_z_slow)
        
        out = self._vgen(z_slow, z_fast)
        out = out.contiguous().view(B, time_points, self.genes_dim)
        return out



if __name__ == "__main__":
    batch_size = 4
    time_points = 16
    gen = Model(batch_size=batch_size, time_points=time_points)
    z_slow = Variable(gen.generate_input(batch_size, time_points))
    out = gen(z_slow)
    print("Output video generator:", out.size())
