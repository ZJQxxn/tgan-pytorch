import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, genes_num, time_points, batch_size):
        super().__init__()
        self.genes_num = genes_num
        self.time_points = time_points
        self.batch_size = batch_size

        self.dc1 = nn.Linear(self.genes_num, 1024)
        self.dc2 = nn.Linear(1024, 512)
        self.dc3 = nn.Linear(512, 256)
        self.dc4 = nn.Linear(256, 128)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)

        # self.c0 = nn.Conv3d(in_channels, mid_ch, 4, 2, 1)
        # self.c1 = nn.Conv3d(mid_ch, mid_ch * 2, 4, 2, 1)
        # self.c2 = nn.Conv3d(mid_ch * 2, mid_ch * 4, 4, 2, 1)
        # self.c3 = nn.Conv3d(mid_ch * 4, mid_ch * 8, 4, 2, 1)
        # self.bn0 = nn.BatchNorm3d(mid_ch)
        # self.bn1 = nn.BatchNorm3d(mid_ch * 2)
        # self.bn2 = nn.BatchNorm3d(mid_ch * 4)
        # self.bn3 = nn.BatchNorm3d(mid_ch * 8)

    def forward(self, x):
        # if self.sequence_first:
        #     x = x.permute(0, 2, 1, 3, 4)
        batch_size, time_points, genes_num = x.size()
        x = x.contiguous().view(batch_size * time_points, genes_num)
        h = F.leaky_relu(self.bn1(self.dc1(x)))
        h = F.leaky_relu(self.bn2(self.dc2(h)))
        h = F.leaky_relu(self.bn3(self.dc3(h)))
        h = F.leaky_relu(self.bn4(self.dc4(h)))
        h = h.view(h.size(0), -1)
        return torch.mean(h, 1)
   
if __name__ == "__main__":
    batch_size = 4
    time_points = 5
    genes_num = 1000

    x = Variable(torch.randn(batch_size, time_points, genes_num))
    discr = Model(genes_num, time_points, batch_size)
    out = discr(x)
    print(out)
